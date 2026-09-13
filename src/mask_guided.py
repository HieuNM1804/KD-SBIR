"""Embedding and attention-guided response distillation; no ranking objective."""
from contextlib import AbstractContextManager
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F


class TeacherAttention(AbstractContextManager):
    """Last-block CLS attention; average heads, then condition on image patches."""
    def __init__(self, visual):
        self.attn = visual.transformer.resblocks[-1].attn
        self.patches = visual.positional_embedding.shape[0] - 1
        self.values = []
        self.handle = None

    def __enter__(self):
        def hook(module, args, kwargs):
            if any(kwargs.get(k) is not None for k in ('attn_mask', 'key_padding_mask')) or kwargs.get('is_causal'):
                raise ValueError('Masked attention not supported by teacher capture')
            if module.bias_k is not None or module.bias_v is not None or module.add_zero_attn:
                raise ValueError('Extra key/value tokens unsupported')
            q, k = [args[i] if len(args)>i else kwargs[name] for i,name in enumerate(('query','key'))]
            if not module.batch_first:q,k=q.transpose(0,1),k.transpose(0,1)
            weights = module.in_proj_weight.chunk(3) if module.in_proj_weight is not None else (module.q_proj_weight,module.k_proj_weight)
            biases = (None,None) if module.in_proj_bias is None else module.in_proj_bias.chunk(3)[:2]
            with torch.autocast(device_type=q.device.type, enabled=False):
                q=F.linear(q[:,:1].float(),weights[0].float(),None if biases[0] is None else biases[0].float())
                k=F.linear(k.float(),weights[1].float(),None if biases[1] is None else biases[1].float())
                b,_,width=q.shape; heads=module.num_heads; dim=width//heads
                q=q.reshape(b,1,heads,dim).transpose(1,2)
                k=k.reshape(b,-1,heads,dim).transpose(1,2)
                a=(q@k.transpose(-1,-2)/math.sqrt(dim)).softmax(-1)[:,:,0,1:1+self.patches].mean(1)
                self.values.append(a/a.sum(-1,keepdim=True).clamp_min(1e-12))
        if self.handle is not None:raise RuntimeError('Capture already active')
        self.values.clear()
        self.handle=self.attn.register_forward_pre_hook(hook,with_kwargs=True)
        return self

    def __exit__(self,*args):
        self.handle.remove();self.handle=None


def spatial_attention(attention, grid):
    side=math.isqrt(attention.shape[-1])
    if side*side != attention.shape[-1]:raise ValueError('Teacher patch layout must be square')
    # Pool on normalized image coordinates, not teacher/student token indices.
    pooled=F.adaptive_avg_pool2d(attention.reshape(-1,1,side,side),(grid,grid)).flatten(1)
    return pooled/pooled.sum(-1,keepdim=True).clamp_min(1e-12)


def make_masks(attention, paths, grid, ratio, seed):
    if not 0 < ratio < 1:raise ValueError('mask_ratio must be in (0,1)')
    scores=spatial_attention(attention,grid).cpu()
    count=max(1,min(grid*grid-1,round(ratio*grid*grid)))
    guided=torch.zeros_like(scores,dtype=torch.bool)
    guided.scatter_(1,scores.argsort(dim=-1,descending=True,stable=True)[:,:count],True)
    random=torch.zeros_like(guided)
    for i,path in enumerate(paths):
        key=int.from_bytes(hashlib.sha256(f'{seed}:{path}'.encode()).digest()[:8],'little')%(2**63-1)
        ids=torch.randperm(grid*grid,generator=torch.Generator().manual_seed(key))[:count]
        random[i,ids]=True
    return scores,guided.reshape(-1,grid,grid),random.reshape(-1,grid,grid)


def apply_mask(images, mask):
    """Zero in CLIP-normalized space = CLIP mean RGB, same for both modalities."""
    single=images.ndim==3
    if single:images,mask=images[None],mask[None]
    if mask.ndim!=3 or len(images)!=len(mask):raise ValueError('Invalid image/mask batch')
    if images.shape[-2]%mask.shape[-2] or images.shape[-1]%mask.shape[-1]:
        raise ValueError('Image dimensions must be divisible by mask grid')
    expanded=mask.to(images.device).repeat_interleave(images.shape[-2]//mask.shape[-2],-2)
    expanded=expanded.repeat_interleave(images.shape[-1]//mask.shape[-1],-1)
    output=images.masked_fill(expanded[:,None],0)
    return output[0] if single else output


class RetrievalProjection(nn.Module):
    """One shared projection for photo/sketch; retained during retrieval."""
    def __init__(self, input_dim, output_dim=1024, seed=42):
        super().__init__()
        generator=torch.Generator().manual_seed(seed)
        self.weight=nn.Parameter(torch.randn(output_dim,input_dim,generator=generator)/math.sqrt(input_dim))

    def forward(self, features):
        with torch.autocast(device_type=features.device.type,enabled=False):
            return F.normalize(F.linear(features.float(),self.weight.float()),dim=-1)


def embedding_loss(student, teacher):
    teacher=F.normalize(teacher.detach().to(student.device).float(),dim=-1)
    return (1-(student.float()*teacher).sum(-1)).mean()


def response_terms(student, masked_student, teacher, masked_teacher):
    t=F.normalize(teacher.detach().to(student.device).float(),dim=-1)
    tm=F.normalize(masked_teacher.detach().to(student.device).float(),dim=-1)
    dt=t-tm; ds=student.float()-masked_student.float()
    error=(ds-dt).square().sum(-1)
    tn,sn=dt.norm(dim=-1),ds.norm(dim=-1)
    valid=(tn>1e-6)&(sn>1e-6)
    cosine=F.cosine_similarity(ds,dt,dim=-1)
    stats={'response_loss':error.mean(), 'teacher_delta':tn.mean(),'student_delta':sn.mean(),
           'response_error_over_energy':error.mean()/dt.square().sum(-1).mean().clamp_min(1e-12),
           'response_cosine':cosine[valid].mean() if valid.any() else cosine.new_zeros(()),
           'response_cosine_valid_fraction':valid.float().mean(),
           'masked_embedding_loss':embedding_loss(masked_student,tm)}
    return error.mean(),stats


def gradient_comparison(base, response, named_parameters):
    """Unweighted loss gradients on one batch, separated by parameter role."""
    named=[(n,p) for n,p in named_parameters if p.requires_grad]
    ga=torch.autograd.grad(base,[p for _,p in named],retain_graph=True,allow_unused=True)
    gb=torch.autograd.grad(response,[p for _,p in named],retain_graph=True,allow_unused=True)
    rows=[]
    for group,tag in [('photo','photo_visual_prompt.'),('sketch','sketch_visual_prompt.'),('projection','retrieval_projection.')]:
        ids=[i for i,(n,p) in enumerate(named) if tag in n]
        if not ids:continue
        a=torch.cat([(torch.zeros_like(named[i][1]) if ga[i] is None else ga[i]).detach().flatten().float() for i in ids])
        b=torch.cat([(torch.zeros_like(named[i][1]) if gb[i] is None else gb[i]).detach().flatten().float() for i in ids])
        na,nb=a.norm().item(),b.norm().item()
        rows.append({'group':group,'embedding_grad_norm':na,'response_grad_norm':nb,
                     'cosine':(a@b).item()/(na*nb) if na*nb>1e-20 else None})
    return rows
