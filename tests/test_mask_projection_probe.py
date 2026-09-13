from argparse import Namespace
from copy import deepcopy
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

CELL=Path(__file__).resolve().parents[1]/'test/kaggle_mask_projection_probe_cell.py'
SOURCE=runpy.run_path(str(CELL),run_name='probe_cell_test')['RUNNER_SOURCE']
PROBE={'__name__':'probe_test','__file__':str(CELL)}
exec(compile(SOURCE,'<probe>','exec'),PROBE)


class ProjectionProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_discovery_requires_explicit_choice_if_ambiguous(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            p=root/'saved_models/mask_embedding_s42_a/final.ckpt'
            p.parent.mkdir(parents=True);p.touch()
            self.assertEqual(PROBE['select_checkpoints'](root,[]),[p])
            q=root/'saved_models/mask_embedding_s42_b/final.ckpt'
            q.parent.mkdir(parents=True);q.touch()
            with self.assertRaisesRegex(ValueError,'Several'):PROBE['select_checkpoints'](root,[])
            self.assertEqual(PROBE['select_checkpoints'](root,[str(q)]),[q])
            self.assertFalse(list(root.glob('*.tmp')))

    def test_centered_spectrum_spread_and_class_decomposition(self):
        x=F.normalize(torch.randn(20,8),dim=-1);labels=torch.arange(20)%4
        result,spectrum=PROBE['feature_statistics'](x,labels)
        z=F.normalize(x,dim=-1).double();centered=z-z.mean(0)
        exact=torch.linalg.svdvals(centered)
        actual=torch.tensor([r['centered_singular_value'] for r in spectrum],dtype=torch.float64)
        torch.testing.assert_close(actual,exact,atol=1e-9,rtol=1e-9)
        self.assertAlmostEqual(result['total_spread'],centered.square().sum().item()/len(x),places=10)
        pair=z@z.T
        self.assertAlmostEqual(result['mean_off_diagonal_cosine'],pair[~torch.eye(len(x),dtype=torch.bool)].mean().item(),places=6)
        within=sum((centered[labels==c]-centered[labels==c].mean(0)).square().sum().item() for c in labels.unique())/len(x)
        self.assertAlmostEqual(result['within_class_spread'],within,places=10)
        collapsed,_=PROBE['feature_statistics'](torch.ones(20,8),labels)
        self.assertLess(collapsed['total_spread'],1e-20)
        self.assertEqual(collapsed['effective_covariance_rank'],0)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_retrieval_matches_original_main(self):
        from src.model import _retrieval_metrics
        features={'sketch':torch.randn(6,12),'photo':torch.randn(104,12)}
        labels={'sketch':torch.arange(6)%3,'photo':torch.arange(104)%3}
        result=PROBE['retrieval'](features,labels,'sketchy_1')
        ap,p,_,_=_retrieval_metrics(features['sketch'].cuda(),features['photo'].cuda(),labels['sketch'],labels['photo'],'sketchy_1')
        self.assertAlmostEqual(result['mAP'],ap.item(),places=6)
        self.assertAlmostEqual(result['precision'],p.item(),places=6)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_seed_reconstruction_and_swapped_projection_match_model(self):
        from clip.model import CLIP,build_model
        from src.model import ZS_SBIR
        args=Namespace(backbone='ViT-B/32',seed=42,n_ctx_visual=3,prompt_depth=3,
                       lambda_domain=0.,lambda_modality=0.,kd_temperature=.07,
                       photo_text_kd_temperature=.15,sketch_text_kd_temperature=.02,
                       teacher_cache_path='',rebuild_teacher_cache=False,teacher_pretrain_epochs=1,
                       lr=.01,momentum=.9,weight_decay=.0005,retrieval_head='mask_guided',mask_strategy='attention',
                       lambda_embedding=1.,lambda_response=1.,mask_gradient_audit=False,dataset='sketchy_1')
        tiny=CLIP(32,16,3,64,4,16,128,64,1,1)
        with patch('src.model._load_teacher',return_value=None),patch('src.model._load_clip_model',return_value=build_model(tiny.state_dict())):
            original=ZS_SBIR(args,['cat','dog']).cuda().eval().requires_grad_(False)
        w0=original.model.retrieval_projection.weight.detach().clone()
        prompt0={n:p.detach().clone() for n,p in original.named_parameters() if '_visual_prompt.' in n}
        images=torch.randn(3,3,16,16,device='cuda')
        initial_features=original.model.encode_student_image(images,'photo').cpu()
        with torch.no_grad():
            for n,p in original.named_parameters():
                if '_visual_prompt.' in n or 'retrieval_projection.' in n:p.add_(torch.randn_like(p)*.03)
        saved={'state_dict':{k:v.cpu().clone() for k,v in original.state_dict().items()},
               'experiment_config':{'args':vars(args),'retrieval_head':'mask_guided'},
               'hyper_parameters':{'classnames':['cat','dog']}}
        before=PROBE['state_hash'](saved['state_dict'])
        rebuilt=PROBE['construct_initial'](saved).cuda()
        torch.testing.assert_close(rebuilt.model.retrieval_projection.weight,w0,rtol=0,atol=0)
        for n,p in rebuilt.named_parameters():
            if n in prompt0:torch.testing.assert_close(p,prompt0[n],rtol=0,atol=0)
        torch.testing.assert_close(rebuilt.model.encode_student_image(images,'photo').cpu(),initial_features,rtol=0,atol=0)
        wt=original.model.retrieval_projection.weight.detach().clone()
        initial_wt=PROBE['project_features']({'photo':initial_features},wt)['photo']
        rebuilt.model.retrieval_projection.weight.copy_(wt)
        torch.testing.assert_close(rebuilt.model.extract_feature(images,'photo').cpu(),initial_wt)
        rebuilt.load_state_dict(saved['state_dict'],strict=True)
        trained=rebuilt.model.encode_student_image(images,'photo').cpu()
        swapped=PROBE['project_features']({'photo':trained},w0)['photo']
        rebuilt.model.retrieval_projection.weight.copy_(w0)
        torch.testing.assert_close(rebuilt.model.extract_feature(images,'photo').cpu(),swapped)
        self.assertEqual(PROBE['state_hash'](saved['state_dict']),before)


if __name__=='__main__':unittest.main()
