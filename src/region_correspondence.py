"""Decision-space KD for unequal teacher/student embedding dimensions.

Row-conditional region matching includes a no-match bin. It is NOT a
one-to-one optimal transport plan, semantic segmentation or causal attention.
All distillation computations run in FP32 even under outer CUDA autocast.
"""
import torch
from torch.nn import functional as F


def observed_visibility(images, modality, grid):
    from src.semantic_region import content_prior
    prior = content_prior(images, modality, grid)
    if modality == 'sketch':
        mean = images.new_tensor([.48145466, .4578275, .40821073])[None, :, None, None]
        std = images.new_tensor([.26862954, .26130258, .27577711])[None, :, None, None]
        observed = ((images.float() * std + mean).mean(1) < .8).flatten(1).any(1)
        prior = prior * observed[:, None]
    return prior.float()


def positive_retrieval_loss(sketch, photo, sketch_ids, photo_ids, temperature):
    """Symmetric multi-positive loss: same category OR exact same photo ID."""
    logits = F.normalize(sketch.float(), dim=-1) @ F.normalize(photo.float(), dim=-1).T / temperature
    mask = sketch_ids[:, None].eq(photo_ids[None, :])
    if not mask.any(1).all() or not mask.any(0).all():
        raise ValueError('Every query/gallery row must have a ground-truth positive')
    def direction(scores, positives):
        return (scores.logsumexp(1) - scores.masked_fill(~positives, -torch.inf).logsumexp(1)).mean()
    return .5 * (direction(logits, mask) + direction(logits.T, mask.T))


def region_similarity(sketch, photo):
    return torch.einsum('nrd,mpd->nmrp', F.normalize(sketch.float(), dim=-1),
                        F.normalize(photo.float(), dim=-1))


def conditional_plan(similarity, photo_visibility, temperature, no_match_score):
    """Shared support = photo region IDs + a dustbin; no feature projection."""
    prior = photo_visibility.float()
    sums = prior.sum(-1, keepdim=True)
    prior = torch.where(sums > 0, prior / sums.clamp_min(1e-12), torch.full_like(prior, 1 / prior.shape[-1]))
    logits = similarity.float() / temperature + prior.clamp_min(1e-30).log().unsqueeze(-2)
    dustbin = logits.new_full((*logits.shape[:-1], 1), no_match_score / temperature)
    return torch.cat((logits, dustbin), -1).softmax(-1)


def local_candidate_scores(similarity, sketch_visibility, photo_visibility, temperature):
    prior = photo_visibility.float().clamp_min(1e-30)
    per_region = temperature * (similarity / temperature + prior.log()[None, :, None, :]).logsumexp(-1)
    vis = sketch_visibility.float()
    vis = vis / vis.sum(-1, keepdim=True).clamp_min(1e-12)
    return (per_region * vis[:, None, :]).sum(-1), per_region


def _losses(sketch_output, photo_output, batch, args):
    s, p = sketch_output['descriptor'].float(), photo_output['descriptor'].float()
    ids = batch['positive_id']
    values = {'retrieval': positive_retrieval_loss(s, p, ids, ids, args.retrieval_temperature)}
    zero = s.sum() * 0 + p.sum() * 0
    values.update(rank=zero, correspondence=zero, teacher_acceptance=zero.detach(), evidence_fraction=zero.detach())
    if args.correspondence_mode == 'gt':
        return args.lambda_retrieval * values['retrieval'], values
    with torch.no_grad():
        gt = ids[:, None].eq(ids[None, :])
        tg = F.normalize(batch['sketch_global'].float(), dim=-1) @ F.normalize(batch['photo_global'].float(), dim=-1).T
        similarity = region_similarity(batch['sketch_crops'], batch['photo_crops'])
        local, per_region = local_candidate_scores(similarity, batch['sketch_visibility'],
                                                   batch['photo_visibility'], args.match_temperature)
        alpha = 0. if args.correspondence_mode == 'global' else args.teacher_local_weight
        scores = (1 - alpha) * tg + alpha * local
        pos = scores.masked_fill(~gt, -torch.inf).argmax(1)
        student_scores = F.normalize(s.detach(), dim=-1) @ F.normalize(p.detach(), dim=-1).T
        selections, accepted = [], []
        for i in range(len(s)):
            negatives = (~gt[i]).nonzero().flatten()
            if len(negatives) == 0:
                selections.append(None); accepted.append(False)
                continue
            if args.retrieval_protocol == 'fg':
                same_class = negatives[batch['category'][negatives].eq(batch['category'][i])]
                if len(same_class):
                    negatives = same_class
            order = student_scores[i, negatives].argsort(descending=True, stable=True)
            negatives = negatives[order[:args.hard_negatives]]
            selected = torch.cat((pos[i:i+1], negatives))
            margin = scores[i, pos[i]] - scores[i, negatives].max()
            ok = bool(margin > args.teacher_min_margin) and bool(batch['sketch_visibility'][i].sum() > 0)
            selections.append(selected); accepted.append(ok)
        accepted = torch.tensor(accepted, device=s.device, dtype=torch.bool)
        values['teacher_acceptance'] = accepted.float().mean()
    ranks, plans, evidence = [], [], []
    ss = region_similarity(sketch_output['regions'], photo_output['regions'])
    for i, selected in enumerate(selections):
        if selected is None or not bool(accepted[i]):
            continue
        with torch.no_grad():
            q = (scores[i, selected] / args.rank_temperature).softmax(-1)
        student_logits = (F.normalize(s[i:i+1], dim=-1) @ F.normalize(p[selected], dim=-1).T).squeeze(0)
        ranks.append(F.kl_div((student_logits / args.rank_temperature).log_softmax(-1), q, reduction='sum'))
        if args.correspondence_mode == 'global' or args.lambda_correspondence == 0:
            continue
        with torch.no_grad():
            # Only rows with positive-vs-hard-negative local evidence supervise matching.
            strength = (per_region[i, selected[0]] - per_region[i, selected[1:]].max(0).values).clamp_min(0)
            weights = strength * batch['sketch_visibility'][i].float()
            teacher_sim = similarity[i, selected]
            if args.correspondence_mode == 'shuffled':
                teacher_sim = similarity[(i + 1) % len(s), selected]
            target = conditional_plan(teacher_sim, batch['photo_visibility'][selected],
                                      args.match_temperature, args.no_match_score)
            if args.correspondence_mode == 'uniform':
                visible = batch['photo_visibility'][selected].float()
                visible = visible / visible.sum(-1, keepdim=True).clamp_min(1e-12)
                target = torch.cat(((1 - target[..., -1:]) * visible[:, None, :], target[..., -1:]), -1)
            weights = weights[None, :] * (1 - target[..., -1])
            evidence.append((weights > 0).float().mean())
        student_plan = conditional_plan(ss[i, selected], batch['photo_visibility'][selected],
                                        args.match_temperature, args.no_match_score)
        divergence = (target * (target.clamp_min(1e-30).log() - student_plan.clamp_min(1e-30).log())).sum(-1)
        if bool(weights.sum() > 0):
            plans.append((divergence * weights).sum() / weights.sum())
    # Acceptance scales KD instead of amplifying a tiny accepted subset.
    if ranks:
        values['rank'] = torch.stack(ranks).sum() / len(s)
    if plans:
        values['correspondence'] = torch.stack(plans).sum() / len(s)
    if evidence:
        values['evidence_fraction'] = torch.stack(evidence).sum() / len(s)
    total = (args.lambda_retrieval * values['retrieval'] + args.lambda_rank * values['rank']
             + args.lambda_correspondence * values['correspondence'])
    return total, values


def correspondence_losses(sketch_output, photo_output, batch, args):
    with torch.autocast(device_type=sketch_output['descriptor'].device.type, enabled=False):
        return _losses(sketch_output, photo_output, batch, args)


@torch.no_grad()
def retrieval_metrics(sketch, photo, sketch_ids, photo_ids, protocol, precision_k=100, chunk=64, fg_gallery='all'):
    if not len(sketch) or not len(photo):
        raise ValueError('Retrieval evaluation requires nonempty queries and gallery')
    device = sketch.device
    photo = F.normalize(photo.float(), dim=-1)
    sid, pid = torch.as_tensor(sketch_ids, device=device), torch.as_tensor(photo_ids, device=device)
    sc = pc = None
    if sid.ndim == 2 and pid.ndim == 2:
        sc, pc = sid[:, 1], pid[:, 1]
        sid, pid = sid[:, 0], pid[:, 0]
    restricted = protocol == 'fg' and fg_gallery == 'category'
    if restricted and (sc is None or pc is None):
        raise ValueError('Known-category FG gallery requires category IDs with instance IDs')
    if protocol == 'fg' and pid.unique().numel() != pid.numel():
        raise ValueError('FG validation gallery photo IDs must be unique')
    totals = {'mAP': 0., 'precision': 0., 'Acc1': 0., 'Acc5': 0.}
    k = min(precision_k, len(photo))
    with torch.autocast(device_type=device.type, enabled=False):
        for start in range(0, len(sketch), chunk):
            scores = F.normalize(sketch[start:start+chunk].float(), dim=-1) @ photo.T
            if restricted:
                scores = scores.masked_fill(~sc[start:start+chunk, None].eq(pc[None, :]), -torch.inf)
            order = scores.argsort(dim=-1, descending=True, stable=True)
            truth = pid[order].eq(sid[start:start+chunk, None])
            if restricted:
                truth = truth & pc[order].eq(sc[start:start+chunk, None])
            counts = truth.sum(-1)
            if (counts == 0).any():
                raise ValueError('A validation sketch has no ground-truth gallery photo')
            ranks = torch.arange(1, len(photo)+1, device=device)
            ap = ((truth.cumsum(-1) / ranks) * truth).sum(-1) / counts
            totals['mAP'] += ap.sum().item()
            divisor = sc[start:start+chunk, None].eq(pc[None, :]).sum(-1).clamp(max=k) if restricted else k
            totals['precision'] += (truth[:, :k].float().sum(-1) / divisor).sum().item()
            totals['Acc1'] += truth[:, 0].sum().item()
            totals['Acc5'] += truth[:, :min(5, len(photo))].any(-1).sum().item()
    values = {name: value / len(sketch) for name, value in totals.items()}
    values.update(queries=len(sketch), gallery=len(photo), precision_k=k, protocol=protocol)
    values['gallery_scope'] = 'known_category' if restricted else 'all'
    candidate_counts = [int(pc.eq(c).sum()) for c in sc.unique()] if restricted else [len(photo)]
    values['gallery_candidates_min'] = min(candidate_counts)
    values['gallery_candidates_max'] = max(candidate_counts)
    values['primary'] = values['Acc1'] if protocol == 'fg' else values['mAP']
    return values
