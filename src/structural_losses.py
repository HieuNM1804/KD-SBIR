"""FP32 objectives for heterogeneous same-image distillation.

Transport is solved on detached costs; the outer objective differentiates
through student costs with a fixed approximate plan (no unrolled solver).
"""
import math
import torch
from torch.nn import functional as F


def norm(x):
    return F.normalize(x.float(), dim=-1, eps=1e-8)


def signature(features, anchors):
    return norm(features) @ norm(anchors).t()


def structure(patches):
    z = norm(patches)
    return (1 - z @ z.transpose(-1, -2)).clamp(0, 2)


def multi_positive_loss(sketch, photo, labels, temperature=0.07):
    logits = norm(sketch) @ norm(photo).t() / temperature
    positive = labels[:, None].eq(labels[None, :])
    if labels.unique().numel() < 2:
        return logits.sum() * 0  # No negative class: do not impose uniformity.
    def direction(scores, mask):
        return -(F.log_softmax(scores, -1) * mask).sum(-1).div(mask.sum(-1)).mean()
    return (direction(logits, positive) + direction(logits.t(), positive.t())) / 2


def semantic_loss(student, teacher, student_anchors, teacher_anchors):
    s = signature(student, student_anchors)
    with torch.no_grad():
        t = signature(teacher, teacher_anchors)
        t = t - t.mean(-1, keepdim=True)
        valid = t.norm(dim=-1) > 1e-6
    s = s - s.mean(-1, keepdim=True)
    values = 1 - (norm(s) * norm(t)).sum(-1)
    return (values * valid).sum() / valid.sum().clamp_min(1)


def contraction_loss(sketch, photo, teacher_sketch, teacher_photo, labels,
                     student_anchors, teacher_anchors, rho=0.5, min_count=2):
    values = []
    for label in labels.unique():
        select = labels == label
        if int(select.sum()) < min_count:
            continue
        # Normalize images before averaging; then normalize each prototype.
        ss = norm(sketch[select]).mean(0, keepdim=True)
        sp = norm(photo[select]).mean(0, keepdim=True)
        delta = signature(ss, student_anchors) - signature(sp, student_anchors)
        with torch.no_grad():
            ts = norm(teacher_sketch[select]).mean(0, keepdim=True)
            tp = norm(teacher_photo[select]).mean(0, keepdim=True)
            target = signature(ts, teacher_anchors) - signature(tp, teacher_anchors)
            bound = rho * target.norm(dim=-1)
        values.append(F.relu(delta.norm(dim=-1) - bound).square().mean())
    if not values:
        return (sketch.sum() + photo.sum()) * 0, 0
    return torch.stack(values).mean(), len(values)


def gw_tensor(ct, cs, plan):
    """Squared-distance GW contraction [B,Nt,Ns], also for inexact marginals."""
    a, b = plan.sum(-1), plan.sum(-2)
    return ((ct.square() @ a.unsqueeze(-1))
            + (cs.square() @ b.unsqueeze(-1)).transpose(-1, -2)
            - 2 * ct @ plan @ cs.transpose(-1, -2))


def semantic_cost(ht, hs):
    # Mean over anchors: changing vocabulary size does not scale the cost.
    return ((ht.square().mean(-1, keepdim=True)
             + hs.square().mean(-1).unsqueeze(-2)
             - 2 * (ht @ hs.transpose(-1, -2)) / ht.shape[-1]).clamp_min(0))


@torch.no_grad()
def sinkhorn(cost, epsilon=0.05, iterations=300, tolerance=1e-4):
    """Log-domain balanced transport with unit total mass, uniform marginals."""
    n, m = cost.shape[-2:]
    logk = -cost / epsilon
    u = torch.zeros_like(cost[..., 0])
    v = torch.zeros_like(cost[..., 0, :])
    for step in range(iterations):
        u = -math.log(n) - torch.logsumexp(logk + v.unsqueeze(-2), dim=-1)
        v = -math.log(m) - torch.logsumexp(logk + u.unsqueeze(-1), dim=-2)
        if step % 10 == 9 or step == iterations - 1:
            plan = (logk + u.unsqueeze(-1) + v.unsqueeze(-2)).exp()
            error = torch.maximum((plan.sum(-1) - 1/n).abs().amax(),
                                  (plan.sum(-2) - 1/m).abs().amax())
            if error <= tolerance:
                break
    if not torch.isfinite(plan).all() or error > tolerance:
        raise RuntimeError(
            f"Sinkhorn marginal error {error.item():.3g}; increase --sinkhorn_iterations "
            "or --transport_epsilon. No unconverged target is silently accepted."
        )
    return plan, error


@torch.no_grad()
def solve_fgw(ct, cs, semantic, alpha=0.5, epsilon=0.05,
              outer_iterations=10, sinkhorn_iterations=300, tolerance=1e-4):
    n, m = ct.shape[-1], cs.shape[-1]
    plan = ct.new_full((ct.shape[0], n, m), 1 / (n*m))
    # Entropic conditional-gradient updates. Line-search evaluates the actual
    # regularized objective, preventing oscillatory full fixed-point updates.
    def objective(p):
        value = alpha * (gw_tensor(ct, cs, p) * p).sum((-1, -2))
        if semantic is not None:
            value = value + (1-alpha) * (semantic*p).sum((-1, -2))
        return value + epsilon * (p * p.clamp_min(1e-30).log()).sum((-1, -2))
    error = ct.new_zeros(())
    last_change = ct.new_zeros(())
    for _ in range(outer_iterations):
        gradient = 2 * alpha * gw_tensor(ct, cs, plan)
        if semantic is not None:
            gradient = gradient + (1-alpha)*semantic
        candidate, error = sinkhorn(gradient, epsilon, sinkhorn_iterations, tolerance)
        best, best_value = plan, objective(plan)
        for rate in (1.0, 0.5, 0.25, 0.125, 0.0625):
            trial = (1-rate)*plan + rate*candidate
            value = objective(trial)
            choose = value < best_value
            best = torch.where(choose[:, None, None], trial, best)
            best_value = torch.minimum(best_value, value)
        last_change = (best-plan).abs().amax()
        plan = best
        if last_change <= tolerance:
            break
    return plan, error, last_change


def transport_loss(ct, student_patches, teacher_semantic=None, student_semantic=None,
                   alpha=0.5, epsilon=0.05, outer_iterations=10,
                   sinkhorn_iterations=300, tolerance=1e-4, return_plan=False):
    ct = ct.detach().float()
    cs = structure(student_patches)
    semantic = None
    if alpha < 1:
        if teacher_semantic is None or student_semantic is None:
            raise ValueError("Semantic signatures are required for alpha < 1.")
        semantic = semantic_cost(teacher_semantic.detach().float(), student_semantic.float())
    plan, marginal, change = solve_fgw(
        ct, cs.detach(), semantic.detach() if semantic is not None else None,
        alpha, epsilon, outer_iterations, sinkhorn_iterations, tolerance,
    )
    structural = (gw_tensor(ct, cs, plan)*plan).sum((-1, -2)).mean()
    sem = (semantic*plan).sum((-1, -2)).mean() if semantic is not None else cs.sum()*0
    # Entropy is constant in the outer gradient because plan is detached.
    entropy = (plan*plan.clamp_min(1e-30).log()).sum((-1, -2)).mean()
    value = alpha*structural + (1-alpha)*sem + epsilon*entropy
    logs = dict(gw_structure=structural.detach(), gw_semantic=sem.detach(),
                       transport_entropy=(-entropy).detach(),
                       transport_marginal_error=marginal, transport_plan_change=change)
    if return_plan:
        logs["_plan"] = plan
    return value, logs


def new_objectives(args, features, labels, student_anchors=None, teacher_anchors=None):
    """Global objectives; original two losses remain in src.losses unchanged."""
    photo, sketch, tp, ts = features[:4]
    total = photo.new_zeros((), dtype=torch.float32)
    logs = {}
    with torch.autocast(device_type=photo.device.type, enabled=False):
        if getattr(args, "lambda_retrieval", 0) > 0:
            value = multi_positive_loss(sketch, photo, labels, args.retrieval_temperature)
            total = total + args.lambda_retrieval * value
            logs["retrieval"] = value.detach()
        if getattr(args, "lambda_semantic", 0) > 0:
            value = (semantic_loss(sketch, ts, student_anchors, teacher_anchors)
                     + semantic_loss(photo, tp, student_anchors, teacher_anchors)) / 2
            total = total + args.lambda_semantic * value
            logs["semantic"] = value.detach()
        if getattr(args, "lambda_contract", 0) > 0:
            value, count = contraction_loss(
                sketch, photo, ts, tp, labels, student_anchors, teacher_anchors,
                args.contract_rho, args.contract_min_count,
            )
            total = total + args.lambda_contract * value
            logs["contract"] = value.detach()
            logs["contract_valid_classes"] = photo.new_tensor(count)
    return total, logs
