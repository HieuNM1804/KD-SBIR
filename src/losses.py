"""Losses used by the DFN5B-to-CLIP SBIR benchmark."""

import torch
from torch.nn import functional as F


def relational_kd_loss(
    student_sketch,
    student_photo,
    teacher_sketch,
    teacher_photo,
    temperature=0.07,
):
    """Match the student and teacher sketch-to-photo similarity distributions."""
    student_device = student_sketch.device
    student_sketch = F.normalize(student_sketch.float(), dim=-1)
    student_photo = F.normalize(student_photo.float(), dim=-1)

    student_logits = student_sketch @ student_photo.t() / temperature
    student_log_probs = F.log_softmax(student_logits, dim=-1)

    with torch.no_grad():
        teacher_sketch = F.normalize(
            teacher_sketch.to(device=student_device, dtype=torch.float32),
            dim=-1,
        )
        teacher_photo = F.normalize(
            teacher_photo.to(device=student_device, dtype=torch.float32),
            dim=-1,
        )
        teacher_logits = teacher_sketch @ teacher_photo.t() / temperature
        teacher_probs = F.softmax(teacher_logits, dim=-1)

    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")


def image_text_kd_loss(
    student_image,
    student_text,
    teacher_image,
    teacher_text,
    temperature=0.1,
):
    """Symmetrically match teacher and student class distributions."""
    student_image = F.normalize(student_image.float(), dim=-1)
    student_text = F.normalize(student_text.float(), dim=-1)
    student_log_probs = F.log_softmax(
        student_image @ student_text.t() / temperature,
        dim=-1,
    )
    student_probs = student_log_probs.exp()

    with torch.no_grad():
        teacher_image = F.normalize(
            teacher_image.to(student_image.device, dtype=torch.float32),
            dim=-1,
        )
        teacher_text = F.normalize(
            teacher_text.to(student_image.device, dtype=torch.float32),
            dim=-1,
        )
        teacher_log_probs = F.log_softmax(
            teacher_image @ teacher_text.t() / temperature,
            dim=-1,
        )
        teacher_probs = teacher_log_probs.exp()

    teacher_to_student = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
    )
    student_to_teacher = (
        (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1).mean()
    )
    return 0.5 * (teacher_to_student + student_to_teacher)


def batch_hard_teacher_triplet_loss(
    sketch_features,
    photo_features,
    labels,
    margin=0.2,
):
    """Symmetric batch-hard triplet loss for jointly trained teacher adapters."""
    sketch_features = F.normalize(sketch_features.float(), dim=-1)
    photo_features = F.normalize(photo_features.float(), dim=-1)
    labels = labels.to(sketch_features.device)

    distance = 1.0 - sketch_features @ photo_features.t()
    positive_mask = labels[:, None].eq(labels[None, :])
    negative_mask = ~positive_mask

    def one_direction(dist):
        valid_negative = negative_mask.any(dim=-1)
        hardest_positive = (
            dist.masked_fill(~positive_mask, -torch.inf).max(dim=-1).values
        )
        hardest_negative = (
            dist.masked_fill(~negative_mask, torch.inf).min(dim=-1).values
        )
        losses = F.relu(hardest_positive - hardest_negative + margin)
        if valid_negative.any():
            return losses[valid_negative].mean()
        return dist.new_zeros(())

    return 0.5 * (one_direction(distance) + one_direction(distance.t()))


def _controlled_teacher_delta(delta, control):
    if control == "verified":
        return delta
    if control == "shuffled":
        # A deterministic identity shuffle preserves the delta distribution but
        # breaks the association between a correction and its anchor.
        return delta.roll(shifts=1, dims=1)
    if control == "reversed":
        return -delta
    raise ValueError(f"Unsupported CoRe control: {control}")


def _margin_correction_direction(
    prompted_query,
    prompted_anchor,
    base_student_query,
    base_student_anchor,
    adapted_teacher_query,
    adapted_teacher_anchor,
    base_teacher_query,
    base_teacher_anchor,
    query_labels,
    anchor_labels,
    teacher_temperature,
    student_temperature,
    hard_negative_topk,
    minimum_teacher_correction,
    maximum_weight,
    control,
):
    """Distill beneficial changes in positive-negative retrieval margins."""
    prompted_query = F.normalize(prompted_query.float(), dim=-1)
    prompted_anchor = F.normalize(prompted_anchor.float(), dim=-1)
    base_student_query = F.normalize(base_student_query.float(), dim=-1)
    base_student_anchor = F.normalize(base_student_anchor.float(), dim=-1)
    student_delta = (
        prompted_query @ prompted_anchor.t()
        - base_student_query @ base_student_anchor.t()
    )

    with torch.no_grad():
        device = prompted_query.device
        adapted_teacher_query = F.normalize(
            adapted_teacher_query.to(device=device, dtype=torch.float32), dim=-1
        )
        adapted_teacher_anchor = F.normalize(
            adapted_teacher_anchor.to(device=device, dtype=torch.float32), dim=-1
        )
        base_teacher_query = F.normalize(
            base_teacher_query.to(device=device, dtype=torch.float32), dim=-1
        )
        base_teacher_anchor = F.normalize(
            base_teacher_anchor.to(device=device, dtype=torch.float32), dim=-1
        )
        base_teacher_scores = base_teacher_query @ base_teacher_anchor.t()
        teacher_delta = (
            adapted_teacher_query @ adapted_teacher_anchor.t() - base_teacher_scores
        )
        teacher_delta = _controlled_teacher_delta(teacher_delta, control)

        query_labels = query_labels.to(device)
        anchor_labels = anchor_labels.to(device)
        positive_mask = query_labels[:, None].eq(anchor_labels[None, :])
        negative_mask = ~positive_mask
        has_positive = positive_mask.any(dim=1)
        has_negative = negative_mask.any(dim=1)

        promoted_positive = teacher_delta.masked_fill(
            ~positive_mask, -torch.inf
        ).argmax(dim=1)

        available_negatives = max(1, int(negative_mask.sum(dim=1).min().item()))
        topk = min(hard_negative_topk, available_negatives)
        hard_negative_indices = (
            base_teacher_scores.masked_fill(~negative_mask, -torch.inf)
            .topk(topk, dim=1)
            .indices
        )
        hard_negative_deltas = teacher_delta.gather(1, hard_negative_indices)
        suppressed_offset = hard_negative_deltas.argmin(dim=1, keepdim=True)
        suppressed_negative = hard_negative_indices.gather(
            1, suppressed_offset
        ).squeeze(1)

        row = torch.arange(len(query_labels), device=device)
        positive_delta = teacher_delta[row, promoted_positive]
        negative_delta = teacher_delta[row, suppressed_negative]
        teacher_correction = positive_delta - negative_delta
        valid = (
            has_positive
            & has_negative
            & teacher_correction.gt(minimum_teacher_correction)
        )

    if not valid.any():
        zero = student_delta.sum() * 0.0
        return zero, {
            "coverage": zero.detach(),
            "teacher_correction": zero.detach(),
            "student_correction": zero.detach(),
            "promote": zero.detach(),
            "suppress": zero.detach(),
            "agreement": zero.detach(),
        }

    row = torch.arange(len(query_labels), device=student_delta.device)
    student_correction = (
        student_delta[row, promoted_positive] - student_delta[row, suppressed_negative]
    )
    teacher_target = torch.sigmoid(teacher_correction.detach() / teacher_temperature)
    pair_loss = F.binary_cross_entropy_with_logits(
        student_correction / student_temperature,
        teacher_target,
        reduction="none",
    )
    weights = teacher_correction.detach().clamp(min=0.0, max=maximum_weight)
    valid_weights = weights[valid]
    valid_weights = valid_weights / valid_weights.mean().clamp_min(1e-6)
    loss = (pair_loss[valid] * valid_weights).mean()

    valid_teacher = teacher_correction[valid]
    valid_student = student_correction[valid]
    return loss, {
        "coverage": valid.float().mean(),
        "teacher_correction": valid_teacher.mean(),
        "student_correction": valid_student.detach().mean(),
        "promote": positive_delta[valid].mean(),
        "suppress": (-negative_delta[valid]).mean(),
        "agreement": valid_student.detach().gt(0).float().mean(),
    }


def cross_modal_margin_correction_loss(
    prompted_photo,
    prompted_sketch,
    base_student_photo,
    base_student_sketch,
    adapted_teacher_photo,
    adapted_teacher_sketch,
    base_teacher_photo,
    base_teacher_sketch,
    labels,
    teacher_temperature=0.05,
    student_temperature=0.05,
    hard_negative_topk=8,
    minimum_teacher_correction=0.0,
    maximum_weight=0.25,
    control="verified",
    direction="bidirectional",
):
    """Transfer adaptation-induced margin corrections in both modalities."""
    common = {
        "query_labels": labels,
        "anchor_labels": labels,
        "teacher_temperature": teacher_temperature,
        "student_temperature": student_temperature,
        "hard_negative_topk": hard_negative_topk,
        "minimum_teacher_correction": minimum_teacher_correction,
        "maximum_weight": maximum_weight,
        "control": control,
    }
    results = []
    if direction in {"bidirectional", "sketch_to_photo"}:
        results.append(
            _margin_correction_direction(
                prompted_sketch,
                prompted_photo,
                base_student_sketch,
                base_student_photo,
                adapted_teacher_sketch,
                adapted_teacher_photo,
                base_teacher_sketch,
                base_teacher_photo,
                **common,
            )
        )
    if direction in {"bidirectional", "photo_to_sketch"}:
        results.append(
            _margin_correction_direction(
                prompted_photo,
                prompted_sketch,
                base_student_photo,
                base_student_sketch,
                adapted_teacher_photo,
                adapted_teacher_sketch,
                base_teacher_photo,
                base_teacher_sketch,
                **common,
            )
        )
    if not results:
        raise ValueError(f"Unsupported CoRe direction: {direction}")

    loss = torch.stack([value[0] for value in results]).mean()
    diagnostics = {
        name: torch.stack([value[1][name] for value in results]).mean()
        for name in results[0][1]
    }
    return loss, diagnostics


def _gap_margin_direction(
    full_student_query,
    full_student_gallery,
    common_student_query,
    common_student_gallery,
    full_teacher_query,
    full_teacher_gallery,
    common_teacher_query,
    common_teacher_gallery,
    query_labels,
    gallery_labels,
    huber_beta,
    minimum_correction,
    maximum_weight,
    control,
):
    """Match full-minus-common margin change on fixed common-state pairs."""
    full_student_query = F.normalize(full_student_query.float(), dim=-1)
    full_student_gallery = F.normalize(full_student_gallery.float(), dim=-1)
    common_student_query = F.normalize(common_student_query.float(), dim=-1)
    common_student_gallery = F.normalize(common_student_gallery.float(), dim=-1)

    device = full_student_query.device
    with torch.no_grad():
        full_teacher_query = F.normalize(
            full_teacher_query.to(device=device, dtype=torch.float32), dim=-1
        )
        full_teacher_gallery = F.normalize(
            full_teacher_gallery.to(device=device, dtype=torch.float32), dim=-1
        )
        common_teacher_query = F.normalize(
            common_teacher_query.to(device=device, dtype=torch.float32), dim=-1
        )
        common_teacher_gallery = F.normalize(
            common_teacher_gallery.to(device=device, dtype=torch.float32), dim=-1
        )
        common_scores = common_teacher_query @ common_teacher_gallery.t()
        full_scores = full_teacher_query @ full_teacher_gallery.t()
        query_labels = query_labels.to(device)
        gallery_labels = gallery_labels.to(device)
        positive_mask = query_labels[:, None].eq(gallery_labels[None, :])
        negative_mask = ~positive_mask
        has_positive = positive_mask.any(dim=1)
        has_negative = negative_mask.any(dim=1)
        positive_index = common_scores.masked_fill(
            ~positive_mask, -torch.inf
        ).argmax(dim=1)
        negative_index = common_scores.masked_fill(
            ~negative_mask, -torch.inf
        ).argmax(dim=1)
        row = torch.arange(len(query_labels), device=device)
        common_margin = (
            common_scores[row, positive_index]
            - common_scores[row, negative_index]
        )
        full_margin = (
            full_scores[row, positive_index]
            - full_scores[row, negative_index]
        )
        verified_correction = full_margin - common_margin
        valid = (
            has_positive
            & has_negative
            & verified_correction.gt(minimum_correction)
        )
        if control == "verified":
            teacher_target = verified_correction
        elif control == "shuffled":
            teacher_target = verified_correction.roll(shifts=1, dims=0)
        elif control == "reversed":
            teacher_target = -verified_correction
        else:
            raise ValueError(f"Unsupported Gap-CoRe control: {control}")

    full_student_scores = full_student_query @ full_student_gallery.t()
    common_student_scores = common_student_query @ common_student_gallery.t()
    student_correction = (
        full_student_scores[row, positive_index]
        - full_student_scores[row, negative_index]
        - common_student_scores[row, positive_index]
        + common_student_scores[row, negative_index]
    )
    if not valid.any():
        zero = student_correction.sum() * 0.0
        return zero, {
            "coverage": zero.detach(),
            "teacher_correction": zero.detach(),
            "student_correction": zero.detach(),
            "agreement": zero.detach(),
            "absolute_error": zero.detach(),
            "common_margin": zero.detach(),
            "full_margin": zero.detach(),
        }

    pair_loss = F.smooth_l1_loss(
        student_correction,
        teacher_target.detach(),
        beta=huber_beta,
        reduction="none",
    )
    weights = verified_correction.detach().clamp(
        min=0.0,
        max=maximum_weight,
    )
    valid_weights = weights[valid]
    valid_weights = valid_weights / valid_weights.mean().clamp_min(1e-6)
    loss = (pair_loss[valid] * valid_weights).mean()

    valid_teacher = teacher_target[valid]
    valid_student = student_correction[valid]
    return loss, {
        "coverage": valid.float().mean(),
        "teacher_correction": valid_teacher.mean(),
        "student_correction": valid_student.detach().mean(),
        "agreement": (
            valid_student.detach().sign().eq(valid_teacher.sign()).float().mean()
        ),
        "absolute_error": (
            valid_student.detach() - valid_teacher
        ).abs().mean(),
        "common_margin": common_margin[valid].mean(),
        "full_margin": full_margin[valid].mean(),
    }


def gap_core_margin_correction_loss(
    full_student_photo,
    full_student_sketch,
    common_student_photo,
    common_student_sketch,
    full_teacher_photo,
    full_teacher_sketch,
    common_teacher_photo,
    common_teacher_sketch,
    labels,
    huber_beta=0.05,
    minimum_correction=0.0,
    maximum_weight=0.25,
    control="verified",
    direction="bidirectional",
):
    common = {
        "query_labels": labels,
        "gallery_labels": labels,
        "huber_beta": huber_beta,
        "minimum_correction": minimum_correction,
        "maximum_weight": maximum_weight,
        "control": control,
    }
    results = []
    if direction in {"bidirectional", "sketch_to_photo"}:
        results.append(
            _gap_margin_direction(
                full_student_sketch,
                full_student_photo,
                common_student_sketch,
                common_student_photo,
                full_teacher_sketch,
                full_teacher_photo,
                common_teacher_sketch,
                common_teacher_photo,
                **common,
            )
        )
    if direction in {"bidirectional", "photo_to_sketch"}:
        results.append(
            _gap_margin_direction(
                full_student_photo,
                full_student_sketch,
                common_student_photo,
                common_student_sketch,
                full_teacher_photo,
                full_teacher_sketch,
                common_teacher_photo,
                common_teacher_sketch,
                **common,
            )
        )
    if not results:
        raise ValueError(f"Unsupported Gap-CoRe direction: {direction}")
    return (
        torch.stack([value[0] for value in results]).mean(),
        {
            name: torch.stack([value[1][name] for value in results]).mean()
            for name in results[0][1]
        },
    )


def _counterfactual_rank_direction(
    full_student_query,
    full_student_gallery,
    common_student_query,
    common_student_gallery,
    swapped_student_query,
    swapped_student_gallery,
    full_teacher_query,
    full_teacher_gallery,
    common_teacher_query,
    common_teacher_gallery,
    swapped_teacher_query,
    swapped_teacher_gallery,
    query_labels,
    gallery_labels,
    hard_negative_topk,
    huber_beta,
    minimum_full_correction,
    minimum_swap_correction,
    maximum_weight,
    swapped_loss_weight,
    control,
):
    """Distill correct/common/swapped margin effects over common hard negatives."""
    student_features = (
        full_student_query,
        full_student_gallery,
        common_student_query,
        common_student_gallery,
        swapped_student_query,
        swapped_student_gallery,
    )
    (
        full_student_query,
        full_student_gallery,
        common_student_query,
        common_student_gallery,
        swapped_student_query,
        swapped_student_gallery,
    ) = tuple(F.normalize(feature.float(), dim=-1) for feature in student_features)

    device = full_student_query.device
    with torch.no_grad():
        teacher_features = (
            full_teacher_query,
            full_teacher_gallery,
            common_teacher_query,
            common_teacher_gallery,
            swapped_teacher_query,
            swapped_teacher_gallery,
        )
        (
            full_teacher_query,
            full_teacher_gallery,
            common_teacher_query,
            common_teacher_gallery,
            swapped_teacher_query,
            swapped_teacher_gallery,
        ) = tuple(
            F.normalize(feature.to(device=device, dtype=torch.float32), dim=-1)
            for feature in teacher_features
        )
        full_scores = full_teacher_query @ full_teacher_gallery.t()
        common_scores = common_teacher_query @ common_teacher_gallery.t()
        swapped_scores = swapped_teacher_query @ swapped_teacher_gallery.t()
        query_labels = query_labels.to(device)
        gallery_labels = gallery_labels.to(device)
        positive_mask = query_labels[:, None].eq(gallery_labels[None, :])
        negative_mask = ~positive_mask
        has_positive = positive_mask.any(dim=1)
        positive_index = common_scores.masked_fill(
            ~positive_mask, -torch.inf
        ).argmax(dim=1)
        negative_candidates = common_scores.masked_fill(
            ~negative_mask, -torch.inf
        )
        topk = min(hard_negative_topk, negative_candidates.shape[1])
        negative_values, negative_index = negative_candidates.topk(topk, dim=1)
        finite_negative = torch.isfinite(negative_values)
        row = torch.arange(len(query_labels), device=device)[:, None]

        def fixed_margins(scores):
            positive = scores[
                torch.arange(len(query_labels), device=device), positive_index
            ][:, None]
            negatives = scores[row, negative_index]
            return positive - negatives

        full_margin = fixed_margins(full_scores)
        common_margin = fixed_margins(common_scores)
        swapped_margin = fixed_margins(swapped_scores)
        full_correction = full_margin - common_margin
        swap_correction = common_margin - swapped_margin
        feasible = has_positive[:, None] & finite_negative
        monotonic = feasible & full_correction.gt(0) & swap_correction.gt(0)
        verified_valid = (
            feasible
            & full_correction.gt(minimum_full_correction)
            & swap_correction.gt(minimum_swap_correction)
        )
        if control == "verified":
            target_full = full_correction
            target_swap = swap_correction
            valid = verified_valid
            weight_full = full_correction
            weight_swap = swap_correction
        elif control == "shuffled":
            target_full = full_correction.roll(shifts=1, dims=0)
            target_swap = swap_correction.roll(shifts=1, dims=0)
            valid = verified_valid.roll(shifts=1, dims=0)
            weight_full = target_full
            weight_swap = target_swap
        elif control == "reversed":
            target_full = -full_correction
            target_swap = -swap_correction
            valid = verified_valid
            weight_full = full_correction
            weight_swap = swap_correction
        else:
            raise ValueError(f"Unsupported CGRD control: {control}")

    full_student_scores = full_student_query @ full_student_gallery.t()
    common_student_scores = common_student_query @ common_student_gallery.t()
    swapped_student_scores = swapped_student_query @ swapped_student_gallery.t()

    def student_fixed_margins(scores):
        positive = scores[
            torch.arange(len(query_labels), device=device), positive_index
        ][:, None]
        negatives = scores[row, negative_index]
        return positive - negatives

    full_student_margin = student_fixed_margins(full_student_scores)
    common_student_margin = student_fixed_margins(common_student_scores)
    swapped_student_margin = student_fixed_margins(swapped_student_scores)
    student_full_correction = full_student_margin - common_student_margin
    student_swap_correction = common_student_margin - swapped_student_margin
    if not valid.any():
        zero = (
            student_full_correction.sum() + student_swap_correction.sum()
        ) * 0.0
        return zero, {
            "coverage": zero.detach(),
            "query_coverage": zero.detach(),
            "monotonicity": monotonic.float().mean(),
            "teacher_full_correction": zero.detach(),
            "teacher_swap_correction": zero.detach(),
            "student_full_correction": zero.detach(),
            "student_swap_correction": zero.detach(),
            "agreement": zero.detach(),
            "absolute_error": zero.detach(),
            "full_margin": zero.detach(),
            "common_margin": zero.detach(),
            "swapped_margin": zero.detach(),
        }

    full_loss = F.smooth_l1_loss(
        student_full_correction,
        target_full.detach(),
        beta=huber_beta,
        reduction="none",
    )
    swap_loss = F.smooth_l1_loss(
        student_swap_correction,
        target_swap.detach(),
        beta=huber_beta,
        reduction="none",
    )
    evidence = torch.minimum(
        weight_full.detach().clamp(min=0.0, max=maximum_weight),
        weight_swap.detach().clamp(min=0.0, max=maximum_weight),
    )
    valid_weights = evidence[valid]
    valid_weights = valid_weights / valid_weights.mean().clamp_min(1e-6)
    pair_loss = full_loss + swapped_loss_weight * swap_loss
    loss = (pair_loss[valid] * valid_weights).mean()

    valid_target_full = target_full[valid]
    valid_target_swap = target_swap[valid]
    valid_student_full = student_full_correction[valid]
    valid_student_swap = student_swap_correction[valid]
    agreement = 0.5 * (
        valid_student_full.detach().sign().eq(valid_target_full.sign()).float().mean()
        + valid_student_swap.detach().sign().eq(valid_target_swap.sign()).float().mean()
    )
    absolute_error = 0.5 * (
        (valid_student_full.detach() - valid_target_full).abs().mean()
        + (valid_student_swap.detach() - valid_target_swap).abs().mean()
    )
    return loss, {
        "coverage": valid.float().mean(),
        "query_coverage": valid.any(dim=1).float().mean(),
        "monotonicity": monotonic.float().mean(),
        "teacher_full_correction": valid_target_full.mean(),
        "teacher_swap_correction": valid_target_swap.mean(),
        "student_full_correction": valid_student_full.detach().mean(),
        "student_swap_correction": valid_student_swap.detach().mean(),
        "agreement": agreement,
        "absolute_error": absolute_error,
        "full_margin": full_margin[valid].mean(),
        "common_margin": common_margin[valid].mean(),
        "swapped_margin": swapped_margin[valid].mean(),
    }


def counterfactual_gap_ranking_loss(
    full_student_photo,
    full_student_sketch,
    common_student_photo,
    common_student_sketch,
    swapped_student_photo,
    swapped_student_sketch,
    full_teacher_photo,
    full_teacher_sketch,
    common_teacher_photo,
    common_teacher_sketch,
    swapped_teacher_photo,
    swapped_teacher_sketch,
    labels,
    hard_negative_topk=8,
    huber_beta=0.02,
    minimum_full_correction=0.0,
    minimum_swap_correction=0.0,
    maximum_weight=0.25,
    swapped_loss_weight=1.0,
    control="verified",
    direction="bidirectional",
):
    common = {
        "query_labels": labels,
        "gallery_labels": labels,
        "hard_negative_topk": hard_negative_topk,
        "huber_beta": huber_beta,
        "minimum_full_correction": minimum_full_correction,
        "minimum_swap_correction": minimum_swap_correction,
        "maximum_weight": maximum_weight,
        "swapped_loss_weight": swapped_loss_weight,
        "control": control,
    }
    results = []
    if direction in {"bidirectional", "sketch_to_photo"}:
        results.append(
            _counterfactual_rank_direction(
                full_student_sketch,
                full_student_photo,
                common_student_sketch,
                common_student_photo,
                swapped_student_sketch,
                swapped_student_photo,
                full_teacher_sketch,
                full_teacher_photo,
                common_teacher_sketch,
                common_teacher_photo,
                swapped_teacher_sketch,
                swapped_teacher_photo,
                **common,
            )
        )
    if direction in {"bidirectional", "photo_to_sketch"}:
        results.append(
            _counterfactual_rank_direction(
                full_student_photo,
                full_student_sketch,
                common_student_photo,
                common_student_sketch,
                swapped_student_photo,
                swapped_student_sketch,
                full_teacher_photo,
                full_teacher_sketch,
                common_teacher_photo,
                common_teacher_sketch,
                swapped_teacher_photo,
                swapped_teacher_sketch,
                **common,
            )
        )
    if not results:
        raise ValueError(f"Unsupported CGRD direction: {direction}")
    return (
        torch.stack([value[0] for value in results]).mean(),
        {
            name: torch.stack([value[1][name] for value in results]).mean()
            for name in results[0][1]
        },
    )


def loss_fn(args, features):
    (
        photo_features,
        sketch_features,
        teacher_photo_features,
        teacher_sketch_features,
        teacher_active,
        student_sketch_text,
        student_photo_text,
        teacher_sketch_text,
        teacher_photo_text,
        base_teacher_photo,
        base_teacher_sketch,
        base_student_photo,
        base_student_sketch,
        labels,
        common_student_photo,
        common_student_sketch,
        common_teacher_photo,
        common_teacher_sketch,
        swapped_student_photo,
        swapped_student_sketch,
        swapped_teacher_photo,
        swapped_teacher_sketch,
    ) = features

    zero = torch.zeros((), device=photo_features.device)

    domain_loss = zero
    if teacher_active and args.lambda_domain > 0:
        domain_loss = relational_kd_loss(
            sketch_features,
            photo_features,
            teacher_sketch_features,
            teacher_photo_features,
            args.kd_temperature,
        )

    photo_text_kd = zero
    sketch_text_kd = zero
    if teacher_active and args.lambda_modality > 0:
        photo_text_kd = image_text_kd_loss(
            photo_features,
            student_photo_text,
            teacher_photo_features,
            teacher_photo_text,
            args.photo_text_kd_temperature,
        )
        sketch_text_kd = image_text_kd_loss(
            sketch_features,
            student_sketch_text,
            teacher_sketch_features,
            teacher_sketch_text,
            args.sketch_text_kd_temperature,
        )
    modality_loss = photo_text_kd + sketch_text_kd

    core_loss = zero
    core_diagnostics = {
        "coverage": zero,
        "teacher_correction": zero,
        "student_correction": zero,
        "promote": zero,
        "suppress": zero,
        "agreement": zero,
    }
    if teacher_active and args.lambda_core > 0:
        core_loss, core_diagnostics = cross_modal_margin_correction_loss(
            prompted_photo=photo_features,
            prompted_sketch=sketch_features,
            base_student_photo=base_student_photo,
            base_student_sketch=base_student_sketch,
            adapted_teacher_photo=teacher_photo_features,
            adapted_teacher_sketch=teacher_sketch_features,
            base_teacher_photo=base_teacher_photo,
            base_teacher_sketch=base_teacher_sketch,
            labels=labels,
            teacher_temperature=args.core_teacher_temperature,
            student_temperature=args.core_student_temperature,
            hard_negative_topk=args.core_hard_negative_topk,
            minimum_teacher_correction=args.core_min_teacher_correction,
            maximum_weight=args.core_max_weight,
            control=args.core_control,
            direction=args.core_direction,
        )

    gap_core_loss = zero
    gap_core_diagnostics = {
        "coverage": zero,
        "teacher_correction": zero,
        "student_correction": zero,
        "agreement": zero,
        "absolute_error": zero,
        "common_margin": zero,
        "full_margin": zero,
    }
    if teacher_active and args.lambda_gap_core > 0:
        gap_core_loss, gap_core_diagnostics = gap_core_margin_correction_loss(
            full_student_photo=photo_features,
            full_student_sketch=sketch_features,
            common_student_photo=common_student_photo,
            common_student_sketch=common_student_sketch,
            full_teacher_photo=teacher_photo_features,
            full_teacher_sketch=teacher_sketch_features,
            common_teacher_photo=common_teacher_photo,
            common_teacher_sketch=common_teacher_sketch,
            labels=labels,
            huber_beta=args.gap_core_huber_beta,
            minimum_correction=args.gap_core_min_correction,
            maximum_weight=args.gap_core_max_weight,
            control=args.gap_core_control,
            direction=args.gap_core_direction,
        )

    cgrd_loss = zero
    cgrd_diagnostics = {
        "coverage": zero,
        "query_coverage": zero,
        "monotonicity": zero,
        "teacher_full_correction": zero,
        "teacher_swap_correction": zero,
        "student_full_correction": zero,
        "student_swap_correction": zero,
        "agreement": zero,
        "absolute_error": zero,
        "full_margin": zero,
        "common_margin": zero,
        "swapped_margin": zero,
    }
    if teacher_active and args.lambda_cgrd > 0:
        cgrd_loss, cgrd_diagnostics = counterfactual_gap_ranking_loss(
            full_student_photo=photo_features,
            full_student_sketch=sketch_features,
            common_student_photo=common_student_photo,
            common_student_sketch=common_student_sketch,
            swapped_student_photo=swapped_student_photo,
            swapped_student_sketch=swapped_student_sketch,
            full_teacher_photo=teacher_photo_features,
            full_teacher_sketch=teacher_sketch_features,
            common_teacher_photo=common_teacher_photo,
            common_teacher_sketch=common_teacher_sketch,
            swapped_teacher_photo=swapped_teacher_photo,
            swapped_teacher_sketch=swapped_teacher_sketch,
            labels=labels,
            hard_negative_topk=args.cgrd_hard_negative_topk,
            huber_beta=args.cgrd_huber_beta,
            minimum_full_correction=args.cgrd_min_full_correction,
            minimum_swap_correction=args.cgrd_min_swap_correction,
            maximum_weight=args.cgrd_max_weight,
            swapped_loss_weight=args.cgrd_swapped_loss_weight,
            control=args.cgrd_control,
            direction=args.cgrd_direction,
        )

    main_objective = (
        args.lambda_domain * domain_loss
        + args.lambda_modality * modality_loss
        + args.lambda_core * core_loss
    )
    gap_objective = args.lambda_gap_core * gap_core_loss
    cgrd_objective = args.lambda_cgrd * cgrd_loss
    total_loss = main_objective + gap_objective + cgrd_objective
    return total_loss, {
        "domain_kd": domain_loss,
        "modality_kd": modality_loss,
        "core_kd": core_loss,
        "gap_core_kd": gap_core_loss,
        "cgrd_kd": cgrd_loss,
        "main_objective": main_objective,
        "gap_objective": gap_objective,
        "cgrd_objective": cgrd_objective,
        **{f"core_{name}": value for name, value in core_diagnostics.items()},
        **{
            f"gap_core_{name}": value
            for name, value in gap_core_diagnostics.items()
        },
        **{f"cgrd_{name}": value for name, value in cgrd_diagnostics.items()},
    }
