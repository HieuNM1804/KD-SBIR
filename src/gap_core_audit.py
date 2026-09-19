"""Teacher-only diagnostics for common/gap prompt decomposition."""

import torch
from torch.nn import functional as F


def _normalized(features):
    return F.normalize(features.float(), dim=-1)


def _shuffle_residual(common, full, seed):
    """Reassign full-minus-common residuals to unrelated image identities."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(len(common), generator=generator)
    residual = full - common
    return _normalized(common + residual[permutation])


def directional_gap_rows(
    full_queries,
    common_queries,
    query_labels,
    full_gallery,
    common_gallery,
    gallery_labels,
    direction,
    shuffle_seed,
):
    """Measure the gap prompt on fixed positive/negative pairs from common state."""
    full_queries = _normalized(full_queries)
    common_queries = _normalized(common_queries)
    full_gallery = _normalized(full_gallery)
    common_gallery = _normalized(common_gallery)
    query_labels = query_labels.cpu()
    gallery_labels = gallery_labels.cpu()

    common_similarity = common_queries @ common_gallery.T
    full_similarity = full_queries @ full_gallery.T
    shuffled_queries = _shuffle_residual(
        common_queries,
        full_queries,
        shuffle_seed,
    )
    shuffled_gallery = _shuffle_residual(
        common_gallery,
        full_gallery,
        shuffle_seed + 1,
    )
    shuffled_similarity = shuffled_queries @ shuffled_gallery.T

    rows = []
    for query_index in range(len(query_labels)):
        positive_mask = gallery_labels.eq(query_labels[query_index])
        negative_mask = ~positive_mask
        if not positive_mask.any() or not negative_mask.any():
            continue

        positive_indices = positive_mask.nonzero(as_tuple=False).flatten()
        negative_indices = negative_mask.nonzero(as_tuple=False).flatten()
        positive_index = positive_indices[
            common_similarity[query_index, positive_indices].argmax()
        ]
        negative_index = negative_indices[
            common_similarity[query_index, negative_indices].argmax()
        ]

        common_positive = common_similarity[query_index, positive_index]
        common_negative = common_similarity[query_index, negative_index]
        full_positive = full_similarity[query_index, positive_index]
        full_negative = full_similarity[query_index, negative_index]
        shuffled_positive = shuffled_similarity[query_index, positive_index]
        shuffled_negative = shuffled_similarity[query_index, negative_index]
        common_margin = common_positive - common_negative
        full_margin = full_positive - full_negative
        shuffled_margin = shuffled_positive - shuffled_negative

        rows.append(
            {
                "direction": direction,
                "query_index": query_index,
                "label": int(query_labels[query_index]),
                "positive_index": int(positive_index),
                "negative_index": int(negative_index),
                "common_positive": float(common_positive),
                "full_positive": float(full_positive),
                "positive_delta": float(full_positive - common_positive),
                "common_negative": float(common_negative),
                "full_negative": float(full_negative),
                "negative_delta": float(full_negative - common_negative),
                "common_margin": float(common_margin),
                "full_margin": float(full_margin),
                "verified_margin_correction": float(full_margin - common_margin),
                "shuffled_margin_correction": float(
                    shuffled_margin - common_margin
                ),
                "verified_minus_shuffled": float(full_margin - shuffled_margin),
                "verified_rank_flip": int(common_margin <= 0 < full_margin),
                "shuffled_rank_flip": int(common_margin <= 0 < shuffled_margin),
            }
        )
    return rows


def build_gap_audit_rows(
    full_sketch,
    common_sketch,
    sketch_labels,
    full_photo,
    common_photo,
    photo_labels,
    seed=42,
):
    rows = directional_gap_rows(
        full_sketch,
        common_sketch,
        sketch_labels,
        full_photo,
        common_photo,
        photo_labels,
        "sketch_to_photo",
        seed,
    )
    rows.extend(
        directional_gap_rows(
            full_photo,
            common_photo,
            photo_labels,
            full_sketch,
            common_sketch,
            sketch_labels,
            "photo_to_sketch",
            seed + 10_000,
        )
    )
    return rows


def bootstrap_mean_interval(values, seed=42, samples=2000):
    values = torch.as_tensor(values, dtype=torch.float64)
    if values.numel() == 0:
        raise ValueError("Cannot bootstrap an empty sample.")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(
        len(values),
        (samples, len(values)),
        generator=generator,
    )
    means = values[indices].mean(dim=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(torch.quantile(means, 0.025)),
        "ci95_high": float(torch.quantile(means, 0.975)),
        "samples": len(values),
    }


def summarize_gap_rows(rows, seed=42, bootstrap_samples=2000):
    if not rows:
        raise ValueError("Gap audit produced no query rows.")
    summary = {"queries": len(rows)}
    names = (
        "positive_delta",
        "negative_delta",
        "verified_margin_correction",
        "shuffled_margin_correction",
        "verified_minus_shuffled",
        "verified_rank_flip",
        "shuffled_rank_flip",
    )
    for index, name in enumerate(names):
        summary[name] = bootstrap_mean_interval(
            [row[name] for row in rows],
            seed=seed + index,
            samples=bootstrap_samples,
        )

    promote = summary["positive_delta"]["mean"]
    suppress = -summary["negative_delta"]["mean"]
    useful_total = max(promote, 0.0) + max(suppress, 0.0)
    summary["positive_contribution_fraction"] = (
        max(promote, 0.0) / useful_total if useful_total > 0 else 0.0
    )
    return summary


def directional_cgrd_rows(
    full_queries,
    common_queries,
    swapped_queries,
    query_labels,
    full_gallery,
    common_gallery,
    swapped_gallery,
    gallery_labels,
    direction,
    hard_negative_topk=8,
):
    """Measure two-sided prompt interventions on fixed common-state pairs."""
    feature_sets = (
        full_queries,
        common_queries,
        swapped_queries,
        full_gallery,
        common_gallery,
        swapped_gallery,
    )
    (
        full_queries,
        common_queries,
        swapped_queries,
        full_gallery,
        common_gallery,
        swapped_gallery,
    ) = tuple(_normalized(features) for features in feature_sets)
    query_labels = query_labels.cpu()
    gallery_labels = gallery_labels.cpu()
    full_similarity = full_queries @ full_gallery.T
    common_similarity = common_queries @ common_gallery.T
    swapped_similarity = swapped_queries @ swapped_gallery.T

    rows = []
    for query_index in range(len(query_labels)):
        positive_mask = gallery_labels.eq(query_labels[query_index])
        negative_mask = ~positive_mask
        if not positive_mask.any() or not negative_mask.any():
            continue
        positive_indices = positive_mask.nonzero(as_tuple=False).flatten()
        negative_indices = negative_mask.nonzero(as_tuple=False).flatten()
        positive_index = positive_indices[
            common_similarity[query_index, positive_indices].argmax()
        ]
        topk = min(hard_negative_topk, len(negative_indices))
        hard_order = common_similarity[query_index, negative_indices].topk(topk).indices
        hard_negative_indices = negative_indices[hard_order]
        for negative_rank, negative_index in enumerate(hard_negative_indices, start=1):
            common_margin = (
                common_similarity[query_index, positive_index]
                - common_similarity[query_index, negative_index]
            )
            full_margin = (
                full_similarity[query_index, positive_index]
                - full_similarity[query_index, negative_index]
            )
            swapped_margin = (
                swapped_similarity[query_index, positive_index]
                - swapped_similarity[query_index, negative_index]
            )
            full_correction = full_margin - common_margin
            swap_correction = common_margin - swapped_margin
            rows.append(
                {
                    "direction": direction,
                    "query_index": query_index,
                    "label": int(query_labels[query_index]),
                    "positive_index": int(positive_index),
                    "negative_index": int(negative_index),
                    "negative_rank": negative_rank,
                    "full_margin": float(full_margin),
                    "common_margin": float(common_margin),
                    "swapped_margin": float(swapped_margin),
                    "full_correction": float(full_correction),
                    "swap_correction": float(swap_correction),
                    "bottleneck_correction": float(
                        torch.minimum(full_correction, swap_correction)
                    ),
                    "monotonic": int(full_correction > 0 and swap_correction > 0),
                }
            )
    return rows


def build_cgrd_audit_rows(
    full_sketch,
    common_sketch,
    swapped_sketch,
    sketch_labels,
    full_photo,
    common_photo,
    swapped_photo,
    photo_labels,
    hard_negative_topk=8,
):
    rows = directional_cgrd_rows(
        full_sketch,
        common_sketch,
        swapped_sketch,
        sketch_labels,
        full_photo,
        common_photo,
        swapped_photo,
        photo_labels,
        "sketch_to_photo",
        hard_negative_topk,
    )
    rows.extend(
        directional_cgrd_rows(
            full_photo,
            common_photo,
            swapped_photo,
            photo_labels,
            full_sketch,
            common_sketch,
            swapped_sketch,
            sketch_labels,
            "photo_to_sketch",
            hard_negative_topk,
        )
    )
    return rows


def summarize_cgrd_rows(rows, seed=42, bootstrap_samples=2000):
    if not rows:
        raise ValueError("CGRD audit produced no query-negative rows.")
    grouped = {}
    for row in rows:
        key = (row["direction"], row["query_index"])
        grouped.setdefault(key, []).append(row)
    summary = {
        "pairs": len(rows),
        "queries": len(grouped),
        "bootstrap_unit": "query",
    }
    for index, name in enumerate(
        (
            "full_correction",
            "swap_correction",
            "bottleneck_correction",
            "monotonic",
        )
    ):
        summary[name] = bootstrap_mean_interval(
            [
                sum(row[name] for row in query_rows) / len(query_rows)
                for query_rows in grouped.values()
            ],
            seed=seed + index,
            samples=bootstrap_samples,
        )
    per_query = [
        sum(row["monotonic"] for row in query_rows) / len(query_rows)
        for query_rows in grouped.values()
    ]
    summary["query_monotonic_fraction"] = bootstrap_mean_interval(
        per_query,
        seed=seed + 10,
        samples=bootstrap_samples,
    )
    return summary
