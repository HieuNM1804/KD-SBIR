"""PromptKD-style distillation over a paired photo-sketch retrieval vocabulary."""

import math

import torch
from torch.nn import functional as F


def select_diverse_candidates(
    features,
    labels,
    class_count,
    candidates_per_class,
):
    """Select deterministic central-then-diverse candidates for every class."""
    if candidates_per_class < 1:
        raise ValueError("candidates_per_class must be at least 1.")
    if len(features) != len(labels):
        raise ValueError("features and labels must have the same length.")

    labels = torch.as_tensor(labels, dtype=torch.long, device="cpu")
    selected = []
    for class_index in range(class_count):
        class_indices = torch.where(labels == class_index)[0]
        if len(class_indices) < candidates_per_class:
            raise ValueError(
                f"Class {class_index} has {len(class_indices)} samples; "
                f"{candidates_per_class} candidates were requested."
            )
        class_features = F.normalize(features[class_indices].float(), dim=-1)
        centroid = F.normalize(class_features.mean(dim=0), dim=0)
        centrality = class_features @ centroid
        first = max(
            range(len(class_indices)),
            key=lambda index: (centrality[index].item(), -index),
        )
        local_selection = [first]
        maximum_similarity = class_features @ class_features[first]
        while len(local_selection) < candidates_per_class:
            available = torch.ones(len(class_indices), dtype=torch.bool)
            available[local_selection] = False
            diversity = 1.0 - maximum_similarity
            score = diversity + 0.05 * centrality
            score[~available] = -torch.inf
            next_index = max(
                range(len(class_indices)),
                key=lambda index: (score[index].item(), -index),
            )
            local_selection.append(next_index)
            maximum_similarity = torch.maximum(
                maximum_similarity,
                class_features @ class_features[next_index],
            )
        selected.append(class_indices[local_selection].clone())
    return torch.stack(selected)


def _mutual_pair_candidates(
    sketch_features,
    photo_features,
    sketch_labels,
    photo_labels,
    mutual_topk,
):
    if mutual_topk < 1:
        raise ValueError("mutual_topk must be at least 1.")
    sketch = F.normalize(sketch_features.float(), dim=-1)
    photo = F.normalize(photo_features.float(), dim=-1)
    sketch_labels = torch.as_tensor(sketch_labels, dtype=torch.long)
    photo_labels = torch.as_tensor(photo_labels, dtype=torch.long)
    similarity = sketch @ photo.t()
    topk = min(mutual_topk, len(sketch), len(photo))
    sketch_top_photo = similarity.topk(topk, dim=1).indices
    photo_top_sketch = similarity.topk(topk, dim=0).indices

    negative = sketch_labels[:, None].ne(photo_labels[None, :])
    sketch_negative = similarity.masked_fill(~negative, -torch.inf).max(dim=1).values
    photo_negative = similarity.masked_fill(~negative, -torch.inf).max(dim=0).values

    pairs = []
    seen = set()
    for sketch_index in range(len(sketch)):
        for photo_index in sketch_top_photo[sketch_index].tolist():
            if sketch_labels[sketch_index] != photo_labels[photo_index]:
                continue
            if not (photo_top_sketch[:, photo_index] == sketch_index).any():
                continue
            key = (sketch_index, photo_index)
            if key in seen:
                continue
            seen.add(key)
            score = similarity[sketch_index, photo_index]
            margin = torch.minimum(
                score - sketch_negative[sketch_index],
                score - photo_negative[photo_index],
            )
            pairs.append((sketch_index, photo_index, score.item(), margin.item()))

    if pairs:
        return pairs, sketch, photo

    raise RuntimeError(
        "No label-consistent mutual teacher retrieval pairs were found. "
        "Increase --retrieval_vocab_mutual_topk or candidate coverage."
    )


def build_retrieval_vocabulary(
    sketch_features,
    photo_features,
    sketch_labels,
    photo_labels,
    vocabulary_size,
    mutual_topk,
    class_count,
):
    """Select reliable, diverse, two-sided landmarks from teacher retrieval."""
    if vocabulary_size < 2:
        raise ValueError("vocabulary_size must be at least 2.")
    pairs, sketch, photo = _mutual_pair_candidates(
        sketch_features,
        photo_features,
        sketch_labels,
        photo_labels,
        mutual_topk,
    )
    if len(pairs) < vocabulary_size:
        raise RuntimeError(
            f"Only {len(pairs)} mutual retrieval pairs are available for a "
            f"vocabulary of {vocabulary_size}. Increase candidates or top-k."
        )

    pair_sketch = torch.tensor([pair[0] for pair in pairs], dtype=torch.long)
    pair_photo = torch.tensor([pair[1] for pair in pairs], dtype=torch.long)
    similarity = torch.tensor([pair[2] for pair in pairs])
    margin = torch.tensor([pair[3] for pair in pairs])
    pair_labels = torch.as_tensor(sketch_labels, dtype=torch.long)[pair_sketch]
    descriptors = F.normalize(sketch[pair_sketch] + photo[pair_photo], dim=-1)
    quality = ((similarity + 1.0) * 0.5).clamp(0.0, 1.0)
    quality = quality * torch.sigmoid(margin / 0.05)

    max_per_class = max(1, math.ceil(2 * vocabulary_size / class_count))
    class_counts = torch.zeros(class_count, dtype=torch.long)
    used_sketch = torch.zeros(len(sketch), dtype=torch.bool)
    used_photo = torch.zeros(len(photo), dtype=torch.bool)
    selected = []
    maximum_similarity = None
    while len(selected) < vocabulary_size:
        available = ~used_sketch[pair_sketch] & ~used_photo[pair_photo]
        available &= class_counts[pair_labels] < max_per_class
        if maximum_similarity is None:
            score = quality.clone()
        else:
            diversity = (1.0 - maximum_similarity).clamp(min=0.0)
            score = quality * (0.25 + 0.75 * diversity)
        score[~available] = -torch.inf
        if not torch.isfinite(score).any():
            raise RuntimeError(
                f"Could select only {len(selected)}/{vocabulary_size} unique "
                "balanced retrieval landmarks. Increase candidate coverage."
            )
        selected_index = max(
            range(len(pairs)),
            key=lambda index: (score[index].item(), -index),
        )
        selected.append(selected_index)
        sketch_index = pair_sketch[selected_index]
        photo_index = pair_photo[selected_index]
        label = pair_labels[selected_index]
        used_sketch[sketch_index] = True
        used_photo[photo_index] = True
        class_counts[label] += 1
        selected_similarity = descriptors @ descriptors[selected_index]
        maximum_similarity = (
            selected_similarity
            if maximum_similarity is None
            else torch.maximum(maximum_similarity, selected_similarity)
        )

    selected = torch.tensor(selected, dtype=torch.long)
    return {
        "sketch_indices": pair_sketch[selected],
        "photo_indices": pair_photo[selected],
        "labels": pair_labels[selected],
        "similarity": similarity[selected],
        "margin": margin[selected],
        "quality": quality[selected],
    }


def directional_vocabulary_kd_loss(
    student_features,
    teacher_features,
    student_landmarks,
    teacher_landmarks,
    student_temperature,
    teacher_temperature,
):
    """Match teacher ranking over landmark identities in one direction."""
    if student_temperature <= 0 or teacher_temperature <= 0:
        raise ValueError("Vocabulary temperatures must be greater than 0.")
    if len(student_landmarks) != len(teacher_landmarks):
        raise ValueError("Teacher and student landmark counts must match.")

    student_features = F.normalize(student_features.float(), dim=-1)
    student_landmarks = F.normalize(student_landmarks.float(), dim=-1)
    student_log_probs = F.log_softmax(
        student_features @ student_landmarks.t() / student_temperature,
        dim=-1,
    )
    with torch.no_grad():
        teacher_features = F.normalize(
            teacher_features.to(student_features.device, dtype=torch.float32),
            dim=-1,
        )
        teacher_landmarks = F.normalize(
            teacher_landmarks.to(student_features.device, dtype=torch.float32),
            dim=-1,
        )
        teacher_probs = F.softmax(
            teacher_features @ teacher_landmarks.t() / teacher_temperature,
            dim=-1,
        )
        entropy = -(teacher_probs * teacher_probs.clamp_min(1e-12).log()).sum(dim=-1)
        confidence = 1.0 - entropy / math.log(teacher_probs.shape[-1])
        confidence = confidence.clamp(min=0.05)

    per_sample = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="none",
    ).sum(dim=-1)
    loss = (confidence * per_sample).sum() / confidence.sum()
    return loss, {
        "teacher_entropy": entropy.mean().detach(),
        "teacher_confidence": confidence.mean().detach(),
    }


def paired_coordinate_consistency(
    sketch_features,
    photo_features,
    photo_landmarks,
    sketch_landmarks,
    temperature,
):
    """Align paired photo/sketch distributions over two-sided landmark IDs."""
    if temperature <= 0:
        raise ValueError("Vocabulary temperature must be greater than 0.")
    sketch = F.normalize(sketch_features.float(), dim=-1)
    photo = F.normalize(photo_features.float(), dim=-1)
    photo_landmarks = F.normalize(photo_landmarks.float(), dim=-1)
    sketch_landmarks = F.normalize(sketch_landmarks.float(), dim=-1)
    sketch_log_probs = F.log_softmax(
        sketch @ photo_landmarks.t() / temperature,
        dim=-1,
    )
    photo_log_probs = F.log_softmax(
        photo @ sketch_landmarks.t() / temperature,
        dim=-1,
    )
    sketch_probs = sketch_log_probs.exp()
    photo_probs = photo_log_probs.exp()
    midpoint = 0.5 * (sketch_probs + photo_probs)
    midpoint_log = midpoint.clamp_min(1e-12).log()
    return 0.5 * (
        F.kl_div(midpoint_log, sketch_probs, reduction="batchmean")
        + F.kl_div(midpoint_log, photo_probs, reduction="batchmean")
    )


def vocabulary_coordinates(features, landmarks, temperature):
    """Return normalized landmark log-probabilities for retrieval inference."""
    features = F.normalize(features.float(), dim=-1)
    landmarks = F.normalize(landmarks.float(), dim=-1)
    coordinates = F.log_softmax(features @ landmarks.t() / temperature, dim=-1)
    coordinates = coordinates - coordinates.mean(dim=-1, keepdim=True)
    return F.normalize(coordinates, dim=-1)
