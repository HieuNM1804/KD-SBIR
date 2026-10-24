"""Streaming classification metrics; labels are used only to score predictions."""

import math

import torch


class ClassificationMetrics:
    def __init__(self, classnames, bins=15):
        if not classnames or bins < 1:
            raise ValueError("Nonempty classes and positive bin count required.")
        self.names = list(classnames)
        self.bins = bins
        self.confusion = torch.zeros(len(classnames), len(classnames), dtype=torch.int64)
        self.top5_hits = 0
        self.nll_sum = 0.0
        self.predicted, self.confidence, self.correct = [], [], []

    def update(self, logits, labels):
        logits = logits.detach().float().cpu()
        labels = labels.detach().long().cpu()
        if logits.ndim != 2 or logits.shape != (len(labels), len(self.names)):
            raise ValueError("Logit/label/class shapes disagree.")
        if not torch.isfinite(logits).all() or ((labels < 0) | (labels >= len(self.names))).any():
            raise ValueError("Invalid logits or labels.")
        logs = logits.log_softmax(-1)
        probs = logs.exp()
        predicted = logits.argmax(-1)
        confidence = probs.gather(1, predicted[:, None]).squeeze(1)
        ranking = logits.topk(min(5, len(self.names)), dim=-1).indices
        correct = predicted.eq(labels)
        self.confusion += torch.bincount(
            labels * len(self.names) + predicted,
            minlength=len(self.names) ** 2,
        ).reshape(len(self.names), len(self.names))
        self.top5_hits += int(ranking.eq(labels[:, None]).any(-1).sum())
        self.nll_sum -= float(logs.gather(1, labels[:, None]).double().sum())
        self.predicted.append(predicted)
        self.confidence.append(confidence)
        self.correct.append(correct)
        return predicted, confidence, ranking

    def finalize(self):
        support = self.confusion.sum(1)
        predicted_count = self.confusion.sum(0)
        count = int(support.sum())
        if not count:
            raise ValueError("No evaluated images.")
        true_positive = self.confusion.diag().double()
        recall = true_positive / support.clamp_min(1)
        precision = true_positive / predicted_count.clamp_min(1)
        f1 = 2 * true_positive / (support + predicted_count).clamp_min(1)
        represented = support > 0
        per_class = [dict(
            classname=name, support=int(support[i]), predicted_count=int(predicted_count[i]),
            correct=int(true_positive[i]), precision=float(precision[i]),
            recall=float(recall[i]), f1=float(f1[i]),
        ) for i, name in enumerate(self.names)]
        pred = torch.cat(self.predicted)
        conf = torch.cat(self.confidence)
        correct = torch.cat(self.correct)

        def quality(mask):
            retained = int(mask.sum())
            return dict(retained=retained, coverage=retained / count,
                        accuracy=float(correct[mask].double().mean()) if retained else None)

        calibration = []
        ece = 0.0
        bin_index = (conf * self.bins).long().clamp_max(self.bins - 1)
        for index in range(self.bins):
            mask = bin_index.eq(index)
            current = quality(mask)
            mean_confidence = float(conf[mask].double().mean()) if current["retained"] else None
            if mean_confidence is not None:
                ece += current["coverage"] * abs(mean_confidence - current["accuracy"])
            calibration.append(dict(lower=index / self.bins, upper=(index + 1) / self.bins,
                                    mean_confidence=mean_confidence, **current))
        threshold_quality = [dict(threshold=threshold, **quality(conf >= threshold))
                             for threshold in (0.5, 0.7, 0.8, 0.9, 0.95)]
        confidence_order = conf.argsort(descending=True, stable=True)
        coverage_quality = []
        for fraction in (.1, .25, .5, .75, 1.):
            keep = torch.zeros(count, dtype=torch.bool)
            keep[confidence_order[:max(1, math.ceil(count * fraction))]] = True
            coverage_quality.append(dict(target_coverage=fraction, **quality(keep)))
        topk_quality = []
        for k in (16, 64, 256):
            keep = torch.zeros(count, dtype=torch.bool)
            for index in range(len(self.names)):
                positions = pred.eq(index).nonzero().flatten()
                order = conf[positions].argsort(descending=True, stable=True)
                keep[positions[order[:k]]] = True
            topk_quality.append(dict(k_per_predicted_class=k, **quality(keep)))
        metrics = dict(
            images=count, classes=len(self.names), represented_classes=int(represented.sum()),
            accuracy=float(true_positive.sum() / count),
            macro_accuracy=float(recall[represented].mean()),
            macro_f1=float(f1[represented].mean()),
            top5_accuracy=self.top5_hits / count, topk=min(5, len(self.names)),
            nll=self.nll_sum / count, ece=ece,
            mean_confidence=float(conf.double().mean()),
            predicted_classes=int((predicted_count > 0).sum()),
            confidence_thresholds=threshold_quality,
            coverage_targets=coverage_quality,
            topk_per_predicted_class=topk_quality,
            calibration_bins=calibration,
        )
        return metrics, per_class
