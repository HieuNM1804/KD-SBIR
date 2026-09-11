"""Training-only state; never registered in the student state_dict."""
import math
import torch
from src.dataset import TeacherFeatureDataset, sample_seed
from src.evidence_references import prototypes, scores
from src.evidence_targets import prepare_targets, select_region
from src.evidence_views import intervene
from src.evidence_losses import evidence_loss


class EvidenceRuntime:
    def __init__(self, args, dataset, payload, paths, labels, photo_labels):
        self.args, self.dataset, self.payload = args, dataset, payload
        self.references = TeacherFeatureDataset(paths, dataset.max_size)
        self.reference_labels = labels
        self.photo_labels = photo_labels
        self.rows = {i: row for row, i in enumerate(payload["metadata"]["photo_indices"])}
        self.student_prototypes = None
        self.epoch = 0
        self.donors = TeacherFeatureDataset(dataset.all_photo_paths, dataset.max_size)

    @classmethod
    def prepare(cls, args, student, dataset):
        result = prepare_targets(args, student, dataset)
        runtime = cls(args, dataset, *result)
        if args.evidence_report_dir:
            from src.evidence_report import write_report
            write_report(runtime, args.evidence_report_dir)
        kinds = ["important", "stable"] if args.evidence_region_kind == "both" else [args.evidence_region_kind]
        eligible = sum(any(select_region(runtime.payload["clean"][row], runtime.payload["masked"][row],
                            runtime.payload["cropped"][row], int(runtime.photo_labels[row]), args,
                            torch.Generator().manual_seed(0), kind) is not None for kind in kinds)
                       for row in range(len(runtime.photo_labels)))
        print(f"[Evidence] Eligible photos: {eligible}/{len(runtime.photo_labels)}")
        if eligible == 0 and not args.evidence_prepare_only:
            raise ValueError("No teacher-eligible evidence photos. Inspect the audit before training.")
        return runtime

    @torch.no_grad()
    def refresh(self, student, epoch):
        self.epoch = int(epoch)
        parameter = next(student.parameters())
        features = []
        for start in range(0, len(self.references), self.args.evidence_teacher_batch_size):
            images = torch.stack([self.references[i] for i in range(start, min(len(self.references), start + self.args.evidence_teacher_batch_size))])
            output = student.encode_student_image(images.to(parameter.device), self.args.evidence_reference)
            features.append(output.float())
        self.student_prototypes = prototypes(torch.cat(features), self.reference_labels,
                                             len(self.dataset.all_categories))
        print(f"[Evidence] Student {self.args.evidence_reference} references refreshed for epoch {epoch}")

    def loss(self, student, batch, features, batch_idx):
        if self.student_prototypes is None:
            raise RuntimeError("Evidence student references were not refreshed")
        args, payload = self.args, self.payload
        generator = torch.Generator().manual_seed(sample_seed(args.seed + 603, self.epoch, batch_idx))
        count = max(1, math.ceil(len(batch[0]) * args.evidence_batch_fraction))
        # Select positions before eligibility filtering; rejected samples are
        # logged instead of silently increasing the number of forward passes.
        positions = torch.randperm(len(batch[0]), generator=generator)[:count].tolist()
        indices, changed, rows, chosen = [], [], [], []
        for pos in positions:
            row = self.rows.get(int(batch[5][pos]))
            if row is None:
                continue
            kind = args.evidence_region_kind
            if kind == "both":
                kind = "important" if torch.randint(2, (), generator=generator).item() else "stable"
            region = select_region(payload["clean"][row], payload["masked"][row],
                                   payload["cropped"][row], int(self.photo_labels[row]), args, generator, kind)
            if region is None:
                continue
            donor = None
            if args.evidence_fill == "donor":
                donor = self.donors[payload["metadata"]["donor_indices"][row]]
            view, _ = intervene(batch[0][pos].detach().float().cpu(), payload["metadata"]["boxes"][region], args.evidence_fill, donor)
            indices.append(pos)
            changed.append(view)
            rows.append(row)
            chosen.append(region)
        zero = features[0].sum() * 0
        logs = {"evidence_selected_fraction": len(indices) / len(positions),
                "evidence_samples": float(len(indices)), "evidence_loss": zero.detach(),
                "evidence_teacher_rms": zero.detach(), "evidence_student_rms": zero.detach()}
        if not indices:
            return zero, logs
        device = features[0].device
        masked_features = student.encode_student_image(torch.stack(changed).to(device), "photo")
        clean_q = scores(features[0][indices], self.student_prototypes)
        masked_q = scores(masked_features, self.student_prototypes)
        clean_t = payload["clean"][rows].to(device)
        masked_t = payload["masked"][rows, chosen].to(device)
        loss = evidence_loss(clean_q, masked_q, clean_t, masked_t, args.evidence_objective)
        logs.update(evidence_loss=loss.detach(),
                    evidence_teacher_rms=(clean_t - masked_t).square().mean().sqrt(),
                    evidence_student_rms=(clean_q - masked_q).detach().square().mean().sqrt())
        return loss, logs
