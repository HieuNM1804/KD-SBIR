"""Fixed training-batch gradient probes without optimizer/RNG side effects."""

import json
import math
from pathlib import Path

import torch
from torch.utils.data import default_collate
from pytorch_lightning import Callback

from src.attention_output_kd import PatchOutputCapture
from src.losses import loss_fn


class AVGradientAudit(Callback):
    def on_train_start(self, trainer, pl_module):
        dataset = trainer.train_dataloader.dataset
        generator = torch.Generator().manual_seed(pl_module.args.seed + 7103)
        self.indices = torch.randperm(len(dataset), generator=generator)[:32].tolist()
        if len(self.indices) < 2:
            raise ValueError("AV gradient audit requires at least two training samples")
        self.batch = default_collate([dataset[(0, i)] for i in self.indices])
        self.records = []
        self.path = Path(trainer.log_dir) / "av_gradient_audit.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.measure(trainer, pl_module, "start")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step in (1, 10, 100):
            self.measure(trainer, pl_module, f"step_{trainer.global_step}")

    def on_train_end(self, trainer, pl_module):
        self.measure(trainer, pl_module, "end")
        print("[AV Gradient Audit] saved:", self.path)

    def measure(self, trainer, module, stage):
        # torch.autograd.grad does not populate/overwrite parameter .grad buffers.
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            batch = [x.to(module.device) for x in self.batch]
            named = [(n, p) for n, p in module.named_parameters()
                     if p.requires_grad and "_visual_prompt." in n]
            if not named:
                raise RuntimeError("No visual prompts found for AV gradient audit")
            params = [p for _, p in named]
            with torch.enable_grad(), PatchOutputCapture(module.model.clip_model.visual) as capture:
                features = module(batch[:5])
                main, _ = loss_fn(module.args, features)
                av = module.av_distillation_loss(
                    capture.values[0], capture.values[1], batch[5], batch[6]
                ) * module.lambda_av

                def gradients(loss):
                    if not loss.requires_grad:
                        return [torch.zeros_like(p, dtype=torch.float32) for p in params]
                    values = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                    return [torch.zeros_like(p, dtype=torch.float32) if g is None
                            else g.detach().float() for p, g in zip(params, values)]

                gm, ga = gradients(main), gradients(av)

            def stats(x, y):
                nx = math.sqrt(sum(t.square().sum().item() for t in x))
                ny = math.sqrt(sum(t.square().sum().item() for t in y))
                dot = sum((a * b).sum().item() for a, b in zip(x, y))
                return {
                    "main_norm": nx, "av_norm": ny,
                    "av_over_main": ny / nx if nx > 1e-12 else None,
                    "cosine": dot / (nx * ny) if nx * ny > 1e-20 else None,
                }

            layers = [{"name": n, **stats([x], [y])}
                      for (n, _), x, y in zip(named, gm, ga)]
            record = {"stage": stage, "global_step": int(trainer.global_step),
                      "epoch": int(trainer.current_epoch),
                      "main_loss": main.detach().item(), "weighted_av_loss": av.detach().item(),
                      **stats(gm, ga), "layers": layers}
            self.records.append(record)
            self.path.write_text(json.dumps({
                "args": dict(vars(module.args)),
                "sample_epoch": 0, "sample_indices": self.indices,
                "notes": "Fixed seen-training batch; raw gradients before clipping/optimizer; no T^2 scaling.",
                "measurements": self.records,
            }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
