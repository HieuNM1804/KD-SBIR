"""Synchronized wall-clock accounting around the unchanged main training methods."""
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
from functools import wraps
import csv
import inspect
import json
from pathlib import Path
import time
from unittest.mock import patch

import torch


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def synchronize():
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


class TrainingCostRecorder:
    def __init__(self, output, metadata=None, clock=time.perf_counter, sync=synchronize):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.clock, self.sync = clock, sync
        self.active = []
        self.started = {}
        self.report = dict(status="running", started_utc=utc_now(), gpu_count=1,
                           metadata=metadata or {}, events=[],
                           scope="single-GPU synchronized wall time, not GPU kernel busy time",
                           accounting="validation is nested in adaptation/fit; do not sum overlapping rows")
        self.save()

    def begin(self, name, **details):
        self.sync()
        event = dict(id=len(self.report["events"]), name=name,
                     parent_id=self.active[-1] if self.active else None,
                     status="running", started_utc=utc_now(), elapsed_seconds=None, **details)
        self.report["events"].append(event)
        self.active.append(event["id"])
        self.save()
        self.started[event["id"]] = self.clock()
        print(f"[Training Cost] start: {name}", flush=True)
        return event

    def end(self, event, status="completed", error=None):
        if event["id"] not in self.started:
            return
        # On a failed CUDA call, preserve the original error and partial report.
        try:
            self.sync()
        except RuntimeError as sync_error:
            event["synchronization_error"] = str(sync_error)
        event.update(status=status, finished_utc=utc_now(),
                     elapsed_seconds=max(0.0, self.clock() - self.started.pop(event["id"])))
        if error:
            event["error"] = error
        self.active.remove(event["id"])
        self.save()
        print(f"[Training Cost] {status}: {event['name']} = {event['elapsed_seconds']:.2f} s", flush=True)

    @contextmanager
    def stage(self, name, **details):
        event = self.begin(name, **details)
        try:
            yield event
        except BaseException as error:
            status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            self.end(event, status, f"{type(error).__name__}: {error}")
            raise
        else:
            self.end(event, event["status"] if event["status"] != "running" else "completed")

    def summary(self):
        def rows(name):
            return [e for e in self.report["events"] if e["name"] == name
                    and e["elapsed_seconds"] is not None]
        def total(name):
            found = rows(name)
            return sum(e["elapsed_seconds"] for e in found) if found else None
        names = ("pipeline", "model_initialization", "teacher_adaptation", "teacher_validation",
                 "teacher_preparation", "teacher_cache_load", "student_distillation",
                 "student_validation", "student_sanity_validation")
        values = {name: total(name) for name in names}
        preparation = values.pop("teacher_preparation")
        # Preparation wraps adaptation + cache generation/loading. Subtract nested
        # teacher adaptation and cache load ONCE to obtain a non-overlapping cost.
        values["teacher_feature_cache_build"] = (
            max(0.0, preparation - (values["teacher_adaptation"] or 0)
                - (values["teacher_cache_load"] or 0))
            if preparation is not None and not self.report.get("cache_hit", False) else None)
        result = {}
        for name, seconds in values.items():
            events = rows(name)
            status = ("completed" if all(e["status"] == "completed" for e in events)
                      else next(e["status"] for e in events if e["status"] != "completed")) if events else "not_run"
            if name == "teacher_feature_cache_build" and seconds is not None:
                prep_events = rows("teacher_preparation")
                status = prep_events[-1]["status"]
            result[name] = dict(seconds=seconds, gpu_hours=seconds / 3600 if seconds is not None else None,
                                status=status, nested_validation=name.endswith("validation"))
        teacher = values["teacher_adaptation"]
        student = values["student_distillation"]
        cache = values["teacher_feature_cache_build"]
        self.report["full_fresh_training_measurement"] = bool(
            self.report["status"] == "completed" and teacher is not None and student is not None
            and not self.report.get("cache_hit", False) and not self.report.get("student_resumed", False))
        self.report["training_gpu_hours_including_validation_excluding_cache"] = (
            (teacher + student) / 3600 if teacher is not None and student is not None else None)
        self.report["training_and_cache_gpu_hours"] = (
            (teacher + student + cache) / 3600
            if teacher is not None and student is not None and cache is not None else None)
        return result

    def save(self):
        self.report["summary"] = self.summary()
        target = self.output / "training_cost.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.report, indent=2, allow_nan=False, default=str)
                             + "\n", encoding="utf-8")
        temporary.replace(target)
        with (self.output / "training_cost.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["stage", "seconds", "gpu_hours", "status", "nested_validation"])
            writer.writeheader()
            writer.writerows(dict(stage=name, **values) for name, values in self.report["summary"].items())

    def finish(self, status, error=None):
        for event_id in list(reversed(self.active)):
            self.end(self.report["events"][event_id], status, error)
        self.report.update(status=status, finished_utc=utc_now())
        if error:
            self.report["error"] = error
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            try:
                self.report["cuda_peak_allocated_mib_process"] = torch.cuda.max_memory_allocated() / 2**20
            except RuntimeError as memory_error:
                self.report["cuda_memory_error"] = str(memory_error)
        self.save()


def scalar_metrics(metrics):
    result = {}
    for name in ("mAP", "precision", "train_loss"):
        value = metrics.get(name)
        if value is not None:
            value = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
            # Failed numeric runs remain inspectable without invalid JSON NaNs.
            result[name] = value if float("-inf") < value < float("inf") else None
    return result


def validation_callback(recorder, callback_base):
    class CostValidationCallback(callback_base):
        def on_validation_start(self, trainer, pl_module):
            name = "student_sanity_validation" if trainer.sanity_checking else "student_validation"
            self.event = recorder.begin(name, epoch=int(trainer.current_epoch))

        def on_validation_end(self, trainer, pl_module):
            self.event["metrics"] = scalar_metrics(trainer.callback_metrics)
            recorder.end(self.event)

    return CostValidationCallback()


@contextmanager
def instrument_training(recorder, model_module, trainer_class, callback_base):
    """Wrap original callables without changing their arguments, outputs or gradients."""
    custom = model_module.CustomCLIP

    def timed(original, name, after=None):
        @wraps(original)
        def wrapper(*args, **kwargs):
            with recorder.stage(name) as event:
                result = original(*args, **kwargs)
                if after:
                    after(args, kwargs, event)
                return result
        return wrapper

    init = model_module.ZS_SBIR.__init__
    @wraps(init)
    def model_init(*args, **kwargs):
        bound = inspect.signature(init).bind(*args, **kwargs)
        config = bound.arguments["args"]
        recorder.report["training_args"] = dict(vars(config))
        recorder.report["student_resumed"] = bool(getattr(config, "ckpt_path", ""))
        with recorder.stage("model_initialization"):
            return init(*args, **kwargs)

    prepare = custom.cache_teacher_features
    @wraps(prepare)
    def preparation(*args, **kwargs):
        bound = inspect.signature(prepare).bind(*args, **kwargs).arguments
        instance = args[0]
        dataset = bound["train_dataset"]
        recorder.report["cache_hit"] = bool(instance.persistent_teacher_cache)
        recorder.report["dataset"] = dict(
            seen_classes=list(dataset.all_categories), seen_sketches=len(dataset.all_sketches_path),
            seen_photos=len(dataset.all_photo_paths),
            validation_sketches=len(bound["val_sketch_loader"].dataset),
            validation_photos=len(bound["val_photo_loader"].dataset))
        with recorder.stage("teacher_preparation"):
            return prepare(*args, **kwargs)

    def teacher_done(args, kwargs, event):
        instance = args[0]
        recorder.report["teacher_best_epoch"] = getattr(instance, "teacher_best_epoch", None)
        recorder.report["teacher_best_precision"] = getattr(instance, "teacher_best_precision", None)

    trainer_init = trainer_class.__init__
    @wraps(trainer_init)
    def make_trainer(*args, **kwargs):
        callbacks = list(kwargs.get("callbacks") or [])
        callbacks.append(validation_callback(recorder, callback_base))
        kwargs = dict(kwargs, callbacks=callbacks)
        return trainer_init(*args, **kwargs)

    fit = trainer_class.fit
    @wraps(fit)
    def fit_student(*args, **kwargs):
        trainer = args[0]
        with recorder.stage("student_distillation") as event:
            result = fit(*args, **kwargs)
            if getattr(trainer, "interrupted", False):
                event["status"] = "interrupted"
                recorder.report["trainer_interrupted"] = True
            recorder.report["student_metrics"] = scalar_metrics(trainer.callback_metrics)
            checkpoint = trainer.checkpoint_callback
            recorder.report["student_checkpoint"] = dict(
                best_model_path=checkpoint.best_model_path,
                last_model_path=checkpoint.last_model_path,
                best_model_score=scalar_metrics({"precision": checkpoint.best_model_score}).get("precision"))
            return result

    teacher_validation = custom._validate_teacher_unseen
    # Bind epoch against the ORIGINAL signature, not the installed wrapper.
    def with_epoch(args, kwargs, event):
        event["epoch"] = inspect.signature(teacher_validation).bind(*args, **kwargs).arguments["epoch"]

    with ExitStack() as stack:
        for cls, attr, replacement in (
            (model_module.ZS_SBIR, "__init__", model_init),
            (custom, "cache_teacher_features", preparation),
            (custom, "_pretrain_teacher_prompts", timed(custom._pretrain_teacher_prompts, "teacher_adaptation", teacher_done)),
            (custom, "_validate_teacher_unseen", timed(teacher_validation, "teacher_validation", with_epoch)),
            (custom, "_load_persistent_teacher_cache", timed(custom._load_persistent_teacher_cache, "teacher_cache_load")),
            (trainer_class, "__init__", make_trainer),
            (trainer_class, "fit", fit_student),
        ):
            stack.enter_context(patch.object(cls, attr, replacement))
        yield


@contextmanager
def local_training_weights(model_module, models_root):
    """Redirect main's original loaders to the already bundled original weights."""
    from src.encoder_cost_models import SPECS
    paths = {key: Path(models_root).resolve() / spec["directory"] / spec["filename"]
             for key, spec in SPECS.items() if key in {"dfn5b_h14", "clip_b32"}}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    create_model = model_module.open_clip.create_model
    def teacher_loader(model_name, *args, **kwargs):
        bound = inspect.signature(create_model).bind(model_name, *args, **kwargs)
        if model_name != "ViT-H-14-quickgelu" or bound.arguments.get("pretrained") != "dfn5b":
            raise ValueError("Training cost supports main's DFN5B ViT-H/14 teacher only")
        bound.arguments["pretrained"] = str(paths["dfn5b_h14"])
        return create_model(*bound.args, **bound.kwargs)
    def student_loader(backbone):
        if backbone != "ViT-B/32":
            raise ValueError("Training cost supports main's CLIP ViT-B/32 student only")
        return str(paths["clip_b32"])
    with patch.object(model_module.open_clip, "create_model", teacher_loader), \
            patch.object(model_module.clip, "download_model", student_loader):
        yield
