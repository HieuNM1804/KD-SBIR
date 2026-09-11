import math


def active(args):
    return getattr(args, "lambda_evidence", 0) > 0 or getattr(args, "evidence_prepare_only", False)


def add_arguments(parser):
    parser.add_argument("--lambda_evidence", type=float, default=0.0)
    parser.add_argument("--evidence_objective", choices=["response", "masked"], default="response")
    parser.add_argument("--evidence_selection", choices=["teacher", "random"], default="teacher")
    parser.add_argument("--evidence_reference", choices=["sketch", "photo"], default="sketch")
    parser.add_argument("--evidence_region_kind", choices=["important", "stable", "both"], default="both")
    parser.add_argument("--evidence_fill", choices=["mean", "donor"], default="mean")
    parser.add_argument("--evidence_area", type=float, default=0.25)
    parser.add_argument("--evidence_min_drop", type=float, default=0.01)
    parser.add_argument("--evidence_stable_rms", type=float, default=0.01)
    parser.add_argument("--evidence_refs_per_class", type=int, default=8)
    parser.add_argument("--evidence_photos_per_class", type=int, default=0,
                        help="0 caches all train photos; use 10 for a small audit")
    parser.add_argument("--evidence_batch_fraction", type=float, default=0.25)
    parser.add_argument("--evidence_teacher_batch_size", type=int, default=8)
    parser.add_argument("--evidence_cache_path", default="evidence_cache/targets.pt")
    parser.add_argument("--evidence_report_dir", default="")
    parser.add_argument("--evidence_prepare_only", action="store_true")


def validate(args):
    for name in ("lambda_evidence", "evidence_min_drop", "evidence_stable_rms"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"--{name} must be finite and nonnegative")
    if not 0 < args.evidence_area < 1:
        raise ValueError("--evidence_area must lie in (0, 1)")
    if not 0 < args.evidence_batch_fraction <= 1:
        raise ValueError("--evidence_batch_fraction must lie in (0, 1]")
    if args.evidence_refs_per_class < 1 or args.evidence_teacher_batch_size < 1:
        raise ValueError("Evidence reference count and microbatch must be positive")
    if args.evidence_photos_per_class < 0:
        raise ValueError("--evidence_photos_per_class must be nonnegative")
    if active(args):
        if args.teacher_pretrain_epochs < 1:
            raise ValueError("Evidence v1 requires a persistent tuned teacher (pretrain_epochs >= 1)")
        if args.n_ctx_visual < 1 or args.max_size != 224:
            raise ValueError("Evidence training requires visual prompts and --max_size 224")
        if not args.evidence_cache_path:
            raise ValueError("Evidence v1 requires --evidence_cache_path")
