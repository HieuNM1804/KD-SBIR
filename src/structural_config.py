"""CLI configuration; all four new loss weights default to zero."""
import math


def teacher_needed(args):
    return any(getattr(args, key, 0) > 0 for key in
               ("lambda_semantic", "lambda_sfgw", "lambda_contract"))


def anchors_needed(args):
    return (getattr(args, "lambda_semantic", 0) > 0
            or getattr(args, "lambda_contract", 0) > 0
            or (getattr(args, "lambda_sfgw", 0) > 0
                and args.local_matching == "fgw" and args.fgw_alpha < 1))


def add_arguments(parser):
    for name in ("retrieval", "semantic", "sfgw", "contract"):
        parser.add_argument("--lambda_"+name, type=float, default=0.0)
    parser.add_argument("--retrieval_temperature", type=float, default=0.07)
    parser.add_argument("--semantic_anchors", choices=("class", "all"), default="all",
                        help="all: class, photo-of-class and sketch-of-class; train classes only.")
    parser.add_argument("--contract_rho", type=float, default=0.5)
    parser.add_argument("--contract_min_count", type=int, default=2)
    parser.add_argument("--student_sampler", choices=("main", "class"), default="main")
    parser.add_argument("--samples_per_class", type=int, default=4)
    parser.add_argument("--local_matching", choices=("fgw", "spatial"), default="fgw")
    parser.add_argument("--teacher_patch_grid", type=int, default=8)
    parser.add_argument("--spatial_grid", type=int, default=7,
                        help="Common grid for spatial baseline only; no upsampling.")
    parser.add_argument("--fgw_alpha", type=float, default=0.5)
    parser.add_argument("--transport_epsilon", type=float, default=0.05)
    parser.add_argument("--gw_iterations", type=int, default=10)
    parser.add_argument("--sinkhorn_iterations", type=int, default=300)
    parser.add_argument("--transport_tolerance", type=float, default=1e-4)
    parser.add_argument("--local_target_mode", choices=("online", "cache"), default="online")
    parser.add_argument("--local_cache_dir", default="local_teacher_cache")
    parser.add_argument("--local_teacher_batch_size", type=int, default=8)
    parser.add_argument("--local_report_dir", default="",
                        help="Optional first-batch transport HTML/JSON diagnostics (FGW only).")


def validate(args, parser):
    for name in ("lambda_retrieval", "lambda_semantic", "lambda_sfgw", "lambda_contract"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(name+" must be finite and nonnegative.")
    for name in ("retrieval_temperature", "transport_epsilon", "transport_tolerance"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(name+" must be finite and positive.")
    for name in ("samples_per_class", "contract_min_count", "teacher_patch_grid",
                 "spatial_grid", "gw_iterations", "sinkhorn_iterations", "local_teacher_batch_size"):
        if getattr(args, name) < 1:
            parser.error(name+" must be positive.")
    if not 0 <= args.fgw_alpha <= 1 or not 0 <= args.contract_rho < 1:
        parser.error("fgw_alpha must be in [0,1]; contract_rho in [0,1).")
    if args.lambda_sfgw > 0 and args.max_size != 224:
        parser.error("Local DFN targets require main's 224x224 input.")
    if args.student_sampler == "class":
        if args.batch_size % args.samples_per_class or args.batch_size // args.samples_per_class < 2:
            parser.error("Class sampler needs >=2 classes and batch_size divisible by samples_per_class.")
    if args.lambda_contract > 0 and args.student_sampler == "class":
        if args.samples_per_class < args.contract_min_count:
            parser.error("samples_per_class is below contract_min_count.")
    if sum(getattr(args, "lambda_"+n) for n in
           ("domain", "modality", "retrieval", "semantic", "sfgw", "contract")) <= 0:
        parser.error("At least one training loss must be enabled.")
    if args.lambda_sfgw > 0 and args.n_ctx_visual == 0:
        parser.error("Local distillation requires trainable visual prompts.")
