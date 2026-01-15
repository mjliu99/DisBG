# src/utils.py
import os
import json
import random
import argparse
import numpy as np
import torch



def set_seed(seed: int, deterministic: bool = True, strict: bool = False):
    """
    seed: random seed
    deterministic:
        - True: best-effort deterministic on CUDA (cudnn deterministic + disable benchmark)
        - False: only set RNG seeds (faster, but more variance)
    strict:
        - True: torch.use_deterministic_algorithms(True) (may throw if non-deterministic ops are used)
        - False: do not enforce at op-level (recommended for PyG/GNN unless you know it works)
    """
    seed = int(seed)

    # ---- python / numpy ----
    random.seed(seed)
    np.random.seed(seed)

    # ---- torch ----
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ---- CUDA deterministic behavior ----
    if deterministic:
        # cuDNN: deterministic convolution kernels (may reduce speed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # cuBLAS: make some GEMM ops deterministic (needs env var)
        # Must be set BEFORE CUDA context is initialized ideally; still usually fine if early.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

        # Optional: disable TF32 to reduce numeric drift (Ampere+)
        # This can improve repeatability at small cost.
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass

    # ---- strict deterministic algorithms (may error with PyG scatter ops) ----
    if strict:
        try:
            torch.use_deterministic_algorithms(True)
            # For older PyTorch:
            os.environ["TORCH_DETERMINISTIC"] = "1"
        except Exception:
            pass


def ensure_dir(path: str):
    if path is None or path == "":
        return
    os.makedirs(path, exist_ok=True)


def save_json(obj, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def fmt_pct(x):
    if x != x:  # nan
        return "nan"
    return f"{x:.2f}%"



def ensure_dir(path: str):
    if path is None or path == "":
        return
    os.makedirs(path, exist_ok=True)


def save_json(obj, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def fmt_pct(x):
    if x != x:  # nan
        return "nan"
    return f"{x:.2f}%"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="DisBG: Disentangled Brain Graph Learning with Counterfactual & Contrastive Training"
    )

    # ======================
    # Basic / Reproducibility
    # ======================
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--dataset", type=str, default="ADHD", choices=["ADHD", "ABIDE"], help="Dataset name")
    parser.add_argument("--batch_size", type=int, default=8, help="Mini-batch size")
    parser.add_argument("--out_dir", type=str, default="results", help="Output directory for logs/ckpt/json")

    # ======================
    # Graph / Data settings
    # ======================
    parser.add_argument("--num_nodes", type=int, default=116, help="Number of nodes per graph")
    # IMPORTANT: your dataset_loader uses corr as node features: x is (N,N) => num_feats should be 116
    parser.add_argument("--num_feats", type=int, default=116, help="Node feature dimension (F)")
    parser.add_argument("--num_classes", type=int, default=2, help="Disease classes")
    parser.add_argument("--num_sex_classes", type=int, default=2, help="Sex classes")
    parser.add_argument("--num_age_classes", type=int, default=4, help="Age classes")

    # ======================
    # Model architecture
    # ======================
    parser.add_argument("--gnn_hidden_dim", type=int, default=64, help="Hidden dimension of GNN encoder")
    parser.add_argument("--gnn_out_dim", type=int, default=64, help="Output dimension of GNN encoder")
    parser.add_argument("--num_gnn_layers", type=int, default=2, help="Number of GNN layers")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")

    # mask generator
    parser.add_argument("--mask_temperature", type=float, default=1.0, help="Mask temperature (passed to EdgeMaskGenerator)")
    parser.add_argument("--mask_hidden_dim", type=int, default=128, help="Hidden dimension for edge mask generator")

    # optional node-proj before maskers (helps when x is low-dim; safe even for 116-d)
    parser.add_argument("--use_mask_node_proj", action="store_true", help="Use node projection before mask generators")
    parser.add_argument("--mask_node_proj_dim", type=int, default=64, help="Node projection dim for mask generators")

    # disentanglement
    parser.add_argument("--bias_scale", type=float, default=0.5, help="Scale bias rep when feeding disease head")

    # ======================
    # Optimization
    # ======================
    parser.add_argument("--num_epochs", type=int, default=100, help="Maximum number of training epochs")
    parser.add_argument("--warmup_epoch", type=int, default=10, help="Warmup epochs (freeze maskers + only disease loss)")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience (fit_one_fold)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--masker_lr_mult", type=float, default=1.0, help="LR multiplier for masker parameters")
    parser.add_argument("--grad_clip", type=float, default=0.0, help="Global grad clip (0=disable)")

    # ======================
    # Loss weights
    # ======================
    parser.add_argument("--lambda_task", type=float, default=1.0, help="Weight for supervised task loss")
    # IMPORTANT: default OFF to avoid hurting disease
    parser.add_argument("--lambda_sensitive", type=float, default=0.02, help="Weight for sensitive CE (sex/age), default OFF")
    parser.add_argument("--lambda_pfs", type=float, default=0.3, help="Weight for counterfactual (PFS) loss")
    parser.add_argument("--lambda_pfs_attr", type=float, default=0.1, help="Downweight demographic parts in PFS loss")
    parser.add_argument("--lambda_supcon", type=float, default=0.02, help="Weight for supervised contrastive loss")
    parser.add_argument("--supcon_temperature", type=float, default=0.07, help="Temperature for supervised contrastive loss")
    parser.add_argument("--lambda_ortho", type=float, default=0.05, help="Weight for orthogonality between u_c and u_b")

    # legacy / kept for compatibility
    parser.add_argument("--lambda_hub", type=float, default=0.0, help="[legacy] hub regularization (unused in new runner)")

    # ✅ NEW: causal sufficiency + true decomposition entropy
    parser.add_argument("--lambda_causal_suf", type=float, default=1.0, help="Weight for causal sufficiency (u_c -> y) term")
    parser.add_argument("--lam_ent_yb", type=float, default=0.2, help="Weight for bias-branch disease entropy (maximize entropy)")

    # ======================
    # Ramps (when to activate PFS/SupCon)
    # ======================
    parser.add_argument("--pfs_start", type=int, default=None, help="Epoch to start PFS ramp (default warmup+10)")
    parser.add_argument("--pfs_end", type=int, default=None, help="Epoch to finish PFS ramp (default 0.6*num_epochs)")
    parser.add_argument("--sup_start", type=int, default=None, help="Epoch to start SupCon ramp (default warmup+10)")
    parser.add_argument("--sup_end", type=int, default=None, help="Epoch to finish SupCon ramp (default 0.7*num_epochs)")

    # ======================
    # Mask regularization
    # ======================
    parser.add_argument("--mask_target_mean", type=float, default=0.2, help="Target mean sparsity for edge masks")
    parser.add_argument("--lambda_maskmean", type=float, default=0.1, help="Weight for mask mean constraint")

    parser.add_argument("--mask_target_std", type=float, default=0.20, help="Target std for edge masks")
    parser.add_argument("--lambda_maskstd", type=float, default=0.05, help="Weight for mask std constraint")
    parser.add_argument("--lambda_maskspread", type=float, default=0.05, help="[legacy] alias of lambda_maskstd")

    # separation penalty (overlap / norm_overlap)
    parser.add_argument("--lambda_mask_dis", type=float, default=0.05, help="Mask separation weight (0=off)")
    parser.add_argument("--mask_dis_mode", type=str, default="norm_overlap", choices=["overlap", "norm_overlap"],
                        help="Mask separation mode")
    parser.add_argument("--mask_topk_ratio", type=float, default=0.1, help="Top-k ratio for Jaccard diagnostic")

    # ✅ NEW: l1 term used in runner v3
    parser.add_argument("--lambda_mask_l1", type=float, default=0.1, help="Extra L1 on mask means (mc_mean + mb_mean)")

    # ✅ NEW: mask guardrails used by your schedule idea
    parser.add_argument("--mask_min_mean", type=float, default=0.08, help="Guard: minimum allowed mean for each mask")
    parser.add_argument("--mask_max_norm_overlap", type=float, default=1.4, help="Guard: maximum allowed norm_overlap")

    # ======================
    # Eval / selection
    # ======================
    parser.add_argument("--fixed_thr", type=float, default=0.5, help="Fixed threshold used for VAL/TEST metrics")
    parser.add_argument("--select_by", type=str, default="val_score",
                        help="Model selection key: val_score/accuracy/f1_score/roc_auc/...")

    # score = acc + beta*f1 - alpha*fairness_sum
    parser.add_argument("--score_f1_beta", type=float, default=0.5, help="beta for F1 in val_score")
    parser.add_argument("--score_fair_alpha", type=float, default=0.01, help="alpha for fairness penalty in val_score")

    # verbosity
    parser.add_argument("--eval_verbose", action="store_true", help="Print verbose eval diagnostics per epoch")
    parser.add_argument("--print_steps", action="store_true", help="Print step-level training logs")
    parser.add_argument("--log_interval", type=int, default=50, help="Step log interval when --print_steps")

    # ======================
    # Counterfactual safety switches
    # ======================
    parser.add_argument("--cf_detach_zc", action="store_true",
                        help="Detach z_c in counterfactual branch (runner default True if absent)")
    parser.add_argument("--cf_detach_zb_for_disease", action="store_true",
                        help="Detach permuted z_b when feeding disease in counterfactual (runner default False)")

    # ✅ NEW: run_5fold currently passes this flag
    parser.add_argument("--cf_no_detach_zb_for_disease", action="store_true",
                        help="Alias: explicitly DO NOT detach z_b perm in disease counterfactual branch")

    # ======================
    # Late-freeze maskers (optional)
    # ======================
    parser.add_argument("--freeze_maskers", action="store_true", help="Enable late freeze for maskers")
    parser.add_argument("--freeze_maskers_at", type=float, default=0.6, help="Freeze maskers after this fraction of epochs")

    # ======================
    # CV settings
    # ======================
    parser.add_argument("--use_5fold", action="store_true", help="Enable 5-fold CV mode")
    parser.add_argument("--fold_id", type=int, default=0, help="Fold index for CV")

    # ======================
    # Two-stage training (new)
    # ======================
    parser.add_argument("--two_stage", action="store_true", help="Enable two-stage training in main/runner")
    parser.add_argument("--stage1_ratio", type=float, default=0.35, help="Stage-1 ratio of total epochs")
    parser.add_argument("--stage2_lr_mult", type=float, default=0.3, help="LR multiplier in stage-2 finetune")

    # ======================
    # Adaptive schedule (new)
    # ======================
    parser.add_argument("--adaptive_schedule", action="store_true", help="Enable adaptive schedule for loss weights")
    parser.add_argument("--w_pfs_init", type=float, default=0.10)
    parser.add_argument("--w_sup_init", type=float, default=0.05)
    parser.add_argument("--w_csuf_init", type=float, default=0.05)
    parser.add_argument("--w_pfs_max", type=float, default=0.35)
    parser.add_argument("--w_sup_max", type=float, default=0.25)
    parser.add_argument("--w_csuf_max", type=float, default=0.20)
    parser.add_argument("--sched_grow", type=float, default=0.04, help="Grow rate for adaptive weights")
    parser.add_argument("--sched_decay", type=float, default=0.10, help="Decay rate when plateau")
    parser.add_argument("--sched_patience", type=int, default=5, help="Plateau patience for adaptive schedule")

    # ======================
    # Diagnostics (legacy; kept to avoid breaking scripts)
    # ======================
    parser.add_argument("--diag_every", type=int, default=10, help="[legacy] Print diagnostics every N epochs")
    parser.add_argument("--diag_probe_batches", type=int, default=5, help="[legacy] Batches used for leakage probe")
    parser.add_argument("--mask_overlap_ratio", type=float, default=0.2, help="[legacy] Top-k ratio for overlap IoU/corr")

    args = parser.parse_args()

    # ---- fill derived defaults for ramps if not explicitly provided ----
    if args.pfs_start is None:
        args.pfs_start = int(args.warmup_epoch) + 10
    if args.sup_start is None:
        args.sup_start = int(args.warmup_epoch) + 10
    if args.pfs_end is None:
        args.pfs_end = int(0.6 * int(args.num_epochs))
    if args.sup_end is None:
        args.sup_end = int(0.7 * int(args.num_epochs))

    # ---- legacy alias sync ----
    # keep lambda_maskstd as the effective one in runner
    if hasattr(args, "lambda_maskspread") and hasattr(args, "lambda_maskstd"):
        if args.lambda_maskstd is None:
            args.lambda_maskstd = float(args.lambda_maskspread)

    # ---- counterfactual flag normalization ----
    # if user explicitly says "no_detach", override detach flag
    if getattr(args, "cf_no_detach_zb_for_disease", False):
        args.cf_detach_zb_for_disease = False

    return args
