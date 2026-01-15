# main.py  (single-fold runner for src/run_5fold.py)
from __future__ import annotations

import os
import copy
import argparse
import math
import numpy as np
import torch
from typing import Dict, Any, Optional

from src.dataset_loader import create_dataloader
from src.utils import set_seed, save_json, ensure_dir
from nets.disbg import DisBGModel
from src.runner import fit_one_fold, train_one_epoch, evaluate

# ----------------------------
# optional W&B (SAFE for sweeps)
# ----------------------------
try:
    import wandb  # type: ignore
except Exception:
    wandb = None


# ----------------------------
# tiny utils
# ----------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise ValueError(f"Invalid boolean value: {v}")


def _safe_float(v) -> Optional[float]:
    """Return finite float if possible, else None (prevents W&B summary/log crashes)."""
    try:
        if v is None:
            return None
        if isinstance(v, (np.number, int, float)):
            x = float(v)
            return x if math.isfinite(x) else None
        if isinstance(v, str):
            x = float(v)
            return x if math.isfinite(x) else None
        return None
    except Exception:
        return None


def _wandb_is_disabled() -> bool:
    v1 = os.environ.get("WANDB_DISABLED", "")
    v2 = os.environ.get("WANDB_MODE", "")
    if str(v1).strip().lower() in ("1", "true", "yes", "y", "on"):
        return True
    if str(v2).strip().lower() == "disabled":
        return True
    return False


def _in_wandb_agent_env() -> bool:
    for k in ("WANDB_SWEEP_ID", "WANDB_RUN_ID", "WANDB_AGENT_ID", "WANDB_LAUNCH_ID"):
        if os.environ.get(k):
            return True
    return False


def _wandb_init_if_needed(args: argparse.Namespace):
    """
    RULES:
    - Only init when --use_wandb is passed.
    - If outer runner disables W&B (WANDB_MODE=disabled / WANDB_DISABLED=true), do nothing.
    - In sweep-agent env, init minimal (agent owns project/entity).
    """
    if wandb is None or _wandb_is_disabled() or (not bool(getattr(args, "use_wandb", False))):
        return None

    # Optional mode
    if getattr(args, "wandb_mode", None):
        os.environ["WANDB_MODE"] = str(args.wandb_mode)

    dataset = getattr(args, "dataset", "NA")
    fold_id = int(getattr(args, "fold_id", 0))
    seed = int(getattr(args, "seed", 0))

    tags = getattr(args, "wandb_tags", "")
    tags = tags.split(",") if tags else None

    in_agent = _in_wandb_agent_env()

    try:
        if in_agent:
            run = wandb.init(tags=tags, notes=getattr(args, "wandb_notes", None))
            # Update config safely (avoid wandb-managed keys)
            safe = vars(args).copy()
            for bad in (
                "wandb_project", "wandb_entity", "wandb_group", "wandb_name",
                "wandb_tags", "wandb_notes", "wandb_mode"
            ):
                safe.pop(bad, None)
            try:
                wandb.config.update(safe, allow_val_change=True)
            except Exception:
                pass
            return run

        project = getattr(args, "wandb_project", None) or os.environ.get("WANDB_PROJECT", "DisBG")
        entity  = getattr(args, "wandb_entity", None) or os.environ.get("WANDB_ENTITY", None)
        group   = getattr(args, "wandb_group", None)  or f"{dataset}_seed{seed}"
        name    = getattr(args, "wandb_name", None)   or f"{dataset}_fold{fold_id}_seed{seed}"

        run = wandb.init(
            project=project,
            entity=entity,
            group=group,
            name=name,
            tags=tags,
            notes=getattr(args, "wandb_notes", None),
            config=vars(args),
        )
        return run

    except Exception as e:
        print(f"[WARN] wandb.init failed: {e}")
        return None


def _wandb_log_dict(run, d: Dict[str, Any], prefix: str, *, step: Optional[int] = None):
    """
    IMPORTANT:
    - Do NOT force step unless you are sure it's monotonically increasing for THIS run.
    - Warmup often restarts at 0 -> causes warnings. So default: step=None.
    """
    if run is None or wandb is None or not isinstance(d, dict):
        return
    if getattr(wandb, "run", None) is None:
        return

    payload = {}
    for k, v in d.items():
        fv = _safe_float(v)
        if fv is not None:
            payload[f"{prefix}/{k}"] = fv

    if not payload:
        return

    try:
        if step is None:
            wandb.log(payload, commit=True)
        else:
            wandb.log(payload, commit=True)
    except Exception:
        pass


# ----------------------------
# optimizer / freeze utils
# ----------------------------
def build_optimizer(
    model: torch.nn.Module,
    base_lr: float,
    weight_decay: float,
    masker_lr_mult: float = 1.0
) -> torch.optim.Optimizer:
    """Adam with param groups: backbone/masker × decay/no_decay."""
    no_decay_keys = ("bias", "norm", "bn", "layernorm")
    groups = {
        "backbone_decay":   {"params": [], "lr": base_lr,                  "weight_decay": weight_decay},
        "backbone_nodecay": {"params": [], "lr": base_lr,                  "weight_decay": 0.0},
        "masker_decay":     {"params": [], "lr": base_lr * masker_lr_mult, "weight_decay": weight_decay},
        "masker_nodecay":   {"params": [], "lr": base_lr * masker_lr_mult, "weight_decay": 0.0},
    }

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        ln = n.lower()
        is_masker = ("mask" in ln) or ("masker" in ln) or ("mask_gen" in ln)
        is_nodecay = any(k in ln for k in no_decay_keys)

        if is_masker and is_nodecay:
            groups["masker_nodecay"]["params"].append(p)
        elif is_masker:
            groups["masker_decay"]["params"].append(p)
        elif is_nodecay:
            groups["backbone_nodecay"]["params"].append(p)
        else:
            groups["backbone_decay"]["params"].append(p)

    param_groups = [g for g in groups.values() if len(g["params"]) > 0]
    return torch.optim.Adam(param_groups)


def freeze_maskers(model: torch.nn.Module, freeze: bool = True) -> None:
    """Freeze/unfreeze all masker-related parameters."""
    for n, p in model.named_parameters():
        ln = n.lower()
        if ("mask" in ln) or ("masker" in ln) or ("mask_gen" in ln):
            p.requires_grad = (not freeze)


# ----------------------------
# parsing
# ----------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    # basics
    p.add_argument("--dataset", type=str, required=True, choices=["ADHD", "ABIDE"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="results")

    # 5fold control
    p.add_argument("--use_5fold", action="store_true")
    p.add_argument("--fold_id", type=int, default=0)

    # reproducibility: split vs train
    p.add_argument("--split_seed", type=int, default=0, help="Seed for DATA SPLIT ONLY (must be fixed across folds)")
    p.add_argument("--dl_num_workers", type=int, default=0, help="DataLoader num_workers (0 for best reproducibility)")

    # optional: deterministic flags (requires your updated set_seed signature)
    p.add_argument("--deterministic", type=str2bool, nargs="?", const=True, default=True)
    p.add_argument("--strict_deterministic", type=str2bool, nargs="?", const=True, default=False)

    # training
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--warmup_epoch", type=int, default=10)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--masker_lr_mult", type=float, default=1.0)
    p.add_argument("--grad_clip", type=float, default=0.0)

    p.add_argument("--stageB_warmup", type=int, default=None)

    # model
    p.add_argument("--gnn_hidden_dim", type=int, default=64)
    p.add_argument("--gnn_out_dim", type=int, default=64)
    p.add_argument("--num_gnn_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--num_feats", type=int, default=116)

    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--num_sex_classes", type=int, default=2)
    p.add_argument("--num_age_classes", type=int, default=4)

    # eval / selection
    p.add_argument("--fixed_thr", type=float, default=0.5)
    p.add_argument("--select_by", type=str, default="val_score",
                   choices=["accuracy", "f1_score", "roc_auc", "val_score", "val_score_bestthr"])
    p.add_argument("--eval_thr_mode", type=str, default="best_acc", choices=["fixed", "best_f1", "best_acc","best_score"])
    p.add_argument("--eval_thr_grid", type=str, default="coarse", choices=["coarse", "fine"])
    p.add_argument("--eval_use_best_thr", action="store_true", default=True)
    p.add_argument("--no_eval_use_best_thr", dest="eval_use_best_thr", action="store_false")
    # score hyperparams
    p.add_argument("--score_f1_beta", type=float, default=0.5)
    p.add_argument("--score_fair_alpha", type=float, default=0.01)
    p.add_argument("--score_fair_w_age", type=float, default=4.0)
    p.add_argument("--score_fair_w_sex", type=float, default=1.0)

    # loss weights
    p.add_argument("--lambda_task", type=float, default=1.0)
    p.add_argument("--lambda_sensitive", type=float, default=0.02)
    p.add_argument("--lambda_pfs", type=float, default=0.3)
    p.add_argument("--lambda_pfs_attr", type=float, default=0.1)
    p.add_argument("--lambda_supcon", type=float, default=0.02)
    p.add_argument("--lambda_ortho", type=float, default=0.05)
    p.add_argument("--supcon_temperature", type=float, default=0.07)
    p.add_argument("--lambda_causal_suf", type=float, default=1.0)
    p.add_argument("--lam_ent_yb", type=float, default=0.1)

    # gradient routing knobs
    p.add_argument("--cf_detach_zc", type=str2bool, nargs="?", const=True, default=False)
    p.add_argument("--no_cf_detach_zc", dest="cf_detach_zc", action="store_false")

    p.add_argument("--cf_detach_zb_for_disease", type=str2bool, nargs="?", const=True, default=True)
    p.add_argument("--no_cf_detach_zb_for_disease", dest="cf_detach_zb_for_disease", action="store_false")
    p.add_argument("--cf_no_detach_zb_for_disease", dest="cf_detach_zb_for_disease", action="store_false")

    p.add_argument("--ent_yb_head_only", type=str2bool, nargs="?", const=True, default=True)
    p.add_argument("--no_ent_yb_head_only", dest="ent_yb_head_only", action="store_false")

    # disease pos weight mode
    p.add_argument("--disease_pos_weight_mode", type=str, default="none", choices=["none", "sqrt", "linear"])
    p.add_argument("--disease_pos_weight", type=float, default=None)

    # freeze
    p.add_argument("--freeze_b_after", type=int, default=25)
    p.add_argument("--freeze_c_after", type=int, default=-1)

    # mask regs
    p.add_argument("--mask_reg_weight", type=float, default=0.6)
    p.add_argument("--mask_temperature", type=float, default=1.0)
    p.add_argument("--mask_hidden_dim", type=int, default=64)
    p.add_argument("--mask_target_mean", type=float, default=0.22)
    p.add_argument("--lambda_maskmean", type=float, default=0.30)
    p.add_argument("--mask_target_std", type=float, default=0.20)
    p.add_argument("--lambda_maskstd", type=float, default=0.00)
    p.add_argument("--lambda_mask_dis", type=float, default=0.35)
    p.add_argument("--mask_dis_mode", type=str, default="norm_overlap", choices=["overlap", "norm_overlap"])
    p.add_argument("--mask_topk_ratio", type=float, default=0.10)
    p.add_argument("--lambda_mask_l1", type=float, default=0.00)

    # two-stage / schedule
    p.add_argument("--two_stage", type=str2bool, nargs="?", const=True, default=True)
    p.add_argument("--no_two_stage", dest="two_stage", action="store_false")
    p.add_argument("--stage1_ratio", type=float, default=0.35)
    p.add_argument("--stage2_lr_mult", type=float, default=0.30)
    p.add_argument("--adaptive_schedule", type=str2bool, nargs="?", const=True, default=True)
    p.add_argument("--no_adaptive_schedule", dest="adaptive_schedule", action="store_false")

    p.add_argument("--w_pfs_init", type=float, default=0.10)
    p.add_argument("--w_sup_init", type=float, default=0.05)
    p.add_argument("--w_csuf_init", type=float, default=0.05)
    p.add_argument("--w_pfs_max", type=float, default=0.25)
    p.add_argument("--w_sup_max", type=float, default=0.25)
    p.add_argument("--w_csuf_max", type=float, default=0.20)
    p.add_argument("--sched_grow", type=float, default=0.04)
    p.add_argument("--sched_decay", type=float, default=0.10)
    p.add_argument("--sched_patience", type=int, default=5)
    p.add_argument("--mask_min_mean", type=float, default=0.08)
    p.add_argument("--mask_max_norm_overlap", type=float, default=1.40)

    # W&B args (per-fold)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_group", type=str, default=None)
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument("--wandb_tags", type=str, default="")
    p.add_argument("--wandb_notes", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default=None)

    return p


def parse_args_known() -> argparse.Namespace:
    p = build_parser()
    args, _ = p.parse_known_args()
    return args


def run_one_fold(args: argparse.Namespace) -> Dict[str, Any]:
    set_seed(int(args.seed), deterministic=True, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_id = int(getattr(args, "fold_id", 0))
    print(f"\n================ Fold {fold_id} (seed={args.seed}) ================\n")
    print(f"Using device: {device}\n")

    run = _wandb_init_if_needed(args)

    try:
        train_loader, val_loader, test_loader = create_dataloader(
            args.dataset,
            batch_size=int(args.batch_size),
            use_5fold=True,
            fold_id=int(fold_id),
            split_seed=int(getattr(args, "split_seed", 0)),
            train_seed=int(args.seed),
            num_workers=int(getattr(args, "dl_num_workers", 0)),
        )


        model = DisBGModel(args).to(device)
        model.args = args  # runner uses this

        if run is not None and wandb is not None:
            try:
                wandb.watch(model, log="gradients", log_freq=50)
            except Exception:
                pass

        num_epochs = int(getattr(args, "num_epochs", 100))
        warmup_epoch = int(getattr(args, "warmup_epoch", 0))
        fixed_thr = float(getattr(args, "fixed_thr", 0.5))

        # -------------------------
        # Stage A: warmup  (NO step passed to wandb.log to avoid monotonic warnings)
        # -------------------------
        warmup_history = {"train": [], "val": []}
        if warmup_epoch > 0:
            print(f"[WARMUP] Freeze maskers for first {warmup_epoch} epochs (true freeze + opt rebuild).")
            freeze_maskers(model, freeze=True)

            opt_warm = build_optimizer(
                model,
                base_lr=float(args.lr),
                weight_decay=float(getattr(args, "weight_decay", 0.0)),
                masker_lr_mult=float(getattr(args, "masker_lr_mult", 1.0)),
            )

            for ep in range(warmup_epoch):
                tr = train_one_epoch(model, train_loader, opt_warm, device, args, epoch=ep)
                va = evaluate(model, val_loader, device, thr=fixed_thr)
                warmup_history["train"].append(tr)
                warmup_history["val"].append(va)

                # DO NOT pass step here
                _wandb_log_dict(run, tr, "warmup/train", step=None)
                _wandb_log_dict(run, va, "warmup/val", step=None)

                print(
                    f"[WARMUP][ep={ep:03d}] train_acc={tr.get('acc', float('nan')):.2f}% | "
                    f"val_acc={va.get('accuracy', float('nan')):.2f}% val_f1={va.get('f1_score', float('nan')):.2f}%"
                )

        # -------------------------
        # Stage B
        # -------------------------
        freeze_maskers(model, freeze=False)
        optimizer = build_optimizer(
            model,
            base_lr=float(args.lr),
            weight_decay=float(getattr(args, "weight_decay", 0.0)),
            masker_lr_mult=float(getattr(args, "masker_lr_mult", 1.0)),
        )

        args_stage = copy.deepcopy(args)

        if getattr(args_stage, "stageB_warmup", None) is None:
            stageb_warm = max(1, min(5, int(warmup_epoch))) if warmup_epoch > 0 else 1
        else:
            stageb_warm = max(0, int(getattr(args_stage, "stageB_warmup")))
        args_stage.warmup_epoch = stageb_warm
        args_stage.num_epochs = max(1, int(num_epochs - warmup_epoch))

        fb = int(getattr(args_stage, "freeze_b_after", -1))
        args_stage.freeze_b_after = (max(0, fb - warmup_epoch) if fb >= 0 else -1)

        fc = int(getattr(args_stage, "freeze_c_after", -1))
        args_stage.freeze_c_after = (max(0, fc - warmup_epoch) if fc >= 0 else -1)

        result = fit_one_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            device=device,
            args=args_stage,
            scheduler=None,
        )

        best_epoch_global = int(result.best_epoch) + int(warmup_epoch)

        fold_payload = {
            "fold_id": int(fold_id),
            "seed": int(args.seed),
            "best_epoch": int(best_epoch_global),
            "fixed_thr": float(fixed_thr),
            "best_val": result.best_val,
            "test_at_best": result.test_at_best,
            "history": {"warmup": warmup_history, "stageB": result.history},
            "stageB_warmup_used": int(stageb_warm),
        }

        out_dir = str(getattr(args, "out_dir", "results"))
        ensure_dir(out_dir)
        out_path = os.path.join(out_dir, f"summary_fold_{fold_id}_main.json")
        save_json(fold_payload, out_path)
        print(f"\n[OK] Saved: {out_path}\n")

        # -------------------------
        # Write final metrics to W&B summary (SAFE)
        # -------------------------
        if run is not None and wandb is not None:
            try:
                # log once (no step)
                final_log = {}

                if isinstance(fold_payload.get("best_val"), dict):
                    for k, v in fold_payload["best_val"].items():
                        fv = _safe_float(v)
                        if fv is not None:
                            final_log[f"best_val/{k}"] = fv
                            run.summary[f"best_val/{k}"] = fv

                if isinstance(fold_payload.get("test_at_best"), dict):
                    for k, v in fold_payload["test_at_best"].items():
                        fv = _safe_float(v)
                        if fv is not None:
                            final_log[f"test_at_best/{k}"] = fv
                            run.summary[f"test_at_best/{k}"] = fv

                run.summary["best_epoch_global"] = int(best_epoch_global)
                run.summary["stageB_warmup_used"] = int(stageb_warm)
                run.summary["fold_id"] = int(fold_id)
                run.summary["seed"] = int(args.seed)

                if final_log:
                    wandb.log(final_log, commit=True)

                try:
                    wandb.save(out_path)
                except Exception:
                    pass

            except Exception as e:
                print(f"[WARN] W&B final summary/log failed: {e}")

        return fold_payload

    finally:
        if run is not None and wandb is not None:
            try:
                run.finish()
            except Exception:
                pass


def main():
    args = parse_args_known()

    if not getattr(args, "use_5fold", False):
        raise RuntimeError("This main.py is designed to be called with --use_5fold --fold_id by src/run_5fold.py")

    print("\n--- Training Configuration (single fold entry) ---")
    print(f"dataset={args.dataset} fold_id={getattr(args,'fold_id',None)} seed={args.seed}")
    print(f"num_epochs={getattr(args,'num_epochs',None)} warmup_epoch={getattr(args,'warmup_epoch',None)} stageB_warmup={getattr(args,'stageB_warmup',None)}")
    print(f"batch_size={args.batch_size} lr={args.lr} weight_decay={getattr(args,'weight_decay',None)}")
    print(f"eval_thr_mode={getattr(args,'eval_thr_mode',None)} eval_thr_grid={getattr(args,'eval_thr_grid',None)} eval_use_best_thr={getattr(args,'eval_use_best_thr',None)}")
    print(f"out_dir={getattr(args,'out_dir',None)}")
    print("-----------------------------------------------\n")

    run_one_fold(args)


if __name__ == "__main__":
    main()
