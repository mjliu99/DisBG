# # src/run_5fold.py
# # -------------------------------------------------------------
# # DisBG 5-fold launcher (W&B sweep-safe, aggregation-to-summary)
# #
# # Guarantees:
# # 1) In sweep/agent env: allow wandb.init() ONLY if wandb.run is None
# #    (agent usually sets env vars, but the program still needs init).
# # 2) If wandb_per_fold is OFF (default in sweep): HARD disable W&B
# #    inside main.py subprocess AND strip sweep env vars to avoid nested init.
# # 3) Always write aggregated metrics to wandb.run.summary
# #    so they appear on W&B page and CSV export.
# # 4) Only call wandb.finish() when *we* created the run (non-sweep).
# # -------------------------------------------------------------

# import os
# import json
# import argparse
# import subprocess
# import numpy as np


# # -------------------------
# # utils
# # -------------------------
# def str2bool(v):
#     if isinstance(v, bool):
#         return v
#     s = str(v).strip().lower()
#     if s in ("1", "true", "t", "yes", "y", "on"):
#         return True
#     if s in ("0", "false", "f", "no", "n", "off"):
#         return False
#     raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


# def _in_wandb_agent_env() -> bool:
#     for k in ("WANDB_SWEEP_ID", "WANDB_RUN_ID", "WANDB_AGENT_ID", "WANDB_LAUNCH_ID"):
#         if os.environ.get(k):
#             return True
#     return False


# def _get_nested(d, key: str):
#     cur = d
#     for k in key.split("."):
#         if not isinstance(cur, dict):
#             return None
#         cur = cur.get(k, None)
#     return cur


# def _collect_numeric(results, key: str):
#     vals = []
#     for r in results:
#         v = _get_nested(r, key)
#         if isinstance(v, (int, float, np.number)) and np.isfinite(float(v)):
#             vals.append(float(v))
#     return vals


# def _format_mean_std(m: float, s: float):
#     if np.isnan(m) or np.isnan(s):
#         return "nan ± nan"
#     return f"{m:.2f} ± {s:.2f}"


# def _load_fold_result(results_dir: str, fold: int):
#     main_path = os.path.join(results_dir, f"summary_fold_{fold}_main.json")
#     summary_path = os.path.join(results_dir, f"summary_fold_{fold}.json")
#     legacy_path = os.path.join(results_dir, f"fold_{fold}.json")

#     for p in (main_path, summary_path, legacy_path):
#         if os.path.exists(p):
#             with open(p, "r", encoding="utf-8") as f:
#                 return json.load(f), p

#     raise FileNotFoundError(
#         f"Missing fold result json. Tried: {main_path} , {summary_path} , {legacy_path}"
#     )


# def _normalize_fold_payload(payload: dict, fold: int, seed: int):
#     out = {"fold_id": int(fold), "seed": int(seed)}

#     # new-style payload from main.py
#     if isinstance(payload, dict) and ("best_val" in payload or "test_at_best" in payload):
#         out["best_epoch"] = payload.get("best_epoch", -1)
#         if isinstance(payload.get("best_val", None), dict):
#             out["val"] = payload["best_val"]
#         if isinstance(payload.get("test_at_best", None), dict):
#             out["test"] = payload["test_at_best"]
#         out["_raw"] = payload
#         return out

#     # legacy fallback
#     out["_raw"] = payload
#     if isinstance(payload, dict):
#         if "best_epoch" in payload:
#             out["best_epoch"] = payload["best_epoch"]
#         if "val" in payload and isinstance(payload["val"], dict):
#             out["val"] = payload["val"]
#         if "test" in payload and isinstance(payload["test"], dict):
#             out["test"] = payload["test"]
#     return out


# # -------------------------
# # args
# # -------------------------
# def build_parser():
#     p = argparse.ArgumentParser()

#     # dataset
#     p.add_argument("--dataset", type=str, choices=["ADHD", "ABIDE"], default="ADHD")

#     # folds
#     p.add_argument("--folds", type=int, default=5)
#     p.add_argument("--base_seed", type=int, default=0)

#     # training
#     p.add_argument("--num_epochs", type=int, default=100)
#     p.add_argument("--warmup_epoch", type=int, default=10)
#     p.add_argument("--patience", type=int, default=30)
#     p.add_argument("--batch_size", type=int, default=8)
#     p.add_argument("--lr", type=float, default=1e-4)
#     p.add_argument("--weight_decay", type=float, default=5e-4)
#     p.add_argument("--masker_lr_mult", type=float, default=1.0)
#     p.add_argument("--grad_clip", type=float, default=0.0)

#     # model
#     p.add_argument("--gnn_hidden_dim", type=int, default=64)
#     p.add_argument("--gnn_out_dim", type=int, default=64)
#     p.add_argument("--num_gnn_layers", type=int, default=2)
#     p.add_argument("--dropout", type=float, default=0.0)
#     p.add_argument("--num_feats", type=int, default=116)

#     p.add_argument("--num_classes", type=int, default=2)
#     p.add_argument("--num_sex_classes", type=int, default=2)
#     p.add_argument("--num_age_classes", type=int, default=4)

#     # eval / selection
#     p.add_argument("--fixed_thr", type=float, default=0.5)
#     p.add_argument(
#         "--select_by",
#         type=str,
#         default="val_score_bestthr",
#         choices=["accuracy", "f1_score", "roc_auc", "val_score", "val_score_bestthr"],
#     )
#     p.add_argument("--eval_thr_mode", type=str, default="best_acc", choices=["fixed", "best_f1", "best_acc","best_score"])
#     p.add_argument("--eval_thr_grid", type=str, default="coarse", choices=["coarse", "fine"])

#     p.add_argument("--eval_use_best_thr", type=str2bool, nargs="?", const=True, default=True)
#     p.add_argument("--no_eval_use_best_thr", dest="eval_use_best_thr", action="store_false")


#     # score hyperparams
#     p.add_argument("--score_f1_beta", type=float, default=0.5)
#     p.add_argument("--score_fair_alpha", type=float, default=0.01)
#     p.add_argument("--score_fair_w_age", type=float, default=4.0)
#     p.add_argument("--score_fair_w_sex", type=float, default=1.0)

#     # loss weights
#     p.add_argument("--lambda_task", type=float, default=1.0)
#     p.add_argument("--lambda_sensitive", type=float, default=0.02)

#     p.add_argument("--lambda_pfs", type=float, default=0.3)
#     p.add_argument("--lambda_pfs_attr", type=float, default=0.1)

#     p.add_argument("--lambda_supcon", type=float, default=0.02)
#     p.add_argument("--lambda_ortho", type=float, default=0.05)
#     p.add_argument("--supcon_temperature", type=float, default=0.07)

#     p.add_argument("--lambda_causal_suf", type=float, default=1.0)
#     p.add_argument("--lam_ent_yb", type=float, default=0.1)

#     # gradient routing knobs
#     p.add_argument("--cf_detach_zc", type=str2bool, nargs="?", const=True, default=False)
#     p.add_argument("--no_cf_detach_zc", dest="cf_detach_zc", action="store_false")

#     p.add_argument("--cf_detach_zb_for_disease", type=str2bool, nargs="?", const=True, default=True)
#     p.add_argument("--no_cf_detach_zb_for_disease", dest="cf_detach_zb_for_disease", action="store_false")

#     p.add_argument("--ent_yb_head_only", type=str2bool, nargs="?", const=True, default=True)
#     p.add_argument("--no_ent_yb_head_only", dest="ent_yb_head_only", action="store_false")

#     # disease pos weight
#     p.add_argument("--disease_pos_weight_mode", type=str, default="none", choices=["none", "sqrt", "linear"])
#     p.add_argument("--disease_pos_weight", type=float, default=None)

#     # early freeze maskers
#     p.add_argument("--freeze_b_after", type=int, default=25)
#     p.add_argument("--freeze_c_after", type=int, default=-1)

#     # mask regs
#     p.add_argument("--mask_reg_weight", type=float, default=0.6)
#     p.add_argument("--mask_temperature", type=float, default=1.0)
#     p.add_argument("--mask_hidden_dim", type=int, default=64)

#     p.add_argument("--mask_target_mean", type=float, default=0.22)
#     p.add_argument("--lambda_maskmean", type=float, default=0.30)

#     p.add_argument("--mask_target_std", type=float, default=0.20)
#     p.add_argument("--lambda_maskstd", type=float, default=0.00)

#     p.add_argument("--lambda_mask_dis", type=float, default=0.35)
#     p.add_argument("--mask_dis_mode", type=str, default="norm_overlap", choices=["overlap", "norm_overlap"])
#     p.add_argument("--mask_topk_ratio", type=float, default=0.1)
#     p.add_argument("--lambda_mask_l1", type=float, default=0.00)

#     # two-stage
#     p.add_argument("--two_stage", type=str2bool, nargs="?", const=True, default=True)
#     p.add_argument("--no_two_stage", dest="two_stage", action="store_false")
#     p.add_argument("--stage1_ratio", type=float, default=0.35)
#     p.add_argument("--stage2_lr_mult", type=float, default=0.30)

#     # adaptive schedule
#     p.add_argument("--adaptive_schedule", type=str2bool, nargs="?", const=True, default=True)
#     p.add_argument("--no_adaptive_schedule", dest="adaptive_schedule", action="store_false")

#     p.add_argument("--w_pfs_init", type=float, default=0.10)
#     p.add_argument("--w_sup_init", type=float, default=0.05)
#     p.add_argument("--w_csuf_init", type=float, default=0.05)

#     p.add_argument("--w_pfs_max", type=float, default=0.35)
#     p.add_argument("--w_sup_max", type=float, default=0.25)
#     p.add_argument("--w_csuf_max", type=float, default=0.20)

#     p.add_argument("--sched_grow", type=float, default=0.04)
#     p.add_argument("--sched_decay", type=float, default=0.10)
#     p.add_argument("--sched_patience", type=int, default=5)

#     p.add_argument("--mask_min_mean", type=float, default=0.08)
#     p.add_argument("--mask_max_norm_overlap", type=float, default=1.40)

#     # io
#     p.add_argument("--results_dir", type=str, default="results")

#     # execution
#     p.add_argument("--python", type=str, default="python", help="python executable")
#     p.add_argument("--main", type=str, default="main.py", help="entry script (main.py)")
#     p.add_argument("--cwd", type=str, default=None, help="working directory for subprocess")

#     # W&B (outer agg)
#     p.add_argument("--use_wandb", action="store_true")
#     p.add_argument("--wandb_project", type=str, default="DisBG-B")
#     p.add_argument("--wandb_entity", type=str, default=None)
#     p.add_argument("--wandb_group", type=str, default=None)
#     p.add_argument("--wandb_tags", type=str, default="")
#     p.add_argument("--wandb_notes", type=str, default=None)
#     p.add_argument("--wandb_mode", type=str, default=None)

#     # per-fold wandb (debug only; should be OFF in sweeps)
#     p.add_argument("--wandb_per_fold", type=str2bool, nargs="?", const=True, default=True)
#     p.add_argument("--no_wandb_per_fold", dest="wandb_per_fold", action="store_false")

#     # log aggregation to W&B
#     p.add_argument("--wandb_log_agg", type=str2bool, nargs="?", const=True, default=True)
#     p.add_argument("--no_wandb_log_agg", dest="wandb_log_agg", action="store_false")

#     p.add_argument("--wandb_name_prefix", type=str, default=None)

#     return p


# # -------------------------
# # W&B helpers
# # -------------------------
# def _make_cfg_summary(args) -> dict:
#     return {
#         "cfg/dataset": str(args.dataset),
#         "cfg/folds": int(args.folds),
#         "cfg/base_seed": int(args.base_seed),
#         "cfg/batch_size": int(args.batch_size),
#         "cfg/lr": float(args.lr),
#         "cfg/weight_decay": float(args.weight_decay),
#         "cfg/gnn_hidden_dim": int(args.gnn_hidden_dim),
#         "cfg/num_gnn_layers": int(args.num_gnn_layers),
#         "cfg/dropout": float(args.dropout),
#         "cfg/lambda_sensitive": float(args.lambda_sensitive),
#         "cfg/lambda_causal_suf": float(args.lambda_causal_suf),
#         "cfg/lam_ent_yb": float(args.lam_ent_yb),
#         "cfg/mask_target_mean": float(args.mask_target_mean),
#         "cfg/mask_target_std": float(args.mask_target_std),
#         "cfg/mask_temperature": float(args.mask_temperature),
#         "cfg/mask_topk_ratio": float(args.mask_topk_ratio),
#         "cfg/two_stage": bool(args.two_stage),
#         "cfg/adaptive_schedule": bool(args.adaptive_schedule),
#         "cfg/select_by": str(args.select_by),
#         "cfg/eval_thr_mode": str(args.eval_thr_mode),
#         "cfg/eval_thr_grid": str(args.eval_thr_grid),
#         "cfg/eval_use_best_thr": bool(args.eval_use_best_thr),
#         "cfg/score_f1_beta": float(args.score_f1_beta),
#         "cfg/score_fair_alpha": float(args.score_fair_alpha),
#     }


# def _wandb_get_run_or_init(args):
#     try:
#         import wandb  # type: ignore
#     except Exception:
#         return None, False

#     if (not getattr(args, "use_wandb", False)) or (not getattr(args, "wandb_log_agg", True)):
#         return None, False

#     in_agent = _in_wandb_agent_env()

#     # sweep/agent: ensure a run exists (agent usually still expects program to init)
#     if in_agent:
#         if getattr(wandb, "run", None) is None:
#             try:
#                 run = wandb.init(
#                     tags=(args.wandb_tags.split(",") if args.wandb_tags else None),
#                     notes=args.wandb_notes,
#                     config=vars(args),
#                 )
#                 return run, False
#             except Exception as e:
#                 print(f"[WARN] wandb.init (agent) failed: {e}")
#                 return None, False

#         # reuse existing run
#         try:
#             cfg = vars(args).copy()
#             cfg.pop("wandb_project", None)
#             cfg.pop("wandb_entity", None)
#             wandb.config.update(cfg, allow_val_change=True)
#         except Exception:
#             pass
#         return wandb.run, False

#     # non-sweep: we create the run
#     if getattr(args, "wandb_mode", None):
#         os.environ["WANDB_MODE"] = str(args.wandb_mode)

#     group = getattr(args, "wandb_group", None) or f"{args.dataset}_seed{getattr(args, 'base_seed', 0)}"
#     prefix = args.wandb_name_prefix or args.dataset
#     name = f"{prefix}_5fold_seed{getattr(args, 'base_seed', 0)}"

#     run = wandb.init(
#         project=getattr(args, "wandb_project", None) or os.environ.get("WANDB_PROJECT"),
#         entity=getattr(args, "wandb_entity", None) or os.environ.get("WANDB_ENTITY"),
#         group=getattr(args, "wandb_group", None) or os.environ.get("WANDB_RUN_GROUP"),
#         name=(f"{(args.wandb_name_prefix or args.dataset)}_5fold_seed{args.base_seed}"),
#         tags=(args.wandb_tags.split(",") if args.wandb_tags else None),
#         notes=args.wandb_notes,
#         config=vars(args),
#     )
#     return run, True


# def _wandb_log(d, step=None):
#     try:
#         import wandb  # type: ignore
#     except Exception:
#         return
#     if getattr(wandb, "run", None) is None:
#         return
#     try:
#         wandb.log(d, step=step, commit=True)
#     except Exception:
#         pass


# def _wandb_summary_update(d):
#     try:
#         import wandb  # type: ignore
#     except Exception:
#         return
#     if getattr(wandb, "run", None) is None:
#         return
#     for k, v in d.items():
#         try:
#             wandb.run.summary[k] = v
#         except Exception:
#             pass
#     try:
#         wandb.run.summary.update()
#     except Exception:
#         pass


# def _wandb_save(path: str):
#     try:
#         import wandb  # type: ignore
#     except Exception:
#         return
#     if getattr(wandb, "run", None) is None:
#         return
#     try:
#         wandb.save(path)
#     except Exception:
#         pass


# def _wandb_finish_if_created(created_by_us: bool):
#     if not created_by_us:
#         return
#     try:
#         import wandb  # type: ignore
#     except Exception:
#         return
#     if getattr(wandb, "run", None) is None:
#         return
#     try:
#         wandb.finish()
#     except Exception:
#         pass


# def _subprocess_env_disable_wandb(parent_env: dict) -> dict:
#     """
#     Hard-disable W&B for subprocess AND strip sweep/agent env vars
#     so main.py won't auto-enable wandb when it detects sweep env.
#     """
#     env = dict(parent_env)

#     for k in ("WANDB_SWEEP_ID", "WANDB_RUN_ID", "WANDB_AGENT_ID", "WANDB_LAUNCH_ID"):
#         env.pop(k, None)

#     env["WANDB_MODE"] = "disabled"
#     env["WANDB_DISABLED"] = "true"
#     return env


# # -------------------------
# # main
# # -------------------------
# def main():
#     parser = build_parser()
#     args, extra_args = parser.parse_known_args()

#     os.makedirs(args.results_dir, exist_ok=True)

#     run, created_by_us = _wandb_get_run_or_init(args)

#     try:
#         # write cfg/* early (visible on run page)
#         if run is not None:
#             _wandb_summary_update(_make_cfg_summary(args))

#         fold_results = []

#         for fold in range(args.folds):
#             seed = args.base_seed + fold
#             print(f"\n================ Fold {fold} (seed={seed}) ================\n")

#             cmd = [
#                 args.python, args.main,
#                 "--dataset", args.dataset,
#                 "--seed", str(seed),
#                 "--split_seed", str(args.base_seed),
#                 "--dl_num_workers", "0",

#                 "--num_epochs", str(args.num_epochs),
#                 "--warmup_epoch", str(args.warmup_epoch),
#                 "--patience", str(args.patience),
#                 "--batch_size", str(args.batch_size),
#                 "--lr", str(args.lr),
#                 "--weight_decay", str(args.weight_decay),
#                 "--masker_lr_mult", str(args.masker_lr_mult),
#                 "--grad_clip", str(args.grad_clip),

#                 "--gnn_hidden_dim", str(args.gnn_hidden_dim),
#                 "--gnn_out_dim", str(args.gnn_out_dim),
#                 "--num_gnn_layers", str(args.num_gnn_layers),
#                 "--dropout", str(args.dropout),
#                 "--num_feats", str(args.num_feats),

#                 "--num_classes", str(args.num_classes),
#                 "--num_sex_classes", str(args.num_sex_classes),
#                 "--num_age_classes", str(args.num_age_classes),

#                 "--fixed_thr", str(args.fixed_thr),
#                 "--select_by", str(args.select_by),

#                 "--eval_thr_mode", str(args.eval_thr_mode),
#                 "--eval_thr_grid", str(args.eval_thr_grid),
#                 ("--eval_use_best_thr" if args.eval_use_best_thr else "--no_eval_use_best_thr"),

#                 "--lambda_task", str(args.lambda_task),
#                 "--lambda_sensitive", str(args.lambda_sensitive),

#                 "--lambda_pfs", str(args.lambda_pfs),
#                 "--lambda_pfs_attr", str(args.lambda_pfs_attr),

#                 "--lambda_supcon", str(args.lambda_supcon),
#                 "--lambda_ortho", str(args.lambda_ortho),
#                 "--supcon_temperature", str(args.supcon_temperature),

#                 "--lambda_causal_suf", str(args.lambda_causal_suf),
#                 "--lam_ent_yb", str(args.lam_ent_yb),

#                 # score hyperparams (threshold selection + val_score)
#                 "--score_f1_beta", str(args.score_f1_beta),
#                 "--score_fair_alpha", str(args.score_fair_alpha),
#                 "--score_fair_w_age", str(args.score_fair_w_age),
#                 "--score_fair_w_sex", str(args.score_fair_w_sex),

#                 ("--cf_detach_zc" if args.cf_detach_zc else "--no_cf_detach_zc"),
#                 ("--cf_detach_zb_for_disease" if args.cf_detach_zb_for_disease else "--no_cf_detach_zb_for_disease"),
#                 ("--ent_yb_head_only" if args.ent_yb_head_only else "--no_ent_yb_head_only"),

#                 "--disease_pos_weight_mode", str(args.disease_pos_weight_mode),
#                 *([] if args.disease_pos_weight is None else ["--disease_pos_weight", str(args.disease_pos_weight)]),

#                 "--freeze_b_after", str(args.freeze_b_after),
#                 "--freeze_c_after", str(args.freeze_c_after),

#                 "--mask_reg_weight", str(args.mask_reg_weight),
#                 "--mask_temperature", str(args.mask_temperature),
#                 "--mask_hidden_dim", str(args.mask_hidden_dim),
#                 "--mask_target_mean", str(args.mask_target_mean),
#                 "--lambda_maskmean", str(args.lambda_maskmean),
#                 "--mask_target_std", str(args.mask_target_std),
#                 "--lambda_maskstd", str(args.lambda_maskstd),
#                 "--lambda_mask_dis", str(args.lambda_mask_dis),
#                 "--mask_dis_mode", str(args.mask_dis_mode),
#                 "--mask_topk_ratio", str(args.mask_topk_ratio),
#                 "--lambda_mask_l1", str(args.lambda_mask_l1),

#                 ("--two_stage" if args.two_stage else "--no_two_stage"),
#                 "--stage1_ratio", str(args.stage1_ratio),
#                 "--stage2_lr_mult", str(args.stage2_lr_mult),

#                 ("--adaptive_schedule" if args.adaptive_schedule else "--no_adaptive_schedule"),
#                 "--w_pfs_init", str(args.w_pfs_init),
#                 "--w_sup_init", str(args.w_sup_init),
#                 "--w_csuf_init", str(args.w_csuf_init),
#                 "--w_pfs_max", str(args.w_pfs_max),
#                 "--w_sup_max", str(args.w_sup_max),
#                 "--w_csuf_max", str(args.w_csuf_max),
#                 "--sched_grow", str(args.sched_grow),
#                 "--sched_decay", str(args.sched_decay),
#                 "--sched_patience", str(args.sched_patience),
#                 "--mask_min_mean", str(args.mask_min_mean),
#                 "--mask_max_norm_overlap", str(args.mask_max_norm_overlap),

#                 "--out_dir", args.results_dir,
#                 "--use_5fold",
#                 "--fold_id", str(fold),
#             ]

#             # forward W&B into main.py ONLY when explicitly enabled AND not in agent env
#             if args.use_wandb and args.wandb_per_fold and (not _in_wandb_agent_env()):
#                 cmd += ["--use_wandb"]
#                 cmd += ["--wandb_project", str(args.wandb_project)]
#                 if args.wandb_entity:
#                     cmd += ["--wandb_entity", str(args.wandb_entity)]
#                 group = args.wandb_group or f"{args.dataset}_seed{args.base_seed}"
#                 cmd += ["--wandb_group", group]
#                 prefix = args.wandb_name_prefix or args.dataset
#                 cmd += ["--wandb_name", f"{prefix}_fold{fold}_seed{seed}"]

#                 if args.wandb_tags:
#                     cmd += ["--wandb_tags", str(args.wandb_tags)]
#                 if args.wandb_notes:
#                     cmd += ["--wandb_notes", str(args.wandb_notes)]
#                 if args.wandb_mode:
#                     cmd += ["--wandb_mode", str(args.wandb_mode)]

#             if extra_args:
#                 cmd += extra_args

#             print("[CMD]", " ".join(cmd))

#             # env handling for subprocess
#             if args.use_wandb and args.wandb_per_fold and (not _in_wandb_agent_env()):
#                 env = os.environ.copy()
#             else:
#                 env = _subprocess_env_disable_wandb(os.environ)

#             subprocess.run(cmd, check=True, cwd=args.cwd, env=env)

#             payload, used_path = _load_fold_result(args.results_dir, fold)
#             one = _normalize_fold_payload(payload, fold=fold, seed=seed)
#             one["_path"] = used_path
#             fold_results.append(one)

#             # fold-level log into OUTER run (history only)
#             if run is not None:
#                 fold_log = {"fold": fold, "seed": seed}
#                 for src_k, dst_k in [
#                     ("val.val_score_bestthr", "fold/val_score_bestthr"),
#                     ("test.accuracy", "fold/test_acc"),
#                     ("test.f1_score", "fold/test_f1"),
#                 ]:
#                     v = _get_nested(one, src_k)
#                     if isinstance(v, (int, float, np.number)) and np.isfinite(float(v)):
#                         fold_log[dst_k] = float(v)
#                 _wandb_log(fold_log, step=fold)

#         # -------------------------
#         # aggregate
#         # -------------------------
#         print("\n================ FINAL RESULTS ================\n")

#         preferred_keys = [
#             "best_epoch",
#             "val.val_score_bestthr",
#             "test.accuracy", "test.precision", "test.recall", "test.roc_auc", "test.f1_score",
#             "test.EO_sex", "test.EO_age", "test.SP_sex", "test.SP_age",
#         ]
#         always_keys = ["fold_id", "seed"]

#         keys = []
#         for k in always_keys + preferred_keys:
#             if len(_collect_numeric(fold_results, k)) > 0:
#                 keys.append(k)

#         summary = {
#             "n_folds": int(args.folds),
#             "dataset": args.dataset,
#             "folds": fold_results,
#             "agg": {},
#             "run_cfg": {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool))},
#         }

#         agg_for_wandb = {}

#         for k in keys:
#             vals = _collect_numeric(fold_results, k)
#             if len(vals) == 0:
#                 continue
#             m = float(np.mean(vals))
#             s = float(np.std(vals))
#             summary["agg"][k] = {"mean": m, "std": s, "n": len(vals)}
#             print(f"{k}: {_format_mean_std(m, s)}")

#             agg_for_wandb[f"sum/agg/{k}_mean"] = m
#             agg_for_wandb[f"sum/agg/{k}_std"] = s

#         out_path = os.path.join(args.results_dir, "summary_5fold.json")
#         with open(out_path, "w", encoding="utf-8") as f:
#             json.dump(summary, f, indent=2)
#         print(f"\n[OK] Saved: {out_path}")

#         # -------------------------
#         # log to W&B (outer run)
#         # -------------------------
#         if run is not None:
#             _wandb_log(agg_for_wandb, step=int(args.folds))

#             payload = {}
#             payload.update(_make_cfg_summary(args))
#             payload.update(agg_for_wandb)
#             payload["sum/agg/summary_path"] = out_path
#             payload["sum/agg/n_folds"] = int(args.folds)

#             _wandb_summary_update(payload)
#             _wandb_save(out_path)

#     finally:
#         _wandb_finish_if_created(created_by_us)


# if __name__ == "__main__":
#     main()

# src/run_5fold.py
# -------------------------------------------------------------
# DisBG 5-fold launcher (W&B sweep-safe, aggregation-to-summary)
# Now supports: K x 5fold (repeat each fold with multiple seeds)
# -------------------------------------------------------------

import os
import json
import argparse
import subprocess
import numpy as np


# -------------------------
# utils
# -------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def _in_wandb_agent_env() -> bool:
    for k in ("WANDB_SWEEP_ID", "WANDB_RUN_ID", "WANDB_AGENT_ID", "WANDB_LAUNCH_ID"):
        if os.environ.get(k):
            return True
    return False


def _get_nested(d, key: str):
    cur = d
    for k in key.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k, None)
    return cur


def _collect_numeric(results, key: str):
    vals = []
    for r in results:
        v = _get_nested(r, key)
        if isinstance(v, (int, float, np.number)) and np.isfinite(float(v)):
            vals.append(float(v))
    return vals


def _format_mean_std(m: float, s: float):
    if np.isnan(m) or np.isnan(s):
        return "nan ± nan"
    return f"{m:.2f} ± {s:.2f}"


def _load_fold_result(results_dir: str, fold: int):
    main_path = os.path.join(results_dir, f"summary_fold_{fold}_main.json")
    summary_path = os.path.join(results_dir, f"summary_fold_{fold}.json")
    legacy_path = os.path.join(results_dir, f"fold_{fold}.json")

    for p in (main_path, summary_path, legacy_path):
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f), p

    raise FileNotFoundError(
        f"Missing fold result json. Tried: {main_path} , {summary_path} , {legacy_path}"
    )


def _normalize_fold_payload(payload: dict, fold: int, seed: int, rep: int):
    out = {"fold_id": int(fold), "seed": int(seed), "rep": int(rep)}

    # new-style payload from main.py
    if isinstance(payload, dict) and ("best_val" in payload or "test_at_best" in payload):
        out["best_epoch"] = payload.get("best_epoch", -1)
        if isinstance(payload.get("best_val", None), dict):
            out["val"] = payload["best_val"]
        if isinstance(payload.get("test_at_best", None), dict):
            out["test"] = payload["test_at_best"]
        out["_raw"] = payload
        return out

    # legacy fallback
    out["_raw"] = payload
    if isinstance(payload, dict):
        if "best_epoch" in payload:
            out["best_epoch"] = payload["best_epoch"]
        if "val" in payload and isinstance(payload["val"], dict):
            out["val"] = payload["val"]
        if "test" in payload and isinstance(payload["test"], dict):
            out["test"] = payload["test"]
    return out


# -------------------------
# args
# -------------------------
def build_parser():
    p = argparse.ArgumentParser()

    # dataset
    p.add_argument("--dataset", type=str, choices=["ADHD", "ABIDE"], default="ADHD")

    # folds + repeats
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--base_seed", type=int, default=0)

    # NEW: repeats per fold
    p.add_argument("--seeds_per_fold", type=int, default=3, help="repeat each fold with K different seeds")
    p.add_argument("--seed_stride", type=int, default=1000, help="seed offset between repeats to reduce correlation")
    p.add_argument("--exp_tag", type=str, default=None, help="subdir tag under results_dir to avoid mixing runs")

    # training
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--warmup_epoch", type=int, default=10)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--masker_lr_mult", type=float, default=1.0)
    p.add_argument("--grad_clip", type=float, default=0.0)

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
    p.add_argument(
        "--select_by",
        type=str,
        default="val_score_bestthr",
        choices=["accuracy", "f1_score", "roc_auc", "val_score", "val_score_bestthr"],
    )
    p.add_argument("--eval_thr_mode", type=str, default="best_acc", choices=["fixed", "best_f1", "best_acc","best_score"])
    p.add_argument("--eval_thr_grid", type=str, default="coarse", choices=["coarse", "fine"])

    p.add_argument("--eval_use_best_thr", type=str2bool, nargs="?", const=True, default=True)
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

    p.add_argument("--ent_yb_head_only", type=str2bool, nargs="?", const=True, default=True)
    p.add_argument("--no_ent_yb_head_only", dest="ent_yb_head_only", action="store_false")

    # disease pos weight
    p.add_argument("--disease_pos_weight_mode", type=str, default="none", choices=["none", "sqrt", "linear"])
    p.add_argument("--disease_pos_weight", type=float, default=None)

    # early freeze maskers
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
    p.add_argument("--mask_topk_ratio", type=float, default=0.1)
    p.add_argument("--lambda_mask_l1", type=float, default=0.00)

    # two-stage
    p.add_argument("--two_stage", type=str2bool, nargs="?", const=True, default=True)
    p.add_argument("--no_two_stage", dest="two_stage", action="store_false")
    p.add_argument("--stage1_ratio", type=float, default=0.35)
    p.add_argument("--stage2_lr_mult", type=float, default=0.30)

    # adaptive schedule
    p.add_argument("--adaptive_schedule", type=str2bool, nargs="?", const=True, default=True)
    p.add_argument("--no_adaptive_schedule", dest="adaptive_schedule", action="store_false")

    p.add_argument("--w_pfs_init", type=float, default=0.10)
    p.add_argument("--w_sup_init", type=float, default=0.05)
    p.add_argument("--w_csuf_init", type=float, default=0.05)

    p.add_argument("--w_pfs_max", type=float, default=0.35)
    p.add_argument("--w_sup_max", type=float, default=0.25)
    p.add_argument("--w_csuf_max", type=float, default=0.20)

    p.add_argument("--sched_grow", type=float, default=0.04)
    p.add_argument("--sched_decay", type=float, default=0.10)
    p.add_argument("--sched_patience", type=int, default=5)

    p.add_argument("--mask_min_mean", type=float, default=0.08)
    p.add_argument("--mask_max_norm_overlap", type=float, default=1.40)

    # io
    p.add_argument("--results_dir", type=str, default="results")

    # execution
    p.add_argument("--python", type=str, default="python", help="python executable")
    p.add_argument("--main", type=str, default="main.py", help="entry script (main.py)")
    p.add_argument("--cwd", type=str, default=None, help="working directory for subprocess")

    # W&B (outer agg)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="DisBG-B")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_group", type=str, default=None)
    p.add_argument("--wandb_tags", type=str, default="")
    p.add_argument("--wandb_notes", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default=None)

    # per-fold wandb (debug only; should be OFF in sweeps)
    p.add_argument("--wandb_per_fold", type=str2bool, nargs="?", const=True, default=True)
    p.add_argument("--no_wandb_per_fold", dest="wandb_per_fold", action="store_false")

    # log aggregation to W&B
    p.add_argument("--wandb_log_agg", type=str2bool, nargs="?", const=True, default=True)
    p.add_argument("--no_wandb_log_agg", dest="wandb_log_agg", action="store_false")

    p.add_argument("--wandb_name_prefix", type=str, default=None)

    return p


# -------------------------
# W&B helpers (unchanged)
# -------------------------
def _make_cfg_summary(args) -> dict:
    return {
        "cfg/dataset": str(args.dataset),
        "cfg/folds": int(args.folds),
        "cfg/base_seed": int(args.base_seed),
        "cfg/seeds_per_fold": int(args.seeds_per_fold),
        "cfg/seed_stride": int(args.seed_stride),
        "cfg/batch_size": int(args.batch_size),
        "cfg/lr": float(args.lr),
        "cfg/weight_decay": float(args.weight_decay),
        "cfg/gnn_hidden_dim": int(args.gnn_hidden_dim),
        "cfg/num_gnn_layers": int(args.num_gnn_layers),
        "cfg/dropout": float(args.dropout),
        "cfg/lambda_sensitive": float(args.lambda_sensitive),
        "cfg/lambda_causal_suf": float(args.lambda_causal_suf),
        "cfg/lam_ent_yb": float(args.lam_ent_yb),
        "cfg/mask_target_mean": float(args.mask_target_mean),
        "cfg/mask_target_std": float(args.mask_target_std),
        "cfg/mask_temperature": float(args.mask_temperature),
        "cfg/mask_topk_ratio": float(args.mask_topk_ratio),
        "cfg/two_stage": bool(args.two_stage),
        "cfg/adaptive_schedule": bool(args.adaptive_schedule),
        "cfg/select_by": str(args.select_by),
        "cfg/eval_thr_mode": str(args.eval_thr_mode),
        "cfg/eval_thr_grid": str(args.eval_thr_grid),
        "cfg/eval_use_best_thr": bool(args.eval_use_best_thr),
        "cfg/score_f1_beta": float(args.score_f1_beta),
        "cfg/score_fair_alpha": float(args.score_fair_alpha),
    }


def _wandb_get_run_or_init(args):
    try:
        import wandb  # type: ignore
    except Exception:
        return None, False

    if (not getattr(args, "use_wandb", False)) or (not getattr(args, "wandb_log_agg", True)):
        return None, False

    in_agent = _in_wandb_agent_env()

    if in_agent:
        if getattr(wandb, "run", None) is None:
            try:
                run = wandb.init(
                    tags=(args.wandb_tags.split(",") if args.wandb_tags else None),
                    notes=args.wandb_notes,
                    config=vars(args),
                )
                return run, False
            except Exception as e:
                print(f"[WARN] wandb.init (agent) failed: {e}")
                return None, False

        try:
            cfg = vars(args).copy()
            cfg.pop("wandb_project", None)
            cfg.pop("wandb_entity", None)
            wandb.config.update(cfg, allow_val_change=True)
        except Exception:
            pass
        return wandb.run, False

    if getattr(args, "wandb_mode", None):
        os.environ["WANDB_MODE"] = str(args.wandb_mode)

    run = wandb.init(
        project=getattr(args, "wandb_project", None) or os.environ.get("WANDB_PROJECT"),
        entity=getattr(args, "wandb_entity", None) or os.environ.get("WANDB_ENTITY"),
        group=getattr(args, "wandb_group", None) or os.environ.get("WANDB_RUN_GROUP"),
        name=(f"{(args.wandb_name_prefix or args.dataset)}_{args.seeds_per_fold}x{args.folds}fold_seed{args.base_seed}"),
        tags=(args.wandb_tags.split(",") if args.wandb_tags else None),
        notes=args.wandb_notes,
        config=vars(args),
    )
    return run, True


def _wandb_log(d, step=None):
    try:
        import wandb  # type: ignore
    except Exception:
        return
    if getattr(wandb, "run", None) is None:
        return
    try:
        wandb.log(d, step=step, commit=True)
    except Exception:
        pass


def _wandb_summary_update(d):
    try:
        import wandb  # type: ignore
    except Exception:
        return
    if getattr(wandb, "run", None) is None:
        return
    for k, v in d.items():
        try:
            wandb.run.summary[k] = v
        except Exception:
            pass
    try:
        wandb.run.summary.update()
    except Exception:
        pass


def _wandb_save(path: str):
    try:
        import wandb  # type: ignore
    except Exception:
        return
    if getattr(wandb, "run", None) is None:
        return
    try:
        wandb.save(path)
    except Exception:
        pass


def _wandb_finish_if_created(created_by_us: bool):
    if not created_by_us:
        return
    try:
        import wandb  # type: ignore
    except Exception:
        return
    if getattr(wandb, "run", None) is None:
        return
    try:
        wandb.finish()
    except Exception:
        pass


def _subprocess_env_disable_wandb(parent_env: dict) -> dict:
    env = dict(parent_env)
    for k in ("WANDB_SWEEP_ID", "WANDB_RUN_ID", "WANDB_AGENT_ID", "WANDB_LAUNCH_ID"):
        env.pop(k, None)
    env["WANDB_MODE"] = "disabled"
    env["WANDB_DISABLED"] = "true"
    return env


# -------------------------
# main
# -------------------------
def main():
    parser = build_parser()
    args, extra_args = parser.parse_known_args()

    # IMPORTANT: put this experiment under a unique tag folder
    exp_tag = args.exp_tag
    if not exp_tag:
        exp_tag = f"{args.dataset}_{args.seeds_per_fold}x{args.folds}fold_seed{args.base_seed}"
    root_out = os.path.join(args.results_dir, exp_tag)
    os.makedirs(root_out, exist_ok=True)

    run, created_by_us = _wandb_get_run_or_init(args)

    try:
        if run is not None:
            _wandb_summary_update(_make_cfg_summary(args))

        all_results = []
        step_counter = 0

        for fold in range(args.folds):
            for rep in range(args.seeds_per_fold):
                # seed schedule: keep split fixed, only training randomness changes
                seed = args.base_seed + fold + rep * args.seed_stride

                # UNIQUE out_dir per run (avoid json overwrite)
                out_dir = os.path.join(root_out, f"fold{fold}_rep{rep}_seed{seed}")
                os.makedirs(out_dir, exist_ok=True)

                print(f"\n================ Fold {fold} / Rep {rep} (seed={seed}) ================\n")

                cmd = [
                    args.python, args.main,
                    "--dataset", args.dataset,
                    "--seed", str(seed),
                    "--split_seed", str(args.base_seed),
                    "--dl_num_workers", "0",

                    "--num_epochs", str(args.num_epochs),
                    "--warmup_epoch", str(args.warmup_epoch),
                    "--patience", str(args.patience),
                    "--batch_size", str(args.batch_size),
                    "--lr", str(args.lr),
                    "--weight_decay", str(args.weight_decay),
                    "--masker_lr_mult", str(args.masker_lr_mult),
                    "--grad_clip", str(args.grad_clip),

                    "--gnn_hidden_dim", str(args.gnn_hidden_dim),
                    "--gnn_out_dim", str(args.gnn_out_dim),
                    "--num_gnn_layers", str(args.num_gnn_layers),
                    "--dropout", str(args.dropout),
                    "--num_feats", str(args.num_feats),

                    "--num_classes", str(args.num_classes),
                    "--num_sex_classes", str(args.num_sex_classes),
                    "--num_age_classes", str(args.num_age_classes),

                    "--fixed_thr", str(args.fixed_thr),
                    "--select_by", str(args.select_by),
                    "--eval_thr_mode", str(args.eval_thr_mode),
                    "--eval_thr_grid", str(args.eval_thr_grid),
                    ("--eval_use_best_thr" if args.eval_use_best_thr else "--no_eval_use_best_thr"),

                    "--lambda_task", str(args.lambda_task),
                    "--lambda_sensitive", str(args.lambda_sensitive),
                    "--lambda_pfs", str(args.lambda_pfs),
                    "--lambda_pfs_attr", str(args.lambda_pfs_attr),

                    "--lambda_supcon", str(args.lambda_supcon),
                    "--lambda_ortho", str(args.lambda_ortho),
                    "--supcon_temperature", str(args.supcon_temperature),

                    "--lambda_causal_suf", str(args.lambda_causal_suf),
                    "--lam_ent_yb", str(args.lam_ent_yb),

                    "--score_f1_beta", str(args.score_f1_beta),
                    "--score_fair_alpha", str(args.score_fair_alpha),
                    "--score_fair_w_age", str(args.score_fair_w_age),
                    "--score_fair_w_sex", str(args.score_fair_w_sex),

                    ("--cf_detach_zc" if args.cf_detach_zc else "--no_cf_detach_zc"),
                    ("--cf_detach_zb_for_disease" if args.cf_detach_zb_for_disease else "--no_cf_detach_zb_for_disease"),
                    ("--ent_yb_head_only" if args.ent_yb_head_only else "--no_ent_yb_head_only"),

                    "--disease_pos_weight_mode", str(args.disease_pos_weight_mode),
                    *([] if args.disease_pos_weight is None else ["--disease_pos_weight", str(args.disease_pos_weight)]),

                    "--freeze_b_after", str(args.freeze_b_after),
                    "--freeze_c_after", str(args.freeze_c_after),

                    "--mask_reg_weight", str(args.mask_reg_weight),
                    "--mask_temperature", str(args.mask_temperature),
                    "--mask_hidden_dim", str(args.mask_hidden_dim),
                    "--mask_target_mean", str(args.mask_target_mean),
                    "--lambda_maskmean", str(args.lambda_maskmean),
                    "--mask_target_std", str(args.mask_target_std),
                    "--lambda_maskstd", str(args.lambda_maskstd),
                    "--lambda_mask_dis", str(args.lambda_mask_dis),
                    "--mask_dis_mode", str(args.mask_dis_mode),
                    "--mask_topk_ratio", str(args.mask_topk_ratio),
                    "--lambda_mask_l1", str(args.lambda_mask_l1),

                    ("--two_stage" if args.two_stage else "--no_two_stage"),
                    "--stage1_ratio", str(args.stage1_ratio),
                    "--stage2_lr_mult", str(args.stage2_lr_mult),

                    ("--adaptive_schedule" if args.adaptive_schedule else "--no_adaptive_schedule"),
                    "--w_pfs_init", str(args.w_pfs_init),
                    "--w_sup_init", str(args.w_sup_init),
                    "--w_csuf_init", str(args.w_csuf_init),
                    "--w_pfs_max", str(args.w_pfs_max),
                    "--w_sup_max", str(args.w_sup_max),
                    "--w_csuf_max", str(args.w_csuf_max),
                    "--sched_grow", str(args.sched_grow),
                    "--sched_decay", str(args.sched_decay),
                    "--sched_patience", str(args.sched_patience),
                    "--mask_min_mean", str(args.mask_min_mean),
                    "--mask_max_norm_overlap", str(args.mask_max_norm_overlap),

                    "--out_dir", out_dir,
                    "--use_5fold",
                    "--fold_id", str(fold),
                ]

                if args.use_wandb and args.wandb_per_fold and (not _in_wandb_agent_env()):
                    cmd += ["--use_wandb"]
                    cmd += ["--wandb_project", str(args.wandb_project)]
                    if args.wandb_entity:
                        cmd += ["--wandb_entity", str(args.wandb_entity)]
                    group = args.wandb_group or f"{args.dataset}_seed{args.base_seed}"
                    cmd += ["--wandb_group", group]
                    prefix = args.wandb_name_prefix or args.dataset
                    cmd += ["--wandb_name", f"{prefix}_fold{fold}_rep{rep}_seed{seed}"]

                    if args.wandb_tags:
                        cmd += ["--wandb_tags", str(args.wandb_tags)]
                    if args.wandb_notes:
                        cmd += ["--wandb_notes", str(args.wandb_notes)]
                    if args.wandb_mode:
                        cmd += ["--wandb_mode", str(args.wandb_mode)]

                if extra_args:
                    cmd += extra_args

                print("[CMD]", " ".join(cmd))

                if args.use_wandb and args.wandb_per_fold and (not _in_wandb_agent_env()):
                    env = os.environ.copy()
                else:
                    env = _subprocess_env_disable_wandb(os.environ)

                subprocess.run(cmd, check=True, cwd=args.cwd, env=env)

                payload, used_path = _load_fold_result(out_dir, fold)
                one = _normalize_fold_payload(payload, fold=fold, seed=seed, rep=rep)
                one["_path"] = used_path
                one["_out_dir"] = out_dir
                all_results.append(one)

                if run is not None:
                    fold_log = {"fold": fold, "rep": rep, "seed": seed}
                    for src_k, dst_k in [
                        ("val.val_score_bestthr", "run/val_score_bestthr"),
                        ("test.accuracy", "run/test_acc"),
                        ("test.f1_score", "run/test_f1"),
                    ]:
                        v = _get_nested(one, src_k)
                        if isinstance(v, (int, float, np.number)) and np.isfinite(float(v)):
                            fold_log[dst_k] = float(v)
                    _wandb_log(fold_log, step=step_counter)
                step_counter += 1

        # -------------------------
        # aggregate across ALL runs (folds x reps)
        # -------------------------
        print("\n================ FINAL RESULTS (ALL RUNS) ================\n")

        preferred_keys = [
            "best_epoch",
            "val.val_score_bestthr",
            "test.accuracy", "test.precision", "test.recall", "test.roc_auc", "test.f1_score",
            "test.EO_sex", "test.EO_age", "test.SP_sex", "test.SP_age",
        ]
        always_keys = ["fold_id", "rep", "seed"]

        keys = []
        for k in always_keys + preferred_keys:
            if len(_collect_numeric(all_results, k)) > 0:
                keys.append(k)

        summary = {
            "n_folds": int(args.folds),
            "seeds_per_fold": int(args.seeds_per_fold),
            "n_runs": int(args.folds * args.seeds_per_fold),
            "dataset": args.dataset,
            "runs": all_results,
            "agg": {},
            "run_cfg": {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool))},
        }

        agg_for_wandb = {}
        for k in keys:
            vals = _collect_numeric(all_results, k)
            if len(vals) == 0:
                continue
            m = float(np.mean(vals))
            s = float(np.std(vals))
            summary["agg"][k] = {"mean": m, "std": s, "n": len(vals)}
            print(f"{k}: {_format_mean_std(m, s)}")

            agg_for_wandb[f"sum/agg/{k}_mean"] = m
            agg_for_wandb[f"sum/agg/{k}_std"] = s

        out_path = os.path.join(root_out, "summary_kxfold.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[OK] Saved: {out_path}")

        if run is not None:
            _wandb_log(agg_for_wandb, step=int(args.folds * args.seeds_per_fold))
            payload = {}
            payload.update(_make_cfg_summary(args))
            payload.update(agg_for_wandb)
            payload["sum/agg/summary_path"] = out_path
            payload["sum/agg/n_folds"] = int(args.folds)
            payload["sum/agg/seeds_per_fold"] = int(args.seeds_per_fold)
            payload["sum/agg/n_runs"] = int(args.folds * args.seeds_per_fold)
            _wandb_summary_update(payload)
            _wandb_save(out_path)

    finally:
        _wandb_finish_if_created(created_by_us)


if __name__ == "__main__":
    main()
