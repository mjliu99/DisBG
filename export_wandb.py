import wandb
import pandas as pd
from collections import Counter

ENTITY  = "mjliujade-federation-university-australia"
PROJECT = "DisBG-next"

GROUP = None          # 例如 "ADHD_sweep_phase1"，不确定就先 None
SWEEP_ID = None       # 例如 "738pcpsq"，可选
ONLY_FINISHED = True

api = wandb.Api()

# ✅ 正确的 viewer 用法
viewer = api.viewer
print("viewer:", viewer)

runs = list(api.runs(f"{ENTITY}/{PROJECT}"))
print("Total runs fetched:", len(runs))

# 🔍 先看真实的 group / state 分布（非常重要）
print("Groups:", Counter([r.group for r in runs]).most_common(10))
print("States:", Counter([r.state for r in runs]).most_common(10))

rows = []
for r in runs:
    if ONLY_FINISHED and r.state != "finished":
        continue
    if GROUP is not None and r.group != GROUP:
        continue
    if SWEEP_ID is not None:
        if r.sweep is None or r.sweep.id != SWEEP_ID:
            continue

    row = {
        "id": r.id,
        "name": r.name,
        "state": r.state,
        "group": r.group,
        "created_at": str(r.created_at),
    }

    # ===== config（按你项目常用的）=====
    for k in [
        "dataset", "folds", "batch_size", "lr", "weight_decay",
        "gnn_hidden_dim", "num_gnn_layers", "dropout",
        "lambda_sensitive", "lambda_causal_suf", "lam_ent_yb",
        "mask_target_mean", "mask_target_std",
        "mask_temperature", "mask_topk_ratio",
        "two_stage", "adaptive_schedule"
    ]:
        if k in r.config:
            row[f"cfg/{k}"] = r.config[k]

    # ===== summary（你论文/分析关心的）=====
    for k in [
        "val/val_score_bestthr",
        "agg/test.accuracy_mean",
        "agg/test.precision_mean",
        "agg/test.roc_auc_mean",
        "agg/test.f1_score_mean",
        "agg/test.EO_sex_mean",
        "agg/test.EO_age_mean",
        "agg/test.SP_sex_mean",
        "agg/test.SP_age_mean",
    ]:
        if k in r.summary:
            row[f"sum/{k}"] = r.summary[k]

    rows.append(row)

df = pd.DataFrame(rows)
out = "wandb_export.csv"
df.to_csv(out, index=False, encoding="utf-8-sig")

print(f"[OK] Saved: {out}  rows={len(df)}  cols={len(df.columns)}")
