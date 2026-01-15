# DisBG: Debiasing Brain Graphs via Causal-Decoupled Subgraph Learning

This repository contains the official implementation of **DisBG**, a causal- and fairness-aware brain graph learning framework for psychiatric disorder detection on neuroimaging datasets.

---

## 1. Environment Setup

### 1.1 Requirements

- Python **3.8**
- **CUDA 11.0** (recommended)
- Conda (recommended)

---

### 1.2 Create Conda Environment

```bash
conda create -n disbg python=3.8
conda activate disbg
```

---

## 2. Install PyTorch (CUDA 11.0)

```bash
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 \
    -f https://download.pytorch.org/whl/torch_stable.html
```

---

## 3. Install PyTorch Geometric (matching PyTorch 1.7.1 + cu110)

```bash
pip install --no-index \
    torch-scatter==2.0.7 \
    torch-sparse==0.6.9 \
    torch-cluster==1.5.9 \
    torch-spline-conv==1.2.1 \
    -f https://data.pyg.org/whl/torch-1.7.1+cu110.html

pip install torch-geometric==1.7.2
```

---

## 4. Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Place Datasets in the Folder

Place dataset inside `.\datasets`.

---

## 5. Usage — Quick Test

Run the project on either dataset:

```bash
python DisBG/main.py --dataset ADHD
python DisBG/main.py --dataset ABIDE
```

If these run without error, your installation is correct.

---

## 6. Running Experiments & Hyperparameter Tuning

It is recommended to use weights and biases (wandb).

General usage:

```bash
python src/run_5fold.py --dataset [DATASET_NAME] [OPTIONS]
```

## 7. Visualization for brain connectivity

General usage:
```bash
python visualize_subgraphs_aal116.py \
  --dataset ADHD \
  --split test \
  --avg --diff \
  --topk 50 \
  --max_batches 999999 \
  --save_npz \
  --save_top_edges \
  --edges_topk 50
```

## 8. Visualization for adjacency matrix

Group-level:
```bash
python visualize_matrices_aal116.py \
  --dataset ADHD \
  --split test \
  --group \
  --max_batches 999999 \
  --export_topk_edges 50
```

Individual-level: 
```bash
python visualize_matrices_aal116.py \
  --dataset ADHD \
  --split test \
  --pair \
  --max_batches 999999 \
  --export_topk_edges 50
```

Sparse Visualization:
```bash
python visualize_matrices_aal116.py \
  --dataset ADHD \
  --split test \
  --group \
  --topk_matrix 50 \
  --export_topk_edges 50
```

## 9. Full CLI Reference (All Arguments)

# -------------------------

# Basics / I/O

# -------------------------

parser.add_argument("--dataset", type=str, required=True, choices=["ADHD", "ABIDE"],
help="Dataset name (required)")
parser.add_argument("--seed", type=int, default=0,
help="Global random seed")
parser.add_argument("--out_dir", type=str, default="results",
help="Directory to save logs/checkpoints/results")

# -------------------------

# 5-Fold Control

# -------------------------

parser.add_argument("--use_5fold", action="store_true",
help="Enable 5-fold mode")
parser.add_argument("--fold_id", type=int, default=0,
help="Fold index (0-based) when --use_5fold is enabled")

# -------------------------

# Reproducibility (Split vs Train)

# -------------------------

parser.add_argument("--split_seed", type=int, default=0,
help="Seed for DATA SPLIT ONLY (keep fixed across folds)")
parser.add_argument("--dl_num_workers", type=int, default=0,
help="DataLoader num_workers (0 for best reproducibility)")
parser.add_argument("--deterministic", type=str2bool, nargs="?", const=True, default=True,
help="Enable deterministic behavior (default: True)")
parser.add_argument("--strict_deterministic", type=str2bool, nargs="?", const=True, default=False,
help="Enable strict deterministic behavior (may reduce speed)")

# -------------------------

# Training

# -------------------------

parser.add_argument("--num_epochs", type=int, default=100,
help="Number of training epochs")
parser.add_argument("--warmup_epoch", type=int, default=10,
help="Warmup epochs (e.g., freeze maskers / train task head first)")
parser.add_argument("--patience", type=int, default=30,
help="Early stopping patience based on --select_by metric")
parser.add_argument("--batch_size", type=int, default=8,
help="Batch size for training")
parser.add_argument("--lr", type=float, default=1e-4,
help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=5e-4,
help="Weight decay")
parser.add_argument("--masker_lr_mult", type=float, default=1.0,
help="Learning-rate multiplier for masker parameters")
parser.add_argument("--grad_clip", type=float, default=0.0,
help="Clip grad norm (0 = disabled)")
parser.add_argument("--stageB_warmup", type=int, default=None,
help="Optional warmup epochs for bias branch / stage-B")

# -------------------------

# Model Hyperparameters

# -------------------------

parser.add_argument("--gnn_hidden_dim", type=int, default=64,
help="Hidden dimension of GNN layers")
parser.add_argument("--gnn_out_dim", type=int, default=64,
help="Output dimension of GNN encoder")
parser.add_argument("--num_gnn_layers", type=int, default=2,
help="Number of GNN layers")
parser.add_argument("--dropout", type=float, default=0.0,
help="Dropout rate")
parser.add_argument("--num_feats", type=int, default=116,
help="Number of ROI nodes/features (AAL116 = 116)")

parser.add_argument("--num_classes", type=int, default=2,
help="Number of disease classes")
parser.add_argument("--num_sex_classes", type=int, default=2,
help="Number of sex classes")
parser.add_argument("--num_age_classes", type=int, default=4,
help="Number of age classes")

# -------------------------

# Evaluation / Model Selection

# -------------------------

parser.add_argument("--fixed_thr", type=float, default=0.5,
help="Fixed threshold for reporting")
parser.add_argument("--select_by", type=str, default="val_score",
choices=["accuracy", "f1_score", "roc_auc", "val_score", "val_score_bestthr"],
help="Metric to select best checkpoint")
parser.add_argument("--eval_thr_mode", type=str, default="best_acc",
choices=["fixed", "best_f1", "best_acc", "best_score"],
help="Threshold selection mode for evaluation")
parser.add_argument("--eval_thr_grid", type=str, default="coarse",
choices=["coarse", "fine"],
help="Threshold search grid resolution")
parser.add_argument("--eval_use_best_thr", action="store_true", default=True,
help="If enabled, report metrics using best threshold (also logs fixed threshold)")
parser.add_argument("--no_eval_use_best_thr", dest="eval_use_best_thr", action="store_false",
help="Disable best-threshold evaluation")

# score hyperparameters (for val_score / val_score_bestthr)

parser.add_argument("--score_f1_beta", type=float, default=0.5,
help="Beta weight for F1 in composite score (val_score)")
parser.add_argument("--score_fair_alpha", type=float, default=0.01,
help="Alpha weight for fairness penalty in composite score (val_score)")
parser.add_argument("--score_fair_w_age", type=float, default=4.0,
help="Weight for age fairness penalty in composite score")
parser.add_argument("--score_fair_w_sex", type=float, default=1.0,
help="Weight for sex fairness penalty in composite score")

# -------------------------

# Loss Weights (Core + Aux)

# -------------------------

parser.add_argument("--lambda_task", type=float, default=1.0,
help="Weight for disease/task objective")
parser.add_argument("--lambda_sensitive", type=float, default=0.02,
help="Weight for sensitive heads (sex/age) CE inside task loss")
parser.add_argument("--lambda_pfs", type=float, default=0.3,
help="Weight for PFS loss")
parser.add_argument("--lambda_pfs_attr", type=float, default=0.1,
help="Alpha for attribute terms inside PFS")
parser.add_argument("--lambda_supcon", type=float, default=0.02,
help="Weight for supervised contrastive (SupCon) loss")
parser.add_argument("--lambda_ortho", type=float, default=0.05,
help="Orthogonality regularizer between causal/bias representations")
parser.add_argument("--supcon_temperature", type=float, default=0.07,
help="Temperature for SupCon loss")
parser.add_argument("--lambda_causal_suf", type=float, default=1.0,
help="Weight for causal sufficiency loss")
parser.add_argument("--lam_ent_yb", type=float, default=0.1,
help="Entropy regularization coefficient on disease logits from bias branch")

# -------------------------

# Counterfactual / Gradient Routing Controls

# -------------------------

parser.add_argument("--cf_detach_zc", type=str2bool, nargs="?", const=True, default=False,
help="Detach z_c in counterfactual branch (stabilize training)")
parser.add_argument("--no_cf_detach_zc", dest="cf_detach_zc", action="store_false",
help="Do not detach z_c in counterfactual branch")

parser.add_argument("--cf_detach_zb_for_disease", type=str2bool, nargs="?", const=True, default=True,
help="Detach permuted z_b when computing disease counterfactual logits")
parser.add_argument("--no_cf_detach_zb_for_disease", dest="cf_detach_zb_for_disease", action="store_false",
help="Do not detach permuted z_b for disease counterfactual logits")
parser.add_argument("--cf_no_detach_zb_for_disease", dest="cf_detach_zb_for_disease", action="store_false",
help="Alias of --no_cf_detach_zb_for_disease")

parser.add_argument("--ent_yb_head_only", type=str2bool, nargs="?", const=True, default=True,
help="Apply entropy regularization to head-only logits (default True)")
parser.add_argument("--no_ent_yb_head_only", dest="ent_yb_head_only", action="store_false",
help="Apply entropy regularization beyond head-only logits")

# -------------------------

# Class Imbalance (Disease Positive Weight)

# -------------------------

parser.add_argument("--disease_pos_weight_mode", type=str, default="none",
choices=["none", "sqrt", "linear"],
help="How to set positive class weight for disease CE")
parser.add_argument("--disease_pos_weight", type=float, default=None,
help="Manual positive class weight for disease CE (overrides mode if set)")

# -------------------------

# Freezing Schedule

# -------------------------

parser.add_argument("--freeze_b_after", type=int, default=25,
help="Freeze bias branch after N epochs (-1 disables)")
parser.add_argument("--freeze_c_after", type=int, default=-1,
help="Freeze causal branch after N epochs (-1 disables)")

# -------------------------

# Mask Regularization

# -------------------------

parser.add_argument("--mask_reg_weight", type=float, default=0.6,
help="Overall mask regularization weight")
parser.add_argument("--mask_temperature", type=float, default=1.0,
help="Mask sampling temperature")
parser.add_argument("--mask_hidden_dim", type=int, default=64,
help="Hidden dimension for edge mask generator")

parser.add_argument("--mask_target_mean", type=float, default=0.22,
help="Target mean sparsity for masks (causal & bias)")
parser.add_argument("--lambda_maskmean", type=float, default=0.30,
help="Penalty weight for (mean - target_mean)^2")

parser.add_argument("--mask_target_std", type=float, default=0.20,
help="Target std for masks to avoid uniform masks")
parser.add_argument("--lambda_maskstd", type=float, default=0.00,
help="Penalty weight for (std - target_std)^2")

parser.add_argument("--lambda_mask_dis", type=float, default=0.35,
help="Penalty weight for mask overlap (or normalized overlap)")
parser.add_argument("--mask_dis_mode", type=str, default="norm_overlap",
choices=["overlap", "norm_overlap"],
help="Disentanglement term type for masks")
parser.add_argument("--mask_topk_ratio", type=float, default=0.10,
help="Top-k ratio for reporting Jaccard overlap (diagnostic)")
parser.add_argument("--lambda_mask_l1", type=float, default=0.00,
help="L1-like penalty on mask means to encourage sparsity")

# -------------------------

# Two-Stage Training & Adaptive Scheduling

# -------------------------

parser.add_argument("--two_stage", type=str2bool, nargs="?", const=True, default=True,
help="Enable two-stage training")
parser.add_argument("--no_two_stage", dest="two_stage", action="store_false",
help="Disable two-stage training")
parser.add_argument("--stage1_ratio", type=float, default=0.35,
help="Stage1 length ratio; stage2 starts at round(stage1_ratio * num_epochs)")
parser.add_argument("--stage2_lr_mult", type=float, default=0.30,
help="Multiply optimizer LR once when entering stage2")

parser.add_argument("--adaptive_schedule", type=str2bool, nargs="?", const=True, default=True,
help="Enable adaptive scheduling for aux losses")
parser.add_argument("--no_adaptive_schedule", dest="adaptive_schedule", action="store_false",
help="Disable adaptive scheduling for aux losses")

parser.add_argument("--w_pfs_init", type=float, default=0.10,
help="Initial dynamic weight for PFS")
parser.add_argument("--w_sup_init", type=float, default=0.05,
help="Initial dynamic weight for SupCon")
parser.add_argument("--w_csuf_init", type=float, default=0.05,
help="Initial dynamic weight for causal sufficiency")

parser.add_argument("--w_pfs_max", type=float, default=0.25,
help="Max dynamic weight for PFS")
parser.add_argument("--w_sup_max", type=float, default=0.25,
help="Max dynamic weight for SupCon")
parser.add_argument("--w_csuf_max", type=float, default=0.20,
help="Max dynamic weight for causal sufficiency")

parser.add_argument("--sched_grow", type=float, default=0.04,
help="Per-epoch growth of dynamic weights when stable")
parser.add_argument("--sched_decay", type=float, default=0.10,
help="Per-epoch decay of dynamic weights when no improvement / collapse")
parser.add_argument("--sched_patience", type=int, default=5,
help="Patience (epochs) before decaying dynamic weights if val does not improve")

parser.add_argument("--mask_min_mean", type=float, default=0.08,
help="If mc_mean or mb_mean < this, treat as collapse")
parser.add_argument("--mask_max_norm_overlap", type=float, default=1.40,
help="If norm_overlap > this, treat as collapse")

# -------------------------

# Weights & Biases (W&B)

# -------------------------

parser.add_argument("--use_wandb", action="store_true",
help="Enable Weights & Biases logging")
parser.add_argument("--wandb_project", type=str, default=None,
help="W&B project name")
parser.add_argument("--wandb_entity", type=str, default=None,
help="W&B entity/user or team name")
parser.add_argument("--wandb_group", type=str, default=None,
help="W&B group name (useful for sweeps / runs)")
parser.add_argument("--wandb_name", type=str, default=None,
help="W&B run name")
parser.add_argument("--wandb_tags", type=str, default="",
help="Comma-separated W&B tags")
parser.add_argument("--wandb_notes", type=str, default=None,
help="W&B notes/description")
parser.add_argument("--wandb_mode", type=str, default=None,
help="W&B mode (e.g., online/offline/disabled)")

---

## 10. Example Experiment Run (1*5fold)

```bash
python src/run_5fold.py \
  --dataset ADHD \
  --batch_size 8 \
  --num_gnn_layers 3 \
  --gnn_hidden_dim 64 \
  --gnn_out_dim 64 \
  --num_epochs 100 \
  --warmup_epoch 10 \
  --two_stage true \
  --stage1_ratio 0.35 \
  --stage2_lr_mult 0.30 \
  --adaptive_schedule true \
  --lr 1e-4 \
  --weight_decay 5e-4 \
  --lambda_task 1.0 \
  --lambda_sensitive 0.06 \
  --lambda_pfs 0.3 \
  --lambda_supcon 0.5 \
  --lambda_causal_suf 2.0 \
  --lam_ent_yb 0.4 \
  --mask_target_mean 0.20 \
  --lambda_maskmean 0.30 \
  --lambda_mask_dis 0.12 \
  --mask_dis_mode norm_overlap \
  --lambda_mask_l1 0.0 \
  --lambda_maskstd 0.0 \
  --fixed_thr 0.35 \
  --select_by val_score

```bash
python src/run_5fold.py \
  --dataset ABIDE \
  --batch_size 8 \
  --num_gnn_layers 3 \
  --gnn_hidden_dim 64 \
  --gnn_out_dim 64 \
  --num_epochs 100 \
  --warmup_epoch 10 \
  --two_stage true \
  --stage1_ratio 0.55 \
  --stage2_lr_mult 0.20 \
  --adaptive_schedule true \
  --lr 2e-4 \
  --weight_decay 5e-4 \
  --dropout 0.05 \
  --lambda_task 1.0 \
  --lambda_sensitive 0.08 \
  --lambda_pfs 0.3 \
  --lambda_pfs_attr 0.20 \
  --lambda_supcon 0.02 \
  --lambda_ortho 0.10 \
  --lambda_causal_suf 0.8 \
  --lam_ent_yb 0.25 \
  --mask_topk_ratio 0.10 \
  --mask_target_mean 0.20 \
  --lambda_maskmean 0.10 \
  --lambda_mask_dis 0.5 \
  --mask_dis_mode norm_overlap \
  --lambda_mask_l1 0.0 \
  --lambda_maskstd 0.0 \
  --freeze_b_after 12 \
  --disease_pos_weight_mode sqrt \
  --score_fair_alpha 0.07 \
  --score_fair_w_age 4.0 \
  --score_fair_w_sex 2.0 \
  --score_f1_beta 0.15 \
  --eval_thr_mode best_score \
  --eval_thr_grid fine \
  --eval_use_best_thr \
  --select_by val_score_bestthr

```