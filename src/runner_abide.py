# src/runner.py
from __future__ import annotations

import os
import json
import math
import time
import random
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ---- optional wandb ----
try:
    import wandb  # type: ignore
except Exception:
    wandb = None
# import wandb
# wandb.init(project="DisBG", entity="mjliujade-federation-university-australia", config=vars(args))


# =============================================================
# W&B
# =============================================================
def _wandb_ready(args) -> bool:
    return (wandb is not None) and (getattr(wandb, "run", None) is not None)


def _wandb_log(d: dict, step: int = None, **kwargs):
    if wandb is None or getattr(wandb, "run", None) is None:
        return
    payload = {}
    for k, v in d.items():
        if isinstance(v, (int, float, np.number)) and np.isfinite(float(v)):
            payload[k] = float(v)
    if payload:
        try:
            wandb.log(payload, commit=True)  # 不传 step，避免非单调警告
        except Exception:
            pass



# =============================================================
# Logger
# =============================================================
class _TeeLogger:
    def __init__(self, filepath: Optional[str]):
        self.filepath = filepath
        self.fp = None
        if filepath is not None:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.fp = open(filepath, "a", encoding="utf-8")

    def close(self):
        if self.fp is not None:
            self.fp.close()
            self.fp = None

    def log(self, msg: str):
        print(msg)
        if self.fp is not None:
            self.fp.write(msg + "\n")
            self.fp.flush()


def _get_fold_id(args) -> int:
    fid = getattr(args, "fold_id", None)
    try:
        return int(fid) if fid is not None else -1
    except Exception:
        return -1


def _get_out_dir(args) -> str:
    out_dir = getattr(args, "out_dir", None)
    if out_dir is None:
        out_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _get_logger(args) -> Optional[_TeeLogger]:
    if args is None:
        return None
    out_dir = _get_out_dir(args)
    fid = _get_fold_id(args)
    path = os.path.join(out_dir, f"log_fold_{fid}.txt")
    return _TeeLogger(path)


# =============================================================
# Seed
# =============================================================
def set_seed(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =============================================================
# SupCon (safe)
# =============================================================
class SupConCrossLoss(nn.Module):
    def __init__(self, tau: float = 0.2, use_normalize: bool = True):
        super().__init__()
        self.tau = float(tau)
        self.use_normalize = bool(use_normalize)

    def forward(self, anchor: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = anchor.device
        B = anchor.size(0)
        if B <= 1:
            return anchor.new_tensor(0.0)

        if self.use_normalize:
            anchor = F.normalize(anchor, dim=1)

        labels = labels.view(B, 1)
        eye = torch.eye(B, device=device)

        pos_mask = (labels == labels.T).float() * (1.0 - eye)
        pos_cnt = pos_mask.sum(dim=1)
        valid = pos_cnt > 0

        logits = (anchor @ anchor.T) / max(self.tau, 1e-6)
        logits = logits * (1.0 - eye) + (-1e9) * eye

        log_den = torch.logsumexp(logits, dim=1)
        log_prob = logits - log_den.view(B, 1)

        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_cnt.clamp(min=1.0)
        if valid.any():
            return (-mean_log_prob_pos[valid]).mean()
        return anchor.new_tensor(0.0)


# =============================================================
# Utilities
# =============================================================
def _pick_attr(obj, names):
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            if v is not None:
                return v
    return None


def _require_label(data, names, label_name: str, allow_missing: bool = False):
    v = _pick_attr(data, names)
    if v is None:
        if allow_missing:
            return None
        keys = list(data.keys()) if hasattr(data, "keys") else []
        raise AttributeError(f"Batch has no {label_name} label. Tried {names}. Available keys: {keys}")
    return v.view(-1).long()


def _to_numpy_1d(x):
    if torch.is_tensor(x):
        return x.detach().view(-1).cpu().numpy()
    return np.asarray(x).reshape(-1)


def _nan(x: float) -> float:
    return float(x) if (x == x) else float("nan")


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logits, dim=-1)
    ent = -(p * torch.log(p + 1e-12)).sum(dim=-1)
    return ent.mean()


# =============================================================
# Mask freeze/unfreeze helpers
# =============================================================
def _set_maskers_trainable(model: nn.Module, trainable: bool):
    for attr in ["mask_gen_c", "mask_gen_b"]:
        if hasattr(model, attr):
            mg = getattr(model, attr)
            if mg is None:
                continue
            for p in mg.parameters():
                p.requires_grad = bool(trainable)


def _set_single_masker_trainable(model: nn.Module, attr: str, trainable: bool):
    if hasattr(model, attr):
        mg = getattr(model, attr)
        if mg is None:
            return
        for p in mg.parameters():
            p.requires_grad = bool(trainable)


def _maskers_trainable_state(model: nn.Module) -> str:
    st = []
    for attr in ["mask_gen_c", "mask_gen_b"]:
        if hasattr(model, attr):
            mg = getattr(model, attr)
            if mg is None:
                st.append(f"{attr}=None")
                continue
            any_train = any(p.requires_grad for p in mg.parameters())
            st.append(f"{attr}={'ON' if any_train else 'OFF'}")
    return ",".join(st) if st else "no_maskers"


# =============================================================
# Adaptive schedule (two-stage + collapse-aware)
# =============================================================
class AdaptiveObjectiveScheduler:
    def __init__(self, args):
        self.args = args
        self.num_epochs = int(getattr(args, "num_epochs", getattr(args, "epochs", 100)))

        self.adaptive_schedule = bool(getattr(args, "adaptive_schedule", True))
        self.two_stage = bool(getattr(args, "two_stage", True))

        self.stage1_ratio = float(getattr(args, "stage1_ratio", 0.35))
        self.stage1_end = int(round(self.stage1_ratio * self.num_epochs))

        self.w_pfs_max = float(getattr(args, "w_pfs_max", 0.25))
        self.w_sup_max = float(getattr(args, "w_sup_max", 0.25))
        self.w_csuf_max = float(getattr(args, "w_csuf_max", 0.20))

        self.w_pfs = float(getattr(args, "w_pfs_init", 0.10))
        self.w_sup = float(getattr(args, "w_sup_init", 0.05))
        self.w_csuf = float(getattr(args, "w_csuf_init", 0.05))

        self.grow = float(getattr(args, "sched_grow", 0.04))
        self.decay = float(getattr(args, "sched_decay", 0.10))
        self.patience = int(getattr(args, "sched_patience", 5))

        self.min_mean = float(getattr(args, "mask_min_mean", 0.08))
        self.max_norm_overlap = float(getattr(args, "mask_max_norm_overlap", 1.40))

        self.stage2_lr_mult = float(getattr(args, "stage2_lr_mult", 0.30))
        self._lr_scaled = False

        self.best_val = -1e18
        self.bad_epochs = 0

    def is_stage2(self, epoch: int) -> bool:
        return self.two_stage and (epoch >= self.stage1_end)

    def maybe_scale_lr(self, optimizer, epoch: int):
        if (optimizer is None) or (not self.is_stage2(epoch)) or self._lr_scaled:
            return
        mult = self.stage2_lr_mult
        if mult <= 0:
            return
        for pg in optimizer.param_groups:
            pg["lr"] = float(pg["lr"]) * mult
        self._lr_scaled = True

    def maybe_freeze_maskers(self, model, epoch: int):
        if (model is None) or (not self.is_stage2(epoch)):
            return
        _set_maskers_trainable(model, trainable=False)

    def export_to_args(self, epoch: int):
        setattr(self.args, "w_pfs_dyn", float(self.w_pfs))
        setattr(self.args, "w_sup_dyn", float(self.w_sup))
        setattr(self.args, "w_csuf_dyn", float(self.w_csuf))
        setattr(self.args, "stage2_active", bool(self.is_stage2(epoch)))

    def update(self, epoch: int, val_score_bestthr: float, mask_diag: dict):
        if not self.adaptive_schedule:
            if self.is_stage2(epoch):
                self.w_pfs = 0.0
                self.w_sup = 0.0
                self.w_csuf = 0.0
            return

        improved = val_score_bestthr > self.best_val + 1e-12
        if improved:
            self.best_val = val_score_bestthr
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        if self.is_stage2(epoch):
            self.w_pfs = 0.0
            self.w_sup = 0.0
            self.w_csuf = 0.0
            return

        mc = float(mask_diag.get("mc_mean", float("nan")))
        mb = float(mask_diag.get("mb_mean", float("nan")))
        no = float(mask_diag.get("norm_overlap", float("nan")))

        collapse = False
        if (mc == mc) and (mb == mb):
            if (mc < self.min_mean) or (mb < self.min_mean):
                collapse = True
        if (no == no) and (no > self.max_norm_overlap):
            collapse = True

        if collapse:
            self.w_pfs = max(0.0, self.w_pfs - 2.0 * self.decay)
            self.w_sup = max(0.0, self.w_sup - 2.0 * self.decay)
            self.w_csuf = max(0.0, self.w_csuf - 2.0 * self.decay)
        elif self.bad_epochs >= self.patience:
            self.w_pfs = max(0.0, self.w_pfs - self.decay)
            self.w_sup = max(0.0, self.w_sup - self.decay)
            self.w_csuf = max(0.0, self.w_csuf - self.decay)
        else:
            self.w_pfs = min(self.w_pfs_max, self.w_pfs + self.grow)
            self.w_sup = min(self.w_sup_max, self.w_sup + self.grow)
            self.w_csuf = min(self.w_csuf_max, self.w_csuf + self.grow)


# =============================================================
# Fairness metrics
# =============================================================
def _group_positive_rate(preds, group_mask):
    if group_mask.sum() == 0:
        return np.nan
    return float((preds[group_mask] == 1).mean())


def _group_tpr(preds, labels, group_mask):
    mask = group_mask & (labels == 1)
    if mask.sum() == 0:
        return np.nan
    return float((preds[mask] == 1).mean())


def _statistical_parity(preds, sens):
    if sens is None:
        return np.nan
    groups = np.unique(sens)
    rates = []
    for g in groups:
        rates.append(_group_positive_rate(preds, sens == g))
    rates = np.array(rates, dtype=float)
    rates = rates[~np.isnan(rates)]
    if rates.size == 0:
        return np.nan
    return float(np.max(rates) - np.min(rates))


def _equal_opportunity(preds, labels, sens):
    if sens is None:
        return np.nan
    groups = np.unique(sens)
    tprs = []
    for g in groups:
        tprs.append(_group_tpr(preds, labels, sens == g))
    tprs = np.array(tprs, dtype=float)
    tprs = tprs[~np.isnan(tprs)]
    if tprs.size == 0:
        return np.nan
    return float(np.max(tprs) - np.min(tprs))


def calculate_metrics(preds, labels, probs, y_s, y_a):
    acc = accuracy_score(labels, preds) if labels.size else float("nan")
    prec = precision_score(labels, preds, zero_division=0) if labels.size else float("nan")
    rec = recall_score(labels, preds, zero_division=0) if labels.size else float("nan")
    f1v = f1_score(labels, preds, zero_division=0) if labels.size else float("nan")
    try:
        auc = roc_auc_score(labels, probs) if labels.size else float("nan")
    except Exception:
        auc = float("nan")

    sp_sex = _statistical_parity(preds, y_s) if labels.size else float("nan")
    eo_sex = _equal_opportunity(preds, labels, y_s) if labels.size else float("nan")
    sp_age = _statistical_parity(preds, y_a) if labels.size else float("nan")
    eo_age = _equal_opportunity(preds, labels, y_a) if labels.size else float("nan")

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1v),
        "roc_auc": float(auc),
        "SP_sex": _nan(sp_sex),
        "EO_sex": _nan(eo_sex),
        "SP_age": _nan(sp_age),
        "EO_age": _nan(eo_age),
    }


# =============================================================
# Score function
# =============================================================
def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        fx = float(x)
        if np.isfinite(fx):
            return fx
    except Exception:
        pass
    return float(default)


def compute_val_score(val_logs: Dict[str, Any], args) -> float:
    """
    val_logs expects values in percent space.
    score = acc + beta*f1 - alpha*( w_sex*(SP_sex+EO_sex) + w_age*(SP_age+EO_age) )

    Key: for multi-group age fairness (4 bins), set w_age > w_sex.
    """
    acc = _safe_float(val_logs.get("accuracy", 0.0), 0.0)
    f1  = _safe_float(val_logs.get("f1_score", 0.0), 0.0)

    sp_sex = _safe_float(val_logs.get("SP_sex", 0.0), 0.0)
    eo_sex = _safe_float(val_logs.get("EO_sex", 0.0), 0.0)
    sp_age = _safe_float(val_logs.get("SP_age", 0.0), 0.0)
    eo_age = _safe_float(val_logs.get("EO_age", 0.0), 0.0)

    # hyperparams (safe defaults if not present)
    alpha = float(getattr(args, "score_fair_alpha", 0.01))
    beta  = float(getattr(args, "score_f1_beta", 0.5))

    # NEW: reweight fairness terms
    # recommended for your case: w_age=4.0~6.0, w_sex=1.0
    w_sex = float(getattr(args, "score_fair_w_sex", 1.0))
    w_age = float(getattr(args, "score_fair_w_age", 4.0))

    fair_sex = sp_sex + eo_sex
    fair_age = sp_age + eo_age

    fairness = w_sex * fair_sex + w_age * fair_age
    return float(acc + beta * f1 - alpha * fairness)



# =============================================================
# Train helpers (pos weight)
# =============================================================
def _ensure_disease_pos_weight(args, data_loader):
    fixed = getattr(args, "disease_pos_weight", None)
    if fixed is not None:
        return float(fixed)

    mode = str(getattr(args, "disease_pos_weight_mode", "none")).lower()

    pos = 0
    neg = 0
    for d in data_loader:
        y = d.y.view(-1).long()
        pos += int((y == 1).sum().item())
        neg += int((y == 0).sum().item())

    ratio = float(neg / (pos + 1e-12))
    if mode in ["none", "off", "no", "0"]:
        w = 1.0
    elif mode in ["linear", "raw", "negpos"]:
        w = ratio
    else:  # sqrt
        w = float(math.sqrt(ratio))

    args.disease_pos_weight = float(w)
    return float(w)


def _topk_jaccard(m1: torch.Tensor, m2: torch.Tensor, k_ratio: float = 0.1) -> float:
    m1 = m1.view(-1)
    m2 = m2.view(-1)
    n = int(m1.numel())
    if n <= 0:
        return float("nan")
    k = max(1, int(n * float(k_ratio)))
    idx1 = torch.topk(m1, k=k, largest=True).indices
    idx2 = torch.topk(m2, k=k, largest=True).indices
    s1 = set(idx1.detach().cpu().tolist())
    s2 = set(idx2.detach().cpu().tolist())
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return float(inter / max(1, union))


def _mask_constraints_from_args(model, device, args) -> Tuple[torch.Tensor, Dict[str, float]]:
    if not (hasattr(model, "mask_gen_c") and hasattr(model, "mask_gen_b")):
        return torch.tensor(0.0, device=device), {}

    m_c = getattr(model.mask_gen_c, "last_mask", None)
    m_b = getattr(model.mask_gen_b, "last_mask", None)
    if (m_c is None) or (m_b is None):
        return torch.tensor(0.0, device=device), {}

    m_c = m_c.view(-1)
    m_b = m_b.view(-1)

    reg_w = float(getattr(args, "mask_reg_weight", 0.6))
    target_mean = float(getattr(args, "mask_target_mean", 0.22))
    lam_mean = float(getattr(args, "lambda_maskmean", 0.30))

    lam_std = float(getattr(args, "lambda_maskstd", 0.0))
    target_std = float(getattr(args, "mask_target_std", 0.20))

    lam_dis = float(getattr(args, "lambda_mask_dis", 0.35))
    dis_mode = str(getattr(args, "mask_dis_mode", "norm_overlap"))

    lam_l1 = float(getattr(args, "lambda_mask_l1", 0.0))

    eps = 1e-6
    mc_mean = m_c.mean()
    mb_mean = m_b.mean()
    loss_mean = (mc_mean - target_mean).pow(2) + (mb_mean - target_mean).pow(2)

    mc_std = m_c.std(unbiased=False)
    mb_std = m_b.std(unbiased=False)
    loss_std = (mc_std - target_std).pow(2) + (mb_std - target_std).pow(2)

    overlap = (m_c * m_b).mean()
    norm_overlap = overlap / (mc_mean * mb_mean + eps)

    # corr(m_c, m_b)
    mc0 = m_c - mc_mean
    mb0 = m_b - mb_mean
    denom = (mc0.std(unbiased=False) * mb0.std(unbiased=False) + eps)
    corr_cb = (mc0 * mb0).mean() / denom

    loss_dis = overlap if dis_mode == "overlap" else norm_overlap
    loss_l1 = mc_mean + mb_mean

    loss = lam_mean * loss_mean + lam_std * loss_std + lam_dis * loss_dis + lam_l1 * loss_l1
    loss = reg_w * loss

    topk_j = _topk_jaccard(m_c, m_b, k_ratio=float(getattr(args, "mask_topk_ratio", 0.10)))

    stats = {
        "mc_mean": float(mc_mean.detach().item()),
        "mb_mean": float(mb_mean.detach().item()),
        "mc_std": float(mc_std.detach().item()),
        "mb_std": float(mb_std.detach().item()),
        "overlap": float(overlap.detach().item()),
        "norm_overlap": float(norm_overlap.detach().item()),
        "corr_cb": float(corr_cb.detach().item()),
        "topk_jaccard": float(topk_j),
        "loss_mean": float(loss_mean.detach().item()),
        "loss_std": float(loss_std.detach().item()),
        "loss_dis": float(loss_dis.detach().item()),
        "loss_l1": float(loss_l1.detach().item()),
        "mask_reg_weight": float(reg_w),
    }
    return loss, stats


# =============================================================
# Threshold selection
# =============================================================
def _threshold_grid(grid: str = "coarse"):
    grid = str(grid).lower()
    if grid == "fine":
        return np.linspace(0.05, 0.95, 181)  # step=0.005
    return np.linspace(0.05, 0.95, 91)  # step=0.01


def best_threshold_by_acc(labels_np: np.ndarray, probs_np: np.ndarray, thrs: np.ndarray):
    """
    Choose threshold maximizing accuracy.
    Tie-break:
      1) higher F1
      2) closer to label_pos_rate (avoid pathological pos_rate)
      3) closer to 0.5
    Returns: best_thr_acc, best_acc, best_f1_at_best_acc
    """
    if labels_np.size == 0 or probs_np.size == 0:
        return 0.5, float("nan"), float("nan")

    label_pos_rate = float((labels_np == 1).mean())

    best_acc = -1.0
    best_thr = 0.5
    best_f1 = -1.0
    best_pr_dist = 1e9
    best_dist05 = 1e9

    for t in thrs:
        pred = (probs_np >= t).astype(np.int64)
        accv = accuracy_score(labels_np, pred)
        f1v = f1_score(labels_np, pred, zero_division=0)
        pos_rate = float(pred.mean()) if pred.size else 0.0

        pr_dist = abs(pos_rate - label_pos_rate)
        dist05 = abs(float(t) - 0.5)

        if (accv > best_acc + 1e-12) or \
           (abs(accv - best_acc) <= 1e-12 and f1v > best_f1 + 1e-12) or \
           (abs(accv - best_acc) <= 1e-12 and abs(f1v - best_f1) <= 1e-12 and pr_dist < best_pr_dist - 1e-12) or \
           (abs(accv - best_acc) <= 1e-12 and abs(f1v - best_f1) <= 1e-12 and abs(pr_dist - best_pr_dist) <= 1e-12 and dist05 < best_dist05 - 1e-12):
            best_acc = float(accv)
            best_thr = float(t)
            best_f1 = float(f1v)
            best_pr_dist = float(pr_dist)
            best_dist05 = float(dist05)

    return best_thr, best_acc, best_f1

def best_threshold_by_score(labels_np, probs_np, y_s, y_a, thrs, args):
    """
    Choose threshold maximizing compute_val_score( percent-space metrics ).
    """
    best_t = 0.5
    best_score = -1e18

    for t in thrs:
        preds = (probs_np >= t).astype(np.int64)
        m = calculate_metrics(preds, labels_np, probs_np, y_s, y_a)

        # convert to percent space for compute_val_score
        val_logs = {
            "accuracy": m["accuracy"] * 100.0,
            "f1_score": m["f1_score"] * 100.0,
            "SP_sex": (m["SP_sex"] * 100.0) if (m["SP_sex"] == m["SP_sex"]) else 0.0,
            "EO_sex": (m["EO_sex"] * 100.0) if (m["EO_sex"] == m["EO_sex"]) else 0.0,
            "SP_age": (m["SP_age"] * 100.0) if (m["SP_age"] == m["SP_age"]) else 0.0,
            "EO_age": (m["EO_age"] * 100.0) if (m["EO_age"] == m["EO_age"]) else 0.0,
        }

        s = compute_val_score(val_logs, args)
        if s > best_score + 1e-12:
            best_score = float(s)
            best_t = float(t)

    return best_t, best_score


def _select_threshold(labels_np, probs_np, args, fixed_thr, ys_np=None, ya_np=None):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    def _safe_get(name, default):
        return getattr(args, name, default) if args is not None else default

    def _threshold_grid(grid: str = "coarse"):
        grid = str(grid).lower()
        if grid == "fine":
            return np.linspace(0.05, 0.95, 181)  # step=0.005
        return np.linspace(0.05, 0.95, 91)  # step=0.01

    def _group_positive_rate(preds, group_mask):
        if group_mask.sum() == 0:
            return np.nan
        return float((preds[group_mask] == 1).mean())

    def _group_tpr(preds, labels, group_mask):
        mask = group_mask & (labels == 1)
        if mask.sum() == 0:
            return np.nan
        return float((preds[mask] == 1).mean())

    def _statistical_parity(preds, sens):
        if sens is None:
            return np.nan
        groups = np.unique(sens)
        rates = []
        for g in groups:
            rates.append(_group_positive_rate(preds, sens == g))
        rates = np.array(rates, dtype=float)
        rates = rates[~np.isnan(rates)]
        if rates.size == 0:
            return np.nan
        return float(np.max(rates) - np.min(rates))

    def _equal_opportunity(preds, labels, sens):
        if sens is None:
            return np.nan
        groups = np.unique(sens)
        tprs = []
        for g in groups:
            tprs.append(_group_tpr(preds, labels, sens == g))
        tprs = np.array(tprs, dtype=float)
        tprs = tprs[~np.isnan(tprs)]
        if tprs.size == 0:
            return np.nan
        return float(np.max(tprs) - np.min(tprs))

    def _nz(x):
        return 0.0 if (x != x) else float(x)

    def _hinge(x, thr):
        # x, thr in [0,1]
        return max(0.0, float(x) - float(thr))

    def _calibrate_mean_only(probs, group, clip01=True):
        """
        Mean-only alignment across groups:
          p_g' = p_g - mean_g + mean_all
        Keeps ranking much better than std scaling.
        """
        if group is None:
            return probs
        probs = probs.astype(np.float64)
        g = group.astype(np.int64)

        mean_all = float(np.mean(probs))
        out = probs.copy()
        for gv in np.unique(g):
            idx = (g == gv)
            if idx.sum() < 5:
                continue
            mg = float(np.mean(probs[idx]))
            out[idx] = probs[idx] - mg + mean_all
        if clip01:
            out = np.clip(out, 0.0, 1.0)
        return out

    # -----------------------
    # guard
    # -----------------------
    if labels_np is None or probs_np is None or labels_np.size == 0 or probs_np.size == 0:
        return float(fixed_thr), "fixed", {"thr_used": float(fixed_thr)}

    mode = str(_safe_get("eval_thr_mode", "best_f1")).lower()
    grid_mode = str(_safe_get("eval_thr_grid", "coarse")).lower()
    thrs = _threshold_grid(grid_mode)

    # -----------------------
    # optional calibration (mean-only strongly recommended)
    # -----------------------
    calib_by = str(_safe_get("thr_calib_by", "none")).lower()
    clip01 = bool(_safe_get("thr_calib_clip", True))
    use_std = bool(_safe_get("thr_calib_use_std", False))  # default False now

    probs_use = probs_np
    if calib_by in ["age", "a"]:
        probs_use = _calibrate_mean_only(probs_use, ya_np, clip01=clip01)
    elif calib_by in ["sex", "s"]:
        probs_use = _calibrate_mean_only(probs_use, ys_np, clip01=clip01)
    elif calib_by in ["age+sex", "sex+age"]:
        probs_use = _calibrate_mean_only(probs_use, ya_np, clip01=clip01)
        probs_use = _calibrate_mean_only(probs_use, ys_np, clip01=clip01)

    # -----------------------
    # scoring
    # -----------------------
    alpha = float(_safe_get("score_fair_alpha", 0.05))
    beta = float(_safe_get("score_f1_beta", 0.15))
    w_age = float(_safe_get("score_fair_w_age", 3.0))
    w_sex = float(_safe_get("score_fair_w_sex", 1.0))

    # NEW: fairness tolerance (in rate space)
    # within tolerance => no penalty
    tol_age = float(_safe_get("score_fair_tol_age", 0.12))  # 12%
    tol_sex = float(_safe_get("score_fair_tol_sex", 0.10))  # 10%

    def _score(preds):
        acc = accuracy_score(labels_np, preds)
        f1v = f1_score(labels_np, preds, zero_division=0)

        sp_sex = _statistical_parity(preds, ys_np)
        eo_sex = _equal_opportunity(preds, labels_np, ys_np)
        sp_age = _statistical_parity(preds, ya_np)
        eo_age = _equal_opportunity(preds, labels_np, ya_np)

        # hinge penalty (only punish beyond tolerance)
        fair_pen = (
            w_sex * (_hinge(_nz(sp_sex), tol_sex) + _hinge(_nz(eo_sex), tol_sex))
            + w_age * (_hinge(_nz(sp_age), tol_age) + _hinge(_nz(eo_age), tol_age))
        )

        score = float(acc + beta * f1v - alpha * fair_pen)
        diag = {
            "acc": float(acc),
            "f1": float(f1v),
            "sp_sex": _nz(sp_sex),
            "eo_sex": _nz(eo_sex),
            "sp_age": _nz(sp_age),
            "eo_age": _nz(eo_age),
            "fair_pen": float(fair_pen),
            "tol_age": float(tol_age),
            "tol_sex": float(tol_sex),
        }
        return score, diag

    # -----------------------
    # select threshold
    # -----------------------
    if mode == "fixed":
        thr = float(fixed_thr)
        preds = (probs_use >= thr).astype(np.int64)
        s, diag = _score(preds)
        return thr, "fixed", {"score_at_thr": float(s), **diag}

    best = {"thr": 0.5, "acc": -1.0, "f1": -1.0, "score": -1e18, "diag": {}}

    for t in thrs:
        preds = (probs_use >= t).astype(np.int64)
        accv = accuracy_score(labels_np, preds)
        f1v = f1_score(labels_np, preds, zero_division=0)

        if mode in ["best_acc", "acc"]:
            key = (accv, f1v, -abs(float(t) - 0.5))
            best_key = (best["acc"], best["f1"], -abs(best["thr"] - 0.5))
            if key > best_key:
                best.update({"thr": float(t), "acc": float(accv), "f1": float(f1v)})

        elif mode in ["best_score", "score"]:
            s, diag = _score(preds)
            if s > best["score"] + 1e-12:
                best.update({"thr": float(t), "score": float(s), "acc": float(accv), "f1": float(f1v), "diag": diag})

        else:  # best_f1
            if f1v > best["f1"] + 1e-12:
                best.update({"thr": float(t), "acc": float(accv), "f1": float(f1v)})

    extra = {
        "thr_used": float(best["thr"]),
        "best_acc": float(best["acc"]),
        "best_f1": float(best["f1"]),
        "calib_by": calib_by,
        "calib_use_std": bool(use_std),
    }
    if mode in ["best_score", "score"]:
        extra["best_score"] = float(best["score"])
        extra.update(best["diag"])

    return float(best["thr"]), mode, extra



# =============================================================
# Eval
# =============================================================
@torch.no_grad()
def eval_epoch(
    model,
    data_loader,
    device,
    split_name: str = "val",
    fixed_thr: float = 0.5,
    verbose: bool = False,
    logger: Optional[_TeeLogger] = None,
):
    model.eval()
    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_examples = 0
    all_labels, all_probs = [], []
    all_y_s, all_y_a = [], []

    sex_names = ["y_s", "y_sex", "sex", "s", "y_gender", "gender"]
    age_names = ["y_a", "y_age", "age", "a"]

    for data in data_loader:
        data = data.to(device)
        x, edge_index, batch = data.x, data.edge_index, data.batch
        y = data.y.view(-1).long()

        y_s = _require_label(data, sex_names, "sex", allow_missing=True)
        y_a = _require_label(data, age_names, "age", allow_missing=True)

        outs = model(x, edge_index, batch)
        if not (isinstance(outs, (tuple, list)) and len(outs) == 8):
            raise RuntimeError("Model forward() must return 8 values: logits_d, logits_s, logits_a, logits_d_b, z_c, z_b, u_c, u_b")
        logits_d = outs[0]

        loss = ce(logits_d, y)
        probs = torch.softmax(logits_d, dim=1)[:, 1]

        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total_examples += bs

        all_labels.append(y.cpu())
        all_probs.append(probs.cpu())
        if y_s is not None:
            all_y_s.append(y_s.cpu())
        if y_a is not None:
            all_y_a.append(y_a.cpu())

    avg_loss = total_loss / max(1, total_examples)

    labels_np = _to_numpy_1d(torch.cat(all_labels)) if len(all_labels) else np.array([], dtype=np.int64)
    probs_np = _to_numpy_1d(torch.cat(all_probs)) if len(all_probs) else np.array([], dtype=np.float64)
    ys_np = _to_numpy_1d(torch.cat(all_y_s)) if len(all_y_s) else None
    ya_np = _to_numpy_1d(torch.cat(all_y_a)) if len(all_y_a) else None

    # fixed
    thr_fixed = float(fixed_thr)
    preds_fixed = (probs_np >= thr_fixed).astype(np.int64) if probs_np.size > 0 else np.array([], dtype=np.int64)
    metrics_fixed = calculate_metrics(preds_fixed, labels_np, probs_np, ys_np, ya_np)
    metrics_fixed["thr_used"] = thr_fixed

    # used
    args = getattr(model, "args", None)
    thr_used, mode_used, extra_diag = _select_threshold(
    labels_np, probs_np, args=args, fixed_thr=thr_fixed, ys_np=ys_np, ya_np=ya_np
    )

    preds_used = (probs_np >= thr_used).astype(np.int64) if probs_np.size > 0 else np.array([], dtype=np.int64)
    metrics_used = calculate_metrics(preds_used, labels_np, probs_np, ys_np, ya_np)
    metrics_used["thr_used"] = float(thr_used)

    # diagnostics
    label_pos_rate = float((labels_np == 1).mean()) if labels_np.size > 0 else float("nan")
    p_mean = float(probs_np.mean()) if probs_np.size > 0 else float("nan")
    p_std = float(probs_np.std()) if probs_np.size > 0 else float("nan")
    pos_rate_fixed = float(preds_fixed.mean()) if preds_fixed.size > 0 else float("nan")
    pos_rate_used = float(preds_used.mean()) if preds_used.size > 0 else float("nan")

    diag = {
        "label_pos_rate": label_pos_rate,
        "p1_mean": p_mean,
        "p1_std": p_std,
        "pos_rate_fixed": pos_rate_fixed,
        "pos_rate_thr_used": pos_rate_used,
        "eval_thr_mode_used": mode_used,
        "thr_used": float(thr_used),
        **extra_diag,
    }

    if verbose:
        tag = split_name.upper()
        msg = (
            f"[{tag}] fixed_thr={thr_fixed:.3f} "
            f"| fixed(acc={metrics_fixed['accuracy']*100:.2f} f1={metrics_fixed['f1_score']*100:.2f}) "
            f"| used({mode_used}) thr={thr_used:.3f} acc={metrics_used['accuracy']*100:.2f} f1={metrics_used['f1_score']*100:.2f} "
            f"| pos_rate_fixed={pos_rate_fixed*100:.2f}% pos_rate_used={pos_rate_used*100:.2f}%"
        )
        (logger.log(msg) if logger is not None else print(msg))

    return avg_loss, metrics_fixed, metrics_used, diag


# =============================================================
# Evaluate/Test (public)
# =============================================================
@torch.no_grad()
def evaluate(model, loader, device, thr: Optional[float] = None):
    """
    Return dict in percent space, and provides:
      - val_score: score on fixed_thr metrics
      - val_score_bestthr: score on thr_used(metrics_used)
    """
    args = getattr(model, "args", None)
    fixed_thr = float(thr) if thr is not None else float(getattr(args, "fixed_thr", 0.5) if args is not None else 0.5)
    verbose = bool(getattr(args, "eval_verbose", False)) if args is not None else False
    logger = _get_logger(args) if verbose else None

    try:
        avg_loss, m_fixed, m_used, diag = eval_epoch(
            model, loader, device, split_name="val", fixed_thr=fixed_thr, verbose=verbose, logger=logger
        )

        use_best_thr = bool(getattr(args, "eval_use_best_thr", True)) if args is not None else True
        m_for_report = m_used if use_best_thr else m_fixed

        tmp_args = args if args is not None else type("Tmp", (), {"score_fair_alpha": 0.01, "score_f1_beta": 0.5})()

        def _pct(m):
            return {
                "accuracy": m["accuracy"] * 100.0,
                "f1_score": m["f1_score"] * 100.0,
                "SP_sex": (m["SP_sex"] * 100.0) if (m["SP_sex"] == m["SP_sex"]) else 0.0,
                "EO_sex": (m["EO_sex"] * 100.0) if (m["EO_sex"] == m["EO_sex"]) else 0.0,
                "SP_age": (m["SP_age"] * 100.0) if (m["SP_age"] == m["SP_age"]) else 0.0,
                "EO_age": (m["EO_age"] * 100.0) if (m["EO_age"] == m["EO_age"]) else 0.0,
            }

        val_score_fixed = compute_val_score(_pct(m_fixed), tmp_args)
        val_score_used = compute_val_score(_pct(m_used), tmp_args)

        out = {
            "loss": float(avg_loss),

            "accuracy": float(m_for_report["accuracy"]) * 100.0,
            "precision": float(m_for_report["precision"]) * 100.0,
            "recall": float(m_for_report["recall"]) * 100.0,
            "f1_score": float(m_for_report["f1_score"]) * 100.0,
            "roc_auc": (float(m_for_report["roc_auc"]) * 100.0) if (m_for_report["roc_auc"] == m_for_report["roc_auc"]) else float("nan"),

            "SP_sex": (float(m_for_report["SP_sex"]) * 100.0) if (m_for_report["SP_sex"] == m_for_report["SP_sex"]) else float("nan"),
            "EO_sex": (float(m_for_report["EO_sex"]) * 100.0) if (m_for_report["EO_sex"] == m_for_report["EO_sex"]) else float("nan"),
            "SP_age": (float(m_for_report["SP_age"]) * 100.0) if (m_for_report["SP_age"] == m_for_report["SP_age"]) else float("nan"),
            "EO_age": (float(m_for_report["EO_age"]) * 100.0) if (m_for_report["EO_age"] == m_for_report["EO_age"]) else float("nan"),

            "fixed_thr": float(m_fixed["thr_used"]),
            "thr_used": float(diag.get("thr_used", m_used.get("thr_used", fixed_thr))),
            "eval_thr_mode_used": str(diag.get("eval_thr_mode_used", "unknown")),

            "best_thr_acc": float(diag.get("best_thr_acc", float("nan"))),
            "best_acc": float(diag.get("best_acc", float("nan"))) * 100.0 if diag.get("best_acc", float("nan")) == diag.get("best_acc", float("nan")) else float("nan"),
            "best_acc_f1": float(diag.get("best_acc_f1", float("nan"))) * 100.0 if diag.get("best_acc_f1", float("nan")) == diag.get("best_acc_f1", float("nan")) else float("nan"),
            "best_thr_f1": float(diag.get("best_thr_f1", float("nan"))),
            "best_f1": float(diag.get("best_f1", float("nan"))) * 100.0 if diag.get("best_f1", float("nan")) == diag.get("best_f1", float("nan")) else float("nan"),

            "label_pos_rate": float(diag.get("label_pos_rate", float("nan"))) * 100.0,
            "pos_rate_fixed": float(diag.get("pos_rate_fixed", float("nan"))) * 100.0,
            "pos_rate_thr_used": float(diag.get("pos_rate_thr_used", float("nan"))) * 100.0,

            # scores
            "val_score": float(val_score_fixed),
            "val_score_bestthr": float(val_score_used),
            "eval_use_best_thr": bool(use_best_thr),
        }
        return out
    finally:
        if logger is not None:
            logger.close()


@torch.no_grad()
def test(model, loader, device, thr: Optional[float] = None):
    args = getattr(model, "args", None)
    fixed_thr = float(thr) if thr is not None else float(getattr(args, "fixed_thr", 0.5) if args is not None else 0.5)
    verbose = bool(getattr(args, "eval_verbose", False)) if args is not None else False
    logger = _get_logger(args) if verbose else None

    try:
        avg_loss, m_fixed, m_used, diag = eval_epoch(
            model, loader, device, split_name="test", fixed_thr=fixed_thr, verbose=verbose, logger=logger
        )

        use_best_thr = bool(getattr(args, "eval_use_best_thr", True)) if args is not None else True
        m_for_report = m_used if use_best_thr else m_fixed

        tmp_args = args if args is not None else type("Tmp", (), {"score_fair_alpha": 0.01, "score_f1_beta": 0.5})()

        def _pct(m):
            return {
                "accuracy": m["accuracy"] * 100.0,
                "f1_score": m["f1_score"] * 100.0,
                "SP_sex": (m["SP_sex"] * 100.0) if (m["SP_sex"] == m["SP_sex"]) else 0.0,
                "EO_sex": (m["EO_sex"] * 100.0) if (m["EO_sex"] == m["EO_sex"]) else 0.0,
                "SP_age": (m["SP_age"] * 100.0) if (m["SP_age"] == m["SP_age"]) else 0.0,
                "EO_age": (m["EO_age"] * 100.0) if (m["EO_age"] == m["EO_age"]) else 0.0,
            }

        score_fixed = compute_val_score(_pct(m_fixed), tmp_args)
        score_used = compute_val_score(_pct(m_used), tmp_args)

        out = {
            "loss": float(avg_loss),

            "accuracy": float(m_for_report["accuracy"]) * 100.0,
            "precision": float(m_for_report["precision"]) * 100.0,
            "recall": float(m_for_report["recall"]) * 100.0,
            "f1_score": float(m_for_report["f1_score"]) * 100.0,
            "roc_auc": (float(m_for_report["roc_auc"]) * 100.0) if (m_for_report["roc_auc"] == m_for_report["roc_auc"]) else float("nan"),

            "SP_sex": (float(m_for_report["SP_sex"]) * 100.0) if (m_for_report["SP_sex"] == m_for_report["SP_sex"]) else float("nan"),
            "EO_sex": (float(m_for_report["EO_sex"]) * 100.0) if (m_for_report["EO_sex"] == m_for_report["EO_sex"]) else float("nan"),
            "SP_age": (float(m_for_report["SP_age"]) * 100.0) if (m_for_report["SP_age"] == m_for_report["SP_age"]) else float("nan"),
            "EO_age": (float(m_for_report["EO_age"]) * 100.0) if (m_for_report["EO_age"] == m_for_report["EO_age"]) else float("nan"),

            "fixed_thr": float(m_fixed["thr_used"]),
            "thr_used": float(diag.get("thr_used", m_used.get("thr_used", fixed_thr))),
            "eval_thr_mode_used": str(diag.get("eval_thr_mode_used", "unknown")),

            "label_pos_rate": float(diag.get("label_pos_rate", float("nan"))) * 100.0,
            "pos_rate_fixed": float(diag.get("pos_rate_fixed", float("nan"))) * 100.0,
            "pos_rate_thr_used": float(diag.get("pos_rate_thr_used", float("nan"))) * 100.0,

            "val_score": float(score_fixed),
            "val_score_bestthr": float(score_used),
            "eval_use_best_thr": bool(use_best_thr),
        }
        return out
    finally:
        if logger is not None:
            logger.close()


# =============================================================
# Training epoch
# =============================================================
def train_epoch(model, data_loader, optimizer, epoch, device, args, logger: Optional[_TeeLogger] = None):
    model.train()

    warmup_epoch = int(getattr(args, "warmup_epoch", 0))

    # ---- trainability gates ----
    if epoch < warmup_epoch:
        _set_maskers_trainable(model, trainable=False)
    else:
        if not bool(getattr(args, "stage2_active", False)):
            _set_maskers_trainable(model, trainable=True)

            freeze_b_after = int(getattr(args, "freeze_b_after", 25))
            if freeze_b_after >= 0 and epoch >= freeze_b_after:
                _set_single_masker_trainable(model, "mask_gen_b", trainable=False)

            freeze_c_after = int(getattr(args, "freeze_c_after", -1))
            if freeze_c_after >= 0 and epoch >= freeze_c_after:
                _set_single_masker_trainable(model, "mask_gen_c", trainable=False)

    print_steps = bool(getattr(args, "print_steps", False))
    log_interval = int(getattr(args, "log_interval", 50))
    grad_clip = float(getattr(args, "grad_clip", 0.0))

    lambda_task = float(getattr(args, "lambda_task", 1.0))
    lambda_sensitive = float(getattr(args, "lambda_sensitive", 0.0))
    lambda_pfs = float(getattr(args, "lambda_pfs", 1.0))
    lambda_supcon = float(getattr(args, "lambda_supcon", 0.5))
    lambda_causal_suf = float(getattr(args, "lambda_causal_suf", 1.0))

    alpha_cf = float(getattr(args, "lambda_pfs_attr", 0.1))
    lam_ortho = float(getattr(args, "lambda_ortho", 0.05))
    lam_ent_yb = float(getattr(args, "lam_ent_yb", 0.1))

    r_pfs = float(getattr(args, "w_pfs_dyn", 0.0))
    r_sup = float(getattr(args, "w_sup_dyn", 0.0))
    r_csuf = float(getattr(args, "w_csuf_dyn", r_sup))

    stage2_active = bool(getattr(args, "stage2_active", False))

    supcon = SupConCrossLoss(
        tau=float(getattr(args, "supcon_temperature", 0.07)),
        use_normalize=True,
    )

    # disease class weight
    w_pos = _ensure_disease_pos_weight(args, data_loader)
    ce_d = nn.CrossEntropyLoss(weight=torch.tensor([1.0, w_pos], device=device))
    ce_s = nn.CrossEntropyLoss()
    ce_a = nn.CrossEntropyLoss()

    total_examples = 0
    total_correct = 0
    total_loss = 0.0

    sex_names = ["y_s", "y_sex", "sex", "s", "y_gender", "gender"]
    age_names = ["y_a", "y_age", "age", "a"]

    # =========================================================
    # NEW: better defaults for gradient routing
    # =========================================================
    # (1) Make PFS disease term actually train encoder_c/mask_c after warmup
    #     Default: False (i.e., do NOT detach z_c)
    cf_detach_zc = bool(getattr(args, "cf_detach_zc", False))

    # (2) Prevent PFS disease from updating encoder_b/mask_b (safer)
    #     Default: True (detach z_b_perm for disease CF)
    cf_detach_zb_for_disease = bool(getattr(args, "cf_detach_zb_for_disease", True))

    # (3) Optional: make entropy(logits_d_b) update ONLY bias head (not encoder_b/mask_b)
    #     Default: True (safer; avoids entropy destabilizing bias branch)
    ent_yb_head_only = bool(getattr(args, "ent_yb_head_only", True))

    # =========================================================

    last_mask_stats: Dict[str, float] = {}
    printed_once = False

    # epoch-avg mask diagnostics
    mask_sum = {"mc_mean": 0.0, "mb_mean": 0.0, "norm_overlap": 0.0, "corr_cb": 0.0}
    mask_count = 0

    for step, data in enumerate(data_loader):
        data = data.to(device)
        x, edge_index, batch = data.x, data.edge_index, data.batch
        y = data.y.view(-1).long()
        y_s = _require_label(data, sex_names, "sex")
        y_a = _require_label(data, age_names, "age")

        optimizer.zero_grad(set_to_none=True)

        outs = model(x, edge_index, batch)
        if not (isinstance(outs, (tuple, list)) and len(outs) == 8):
            raise RuntimeError(
                "Model forward() must return 8 values: logits_d, logits_s, logits_a, logits_d_b, z_c, z_b, u_c, u_b"
            )
        logits_d, logits_s, logits_a, logits_d_b, z_c, z_b, u_c, u_b = outs

        # -------------------------
        # (A) main task losses
        # -------------------------
        loss_task_d = ce_d(logits_d, y)
        loss_task_s = ce_s(logits_s, y_s)
        loss_task_a = ce_a(logits_a, y_a)

        # entropy regularization on bias disease head
        if epoch >= warmup_epoch and lam_ent_yb != 0.0:
            if ent_yb_head_only:
                # ENTROPY updates only classifier_disease_bias (head), not encoder_b/mask_b
                if not hasattr(model, "classifier_disease_bias"):
                    raise AttributeError("ent_yb_head_only=True requires model.classifier_disease_bias.")
                z_for_b_det = torch.cat([z_c.detach(), z_b.detach()], dim=-1)
                logits_d_b_ent = model.classifier_disease_bias(z_for_b_det)
                loss_ent_yb = entropy_from_logits(logits_d_b_ent)
            else:
                # original behavior: entropy can flow into bias branch
                loss_ent_yb = entropy_from_logits(logits_d_b)
        else:
            loss_ent_yb = torch.tensor(0.0, device=device)

        loss_task = loss_task_d + lambda_sensitive * (loss_task_s + loss_task_a)
        loss_task = loss_task - lam_ent_yb * loss_ent_yb

        # -------------------------
        # (B) counterfactual PFS
        # -------------------------
        if not hasattr(model, "classify_counterfactual"):
            raise AttributeError("Model must implement classify_counterfactual(z_c, z_b_perm).")

        perm = torch.randperm(y.size(0), device=device)

        # KEY CHANGE: by default, do NOT detach z_c after warmup,
        # so loss_pfs_d can shape encoder_c + mask_gen_c
        if epoch < warmup_epoch:
            zc_cf = z_c.detach()
        else:
            zc_cf = z_c.detach() if cf_detach_zc else z_c

        # Safer default: detach bias for disease CF (do not push encoder_b via PFS disease)
        zb_perm = z_b[perm].detach() if cf_detach_zb_for_disease else z_b[perm]

        logits_d_cf, logits_s_cf, logits_a_cf = model.classify_counterfactual(zc_cf, zb_perm)

        loss_pfs_d = ce_d(logits_d_cf, y)
        loss_pfs_attr = ce_s(logits_s_cf, y_s[perm]) + ce_a(logits_a_cf, y_a[perm])
        loss_pfs = loss_pfs_d + alpha_cf * loss_pfs_attr

        # -------------------------
        # (C) SupCon + Ortho
        # -------------------------
        supcon_active = (r_sup > 0.0) and (epoch >= warmup_epoch) and (not stage2_active)
        if supcon_active and (lambda_supcon != 0.0):
            loss_sup_c = supcon(u_c, y)
            loss_sup_b = supcon(u_b, y_s) + supcon(u_b, y_a)

            uc_n = F.normalize(u_c, dim=1)
            ub_n = F.normalize(u_b, dim=1)
            loss_ortho = (torch.sum(uc_n * ub_n, dim=1).pow(2)).mean()
        else:
            loss_sup_c = torch.tensor(0.0, device=device)
            loss_sup_b = torch.tensor(0.0, device=device)
            loss_ortho = torch.tensor(0.0, device=device)

        # -------------------------
        # (D) causal sufficiency (on u_c)
        # -------------------------
        if (epoch >= warmup_epoch) and (r_csuf > 0.0) and (not stage2_active) and (float(lambda_causal_suf) != 0.0):
            loss_causal_suf = supcon(u_c, y)
        else:
            loss_causal_suf = torch.tensor(0.0, device=device)

        # -------------------------
        # (E) mask regularization + stats
        # -------------------------
        loss_mask, mask_stats = _mask_constraints_from_args(model, device, args)
        if mask_stats:
            last_mask_stats = mask_stats

        # -------------------------
        # (F) total loss
        # -------------------------
        if epoch < warmup_epoch:
            loss = loss_task_d
        else:
            loss = (
                lambda_task * loss_task
                + (lambda_pfs * r_pfs) * loss_pfs
                + (lambda_supcon * r_sup) * (loss_sup_c + loss_sup_b + lam_ortho * loss_ortho)
                + (lambda_causal_suf * r_csuf) * loss_causal_suf
                + loss_mask
            )

        if (step == 0) and (logger is not None) and (not printed_once):
            logger.log(f"[DBG][ep={epoch:03d}] disease_pos_weight={float(w_pos):.4f} mode={getattr(args,'disease_pos_weight_mode','none')}")
            logger.log(f"[DBG][ep={epoch:03d}] cf_detach_zc={cf_detach_zc} cf_detach_zb_for_disease={cf_detach_zb_for_disease} ent_yb_head_only={ent_yb_head_only}")
            if last_mask_stats:
                logger.log(
                    f"[DBG][ep={epoch:03d}] mask(step0): "
                    f"mc_mean={last_mask_stats.get('mc_mean',float('nan')):.4f} "
                    f"mb_mean={last_mask_stats.get('mb_mean',float('nan')):.4f} "
                    f"norm_overlap={last_mask_stats.get('norm_overlap',float('nan')):.4f} "
                    f"corr_cb={last_mask_stats.get('corr_cb',float('nan')):.4f} "
                    f"topkJ={last_mask_stats.get('topk_jaccard',float('nan')):.4f}"
                )
            printed_once = True

        # backward & step
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        # stats
        bs = y.size(0)
        total_examples += bs
        total_loss += float(loss.item()) * bs
        total_correct += int((logits_d.argmax(dim=1) == y).sum().item())

        if mask_stats:
            mask_sum["mc_mean"] += float(mask_stats["mc_mean"]) * bs
            mask_sum["mb_mean"] += float(mask_stats["mb_mean"]) * bs
            mask_sum["norm_overlap"] += float(mask_stats["norm_overlap"]) * bs
            mask_sum["corr_cb"] += float(mask_stats["corr_cb"]) * bs
            mask_count += bs

        if print_steps and (step % log_interval == 0 or step == len(data_loader) - 1):
            extra = ""
            if last_mask_stats:
                extra = (
                    f" | mc_mean={last_mask_stats.get('mc_mean',float('nan')):.3f}"
                    f" mb_mean={last_mask_stats.get('mb_mean',float('nan')):.3f}"
                    f" no={last_mask_stats.get('norm_overlap',float('nan')):.3f}"
                    f" corr={last_mask_stats.get('corr_cb',float('nan')):.3f}"
                )
            msg = (
                f"[TRAIN][ep={epoch:03d} step={step:04d}] "
                f"task_d={loss_task_d.item():.4f} task_total={loss_task.item():.4f} ent_yb={loss_ent_yb.item():.4f} | "
                f"pfs_d={loss_pfs_d.item():.4f} | "
                f"sup_c={loss_sup_c.item():.4f} sup_b={loss_sup_b.item():.4f} csuf={loss_causal_suf.item():.4f} "
                f"ortho={loss_ortho.item():.4f} | "
                f"mask={loss_mask.item():.4f} | total={loss.item():.4f}"
                + extra
            )
            (logger.log(msg) if logger is not None else print(msg))

    avg_loss = total_loss / max(1, total_examples)
    avg_acc = total_correct / max(1, total_examples)

    mask_avg: Dict[str, float] = {}
    if mask_count > 0:
        mask_avg = {k: mask_sum[k] / mask_count for k in mask_sum}

    return avg_loss, avg_acc, mask_avg


def train_one_epoch(model, loader, optimizer, device, args, epoch: int):
    logger = _get_logger(args)
    try:
        avg_loss, avg_acc, mask_avg = train_epoch(
            model=model,
            data_loader=loader,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
            args=args,
            logger=logger,
        )

        if logger is not None:
            logger.log(f"[TRAIN][ep={epoch:03d}] loss={avg_loss:.4f} acc={avg_acc*100:.2f}%")
            if mask_avg:
                logger.log(
                    f"[TRAIN][ep={epoch:03d}] mask_avg: "
                    f"mc_mean={mask_avg.get('mc_mean',float('nan')):.4f} "
                    f"mb_mean={mask_avg.get('mb_mean',float('nan')):.4f} "
                    f"norm_overlap={mask_avg.get('norm_overlap',float('nan')):.4f} "
                    f"corr_cb={mask_avg.get('corr_cb',float('nan')):.4f}"
                )

        return {
            "loss": float(avg_loss),
            "acc": float(avg_acc) * 100.0,
            "mask_avg": mask_avg,
        }
    finally:
        if logger is not None:
            logger.close()


# =============================================================
# Optimizer builder (supports masker_lr_mult)
# =============================================================
def build_optimizer(model: nn.Module, args) -> torch.optim.Optimizer:
    lr = float(getattr(args, "lr", 1e-4))
    wd = float(getattr(args, "weight_decay", 5e-4))
    masker_lr_mult = float(getattr(args, "masker_lr_mult", 1.0))

    base_params: List[nn.Parameter] = []
    masker_params: List[nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = name.lower()
        if ("mask_gen" in n) or ("masker" in n) or ("mask" in n and ("mask_gen_c" in n or "mask_gen_b" in n)):
            masker_params.append(p)
        else:
            base_params.append(p)

    param_groups = [
        {"params": base_params, "lr": lr, "weight_decay": wd},
    ]
    if len(masker_params) > 0:
        param_groups.append({"params": masker_params, "lr": lr * masker_lr_mult, "weight_decay": wd})

    opt = torch.optim.Adam(param_groups)
    return opt


# =============================================================
# Fit loop
# =============================================================
@dataclass
class FitResult:
    best_epoch: int
    best_val: Dict[str, float]
    test_at_best: Dict[str, float]
    history: Dict[str, list]


def _score_from_metrics(m: Dict[str, float], key: str) -> float:
    v = m.get(key, float("nan"))
    if v != v:
        return -1e18
    return float(v)


def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, extra: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"epoch": int(epoch), "model": model.state_dict(), "optimizer": optimizer.state_dict(), "extra": extra}, path)


def load_checkpoint(path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


def fit_one_fold(
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    optimizer: torch.optim.Optimizer,
    device,
    args,
    scheduler: Optional[Any] = None,
) -> FitResult:
    model.args = args

    out_dir = _get_out_dir(args)
    fold_id = _get_fold_id(args)
    logger = _get_logger(args)

    num_epochs = int(getattr(args, "num_epochs", 100))
    patience = int(getattr(args, "patience", 30))
    fixed_thr = float(getattr(args, "fixed_thr", 0.5))
    select_by = str(getattr(args, "select_by", "val_score"))

    ckpt_path = os.path.join(out_dir, f"ckpt_best_fold_{fold_id}.pt")
    summary_path = os.path.join(out_dir, f"summary_fold_{fold_id}.json")

    obj_sched = AdaptiveObjectiveScheduler(args)
    obj_sched.export_to_args(epoch=0)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1": [],
        "val_auc": [],
        "val_score": [],
        "val_score_bestthr": [],
        "w_pfs": [],
        "w_sup": [],
        "w_csuf": [],
        "stage2": [],
        "mask_mc_mean": [],
        "mask_mb_mean": [],
        "mask_norm_overlap": [],
        "mask_corr_cb": [],
        "masker_state": [],
    }

    best_epoch = -1
    best_val: Dict[str, float] = {}
    best_score = -1e18
    bad = 0

    t0 = time.time()
    try:
        if logger is not None:
            logger.log(f"================= FIT START (fold={fold_id}) =================")
            logger.log(
                f"[CFG] epochs={num_epochs} warmup={getattr(args,'warmup_epoch',0)} patience={patience} "
                f"fixed_thr={fixed_thr} select_by={select_by} | "
                f"eval_thr_mode={getattr(args,'eval_thr_mode','best_acc')} eval_thr_grid={getattr(args,'eval_thr_grid','coarse')} | "
                f"disease_pos_weight_mode={getattr(args,'disease_pos_weight_mode','none')} "
                f"freeze_b_after={getattr(args,'freeze_b_after',25)} | "
                f"two_stage={getattr(args,'two_stage',True)} stage1_ratio={getattr(args,'stage1_ratio',0.35)} stage2_lr_mult={getattr(args,'stage2_lr_mult',0.30)} | "
                f"adaptive_schedule={getattr(args,'adaptive_schedule',True)}"
            )

        for epoch in range(num_epochs):
            obj_sched.maybe_freeze_maskers(model, epoch)
            obj_sched.maybe_scale_lr(optimizer, epoch)
            obj_sched.export_to_args(epoch=epoch)

            tr = train_one_epoch(model, train_loader, optimizer, device, args, epoch)
            va = evaluate(model, val_loader, device, thr=fixed_thr)

            if scheduler is not None and hasattr(scheduler, "step"):
                try:
                    if "ReduceLROnPlateau" in scheduler.__class__.__name__:
                        scheduler.step(va.get(select_by, va.get("val_score", 0.0)))
                    else:
                        scheduler.step()
                except TypeError:
                    scheduler.step()

            # scheduler uses "bestthr score" to detect collapse/overlap
            vbest = float(va.get("val_score_bestthr", va.get("val_score", 0.0)))
            mavg = tr.get("mask_avg", {}) if isinstance(tr, dict) else {}
            obj_sched.update(epoch, vbest, mavg)
            obj_sched.export_to_args(epoch=epoch)

            history["train_loss"].append(float(tr["loss"]))
            history["train_acc"].append(float(tr["acc"]))
            history["val_loss"].append(float(va["loss"]))
            history["val_accuracy"].append(float(va["accuracy"]))
            history["val_f1"].append(float(va["f1_score"]))
            history["val_auc"].append(float(va["roc_auc"]))
            history["val_score"].append(float(va.get("val_score", float("nan"))))
            history["val_score_bestthr"].append(float(va.get("val_score_bestthr", float("nan"))))

            history["w_pfs"].append(float(getattr(args, "w_pfs_dyn", 0.0)))
            history["w_sup"].append(float(getattr(args, "w_sup_dyn", 0.0)))
            history["w_csuf"].append(float(getattr(args, "w_csuf_dyn", 0.0)))
            history["stage2"].append(bool(getattr(args, "stage2_active", False)))

            history["mask_mc_mean"].append(float(mavg.get("mc_mean", float("nan"))))
            history["mask_mb_mean"].append(float(mavg.get("mb_mean", float("nan"))))
            history["mask_norm_overlap"].append(float(mavg.get("norm_overlap", float("nan"))))
            history["mask_corr_cb"].append(float(mavg.get("corr_cb", float("nan"))))
            history["masker_state"].append(_maskers_trainable_state(model))

            score = _score_from_metrics(va, select_by)
            improved = score > best_score + 1e-12

            if logger is not None:
                logger.log(
                    f"[EPOCH {epoch:03d}/{num_epochs-1:03d}] "
                    f"train: loss={tr['loss']:.4f} acc={tr['acc']:.2f}% | "
                    f"val: loss={va['loss']:.4f} acc={va['accuracy']:.2f}% f1={va['f1_score']:.2f}% auc={va['roc_auc']:.2f}% "
                    f"val_score={va.get('val_score', float('nan')):.4f} val_score_bestthr={va.get('val_score_bestthr', float('nan')):.4f} | "
                    f"dyn(w_pfs={getattr(args,'w_pfs_dyn',0.0):.3f}, w_sup={getattr(args,'w_sup_dyn',0.0):.3f}, w_csuf={getattr(args,'w_csuf_dyn',0.0):.3f}, stage2={getattr(args,'stage2_active',False)}) | "
                    f"mask(mc={mavg.get('mc_mean',float('nan')):.3f}, mb={mavg.get('mb_mean',float('nan')):.3f}, "
                    f"no={mavg.get('norm_overlap',float('nan')):.3f}, corr={mavg.get('corr_cb',float('nan')):.3f}) | "
                    f"masker_state={_maskers_trainable_state(model)} | "
                    f"best({select_by})={'*' if improved else '-'} {best_score:.4f} -> {score:.4f}"
                )

            if _wandb_ready(args):
                _wandb_log({
                    "epoch": epoch,
                    "train/loss": tr.get("loss", float("nan")),
                    "train/acc": tr.get("acc", float("nan")),
                    "val/loss": va.get("loss", float("nan")),
                    "val/acc": va.get("accuracy", float("nan")),
                    "val/f1": va.get("f1_score", float("nan")),
                    "val/auc": va.get("roc_auc", float("nan")),
                    "val/val_score": va.get("val_score", float("nan")),
                    "val/val_score_bestthr": va.get("val_score_bestthr", float("nan")),
                    "sched/w_pfs": getattr(args, "w_pfs_dyn", 0.0),
                    "sched/w_sup": getattr(args, "w_sup_dyn", 0.0),
                    "sched/w_csuf": getattr(args, "w_csuf_dyn", 0.0),
                    "sched/stage2": 1.0 if bool(getattr(args, "stage2_active", False)) else 0.0,
                    "mask/mc_mean": mavg.get("mc_mean", float("nan")),
                    "mask/mb_mean": mavg.get("mb_mean", float("nan")),
                    "mask/norm_overlap": mavg.get("norm_overlap", float("nan")),
                    "mask/corr_cb": mavg.get("corr_cb", float("nan")),
                })

            if improved:
                best_score = score
                best_epoch = epoch
                best_val = dict(va)

                extra = {
                    "fold_id": fold_id,
                    "best_epoch": best_epoch,
                    "best_val": best_val,
                }
                save_checkpoint(ckpt_path, model, optimizer, epoch, extra)
                if logger is not None:
                    logger.log(f"[CKPT] Saved best -> {ckpt_path}")
                bad = 0
            else:
                bad += 1

            if bad >= patience:
                if logger is not None:
                    logger.log(f"[EARLY STOP] no improvement for {patience} epochs. best_epoch={best_epoch}")
                break

        if os.path.exists(ckpt_path):
            _ = load_checkpoint(ckpt_path, model, optimizer=None, map_location=device)
            if logger is not None:
                logger.log(f"[CKPT] Loaded best for testing: {ckpt_path}")

        te = test(model, test_loader, device, thr=fixed_thr)

        elapsed = time.time() - t0
        if logger is not None:
            logger.log("=============== FIT DONE ===============")
            logger.log(
                f"[BEST] epoch={best_epoch} | val_acc={best_val.get('accuracy', float('nan')):.2f}% "
                f"| val_f1={best_val.get('f1_score', float('nan')):.2f}% | val_auc={best_val.get('roc_auc', float('nan')):.2f}% "
                f"| val_score={best_val.get('val_score', float('nan')):.4f} | val_score_bestthr={best_val.get('val_score_bestthr', float('nan')):.4f}"
            )
            logger.log(
                f"[TEST@BEST] thr_used={te.get('thr_used', float('nan')):.3f} "
                f"| acc={te['accuracy']:.2f}% f1={te['f1_score']:.2f}% auc={te['roc_auc']:.2f}% "
                f"| SP_sex={te['SP_sex']:.2f}% EO_sex={te['EO_sex']:.2f}% SP_age={te['SP_age']:.2f}% EO_age={te['EO_age']:.2f}%"
            )
            logger.log(f"[TIME] {elapsed/60.0:.2f} min")

        summary = {
            "fold_id": fold_id,
            "best_epoch": int(best_epoch),
            "select_by": select_by,
            "fixed_thr": fixed_thr,
            "best_val": best_val,
            "test_at_best": te,
            "history": history,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        if logger is not None:
            logger.log(f"[OK] Saved: {summary_path}")

        return FitResult(best_epoch=best_epoch, best_val=best_val, test_at_best=te, history=history)

    finally:
        if logger is not None:
            logger.close()