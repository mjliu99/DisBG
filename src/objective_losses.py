# loss_objective.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        features = F.normalize(features, dim=1)
        labels = labels.contiguous().view(-1, 1)
        B = features.shape[0]

        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.ones_like(mask) - torch.eye(B, device=device)
        mask = mask * logits_mask

        logits = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mask_sum = mask.sum(1)
        valid = mask_sum > 0
        mask_sum = mask_sum.clamp(min=1.0)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        loss = -mean_log_prob_pos[valid].mean() if valid.any() else features.new_tensor(0.0)
        return loss


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Return mean entropy H(p) where p = softmax(logits).
    """
    p = F.softmax(logits, dim=-1)
    ent = -(p * torch.log(p + 1e-12)).sum(dim=-1)
    return ent.mean()


class DisBGDecompLoss(nn.Module):
    """
    Decomposition supervision:
      - causal branch: predict disease y
      - bias branch: predict sensitive sex/age
      - bias branch: remove y info via MAX entropy on disease logits (uniform)
      - optional: supervised contrastive on z_c (y) and z_b (sex/age)
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.ce = nn.CrossEntropyLoss()

        # optional contrastive
        self.use_supcon = bool(getattr(args, "use_supcon", False))
        tau = float(getattr(args, "supcon_tau", 0.07))
        self.supcon = SupConLoss(temperature=tau)

        # weights
        self.lam_y   = float(getattr(args, "lam_y", 1.0))
        self.lam_sex = float(getattr(args, "lam_sex", 1.0))
        self.lam_age = float(getattr(args, "lam_age", 1.0))

        # bias->y entropy (maximize entropy => subtract in total loss)
        self.lam_ent_yb = float(getattr(args, "lam_ent_yb", 0.1))

        # supcon weights
        self.lam_supcon_y   = float(getattr(args, "lam_supcon_y", 0.0))
        self.lam_supcon_sex = float(getattr(args, "lam_supcon_sex", 0.0))
        self.lam_supcon_age = float(getattr(args, "lam_supcon_age", 0.0))

    def forward(
        self,
        logits_d, logits_s, logits_a, logits_d_b,
        z_c, z_b,
        y, sex, age
    ):
        # CE losses
        loss_y   = self.ce(logits_d, y)
        loss_sex = self.ce(logits_s, sex)
        loss_age = self.ce(logits_a, age)

        # bias branch should contain *as little disease info as possible*
        ent_yb = entropy_from_logits(logits_d_b)  # maximize it

        loss = (
            self.lam_y * loss_y
            + self.lam_sex * loss_sex
            + self.lam_age * loss_age
            - self.lam_ent_yb * ent_yb
        )

        # optional supcon
        loss_supcon_y = z_c.new_tensor(0.0)
        loss_supcon_s = z_c.new_tensor(0.0)
        loss_supcon_a = z_c.new_tensor(0.0)
        if self.use_supcon:
            if self.lam_supcon_y > 0:
                loss_supcon_y = self.supcon(z_c, y)
                loss = loss + self.lam_supcon_y * loss_supcon_y
            if self.lam_supcon_sex > 0:
                loss_supcon_s = self.supcon(z_b, sex)
                loss = loss + self.lam_supcon_sex * loss_supcon_s
            if self.lam_supcon_age > 0:
                loss_supcon_a = self.supcon(z_b, age)
                loss = loss + self.lam_supcon_age * loss_supcon_a

        logs = {
            "loss_total": float(loss.detach().cpu()),
            "loss_y": float(loss_y.detach().cpu()),
            "loss_sex": float(loss_sex.detach().cpu()),
            "loss_age": float(loss_age.detach().cpu()),
            "ent_y_from_bias": float(ent_yb.detach().cpu()),
            "loss_supcon_y": float(loss_supcon_y.detach().cpu()),
            "loss_supcon_sex": float(loss_supcon_s.detach().cpu()),
            "loss_supcon_age": float(loss_supcon_a.detach().cpu()),
        }
        return loss, logs
