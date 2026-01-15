# nets/disbg.py
import torch
import torch.nn as nn

from nets.mask_generator import EdgeMaskGenerator
from nets.gcn import GCNEncoder


class DisBGModel(nn.Module):
    """
    DisBG (Edge-mask decomposition)

    forward returns:
      logits_d, logits_s, logits_a, logits_d_b, z_c, z_b, u_c, u_b

    Important knobs (read from args):
      - cf_detach_zc (bool, default False):
          If True, disease counterfactual path does NOT backprop to encoder_c/mask_c.
      - cf_detach_zb_for_disease (bool, default True):
          If True, disease counterfactual path does NOT backprop to encoder_b/mask_b.
      - ent_yb_head_only (bool, default True):
          If True, entropy regularization on logits_d_b updates only classifier_disease_bias,
          not encoder_b/mask_b (and not encoder_c either).
    """

    def __init__(self, args):
        super().__init__()

        # ----- misc -----
        self.args = args
        self.dropout = float(getattr(args, "dropout", 0.5))
        # keep backward-compat: run_5fold uses --mask_temperature; some older code uses --mask_tau
        self.mask_tau = float(getattr(args, "mask_temperature", getattr(args, "mask_tau", 1.0)))
        self.bias_scale = float(getattr(args, "bias_scale", 0.5))

        # ----- gradient routing knobs (critical) -----
        self.cf_detach_zc = bool(getattr(args, "cf_detach_zc", False))
        self.cf_detach_zb_for_disease = bool(getattr(args, "cf_detach_zb_for_disease", True))
        self.ent_yb_head_only = bool(getattr(args, "ent_yb_head_only", True))

        # ---- optional: node projection for mask generator ----
        self.use_mask_node_proj = bool(getattr(args, "use_mask_node_proj", False))
        mask_node_proj_dim = int(getattr(args, "mask_node_proj_dim", 0))
        num_feats = int(getattr(args, "num_feats", 116))

        if self.use_mask_node_proj:
            if mask_node_proj_dim <= 0:
                raise ValueError("use_mask_node_proj=True requires mask_node_proj_dim > 0")
            self.mask_node_proj = nn.Linear(num_feats, mask_node_proj_dim)
            mask_in_dim = mask_node_proj_dim
        else:
            self.mask_node_proj = None
            mask_in_dim = num_feats

        # ---- encoders: split into causal & bias ----
        self.encoder_c = GCNEncoder(
            int(args.num_feats),
            int(args.gnn_hidden_dim),
            int(args.gnn_out_dim),
            int(args.num_gnn_layers),
            float(self.dropout),
        )
        self.encoder_b = GCNEncoder(
            int(args.num_feats),
            int(args.gnn_hidden_dim),
            int(args.gnn_out_dim),
            int(args.num_gnn_layers),
            float(self.dropout),
        )

        # ---- mask generators ----
        mask_hidden = int(getattr(args, "mask_hidden_dim", 64))
        self.mask_gen_c = EdgeMaskGenerator(
            in_dim=mask_in_dim,
            hidden_dim=mask_hidden,
            temperature=float(self.mask_tau),
        )
        self.mask_gen_b = EdgeMaskGenerator(
            in_dim=mask_in_dim,
            hidden_dim=mask_hidden,
            temperature=float(self.mask_tau),
        )

        # ---- heads ----
        in_dim = int(args.gnn_out_dim) * 2
        self.classifier_disease = nn.Linear(in_dim, int(args.num_classes))
        self.classifier_sex = nn.Linear(in_dim, int(args.num_sex_classes))
        self.classifier_age = nn.Linear(in_dim, int(args.num_age_classes))

        # disease head from bias branch (for entropy regularization)
        self.classifier_disease_bias = nn.Linear(in_dim, int(args.num_classes))

    # ----------------------------
    # helpers: apply detaches safely
    # ----------------------------
    def _maybe_detach(self, t: torch.Tensor, flag: bool) -> torch.Tensor:
        return t.detach() if flag else t

    def classify_counterfactual(self, z_c: torch.Tensor, z_b_perm: torch.Tensor):
        """
        z_c: (B, D) from encoder_c
        z_b_perm: (B, D) permuted bias embedding (counterfactual)

        Returns:
          logits_d_cf, logits_s_cf, logits_a_cf
        """
        # For disease CF: decide who gets gradients
        zc_for_d = self._maybe_detach(z_c, self.cf_detach_zc)  # default False => allow gradients
        zb_for_d = self._maybe_detach(z_b_perm, self.cf_detach_zb_for_disease)  # default True => block bias grads

        z_for_c = torch.cat([zc_for_d, self.bias_scale * zb_for_d], dim=-1)

        # For sensitive attribute heads we keep the usual: do NOT let them shape z_c
        z_for_b = torch.cat([z_c.detach(), z_b_perm], dim=-1)

        return (
            self.classifier_disease(z_for_c),
            self.classifier_sex(z_for_b),
            self.classifier_age(z_for_b),
        )

    def forward(self, x, edge_index, batch=None):
        """
        x: (N, F)
        edge_index: (2, E)
        batch: (N,) or None
        """

        # mask input
        if self.use_mask_node_proj and self.mask_node_proj is not None:
            x_mask = self.mask_node_proj(x)
        else:
            x_mask = x

        # edge masks
        m_c = self.mask_gen_c(x_mask, edge_index)  # (E,)
        m_b = self.mask_gen_b(x_mask, edge_index)  # (E,)

        # embeddings
        z_c = self.encoder_c(x, edge_index, edge_weight=m_c, batch=batch)  # (B, D)
        z_b = self.encoder_b(x, edge_index, edge_weight=m_b, batch=batch)  # (B, D)

        u_c, u_b = z_c, z_b

        # disease uses causal; block bias gradients by default to avoid mask_b collapse
        z_for_c = torch.cat([z_c, self.bias_scale * z_b.detach()], dim=-1)

        # sensitive heads use bias; block z_c grads
        z_for_b = torch.cat([z_c.detach(), z_b], dim=-1)

        logits_d = self.classifier_disease(z_for_c)      # disease from causal
        logits_s = self.classifier_sex(z_for_b)          # sex from bias
        logits_a = self.classifier_age(z_for_b)          # age from bias

        # entropy(disease|bias): head-only option
        if self.ent_yb_head_only:
            # Only update classifier_disease_bias, not encoder_b/mask_b
            logits_d_b = self.classifier_disease_bias(z_for_b.detach())
        else:
            logits_d_b = self.classifier_disease_bias(z_for_b)

        return logits_d, logits_s, logits_a, logits_d_b, z_c, z_b, u_c, u_b
