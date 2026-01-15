# nets/mask_generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeMaskGenerator(nn.Module):
    """
    Edge mask generator:
      x: (N, F) node features
      edge_index: (2, E)
    returns:
      m: (E,) in (0,1)
    """
    def __init__(self, in_dim: int, hidden_dim: int, temperature: float = 1.0):
        super().__init__()
        self.temperature = float(temperature)

        self.mlp = nn.Sequential(
            nn.Linear(2 * in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.last_mask = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index  # (E,), (E,)
        x_i = x[row]           # (E, F)
        x_j = x[col]           # (E, F)
        edge_feat = torch.cat([x_i, x_j], dim=-1)  # (E, 2F)

        logits = self.mlp(edge_feat).squeeze(-1)   # (E,)
        # temperature-aware sigmoid
        m = torch.sigmoid(logits / max(self.temperature, 1e-6)).view(-1)  # (E,)

        self.last_mask = m
        return m
