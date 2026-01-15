# nets/gcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GCNEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList()

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(num_layers):
            self.convs.append(GCNConv(dims[i], dims[i + 1]))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None,
        batch: torch.Tensor = None,
    ) -> torch.Tensor:
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        # edge_weight can be None or (E,)
        for layer_idx, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_weight)  # ✅ use edge_weight
            if layer_idx != len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)

        g_emb = global_mean_pool(x, batch)  # (B, out_dim)
        return g_emb
