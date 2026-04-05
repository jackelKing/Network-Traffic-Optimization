"""
Graph Neural Network encoder for traffic routing.

Replaces (or augments) the attention-based TrafficNet with a
GNN that explicitly encodes the network topology as a graph.

Requirements: pip install torch-geometric
If torch-geometric is not installed, the encoder gracefully falls
back to a plain MLP so training is never broken.
"""

import torch
import torch.nn as nn
import numpy as np

try:
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("⚠️  torch-geometric not installed — GNN encoder will use MLP fallback")


class GNNTrafficEncoder(nn.Module):
    """
    Two-layer GAT (Graph Attention Network) over the router graph.

    node_feat_dim : features per node (OBS_PER_NODE from obs layout)
    hidden_dim    : internal GAT channels
    out_dim       : final embedding per node
    num_heads     : GAT attention heads

    Forward input:
        node_feats  : (B*N, node_feat_dim)
        edge_index  : (2, E) — topology edges (same for all batch items)
        batch_vec   : (B*N,) — maps each node to its graph in the batch
    Forward output:
        graph_emb   : (B, out_dim) — one vector per graph (mean pool)
    """

    def __init__(self, node_feat_dim=8, hidden_dim=64, out_dim=256, num_heads=4):
        super().__init__()
        self.use_gnn = HAS_PYG

        if HAS_PYG:
            self.gat1  = GATConv(node_feat_dim, hidden_dim,
                                  heads=num_heads, concat=True,  dropout=0.05)
            self.gat2  = GATConv(hidden_dim * num_heads, out_dim,
                                  heads=1,         concat=False, dropout=0.05)
            self.norm1 = nn.LayerNorm(hidden_dim * num_heads)
            self.norm2 = nn.LayerNorm(out_dim)
        else:
            # Fallback MLP
            self.mlp = nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),       nn.ReLU(),
            )

    def forward(self, node_feats, edge_index=None, batch_vec=None):
        if self.use_gnn and edge_index is not None:
            import torch.nn.functional as F
            x = F.elu(self.norm1(self.gat1(node_feats, edge_index)))
            x = F.elu(self.norm2(self.gat2(x,           edge_index)))
            if batch_vec is not None:
                return global_mean_pool(x, batch_vec)   # (B, out_dim)
            return x.mean(dim=0, keepdim=True)
        else:
            x = self.mlp(node_feats)
            if batch_vec is not None:
                B = int(batch_vec.max().item()) + 1
                out = torch.stack([x[batch_vec == b].mean(0) for b in range(B)])
                return out
            return x.mean(dim=0, keepdim=True)


def build_edge_index_for_topo(topo_type, num_nodes):
    """
    Build a static edge_index tensor for the current topology.
    Used to pass graph structure into the GNN encoder.
    """
    import torch, math
    edges = []

    if topo_type == "linear":
        for i in range(num_nodes - 1):
            edges += [[i, i+1], [i+1, i]]

    elif topo_type == "grid":
        side = int(math.ceil(math.sqrt(num_nodes)))
        for i in range(num_nodes):
            r, c = divmod(i, side)
            if c + 1 < side and i + 1 < num_nodes:
                edges += [[i, i+1], [i+1, i]]
            if r + 1 < side and i + side < num_nodes:
                edges += [[i, i+side], [i+side, i]]

    else:   # random / fallback: nearest neighbours (ring)
        for i in range(num_nodes):
            j = (i + 1) % num_nodes
            edges += [[i, j], [j, i]]

    if not edges:
        edges = [[0, 0]]  # self-loop guard for single-node edge case

    ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return ei
