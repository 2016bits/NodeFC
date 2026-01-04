import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from torch_geometric.nn.pool import global_add_pool
from torch_geometric.utils import softmax

class ClaimHeteroGNN(nn.Module):
    """
    S/N hetero graph classifier.
    - sentence/entity node init: projected encoder embeddings
    - message passing: per-relation SAGEConv via HeteroConv (easy & strong baseline)
    - readout: claim-conditioned attentive pooling over sentence nodes
    - classification: binary (SUPPORTS/REFUTES)
    """
    def __init__(self, in_dim: int, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.dropout = dropout

        self.proj_s = Linear(in_dim, hidden_dim)
        self.proj_n = Linear(in_dim, hidden_dim)
        self.proj_c = Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HeteroConv(
                    {
                        ("sentence", "ss", "sentence"): SAGEConv((-1, -1), hidden_dim),
                        ("sentence", "sn", "entity"):   SAGEConv((-1, -1), hidden_dim),
                        ("entity",   "ns", "sentence"): SAGEConv((-1, -1), hidden_dim),
                    },
                    aggr="sum",
                )
            )

        # attention conditioned on claim
        self.att_c = Linear(hidden_dim, hidden_dim, bias=False)
        self.att_s = Linear(hidden_dim, hidden_dim, bias=False)

        # classifier (binary)
        fuse_dim = hidden_dim * 4  # [c, g, |c-g|, c*g]
        self.mlp = nn.Sequential(
            nn.Linear(fuse_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def attentive_pool_sentences(self, h_s: torch.Tensor, batch_s: torch.Tensor, c_h: torch.Tensor) -> torch.Tensor:
        """
        h_s: [Ns, H]
        batch_s: [Ns] graph ids
        c_h: [B, H] claim representation
        Return g: [B, H]
        """
        c_proj = self.att_c(c_h)         # [B,H]
        s_proj = self.att_s(h_s)         # [Ns,H]
        scores = (s_proj * c_proj[batch_s]).sum(dim=-1)  # [Ns]
        alpha = softmax(scores, batch_s)                 # [Ns], group softmax by batch
        g = global_add_pool(alpha.unsqueeze(-1) * h_s, batch_s)  # [B,H]
        return g

    def forward(self, data):
        # node init
        x_s = F.relu(self.proj_s(data["sentence"].x))
        x_n = F.relu(self.proj_n(data["entity"].x)) if data["entity"].num_nodes > 0 else data["entity"].x
        c_h = F.relu(self.proj_c(data.claim_emb))   # [B, H]

        x_dict = {"sentence": x_s, "entity": x_n}

        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
            x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x_dict.items()}

        h_s = x_dict["sentence"]
        batch_s = data["sentence"].batch  # created by PyG Batch

        g = self.attentive_pool_sentences(h_s, batch_s, c_h)

        z = torch.cat([c_h, g, torch.abs(c_h - g), c_h * g], dim=-1)
        logits = self.mlp(z)
        return logits
