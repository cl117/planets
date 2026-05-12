"""
Sequence-aware GNN for synthetic biology circuits.

Input:
  - part DNA sequences (nodes)
  - interactions (directed edges with relation types)

Encoder options:
  - DNABERT2 (Hugging Face)
  - HyenaDNA (Hugging Face)

Outputs (multi-task):
  - node-level prediction
  - edge-level prediction
  - graph-level prediction
  - optional product-level regression/classification
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError as e:
    raise ImportError("Please install transformers: pip install transformers") from e

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GATv2Conv, global_mean_pool
except ImportError as e:
    raise ImportError(
        "Please install torch-geometric and its deps: https://pytorch-geometric.readthedocs.io/"
    ) from e


RELATION2ID = {
    "activate": 0,
    "repress": 1,
    "physical": 2,
    "unknown": 3,
}


@dataclass
class ModelConfig:
    hf_model_name: str = "zhihan1996/DNABERT-2-117M"
    gnn_hidden_dim: int = 256
    gnn_layers: int = 3
    gnn_heads: int = 4
    dropout: float = 0.2
    relation_vocab_size: int = len(RELATION2ID)

    node_out_dim: int = 8
    edge_out_dim: int = 4
    graph_out_dim: int = 3  # e.g. bistable / oscillatory / dynamic
    product_out_dim: int = 1


class DNASequenceEncoder(nn.Module):
    """Encode DNA sequence to dense embedding with a pretrained DNA foundation model."""

    def __init__(self, model_name: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.hidden_size = self.backbone.config.hidden_size

    def forward(self, sequences: List[str], device: torch.device) -> torch.Tensor:
        # Batch tokenize raw DNA strings
        toks = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        toks = {k: v.to(device) for k, v in toks.items()}

        out = self.backbone(**toks)
        # Use [CLS] embedding if available, else mean pooling
        if hasattr(out, "last_hidden_state"):
            h = out.last_hidden_state
            if h.size(1) > 0:
                x = h[:, 0, :]
            else:
                x = h.mean(dim=1)
        else:
            x = out[0][:, 0, :]
        return x


class SeqAwareGNN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.seq_encoder = DNASequenceEncoder(cfg.hf_model_name)

        self.relation_emb = nn.Embedding(cfg.relation_vocab_size, cfg.gnn_hidden_dim)
        self.node_proj = nn.Linear(self.seq_encoder.hidden_size, cfg.gnn_hidden_dim)

        self.convs = nn.ModuleList()
        self.edge_mlps = nn.ModuleList()
        for _ in range(cfg.gnn_layers):
            conv = GATv2Conv(
                in_channels=cfg.gnn_hidden_dim,
                out_channels=cfg.gnn_hidden_dim // cfg.gnn_heads,
                heads=cfg.gnn_heads,
                dropout=cfg.dropout,
                edge_dim=cfg.gnn_hidden_dim,
            )
            self.convs.append(conv)
            self.edge_mlps.append(
                nn.Sequential(
                    nn.Linear(cfg.gnn_hidden_dim, cfg.gnn_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(cfg.dropout),
                )
            )

        self.node_head = nn.Linear(cfg.gnn_hidden_dim, cfg.node_out_dim)
        self.edge_head = nn.Sequential(
            nn.Linear(cfg.gnn_hidden_dim * 2 + cfg.gnn_hidden_dim, cfg.gnn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.gnn_hidden_dim, cfg.edge_out_dim),
        )
        self.graph_head = nn.Linear(cfg.gnn_hidden_dim, cfg.graph_out_dim)
        self.product_head = nn.Linear(cfg.gnn_hidden_dim, cfg.product_out_dim)

    def _message_passing(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_rel_ids: torch.Tensor,
    ) -> torch.Tensor:
        edge_attr = self.relation_emb(edge_rel_ids)
        for conv, edge_mlp in zip(self.convs, self.edge_mlps):
            e = edge_mlp(edge_attr)
            x = conv(x, edge_index, e)
            x = F.elu(x)
            x = F.dropout(x, p=self.cfg.dropout, training=self.training)
        return x

    def forward(
        self,
        sequences: List[str],
        edge_index: torch.Tensor,
        edge_rel_ids: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        device = edge_index.device
        seq_x = self.seq_encoder(sequences, device=device)
        x = self.node_proj(seq_x)

        h = self._message_passing(x, edge_index, edge_rel_ids)

        node_logits = self.node_head(h)

        src, dst = edge_index[0], edge_index[1]
        edge_rel = self.relation_emb(edge_rel_ids)
        edge_feat = torch.cat([h[src], h[dst], edge_rel], dim=-1)
        edge_logits = self.edge_head(edge_feat)

        if batch_index is None:
            batch_index = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        graph_emb = global_mean_pool(h, batch_index)
        graph_logits = self.graph_head(graph_emb)
        product_pred = self.product_head(graph_emb)

        return {
            "node_logits": node_logits,
            "edge_logits": edge_logits,
            "graph_logits": graph_logits,
            "product_pred": product_pred,
            "node_embeddings": h,
            "graph_embeddings": graph_emb,
        }


def build_pyg_data(
    part_sequences: List[str],
    interactions: List[Dict],
    graph_label: Optional[int] = None,
    node_labels: Optional[List[int]] = None,
    edge_labels: Optional[List[int]] = None,
    product_target: Optional[float] = None,
) -> Data:
    """
    Build PyG Data from symbolic interactions.

    interactions example:
    [
      {"src": 0, "dst": 2, "type": "repress"},
      {"src": 1, "dst": 2, "type": "activate"},
    ]
    """
    edge_tuples = [(it["src"], it["dst"]) for it in interactions]
    edge_index = torch.tensor(edge_tuples, dtype=torch.long).t().contiguous()

    edge_rel_ids = torch.tensor(
        [RELATION2ID.get(it.get("type", "unknown"), RELATION2ID["unknown"]) for it in interactions],
        dtype=torch.long,
    )

    data = Data(edge_index=edge_index, edge_rel_ids=edge_rel_ids)
    data.part_sequences = part_sequences

    if graph_label is not None:
        data.y_graph = torch.tensor([graph_label], dtype=torch.long)
    if node_labels is not None:
        data.y_node = torch.tensor(node_labels, dtype=torch.long)
    if edge_labels is not None:
        data.y_edge = torch.tensor(edge_labels, dtype=torch.long)
    if product_target is not None:
        data.y_product = torch.tensor([[product_target]], dtype=torch.float)

    return data


def compute_multitask_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Batch,
    w_node: float = 1.0,
    w_edge: float = 1.0,
    w_graph: float = 1.0,
    w_product: float = 1.0,
) -> torch.Tensor:
    loss = 0.0
    if hasattr(batch, "y_node"):
        loss = loss + w_node * F.cross_entropy(outputs["node_logits"], batch.y_node)
    if hasattr(batch, "y_edge"):
        loss = loss + w_edge * F.cross_entropy(outputs["edge_logits"], batch.y_edge)
    if hasattr(batch, "y_graph"):
        loss = loss + w_graph * F.cross_entropy(outputs["graph_logits"], batch.y_graph)
    if hasattr(batch, "y_product"):
        loss = loss + w_product * F.mse_loss(outputs["product_pred"], batch.y_product)
    return loss


def demo_train_step(device: str = "cpu") -> None:
    """Minimal runnable example."""
    cfg = ModelConfig()
    model = SeqAwareGNN(cfg).to(device)

    seqs = [
        "TTGACATATAAT",  # promoter
        "AGGAGG",        # RBS
        "ATGGCCATTGTAATGGG",  # CDS
        "TTATTT",        # terminator-like short example
    ]
    interactions = [
        {"src": 0, "dst": 2, "type": "activate"},
        {"src": 2, "dst": 0, "type": "repress"},
        {"src": 1, "dst": 2, "type": "physical"},
    ]

    data = build_pyg_data(
        part_sequences=seqs,
        interactions=interactions,
        graph_label=1,
        node_labels=[0, 1, 2, 3],
        edge_labels=[0, 1, 2],
        product_target=0.73,
    )
    batch = Batch.from_data_list([data]).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    model.train()
    outputs = model(
        sequences=batch.part_sequences,
        edge_index=batch.edge_index,
        edge_rel_ids=batch.edge_rel_ids,
        batch_index=batch.batch,
    )
    loss = compute_multitask_loss(outputs, batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print({k: tuple(v.shape) for k, v in outputs.items() if isinstance(v, torch.Tensor)})
    print(f"train_loss={loss.item():.4f}")


if __name__ == "__main__":
    demo_train_step(device="cpu")
