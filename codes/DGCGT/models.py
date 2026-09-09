"""DGCGT and MarsGT-style heterogeneous graph transformer models."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph import HeteroGraph, NODE_TYPES

def _key(relation):
    return "__".join(relation)

class HeteroHGTLayer(nn.Module):
    """Apply relation aware multi head attention to graph nodes."""

    def __init__(self, hidden_dim: int, heads: int, dropout: float, relations: tuple):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.q = nn.ModuleDict({node: nn.Linear(hidden_dim, hidden_dim) for node in NODE_TYPES})
        self.k = nn.ModuleDict({_key(rel): nn.Linear(hidden_dim, hidden_dim) for rel in relations})
        self.v = nn.ModuleDict({_key(rel): nn.Linear(hidden_dim, hidden_dim) for rel in relations})
        self.out = nn.ModuleDict({node: nn.Linear(hidden_dim, hidden_dim) for node in NODE_TYPES})
        self.norm = nn.ModuleDict({node: nn.LayerNorm(hidden_dim) for node in NODE_TYPES})
        self.relation_scale = nn.ParameterDict(
            {_key(rel): nn.Parameter(torch.ones(heads)) for rel in relations}
        )
        self.dropout = dropout

    def forward(self, h: Dict[str, torch.Tensor], slots: Dict[tuple, torch.Tensor]):
        updates = {node: torch.zeros_like(h[node]) for node in h}
        for relation, neighbor_idx in slots.items():
            src_type, _, dst_type = relation
            n_dst, k = neighbor_idx.shape
            if n_dst == 0 or k == 0:
                continue
            q = self.q[dst_type](h[dst_type]).view(n_dst, self.heads, self.head_dim)
            key = _key(relation)
            ks = self.k[key](h[src_type]).view(-1, self.heads, self.head_dim)
            vs = self.v[key](h[src_type]).view(-1, self.heads, self.head_dim)
            flat = neighbor_idx.reshape(-1)
            knb = ks.index_select(0, flat).view(n_dst, k, self.heads, self.head_dim)
            vnb = vs.index_select(0, flat).view(n_dst, k, self.heads, self.head_dim)
            scores = (q[:, None, :, :] * knb).sum(-1) / math.sqrt(self.head_dim)
            scores = scores * self.relation_scale[key]
            weights = torch.softmax(scores, dim=1)
            weights = F.dropout(weights, p=self.dropout, training=self.training)
            msg = (weights.unsqueeze(-1) * vnb).sum(dim=1).view(n_dst, -1)
            updates[dst_type] = updates[dst_type] + msg
        return {
            node: self.norm[node](
                h[node] + F.dropout(F.gelu(self.out[node](updates[node])), p=self.dropout, training=self.training)
            )
            for node in h
        }

class HeteroHGTEncoder(nn.Module):
    """Encode heterogeneous graph node features with HGT layers."""

    def __init__(self, input_dims: Dict[str, int], hidden_dim: int = 64, heads: int = 4, layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.input = nn.ModuleDict(
            {node: nn.Linear(input_dims[node], hidden_dim) for node in input_dims}
        )
        relations = tuple(
            rel for rel in (
                ("cell", "knn", "cell"),
                ("gene", "expressed_by", "cell"),
                ("peak", "accessible_by", "cell"),
                ("cell", "expresses", "gene"),
                ("peak", "regulated_by", "gene"),
                ("cell", "accessible", "peak"),
                ("gene", "regulates", "peak"),
            )
            if rel[2] in input_dims and rel[0] in input_dims
        )
        self.layers = nn.ModuleList(
            [HeteroHGTLayer(hidden_dim, heads, dropout, relations) for _ in range(layers)]
        )
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, graph: HeteroGraph, cell_override=None):
        x = dict(graph.x)
        if cell_override is not None:
            x["cell"] = cell_override
        h = {node: F.gelu(self.input[node](x[node])) for node in x}
        for layer in self.layers:
            h = layer(h, graph.slots)
        return h

class MarsGTLite(nn.Module):
    """Provide the memory optimized MarsGT style comparison model."""

    def __init__(self, input_dims, hidden_dim=64, heads=4, layers=1, dropout=0.3):
        super().__init__()
        self.encoder = HeteroHGTEncoder(input_dims, hidden_dim, heads, layers, dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, graph: HeteroGraph, enhanced_cell=None):
        return self.classifier(self.encoder(graph)["cell"]).squeeze(-1)

class DGCGTMultiOmics(nn.Module):
    """Combine raw and SISD DCC enhanced branches for rare cell detection."""

    def __init__(self, input_dims, enhanced_dim=None, hidden_dim=64, heads=4, layers=1, dropout=0.3, use_ff=True, use_gate=False):
        super().__init__()
        self.use_ff = use_ff
        self.use_gate = use_gate
        if enhanced_dim is None:
            enhanced_dim = input_dims["cell"]
        self.encoder_raw = HeteroHGTEncoder(input_dims, hidden_dim, heads, layers, dropout)
        enhanced_dims = dict(input_dims)
        enhanced_dims["cell"] = enhanced_dim
        self.encoder_enhanced = HeteroHGTEncoder(enhanced_dims, hidden_dim, heads, layers, dropout)

        self.shortcut = nn.Sequential(
            nn.LayerNorm(input_dims["cell"]),
            nn.Linear(input_dims["cell"], hidden_dim),
            nn.GELU(),
        )
        if use_ff and use_gate:
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 3),
            )
            classifier_dim = hidden_dim
        else:
            classifier_dim = hidden_dim * 3 if use_ff else hidden_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_dim),
            nn.Linear(classifier_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, graph: HeteroGraph, enhanced_cell=None):
        if enhanced_cell is None:
            enhanced_cell = graph.x["cell"]
        raw_h = self.encoder_raw(graph)["cell"]
        enhanced_h = self.encoder_enhanced(graph, cell_override=enhanced_cell)["cell"]

        enhanced_h = enhanced_h + self.shortcut(graph.x["cell"])
        if self.use_ff:
            shortcut_h = self.shortcut(graph.x["cell"])
            if self.use_gate:
                gate = torch.softmax(self.gate(torch.cat([raw_h, enhanced_h, shortcut_h], dim=-1)), dim=-1)
                fused = (
                    gate[:, 0:1] * raw_h
                    + gate[:, 1:2] * enhanced_h
                    + gate[:, 2:3] * shortcut_h
                )
            else:
                fused = torch.cat([raw_h, enhanced_h, shortcut_h], dim=-1)
        else:
            fused = enhanced_h
        return self.classifier(fused).squeeze(-1)

