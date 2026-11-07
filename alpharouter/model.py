from __future__ import annotations

"""Transformer router used by the dynamic AlphaRouter experiment."""

import torch
from torch import nn


class DynamicExpertRouter(nn.Module):
    """Map a causal market history to a hard decision over shared experts.

    The router predicts four expert logits. The deployment gate then blends
    the selected expert with the reference under Eq. (18).
    """

    def __init__(
        self,
        input_features: int,
        experts: int = 4,
        window: int = 30,
        hidden: int = 128,
        heads: int = 8,
        layers: int = 6,
        feedforward: int = 256,
        dropout: float = 0.10,
        evidence_features: int = 4,
    ) -> None:
        super().__init__()
        self.experts = int(experts)
        self.evidence_features = int(evidence_features)
        self.input = nn.Linear(input_features, hidden)
        self.position = nn.Parameter(torch.zeros(1, window, hidden))
        encoder = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(encoder, num_layers=layers)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, experts),
        )
        # The same small scoring network is applied to every expert.  It turns
        # observable short, medium and long trailing portfolio evidence into an
        # additive routing score.  Sharing the scorer avoids assigning a fixed
        # semantic meaning to an expert index and leaves the Transformer to
        # learn state-dependent corrections.
        evidence_hidden = max(16, hidden // 4)
        self.evidence_head = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, evidence_hidden),
            nn.GELU(),
            nn.Linear(evidence_hidden, 1),
        )
        self.evidence_scale = nn.Parameter(torch.tensor(0.0))
        nn.init.normal_(self.position, std=0.02)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        nn.init.zeros_(self.evidence_head[-1].weight)
        nn.init.zeros_(self.evidence_head[-1].bias)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if state.ndim != 3:
            raise ValueError("state must have shape [batch, time, features]")
        if state.shape[1] > self.position.shape[1]:
            raise ValueError("state window exceeds positional encoding")
        encoded = self.input(state) + self.position[:, : state.shape[1]]
        hidden = self.temporal(encoded)
        # The last causal token represents the information set immediately
        # before allocation.  Mean pooling is added to retain slow conditions.
        pooled = self.norm(0.5 * hidden[:, -1] + 0.5 * hidden.mean(dim=1))
        logits = self.head(pooled)
        if self.evidence_features:
            if state.shape[2] < self.evidence_features:
                raise ValueError(
                    "state does not contain the requested expert evidence channels"
                )
            evidence = state[:, :, -self.evidence_features :]
            short = evidence[:, -min(5, evidence.shape[1]) :].mean(dim=1)
            medium = evidence[:, -min(15, evidence.shape[1]) :].mean(dim=1)
            long = evidence.mean(dim=1)
            volatility = evidence.std(dim=1, unbiased=False)
            downside = torch.sqrt(torch.relu(-evidence).square().mean(dim=1) + 1e-8)
            statistics = torch.stack(
                [short, medium, long, volatility, downside], dim=-1
            )
            evidence_logits = self.evidence_head(statistics).squeeze(-1)
            # softplus keeps the evidence contribution non-negative while the
            # zero initialization starts training from a modest, stable scale.
            logits = (
                logits + nn.functional.softplus(self.evidence_scale) * evidence_logits
            )
        return logits, torch.softmax(logits, dim=1)

    @staticmethod
    def allocate(gate: torch.Tensor, experts: torch.Tensor) -> torch.Tensor:
        if gate.ndim != 2 or experts.ndim != 3:
            raise ValueError(
                "gate [batch, experts] and experts [batch, experts, assets] required"
            )
        return torch.einsum("be,bea->ba", gate, experts)
