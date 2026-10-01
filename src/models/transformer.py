"""Transformer sequence classifier for Task A (question -> provision label).

Pretrained encoder (transfer learning) + pooling + dropout + linear head, fine-tuned end to
end. Used as (1) the Transformer baseline and (2) a source of cheap signals (confidence,
entropy, margin) for the difficulty estimator in later phases.
"""
from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel


class TransformerClassifier(nn.Module):
    def __init__(self, model_name: str, n_classes: int, dropout: float = 0.1, pooling: str = "cls") -> None:
        super().__init__()
        # transformers>=5 loads the checkpoint's stored dtype by default (deberta-v3-base is fp16),
        # and AdamW on fp16 weights diverges to NaN. Keep fp32 master weights; bf16 is autocast only.
        self.encoder = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        hidden = self.encoder.config.hidden_size
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_classes)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return hidden[:, 0]
        if self.pooling == "mean":
            m = mask.unsqueeze(-1).to(hidden.dtype)
            return (hidden * m).sum(1) / m.sum(1).clamp(min=1.0)
        raise ValueError(self.pooling)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.head(self.dropout(self.pool(out.last_hidden_state, attention_mask)))
