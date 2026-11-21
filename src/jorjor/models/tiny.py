from __future__ import annotations


def require_torch():
    """Import torch with a clear message for local environments without PyTorch."""

    try:
        import torch
        from torch import nn
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for tiny model training") from exc
    return torch, nn


def build_tiny_model(vocab_size: int, action_size: int, opponent_card_size: int, hidden_size: int):
    """Build the tiny pooled-token supervised model."""

    torch, nn = require_torch()

    class TinyLeducModel(nn.Module):
        """Tiny model with policy, value, and opponent-card heads."""

        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.body = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
            )
            self.policy_head = nn.Linear(hidden_size, action_size)
            self.value_head = nn.Linear(hidden_size, 1)
            self.opponent_card_head = nn.Linear(hidden_size, opponent_card_size)

        def encode_layers(self, input_ids):
            """Return pooled and hidden representations for probing."""

            mask = input_ids >= 0
            safe_ids = input_ids.clamp_min(0)
            embedded = self.embedding(safe_ids)
            masked = embedded * mask.unsqueeze(-1)
            lengths = mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
            pooled = masked.sum(dim=1) / lengths
            hidden = self.body(pooled)
            return {
                "embedding_pool": pooled,
                "body_hidden": hidden,
            }

        def forward(self, input_ids):
            layers = self.encode_layers(input_ids)
            hidden = layers["body_hidden"]
            return {
                "policy_logits": self.policy_head(hidden),
                "value": self.value_head(hidden).squeeze(-1),
                "opponent_card_logits": self.opponent_card_head(hidden),
            }

    return TinyLeducModel()
