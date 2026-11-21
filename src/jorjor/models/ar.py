from __future__ import annotations

from .tiny import require_torch


def build_tiny_ar_model(
    vocab_size: int,
    action_size: int,
    opponent_card_size: int,
    hidden_size: int,
    num_heads: int,
    num_layers: int,
):
    """Build a tiny causal Transformer supervised model."""

    torch, nn = require_torch()

    class TinyArModel(nn.Module):
        """Causal Transformer with policy, value, and opponent-hand heads."""

        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.position_embedding = nn.Embedding(128, hidden_size)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 2,
                dropout=0.0,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.policy_head = nn.Linear(hidden_size, action_size)
            self.value_head = nn.Linear(hidden_size, 1)
            self.opponent_card_head = nn.Linear(hidden_size, opponent_card_size)

        def encode_layers(self, input_ids):
            """Return final-token sequence representations for probing."""

            mask = input_ids >= 0
            safe_ids = input_ids.clamp_min(0)
            batch_size, seq_len = safe_ids.shape
            positions = torch.arange(seq_len, device=safe_ids.device).unsqueeze(0).expand(batch_size, seq_len)
            embedded = self.embedding(safe_ids) + self.position_embedding(positions)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=safe_ids.device, dtype=torch.bool),
                diagonal=1,
            )
            last_indices = mask.sum(dim=1).clamp_min(1) - 1
            row_indices = torch.arange(batch_size, device=safe_ids.device)
            layers = {
                "embedding_final": embedded[row_indices, last_indices],
            }
            encoded = embedded
            for layer_index, layer in enumerate(self.encoder.layers):
                encoded = layer(encoded, src_mask=causal_mask, src_key_padding_mask=~mask)
                layers[f"encoder_layer_{layer_index}"] = encoded[row_indices, last_indices]
            if self.encoder.norm is not None:
                encoded = self.encoder.norm(encoded)
            layers["sequence_hidden"] = encoded[row_indices, last_indices]
            return layers

        def forward(self, input_ids):
            layers = self.encode_layers(input_ids)
            hidden = layers["sequence_hidden"]
            return {
                "policy_logits": self.policy_head(hidden),
                "value": self.value_head(hidden).squeeze(-1),
                "opponent_card_logits": self.opponent_card_head(hidden),
            }

    return TinyArModel()
