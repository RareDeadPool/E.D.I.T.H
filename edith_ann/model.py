"""Multi-Task Artificial Neural Network (ANN) Architecture for E.D.I.T.H. Intelligence Core V2."""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import PositionalEncoding, TransformerEncoderBlock


class EdithNeuralNetwork(nn.Module):
    """
    E.D.I.T.H. Custom Multi-Task Neural Network V2.
    
    Cognitive outputs:
    1. Intent Classification (high-level user goal)
    2. Slot / Entity Sequence Tagging (token-level parameter extraction)
    3. Action Policy Selection (executable tool/action dispatch)
    4. Operational Confidence / Out-of-Distribution Calibration (operational certainty)
    5. Sequence Complexity Prediction (atomic single-step vs compound multi-step)
    """

    def __init__(
        self,
        vocab_size: int,
        num_intents: int,
        num_slots: int,
        num_actions: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_intents = num_intents
        self.num_slots = num_slots
        self.num_actions = num_actions
        self.max_seq_len = max_seq_len

        # 1. Input Representation Layer
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_seq_len=max_seq_len, dropout=dropout)
        self.embed_norm = nn.LayerNorm(d_model)

        # 2. Transformer Encoder Backbone
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # 3. Intent Classification Head
        self.intent_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_intents),
        )

        # 4. Token-level Slot Tagging Head (BIO entity tagging)
        self.slot_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_slots),
        )

        # 5. Action Policy Selection Head
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_actions),
        )

        # 6. Operational Confidence Head (predicts command actionability score)
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # 7. Sequence Complexity Head (predicts single-step vs multi-step plan requirement)
        self.complexity_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform distribution."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] with 1 for tokens and 0 for padding
        Returns:
            Dictionary containing:
                - intent_logits: [batch_size, num_intents]
                - slot_logits: [batch_size, seq_len, num_slots]
                - action_logits: [batch_size, num_actions]
                - confidence: [batch_size, 1]
                - complexity_logits: [batch_size, 2]
                - contextual_embeddings: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = input_ids.size()

        # Build attention mask for MHA if provided: [batch_size, 1, 1, seq_len]
        mask_mha = None
        if attention_mask is not None:
            mask_mha = attention_mask.unsqueeze(1).unsqueeze(2)

        # 1. Embeddings
        x = self.token_embedding(input_ids) * (self.d_model ** 0.5)
        x = self.pos_encoder(x)
        x = self.embed_norm(x)

        # 2. Transformer Backbone Pass
        for layer in self.encoder_layers:
            x = layer(x, mask=mask_mha)
        x = self.encoder_norm(x)  # [batch_size, seq_len, d_model]

        # 3. Context Pooling for Sentence-Level Heads (Intent, Action, Confidence, Complexity)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
            sum_embeddings = torch.sum(x * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask  # [batch_size, d_model]
        else:
            pooled = x[:, 0, :]  # Fallback to BOS representation

        # 4. Multi-Task Cognitive Predictions
        intent_logits = self.intent_head(pooled)
        slot_logits = self.slot_head(x)  # [batch_size, seq_len, num_slots]
        action_logits = self.action_head(pooled)
        confidence = self.confidence_head(pooled)
        complexity_logits = self.complexity_head(pooled)

        return {
            "intent_logits": intent_logits,
            "slot_logits": slot_logits,
            "action_logits": action_logits,
            "confidence": confidence,
            "complexity_logits": complexity_logits,
            "contextual_embeddings": x,
        }

    def count_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
