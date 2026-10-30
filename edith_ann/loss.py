"""Multi-Task Loss Module for E.D.I.T.H. Intelligence Core V2."""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdithMultiTaskLoss(nn.Module):
    """
    Combined multi-task loss with calibrated operational uncertainty:
    L = w_intent * L_intent + w_slot * L_slot + w_action * L_action + w_conf * L_conf + w_comp * L_comp
    """

    def __init__(
        self,
        intent_weight: float = 1.0,
        slot_weight: float = 1.5,
        action_weight: float = 1.0,
        confidence_weight: float = 0.5,
        complexity_weight: float = 0.5,
        ignore_slot_index: int = -100,
        slot_class_weights: Optional[torch.Tensor] = None,
        slot_label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.intent_weight = intent_weight
        self.slot_weight = slot_weight
        self.action_weight = action_weight
        self.confidence_weight = confidence_weight
        self.complexity_weight = complexity_weight
        self.ignore_slot_index = ignore_slot_index
        self.slot_class_weights = slot_class_weights
        self.slot_label_smoothing = slot_label_smoothing

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: Output dictionary from EdithNeuralNetwork
            targets: Dictionary with:
                - intent: [batch_size] (int64)
                - slot: [batch_size, seq_len] (int64, padded with ignore_slot_index)
                - action: [batch_size] (int64)
                - confidence: [batch_size, 1] (float32)
                - complexity: [batch_size] (int64, optional)
        """
        # 1. Intent Cross-Entropy Loss
        loss_intent = F.cross_entropy(
            predictions["intent_logits"],
            targets["intent"]
        )

        # 2. Slot Tagging Cross-Entropy Loss (flattened, ignoring padding)
        slot_logits = predictions["slot_logits"].view(-1, predictions["slot_logits"].size(-1))
        slot_targets = targets["slot"].view(-1)
        w = self.slot_class_weights.to(slot_logits.device) if self.slot_class_weights is not None else None
        loss_slot = F.cross_entropy(
            slot_logits,
            slot_targets,
            weight=w,
            ignore_index=self.ignore_slot_index,
            label_smoothing=self.slot_label_smoothing
        )

        # 3. Action Policy Cross-Entropy Loss
        loss_action = F.cross_entropy(
            predictions["action_logits"],
            targets["action"]
        )

        # 4. Confidence Calibration Loss (Binary Cross-Entropy)
        loss_conf = F.binary_cross_entropy(
            predictions["confidence"],
            targets["confidence"]
        )

        # 5. Complexity Loss (if provided)
        loss_comp = torch.tensor(0.0, device=predictions["intent_logits"].device)
        if "complexity" in targets and "complexity_logits" in predictions:
            loss_comp = F.cross_entropy(
                predictions["complexity_logits"],
                targets["complexity"]
            )

        # Total Weighted Multi-Task Loss
        total_loss = (
            self.intent_weight * loss_intent +
            self.slot_weight * loss_slot +
            self.action_weight * loss_action +
            self.confidence_weight * loss_conf +
            self.complexity_weight * loss_comp
        )

        return {
            "total_loss": total_loss,
            "intent_loss": loss_intent,
            "slot_loss": loss_slot,
            "action_loss": loss_action,
            "confidence_loss": loss_conf,
            "complexity_loss": loss_comp,
        }
