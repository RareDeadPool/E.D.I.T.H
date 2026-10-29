"""Learned Context-Conditioned Appraisal and Temporal Affective Neural Network.

Implements:
1. Context-Conditioned Appraisal Model:
   Takes structured representations of the event, active goals, prior expectations,
   previous affect, and operational constraints to predict 12 continuous appraisal dimensions
   and calibration confidence.
2. Temporal Affect State Transition Model:
   Recurrent GRU cell modeling multi-timescale affective dynamics (fast momentary response,
   slow baseline mood drift, exponential decay, and flow-state likelihood).
3. Calibration & Out-of-Distribution Handling:
   Outputs epistemic uncertainty and avoids forced emotional classifications.
"""

from typing import Dict, Any, Optional, Tuple
import os
import math
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.cognitive_state import ContinuousAffectState, OperationalResourceState
from engine.appraisal import AppraisalFrame


# 12 Standard Appraisal Dimensions
APPRAISAL_DIMENSIONS = [
    "novelty",
    "goal_relevance",
    "goal_congruence",
    "expectation_violation",
    "agency",
    "controllability",
    "coping_potential",
    "uncertainty",
    "effort_required",
    "task_progress",
    "social_significance",
    "norm_consistency"
]

AFFECT_DIMENSIONS = [
    "valence",
    "arousal",
    "dominance",
    "uncertainty",
    "frustration",
    "engagement",
    "social_trust"
]


class ContextConditionedAppraisalNN(nn.Module):
    """Neural appraisal estimator conditioned on event semantics, goals, and history."""

    def __init__(
        self,
        event_dim: int = 64,
        context_dim: int = 32,
        hidden_dim: int = 128,
        num_appraisal_dims: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_appraisal_dims = num_appraisal_dims

        # Context fusion
        self.input_fusion = nn.Sequential(
            nn.Linear(event_dim + context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # Appraisal projection head: predicts continuous dimensions in [-1, 1] or [0, 1]
        self.appraisal_head = nn.Linear(hidden_dim, num_appraisal_dims)

        # Epistemic uncertainty / calibration head: predicts variance or confidence [0, 1]
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, event_feat: torch.Tensor, context_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            event_feat: [B, event_dim]
            context_feat: [B, context_dim]
            
        Returns:
            appraisals: [B, num_appraisal_dims] in range [-1, 1] (or [0, 1] for unipolar)
            uncertainty: [B, 1] in range [0, 1]
        """
        combined = torch.cat([event_feat, context_feat], dim=-1)
        hidden = self.input_fusion(combined)

        raw_appraisals = self.appraisal_head(hidden)
        # Bounded activation: Tanh maps to [-1.0, 1.0]
        appraisals = torch.tanh(raw_appraisals)

        uncertainty = self.uncertainty_head(hidden)
        return appraisals, uncertainty


class TemporalAffectiveGRU(nn.Module):
    """Recurrent multi-timescale affective transition network."""

    def __init__(
        self,
        appraisal_dim: int = 12,
        affect_dim: int = 7,
        condition_dim: int = 8,
        hidden_dim: int = 48
    ):
        super().__init__()
        self.affect_dim = affect_dim
        self.hidden_dim = hidden_dim

        # Input to GRU: appraisals (12) + prior affect (7) + cognitive condition (8) + delta_t (1)
        input_dim = appraisal_dim + affect_dim + condition_dim + 1
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)

        # Momentary affect update head (fast timescale)
        self.momentary_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, affect_dim)
        )

        # Baseline mood drift head (slow timescale)
        self.baseline_drift_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, affect_dim),
            nn.Tanh()  # Small relative drift in [-1, 1]
        )

        # Flow likelihood head [0, 1]
        self.flow_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        appraisals: torch.Tensor,
        prior_affect: torch.Tensor,
        condition: torch.Tensor,
        delta_t: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update affective states across temporal scales.
        
        Args:
            appraisals: [B, 12]
            prior_affect: [B, 7]
            condition: [B, 8]
            delta_t: [B, 1]
            hidden_state: [B, hidden_dim]
            
        Returns:
            momentary_affect: [B, 7]
            baseline_drift: [B, 7]
            flow_likelihood: [B, 1]
            next_hidden: [B, hidden_dim]
        """
        B = appraisals.size(0)
        if hidden_state is None:
            hidden_state = torch.zeros(B, self.hidden_dim, device=appraisals.device)

        inputs = torch.cat([appraisals, prior_affect, condition, delta_t], dim=-1)
        next_hidden = self.gru_cell(inputs, hidden_state)

        # Momentary affect bounds:
        # valence [-1, 1] via tanh; others [0, 1] via sigmoid
        raw_mom = self.momentary_head(next_hidden)
        val = torch.tanh(raw_mom[:, 0:1])
        others = torch.sigmoid(raw_mom[:, 1:])
        momentary_affect = torch.cat([val, others], dim=-1)

        # Baseline mood drift: scaled slow delta
        baseline_drift = self.baseline_drift_head(next_hidden) * 0.05

        # Flow likelihood
        flow_likelihood = self.flow_head(next_hidden)

        return momentary_affect, baseline_drift, flow_likelihood, next_hidden


class CognitiveNeuralSystem(nn.Module):
    """End-to-end trainable cognitive appraisal and affective transition system."""

    def __init__(
        self,
        vocab_size: int = 1000,
        event_dim: int = 64,
        context_dim: int = 32,
        hidden_dim: int = 128,
        device: str = "cpu"
    ):
        super().__init__()
        self.device = torch.device(device)

        # Lightweight hash/bag-of-words text projection
        self.text_embed = nn.EmbeddingBag(vocab_size, event_dim, mode="mean")
        self.appraisal_nn = ContextConditionedAppraisalNN(
            event_dim=event_dim, context_dim=context_dim, hidden_dim=hidden_dim
        )
        self.temporal_gru = TemporalAffectiveGRU(
            appraisal_dim=12, affect_dim=7, condition_dim=8, hidden_dim=48
        )
        self.to(self.device)

    def extract_context_features(
        self,
        goal_urgency: float,
        goal_progress: float,
        prior_affect: ContinuousAffectState,
        cognitive_load: float,
        prediction_error: float
    ) -> torch.Tensor:
        """Vectorize context and prior conditions into a fixed 32-dim tensor."""
        vec = [
            goal_urgency,
            goal_progress,
            prior_affect.valence,
            prior_affect.arousal,
            prior_affect.dominance,
            prior_affect.uncertainty,
            prior_affect.frustration,
            prior_affect.engagement,
            prior_affect.social_trust,
            cognitive_load,
            prediction_error,
            # Padding to 32 dimensions
            *[0.0] * 21
        ]
        return torch.tensor([vec[:32]], dtype=torch.float32, device=self.device)

    @staticmethod
    def _deterministic_token_id(word: str, vocab_size: int = 1000) -> int:
        digest = hashlib.md5(word.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % vocab_size

    def extract_event_features(self, text: str) -> torch.Tensor:
        """Convert utterance or event string to a dense 64-dim representation with process-invariant hashing."""
        words = text.lower().split()
        if not words:
            indices = torch.tensor([0], dtype=torch.long, device=self.device)
        else:
            indices = torch.tensor([self._deterministic_token_id(w) for w in words], dtype=torch.long, device=self.device)
        return self.text_embed(indices.unsqueeze(0))

    def predict_appraisal(
        self,
        event_text: str,
        goal_urgency: float = 0.5,
        goal_progress: float = 0.0,
        prior_affect: Optional[ContinuousAffectState] = None,
        cognitive_load: float = 0.2,
        prediction_error: float = 0.0,
        with_context: bool = True
    ) -> Tuple[Dict[str, float], float]:
        """Predict continuous appraisal dimensions and confidence from context."""
        self.eval()
        with torch.no_grad():
            aff = prior_affect or ContinuousAffectState()
            event_feat = self.extract_event_features(event_text)
            if with_context:
                ctx_feat = self.extract_context_features(
                    goal_urgency, goal_progress, aff, cognitive_load, prediction_error
                )
            else:
                ctx_feat = torch.zeros((1, 32), dtype=torch.float32, device=self.device)
            appraisals_t, uncert_t = self.appraisal_nn(event_feat, ctx_feat)

            appr_vals = appraisals_t[0].cpu().numpy().tolist()
            uncert = float(uncert_t[0, 0].item())

            appraisal_dict = {
                dim_name: float(appr_vals[i])
                for i, dim_name in enumerate(APPRAISAL_DIMENSIONS)
            }
            # Confidence is inverse of uncertainty
            confidence = max(0.05, min(0.99, 1.0 - uncert))
            return appraisal_dict, confidence

    def predict_transition(
        self,
        appraisal_dict: Dict[str, float],
        prior_affect: ContinuousAffectState,
        cognitive_load: float = 0.2,
        prediction_error: float = 0.0,
        delta_t_seconds: float = 1.0,
        hidden_state: Optional[torch.Tensor] = None,
        with_context: bool = True
    ) -> Tuple[ContinuousAffectState, Dict[str, float], float, torch.Tensor]:
        """Predict next affective state and flow probability."""
        self.eval()
        with torch.no_grad():
            agency_map = {"self": 0.0, "user": 0.33, "system": 0.66, "environment": 1.0}
            appr_vec = []
            for d in APPRAISAL_DIMENSIONS:
                val = appraisal_dict.get(d, 0.0)
                if isinstance(val, str):
                    val = agency_map.get(val.lower(), 0.5)
                appr_vec.append(float(val))
            appr_t = torch.tensor([appr_vec], dtype=torch.float32, device=self.device)

            if with_context:
                aff_vec = [
                    prior_affect.valence,
                    prior_affect.arousal,
                    prior_affect.dominance,
                    prior_affect.uncertainty,
                    prior_affect.frustration,
                    prior_affect.engagement,
                    prior_affect.social_trust
                ]
                cond_vec = [cognitive_load, prediction_error, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                aff_vec = [0.0, 0.3, 0.5, 0.2, 0.0, 0.5, 0.7]
                cond_vec = [0.0] * 8

            aff_t = torch.tensor([aff_vec], dtype=torch.float32, device=self.device)
            cond_t = torch.tensor([cond_vec], dtype=torch.float32, device=self.device)
            dt_t = torch.tensor([[min(10.0, max(0.01, delta_t_seconds / 60.0))]], dtype=torch.float32, device=self.device)

            mom_t, drift_t, flow_t, next_h = self.temporal_gru(
                appr_t, aff_t, cond_t, dt_t, hidden_state=hidden_state
            )

            mom_vals = mom_t[0].cpu().numpy().tolist()
            drift_vals = drift_t[0].cpu().numpy().tolist()
            flow_val = float(flow_t[0, 0].item())

            new_affect = ContinuousAffectState(
                valence=max(-1.0, min(1.0, mom_vals[0])),
                arousal=max(0.0, min(1.0, mom_vals[1])),
                dominance=max(0.0, min(1.0, mom_vals[2])),
                uncertainty=max(0.0, min(1.0, mom_vals[3])),
                frustration=max(0.0, min(1.0, mom_vals[4])),
                engagement=max(0.0, min(1.0, mom_vals[5])),
                social_trust=max(0.0, min(1.0, mom_vals[6]))
            )

            drift_dict = {
                AFFECT_DIMENSIONS[i]: float(drift_vals[i])
                for i in range(len(AFFECT_DIMENSIONS))
            }

            return new_affect, drift_dict, flow_val, next_h

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "model_state_dict": self.state_dict(),
            "vocab_size": 1000,
            "version": "v5.0.0"
        }, path)

    def load_checkpoint(self, path: str) -> bool:
        """Load model checkpoint if exists."""
        if not os.path.exists(path):
            return False
        try:
            ckpt = torch.load(path, map_location=self.device)
            self.load_state_dict(ckpt["model_state_dict"])
            return True
        except Exception as e:
            return False
