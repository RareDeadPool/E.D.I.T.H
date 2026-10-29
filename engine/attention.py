"""Adaptive Attention Allocation Engine for E.D.I.T.H. V5.

Determines what information, context, and memories should be brought into
working focus based on goal relevance, uncertainty, prediction error, and urgency.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from .cognitive_state import CognitiveState


@dataclass
class AttentionSalience:
    """Salience weights across different cognitive dimensions."""
    goal_relevance_weight: float = 0.35
    prediction_error_weight: float = 0.25
    uncertainty_weight: float = 0.20
    urgency_weight: float = 0.10
    novelty_weight: float = 0.10


class AdaptiveAttentionEngine:
    """Allocates computational focus and filters context dynamically."""

    def __init__(self, weights: Optional[AttentionSalience] = None):
        self.weights = weights or AttentionSalience()

    def rank_context_items(
        self,
        candidate_items: List[Dict[str, Any]],
        state: CognitiveState
    ) -> List[Dict[str, Any]]:
        """
        Rank and filter candidate contextual items (memories, entities, subtasks)
        using multi-factor salience scoring.
        """
        if not candidate_items:
            return []

        active_goal = (state.goals.active_goal or "").lower()
        active_goal_tokens = set(active_goal.split())
        pred_error = state.condition.prediction_error
        urgency = state.goals.urgency

        scored: List[Tuple[float, Dict[str, Any]]] = []

        for item in candidate_items:
            text = str(item.get("text", "") or item.get("name", "") or item.get("key", "")).lower()
            tokens = set(text.split())

            # 1. Goal Relevance
            overlap = len(active_goal_tokens.intersection(tokens)) if active_goal_tokens else 0
            rel_score = min(1.0, overlap * 0.4 + (0.3 if item.get("is_active_task") else 0.0))

            # 2. Uncertainty & Information Need
            unc_score = float(item.get("uncertainty", 0.2))

            # 3. Prediction Error Salience (items that failed or violated expectations need focus)
            err_score = 0.8 if item.get("has_error") or item.get("unverified") else (pred_error * 0.5)

            # 4. Novelty & Recency
            novelty_score = float(item.get("novelty", 0.3))

            salience = (
                self.weights.goal_relevance_weight * rel_score +
                self.weights.prediction_error_weight * err_score +
                self.weights.uncertainty_weight * unc_score +
                self.weights.urgency_weight * urgency +
                self.weights.novelty_weight * novelty_score
            )

            scored.append((salience, item))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Attach computed attention salience to items
        ranked = []
        for s, it in scored:
            it_copy = dict(it)
            it_copy["attention_salience"] = round(s, 3)
            ranked.append(it_copy)

        return ranked

    def should_trigger_deep_deliberation(self, state: CognitiveState) -> Tuple[bool, str]:
        """
        Determine if the current situation justifies pausing fast heuristics
        to engage in deeper multi-step alternative exploration.
        """
        # Condition A: Repeated errors or high prediction error
        if state.condition.error_accumulation > 0.5:
            return True, "error_accumulation_exceeds_threshold"

        # Condition B: High goal priority with high uncertainty
        if state.goals.priority > 0.7 and state.condition.plan_confidence < 0.6:
            return True, "high_priority_low_confidence"

        # Condition C: Detected contradiction in perception/beliefs
        if len(state.perception.detected_contradictions) > 0:
            return True, "detected_belief_contradiction"

        return False, "fast_heuristic_sufficient"
