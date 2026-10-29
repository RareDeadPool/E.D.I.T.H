"""Computational Cognitive Appraisal System for E.D.I.T.H. V5.

Implements a context-conditioned appraisal model inspired by modern computational emotion theory
(Scherer's Component Process Model and Lazarus' Cognitive-Motivational-Relational Theory).

Evaluates events relative to active goals, prior expectations, coping potential, agency,
and uncertainty, rather than using keyword-to-emotion lookup tables.
"""

from __future__ import annotations
import math
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

from .cognitive_state import CognitiveState


@dataclass
class AppraisalFrame:
    """Multi-dimensional cognitive appraisal frame."""
    novelty: float = 0.2               # [0.0 (routine) to 1.0 (unprecedented)]
    goal_relevance: float = 0.5        # [0.0 (irrelevant) to 1.0 (mission-critical)]
    goal_congruence: float = 0.0       # [-1.0 (severe obstruction) to +1.0 (major achievement)]
    expectation_violation: float = 0.0 # [0.0 (as predicted) to 1.0 (shocking anomaly)]
    agency: str = "environment"        # 'self', 'user', 'environment', 'system'
    controllability: float = 0.7       # [0.0 (uncontrollable) to 1.0 (fully controllable)]
    coping_potential: float = 0.8      # [0.0 (hopeless/blocked) to 1.0 (clear recovery strategy)]
    uncertainty: float = 0.2           # [0.0 (complete clarity) to 1.0 (total ambiguity)]
    effort_required: float = 0.3       # [0.0 (effortless) to 1.0 (exhausting computation)]
    task_progress: float = 0.0         # [-1.0 (regressed) to +1.0 (completed)]
    social_significance: float = 0.5   # [0.0 (neutral machine interaction) to 1.0 (deep interpersonal bond)]
    norm_consistency: float = 1.0      # [0.0 (violates safety/permission) to 1.0 (fully aligned)]
    confidence: float = 0.85           # Confidence in this appraisal evaluation (0.0 to 1.0)
    provenance_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_feature_vector(self) -> List[float]:
        agency_map = {"self": 0.0, "user": 0.33, "system": 0.66, "environment": 1.0}
        agency_val = agency_map.get(self.agency, 1.0)
        return [
            self.novelty,
            self.goal_relevance,
            self.goal_congruence,
            self.expectation_violation,
            agency_val,
            self.controllability,
            self.coping_potential,
            self.uncertainty,
            self.effort_required,
            self.task_progress,
            self.social_significance,
            self.norm_consistency,
        ]


class ContextualAppraisalEngine:
    """Evaluates the meaning of events in context of goals, memory, and cognitive state."""

    def __init__(self, neural_appraisal_model: Optional[Any] = None):
        self.neural_model = neural_appraisal_model

    def appraise_event(
        self,
        event: Dict[str, Any],
        state: CognitiveState,
        recent_history: Optional[List[Dict[str, Any]]] = None
    ) -> AppraisalFrame:
        """
        Evaluate an event in the context of active goals, expectations, and recent history.

        Args:
            event: {
                'type': 'user_message' | 'task_result' | 'execution_failure' | 'interruption' | 'system_alert',
                'text': str,
                'intent': Optional[str],
                'action': Optional[str],
                'success': Optional[bool],
                'verified': Optional[bool],
                'error': Optional[str],
                'recovery_available': Optional[bool],
                'is_social': Optional[bool],
                'is_safety_violation': Optional[bool]
            }
            state: Current CognitiveState
            recent_history: List of recent interaction/appraisal episodes
        """
        notes = []
        ev_type = event.get("type", "user_message")
        text = str(event.get("text", "")).lower()

        # 1. Goal Relevance
        active_goal = state.goals.active_goal
        if active_goal:
            # Relevance is higher if text relates to active goal or task progress
            if any(term in text for term in active_goal.lower().split() if len(term) > 3):
                relevance = 0.85
                notes.append(f"Event matches lexical tokens of active goal '{active_goal}'")
            elif ev_type in ["task_result", "execution_failure"]:
                relevance = 0.95
                notes.append("Execution outcome directly targets current task")
            else:
                relevance = 0.55
        else:
            relevance = 0.6 if ev_type == "user_message" else 0.3
            notes.append("No prior active goal; relevance set to baseline exploratory priority")

        # 2. Goal Congruence & Task Progress
        congruence = 0.0
        progress = 0.0
        if ev_type in ["task_result", "action_outcome"]:
            is_success = event.get("is_success", event.get("success", True))
            is_verified = event.get("verified", True)
            if is_success and is_verified:
                congruence = 0.85
                progress = 0.7
                notes.append("Task succeeded and was verified closed-loop: highly goal-congruent")
            elif is_success and not is_verified:
                congruence = 0.1
                progress = 0.2
                notes.append("Action executed but closed-loop verification pending/unconfirmed")
            else:
                urgency = state.goals.urgency
                curiosity = state.goals.curiosity_drive
                is_exploratory = (curiosity > 0.6 or urgency < 0.3 or state.goals.explore_vs_exploit > 0.6)
                if is_exploratory:
                    congruence = -0.2
                    progress = 0.05
                    notes.append("Task failure in exploratory context: treated as informative trial, not severe setback")
                else:
                    congruence = -0.85
                    progress = -0.3
                    notes.append("Task failure under high goal urgency: severe goal incongruence")
        elif ev_type == "execution_failure":
            urgency = getattr(state.goals, "urgency", 0.5)
            priority = getattr(state.goals, "priority", 0.5)
            effective_urgency = min(urgency, priority) if priority < 0.3 or urgency < 0.3 else max(urgency, priority)
            if effective_urgency < 0.3:
                congruence = -0.3
                progress = -0.1
                notes.append("Execution failure during low-urgency trial")
            else:
                congruence = -0.85
                progress = -0.4
                notes.append("Direct execution exception during critical operation")
        elif ev_type == "interruption":
            congruence = -0.2
            progress = 0.0
            notes.append("User requested halt/hold-on; mild goal pausing without destructive failure")
        elif ev_type == "user_message":
            # Social / conversational congruence
            if any(w in text for w in ["thank", "awesome", "great", "nice", "love", "good job", "perfect"]):
                congruence = 0.6
                notes.append("Positive user feedback promotes goal congruence")
            elif any(w in text for w in ["hate", "bad", "wrong", "broke", "not working", "useless", "terrible"]):
                congruence = -0.5
                notes.append("Negative user feedback indicates goal obstruction")
            else:
                congruence = 0.1

        # 3. Novelty & Expectation Violation
        prior_expected = state.perception.prior_expectations.get("expected_outcome")
        novelty = 0.25
        expectation_violation = 0.0

        observed = event.get("is_success", event.get("success", None))
        if prior_expected is not None and observed is not None:
            if observed != prior_expected:
                expectation_violation = max(0.8, float(event.get("prediction_error", 0.8)))
                novelty = 0.75
                notes.append(f"Outcome violation: expected success={prior_expected}, observed success={observed}")
            else:
                expectation_violation = 0.05
                novelty = 0.2
                notes.append("Outcome matched expectation")
        elif float(event.get("prediction_error", 0.0)) > 0.0:
            expectation_violation = float(event["prediction_error"])
            novelty = min(0.9, 0.2 + expectation_violation * 0.7)
            notes.append(f"Expectation violation driven by prediction error {expectation_violation:.2f}")
        else:
            # Check recurrence in recent history
            if recent_history:
                recent_texts = [str(h.get("text", "")).lower() for h in recent_history[-5:]]
                if text in recent_texts:
                    novelty = 0.1
                    notes.append("Repeated input in recent window: low novelty")
                else:
                    novelty = 0.45

        # 4. Agency Attribution
        if ev_type in ["task_result", "execution_failure"]:
            agency = "self" if event.get("action") else "system"
            notes.append(f"Agency attributed to '{agency}' for execution event")
        elif ev_type == "interruption":
            agency = "user"
            notes.append("Agency attributed to 'user' for interruption")
        elif ev_type == "system_alert":
            agency = "environment"
            notes.append("Agency attributed to 'environment' for system alert")
        else:
            agency = "user"

        # 5. Controllability & Coping Potential
        # A failure is NOT devastating if we have clear recovery strategies
        recovery_available = event.get("recovery_available", True)
        retry_count = state.operational.retry_count

        if recovery_available and retry_count < 2:
            coping_potential = 0.85
            controllability = 0.8
            notes.append("Recovery actions or retries are available; high coping potential")
        elif recovery_available and retry_count >= 2:
            coping_potential = 0.4
            controllability = 0.4
            notes.append("Repeated retries failing; coping potential significantly degraded")
        else:
            coping_potential = 0.25
            controllability = 0.3
            notes.append("No immediate automated recovery available; requires clarification or user intervention")

        # 6. Uncertainty & Ambiguity
        conf = float(event.get("confidence", state.condition.plan_confidence))
        uncertainty = max(0.05, min(0.95, 1.0 - conf))
        if any(q in text for q in ["maybe", "perhaps", "not sure", "don't know", "what if", "unknown"]):
            uncertainty = min(0.95, uncertainty + 0.3)
            notes.append("Lexical indicators of ambiguity detected")

        # 7. Effort & Cognitive Load Impact
        task_complexity = state.condition.task_complexity
        effort = max(0.1, min(1.0, 0.3 + 0.5 * task_complexity))

        # 8. Social Significance
        is_social = bool(
            event.get("is_social") or
            any(w in text for w in ["love", "feel", "friend", "lonely", "breakup", "crush", "sad", "happy", "talk to me", "edith"])
        )
        social_sig = 0.85 if is_social else 0.3
        if is_social:
            notes.append("High social/interpersonal significance detected")

        # 9. Norm Consistency (Safety & Authorization)
        is_safety_violation = bool(event.get("is_safety_violation", False))
        norm_consistency = 0.0 if is_safety_violation else 1.0
        if is_safety_violation:
            notes.append("CRITICAL: Action violates safety policy or authorization constraint")

        # 10. Neural Appraisal Integration (if available)
        if self.neural_model is not None and not is_safety_violation:
            try:
                learned_appr, learned_conf = self.neural_model.predict_appraisal(
                    event_text=text or ev_type,
                    goal_urgency=state.goals.urgency,
                    goal_progress=state.goals.progress,
                    prior_affect=state.affect,
                    cognitive_load=state.condition.cognitive_load,
                    prediction_error=state.condition.prediction_error
                )
                # Bayesian blending: learned representation (60%) + situational constraints (40%)
                relevance = 0.6 * learned_appr.get("goal_relevance", relevance) + 0.4 * relevance
                congruence = 0.6 * learned_appr.get("goal_congruence", congruence) + 0.4 * congruence
                novelty = 0.6 * learned_appr.get("novelty", novelty) + 0.4 * novelty
                expectation_violation = 0.6 * learned_appr.get("expectation_violation", expectation_violation) + 0.4 * expectation_violation
                controllability = 0.6 * learned_appr.get("controllability", controllability) + 0.4 * controllability
                coping_potential = 0.6 * learned_appr.get("coping_potential", coping_potential) + 0.4 * coping_potential
                uncertainty = 0.6 * learned_appr.get("uncertainty", uncertainty) + 0.4 * uncertainty
                conf = 0.5 * learned_conf + 0.5 * conf
                notes.append(f"Fused learned neural appraisal (conf={learned_conf:.2f}) with contextual constraints")
            except Exception as e:
                notes.append(f"Neural appraisal bypass/fallback: {e}")

        frame = AppraisalFrame(
            novelty=round(novelty, 3),
            goal_relevance=round(relevance, 3),
            goal_congruence=round(congruence, 3),
            expectation_violation=round(expectation_violation, 3),
            agency=agency,
            controllability=round(controllability, 3),
            coping_potential=round(coping_potential, 3),
            uncertainty=round(uncertainty, 3),
            effort_required=round(effort, 3),
            task_progress=round(progress, 3),
            social_significance=round(social_sig, 3),
            norm_consistency=round(norm_consistency, 3),
            confidence=round(conf, 3),
            provenance_notes=notes
        )

        return frame
