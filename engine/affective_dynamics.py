"""Temporal Affective Dynamics and Mood Engine for E.D.I.T.H. V5.

Implements dual-timescale computational affect:
1. Momentary Affect: Fast-responding state reflecting immediate appraisals and prediction errors.
2. Baseline Mood: Slower-moving integrative bias reflecting sustained conditions and unresolved goals.

Provides continuous dimensional updates, temporal decay, recovery mechanics,
and diagnostic attribution traces without naive keyword-to-emotion tables.
"""

from __future__ import annotations
import math
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

from .cognitive_state import ContinuousAffectState, CognitiveState
from .appraisal import AppraisalFrame


@dataclass
class AffectiveTransitionDiagnostic:
    """Explains why and how the affective state changed with rigorous cognitive provenance."""
    timestamp: float
    trigger_event: str
    appraisal_congruence: float
    appraisal_expectation_violation: float
    momentary_shift: Dict[str, float]
    baseline_drift: Dict[str, float]
    dominant_disposition: str
    attribution_reason: str
    previous_state: Dict[str, Any] = field(default_factory=dict)
    current_event_representation: Dict[str, Any] = field(default_factory=dict)
    goal_and_memory_context: Dict[str, Any] = field(default_factory=dict)
    appraisal_model_output: Dict[str, Any] = field(default_factory=dict)
    temporal_model_input: Optional[Dict[str, Any]] = None
    temporal_model_output: Optional[Dict[str, Any]] = None
    deterministic_constraints_applied: List[str] = field(default_factory=list)
    final_state_update: Dict[str, Any] = field(default_factory=dict)
    attribution_provenance: str = "deterministic_heuristic_fallback"
    uncertainty_and_evidence: Dict[str, Any] = field(default_factory=dict)


class AffectiveDynamicsEngine:
    """Manages continuous dimensional affect, dual-timescale mood updates, and recovery."""

    # Timescale constants
    FAST_ALPHA = 0.60   # Momentary update weight
    SLOW_BETA = 0.05    # Baseline mood moving integration weight
    TIME_DECAY_HALF_LIFE = 60.0  # Seconds for momentary affect to decay 50% toward baseline

    def __init__(self, temporal_neural_model: Optional[Any] = None):
        self.temporal_neural_model = temporal_neural_model
        self.hidden_state: Optional[Any] = None
        self.diagnostics_log: List[AffectiveTransitionDiagnostic] = []

    def update_affect(
        self,
        state: CognitiveState,
        appraisal: AppraisalFrame,
        event_summary: str = "",
        elapsed_seconds: Optional[float] = None,
        event_representation: Optional[Dict[str, Any]] = None,
        goal_context: Optional[Dict[str, Any]] = None
    ) -> CognitiveState:
        """
        Perform unified dual-timescale temporal state update.

        1. Computes target momentary affect vector via learned GRU or heuristic fallback.
        2. Applies time decay from previous momentary state toward baseline mood.
        3. Integrates new appraisal with decayed momentary state (Fast timescale).
        4. Drifts baseline mood slowly toward sustained condition (Slow timescale).
        5. Updates CognitiveState and records transparent diagnostics.
        """
        now = time.time()
        dt = elapsed_seconds if elapsed_seconds is not None else max(0.0, now - state.timestamp)

        prev_mom = state.momentary_affect
        prev_base = state.baseline_mood
        prev_state_dict = {
            "momentary_affect": prev_mom.to_dict(),
            "baseline_mood": prev_base.to_dict()
        }

        constraints_applied: List[str] = []

        # Step 1: Temporal decay of momentary affect toward baseline mood
        decay_factor = math.exp(-dt / self.TIME_DECAY_HALF_LIFE) if dt > 0 else 1.0
        decayed_valence = prev_base.valence + (prev_mom.valence - prev_base.valence) * decay_factor
        decayed_arousal = prev_base.arousal + (prev_mom.arousal - prev_base.arousal) * decay_factor
        decayed_dominance = prev_base.dominance + (prev_mom.dominance - prev_base.dominance) * decay_factor
        decayed_frustration = prev_base.frustration + (prev_mom.frustration - prev_base.frustration) * decay_factor
        decayed_engagement = prev_base.engagement + (prev_mom.engagement - prev_base.engagement) * decay_factor
        constraints_applied.append(
            f"exponential_decay(half_life={self.TIME_DECAY_HALF_LIFE}s, dt={dt:.2f}s, factor={decay_factor:.3f})"
        )

        # Step 2: Compute target momentary affect via GRU or deterministic baseline
        temporal_in = None
        temporal_out = None
        provenance = "deterministic_heuristic_fallback"
        neural_drift_dict: Dict[str, float] = {}

        if self.temporal_neural_model is not None:
            try:
                cog_load = getattr(getattr(state, "condition", None), "cognitive_load", 0.2)
                pred_err = float(getattr(getattr(state, "condition", None), "prediction_error", 0.0))
                appr_dict = appraisal.to_dict()

                temporal_in = {
                    "appraisal_dim_count": len(appr_dict),
                    "prior_affect": prev_mom.to_dict(),
                    "cognitive_load": round(cog_load, 3),
                    "prediction_error": round(pred_err, 3),
                    "delta_t_seconds": round(dt, 2)
                }

                neural_mom, drift_dict, flow_val, next_h = self.temporal_neural_model.predict_transition(
                    appraisal_dict=appr_dict,
                    prior_affect=prev_mom,
                    cognitive_load=cog_load,
                    prediction_error=pred_err,
                    delta_t_seconds=dt,
                    hidden_state=self.hidden_state
                )
                self.hidden_state = next_h
                neural_drift_dict = drift_dict

                # Grounded cognitive fusion: blend learned neural dynamics with grounded appraisal signals
                appr_grounded_val = appraisal.goal_congruence * 0.7 + (appraisal.task_progress * 0.3)
                target_valence = 0.5 * neural_mom.valence + 0.5 * appr_grounded_val
                target_arousal = max(0.1, min(1.0, 0.5 * neural_mom.arousal + 0.5 * (0.2 + 0.4 * appraisal.expectation_violation + 0.3 * appraisal.novelty + 0.2 * appraisal.effort_required)))
                target_dominance = max(0.0, min(1.0, 0.5 * neural_mom.dominance + 0.5 * (appraisal.controllability * 0.5 + appraisal.coping_potential * 0.5)))
                target_uncertainty = max(0.0, min(1.0, 0.5 * neural_mom.uncertainty + 0.5 * appraisal.uncertainty))

                if appraisal.goal_congruence < 0:
                    appr_grounded_fru = min(1.0, abs(appraisal.goal_congruence) * (1.2 - 0.5 * appraisal.coping_potential))
                    target_frustration = max(neural_mom.frustration, appr_grounded_fru * 0.75)
                else:
                    target_frustration = min(neural_mom.frustration, max(0.0, prev_mom.frustration * 0.4 - 0.2 * appraisal.goal_congruence))

                target_engagement = max(0.0, min(1.0, 0.5 * neural_mom.engagement + 0.5 * (0.3 + 0.4 * appraisal.goal_relevance + 0.3 * appraisal.controllability)))
                target_trust = max(0.0, min(1.0, 0.5 * neural_mom.social_trust + 0.5 * prev_base.social_trust))

                temporal_out = {
                    "neural_momentary": neural_mom.to_dict(),
                    "drift_vector": {k: round(v, 4) for k, v in drift_dict.items()},
                    "flow_likelihood": round(flow_val, 4)
                }
                provenance = "neural_gru_fused"
                constraints_applied.append("neural_temporal_gru_evaluated")
                constraints_applied.append("grounded_appraisal_neural_fusion")
            except Exception as err:
                provenance = "deterministic_heuristic_fallback"
                constraints_applied.append(f"neural_gru_exception_fallback({str(err)[:50]})")
                self.temporal_neural_model = None

        if provenance == "deterministic_heuristic_fallback":
            # Goal Congruence drives Valence directly
            target_valence = appraisal.goal_congruence * 0.7 + (appraisal.task_progress * 0.3)

            # Novelty and Expectation Violation drive Arousal
            target_arousal = max(
                0.1,
                min(1.0, 0.2 + 0.4 * appraisal.expectation_violation + 0.3 * appraisal.novelty + 0.2 * appraisal.effort_required)
            )

            # Controllability and Coping Potential drive Dominance / Agency
            target_dominance = appraisal.controllability * 0.5 + appraisal.coping_potential * 0.5

            # Uncertainty directly maps to cognitive uncertainty
            target_uncertainty = appraisal.uncertainty

            # Frustration emerges when: Goal Congruence is negative AND Expectation Violation or low Coping occurs
            if appraisal.goal_congruence < 0:
                target_frustration = min(1.0, abs(appraisal.goal_congruence) * (1.2 - 0.5 * appraisal.coping_potential))
            else:
                target_frustration = max(0.0, prev_mom.frustration * 0.4 - 0.2 * appraisal.goal_congruence)

            # Engagement increases with Goal Relevance and moderate Arousal/Controllability
            target_engagement = min(1.0, 0.3 + 0.4 * appraisal.goal_relevance + 0.3 * appraisal.controllability)

            # Social Trust increases with positive Social Significance and Norm Consistency
            target_trust = min(1.0, prev_base.social_trust * 0.8 + 0.2 * (appraisal.social_significance * appraisal.norm_consistency))
            constraints_applied.append("deterministic_appraisal_formulas_applied")

        # Acute sudden shift override for severe events
        is_acute = (appraisal.expectation_violation > 0.6) or (appraisal.goal_congruence < -0.7)
        alpha = min(0.95, self.FAST_ALPHA * 1.5) if is_acute else self.FAST_ALPHA
        if is_acute:
            constraints_applied.append(f"acute_event_shift_alpha_boosted({alpha:.2f})")

        # Step 3: Fast Timescale Update (New Momentary Affect)
        new_mom_valence = (1.0 - alpha) * decayed_valence + alpha * target_valence
        new_mom_arousal = (1.0 - alpha) * decayed_arousal + alpha * target_arousal
        new_mom_dominance = (1.0 - alpha) * decayed_dominance + alpha * target_dominance
        new_mom_uncertainty = (1.0 - alpha) * prev_mom.uncertainty + alpha * target_uncertainty
        new_mom_frustration = (1.0 - alpha) * decayed_frustration + alpha * target_frustration
        new_mom_engagement = (1.0 - alpha) * decayed_engagement + alpha * target_engagement
        new_mom_trust = (1.0 - alpha) * prev_mom.social_trust + alpha * target_trust

        new_momentary = ContinuousAffectState(
            valence=new_mom_valence,
            arousal=new_mom_arousal,
            dominance=new_mom_dominance,
            uncertainty=new_mom_uncertainty,
            frustration=new_mom_frustration,
            engagement=new_mom_engagement,
            social_trust=new_mom_trust,
        )
        new_momentary.clamp()
        constraints_applied.append("clamped_momentary_affect_bounds")

        # Step 4: Slow Timescale Update (Baseline Mood Drift)
        beta = self.SLOW_BETA
        neural_val_drift = neural_drift_dict.get("valence", 0.0) * 0.05
        neural_fru_drift = neural_drift_dict.get("frustration", 0.0) * 0.05

        new_base_valence = (1.0 - beta) * prev_base.valence + beta * new_momentary.valence + neural_val_drift
        new_base_arousal = (1.0 - beta) * prev_base.arousal + beta * new_momentary.arousal
        new_base_dominance = (1.0 - beta) * prev_base.dominance + beta * new_momentary.dominance
        new_base_frustration = (1.0 - beta) * prev_base.frustration + beta * new_momentary.frustration + neural_fru_drift
        new_base_engagement = (1.0 - beta) * prev_base.engagement + beta * new_momentary.engagement
        new_base_trust = (1.0 - beta) * prev_base.social_trust + beta * new_momentary.social_trust

        new_baseline = ContinuousAffectState(
            valence=new_base_valence,
            arousal=new_base_arousal,
            dominance=new_base_dominance,
            uncertainty=(1.0 - beta) * prev_base.uncertainty + beta * new_momentary.uncertainty,
            frustration=new_base_frustration,
            engagement=new_base_engagement,
            social_trust=new_base_trust,
        )
        new_baseline.clamp()
        constraints_applied.append("clamped_baseline_mood_bounds")

        # Step 5: Interpretation & Diagnostics
        disposition = self.interpret_affect_disposition(new_momentary, new_baseline)

        shift_summary = {
            "delta_valence": round(new_momentary.valence - prev_mom.valence, 3),
            "delta_frustration": round(new_momentary.frustration - prev_mom.frustration, 3),
            "delta_arousal": round(new_momentary.arousal - prev_mom.arousal, 3),
        }
        drift_summary = {
            "baseline_valence": round(new_baseline.valence, 3),
            "baseline_frustration": round(new_baseline.frustration, 3),
        }

        reason_str = (
            f"Congruence={appraisal.goal_congruence:+.2f}, Violation={appraisal.expectation_violation:.2f}, "
            f"Coping={appraisal.coping_potential:.2f} [{provenance}] -> {disposition}"
        )

        final_state_dict = {
            "momentary_affect": new_momentary.to_dict(),
            "baseline_mood": new_baseline.to_dict()
        }

        diag = AffectiveTransitionDiagnostic(
            timestamp=now,
            trigger_event=event_summary or appraisal.agency,
            appraisal_congruence=appraisal.goal_congruence,
            appraisal_expectation_violation=appraisal.expectation_violation,
            momentary_shift=shift_summary,
            baseline_drift=drift_summary,
            dominant_disposition=disposition,
            attribution_reason=reason_str,
            previous_state=prev_state_dict,
            current_event_representation=event_representation or {"summary": event_summary or appraisal.agency},
            goal_and_memory_context=goal_context or {
                "active_goal": getattr(getattr(state, "goals", None), "active_goal", None),
                "goal_progress": getattr(getattr(state, "goals", None), "progress", 0.0),
                "cognitive_load": getattr(getattr(state, "condition", None), "cognitive_load", 0.2),
                "prediction_error": float(getattr(getattr(state, "condition", None), "prediction_error", 0.0))
            },
            appraisal_model_output=appraisal.to_dict(),
            temporal_model_input=temporal_in,
            temporal_model_output=temporal_out,
            deterministic_constraints_applied=constraints_applied,
            final_state_update=final_state_dict,
            attribution_provenance=provenance,
            uncertainty_and_evidence={
                "appraisal_confidence": getattr(appraisal, "confidence", 0.85),
                "epistemic_uncertainty": appraisal.uncertainty,
                "neural_active": self.temporal_neural_model is not None
            }
        )
        self.diagnostics_log.append(diag)
        if len(self.diagnostics_log) > 100:
            self.diagnostics_log.pop(0)

        # Commit updates to state
        state.momentary_affect = new_momentary
        state.baseline_mood = new_baseline
        state.last_appraisal_trace = appraisal.to_dict()
        state.advance_version(reason=reason_str)

        return state

    @staticmethod
    def interpret_affect_disposition(
        momentary: ContinuousAffectState,
        baseline: ContinuousAffectState
    ) -> str:
        """
        Derive an interpretable, nuanced affective disposition for logging and response strategy.
        Supports mixed states (e.g. frustrated yet engaged; uncertain yet calm).
        """
        v = momentary.valence
        a = momentary.arousal
        d = momentary.dominance
        u = momentary.uncertainty
        f = momentary.frustration
        eng = momentary.engagement

        if f > 0.4 and d >= 0.5:
            return "determined_resilience"  # Facing obstacle, but confident in coping
        elif f > 0.5 and d < 0.4:
            return "frustrated_blocked"     # Obstacle with low coping potential
        elif u > 0.5 and a > 0.5:
            return "vigilant_curious"       # High uncertainty + high activation
        elif v > 0.3 and eng > 0.6 and a >= 0.4:
            return "engaged_flow"           # Positive, high task immersion
        elif v > 0.4 and a < 0.4:
            return "calm_content"           # Satisfied, low arousal
        elif u > 0.5 and d < 0.4:
            return "cautious_uncertain"     # Unsure, requesting clarification
        elif v < -0.3 and a < 0.4:
            return "sympathetic_quiet"      # Shared sorrow / gentle quiet tone
        else:
            return "focused_operational"    # Balanced default state
