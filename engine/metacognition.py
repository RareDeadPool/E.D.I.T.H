"""Metacognitive Self-Monitoring, Confidence Calibration and Failure Diagnosis for E.D.I.T.H. V5.

Implements:
1. Knowledge State Epistemics: Distinguishes Known, Believed, Inferred, Unverified, and Unknown.
2. Calibrated Confidence: Integrates neural predictions, entity completeness, tool availability,
   and procedural reliability history (no hardcoded 100% confidence).
3. Prediction Error Tracking: Quantifies expected vs observed discrepancies.
4. Structured Failure Diagnosis: Classifies failures to select targeted recovery strategies.
"""

from __future__ import annotations
import math
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from .cognitive_state import CognitiveState


@dataclass
class EpistemicBeliefReport:
    """Explicit taxonomy of agent's knowledge regarding a topic/query."""
    known_facts: List[str] = field(default_factory=list)
    believed_inferences: List[Dict[str, Any]] = field(default_factory=list)
    unverified_claims: List[str] = field(default_factory=list)
    explicit_unknowns: List[str] = field(default_factory=list)
    calibration_score: float = 0.85


@dataclass
class FailureDiagnosis:
    """Structured diagnosis of an execution or planning failure."""
    failure_category: str  # 'understanding_error', 'missing_information', 'strategy_defect', 'tool_unavailability', 'execution_error', 'environment_change'
    root_cause_explanation: str
    recommended_recovery: str  # 'ask_user_clarification', 'fallback_tool', 'retry_with_backoff', 'abort_and_notify'
    recovery_confidence: float


class MetacognitiveEngine:
    """Self-monitoring and reflective diagnostic engine for E.D.I.T.H."""

    def __init__(self):
        self.error_history: List[float] = []

    def calibrate_confidence(
        self,
        neural_confidence: float,
        intent_prob: float,
        missing_slots: List[str],
        procedure_reliability: Optional[float] = None,
        operational_retry_count: int = 0
    ) -> float:
        """
        Compute a well-calibrated confidence score combining model certainty,
        slot completeness, procedural success history, and operational retries.
        """
        # Baseline from neural prediction
        base = max(0.1, min(0.98, (neural_confidence + intent_prob) / 2.0))

        # Penalty for missing essential slots
        slot_penalty = len(missing_slots) * 0.25

        # Reward / penalty from procedural memory track record
        proc_factor = procedure_reliability if procedure_reliability is not None else 0.8

        # Penalty for repeated retries on same task
        retry_penalty = min(0.6, operational_retry_count * 0.2)

        raw = base * proc_factor - slot_penalty - retry_penalty
        calibrated = max(0.05, min(0.99, raw))

        return round(calibrated, 3)

    def track_prediction_error(
        self,
        state: CognitiveState,
        expected_success: bool,
        actual_success: bool,
        verified: bool = True
    ) -> float:
        """
        Compare predicted outcome with actual observation.
        Updates prediction_error and error_accumulation in cognitive state.
        """
        exp_val = 1.0 if expected_success else 0.0
        act_val = 1.0 if (actual_success and verified) else (0.4 if actual_success else 0.0)

        error_magnitude = abs(exp_val - act_val)

        # Update state condition
        state.condition.prediction_error = round(error_magnitude, 3)

        # Cumulative error tracking with decay
        current_accum = state.condition.error_accumulation
        new_accum = current_accum * 0.7 + error_magnitude * 0.5
        state.condition.error_accumulation = round(min(1.0, new_accum), 3)

        self.error_history.append(error_magnitude)
        if len(self.error_history) > 50:
            self.error_history.pop(0)

        return error_magnitude

    def diagnose_failure(
        self,
        step_type: str,
        target: str,
        error_message: str,
        structured_understanding: Optional[Dict[str, Any]] = None,
        tool_available: bool = True,
        retry_count: int = 0
    ) -> FailureDiagnosis:
        """
        Classify failure mode into actionable cognitive categories:
        - understanding_error: neural model extracted wrong entities or intent.
        - missing_information: parameters missing or ambiguous.
        - tool_unavailability: application or executable not found on disk.
        - strategy_defect: repeated failure using same approach.
        - execution_error: transient OS/permissions/timeout failure.
        - environment_change: target file moved or window closed.
        """
        err_lower = error_message.lower()

        # 1. Tool / App Unavailability
        if any(w in err_lower for w in ["not found", "cannot find", "no such file", "unrecognized application", "not installed"]):
            return FailureDiagnosis(
                failure_category="tool_unavailability",
                root_cause_explanation=f"Target executable or file '{target}' is not accessible on the host filesystem.",
                recommended_recovery="fallback_tool",
                recovery_confidence=0.75
            )

        # 2. Missing Information / Ambiguity
        if any(w in err_lower for w in ["missing", "ambiguous", "which one", "clarification required", "no recipient"]):
            return FailureDiagnosis(
                failure_category="missing_information",
                root_cause_explanation="Essential parameters required to proceed are absent or ambiguous.",
                recommended_recovery="ask_user_clarification",
                recovery_confidence=0.90
            )

        # 3. Strategy Defect (Repeated failures)
        if retry_count >= 2:
            return FailureDiagnosis(
                failure_category="strategy_defect",
                root_cause_explanation=f"Strategy '{step_type}' failed across {retry_count} consecutive attempts; plan is likely defective.",
                recommended_recovery="abort_and_notify",
                recovery_confidence=0.85
            )

        # 4. Environment / Window State Change
        if any(w in err_lower for w in ["timed out", "unresponsive", "hung", "access denied", "permission denied"]):
            return FailureDiagnosis(
                failure_category="execution_error",
                root_cause_explanation=f"Operating system returned permission, timeout, or responsiveness error: {error_message}",
                recommended_recovery="retry_with_backoff",
                recovery_confidence=0.60
            )

        # Default fallback: Understanding / Strategy mismatch
        return FailureDiagnosis(
            failure_category="understanding_error",
            root_cause_explanation=f"Action '{step_type}' on '{target}' yielded unexpected error: {error_message}",
            recommended_recovery="ask_user_clarification",
            recovery_confidence=0.70
        )
