"""Integrated 13-Stage Cognitive Cycle Coordinator for E.D.I.T.H. V5.

Orchestrates the unified cognitive loop across user interactions, memory retrieval,
appraisal, affective dynamics, attention allocation, goal arbitration, metacognition,
execution observation, prediction-error tracking, and reflection.
"""

from __future__ import annotations
import os
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

from .cognitive_state import CognitiveState
from .appraisal import ContextualAppraisalEngine, AppraisalFrame
from .affective_dynamics import AffectiveDynamicsEngine
from .multi_memory import MultiSystemMemory
from .motivation_flow import MotivationAndFlowEngine, FlowStateEvaluation
from .attention import AdaptiveAttentionEngine
from .metacognition import MetacognitiveEngine, FailureDiagnosis
from .social_cognition import SocialCognitionEngine, EmpathicInference
from .resource_monitor import OperationalResourceMonitor
from .experience_recorder import ExperienceRecorder, FeedbackInterpreter, FeedbackSignal, FeedbackCategory
from .adaptive_communication import AdaptiveCommunicationEngine


@dataclass
class CognitiveCycleTrace:
    """Trace of a complete cognitive cycle execution."""
    cycle_id: str
    timestamp: float
    user_command: str
    appraisal: Dict[str, Any]
    momentary_affect: Dict[str, float]
    baseline_mood: Dict[str, float]
    disposition: str
    flow_evaluation: Dict[str, Any]
    empathic_inference: Dict[str, Any]
    attention_focus: Optional[str]
    retrieved_memory_count: int
    prediction_error: float
    calibrated_confidence: float
    reflection_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "timestamp": self.timestamp,
            "user_command": self.user_command,
            "appraisal": self.appraisal,
            "momentary_affect": self.momentary_affect,
            "baseline_mood": self.baseline_mood,
            "disposition": self.disposition,
            "flow_evaluation": self.flow_evaluation,
            "empathic_inference": self.empathic_inference,
            "attention_focus": self.attention_focus,
            "retrieved_memory_count": self.retrieved_memory_count,
            "prediction_error": self.prediction_error,
            "calibrated_confidence": self.calibrated_confidence,
            "reflection_notes": self.reflection_notes,
        }


class EdithCognitiveCycle:
    """Unified Cognitive Agent Controller integrating all sub-architectures."""

    SNAPSHOT_FILE = os.path.join("config", "cognitive_state_snapshot.json")

    def __init__(self, session_id: str = "edith_v5_main", memory_storage_dir: Optional[str] = None):
        self.session_id = session_id
        if memory_storage_dir:
            self.storage_dir = memory_storage_dir
            self.SNAPSHOT_FILE = os.path.join(memory_storage_dir, f"cognitive_snapshot_{session_id}.json")
            self.memory = MultiSystemMemory(storage_dir=memory_storage_dir)
        else:
            self.storage_dir = os.path.join("config", "cognitive_memory")
            self.SNAPSHOT_FILE = os.path.join("config", "cognitive_state_snapshot.json")
            self.memory = MultiSystemMemory()

        # 1. State Persistence: Restore state only if session matches to prevent cross-session contamination
        loaded_state = CognitiveState.load_snapshot(self.SNAPSHOT_FILE)
        if loaded_state and loaded_state.session_id == session_id:
            self.state = loaded_state
        else:
            self.state = CognitiveState(session_id=session_id)

        # 2. Optional Neural Appraisal System
        neural_model = None
        ckpt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "checkpoints", "cognitive_appraisal.pt")
        if os.path.exists(ckpt_path):
            try:
                from .learned_appraisal_model import CognitiveNeuralSystem
                neural_model = CognitiveNeuralSystem()
                neural_model.load_checkpoint(ckpt_path)
            except Exception:
                neural_model = None

        # 3. Cognitive Subsystems
        self.neural_model = neural_model
        self.appraisal_engine = ContextualAppraisalEngine(neural_appraisal_model=neural_model)
        self.affect_engine = AffectiveDynamicsEngine(temporal_neural_model=neural_model)
        self.motivation_flow = MotivationAndFlowEngine()
        self.attention = AdaptiveAttentionEngine()
        self.metacognition = MetacognitiveEngine()
        self.social_cognition = SocialCognitionEngine()
        self.resource_monitor = OperationalResourceMonitor()

        # 4. Experience-Based Learning & Communication Adaptation (V5.2)
        exp_dir = os.path.join(self.storage_dir, "experience_data")
        self.experience_recorder = ExperienceRecorder(storage_dir=exp_dir)
        self.feedback_interpreter = FeedbackInterpreter()
        comm_dir = os.path.join(self.storage_dir, "communication")
        self.adaptive_communication = AdaptiveCommunicationEngine(config_dir=comm_dir)

        # Audit history of completed cycles
        self.cycle_traces: List[CognitiveCycleTrace] = []

    # -------------------------------------------------------------------------
    # STAGE 1 - 8: Pre-Execution Cognitive Processing
    # -------------------------------------------------------------------------
    def pre_action_cycle(
        self,
        user_command: str,
        neural_prediction: Dict[str, Any],
        is_interruption: bool = False
    ) -> Dict[str, Any]:
        """
        Executes Cognitive Stages 1 through 8:
        1. Perceive -> 2. Interpret -> 3. Retrieve Memory -> 4. Update World
        -> 5. Appraise -> 6. Update Affect/Mood -> 7. Allocate Attention -> 8. Arbitrate/Decide.
        """
        t0 = time.time()
        notes: List[str] = []

        # 1. Perceive: Raw message and neural predictions
        raw_cmd = user_command.strip()
        intent = neural_prediction.get("intent", "unknown")
        action = neural_prediction.get("action", "noop")
        neural_conf = float(neural_prediction.get("confidence", 0.8))
        intent_prob = float(neural_prediction.get("intent_confidence", 0.8))
        entities = neural_prediction.get("entities", {})

        # 2. Interpret: Update situation understanding in state
        self.state.perception.current_interpretation = f"{intent} -> {action}"
        self.state.perception.entities = entities

        # 3. Retrieve Memory: Working memory, relevant episodes, semantic facts
        self.memory.set_working_item("last_user_query", raw_cmd, salience=0.9, item_type="evidence")
        relevant_episodes = self.memory.retrieve_relevant_episodes(raw_cmd, limit=3)
        if relevant_episodes:
            notes.append(f"Retrieved {len(relevant_episodes)} relevant episodic memories.")

        # Check for known procedural skill
        known_skill = self.memory.get_reliable_procedure(action)
        proc_reliability = known_skill.reliability if known_skill else None

        # 4. Update World & Goals
        is_dialogue = intent in ["chitchat", "dialogue"] or action == "respond_dialogue"
        if not is_dialogue and intent != "unsupported_ood":
            self.state.goals.active_goal = raw_cmd
            self.state.goals.priority = 0.7
        elif is_dialogue:
            self.state.goals.priority = 0.4

        # 5. Appraise: Context-conditioned cognitive appraisal
        event_repr = {
            "type": "interruption" if is_interruption else ("user_message" if is_dialogue else "task_request"),
            "text": raw_cmd,
            "intent": intent,
            "action": action,
            "confidence": neural_conf,
            "is_social": is_dialogue,
            "recovery_available": True
        }
        recent_ep_dicts = [e.context_tags for e in relevant_episodes]
        appraisal_frame = self.appraisal_engine.appraise_event(event_repr, self.state)

        # 6. Affective Dynamics: Update Momentary Affect and Baseline Mood
        self.affect_engine.update_affect(
            state=self.state,
            appraisal=appraisal_frame,
            event_summary=raw_cmd[:40],
            event_representation=event_repr,
            goal_context={
                "active_goal": self.state.goals.active_goal,
                "priority": self.state.goals.priority,
                "progress": self.state.goals.progress,
                "cognitive_load": getattr(self.state.condition, "cognitive_load", 0.2),
                "prediction_error": float(getattr(self.state.condition, "prediction_error", 0.0))
            }
        )
        disposition = self.affect_engine.interpret_affect_disposition(
            self.state.momentary_affect, self.state.baseline_mood
        )

        # Social perspective taking
        empathy_inf = self.social_cognition.analyze_social_context(raw_cmd, self.state)

        # 7. Calibrate Confidence
        missing_slots = [k for k, v in entities.items() if not v]
        calibrated_conf = self.metacognition.calibrate_confidence(
            neural_confidence=neural_conf,
            intent_prob=intent_prob,
            missing_slots=missing_slots,
            procedure_reliability=proc_reliability,
            operational_retry_count=self.state.operational.retry_count
        )
        self.state.condition.plan_confidence = calibrated_conf

        # 8. Allocate Attention & Flow State
        flow_eval = self.motivation_flow.evaluate_flow_state(self.state, recent_interruption=is_interruption)
        deep_delib, delib_reason = self.attention.should_trigger_deep_deliberation(self.state)
        self.state.condition.deep_reasoning_needed = deep_delib
        if deep_delib:
            notes.append(f"Deep deliberation triggered: {delib_reason}")

        # Set attention focus
        if is_interruption:
            self.state.condition.attention_focus = "user_interruption"
        elif self.state.in_flow_state:
            self.state.condition.attention_focus = "active_goal_flow"
        elif empathy_inf.detected_core_need == "emotional_validation":
            self.state.condition.attention_focus = "empathic_attunement"
        else:
            self.state.condition.attention_focus = f"task::{action}"

        # Record prior expectation for stage 10/11
        self.state.perception.prior_expectations = {
            "expected_outcome": True if calibrated_conf >= 0.55 else False,
            "calibrated_confidence": calibrated_conf,
            "action": action
        }

        # Build Cycle Trace Snapshot
        cycle_id = f"cycle_{int(time.time() * 1000)}"
        trace = CognitiveCycleTrace(
            cycle_id=cycle_id,
            timestamp=t0,
            user_command=raw_cmd,
            appraisal=appraisal_frame.to_dict(),
            momentary_affect=asdict(self.state.momentary_affect),
            baseline_mood=asdict(self.state.baseline_mood),
            disposition=disposition,
            flow_evaluation=asdict(flow_eval),
            empathic_inference=asdict(empathy_inf),
            attention_focus=self.state.condition.attention_focus,
            retrieved_memory_count=len(relevant_episodes),
            prediction_error=self.state.condition.prediction_error,
            calibrated_confidence=calibrated_conf,
            reflection_notes=notes,
        )
        self.cycle_traces.append(trace)
        if len(self.cycle_traces) > 100:
            self.cycle_traces.pop(0)

        # Adaptive Communication Plan (V5.2)
        comm_plan = self.adaptive_communication.determine_turn_communication_plan(raw_cmd, task_intent=intent)

        # Snapshot state
        self.state.save_snapshot(self.SNAPSHOT_FILE)

        return {
            "cycle_id": cycle_id,
            "calibrated_confidence": calibrated_conf,
            "disposition": disposition,
            "flow": flow_eval,
            "empathy": empathy_inf,
            "appraisal": appraisal_frame,
            "deep_deliberation": deep_delib,
            "communication_plan": comm_plan,
            "support_strategy": empathy_inf.suggested_support_strategy
        }

    # -------------------------------------------------------------------------
    # STAGE 9 - 13: Post-Execution Cognitive Processing & Reflection
    # -------------------------------------------------------------------------
    def post_action_cycle(
        self,
        user_command: str,
        action: str,
        execution_result: Dict[str, Any],
        verified: bool = True,
        latency_ms: float = 0.0
    ):
        """
        Executes Cognitive Stages 9 through 13:
        9. Execute (observed) -> 10. Inspect Outcome -> 11. Compare Predicted vs Observed
        -> 12. Update Beliefs & Consolidate Memory -> 13. Reflect on Failures / Success.
        """
        status = execution_result.get("status", "success")
        is_success = (status in ["success", "warning"])
        err_msg = execution_result.get("message", "")

        # 10. Inspect & 11. Compare: Prediction Error Tracking
        prior_expected = self.state.perception.prior_expectations.get("expected_outcome", True)
        pred_error = self.metacognition.track_prediction_error(
            state=self.state,
            expected_success=prior_expected,
            actual_success=is_success,
            verified=verified
        )

        # Update Operational Resource Monitor
        self.resource_monitor.update_operational_condition(
            state=self.state,
            last_latency_ms=latency_ms,
            action_failed=not is_success
        )

        # 12. Update Beliefs, Procedural Memory & Episodes
        lesson = None
        if not is_success:
            diagnosis = self.metacognition.diagnose_failure(
                step_type=action,
                target=execution_result.get("target", ""),
                error_message=err_msg,
                retry_count=self.state.operational.retry_count
            )
            lesson = f"{diagnosis.failure_category}: {diagnosis.recommended_recovery}"

        # Register procedural skill track record
        self.memory.register_procedure_outcome(
            skill_name=action,
            goal_pattern=self.state.goals.active_goal or action,
            action_sequence=[action],
            success=is_success and verified,
            failure_reason=err_msg if not is_success else None,
            recovery_strategy=lesson
        )

        # Record episodic memory trace
        self.memory.record_episode(
            goal=self.state.goals.active_goal,
            user_input=user_command,
            action_taken=action,
            outcome_status=status,
            verified=verified,
            prediction_error=pred_error,
            appraisal_congruence=0.8 if is_success else -0.7,
            lesson_learned=lesson,
            context_tags=[action, status]
        )

        # Log autobiographical trace
        self.memory.log_autobiographical_action(
            action=action,
            target=execution_result.get("target", "system"),
            result=status,
            verified=verified
        )

        # 13. Reflect, Appraise Outcome & Update Affect
        outcome_event_repr = {
            "type": "action_outcome",
            "text": f"Action {action} produced status {status}: {err_msg or 'ok'}",
            "intent": "action_outcome",
            "action": action,
            "is_success": is_success,
            "verified": verified,
            "prediction_error": pred_error,
            "recovery_available": (lesson is not None)
        }
        outcome_appraisal = self.appraisal_engine.appraise_event(outcome_event_repr, self.state)
        self.affect_engine.update_affect(
            state=self.state,
            appraisal=outcome_appraisal,
            event_summary=f"Outcome: {action}->{status}",
            event_representation=outcome_event_repr,
            goal_context={
                "active_goal": self.state.goals.active_goal,
                "goal_progress": self.state.goals.progress,
                "action": action,
                "verified": verified,
                "prediction_error": pred_error
            }
        )

        if is_success and verified:
            self.state.goals.progress = 1.0
            self.state.condition.error_accumulation = max(0.0, self.state.condition.error_accumulation - 0.2)
        else:
            self.state.goals.progress = max(0.0, self.state.goals.progress - 0.1)

        # 14. Interpret Feedback & Adapt Communication (V5.2)
        feedback_sig = self.feedback_interpreter.interpret_feedback(
            user_utterance=user_command,
            prior_action=action,
            prior_verified=verified
        )
        if feedback_sig.category == FeedbackCategory.STYLE_PREFERENCE:
            self.adaptive_communication.update_from_feedback(
                feedback_category=feedback_sig.category.value,
                feedback_details=feedback_sig.details
            )

        # 15. Record Interaction Event (Privacy-preserving, opt-in)
        resolved_int = self.state.perception.current_interpretation.split(" -> ")[0] if self.state.perception.current_interpretation else "task"
        self.experience_recorder.record_interaction(
            user_request=user_command,
            resolved_intent=resolved_int,
            resolved_action=action,
            entities=self.state.perception.entities,
            contextual_references={},
            prior_confidence=float(self.state.perception.prior_expectations.get("calibrated_confidence", 0.5)),
            calibrated_confidence=float(self.state.perception.prior_expectations.get("calibrated_confidence", 0.5)),
            predicted_plan=[action],
            execution_status=status,
            verified=verified,
            error_message=err_msg if not is_success else None,
            user_correction=user_command if feedback_sig.category == FeedbackCategory.EXPLICIT_CORRECTION else None,
            task_completed=(is_success and verified),
            influencing_procedure=action,
            observed_facts={"status": status, "verified": verified, "action": action},
            inferred_explanations={"prediction_error": pred_error, "lesson": lesson},
            latency_ms=latency_ms
        )

        # Persist memory consolidation and state snapshot
        self.memory.consolidate_and_save()
        self.state.save_snapshot(self.SNAPSHOT_FILE)

        return {
            "status": status,
            "prediction_error": round(pred_error, 4),
            "verified": verified,
            "lesson_learned": lesson,
            "disposition": self.affect_engine.interpret_affect_disposition(self.state.momentary_affect, self.state.baseline_mood),
            "feedback_signal": asdict(feedback_sig),
            "communication_preferences": self.adaptive_communication.preferences.to_dict()
        }

    def get_last_trace(self) -> Optional[Dict[str, Any]]:
        """Return the most recent cognitive cycle trace for diagnostics."""
        return self.cycle_traces[-1].to_dict() if self.cycle_traces else None
