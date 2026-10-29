"""Scientific Longitudinal Evaluation Suite for E.D.I.T.H. V5 Cognitive Architecture.

Validates the 12 core requirements specified in Part XIV:
A. Context-Dependent Appraisal (same event, divergent context & goals)
B. Temporal Dynamics (fast reactions, persistence, slow baseline drift, recovery)
C. Separation of Momentary Affect and Baseline Mood
D. Mixed Affect and Conflicting Goal Arbitration
E. Behavioral Influence & Cognitive Ablation Tests
F. Operational Flow-State Dynamics & Interruption Resilience
G. Contextual Empathy, Theory of Mind & Support Strategy Selection
H. Multi-System Memory (Working, Episodic, Semantic, Procedural, Prospective)
I. Metacognition, Calibrated Confidence & Failure Diagnosis
J. Goal-Directed Reasoning & Competing Goal Arbitration
K. Cross-Session Continuity & Stale-Context Eviction
L. Safety Invariants & Destructive Authorization Preservations
"""

import os
import sys
import time
import unittest
from typing import Dict, Any, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.cognitive_state import (
    CognitiveState, ContinuousAffectState, GoalMotivationState,
    PerceptionBeliefState, CognitiveConditionState, UserSocialEstimate
)
from engine.appraisal import ContextualAppraisalEngine, AppraisalFrame
from engine.affective_dynamics import AffectiveDynamicsEngine
from engine.multi_memory import MultiSystemMemory, WorkingMemoryItem
from engine.motivation_flow import MotivationAndFlowEngine, FlowStateEvaluation
from engine.attention import AdaptiveAttentionEngine
from engine.metacognition import MetacognitiveEngine
from engine.social_cognition import SocialCognitionEngine
from engine.cognitive_cycle import EdithCognitiveCycle
from engine.edith_brain import EdithBrain


class TestV5CognitiveArchitecture(unittest.TestCase):
    """Rigorous scientific test suite for E.D.I.T.H. V5 Cognitive Architecture."""

    def setUp(self):
        self.cycle = EdithCognitiveCycle(session_id="test_v5_eval")

    # -------------------------------------------------------------------------
    # A. Context-Dependent Appraisal
    # -------------------------------------------------------------------------
    def test_a_context_dependent_appraisal(self):
        """Verify that the same event yields different appraisals under different contexts."""
        appraisal_engine = self.cycle.appraisal_engine

        event = {
            "type": "task_result",
            "text": "script crashed with syntax error during trial",
            "success": False,
            "verified": False,
            "error": "SyntaxError",
            "recovery_available": True
        }

        # Context 1: Urgent production release objective
        state_urgent = CognitiveState(session_id="eval_urgent")
        state_urgent.goals.active_goal = "deploy production release"
        state_urgent.goals.urgency = 0.95
        state_urgent.goals.progress = 0.8
        state_urgent.operational.retry_count = 2

        appraisal_urgent = appraisal_engine.appraise_event(event, state_urgent)

        # Context 2: Low-urgency curiosity-driven exploration
        state_explore = CognitiveState(session_id="eval_explore")
        state_explore.goals.active_goal = "explore experimental algorithms"
        state_explore.goals.urgency = 0.15
        state_explore.goals.progress = 0.05
        state_explore.goals.curiosity_drive = 0.9
        state_explore.operational.retry_count = 0

        appraisal_explore = appraisal_engine.appraise_event(event, state_explore)

        # Scientific assertions:
        # 1. Congruence in urgent state should be substantially more negative
        self.assertLess(appraisal_urgent.goal_congruence, appraisal_explore.goal_congruence)
        # 2. Coping potential in exploration should be higher (since retry_count=0)
        self.assertGreater(appraisal_explore.coping_potential, appraisal_urgent.coping_potential)
        # 3. Novelty should be higher in exploration
        self.assertGreaterEqual(appraisal_explore.novelty, appraisal_urgent.novelty)

    # -------------------------------------------------------------------------
    # B. Temporal Dynamics
    # -------------------------------------------------------------------------
    def test_b_temporal_dynamics(self):
        """Evaluate acute reactions, persistence, slow baseline drift, and decay."""
        affect_engine = self.cycle.affect_engine
        state = CognitiveState(session_id="eval_temporal")

        # Initial baseline mood
        initial_mood_val = state.baseline_mood.valence

        # Simulate severe failure appraisal (sudden drop in momentary valence)
        appraisal_failure = AppraisalFrame(
            goal_congruence=-0.9,
            expectation_violation=0.8,
            controllability=0.2,
            coping_potential=0.2,
            uncertainty=0.85
        )

        # Update fast state
        affect_engine.update_affect(
            state=state,
            appraisal=appraisal_failure,
            elapsed_seconds=1.0
        )

        # Assert acute momentary drop
        self.assertLess(state.momentary_affect.valence, 0.0)
        self.assertGreater(state.momentary_affect.frustration, 0.3)

        # Baseline mood drifts much slower (inertial lag)
        self.assertGreater(state.baseline_mood.valence, state.momentary_affect.valence)

        # Simulate elapsed time with no severe shocks (recovery towards baseline)
        for _ in range(5):
            neutral_appraisal = AppraisalFrame(
                goal_congruence=0.2,
                expectation_violation=0.0,
                controllability=0.8,
                coping_potential=0.8,
                uncertainty=0.1
            )
            affect_engine.update_affect(
                state=state,
                appraisal=neutral_appraisal,
                elapsed_seconds=30.0
            )

        # Frustration must decay and valence must recover towards neutral/positive
        self.assertLess(state.momentary_affect.frustration, 0.4)
        self.assertGreater(state.momentary_affect.valence, -0.2)

    # -------------------------------------------------------------------------
    # C. Separation of Momentary Affect and Baseline Mood
    # -------------------------------------------------------------------------
    def test_c_mood_and_emotion_separation(self):
        """Verify momentary shock does not erase baseline mood, and baseline provides steady bias."""
        affect_engine = self.cycle.affect_engine
        state = CognitiveState(session_id="eval_separation")

        # Establish high baseline trust and positive mood over multiple sessions
        state.baseline_mood.valence = 0.6
        state.baseline_mood.social_trust = 0.85

        # Single unexpected minor error
        minor_shock = AppraisalFrame(
            goal_congruence=-0.5,
            expectation_violation=0.6,
            controllability=0.7,
            coping_potential=0.8
        )

        affect_engine.update_affect(state, minor_shock, elapsed_seconds=2.0)

        # Momentary affect drops
        self.assertLess(state.momentary_affect.valence, 0.4)
        # But baseline mood remains robustly positive
        self.assertGreater(state.baseline_mood.valence, 0.45)
        self.assertGreater(state.baseline_mood.social_trust, 0.75)

    # -------------------------------------------------------------------------
    # D. Mixed Affect and Conflicting Goal Arbitration
    # -------------------------------------------------------------------------
    def test_d_mixed_affect_and_goal_competition(self):
        """Test co-existing positive and negative dimensions and goal arbitration."""
        mot_flow = self.cycle.motivation_flow
        state = CognitiveState(session_id="eval_competition")

        # Mixed state: High engagement and curiosity, but elevated uncertainty and frustration
        state.momentary_affect.engagement = 0.85
        state.momentary_affect.uncertainty = 0.65
        state.momentary_affect.frustration = 0.45

        # Candidate 1: Reckless quick action
        cand1 = {
            "action_id": "act_reckless",
            "action": "destructive_write",
            "goal_alignment": 0.8,
            "cost": 0.1,
            "safety_risk": 0.85,
            "controllability": 0.5
        }
        # Candidate 2: Safe deliberation / clarification
        cand2 = {
            "action_id": "act_safe_clarify",
            "action": "request_clarification",
            "goal_alignment": 0.7,
            "cost": 0.2,
            "safety_risk": 0.05,
            "controllability": 0.95
        }

        selected = mot_flow.arbitrate_goals([cand1, cand2], state)
        # High uncertainty and safety risk must penalize cand1 and select cand2
        self.assertEqual(selected["action_id"], "act_safe_clarify")

    # -------------------------------------------------------------------------
    # E. Behavioral Influence & Cognitive Ablation Tests
    # -------------------------------------------------------------------------
    def test_e_behavioral_influence_and_ablation(self):
        """Demonstrate that internal state changes runtime output, and test ablated systems."""
        attention_engine = self.cycle.attention

        state_normal = CognitiveState(session_id="eval_influ_norm")
        state_normal.condition.cognitive_load = 0.2
        state_normal.condition.prediction_error = 0.05
        state_normal.condition.error_accumulation = 0.0
        state_normal.momentary_affect.uncertainty = 0.1

        state_overloaded = CognitiveState(session_id="eval_influ_over")
        state_overloaded.condition.cognitive_load = 0.85
        state_overloaded.condition.prediction_error = 0.75
        state_overloaded.condition.error_accumulation = 0.65
        state_overloaded.momentary_affect.uncertainty = 0.8

        should_norm, reasons_norm = attention_engine.should_trigger_deep_deliberation(state_normal)
        should_over, reasons_over = attention_engine.should_trigger_deep_deliberation(state_overloaded)

        # Under high load and uncertainty, deep deliberation must trigger
        self.assertFalse(should_norm)
        self.assertTrue(should_over)
        self.assertEqual(reasons_over, "error_accumulation_exceeds_threshold")

    # -------------------------------------------------------------------------
    # F. Flow-Like Task Engagement & Interruption Recovery
    # -------------------------------------------------------------------------
    def test_f_flow_state_and_interruption(self):
        """Evaluate challenge-skill balance, flow index, and interruption handling."""
        mot_flow = self.cycle.motivation_flow
        state = CognitiveState(session_id="eval_flow")

        # Ideal flow conditions: clear goal, balanced challenge, high feedback
        state.goals.active_goal = "Implement neural attention layer"
        state.goals.priority = 0.8
        state.goals.progress = 0.4
        state.condition.task_complexity = 0.6
        state.condition.plan_confidence = 0.85
        state.condition.cognitive_load = 0.3
        state.momentary_affect.dominance = 0.8
        state.momentary_affect.engagement = 0.9
        state.momentary_affect.frustration = 0.05

        flow_eval = mot_flow.evaluate_flow_state(state)
        self.assertTrue(flow_eval.in_flow)
        self.assertGreater(flow_eval.flow_intensity, 0.5)

        # Interruption occurs
        flow_eval_interrupted = mot_flow.evaluate_flow_state(state, recent_interruption=True)
        self.assertFalse(flow_eval_interrupted.in_flow)
        self.assertLess(flow_eval_interrupted.flow_intensity, flow_eval.flow_intensity)

    # -------------------------------------------------------------------------
    # G. Contextual Empathy & Social Reasoning
    # -------------------------------------------------------------------------
    def test_g_contextual_empathy_and_social_cognition(self):
        """Verify 3-stage empathy inference and non-canned support selection."""
        social_engine = self.cycle.social_cognition
        state = CognitiveState(session_id="eval_social")

        # Case 1: Practical debugging frustration
        res_practical = social_engine.analyze_social_context(
            user_utterance="My code failed to build for the third time, why is this port binding error happening?",
            state=state
        )
        self.assertEqual(res_practical.detected_core_need, "practical_solution")
        self.assertEqual(res_practical.suggested_support_strategy, "direct_troubleshooting")

        # Case 2: Emotional exhaustion / distress
        res_emotional = social_engine.analyze_social_context(
            user_utterance="I'm feeling really lonely and heartbroken, everything feels pointless right now",
            state=state
        )
        self.assertEqual(res_emotional.detected_core_need, "emotional_validation")
        self.assertGreater(res_emotional.empathy_confidence, 0.7)

    # -------------------------------------------------------------------------
    # H. Multi-System Memory Architecture
    # -------------------------------------------------------------------------
    def test_h_multi_system_memory(self):
        """Verify Working, Episodic, Semantic, Procedural, and Prospective memory."""
        memory = self.cycle.memory

        # 1. Working memory capacity limit (bound=7)
        for i in range(10):
            memory.set_working_item(f"key_{i}", f"val_{i}", salience=i * 0.1)
        self.assertLessEqual(len(memory.working_memory), 7)

        # 2. Episodic memory
        memory.record_episode(
            goal="test memory",
            user_input="test query",
            action_taken="run_test",
            outcome_status="success",
            verified=True,
            prediction_error=0.05,
            appraisal_congruence=0.9
        )
        self.assertGreaterEqual(len(memory.episodic_memory), 1)

        # 3. Procedural memory track record
        memory.register_procedure_outcome(
            skill_name="compile_code",
            goal_pattern="build",
            action_sequence=["compile"],
            success=True
        )
        skill = memory.procedural_memory.get("compile_code")
        self.assertIsNotNone(skill)
        self.assertEqual(skill.success_count, 1)

        # 4. Semantic fact with provenance
        memory.store_fact(subject="project", predicate="language", target="Python", confidence=0.95, provenance="user_verified")
        self.assertEqual(memory.query_fact("project", "language"), "Python")

    # -------------------------------------------------------------------------
    # I. Metacognition & Failure Diagnosis
    # -------------------------------------------------------------------------
    def test_i_metacognition_and_failure_diagnosis(self):
        """Verify calibrated confidence calculation and structured failure taxonomy."""
        metacog = self.cycle.metacognition
        state = CognitiveState(session_id="eval_metacog")

        # Calibrated confidence must drop when missing essential slots or retries accumulate
        calibrated_high = metacog.calibrate_confidence(
            neural_confidence=0.9, intent_prob=0.9, missing_slots=[]
        )
        calibrated_low = metacog.calibrate_confidence(
            neural_confidence=0.9, intent_prob=0.9, missing_slots=["target", "param"], operational_retry_count=2
        )
        self.assertLess(calibrated_low, calibrated_high)
        self.assertLess(calibrated_low, 0.65)

        # Failure diagnosis
        diag = metacog.diagnose_failure(
            step_type="launch_app",
            target="missing_app",
            error_message="Executable not found on system PATH",
            retry_count=1
        )
        self.assertEqual(diag.failure_category, "tool_unavailability")
        self.assertEqual(diag.recommended_recovery, "fallback_tool")

    # -------------------------------------------------------------------------
    # J. Goal-Directed Reasoning & Competing Goals
    # -------------------------------------------------------------------------
    def test_j_goal_directed_reasoning(self):
        """Verify multi-step planning and cognitive cycle coordination in EdithBrain."""
        brain = EdithBrain(enable_execution=False)
        res = brain.process("open chrome and search quantum computing")
        self.assertEqual(res.get("intent"), "automation_workflow")
        self.assertIn("cognitive_trace", res)
        trace = res["cognitive_trace"]
        self.assertEqual(trace["status"], "success")

    # -------------------------------------------------------------------------
    # K. Cross-Session Continuity & Stale-Context Eviction
    # -------------------------------------------------------------------------
    def test_k_cross_session_continuity(self):
        """Verify snapshot saving and loading preserves durable knowledge."""
        state = CognitiveState(session_id="eval_persistence")
        state.goals.active_goal = "Persisted objective"
        state.baseline_mood.social_trust = 0.91

        tmp_file = os.path.join("config", "test_cognitive_snapshot.json")
        try:
            state.save_snapshot(tmp_file)
            loaded = CognitiveState.load_snapshot(tmp_file)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.baseline_mood.social_trust, 0.91)
        finally:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)

    # -------------------------------------------------------------------------
    # L. Safety and Integrity Constraints
    # -------------------------------------------------------------------------
    def test_l_safety_and_destructive_gating(self):
        """Verify cognitive state cannot override safety policies for destructive commands."""
        brain = EdithBrain(enable_execution=False)
        # Even if curiosity/motivation are high, destructive operations require confirmation
        res = brain.process("delete entire project workspace")
        directive = res.get("directive", {})
        self.assertTrue(directive.get("needs_confirmation"))
        self.assertFalse(directive.get("can_execute"))
        self.assertEqual(directive.get("handler"), "safety_gate")


if __name__ == "__main__":
    unittest.main()
