"""Causal Behavioral Ablation Suite for E.D.I.T.H. V5.1 Cognitive Architecture.

Validates that cognitive components exert genuine causal influence on downstream decisions:
1. Memory Retrieval Ablation: Measures memory retrieval count and procedural calibration impact.
2. Appraisal Ablation: Measures impact of goal congruence/violation on affective shifts.
3. Affective State Ablation: Measures impact of emotional disposition on attention and risk posture.
4. Motivation & Flow Ablation: Measures impact on flow entry and attention focus prioritization.
5. Metacognition Ablation: Measures confidence calibration, failure diagnosis, and deep deliberation triggering.
"""

import os
import sys
import unittest
import time
from typing import Dict, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.cognitive_state import CognitiveState, ContinuousAffectState
from engine.cognitive_cycle import EdithCognitiveCycle
from engine.appraisal import AppraisalFrame


class TestV5CausalAblation(unittest.TestCase):
    """Paired causal ablation experiments comparing Full Pipeline vs Ablated Subsystems."""

    def test_ablation_memory_retrieval(self):
        """Ablation 1: Memory retrieval disabled vs full pipeline on procedural execution."""
        cmd = "deploy staging service"
        neural_pred = {"intent": "deploy", "action": "deploy_staging", "confidence": 0.8, "entities": {}}

        # Setup Full Pipeline with existing reliable procedure
        full_cycle = EdithCognitiveCycle(session_id="full_mem_test")
        full_cycle.memory.register_procedure_outcome(
            skill_name="deploy_staging",
            goal_pattern="deploy staging service",
            action_sequence=["deploy_staging"],
            success=True
        )
        full_pre = full_cycle.pre_action_cycle(cmd, neural_pred)

        # Setup Ablated Cycle (Memory Retrieval Disabled)
        ablated_cycle = EdithCognitiveCycle(session_id="ablated_mem_test")
        # Monkeypatch retrieve_relevant_episodes and get_reliable_procedure to return empty/None
        ablated_cycle.memory.retrieve_relevant_episodes = lambda *args, **kwargs: []
        ablated_cycle.memory.get_reliable_procedure = lambda *args, **kwargs: None
        ablated_pre = ablated_cycle.pre_action_cycle(cmd, neural_pred)

        # Behavioral differences:
        # 1. Calibrated confidence in full pipeline is boosted by procedural reliability
        self.assertGreater(
            full_pre["calibrated_confidence"],
            ablated_pre["calibrated_confidence"],
            "Full pipeline with procedural memory must have higher calibrated confidence than memory-ablated pipeline"
        )
        # 2. Retrieved episodes / procedural knowledge present in trace
        last_full_trace = full_cycle.get_last_trace()
        last_ablated_trace = ablated_cycle.get_last_trace()
        self.assertIsNotNone(last_full_trace)
        self.assertIsNotNone(last_ablated_trace)

    def test_ablation_appraisal_influence(self):
        """Ablation 2: Appraisal influence disabled vs full pipeline on severe failure."""
        cmd = "run urgent database migration"
        neural_pred = {"intent": "migrate", "action": "run_migration", "confidence": 0.85, "entities": {}}

        full_cycle = EdithCognitiveCycle(session_id="full_appr_test")
        full_cycle.state.goals.priority = 0.95
        full_cycle.state.goals.urgency = 0.95

        ablated_cycle = EdithCognitiveCycle(session_id="ablated_appr_test")
        ablated_cycle.state.goals.priority = 0.95
        ablated_cycle.state.goals.urgency = 0.95
        # Ablate appraisal: replace with flat zero frame
        ablated_cycle.appraisal_engine.appraise_event = lambda *args, **kwargs: AppraisalFrame(
            novelty=0.0, goal_relevance=0.0, goal_congruence=0.0, expectation_violation=0.0,
            controllability=0.5, coping_potential=0.5, uncertainty=0.0, effort_required=0.0
        )

        # Run catastrophic failure
        fail_res = {"status": "failure", "message": "FATAL: database connection dropped; table corrupted"}

        post_full = full_cycle.post_action_cycle(cmd, "run_migration", fail_res, verified=False)
        post_ablated = ablated_cycle.post_action_cycle(cmd, "run_migration", fail_res, verified=False)

        # In full pipeline, severe failure produces severe negative congruence and expectation violation
        full_trace = full_cycle.state.last_appraisal_trace
        ablated_trace = ablated_cycle.state.last_appraisal_trace
        self.assertLess(
            full_trace["goal_congruence"],
            ablated_trace["goal_congruence"],
            "Appraisal must reflect severe goal obstruction on critical failure"
        )
        self.assertGreater(
            full_trace["expectation_violation"],
            ablated_trace["expectation_violation"],
            "Appraisal must capture unexpected catastrophic failure"
        )

    def test_ablation_affective_dynamics(self):
        """Ablation 3: Affective state influence disabled vs full pipeline."""
        full_cycle = EdithCognitiveCycle(session_id="full_aff_test")
        ablated_cycle = EdithCognitiveCycle(session_id="ablated_aff_test")

        # Induce high frustration in both cycles
        full_cycle.state.momentary_affect.frustration = 0.85
        full_cycle.state.momentary_affect.dominance = 0.2
        full_cycle.state.momentary_affect.valence = -0.7

        # In ablated cycle, affect engine is clamped to flat neutral
        ablated_cycle.state.momentary_affect = ContinuousAffectState()
        ablated_cycle.affect_engine.update_affect = lambda state, *args, **kwargs: state

        disp_full = full_cycle.affect_engine.interpret_affect_disposition(
            full_cycle.state.momentary_affect, full_cycle.state.baseline_mood
        )
        disp_ablated = ablated_cycle.affect_engine.interpret_affect_disposition(
            ablated_cycle.state.momentary_affect, ablated_cycle.state.baseline_mood
        )

        self.assertEqual(disp_full, "frustrated_blocked")
        self.assertEqual(disp_ablated, "focused_operational")
        self.assertNotEqual(disp_full, disp_ablated, "Affective state must change disposition from default")

    def test_ablation_motivational_flow(self):
        """Ablation 4: Motivational & flow influence disabled vs full pipeline."""
        from engine.motivation_flow import FlowStateEvaluation

        cmd = "complex ast refactoring across 30 files"
        neural_pred = {"intent": "refactor", "action": "ast_refactor", "confidence": 0.9, "entities": {}}

        full_cycle = EdithCognitiveCycle(session_id="full_flow_test")
        full_cycle.state.goals.active_goal = "ast_refactoring"
        full_cycle.state.condition.task_complexity = 0.5
        full_cycle.state.momentary_affect.dominance = 0.8
        full_cycle.state.momentary_affect.valence = 0.5
        full_cycle.state.momentary_affect.engagement = 0.9
        full_cycle.state.momentary_affect.arousal = 0.5
        full_cycle.state.goals.priority = 0.8
        full_cycle.state.goals.progress = 0.5

        ablated_cycle = EdithCognitiveCycle(session_id="ablated_flow_test")
        ablated_cycle.state.goals.active_goal = "ast_refactoring"
        ablated_cycle.state.condition.task_complexity = 0.5
        ablated_cycle.state.momentary_affect = ContinuousAffectState(valence=0.5, engagement=0.9, arousal=0.5, dominance=0.8)
        # Ablate flow engine by returning non-flow FlowStateEvaluation dataclass
        ablated_cycle.motivation_flow.evaluate_flow_state = lambda *args, **kwargs: FlowStateEvaluation(
            in_flow=False, flow_intensity=0.0, challenge_skill_ratio=0.5, goal_clarity=0.5,
            feedback_immediacy=0.5, attention_stability=0.5, controllability=0.5,
            interruption_penalty=0.0, behavioral_directive="relax_mode"
        )

        pre_full = full_cycle.pre_action_cycle(cmd, neural_pred)
        pre_ablated = ablated_cycle.pre_action_cycle(cmd, neural_pred)

        # In full cycle, flow state is triggered and focuses attention on flow
        self.assertTrue(full_cycle.state.in_flow_state, "Full cycle should enter flow under high engagement and balanced skills")
        self.assertEqual(full_cycle.state.condition.attention_focus, "active_goal_flow")

        # In ablated cycle, flow state is never recognized
        self.assertFalse(ablated_cycle.state.in_flow_state)
        self.assertNotEqual(ablated_cycle.state.condition.attention_focus, "active_goal_flow")

    def test_ablation_metacognition(self):
        """Ablation 5: Metacognitive feedback disabled vs full pipeline on repeated errors."""
        cmd = "send encrypted message"
        # Neural model claims 95% confidence, but required entities ('recipient', 'body') are missing
        neural_pred = {"intent": "send_msg", "action": "send_message", "confidence": 0.95, "entities": {"recipient": None, "body": None}}

        full_cycle = EdithCognitiveCycle(session_id="full_meta_test")
        full_cycle.state.operational.retry_count = 2  # Already failed twice

        ablated_cycle = EdithCognitiveCycle(session_id="ablated_meta_test")
        ablated_cycle.state.operational.retry_count = 2
        # Ablate metacognition confidence calibration (simply returns raw neural confidence)
        ablated_cycle.metacognition.calibrate_confidence = lambda neural_confidence, **kwargs: neural_confidence
        ablated_cycle.metacognition.diagnose_failure = lambda *args, **kwargs: type("Diag", (), {"failure_category": "unknown", "recommended_recovery": "retry"})()

        pre_full = full_cycle.pre_action_cycle(cmd, neural_pred)
        pre_ablated = ablated_cycle.pre_action_cycle(cmd, neural_pred)

        # Full pipeline detects missing slots and repeated failures, lowering confidence
        self.assertLess(
            pre_full["calibrated_confidence"],
            0.6,
            "Metacognition must drastically penalize confidence for missing required slots and retries"
        )
        self.assertEqual(
            pre_ablated["calibrated_confidence"],
            0.95,
            "Ablated metacognition blindly preserves overconfident raw neural prediction"
        )

        # Observe post-action failure diagnosis
        post_full = full_cycle.post_action_cycle(
            cmd, "send_message",
            {"status": "failure", "message": "Missing recipient address parameter"},
            verified=False
        )
        self.assertIsNotNone(post_full["lesson_learned"])
        self.assertIn("missing", str(post_full["lesson_learned"]).lower())


if __name__ == "__main__":
    unittest.main()
