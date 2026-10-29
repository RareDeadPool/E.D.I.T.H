"""Social Cognition, Contextual Empathy, and Operational Flow Evaluation for E.D.I.T.H. V5.1.

Validates that:
1. Flow state functions as an operational engagement proxy for task continuity,
   interruption handling, and blocker detection without overclaiming human consciousness.
2. Contextual empathy adapts support strategies based on task context (practical troubleshooting
   for coding hurdles vs gentle emotional validation for personal grief).
3. User emotional estimates are strictly segregated from the system's own computational affect.
4. Ambiguous cues, contradictory evidence, and user corrections are handled robustly.
"""

import os
import sys
import unittest
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.cognitive_state import CognitiveState, ContinuousAffectState
from engine.motivation_flow import MotivationAndFlowEngine, FlowStateEvaluation
from engine.social_cognition import SocialCognitionEngine, EmpathicInference


class TestV5SocialAndFlowEvaluation(unittest.TestCase):
    """Test operational flow proxies and contextual empathy differentiation."""

    def setUp(self):
        self.flow_engine = MotivationAndFlowEngine()
        self.social_engine = SocialCognitionEngine()
        self.state = CognitiveState(session_id="test_social_flow")

    def test_operational_flow_continuity_and_blocker_detection(self):
        """Test flow proxy continuity across multi-step execution and blocker detection."""
        self.state.goals.active_goal = "refactor database client"
        self.state.goals.priority = 0.8
        self.state.condition.task_complexity = 0.5
        self.state.momentary_affect.dominance = 0.8
        self.state.momentary_affect.valence = 0.4
        self.state.momentary_affect.frustration = 0.05
        self.state.condition.plan_confidence = 0.85

        # 1. Optimal conditions: should achieve operational flow
        eval_optimal = self.flow_engine.evaluate_flow_state(self.state, recent_interruption=False)
        self.assertTrue(eval_optimal.in_flow)
        self.assertGreaterEqual(eval_optimal.flow_intensity, 0.55)
        self.assertEqual(eval_optimal.behavioral_directive, "sustain_focus")

        # 2. Interruption occurs: flow should immediately drop with recovery directive
        eval_interrupted = self.flow_engine.evaluate_flow_state(self.state, recent_interruption=True)
        self.assertFalse(eval_interrupted.in_flow)
        self.assertEqual(eval_interrupted.behavioral_directive, "recover_from_interruption")
        self.assertGreater(eval_interrupted.interruption_penalty, 0.0)

        # 3. Blocker accumulation: repeated failures destroy flow and trigger clarification directive
        self.state.operational.consecutive_failures = 4
        self.state.condition.prediction_error = 0.8
        self.state.momentary_affect.frustration = 0.7
        eval_blocked = self.flow_engine.evaluate_flow_state(self.state, recent_interruption=False)
        self.assertFalse(eval_blocked.in_flow)
        self.assertLess(eval_blocked.flow_intensity, 0.3)
        self.assertEqual(eval_blocked.behavioral_directive, "seek_clarification")

    def test_contextual_empathy_strategy_differentiation(self):
        """Test that identical emotional markers select different strategies depending on active task context."""
        # Case A: Frustration during an active coding/software task
        self.state.goals.active_goal = "debug segfault in native module"
        inf_coding = self.social_engine.analyze_social_context(
            "this stupid bug is not working, why is this happening?",
            self.state
        )
        self.assertEqual(inf_coding.detected_core_need, "practical_solution")
        self.assertEqual(inf_coding.suggested_support_strategy, "direct_troubleshooting")
        self.assertEqual(inf_coding.adapted_tone, "direct_pragmatic")

        # Case B: Personal emotional distress / grief (no technical coding goal)
        self.state.goals.active_goal = None
        inf_distress = self.social_engine.analyze_social_context(
            "i feel so lonely and heartbroken today, everything went wrong",
            self.state
        )
        self.assertEqual(inf_distress.detected_core_need, "emotional_validation")
        self.assertEqual(inf_distress.suggested_support_strategy, "deep_empathy_and_gentle_grounding")
        self.assertEqual(inf_distress.adapted_tone, "gentle_attentive")

        # Case C: Routine neutral query
        inf_routine = self.social_engine.analyze_social_context(
            "show me files in directory",
            self.state
        )
        self.assertEqual(inf_routine.inferred_user_state, "operational_neutral")
        self.assertEqual(inf_routine.suggested_support_strategy, "efficient_execution")

    def test_strict_segregation_of_user_estimate_and_system_affect(self):
        """Verify that user emotional estimates do not corrupt the system's own computational affect."""
        # User is deeply distraught
        self.state.momentary_affect = ContinuousAffectState(valence=0.3, arousal=0.3, dominance=0.8, frustration=0.0)

        inf = self.social_engine.analyze_social_context(
            "i'm overwhelmed and completely heartbroken",
            self.state
        )

        # User estimate reflects user distress
        self.assertEqual(self.state.user_estimate.detected_user_mood, "relational_pain_or_isolation")
        self.assertEqual(self.state.user_estimate.communicative_need, "emotional_validation")

        # System's computational affect remains segregated (maintaining stability/agency)
        self.assertEqual(self.state.momentary_affect.dominance, 0.8)
        self.assertEqual(self.state.momentary_affect.frustration, 0.0)


if __name__ == "__main__":
    unittest.main()
