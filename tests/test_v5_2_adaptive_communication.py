"""Unit and Integration Tests for Adaptive Communication & Evidence-Based Empathy (V5.2).

Validates:
1. Communication style adaptation from user feedback (verbosity, directness, autonomy).
2. Prevention of single-interaction permanent lock-in via bounded deltas (0.20) and temporal decay.
3. Repeated feedback reinforcement up to ceiling (0.95).
4. Dynamic per-turn explicit instruction overrides ('explain in detail this time').
5. Nuanced social cognition distinguishing:
   - Seeking practical advice vs seeking emotional support vs concise answers
   - Space to vent vs active problem-solving
   - Frustration with tool vs frustration with coworker
   - Sarcasm/irony vs genuine praise
   - Mixed ambivalent emotions
   - Weak evidence neutral fallback
6. Strict segregation between estimated user state and agent computational affective state.
"""

import os
import sys
import unittest
import tempfile
import shutil
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.cognitive_state import CognitiveState
from engine.adaptive_communication import AdaptiveCommunicationEngine, CommunicationStylePreferences
from engine.social_cognition import SocialCognitionEngine, EmpathicInference


class TestV52AdaptiveCommunication(unittest.TestCase):
    """Test suite for adaptive communication and nuanced evidence-based social cognition."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.comm_engine = AdaptiveCommunicationEngine(config_dir=self.test_dir)
        self.social_engine = SocialCognitionEngine(personality_config_path=os.path.join(self.test_dir, "personality.json"))
        self.state = CognitiveState(session_id="test_comm_session")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_communication_style_adaptation_and_no_lock_in(self):
        """Verify that style adapts incrementally without permanently locking in from a single feedback."""
        # Initial defaults
        self.assertEqual(self.comm_engine.preferences.verbosity, "balanced")
        self.assertEqual(self.comm_engine.preferences.verbosity_confidence, 0.5)

        # Single feedback: "Please be more concise"
        updated = self.comm_engine.update_from_feedback("style_preference", {"preference": "concise"})
        self.assertTrue(updated)
        self.assertEqual(self.comm_engine.preferences.verbosity, "concise")
        # Confidence must be bounded, NOT locked to 1.0
        self.assertEqual(self.comm_engine.preferences.verbosity_confidence, 0.60)

        # Temporal decay: simulate 5 days passing
        self.comm_engine.preferences.last_updated = time.time() - (5 * 86400.0)
        self.comm_engine._apply_temporal_decay(decay_rate_per_day=0.05)
        self.assertLess(self.comm_engine.preferences.verbosity_confidence, 0.60, "Inactive preferences must decay toward baseline")

    def test_repeated_feedback_reinforcement(self):
        """Verify that repeated consistent feedback progressively strengthens preference up to ceiling."""
        for _ in range(5):
            self.comm_engine.update_from_feedback("style_preference", {"preference": "concise"})

        self.assertEqual(self.comm_engine.preferences.verbosity, "concise")
        self.assertGreaterEqual(self.comm_engine.preferences.verbosity_confidence, 0.90)
        self.assertLessEqual(self.comm_engine.preferences.verbosity_confidence, 0.95, "Confidence must not exceed safety ceiling 0.95")

    def test_per_turn_override(self):
        """Verify that explicit per-turn requests override baseline preference without corrupting long-term profile."""
        # Long term preference is concise
        self.comm_engine.preferences.verbosity = "concise"
        self.comm_engine.preferences.verbosity_confidence = 0.90

        # Current request explicitly asks for detailed explanation
        plan = self.comm_engine.determine_turn_communication_plan("Explain the database schema in detail this time")
        self.assertEqual(plan["verbosity"], "detailed_technical")
        self.assertEqual(plan["directness"], "step_by_step")
        self.assertTrue(plan["override_applied"])

        # Long-term preference must remain untouched
        self.assertEqual(self.comm_engine.preferences.verbosity, "concise")

    def test_nuanced_social_cognition_situations(self):
        """Verify distinct communicative stance classification across 7 nuanced interaction scenarios."""
        # 1. Concise Demand
        inf_concise = self.social_engine.analyze_social_context("Just give me the IP address, no explanation", self.state)
        self.assertEqual(inf_concise.detected_core_need, "concise_direct_answer")
        self.assertEqual(inf_concise.suggested_support_strategy, "concise_direct_answer")

        # 2. Venting Space Needed
        inf_vent = self.social_engine.analyze_social_context("I just need to vent about what happened in the meeting today...", self.state)
        self.assertEqual(inf_vent.detected_core_need, "space_to_vent")
        self.assertEqual(inf_vent.suggested_support_strategy, "reflective_listening")

        # 3. Frustration with Tool vs Frustration with Person
        inf_tool = self.social_engine.analyze_social_context("The compiler error makes no sense, connection refused on port 5432", self.state)
        self.assertEqual(inf_tool.inferred_user_state, "task_blocked_frustration")
        self.assertEqual(inf_tool.suggested_support_strategy, "direct_troubleshooting")

        inf_person = self.social_engine.analyze_social_context("My manager promised to review the code and didn't do their part", self.state)
        self.assertEqual(inf_person.inferred_user_state, "interpersonal_friction")
        self.assertEqual(inf_person.suggested_support_strategy, "reflective_listening_then_advice")

        # 4. Sarcasm / Irony
        inf_sarcasm = self.social_engine.analyze_social_context("Oh great, another error, exactly what I needed today!", self.state)
        self.assertEqual(inf_sarcasm.inferred_user_state, "sarcastic_frustration")
        self.assertEqual(inf_sarcasm.suggested_support_strategy, "direct_troubleshooting")

        # 5. Mixed Conflicted Emotions
        inf_mixed = self.social_engine.analyze_social_context("I'm excited about the new job offer but terrified of relocating", self.state)
        self.assertEqual(inf_mixed.inferred_user_state, "ambivalent_uncertainty")
        self.assertEqual(inf_mixed.suggested_support_strategy, "validate_both_sides_collaboratively")

        # 6. Weak Evidence Neutral Fallback
        inf_neutral = self.social_engine.analyze_social_context("What time is the team sync today?", self.state)
        self.assertEqual(inf_neutral.inferred_user_state, "operational_neutral")
        self.assertEqual(inf_neutral.empathy_confidence, 0.60, "Weak evidence must not produce overconfident emotional guesses")

    def test_strict_state_segregation(self):
        """Verify that user social estimates do not alter E.D.I.T.H.'s own computational PAD state."""
        initial_valence = self.state.momentary_affect.valence
        initial_arousal = self.state.momentary_affect.arousal

        self.social_engine.analyze_social_context("I'm so heartbroken and crying, my partner left me", self.state)

        # User estimate must be populated
        self.assertEqual(self.state.user_estimate.detected_user_mood, "relational_pain_or_isolation")

        # But E.D.I.T.H.'s own internal computational affect must be strictly unpolluted
        self.assertEqual(self.state.momentary_affect.valence, initial_valence)
        self.assertEqual(self.state.momentary_affect.arousal, initial_arousal)


if __name__ == "__main__":
    unittest.main()
