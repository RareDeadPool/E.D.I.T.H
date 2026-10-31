"""
Permanent Regression Suite: tests/test_v14_transcript_replay.py
Reproduces the 7 microphone session failures verbatim from Section 1 of the E.D.I.T.H. V14 specification.
Verifies that all 7 scenarios strictly adhere to the Output Contract and pass the Leak Linter.
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.edith_brain import EdithBrain
from engine.mind_router import get_mind_router
from engine.output_contract import LeakLinter, TTSSanitizer, speak
from engine.personality import EdithPersonality


class TestV14TranscriptReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.brain = EdithBrain(enable_execution=False)
        cls.mind = get_mind_router(backend="edith_lm")

        # Seed known user facts for memory evaluation
        cls.mind.memory.add_fact("Harshan", "relationship", "friend")
        cls.mind.memory.add_fact("Aditya", "preference", "dark mode")

    def assert_no_leaks(self, text: str):
        """Asserts that text passes the non-negotiable LeakLinter without any internal leaks."""
        self.assertTrue(len(text) > 0, "Spoken text must not be empty.")
        is_clean, violations = LeakLinter.lint(text)
        self.assertTrue(
            is_clean,
            f"LeakLinter detected internal terminology in user-facing speech: {violations} | Text: '{text}'"
        )

    # -------------------------------------------------------------------------
    # Scenario 1: "I'll allow you" (STT mishear / conversational ambiguity)
    # -------------------------------------------------------------------------
    def test_scenario_1_ill_allow_you_mishear(self):
        """1. 'I'll allow you' -> Must NOT execute system_utility. Must ask human clarification."""
        query = "I'll allow you"
        res = self.brain.process(query, execute=False)

        directive = res["directive"]
        msg = directive["message"]

        # Property checks
        self.assertFalse(directive.get("can_execute", False), "Ambiguous utterance must not execute system actions.")
        self.assertNotEqual(res.get("action"), "exec_system_utility")
        self.assertNotIn("system_utility", msg.lower())

        # Must recognize as likely mishear or conversational ambiguity
        self.assertIn("allow you", msg.lower())
        self.assertTrue(
            any(w in msg.lower() for w in ["mean", "clarify", "catch", "caught"]),
            f"Expected clarification in response, got: '{msg}'"
        )

        # Output Contract check
        self.assert_no_leaks(msg)

    # -------------------------------------------------------------------------
    # Scenario 2: "How are you" (Dynamic, varied, non-canned dialogue)
    # -------------------------------------------------------------------------
    def test_scenario_2_how_are_you_dynamic(self):
        """2. 'How are you' -> Must never be the canned 'Running at peak performance…'. Must vary."""
        query = "How are you"

        replies = set()
        for _ in range(5):
            res = self.brain.process(query, execute=False)
            msg = res["directive"]["message"]

            # Must never be the canned V12 f-string
            self.assertNotIn("running at peak performance", msg.lower())
            self.assertNotIn("all good on my end", msg.lower())

            # Output Contract check
            self.assert_no_leaks(msg)
            replies.add(msg)

        # Across 5 runs, the reply must show generative variation
        self.assertGreater(len(replies), 1, "Replies to 'How are you' must vary and not be identical static strings.")

    # -------------------------------------------------------------------------
    # Scenario 3: "Subscribe me" (Twice: Low confidence execution block & Repetition adaptation)
    # -------------------------------------------------------------------------
    def test_scenario_3_subscribe_me_low_confidence_and_repetition(self):
        """
        3. 'Subscribe me' (twice) ->
           Turn 1: Low confidence (e.g. 25.8%) must NEVER execute. Ask clarifying question.
           Turn 2: Detect repetition and change strategy. Never leak context_control or unbound handler.
        """
        # Turn 1:
        res1 = self.brain.process("Subscribe me", execute=False)
        dir1 = res1["directive"]
        msg1 = dir1["message"]

        self.assertFalse(dir1.get("can_execute", False), "Low confidence action must never execute.")
        self.assertNotEqual(dir1.get("handler"), "context_control")
        self.assertNotIn("contextual control resolved", msg1.lower())
        self.assertNotIn("not yet bound", msg1.lower())
        self.assertIn("subscribe", msg1.lower())
        self.assert_no_leaks(msg1)

        # Turn 2 (Verbatim repeat):
        res2 = self.brain.process("Subscribe me", execute=False)
        dir2 = res2["directive"]
        msg2 = dir2["message"]

        self.assertFalse(dir2.get("can_execute", False))
        self.assertNotEqual(msg1, msg2, "Repetition must trigger an adaptive strategy change.")
        self.assertTrue(
            any(w in msg2.lower() for w in ["still", "stuck", "channel", "page", "open"]),
            f"Expected adaptive repetition recovery, got: '{msg2}'"
        )
        self.assert_no_leaks(msg2)

    # -------------------------------------------------------------------------
    # Scenario 4: "Tell me something about myself" (Self-reference memory retrieval)
    # -------------------------------------------------------------------------
    def test_scenario_4_tell_me_something_about_myself(self):
        """4. 'Tell me something about myself' -> Must answer naturally about user, never emit classifier labels."""
        query = "Tell me something about myself"
        res = self.brain.process(query, execute=False)

        msg = res["directive"]["message"]

        # Must never leak classifier labels or internal thought tags
        self.assertNotIn("command classified as", msg.lower())
        self.assertNotIn("exec_project_action", msg.lower())
        self.assertNotIn("exec_read_metrics", msg.lower())
        self.assertNotIn("saialog", msg.lower())

        # Must answer about user and mention known context
        self.assertTrue(
            any(name in msg for name in ["Aditya", "Harshan", "working", "days"]),
            f"Expected personal context in response, got: '{msg}'"
        )

        # Output Contract check
        self.assert_no_leaks(msg)

    # -------------------------------------------------------------------------
    # Scenario 5: "What do you know about me" (Human verbalization, never DB syntax)
    # -------------------------------------------------------------------------
    def test_scenario_5_what_do_you_know_about_me(self):
        """5. 'What do you know about me' -> Verbalize memory like a human, never 'X is relationship (friend)'."""
        query = "What do you know about me"
        res = self.brain.process(query, execute=False)

        msg = res["directive"]["message"]

        # Must NEVER contain database schema syntax
        self.assertNotIn("is relationship (friend)", msg)
        self.assertNotIn("relationship (", msg)
        self.assertNotIn("semantic memory:", msg.lower())

        # Must contain conversational verbalization of the relationship
        self.assertTrue("Harshan" in msg or "friend" in msg, f"Expected friend context verbalized, got: '{msg}'")

        # Output Contract check
        self.assert_no_leaks(msg)

    # -------------------------------------------------------------------------
    # Scenario 6: Startup Greeting (Varied and generated, not rigid template)
    # -------------------------------------------------------------------------
    def test_scenario_6_startup_greeting_varied(self):
        """6. Startup greeting -> Must be generated and varied, never rigid template concatenation."""
        personality = EdithPersonality()

        greetings = set()
        for _ in range(5):
            g = personality.generate_startup_greeting()

            # Must not be the rigid template
            self.assertNotIn("consistency is key. 💯", g.lower())
            self.assert_no_leaks(g)
            greetings.add(g)

        self.assertGreater(len(greetings), 1, "Startup greetings must show variation across sessions.")

    # -------------------------------------------------------------------------
    # Scenario 7: UI Noise & Speak Choke Point Verification
    # -------------------------------------------------------------------------
    def test_scenario_7_ui_noise_silenced_in_normal_mode(self):
        """7. Normal mode must strictly silence all telemetry cards and execution echoes."""
        # Test the speak bottleneck
        test_text = "I've opened Spotify for you."
        spoken = speak(test_text, voice_mode=False, debug_mode=False, strict_raise=True)
        self.assertEqual(spoken, test_text)

        # Verify that leaking text raises in strict mode
        leaking_text = "Executing exec_system_utility with operational confidence 0.25."
        with self.assertRaises(ValueError):
            speak(leaking_text, voice_mode=False, debug_mode=False, strict_raise=True)

        # Verify TTS Sanitizer strips symbols and formatting
        messy_text = "**Good morning!** I can help with that (definitely) -> [Spotify] 🚀"
        sanitized = TTSSanitizer.sanitize(messy_text)
        self.assertNotIn("**", sanitized)
        self.assertNotIn("(", sanitized)
        self.assertNotIn(")", sanitized)
        self.assertNotIn("->", sanitized)
        self.assertNotIn("🚀", sanitized)
        self.assert_no_leaks(sanitized)


if __name__ == "__main__":
    unittest.main()
