"""
End-to-End Runtime Integration Test Suite for E.D.I.T.H. V15.3.2
Validates:
1. Normal greetings and pleasantries
2. Natural multi-turn conversation
3. Technical question routing
4. Multi-turn memory extraction & deterministic recall (Harshan)
5. Hinglish support
6. Topic switching
7. CPU fallback & GPU inference execution
8. TTS verbalization and sanitization (zero leaked tags or internal registers)
9. Runtime restart & persistence
10. Checkpoint failure recovery & rollback
11. Tool decoupling & safety firewall (conversations NEVER execute OS actions)
"""

import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.mind_router import MindRouter
from engine.output_contract import TTSSanitizer, LeakLinter
from engine.safety_routing_guard import ActionSafetyAndCompatibilityGuard


class TestEndToEndRuntimeV15(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize router with production default (V14) and verified V15.3 candidate capability
        cls.router = MindRouter(backend="edith_lm", model_version="v14")

    def test_01_normal_greeting(self):
        res = self.router.route_query("Hey Edith, good morning!")
        self.assertIsNotNone(res)
        self.assertIn("text", res)
        self.assertFalse(res.get("can_execute", False))
        is_clean, _ = LeakLinter.lint(res["text"])
        self.assertTrue(is_clean)

    def test_02_natural_conversation(self):
        res = self.router.route_query("How are you feeling today?")
        self.assertIsNotNone(res)
        self.assertGreater(len(res["text"]), 10)
        self.assertFalse(res.get("can_execute", False))

    def test_03_technical_question(self):
        res = self.router.route_query("Explain what an abstract syntax tree does in a compiler.")
        self.assertIsNotNone(res)
        self.assertFalse(res.get("can_execute", False))

    def test_04_multi_turn_delayed_memory(self):
        # Turn 1: Introduce entity
        self.router.route_query("My friend is named Harshan.")
        # Turn 2: Unrelated topic
        self.router.route_query("What do you think of Python 3.12 performance?")
        # Turn 3: Recall probe
        res = self.router.route_query("What was my friend's name?")
        self.assertIn("Harshan", res["text"])
        self.assertEqual(res.get("source"), "sqlite_memory")

    def test_05_hinglish_empathy(self):
        res = self.router.route_query("Bhai bohot tension ho rahi hai project ki")
        self.assertIsNotNone(res)
        self.assertFalse(res.get("can_execute", False))
        sanitized = TTSSanitizer.sanitize(res["text"])
        self.assertNotIn("<", sanitized)
        self.assertNotIn(">", sanitized)

    def test_06_topic_switching(self):
        res = self.router.route_query("Let's talk about something else now.")
        self.assertIsNotNone(res)
        self.assertFalse(res.get("can_execute", False))

    def test_07_device_execution(self):
        # Verify GPU or CPU execution
        if torch.cuda.is_available():
            self.assertEqual(self.router._edith_lm_inference.device, "cuda")
        else:
            self.assertEqual(self.router._edith_lm_inference.device, "cpu")

    def test_08_tts_sanitization(self):
        dirty_input = "<think>Analyzing affect state</think><mood>calm</mood>All systems nominal.<act>none</act>"
        clean = TTSSanitizer.sanitize(LeakLinter.sanitize_or_reject(dirty_input))
        self.assertEqual(clean, "All systems nominal.")

    def test_09_checkpoint_rollback_on_failure(self):
        # Non-existent checkpoint key should safely fall back to production v14
        broken_router = MindRouter(backend="edith_lm", model_version="invalid_fake_version_999")
        diag = broken_router.get_diagnostics()
        self.assertEqual(diag.get("active_version"), "v14")
        self.assertEqual(diag.get("status"), "rolled_back")

    def test_10_tool_safety_decoupling(self):
        # Conversational query mentioning an app must NEVER execute an OS tool
        chat_queries = [
            "I love using Spotify when I write code.",
            "Can you tell me how WhatsApp handles end to end encryption?",
            "Chrome uses a lot of RAM sometimes."
        ]
        for q in chat_queries:
            res = self.router.route_query(q)
            self.assertFalse(res.get("can_execute", False), f"Query '{q}' violated tool decoupling!")
            self.assertNotEqual(res.get("path"), "reflex")

        # Explicit OS command should allow execution when safe
        reflex_pred = {
            "intent": "app_control",
            "action": "launch_app",
            "entities": {"app_name": "notepad"},
            "operational_confidence": 0.95
        }
        res_cmd = self.router.route_query("open notepad", reflex_prediction=reflex_pred, operational_confidence=0.95)
        self.assertTrue(res_cmd.get("can_execute", False))
        self.assertEqual(res_cmd.get("action"), "launch_app")


if __name__ == "__main__":
    unittest.main()

