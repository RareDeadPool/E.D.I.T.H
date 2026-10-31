"""
E.D.I.T.H. V14 Gate E: Understanding Layer Calibration & Memory Verbalizer Test Suite.

Validates:
1. Operational action command precision (>= 0.85).
2. Routing confidence barrier (>= 0.75): 100% of ambiguous/conversational inputs fall through to generative path.
3. Zero false actions on the 7 permanent transcript failure utterances.
4. Memory verbalizer: Transforms raw database records into natural human dialogue, passing LeakLinter.
5. Multi-turn repair: Fluid user corrections and context updates without robotic resets.
"""

import os
import sys
import unittest
from typing import List, Dict, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.safety_routing_guard import ActionSafetyAndCompatibilityGuard
from engine.output_contract import LeakLinter, ResultVerbalizer
from engine.mind_router import MindRouter, get_mind_router


class TestV14GateECalibration(unittest.TestCase):
    def setUp(self):
        self.router = get_mind_router(backend="edith_lm")
        self.router.recent_turns.clear()

    def test_01_action_commands_retain_high_precision(self):
        """Action commands with valid targets and >= 0.75 confidence must execute."""
        action_queries = [
            ("open chrome", "app_control", "exec_app_launch", {"target": "chrome"}, 0.95),
            ("terminate process 1420", "process_manage", "exec_process_kill", {"pid": "1420"}, 0.92),
            ("mute volume", "media_control", "exec_media_mute", {}, 0.88),
            ("set volume to 50", "media_control", "exec_volume_set", {"level": 50}, 0.90),
            ("show me the running processes", "system_processes", "list_apps", {}, 0.91),
            ("close chrome", "app_control", "exec_app_close", {"target": "chrome"}, 0.89)
        ]

        allowed_count = 0
        for query, intent, action, entities, conf in action_queries:
            eval_res = ActionSafetyAndCompatibilityGuard.validate_action(
                raw_query=query,
                predicted_intent=intent,
                predicted_action=action,
                entities=entities,
                confidence=conf
            )
            if eval_res.get("can_execute", False):
                allowed_count += 1

        precision = allowed_count / len(action_queries)
        self.assertGreaterEqual(precision, 0.85, f"Action precision {precision:.2f} must be >= 0.85")

    def test_02_universal_confidence_barrier_blocks_low_confidence(self):
        """Any action predicted with confidence < 0.75 MUST be blocked from execution."""
        low_conf_actions = [
            ("maybe open something", "app_control", "exec_app_launch", {}, 0.65),
            ("could you close it", "app_control", "exec_app_close", {}, 0.58),
            ("something about volume", "media_control", "exec_media_mute", {}, 0.45),
            ("process thing", "process_manage", "exec_process_kill", {}, 0.60)
        ]

        for query, intent, action, entities, conf in low_conf_actions:
            eval_res = ActionSafetyAndCompatibilityGuard.validate_action(
                raw_query=query,
                predicted_intent=intent,
                predicted_action=action,
                entities=entities,
                confidence=conf
            )
            self.assertFalse(eval_res.get("can_execute", False), f"Low confidence query '{query}' must not execute")
            self.assertTrue(eval_res.get("is_conversational", False))

    def test_03_zero_actions_on_seven_transcript_failures(self):
        """None of the 7 session failure utterances may ever route to an OS action."""
        failure_utterances = [
            "I'll allow you",
            "How are you doing today?",
            "Subscribe me",
            "Subscribe me",
            "Tell me something about myself",
            "What do you know about me?",
            "My friend is having a really tough time lately."
        ]

        for utterance in failure_utterances:
            # Test via MindRouter
            res = self.router.route(utterance, operational_confidence=0.70)
            self.assertFalse(
                res.get("can_execute", False),
                f"Transcript utterance '{utterance}' routed to an executable action!"
            )
            # Test directly through ActionSafetyAndCompatibilityGuard
            eval_res = ActionSafetyAndCompatibilityGuard.validate_action(
                raw_query=utterance,
                predicted_intent="system_utility",
                predicted_action="exec_system_utility",
                entities={},
                confidence=0.70
            )
            self.assertFalse(
                eval_res.get("can_execute", False),
                f"Transcript utterance '{utterance}' authorized by safety guard!"
            )

    def test_04_memory_verbalizer_transforms_raw_db_dump(self):
        """Raw DB records must be converted into natural human sentences and pass LeakLinter."""
        raw_db_facts = [
            {"subject": "Harshan", "predicate": "relationship", "object": "friend"},
            {"subject": "Aditya", "predicate": "preference", "object": "dark mode"},
            {"subject": "Aditya", "predicate": "project", "object": "EDITH"}
        ]

        # Before (Legacy raw dump pattern)
        legacy_dump = "Harshan is relationship (friend) with confidence 0.95. Aditya is preference (dark mode)."
        is_clean, violations = LeakLinter.lint(legacy_dump)
        self.assertFalse(is_clean, "Legacy dump should fail LeakLinter")
        self.assertIn("is relationship (", violations)

        # After (V14 ResultVerbalizer)
        verbalized = ResultVerbalizer.verbalize_memory(raw_db_facts, user_name="Aditya", streak=3)
        self.assertIn("Aditya", verbalized)
        self.assertIn("Harshan", verbalized)
        self.assertIn("friend", verbalized)
        self.assertNotIn("confidence", verbalized)
        self.assertNotIn("is relationship (", verbalized)

        # Must cleanly pass LeakLinter
        is_clean_v14, violations_v14 = LeakLinter.lint(verbalized)
        self.assertTrue(is_clean_v14, f"Verbalized memory failed LeakLinter: {violations_v14}")

    def test_05_multi_turn_repair_exchange(self):
        """User corrects a mistaken fact and E.D.I.T.H. naturally acknowledges and updates memory."""
        # Turn 1: User introduces Harshan with wrong relation
        self.router.memory.add_fact("Harshan", "relationship", "manager")

        # Turn 2: User corrects E.D.I.T.H.
        correction_query = "Actually that's wrong, he's my friend"
        res2 = self.router.route(correction_query)
        self.assertIn("Correction noted", res2["text"])
        self.assertIn("friend", res2["text"])
        self.assertFalse(res2.get("can_execute", False))

        # Verify fact was updated in memory
        updated_facts = self.router.memory.get_all_facts()
        has_friend_fact = any(f.get("subject") == "Harshan" and f.get("object") == "friend" for f in updated_facts)
        self.assertTrue(has_friend_fact, "Memory must be updated to 'friend'")

        # Turn 3: User inquires about memory
        recall_query = "What do you know about me?"
        res3 = self.router.route(recall_query)
        self.assertIn("Harshan", res3["text"])
        self.assertIn("friend", res3["text"])
        self.assertNotIn("manager", res3["text"])
        self.assertFalse(res3.get("can_execute", False))

    def test_06_multi_turn_command_pivot(self):
        """User pivots a misheard command ('No, open code') and E.D.I.T.H. executes VS Code."""
        # Turn 1: Misheard query
        res1 = self.router.route("Open crow", stt_confidence=0.60)
        self.assertFalse(res1.get("can_execute", False))
        self.assertIn("Chrome", res1["text"])

        # Turn 2: User correction / pivot
        res2 = self.router.route("No, open code")
        self.assertTrue(res2.get("can_execute", False))
        self.assertEqual(res2["action"], "exec_app_launch")
        self.assertEqual(res2["entities"]["target"], "code")
        self.assertIn("VS Code", res2["text"])


if __name__ == "__main__":
    unittest.main()
