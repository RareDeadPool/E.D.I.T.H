"""Comprehensive Natural Intelligence & Cognitive Task Test Suite for E.D.I.T.H.

Tests:
1. Natural Language Paraphrase Handling & Slot Extraction
2. Multi-turn Conversational Context, Ordinal & Anaphora Resolution
3. Structured Multi-Step Planning & Dependency Output Chaining
4. Safety Gates for Consequential Operations (File deletion, process killing)
5. Out-of-Domain & Ambiguity Graceful Handling
"""

import unittest
import os
import sys

# Ensure repository root is in sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from engine.edith_brain import EdithBrain
from runtime.context import ConversationContext
from engine.planner import TaskPlanner, PlanStep, TaskPlan


class TestNaturalParaphrases(unittest.TestCase):
    """Test natural language paraphrasing across multiple intent categories."""

    @classmethod
    def setUpClass(cls):
        cls.brain = EdithBrain(enable_execution=False)

    def test_app_launch_paraphrases(self):
        queries = [
            "bring up my browser",
            "fire up notepad",
            "can you open chrome for me",
            "start up spotify",
            "open the terminal",
        ]
        for q in queries:
            with self.subTest(query=q):
                res = self.brain.process(q)
                self.assertIn(res["intent"], ["app_launch", "automation_workflow"])
                self.assertGreaterEqual(res["operational_confidence"], 0.7)

    def test_web_search_paraphrases(self):
        queries = [
            "search online for quantum computing",
            "look up latest machine learning models",
            "search google for python tips",
        ]
        for q in queries:
            with self.subTest(query=q):
                res = self.brain.process(q)
                self.assertIn(res["intent"], ["web_search", "automation_workflow"])

    def test_system_control_paraphrases(self):
        queries = [
            "turn the volume way down",
            "make it a bit louder",
            "lower brightness to 50%",
            "mute the audio",
        ]
        for q in queries:
            with self.subTest(query=q):
                res = self.brain.process(q)
                self.assertIn(res["intent"], ["system_control", "media_control"])

    def test_diagnostics_paraphrases(self):
        queries = [
            "how much storage do I have left",
            "is my internet working",
            "check cpu and ram usage",
            "ping google.com",
        ]
        for q in queries:
            with self.subTest(query=q):
                res = self.brain.process(q)
                self.assertIn(res["intent"], ["system_query", "system_metrics", "network_task", "system_utility"])


class TestMultiTurnContextAndAnaphora(unittest.TestCase):
    """Test conversational state, ordinal extraction, and pronoun resolution."""

    def setUp(self):
        self.brain = EdithBrain(enable_execution=False)

    def test_ordinal_and_pronoun_resolution(self):
        ctx = self.brain.context
        # Simulate Turn 1: Search returned candidate files
        candidates = [
            {"name": "resume_2024.pdf", "path": "C:/Users/Aditya Sawant/Desktop/resume_2024.pdf"},
            {"name": "resume_2025.pdf", "path": "C:/Users/Aditya Sawant/Desktop/resume_2025.pdf"},
            {"name": "notes.txt", "path": "C:/Users/Aditya Sawant/Desktop/notes.txt"},
        ]
        ctx.set_search_results(candidates)
        ctx.record_turn("find my resume on desktop", "file_operation", "exec_file_action", {"status": "success"})

        # Turn 2: "open the second one"
        turn2_query = "open the second one"
        resolved_t2 = ctx.resolve_anaphora(turn2_query)
        self.assertIn("resume_2025.pdf", resolved_t2)

        res2 = self.brain.process(turn2_query)
        directive2 = res2["directive"]
        self.assertEqual(directive2["handler"], "file_operation")
        self.assertEqual(directive2["parameters"]["subaction"], "open")
        self.assertIn("resume_2025.pdf", directive2["parameters"]["source"])

        # Update context with Turn 2 result
        ctx.last_file = "resume_2025.pdf"
        ctx.last_path = "C:/Users/Aditya Sawant/Desktop/resume_2025.pdf"
        ctx.record_turn(turn2_query, "file_operation", "exec_file_action", {"status": "success"})

        # Turn 3: "delete it"
        turn3_query = "delete it"
        resolved_t3 = ctx.resolve_anaphora(turn3_query)
        self.assertIn("resume_2025.pdf", resolved_t3)

        res3 = self.brain.process(turn3_query)
        directive3 = res3["directive"]
        self.assertEqual(directive3["handler"], "file_operation")
        self.assertEqual(directive3["parameters"]["subaction"], "delete")
        self.assertIn("resume_2025.pdf", directive3["parameters"]["source"])

    def test_idiom_safety(self):
        """Ensure common idioms containing 'it' are not corrupted by anaphora."""
        ctx = self.brain.context
        ctx.last_file = "report.pdf"
        ctx.last_path = "C:/Users/Aditya Sawant/Desktop/report.pdf"

        self.assertEqual(ctx.resolve_anaphora("what time is it"), "what time is it")
        self.assertEqual(ctx.resolve_anaphora("who are you"), "who are you")


class TestMultiStepTaskPlanner(unittest.TestCase):
    """Test structured task planning, output chaining, and safety confirmation gates."""

    def setUp(self):
        self.planner = TaskPlanner()

    def test_consequential_action_safety_gate(self):
        raw_steps = ["delete it", "kill 4096"]
        directives = [
            {"handler": "file_operation", "parameters": {"subaction": "delete", "source": "it"}},
            {"handler": "kill_process", "parameters": {}}
        ]
        plan = self.planner.create_plan(
            goal="delete it and kill process",
            raw_steps=raw_steps,
            step_directives=directives
        )
        self.assertTrue(plan.steps[0].is_consequential)
        self.assertTrue(plan.steps[0].requires_confirmation)
        self.assertTrue(plan.steps[1].is_consequential)
        self.assertTrue(plan.steps[1].requires_confirmation)
        self.assertIsNotNone(plan.confirmation_needed)

    def test_output_chaining(self):
        raw_steps = ["find notes.txt", "open it"]
        directives = [
            {"handler": "file_operation", "parameters": {"subaction": "find", "query": "notes.txt"}},
            {"handler": "file_operation", "parameters": {"subaction": "open", "source": ""}}
        ]
        plan = self.planner.create_plan(
            goal="find notes.txt and open it",
            raw_steps=raw_steps,
            step_directives=directives
        )
        self.assertEqual(len(plan.steps), 2)
        self.assertIn(1, plan.steps[1].dependencies)

        # Simulate execution of step 1 returning a resolved path
        res1 = {"status": "success", "file": "C:/Users/Aditya Sawant/Desktop/notes.txt"}
        plan.steps[0].result = res1
        plan.steps[0].status = "completed"

        # Apply chaining
        self.planner.pipe_step_dependencies(plan.steps[1], {1: res1})
        self.assertEqual(plan.steps[1].parameters["source"], "C:/Users/Aditya Sawant/Desktop/notes.txt")


class TestOutOfDomainAndAmbiguity(unittest.TestCase):
    """Test graceful handling of out-of-domain and ambiguous requests."""

    @classmethod
    def setUpClass(cls):
        cls.brain = EdithBrain(enable_execution=False)

    def test_unsupported_ood(self):
        res = self.brain.process("fly a spaceship to mars")
        self.assertEqual(res["intent"], "unsupported_ood")
        self.assertEqual(res["action"], "request_clarification")
        self.assertFalse(res["directive"]["can_execute"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
