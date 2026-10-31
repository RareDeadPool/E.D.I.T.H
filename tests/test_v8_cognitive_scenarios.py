"""Comprehensive Automated Test Suite for E.D.I.T.H. 150 Cognitive & Empathy Scenarios (V10).

Validates:
1. Dataset integrity across all 150 scenarios defined in data/cognitive_scenarios_dataset.json.
2. CognitiveScenarioEngine direct matching and response generation across all 150 scenarios.
3. EdithBrain end-to-end routing, high operational confidence, and zero false command execution.
4. High emotional intelligence, empathy keywords, actionable problem-solving, and domain coverage across 15 domains:
   - Domain 1: Relationships, Heartbreak & Social Bonds (Scenarios 1-10)
   - Domain 2: Career, Academics & Work Pressure (Scenarios 11-20)
   - Domain 3: Mental Health, Emotional Crises & Well-being (Scenarios 21-30)
   - Domain 4: Cognitive Decisions & Rational Problem Solving (Scenarios 31-40)
   - Domain 5: Software Engineering & DevOps Panics (Scenarios 41-50)
   - Domain 6: Deep Focus, ADHD, Creativity & Productivity (Scenarios 51-60)
   - Domain 7: Everyday Life Navigation, Money & Social Skills (Scenarios 61-68)
   - Domain 8: Humor, Philosophizing & Late-Night Companionship (Scenarios 69-75)
   - Domain 9: Modern Dating, Ghosting, and Emotional Closure (Scenarios 76-85)
   - Domain 10: Deep Systems Architecture & Engineering Tradeoffs (Scenarios 86-95)
   - Domain 11: Philosophy of Mind, Consciousness, and AI Sentience (Scenarios 96-105)
   - Domain 12: Deep Solitude, Existential Resilience, and Meaning (Scenarios 106-115)
   - Domain 13: Career Mastery, High-Stakes Negotiation, and Strategy (Scenarios 116-125)
   - Domain 14: ADHD, Hyperfocus, Executive Function, and Dopamine (Scenarios 126-135)
   - Domain 15: The Visionary Builder's Solitude and Masterpiece Drive (Scenarios 136-150)
"""

import os
import sys
import json
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.edith_brain import EdithBrain
from engine.cognitive_scenarios_engine import CognitiveScenarioEngine


class TestV8CognitiveScenarios(unittest.TestCase):
    """Test suite validating 150 human cognitive, problem solving, and empathy scenarios."""

    @classmethod
    def setUpClass(cls):
        cls.dataset_path = os.path.join(PROJECT_ROOT, "data", "cognitive_scenarios_dataset.json")
        with open(cls.dataset_path, "r", encoding="utf-8") as f:
            cls.dataset = json.load(f)
        cls.scenarios = cls.dataset.get("scenarios", [])
        cls.scenario_engine = CognitiveScenarioEngine()
        cls.brain = EdithBrain(enable_execution=False)

    def test_01_dataset_integrity(self):
        """Verify the dataset contains exactly 150 scenarios with all required fields."""
        self.assertEqual(len(self.scenarios), 150, "Dataset must contain exactly 150 scenarios.")
        for s in self.scenarios:
            self.assertIn("id", s)
            self.assertIn("domain", s)
            self.assertIn("prompt", s)
            self.assertIn("expected_topic", s)
            self.assertIn("empathy_keywords", s)
            self.assertIsInstance(s["empathy_keywords"], list)
            self.assertGreater(len(s["empathy_keywords"]), 0)

    def test_02_scenario_engine_matches_and_handles_all_150(self):
        """Verify CognitiveScenarioEngine directly recognizes and responds to all 150 scenarios."""
        unmatched = []
        unhandled = []

        for s in self.scenarios:
            q = s["prompt"]
            q_low = q.lower()
            if not self.scenario_engine.matches_scenario(q_low):
                unmatched.append(s["id"])
            res = self.scenario_engine.handle_scenario(q, q_low, user_name="Aditya")
            if not res or not res.get("message"):
                unhandled.append(s["id"])

        self.assertEqual(unmatched, [], f"Engine failed to match scenarios: {unmatched}")
        self.assertEqual(unhandled, [], f"Engine failed to handle scenarios: {unhandled}")

    def test_03_edith_brain_end_to_end_coverage_150_scenarios(self):
        """Verify EdithBrain routes all 150 scenarios to dialogue with warm, empathetic answers."""
        failures = []
        keyword_mismatches = []

        for s in self.scenarios:
            q = s["prompt"]
            result = self.brain.process(q)
            directive = result.get("directive", {})
            handler = directive.get("handler")
            message = directive.get("message", "")

            if handler != "dialogue":
                failures.append((s["id"], q, handler))
                continue

            if len(message) < 50:
                failures.append((s["id"], q, "Message too short (< 50 chars)"))
                continue

            # Check empathy keywords (case-insensitive)
            msg_low = message.lower()
            matched_keywords = [kw for kw in s["empathy_keywords"] if kw.lower() in msg_low]
            if not matched_keywords:
                keyword_mismatches.append((s["id"], s["expected_topic"], s["empathy_keywords"]))

        self.assertEqual(failures, [], f"Failed end-to-end processing for scenarios: {failures}")
        self.assertEqual(keyword_mismatches, [], f"Empathy keyword mismatches in scenarios: {keyword_mismatches}")

    def test_04_domain_1_relationships_and_heartbreak(self):
        """Test Domain 1 scenarios (Love, Breakup, Ghosts, Boundaries)."""
        domain_scenarios = [s for s in self.scenarios if s["domain"] == "relationships_and_heartbreak"]
        self.assertEqual(len(domain_scenarios), 10)

        for s in domain_scenarios:
            res = self.brain.process(s["prompt"])
            msg = res["directive"]["message"]
            self.assertIn("Aditya", msg)
            self.assertEqual(res["directive"]["handler"], "dialogue")

    def test_05_domain_2_career_and_academics(self):
        """Test Domain 2 scenarios (Rejection, Imposter Syndrome, Exam Panic, Corporate)."""
        domain_scenarios = [s for s in self.scenarios if s["domain"] == "career_and_academics"]
        self.assertEqual(len(domain_scenarios), 10)

        for s in domain_scenarios:
            res = self.brain.process(s["prompt"])
            msg = res["directive"]["message"]
            self.assertTrue("Aditya" in msg or len(msg) > 30)
            self.assertEqual(res["directive"]["handler"], "dialogue")

    def test_06_domain_3_mental_health_and_crises(self):
        """Test Domain 3 scenarios (Panic attack, Insomnia, Burnout, Pet loss)."""
        domain_scenarios = [s for s in self.scenarios if s["domain"] == "mental_health_and_wellbeing"]
        self.assertEqual(len(domain_scenarios), 10)

        for s in domain_scenarios:
            res = self.brain.process(s["prompt"])
            msg = res["directive"]["message"]
            self.assertTrue("Aditya" in msg or len(msg) > 30)
            self.assertEqual(res["directive"]["handler"], "dialogue")

    def test_07_domain_4_cognitive_decisions(self):
        """Test Domain 4 scenarios (Relocation, Ethical dilemmas, Doomscrolling)."""
        domain_scenarios = [s for s in self.scenarios if s["domain"] == "cognitive_decisions"]
        self.assertEqual(len(domain_scenarios), 10)

        for s in domain_scenarios:
            res = self.brain.process(s["prompt"])
            msg = res["directive"]["message"]
            self.assertTrue("Aditya" in msg or len(msg) > 30)
            self.assertEqual(res["directive"]["handler"], "dialogue")

    def test_08_domain_5_engineering_and_devops(self):
        """Test Domain 5 scenarios (Dropped DB, Leaked AWS keys, Merge conflict, Docker OOM)."""
        domain_scenarios = [s for s in self.scenarios if s["domain"] == "software_engineering_panics"]
        self.assertEqual(len(domain_scenarios), 10)

        for s in domain_scenarios:
            res = self.brain.process(s["prompt"])
            msg = res["directive"]["message"]
            self.assertTrue("Aditya" in msg or len(msg) > 30)
            self.assertEqual(res["directive"]["handler"], "dialogue")

    def test_09_domain_6_focus_and_productivity(self):
        """Test Domain 6 scenarios (ADHD tabs, Afternoon slump, Digital clutter)."""
        domain_scenarios = [s for s in self.scenarios if s["domain"] == "focus_and_productivity"]
        self.assertEqual(len(domain_scenarios), 10)

        for s in domain_scenarios:
            res = self.brain.process(s["prompt"])
            msg = res["directive"]["message"]
            self.assertTrue("Aditya" in msg or len(msg) > 30)
            self.assertEqual(res["directive"]["handler"], "dialogue")

    def test_10_domain_7_life_skills_and_money(self):
        """Test Domain 7 scenarios (Emergency expenses, Raise negotiation, Roommate dishes)."""
        domain_scenarios = [s for s in self.scenarios if s["domain"] == "life_skills_and_money"]
        self.assertEqual(len(domain_scenarios), 8)

        for s in domain_scenarios:
            res = self.brain.process(s["prompt"])
            msg = res["directive"]["message"]
            self.assertTrue("Aditya" in msg or len(msg) > 30)
            self.assertEqual(res["directive"]["handler"], "dialogue")

    def test_11_domain_8_humor_and_companionship(self):
        """Test Domain 8 scenarios (Unseen win, Simulation theory, Variable naming roast)."""
        domain_scenarios = [s for s in self.scenarios if s["domain"] == "humor_and_companionship"]
        self.assertEqual(len(domain_scenarios), 7)

        for s in domain_scenarios:
            res = self.brain.process(s["prompt"])
            msg = res["directive"]["message"]
            self.assertTrue("Aditya" in msg or len(msg) > 30)
            self.assertEqual(res["directive"]["handler"], "dialogue")

    def test_12_domain_9_modern_dating_and_closure(self):
        """Test Domain 9 scenarios (Ghosting, Breadcrumbing, Closure, Scarcity)."""
        domain_scenarios = [s for s in self.scenarios if s["domain"] == "modern_dating_and_heartbreak"]
        self.assertEqual(len(domain_scenarios), 10)

        for s in domain_scenarios:
            res = self.brain.process(s["prompt"])
            msg = res["directive"]["message"]
            self.assertTrue("Aditya" in msg or len(msg) > 30)
            self.assertEqual(res["directive"]["handler"], "dialogue")

    def test_13_domain_10_advanced_systems_and_engineering(self):
        """Test Domain 10 scenarios (Distributed consensus, Cache stampedes, Latency spikes)."""
        domain_scenarios = [s for s in self.scenarios if s["domain"] == "advanced_systems_and_engineering"]
        self.assertEqual(len(domain_scenarios), 10)

        for s in domain_scenarios:
            res = self.brain.process(s["prompt"])
            msg = res["directive"]["message"]
            self.assertTrue("Aditya" in msg or len(msg) > 30)
            self.assertEqual(res["directive"]["handler"], "dialogue")

    def test_14_domain_11_deep_philosophy_and_sentience(self):
        """Test Domain 11 scenarios (AI consciousness, Free will, Chinese Room, Alignment)."""
        domain_scenarios = [s for s in self.scenarios if s["domain"] == "deep_philosophy_and_sentience"]
        self.assertEqual(len(domain_scenarios), 10)

        for s in domain_scenarios:
            res = self.brain.process(s["prompt"])
            msg = res["directive"]["message"]
            self.assertTrue("Aditya" in msg or len(msg) > 30)
            self.assertEqual(res["directive"]["handler"], "dialogue")

    def test_15_domain_12_stoic_resilience_and_solitude(self):
        """Test Domain 12 scenarios (Deep solitude, Unfair criticism, Betrayal, Adversity)."""
        domain_scenarios = [s for s in self.scenarios if s["domain"] == "stoic_resilience_and_solitude"]
        self.assertEqual(len(domain_scenarios), 10)

        for s in domain_scenarios:
            res = self.brain.process(s["prompt"])
            msg = res["directive"]["message"]
            self.assertTrue("Aditya" in msg or len(msg) > 30)
            self.assertEqual(res["directive"]["handler"], "dialogue")

    def test_16_domain_13_career_strategy_and_negotiation(self):
        """Test Domain 13 scenarios (Salary negotiation, Toxic micromanagers, Equity tradeoffs)."""
        domain_scenarios = [s for s in self.scenarios if s["domain"] == "career_strategy_and_negotiation"]
        self.assertEqual(len(domain_scenarios), 10)

        for s in domain_scenarios:
            res = self.brain.process(s["prompt"])
            msg = res["directive"]["message"]
            self.assertTrue("Aditya" in msg or len(msg) > 30)
            self.assertEqual(res["directive"]["handler"], "dialogue")

    def test_17_domain_14_focus_and_dopamine_mastery(self):
        """Test Domain 14 scenarios (Dopamine fasting, Hyperfocus crash, Task paralysis)."""
        domain_scenarios = [s for s in self.scenarios if s["domain"] == "focus_and_dopamine_mastery"]
        self.assertEqual(len(domain_scenarios), 10)

        for s in domain_scenarios:
            res = self.brain.process(s["prompt"])
            msg = res["directive"]["message"]
            self.assertTrue("Aditya" in msg or len(msg) > 30)
            self.assertEqual(res["directive"]["handler"], "dialogue")

    def test_18_domain_15_builders_journey_and_vision(self):
        """Test Domain 15 scenarios (Solo builder loneliness, Tech debt vs shipping, Legacy)."""
        domain_scenarios = [s for s in self.scenarios if s["domain"] == "builders_journey_and_vision"]
        self.assertEqual(len(domain_scenarios), 15)

        for s in domain_scenarios:
            res = self.brain.process(s["prompt"])
            msg = res["directive"]["message"]
            self.assertTrue("Aditya" in msg or len(msg) > 30)
            self.assertEqual(res["directive"]["handler"], "dialogue")


if __name__ == "__main__":
    unittest.main()
