"""Comprehensive Test Suite for E.D.I.T.H. Human Dialogue & Problem-Solving Engine (V6).

Validates:
1. Social conversations, greetings, persona identity, capabilities, and gratitude.
2. Emotional intelligence and human empathy (stress, fatigue, accomplishment, bug frustration).
3. Technical problem-solving across system slowdown, Wi-Fi connectivity, disk space, and frozen apps.
4. Multi-turn diagnostic follow-up continuity (suggesting diagnostics -> user confirmation -> live execution).
5. Document & file workflow guidance (converting images to PDF).
6. Software development & coding assistance (ModuleNotFoundError, Git merge conflicts, Python venv, merge vs rebase).
7. Natural language everyday budget & financial problem calculations.
8. Preservation of physical/unsupported OOD queries (spaceship to Mars, cakes on Pluto).
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.edith_brain import EdithBrain


class TestEdithHumanDialogueAndProblemSolving(unittest.TestCase):
    """Test suite for human-like communication and problem-solving reasoning in E.D.I.T.H."""

    def setUp(self):
        self.brain = EdithBrain(enable_execution=False)

    def test_01_social_greetings_and_identity(self):
        """Test natural greetings, identity explanation, and capability summary."""
        # Greeting
        res1 = self.brain.process("hello edith")
        self.assertEqual(res1["intent"], "chitchat")
        self.assertEqual(res1["directive"]["handler"], "dialogue")
        self.assertIn("Aditya", res1["directive"]["message"])

        # Identity
        res2 = self.brain.process("who are you")
        self.assertEqual(res2["directive"]["handler"], "dialogue")
        self.assertIn("E.D.I.T.H.", res2["directive"]["message"])
        self.assertIn("Even Dead, I'm The Hero", res2["directive"]["message"])

        # Capabilities
        res3 = self.brain.process("what can you do for me")
        self.assertEqual(res3["directive"]["handler"], "dialogue")
        msg = res3["directive"]["message"]
        self.assertIn("Desktop & App Control", msg)
        self.assertIn("Messaging & WhatsApp", msg)
        self.assertIn("Files & Workspaces", msg)
        self.assertIn("System Tools & Metrics", msg)
        self.assertIn("Problem Solving", msg)

    def test_02_emotional_empathy_and_support(self):
        """Test empathetic responses to stress, success, and developer frustration."""
        # Stress / Overwhelm
        res_stress = self.brain.process("I feel really stressed today with all this work")
        msg_stress = res_stress["directive"]["message"]
        self.assertIn("Aditya", msg_stress)
        self.assertIn("20-20-20 rule", msg_stress)
        self.assertIn("Spotify", msg_stress)

        # Celebration / Milestone
        res_success = self.brain.process("I finished my project and all tests passed!")
        msg_success = res_success["directive"]["message"]
        self.assertIn("congratulations", msg_success.lower())

        # Frustration with bugs
        res_frust = self.brain.process("nothing is working and I hate bugs")
        msg_frust = res_frust["directive"]["message"]
        self.assertIn("frustration", msg_frust.lower())
        self.assertIn("step-by-step", msg_frust.lower())

    def test_03_technical_system_lag_troubleshooting(self):
        """Test computer performance troubleshooting."""
        res = self.brain.process("my computer is running really slow")
        msg = res["directive"]["message"]
        self.assertIn("sluggish", msg.lower())
        self.assertIn("Task Manager", msg)
        self.assertIn("Ctrl + Shift + Esc", msg)
        self.assertIn("metrics", msg.lower())

    def test_04_network_wifi_troubleshooting(self):
        """Test Wi-Fi and network connectivity diagnostic advice."""
        res = self.brain.process("how do I fix my wifi connection")
        msg = res["directive"]["message"]
        self.assertIn("ipconfig /flushdns", msg)
        self.assertIn("8.8.8.8", msg)
        self.assertIn("router", msg.lower())

    def test_05_multi_turn_metrics_diagnostic_follow_up(self):
        """Test multi-turn flow: system slow diagnosis -> user confirmation -> live metrics display."""
        # Turn 1: Problem diagnosis
        res1 = self.brain.process("my computer is running really slow")
        self.assertIn("metrics", res1["directive"]["message"].lower())

        # Turn 2: Follow-up confirmation
        res2 = self.brain.process("yes check it")
        msg2 = res2["directive"]["message"]
        self.assertIn("Resource Utilization", msg2)
        self.assertIn("CPU", msg2)
        self.assertIn("Memory", msg2)
        self.assertIn("Storage", msg2)

    def test_06_multi_turn_network_ping_follow_up(self):
        """Test multi-turn flow: Wi-Fi troubleshooting -> user confirmation -> network ping test."""
        # Turn 1: Problem diagnosis
        res1 = self.brain.process("how do I fix my wifi connection")
        self.assertIn("ping diagnostic", res1["directive"]["message"].lower())

        # Turn 2: Follow-up confirmation
        res2 = self.brain.process("yes please check")
        msg2 = res2["directive"]["message"]
        self.assertIn("Network Diagnostic", msg2)
        self.assertIn("8.8.8.8", msg2)

    def test_07_storage_disk_cleanup(self):
        """Test storage space exhaustion troubleshooting."""
        res = self.brain.process("how do I free up disk space on my C drive")
        msg = res["directive"]["message"]
        self.assertIn("%temp%", msg)
        self.assertIn("cleanmgr", msg)
        self.assertIn("Recycle Bin", msg)

    def test_08_frozen_app_troubleshooting(self):
        """Test unresponsive window / frozen app troubleshooting."""
        res = self.brain.process("an app is frozen and not responding")
        msg = res["directive"]["message"]
        self.assertIn("Alt + F4", msg)
        self.assertIn("Ctrl + Shift + Esc", msg)
        self.assertIn("taskkill", msg)

    def test_09_file_conversion_guidance(self):
        """Test practical document guidance (converting images to PDF)."""
        res = self.brain.process("how do I convert images to pdf")
        msg = res["directive"]["message"]
        self.assertIn("Microsoft Print to PDF", msg)
        self.assertIn("Pillow", msg)

    def test_10_software_dev_troubleshooting(self):
        """Test software engineering bugs (ModuleNotFoundError, Git conflict, Python venv)."""
        # Python ModuleNotFoundError
        res1 = self.brain.process("ModuleNotFoundError: No module named requests")
        msg1 = res1["directive"]["message"]
        self.assertIn("pip install requests", msg1)
        self.assertIn(".venv", msg1)

        # Git merge conflicts
        res2 = self.brain.process("how do I fix git merge conflicts")
        msg2 = res2["directive"]["message"]
        self.assertIn("git status", msg2)
        self.assertIn("<<<<<<< HEAD", msg2)
        self.assertIn("git commit", msg2)

        # Python virtual environment
        res3 = self.brain.process("how to create a python virtual environment")
        msg3 = res3["directive"]["message"]
        self.assertIn("python -m venv .venv", msg3)
        self.assertIn("Activate.ps1", msg3)

    def test_11_natural_language_budget_word_problem(self):
        """Test natural language budget word problems with calculations and financial breakdown."""
        query = "Help me calculate my monthly budget: 25000 for rent, 8000 for groceries, 3500 for bills and 2000 for fuel"
        res = self.brain.process(query)
        msg = res["directive"]["message"]
        self.assertIn("38,500.00", msg)
        self.assertIn("Rent", msg)
        self.assertIn("Groceries", msg)
        self.assertIn("Bills", msg)
        self.assertIn("Fuel", msg)
        self.assertIn("64.9%", msg)

    def test_12_productivity_and_decision_advice(self):
        """Test productivity and technical decision-making advice."""
        # Prioritization advice
        res1 = self.brain.process("how should I prioritize my tasks today")
        msg1 = res1["directive"]["message"]
        self.assertIn("Eisenhower Matrix", msg1)

        # Git merge vs rebase
        res2 = self.brain.process("what is the difference between git merge and rebase")
        msg2 = res2["directive"]["message"]
        self.assertIn("merge commit", msg2)
        self.assertIn("rewrites", msg2.lower())

    def test_13_small_talk_humor_and_gratitude(self):
        """Test jokes, appreciation, and farewells."""
        # Joke
        res_joke = self.brain.process("tell me a joke")
        self.assertTrue(len(res_joke["directive"]["message"]) > 10)

        # Gratitude
        res_thanks = self.brain.process("thank you so much edith")
        self.assertIn("Aditya", res_thanks["directive"]["message"])

        # Farewell
        res_bye = self.brain.process("goodbye edith")
        self.assertIn("Aditya", res_bye["directive"]["message"])

    def test_14_preservation_of_ood_benchmark(self):
        """Verify that genuine physical/unsupported OOD requests remain unsupported_ood."""
        res1 = self.brain.process("fly a spaceship to mars")
        self.assertEqual(res1["intent"], "unsupported_ood")
        self.assertFalse(res1["directive"]["can_execute"])

        res2 = self.brain.process("bake me a chocolate cake on pluto")
        self.assertEqual(res2["intent"], "unsupported_ood")
        self.assertFalse(res2["directive"]["can_execute"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
