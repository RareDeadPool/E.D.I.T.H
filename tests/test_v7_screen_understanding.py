"""Unit and Integration Test Suite for E.D.I.T.H. V7 Screen Understanding Pipeline.

Validates:
1. Active foreground window metadata retrieval.
2. Error and crash dialog pattern detection in visible text and titles.
3. Grounded reference interpretation ("What do you think of this?", "Why isn't this working?").
4. Overview synthesis ("What is going on on my screen?").
5. Privacy controls (pause and resume observation).
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runtime.screen_interpreter import ScreenInterpreter
from runtime.screen_engine import ScreenElement


class TestScreenInterpreter(unittest.TestCase):
    """Test suite for ScreenInterpreter."""

    def setUp(self):
        self.interpreter = ScreenInterpreter(enable_ocr=False)

    def test_01_privacy_pause(self):
        """Test observation pause stops screen reading for privacy."""
        self.interpreter.pause_observation(True)
        res = self.interpreter.observe()
        self.assertEqual(res["status"], "paused")
        self.assertIn("paused", res["message"])

        self.interpreter.pause_observation(False)
        self.assertFalse(self.interpreter.observation_paused)

    def test_02_detect_python_code_exception(self):
        """Test error detection from Python tracebacks."""
        traceback_text = (
            'Traceback (most recent call last):\n'
            '  File "c:/Users/Aditya/project/main.py", line 42, in <module>\n'
            'ModuleNotFoundError: No module named \'requests\''
        )
        err = self.interpreter._detect_errors_in_text(traceback_text, "main.py - Visual Studio Code")
        self.assertTrue(err["has_error"])
        self.assertEqual(err["error_type"], "ModuleNotFoundError")
        self.assertIn("requests", err["error_message"])
        self.assertIn("42", err["location"])

    def test_03_detect_windows_error_dialog(self):
        """Test detection of Windows application error messages."""
        dialog_text = "An error occurred while saving the file. Access denied."
        err = self.interpreter._detect_errors_in_text(dialog_text, "Save File Error")
        self.assertTrue(err["has_error"])
        self.assertEqual(err["category"], "app_error")

    def test_04_interpret_why_isnt_this_working(self):
        """Test grounded error troubleshooting response."""
        mock_obs = {
            "window": {"app": "Visual Studio Code", "title": "test_script.py"},
            "element_count": 10,
            "buttons": ["Run", "Debug"],
            "labels": [],
            "error_detected": True,
            "error_details": {
                "has_error": True,
                "error_type": "SyntaxError",
                "error_message": "invalid syntax",
                "location": "test_script.py (line 15)"
            }
        }
        with patch.object(self.interpreter, "observe", return_value=mock_obs):
            res = self.interpreter.interpret("Why isn't this working?")
            self.assertEqual(res["status"], "success")
            self.assertEqual(res["error_type"], "SyntaxError")
            self.assertIn("SyntaxError", res["message"])
            self.assertIn("line 15", res["message"])

    def test_05_interpret_screen_overview(self):
        """Test 'What is going on on my screen?' overview response."""
        mock_obs = {
            "window": {"app": "Google Chrome", "title": "GitHub - Repository Pull Requests"},
            "element_count": 25,
            "buttons": ["New pull request", "Filter"],
            "labels": ["Open", "Closed"],
            "error_detected": False,
            "error_details": {"has_error": False}
        }
        with patch.object(self.interpreter, "observe", return_value=mock_obs):
            res = self.interpreter.interpret("What is going on on my screen?")
            self.assertEqual(res["status"], "success")
            self.assertIn("Google Chrome", res["message"])
            self.assertIn("GitHub - Repository Pull Requests", res["message"])
            self.assertIn("New pull request", res["message"])

    def test_06_interpret_grounded_this(self):
        """Test 'What do you think of this?' with an active traceback."""
        mock_obs = {
            "window": {"app": "Visual Studio Code", "title": "api.py"},
            "element_count": 12,
            "buttons": [],
            "labels": [],
            "error_detected": True,
            "error_details": {
                "has_error": True,
                "error_type": "KeyError",
                "error_message": "'user_id'",
                "location": "api.py (line 88)"
            }
        }
        with patch.object(self.interpreter, "observe", return_value=mock_obs):
            res = self.interpreter.interpret("What do you think of this?")
            self.assertEqual(res["status"], "success")
            self.assertIn("KeyError", res["message"])
            self.assertIn("'user_id'", res["message"])


if __name__ == "__main__":
    unittest.main()
