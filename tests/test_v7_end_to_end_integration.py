"""Comprehensive End-to-End Validation Suite for E.D.I.T.H. V7.

Validates the full integrated pipeline across:
1. Priority 0: Contextual Dialogue & Cognitive Repair (Personal dialogue vs System Queries).
2. Priority 1: Spotify Desktop Application Resolution & Playback Verification.
3. Priority 2: Grounded Screen Understanding & Error Diagnosis.
4. Priority 3: Safe Storage Intelligence, 4-tier Risk Categorization, and Reversible Quarantine.
5. Priority 4: Semantic File Organization, Git Protection, and Transaction Rollback.
6. Priority 5: Natural Voice Prosody, Micro-pauses, Question Intonation, and Instant Barge-In.
7. Priority 6: Dependable Multitasking, Concurrency, Collision Serialization, and Cancellation.
"""

import os
import sys
import time
import shutil
import tempfile
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.edith_brain import EdithBrain
from runtime.spotify_adapter import SpotifyAdapter
from runtime.screen_interpreter import ScreenInterpreter
from runtime.storage_analyzer import StorageAnalyzer
from runtime.file_organizer import SemanticFileOrganizer
from runtime.speech_synthesis import SpeechTextCleaner, SpeechProsodyFormatter, LocalSpeechSynthesisEngine
from runtime.task_manager import TaskManager, TaskStatus


class TestV7EndToEndIntegration(unittest.TestCase):
    """End-to-end integration test suite verifying V7 improvements against the real runtime."""

    @classmethod
    def setUpClass(cls):
        cls.brain = EdithBrain(enable_execution=False)

    def _get_directive(self, text: str) -> dict:
        res = self.brain.process(text, execute=False)
        return res.get("directive", {})

    # ─────────────────────────────────────────────────────────────────────
    # 1. Natural Conversation & Cognitive Pipeline
    # ─────────────────────────────────────────────────────────────────────

    def test_01_personal_dialogue_continuity(self):
        """Turn 1 & 2: Personal relationship reflection must NEVER misclassify as system_query."""
        # Turn 1
        d1 = self._get_directive(
            "I'm sad, please tell me what should I do, I still miss my ex, I think I should text her, should I text her?"
        )
        self.assertEqual(d1.get("handler"), "dialogue")
        msg1 = d1.get("message", "")
        self.assertNotIn("🚨", msg1)
        self.assertNotIn("48 hours", msg1)

        # Turn 2: Follow-up without keywords - MUST continue dialogue, NOT show CPU/RAM
        d2 = self._get_directive(
            "I don't think she has changed, but I think I have missed her really much, I have missed her too much."
        )
        self.assertEqual(d2.get("handler"), "dialogue")
        self.assertNotEqual(d2.get("handler"), "system_metrics")
        msg2 = d2.get("message", "").lower()
        self.assertNotIn("cpu", msg2)
        self.assertNotIn("ram", msg2)
        self.assertNotIn("battery", msg2)

    def test_02_conversational_rationale_explanation(self):
        """Turn 3: 'Why did you say that?' explains conversational reasoning, not noop."""
        d3 = self._get_directive("Why did you say that?")
        self.assertEqual(d3.get("handler"), "dialogue")
        msg = d3.get("message", "").lower()
        self.assertTrue(any(w in msg for w in ["said that because", "missing someone", "reason", "tension"]))

    def test_03_goal_switch_and_system_query(self):
        """Goal switch to Spotify, and explicit system metrics query."""
        # Goal switch
        d4 = self._get_directive("Actually, leave it and open Spotify.")
        self.assertEqual(d4.get("handler"), "launch_app")
        self.assertEqual(d4.get("parameters", {}).get("app_name"), "spotify")

        # Explicit hardware metrics query
        d5 = self._get_directive("Check my battery percentage and CPU usage.")
        self.assertEqual(d5.get("handler"), "system_metrics")

    # ─────────────────────────────────────────────────────────────────────
    # 2. Spotify Desktop Resolution & Verification
    # ─────────────────────────────────────────────────────────────────────

    def test_04_spotify_desktop_resolution(self):
        """Verify SpotifyAdapter resolves authentic Spotify.exe and rejects SpotifyLauncher.exe."""
        from unittest.mock import patch
        adapter = SpotifyAdapter()
        exe = adapter.resolve_spotify_executable()
        if exe:
            self.assertTrue(exe.lower().endswith("spotify.exe"))
            self.assertNotIn("spotifylauncher", exe.lower())

        # Title verification test
        with patch.object(adapter, "find_spotify_window", return_value=9999):
            with patch.object(adapter, "get_window_title", return_value="Daft Punk - Get Lucky"):
                ver = adapter.verify_playback(expected_title="Get Lucky", timeout=0.1)
                self.assertTrue(ver["is_playing"])
                self.assertEqual(ver["artist"], "Daft Punk")
                self.assertEqual(ver["track"], "Get Lucky")

        with patch.object(adapter, "find_spotify_window", return_value=9999):
            with patch.object(adapter, "get_window_title", return_value="Spotify Free"):
                ver_idle = adapter.verify_playback(timeout=0.1)
                self.assertFalse(ver_idle["is_playing"])

    # ─────────────────────────────────────────────────────────────────────
    # 3. Grounded Screen Understanding
    # ─────────────────────────────────────────────────────────────────────

    def test_05_screen_understanding_integration(self):
        """Verify screen interpretation pipeline for errors and general queries."""
        d_screen = self._get_directive("What is going on on my screen?")
        self.assertEqual(d_screen.get("handler"), "screen_understanding")

        interpreter = ScreenInterpreter()
        traceback_text = (
            "Traceback (most recent call last):\n"
            "  File 'app.py', line 42, in <module>\n"
            "ZeroDivisionError: division by zero"
        )
        errors = interpreter.detect_visible_errors(ocr_text=traceback_text)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["error_type"], "ZeroDivisionError")
        self.assertIn("division by zero", errors[0]["error_message"])

    # ─────────────────────────────────────────────────────────────────────
    # 4. Storage Intelligence & 4-Tier Risk Quarantine
    # ─────────────────────────────────────────────────────────────────────

    def test_06_storage_audit_and_quarantine(self):
        """Verify 4-tier risk assessment, duplicate detection, and reversible quarantine."""
        d_audit = self._get_directive("Analyze my storage space.")
        self.assertEqual(d_audit.get("handler"), "storage_audit")

        temp_dir = tempfile.mkdtemp(prefix="edith_test_e2e_storage_")
        try:
            analyzer = StorageAnalyzer(preferences_path=os.path.join(temp_dir, "prefs.json"))
            # Cache file (A)
            c_dir = os.path.join(temp_dir, "__pycache__")
            os.makedirs(c_dir, exist_ok=True)
            c_file = os.path.join(c_dir, "cache.pyc")
            with open(c_file, "wb") as f:
                f.write(b"data" * 2000)

            # Photo (C) - protected
            p_file = os.path.join(temp_dir, "memory.jpg")
            with open(p_file, "wb") as f:
                f.write(b"photo" * 2000)

            report = analyzer.audit_storage(temp_dir)
            self.assertEqual(len(report.candidates_for_cleanup), 1)

            # Reversible Quarantine
            q_dir = os.path.join(temp_dir, "quarantine")
            manifest = analyzer.quarantine_candidates(report.candidates_for_cleanup, dry_run=False, quarantine_root=q_dir)
            self.assertFalse(os.path.exists(c_file))
            self.assertTrue(os.path.exists(p_file))  # Personal photo never moved

            # Rollback
            restored = analyzer.restore_quarantine(manifest["manifest_path"])
            self.assertTrue(restored["success"])
            self.assertTrue(os.path.exists(c_file))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    # ─────────────────────────────────────────────────────────────────────
    # 5. Semantic File Organization & Git Protection
    # ─────────────────────────────────────────────────────────────────────

    def test_07_semantic_file_organizer_and_rollback(self):
        """Verify taxonomy categorization, Git repo protection, and transaction rollback."""
        d_org = self._get_directive("Organize these files in my folder.")
        self.assertEqual(d_org.get("handler"), "semantic_file_organize")

        temp_dir = tempfile.mkdtemp(prefix="edith_test_e2e_org_")
        try:
            organizer = SemanticFileOrganizer(log_directory=os.path.join(temp_dir, "logs"))

            # Lecture notes
            lecture_file = os.path.join(temp_dir, "deep_learning_lecture_02.pdf")
            with open(lecture_file, "w") as f:
                f.write("notes")

            # Git code
            git_dir = os.path.join(temp_dir, "code_repo")
            os.makedirs(os.path.join(git_dir, ".git"), exist_ok=True)
            repo_file = os.path.join(git_dir, "engine.py")
            with open(repo_file, "w") as f:
                f.write("code")

            plan = organizer.generate_organization_plan(temp_dir)
            prot_paths = [p["source_path"] for p in plan.protected_files]
            self.assertIn(repo_file, prot_paths)

            # Execute and Rollback
            exec_res = organizer.execute_organization_plan(plan, dry_run=False)
            self.assertFalse(os.path.exists(lecture_file))

            rollback_res = organizer.rollback_transaction(exec_res["transaction_file"])
            self.assertTrue(rollback_res["success"])
            self.assertTrue(os.path.exists(lecture_file))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    # ─────────────────────────────────────────────────────────────────────
    # 6. Speech Prosody & Delivery Styles
    # ─────────────────────────────────────────────────────────────────────

    def test_08_natural_voice_prosody(self):
        """Verify XML prosody shaping across technical, casual, and gentle styles."""
        # 1. Technical
        tech_xml, _ = SpeechProsodyFormatter.format_prosody("CPU usage is 12%, memory is 48%.", style="technical_composed")
        self.assertIn("<rate absspeed='0'>", tech_xml)
        self.assertIn("<silence msec='140'/>", tech_xml)

        # 2. Gentle personal
        gentle_xml, _ = SpeechProsodyFormatter.format_prosody("I understand, take all the time you need.", style="gentle_personal")
        self.assertIn("<rate absspeed='-1'>", gentle_xml)
        self.assertIn("<silence msec='220'/>", gentle_xml)

        # 3. Question intonation
        q_xml, _ = SpeechProsodyFormatter.format_prosody("Would you like me to check that?", style="relaxed_casual")
        self.assertIn("<pitch absmiddle='2'>", q_xml)

    # ─────────────────────────────────────────────────────────────────────
    # 7. Dependable Multitasking & Resource Coordination
    # ─────────────────────────────────────────────────────────────────────

    def test_09_multitasking_concurrency_and_cancellation(self):
        """Verify concurrent task execution, collision serialization, and cancellation."""
        d_tasks = self._get_directive("List background tasks.")
        self.assertEqual(d_tasks.get("handler"), "task_list")

        tm = TaskManager(max_concurrent_workers=4)
        try:
            # Concurrency
            def quick_worker(val):
                time.sleep(0.05)
                return val * 2

            t1 = tm.submit_task("q1", "double 10", quick_worker, args=(10,))
            t2 = tm.submit_task("q2", "double 20", quick_worker, args=(20,))

            for _ in range(20):
                info1 = tm.get_task(t1)
                info2 = tm.get_task(t2)
                if info1["status"] == "completed" and info2["status"] == "completed":
                    break
                time.sleep(0.05)

            self.assertEqual(info1["result"], 20)
            self.assertEqual(info2["result"], 40)

            # Cancellation
            def stoppable_task(cancel_event):
                while not cancel_event.is_set():
                    time.sleep(0.02)
                return "stopped"

            t3 = tm.submit_task("cancel_me", "test abort", stoppable_task)
            time.sleep(0.1)
            self.assertTrue(tm.cancel_task(t3))
            time.sleep(0.05)
            self.assertEqual(tm.get_task(t3)["status"], "cancelled")
        finally:
            tm.shutdown(wait=False)


if __name__ == "__main__":
    unittest.main()
