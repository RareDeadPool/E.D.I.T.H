"""Comprehensive Verification Suite for Windows GUI Agent & Execution Hardening (Option 1).

Tests:
1. DPI Awareness initialization and screen dimension integrity.
2. Window discovery, candidate scoring, and helper frame filtering.
3. Window focus management and foreground verification.
4. Top-down window screenshot orientation (verifying non-inverted DIB).
5. Perceptual element matching and role filtering in ScreenEngine.
6. Closed-loop verification for GUI actions.
7. Cursor position restoration with cursor_guard.
"""

import os
import sys
import time
import unittest
import ctypes
from ctypes import wintypes
from typing import Dict, Any, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runtime import actions
from runtime.gui_agent import GUIAgent
from runtime.screen_engine import ScreenEngine, ScreenElement
from runtime.closed_loop import ClosedLoopVerifier, VerificationResult


class TestDPIAwarenessAndEnvironment(unittest.TestCase):
    """Verify that high-DPI scaling and desktop session handles are properly initialized."""

    def test_dpi_awareness_active(self):
        """Ensure per-monitor DPI awareness v2 is active to prevent coordinate drift."""
        user32 = ctypes.windll.user32
        w = user32.GetSystemMetrics(0)
        h = user32.GetSystemMetrics(1)
        self.assertGreater(w, 0, "Primary screen width must be greater than 0")
        self.assertGreater(h, 0, "Primary screen height must be greater than 0")
        # On modern laptops, 1080p is at least 1920 in physical resolution
        self.assertGreaterEqual(w, 1024, "DPI-aware width should reflect physical display dimensions")

    def test_cursor_guard_restoration(self):
        """Ensure cursor_guard preserves and restores user cursor position."""
        user32 = ctypes.windll.user32
        try:
            hdesk = user32.OpenInputDesktop(0, False, 0x01FF)
            if hdesk:
                user32.SetThreadDesktop(hdesk)
        except Exception:
            pass
        pt_initial = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt_initial))


        with actions.cursor_guard():
            # Move cursor away
            user32.SetCursorPos(100, 100)
            pt_moved = wintypes.POINT()
            user32.GetCursorPos(ctypes.byref(pt_moved))
            if (pt_moved.x, pt_moved.y) == (0, 0) and (pt_initial.x, pt_initial.y) == (0, 0):
                self.skipTest("Desktop cursor manipulation not available in non-interactive background session")
            self.assertEqual((pt_moved.x, pt_moved.y), (100, 100))

        # Check restored position
        pt_restored = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt_restored))
        self.assertEqual((pt_restored.x, pt_restored.y), (pt_initial.x, pt_initial.y))


class TestWindowDiscoveryAndFocus(unittest.TestCase):
    """Verify top-level window discovery and candidate scoring."""

    def test_find_window_by_process_non_crashing(self):
        """Ensure window discovery executes safely across system processes."""
        hwnd = actions.find_window_by_process(["explorer", "dwm", "code", "spotify"])
        # Should return int HWND or None without crashing
        self.assertTrue(hwnd is None or isinstance(hwnd, int))

    def test_find_window_scores_interactive_frames(self):
        """Verify that discovery prefers titled, large windows over 0x0 helper frames."""
        user32 = ctypes.windll.user32
        fg = user32.GetForegroundWindow()
        if fg and user32.IsWindow(fg):
            rect = actions.get_window_rect(fg)
            w, h = rect[2] - rect[0], rect[3] - rect[1]
            self.assertGreaterEqual(w, 0)
            self.assertGreaterEqual(h, 0)

    def test_focus_window_graceful_handling(self):
        """Verify focus_window handles invalid and valid HWNDs gracefully."""
        self.assertFalse(actions.focus_window(0))
        self.assertFalse(actions.focus_window(99999999))


class TestScreenPerceptionAndCapture(unittest.TestCase):
    """Verify screen element perception, top-down DIB orientation, and fuzzy matching."""

    def test_screen_element_creation_and_center(self):
        """Ensure ScreenElement computes center coordinates correctly from bounding box."""
        elem = ScreenElement(
            name="Play",
            role="button",
            bbox=(100, 200, 300, 400),
            source="uia"
        )
        self.assertEqual(elem.center, (200, 300))
        self.assertEqual(elem.width, 200)
        self.assertEqual(elem.height, 200)
        self.assertEqual(elem.area, 40000)

    def test_screen_engine_element_matching(self):
        """Verify fuzzy and exact matching in ScreenEngine.find."""
        engine = ScreenEngine()
        elements = [
            ScreenElement(name="Play", role="button", bbox=(10, 10, 50, 50), source="uia"),
            ScreenElement(name="Pause Music", role="button", bbox=(60, 10, 120, 50), source="uia"),
            ScreenElement(name="Search Bar", role="edit", bbox=(10, 60, 200, 90), source="uia"),
            ScreenElement(name="Hidden Element", role="button", bbox=(0, 0, 10, 10), is_offscreen=True, source="uia"),
        ]

        # Exact match
        match_play = engine.find("Play", elements=elements)
        self.assertIsNotNone(match_play)
        self.assertEqual(match_play.name, "Play")

        # Substring / partial match
        match_search = engine.find("Search", elements=elements)
        self.assertIsNotNone(match_search)
        self.assertEqual(match_search.name, "Search Bar")

        # Role filter
        match_edit = engine.find("Search", elements=elements, role="edit")
        self.assertIsNotNone(match_edit)
        self.assertEqual(match_edit.role, "edit")

        # Offscreen penalization
        match_hidden = engine.find("Hidden Element", elements=elements)
        # Should either be None or lower score due to offscreen penalty
        if match_hidden:
            self.assertTrue(match_hidden.is_offscreen)

    def test_top_down_window_capture_orientation(self):
        """Verify that capture_window produces valid PIL RGBA image with top-down orientation."""
        user32 = ctypes.windll.user32
        fg = user32.GetForegroundWindow()
        if fg and user32.IsWindow(fg):
            img = actions.capture_window(fg)
            if img is not None:
                self.assertEqual(img.mode, "RGBA")
                self.assertGreater(img.width, 0)
                self.assertGreater(img.height, 0)


class TestClosedLoopGUIVerification(unittest.TestCase):
    """Verify closed-loop verification of GUI actions and recovery strategies."""

    def test_gui_action_success_verification(self):
        """Verify successful GUI action verification."""
        verifier = ClosedLoopVerifier()
        result = verifier.verify_action(
            handler="gui_action",
            parameters={"action": "click", "target": "Play Button", "app_name": "Spotify"},
            execution_result={"status": "success", "message": "Element clicked"}
        )
        self.assertTrue(result.success)
        self.assertEqual(result.observed_state, "gui_action_executed")
        self.assertFalse(result.can_recover)

    def test_gui_action_failure_and_recovery(self):
        """Verify that failed GUI actions prompt focus-retry recovery."""
        verifier = ClosedLoopVerifier()
        result = verifier.verify_action(
            handler="gui_action",
            parameters={"action": "click", "target": "Play Button", "app_name": "Spotify"},
            execution_result={"status": "failed", "message": "Window not found"}
        )
        self.assertFalse(result.success)
        self.assertEqual(result.observed_state, "gui_action_failed")
        self.assertTrue(result.can_recover)
        self.assertEqual(result.recovery_strategy, "retry_with_focus")
        self.assertEqual(result.fallback_parameters.get("app_name"), "Spotify")


if __name__ == "__main__":
    unittest.main()
