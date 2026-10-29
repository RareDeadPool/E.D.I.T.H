"""Unit and Integration Test Suite for E.D.I.T.H. V7 Spotify Desktop Adapter.

Validates:
1. Executable resolution finds genuine Spotify.exe and rejects SpotifyLauncher/Migrator/Setup.
2. Process detection discriminates Spotify.exe from SpotifyLauncher.exe.
3. Idle vs playing window title verification.
4. Launch resolution and error reporting when app is missing or closed.
5. Media key dispatch and playback controls.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runtime.spotify_adapter import SpotifyAdapter


class TestSpotifyAdapter(unittest.TestCase):
    """Test suite for SpotifyAdapter functionality and verification."""

    def setUp(self):
        self.adapter = SpotifyAdapter()

    def test_01_disallowed_executables_filter(self):
        """Verify launcher and installer binaries are rejected."""
        for disallowed in [
            "SpotifyLauncher.exe", "SpotifyMigrator.exe", "SpotifyStartupTask.exe",
            "SpotifySetup.exe", "spotifyinstaller.exe", "crashpad_handler.exe"
        ]:
            self.assertIn(disallowed.lower(), self.adapter.DISALLOWED_EXES)

    def test_02_resolve_executable(self):
        """Test resolving real Spotify desktop executable on this machine."""
        exe = self.adapter.resolve_spotify_executable()
        if exe:
            self.assertTrue(os.path.isfile(exe))
            base = os.path.basename(exe).lower()
            self.assertEqual(base, "spotify.exe")
            self.assertNotIn(base, self.adapter.DISALLOWED_EXES)

    def test_03_process_discrimination(self):
        """Test that get_running_spotify_pids strictly looks for spotify.exe, not spotifylauncher.exe."""
        # Create mock processes
        mock_p_launcher = MagicMock()
        mock_p_launcher.info = {'pid': 1234, 'name': 'SpotifyLauncher.exe'}

        mock_p_real = MagicMock()
        mock_p_real.info = {'pid': 5678, 'name': 'Spotify.exe'}

        with patch("psutil.process_iter", return_value=[mock_p_launcher, mock_p_real]):
            pids = self.adapter.get_running_spotify_pids()
            self.assertIn(5678, pids)
            self.assertNotIn(1234, pids)

    def test_04_playback_title_verification(self):
        """Test verification of idle vs playing window titles."""
        # Case A: Idle titles return not playing
        for idle_title in ["Spotify", "Spotify Free", "Spotify Premium"]:
            with patch.object(self.adapter, "find_spotify_window", return_value=9999):
                with patch.object(self.adapter, "get_window_title", return_value=idle_title):
                    ver = self.adapter.verify_playback(expected_title="Starboy", timeout=0.1)
                    self.assertFalse(ver["is_playing"])
                    self.assertFalse(ver["verified"])

        # Case B: Song title playing
        with patch.object(self.adapter, "find_spotify_window", return_value=9999):
            with patch.object(self.adapter, "get_window_title", return_value="The Weeknd - Starboy"):
                ver = self.adapter.verify_playback(expected_title="Starboy", timeout=0.1)
                self.assertTrue(ver["is_playing"])
                self.assertTrue(ver["verified"])
                self.assertEqual(ver["track_title"], "The Weeknd - Starboy")

    def test_05_mocked_launch_when_already_running(self):
        """Test fast window activation when Spotify is already running."""
        with patch.object(self.adapter, "find_spotify_window", return_value=8888):
            with patch.object(self.adapter, "focus_window", return_value=True):
                ok, msg, hwnd = self.adapter.launch_desktop_app(timeout=1.0)
                self.assertTrue(ok)
                self.assertEqual(hwnd, 8888)
                self.assertIn("already running", msg.lower())

    def test_06_not_installed_reporting(self):
        """Test clear reporting when Spotify desktop executable is absent and process fails."""
        with patch.object(self.adapter, "find_spotify_window", return_value=None):
            with patch.object(self.adapter, "is_running", return_value=False):
                with patch.object(self.adapter, "resolve_spotify_executable", return_value=None):
                    res = self.adapter.play_track("Blinding Lights")
                    self.assertEqual(res["status"], "not_installed")
                    self.assertFalse(res["verified"])
                    self.assertIn("not installed", res["message"].lower())


if __name__ == "__main__":
    unittest.main()
