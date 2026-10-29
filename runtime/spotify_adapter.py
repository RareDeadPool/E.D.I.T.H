"""Robust, Production-Grade Spotify Desktop Adapter for E.D.I.T.H.

Resolves installed desktop Spotify application, strictly filters out launchers/installers/bootstrappers,
detects running Spotify processes, waits for window readiness, executes playback actions, and
verifies actual playback state via native Win32 window metadata and media key control.
"""

import os
import re
import sys
import time
import glob
import ctypes
import shutil
import urllib.parse
import subprocess
from ctypes import wintypes
from typing import Dict, Any, Optional, List, Tuple

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

# Virtual Key Codes
VK_CONTROL = 0x11
VK_RETURN = 0x0D
VK_BACK = 0x08
VK_SPACE = 0x20
VK_KEY_A = 0x41
VK_KEY_L = 0x4C
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_MEDIA_STOP = 0xB2
VK_MEDIA_PLAY_PAUSE = 0xB3


def _send_key(vk: int, up: bool = False):
    flags = 0x0002 if up else 0
    user32.keybd_event(vk, 0, flags, 0)


def press_media_key(vk: int):
    """Dispatch a media hardware keypress event."""
    _send_key(vk, up=False)
    time.sleep(0.04)
    _send_key(vk, up=True)
    time.sleep(0.04)


class SpotifyAdapter:
    """Robust manager for Spotify desktop application launching, playback and state verification."""

    DISALLOWED_EXES = {
        "spotifylauncher.exe", "spotifymigrator.exe", "spotifystartuptask.exe",
        "spotifysetup.exe", "spotifyinstaller.exe", "crashpad_handler.exe",
        "save-to-spotify.exe", "spotifyxboxgamebarwebview.exe"
    }

    IDLE_TITLES = {"spotify", "spotify free", "spotify premium", "spotify desktop"}

    def __init__(self):
        self.cached_exe_path: Optional[str] = None

    def resolve_spotify_executable(self) -> Optional[str]:
        """
        Locate the actual Spotify.exe desktop binary on the current Windows installation.
        Filters out installers, bootstrappers, launchers, and migrators.
        """
        if self.cached_exe_path and os.path.isfile(self.cached_exe_path):
            return self.cached_exe_path

        candidates: List[str] = []

        # 1. Check running process path first (if already running)
        try:
            import psutil
            for p in psutil.process_iter(['name', 'exe']):
                pname = (p.info['name'] or '').lower()
                if pname == "spotify.exe":
                    pexe = p.info.get('exe')
                    if pexe and os.path.isfile(pexe):
                        base = os.path.basename(pexe).lower()
                        if base not in self.DISALLOWED_EXES:
                            self.cached_exe_path = pexe
                            return pexe
        except Exception:
            pass

        # 2. Windows App Execution Alias (default for modern Store Spotify)
        which_path = shutil.which("spotify")
        if which_path and os.path.isfile(which_path):
            base = os.path.basename(which_path).lower()
            if base not in self.DISALLOWED_EXES:
                candidates.append(which_path)

        local_app_data = os.environ.get("LOCALAPPDATA", "")
        if local_app_data:
            app_alias = os.path.join(local_app_data, "Microsoft", "WindowsApps", "Spotify.exe")
            if os.path.isfile(app_alias):
                candidates.append(app_alias)

        # 3. Standard Roaming AppData Win32 installation
        app_data = os.environ.get("APPDATA", "")
        if app_data:
            roaming_exe = os.path.join(app_data, "Spotify", "Spotify.exe")
            if os.path.isfile(roaming_exe):
                candidates.append(roaming_exe)

        # 4. WindowsApps Package directory
        prog_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        windows_apps_glob = os.path.join(prog_files, "WindowsApps", "SpotifyAB.SpotifyMusic_*", "Spotify.exe")
        for match in glob.glob(windows_apps_glob):
            base = os.path.basename(match).lower()
            if base not in self.DISALLOWED_EXES:
                candidates.append(match)

        # 5. Traditional Program Files
        candidates.extend([
            os.path.join(prog_files, "Spotify", "Spotify.exe"),
            os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "Spotify", "Spotify.exe"),
            os.path.join(local_app_data, "Spotify", "Spotify.exe"),
        ])

        for path in candidates:
            if os.path.isfile(path):
                base = os.path.basename(path).lower()
                # Ensure it is genuinely Spotify.exe and not a bootstrapper/launcher
                if base == "spotify.exe" and base not in self.DISALLOWED_EXES:
                    # Exclude downloads, temp or desktop installers
                    path_low = path.lower()
                    if not any(bad in path_low for bad in ["downloads", "temp", "tmp", "setup", "installer"]):
                        self.cached_exe_path = path
                        return path

        return None

    def get_running_spotify_pids(self) -> List[int]:
        """Return list of active process IDs running the actual Spotify.exe (excluding launchers)."""
        pids = []
        try:
            import psutil
            for p in psutil.process_iter(['pid', 'name']):
                pname = (p.info['name'] or '').lower()
                if pname == "spotify.exe":
                    pids.append(p.info['pid'])
        except Exception:
            pass
        return pids

    def is_running(self) -> bool:
        """Check whether Spotify.exe desktop process is currently active."""
        return len(self.get_running_spotify_pids()) > 0

    def find_spotify_window(self) -> Optional[int]:
        """Find and return HWND for the main interactive Spotify desktop window."""
        spot_pids = set(self.get_running_spotify_pids())
        candidates = []

        def enum_win(hwnd, _):
            if not user32.IsWindow(hwnd):
                return True
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if spot_pids and pid.value not in spot_pids:
                return True

            cbuf = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd, cbuf, 256)
            cls_name = cbuf.value

            tbuf = ctypes.create_unicode_buffer(512)
            user32.GetWindowTextW(hwnd, tbuf, 512)
            title = tbuf.value.strip()

            rect = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            w = rect.right - rect.left
            h = rect.bottom - rect.top

            # Main interactive Spotify window criteria
            if w >= 300 and h >= 200:
                is_vis = user32.IsWindowVisible(hwnd)
                score = w * h
                if is_vis:
                    score += 10000000
                if cls_name in ["Chrome_WidgetWin_1", "Spotify"]:
                    score += 5000000
                if title:
                    score += 2000000
                candidates.append((score, hwnd, title))
            return True

        WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        user32.EnumWindows(WNDENUMPROC(enum_win), 0)

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]
        return None

    def get_window_title(self, hwnd: Optional[int] = None) -> str:
        """Get the current top-level window title of Spotify."""
        h = hwnd or self.find_spotify_window()
        if not h:
            return ""
        tbuf = ctypes.create_unicode_buffer(512)
        user32.GetWindowTextW(h, tbuf, 512)
        return tbuf.value.strip()

    def focus_window(self, hwnd: int) -> bool:
        """Reliably bring Spotify window to the foreground."""
        try:
            user32.AllowSetForegroundWindow(0xFFFFFFFF)
            if user32.IsIconic(hwnd):
                user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            else:
                user32.ShowWindow(hwnd, 5)  # SW_SHOW

            # Alt key trick to bypass Windows foreground lock
            user32.keybd_event(0x12, 0, 0, 0)
            user32.keybd_event(0x12, 0, 0x0002, 0)
            user32.SetForegroundWindow(hwnd)
            return True
        except Exception:
            return False

    def launch_desktop_app(self, timeout: float = 10.0) -> Tuple[bool, str, Optional[int]]:
        """
        Open or activate the installed Spotify desktop app.
        Returns: (success: bool, status_message: str, hwnd: Optional[int])
        """
        # 1. Check if already running with a main window
        existing_hwnd = self.find_spotify_window()
        if existing_hwnd:
            self.focus_window(existing_hwnd)
            return (True, "Spotify is already running and focused.", existing_hwnd)

        # 2. Resolve genuine desktop executable
        exe_path = self.resolve_spotify_executable()
        if exe_path:
            try:
                subprocess.Popen([exe_path], shell=False)
            except Exception as e:
                # Fallback to protocol URI
                os.system("start spotify:")
        else:
            # Fallback to registered Windows protocol URI
            os.system("start spotify:")

        # 3. Wait for Spotify desktop process and window readiness
        t0 = time.time()
        ready_hwnd = None
        while time.time() - t0 < timeout:
            time.sleep(0.3)
            hwnd = self.find_spotify_window()
            if hwnd:
                ready_hwnd = hwnd
                break

        if ready_hwnd:
            self.focus_window(ready_hwnd)
            return (True, "Spotify desktop application launched and ready.", ready_hwnd)
        elif self.is_running():
            return (True, "Spotify process is active, waiting for GUI mount.", None)
        else:
            return (False, "Spotify desktop application is not installed or failed to start.", None)

    def play_track(self, query: str, platform: str = "spotify") -> Dict[str, Any]:
        """
        Search and initiate playback for a song query on Spotify desktop.
        Guarantees that the real desktop application is used and tracks resulting playback state.
        """
        clean_query = query.strip()
        # Launch or focus Spotify desktop
        ok, msg, hwnd = self.launch_desktop_app(timeout=10.0)
        if not ok and not self.is_running():
            return {
                "status": "not_installed",
                "verified": False,
                "message": "Spotify desktop application is not installed on this computer. Please install it from spotify.com or Microsoft Store.",
                "app": "spotify"
            }

        # Wait briefly for Spotify Electron interface hydration if it just launched
        time.sleep(0.6)
        if not hwnd:
            hwnd = self.find_spotify_window()

        if hwnd:
            self.focus_window(hwnd)
            time.sleep(0.2)

        if clean_query:
            # Method A: Navigate directly using Spotify Search URI
            encoded = urllib.parse.quote_plus(clean_query)
            os.system(f"start spotify:search:{encoded}")
            time.sleep(1.2)

            # Re-focus and press enter or space to begin playback on the top result
            if hwnd or self.find_spotify_window():
                h = hwnd or self.find_spotify_window()
                self.focus_window(h)
                time.sleep(0.3)
                # Press Enter to start playing top track in search results
                press_media_key(VK_RETURN)
                time.sleep(0.4)
                press_media_key(VK_MEDIA_PLAY_PAUSE)
        else:
            # Play / Resume current track
            press_media_key(VK_MEDIA_PLAY_PAUSE)

        # Verification step
        ver = self.verify_playback(expected_title=clean_query, timeout=3.5)
        return {
            "status": "success" if ver.get("is_playing") else "playback_initiated",
            "verified": ver.get("verified", False),
            "is_playing": ver.get("is_playing", False),
            "current_track": ver.get("track_title", clean_query),
            "window_title": ver.get("raw_window_title", ""),
            "message": f"Playing '{ver.get('track_title') or clean_query}' on Spotify." if ver.get("is_playing") else f"Sent playback command for '{clean_query}' to Spotify."
        }

    def verify_playback(self, expected_title: str = "", timeout: float = 3.5) -> Dict[str, Any]:
        """
        Verify whether Spotify is actively playing music.
        When Spotify is playing, its top-level window title displays:
        '{Artist} - {Track}' or '{Track} - {Artist}'.
        When paused or stopped, it displays 'Spotify' or 'Spotify Premium' or 'Spotify Free'.
        """
        t0 = time.time()
        last_title = ""
        while time.time() - t0 < timeout:
            hwnd = self.find_spotify_window()
            if hwnd:
                title = self.get_window_title(hwnd)
                last_title = title
                title_low = title.lower()

                # If window title is not the default idle title, a track is playing!
                if title and title_low not in self.IDLE_TITLES and len(title) > 2:
                    artist = None
                    track = title
                    if " - " in title:
                        parts = title.split(" - ", 1)
                        artist = parts[0].strip()
                        track = parts[1].strip()
                    return {
                        "verified": True,
                        "is_playing": True,
                        "track_title": title,
                        "artist": artist,
                        "track": track,
                        "raw_window_title": title,
                        "method": "window_title_inspection"
                    }
            time.sleep(0.3)

        return {
            "verified": False,
            "is_playing": False,
            "track_title": expected_title,
            "artist": None,
            "track": expected_title,
            "raw_window_title": last_title,
            "method": "window_title_inspection"
        }

    def control_playback(self, action: str) -> Dict[str, Any]:
        """Control playback state: play, pause, resume, next, previous, stop."""
        act = action.strip().lower()
        if act in ["next", "skip"]:
            press_media_key(VK_MEDIA_NEXT_TRACK)
            msg = "Skipped to next track on Spotify."
        elif act in ["prev", "previous", "back"]:
            press_media_key(VK_MEDIA_PREV_TRACK)
            msg = "Returned to previous track on Spotify."
        elif act in ["stop"]:
            press_media_key(VK_MEDIA_STOP)
            msg = "Stopped Spotify playback."
        else:  # play, pause, resume, toggle
            press_media_key(VK_MEDIA_PLAY_PAUSE)
            msg = "Toggled Spotify playback."

        time.sleep(0.4)
        ver = self.verify_playback(timeout=1.5)
        return {
            "status": "success",
            "action": act,
            "verified": ver.get("verified", False),
            "is_playing": ver.get("is_playing", False),
            "current_track": ver.get("track_title", ""),
            "message": msg
        }
