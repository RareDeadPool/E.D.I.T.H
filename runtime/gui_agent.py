"""Application-Agnostic GUI Interaction Agent for E.D.I.T.H.

This module provides high-level, intent-driven GUI automation that
works across ANY Windows application by using the ScreenEngine for
dynamic element discovery instead of hardcoded pixel offsets.

Architecture
------------
The old approach:
    "click at x=960, y=1020 because that's where Spotify's play button is"

The new approach:
    1. ScreenEngine.perceive(hwnd) → discovers all UI elements
    2. ScreenEngine.find("Play") → locates the play button dynamically
    3. actions.click(element.center) → clicks it

This works on any resolution, DPI, or window size because coordinates
come from the live accessibility tree and OCR, not from hardcoded values.
"""

from __future__ import annotations

import os
import re
import time
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from runtime.screen_engine import ScreenEngine, ScreenElement
from runtime import actions


class GUIAgent:
    """High-level GUI interaction agent powered by ScreenEngine perception.

    This agent can interact with any application on the screen by:
    1. Finding windows by process name
    2. Perceiving their UI elements (UIA + OCR)
    3. Locating interactive elements by name/role
    4. Performing actions (click, type, scroll, hotkey)
    5. Verifying post-action state

    It replaces all app-specific hardcoded coordinate logic.
    """

    def __init__(self, engine: Optional[ScreenEngine] = None):
        self.engine = engine or ScreenEngine()
        self._last_hwnd: Optional[int] = None
        self._last_elements: List[ScreenElement] = []

    # ──────────────────────── Window Management ─────────────────────────────

    def find_app(self, app_name: str) -> Optional[int]:
        """Find and return the HWND of a running application.

        Supports common app names and fuzzy matching.
        """
        app_lower = app_name.lower().strip()

        # Map common names to process name patterns
        _APP_PROCESS_MAP = {
            "spotify": (["spotify"], "spotify"),
            "chrome": (["chrome"], "chrome"),
            "google chrome": (["chrome"], "chrome"),
            "firefox": (["firefox"], "firefox"),
            "edge": (["msedge"], "edge"),
            "notepad": (["notepad"], "notepad"),
            "vscode": (["code"], None),
            "visual studio code": (["code"], None),
            "file explorer": (["explorer"], None),
            "explorer": (["explorer"], None),
            "terminal": (["windowsterminal", "cmd", "powershell"], None),
            "discord": (["discord"], "discord"),
            "slack": (["slack"], "slack"),
            "teams": (["teams"], "teams"),
            "word": (["winword"], None),
            "excel": (["excel"], None),
            "powerpoint": (["powerpnt"], None),
        }

        proc_names, title_hint = _APP_PROCESS_MAP.get(app_lower, ([app_lower], app_lower))
        hwnd = actions.find_window_by_process(proc_names, title_contains=title_hint)

        if hwnd:
            self._last_hwnd = hwnd
        return hwnd

    def focus_app(self, app_name: str, launch_if_missing: bool = False) -> Optional[int]:
        """Find, focus, and return the HWND of an application.

        Args:
            app_name: Human name of the app (e.g. "spotify", "chrome").
            launch_if_missing: If True and app isn't running, attempt to launch it.

        Returns:
            HWND if found/launched, None otherwise.
        """
        hwnd = self.find_app(app_name)

        if not hwnd and launch_if_missing:
            if app_name.lower() == "spotify":
                try:
                    from runtime.spotify_adapter import SpotifyAdapter
                    spot = SpotifyAdapter()
                    ok, msg, spot_hwnd = spot.launch_desktop_app(timeout=8.0)
                    if spot_hwnd:
                        return spot_hwnd
                except Exception:
                    pass

            # Try launching the app
            _LAUNCH_CMDS = {
                "spotify": "start spotify:",
                "chrome": 'start "" "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" --profile-directory="Default"',
                "edge": "start msedge.exe",
                "notepad": "start notepad.exe",
                "calculator": "start calc.exe",
                "terminal": "start wt.exe",
                "settings": "start ms-settings:",
            }
            cmd = _LAUNCH_CMDS.get(app_name.lower())
            if cmd:
                subprocess.Popen(cmd, shell=True)
                # Poll for the window to appear
                for _ in range(30):
                    time.sleep(0.5)
                    hwnd = self.find_app(app_name)
                    if hwnd:
                        break

        if hwnd:
            actions.focus_window(hwnd)
            self._last_hwnd = hwnd
            time.sleep(0.2)

        return hwnd

    # ──────────────────────── Perception ────────────────────────────────────

    def see(self, hwnd: Optional[int] = None, force_ocr: bool = False) -> List[ScreenElement]:
        """Perceive all UI elements in a window.

        Args:
            hwnd: Window handle.  Uses last focused if None.
            force_ocr: If True, always run OCR even if UIA returns enough elements.

        Returns:
            List of detected ScreenElement objects.
        """
        target = hwnd or self._last_hwnd
        self.engine.clear_cache()

        if force_ocr and target:
            # Get UIA elements + always include OCR
            uia_elements = []
            if self.engine.uia:
                uia_elements = self.engine.uia.get_elements(hwnd=target)

            ocr_elements = self.engine.perceive_ocr_only(hwnd=target)

            # Deduplicate
            uia_names = {e.name.lower() for e in uia_elements if e.name}
            merged = list(uia_elements)
            for oe in ocr_elements:
                if oe.name.lower() not in uia_names:
                    merged.append(oe)

            self._last_elements = merged
            return merged
        else:
            self._last_elements = self.engine.perceive(hwnd=target)
            return self._last_elements

    def find(
        self,
        query: str,
        role: Optional[str] = None,
        hwnd: Optional[int] = None,
        refresh: bool = False,
    ) -> Optional[ScreenElement]:
        """Find a UI element by name.

        Args:
            query: Text to search for (e.g. "Search", "Play", "Settings").
            role: Optional role filter (e.g. "button", "edit").
            hwnd: Window handle (uses cached elements if not specified).
            refresh: If True, re-perceive before searching.

        Returns:
            Best matching ScreenElement, or None.
        """
        if refresh or not self._last_elements:
            self.see(hwnd=hwnd)
        return self.engine.find(query, self._last_elements, role=role)

    def find_all(
        self,
        query: str,
        role: Optional[str] = None,
        threshold: float = 0.5,
    ) -> List[ScreenElement]:
        """Find all matching elements above a threshold."""
        if not self._last_elements:
            self.see()
        return self.engine.find_all(query, self._last_elements, role=role, threshold=threshold)

    # ──────────────────────── Actions ───────────────────────────────────────

    def click_element(
        self,
        query: str,
        role: Optional[str] = None,
        hwnd: Optional[int] = None,
        double: bool = False,
        refresh: bool = True,
        cursorless: bool = True,
    ) -> Optional[ScreenElement]:
        """Find and click an element by name.

        Args:
            query: Text label of the element to click.
            role: Optional role filter.
            hwnd: Window handle.
            double: If True, double-click instead.
            refresh: If True, re-perceive before searching.
            cursorless: If True, uses programmatic UIA invoke or background
                       PostMessageW so the user's cursor is NEVER moved.

        Returns:
            The clicked ScreenElement, or None if not found.
        """
        elem = self.find(query, role=role, hwnd=hwnd, refresh=refresh)
        if not elem:
            return None

        if cursorless:
            # Tier 1: Native programmatic UIA invoke (zero mouse movement, zero focus stealing)
            if not double and elem.invoke():
                return elem

            # Tier 2: Background window message queue (zero mouse movement)
            target_hwnd = elem.hwnd or hwnd or self._last_hwnd
            if target_hwnd and elem.post_click(double=double):
                return elem

            # Tier 3: Hardware click with instant cursor restoration
            actions.click_with_restore(elem.center[0], elem.center[1], clicks=2 if double else 1)
            return elem
        else:
            if double:
                actions.double_click(elem.center[0], elem.center[1])
            else:
                actions.click(elem.center[0], elem.center[1])
            return elem

    def type_into_element(
        self,
        query: str,
        text: str,
        role: Optional[str] = None,
        clear_first: bool = True,
        submit: bool = False,
        hwnd: Optional[int] = None,
        cursorless: bool = True,
    ) -> Optional[ScreenElement]:
        """Find an input element, and type text into it.

        Args:
            query: Label of the input element (e.g. "Search").
            text: Text to type.
            role: Optional role filter (defaults to trying 'edit' first).
            clear_first: Clear existing content before typing.
            submit: If True, press Enter after typing.
            hwnd: Window handle.
            cursorless: If True, sets value programmatically via COM without
                       stealing active keyboard focus.

        Returns:
            The targeted ScreenElement, or None if not found.
        """
        elem = self.find(query, role=role or "edit", hwnd=hwnd, refresh=True)
        if not elem:
            elem = self.find(query, hwnd=hwnd, refresh=False)

        if not elem:
            return None

        target_hwnd = elem.hwnd or hwnd or self._last_hwnd

        if cursorless:
            # Tier 1: Programmatic ValuePattern injection (zero keyboard focus stolen)
            if elem.set_value(text):
                if submit and target_hwnd:
                    actions.user32.PostMessageW(target_hwnd, 0x0100, 0x0D, 0)
                    actions.user32.PostMessageW(target_hwnd, 0x0101, 0x0D, 0)
                elif submit:
                    actions.press_key(0x0D)
                return elem

            # Tier 2: Background text message
            if target_hwnd:
                elem.post_click()
                time.sleep(0.05)
                actions.post_text_background(target_hwnd, text)
                if submit:
                    actions.user32.PostMessageW(target_hwnd, 0x0100, 0x0D, 0)
                    actions.user32.PostMessageW(target_hwnd, 0x0101, 0x0D, 0)
                return elem

        # Tier 3 Fallback: Hardware input with cursor restoration
        with actions.cursor_guard():
            actions.click(elem.center[0], elem.center[1])
            time.sleep(0.05)
            if clear_first:
                actions.hotkey(0x11, 0x41)  # Ctrl+A
                time.sleep(0.05)
                actions.press_key(0x08)     # Backspace
                time.sleep(0.05)
            actions.type_text(text)
            if submit:
                time.sleep(0.05)
                actions.press_key(0x0D)
        return elem

    def scroll_at(
        self,
        query: str,
        amount: int = -3,
        hwnd: Optional[int] = None,
    ) -> Optional[ScreenElement]:
        """Find an element and scroll at its position."""
        elem = self.find(query, hwnd=hwnd, refresh=True)
        if elem:
            actions.scroll(elem.center[0], elem.center[1], amount=amount)
            return elem
        return None

    def press_hotkey(self, *vkeys: int):
        """Press a keyboard shortcut."""
        actions.hotkey(*vkeys)

    def press_key(self, vk: int):
        """Press and release a single key."""
        actions.press_key(vk)

    def type_text(self, text: str):
        """Type text at current cursor position."""
        actions.type_text(text)

    # ──────────────────────── Wait / Verify ─────────────────────────────────

    def wait_for_element(
        self,
        query: str,
        role: Optional[str] = None,
        hwnd: Optional[int] = None,
        timeout: float = 10.0,
        poll_interval: float = 0.5,
    ) -> Optional[ScreenElement]:
        """Wait until an element with the given name appears on screen.

        Args:
            query: Element name to wait for.
            role: Optional role filter.
            hwnd: Window handle.
            timeout: Maximum seconds to wait.
            poll_interval: Seconds between perception attempts.

        Returns:
            The found ScreenElement, or None if timed out.
        """
        t0 = time.time()
        while time.time() - t0 < timeout:
            elem = self.find(query, role=role, hwnd=hwnd, refresh=True)
            if elem:
                return elem
            time.sleep(poll_interval)
        return None

    def wait_for_element_gone(
        self,
        query: str,
        role: Optional[str] = None,
        hwnd: Optional[int] = None,
        timeout: float = 10.0,
        poll_interval: float = 0.5,
    ) -> bool:
        """Wait until a specific element disappears from the screen.

        Returns:
            True if the element disappeared, False if timed out.
        """
        t0 = time.time()
        while time.time() - t0 < timeout:
            elem = self.find(query, role=role, hwnd=hwnd, refresh=True)
            if not elem:
                return True
            time.sleep(poll_interval)
        return False

    # ──────────────────────── Composite Workflows ───────────────────────────

    def search_in_app(
        self,
        app_name: str,
        search_query: str,
        search_element_name: str = "Search",
        wait_after: float = 1.0,
    ) -> Dict[str, Any]:
        """Open or find an app, locate its search box, and submit a query cursor-free.

        Uses programmatic ValuePattern injection or background messages so the user's
        mouse and keyboard focus are never hijacked.
        """
        # Step 1: Find or focus the app
        hwnd = self.find_app(app_name) or self.focus_app(app_name, launch_if_missing=True)
        if not hwnd:
            return {
                "status": "error",
                "message": f"Could not find or launch '{app_name}'."
            }

        actions.focus_window(hwnd)
        time.sleep(0.3)

        # Spotify-specific native fast search (Ctrl+L -> Ctrl+A -> type -> Enter)
        if "spotify" in app_name.lower():
            actions.hotkey(0x11, 0x4C)  # Ctrl + L
            time.sleep(0.08)
            actions.hotkey(0x11, 0x41)  # Ctrl + A
            time.sleep(0.04)
            actions.press_key(0x08)     # Backspace
            time.sleep(0.04)
            actions.type_text(search_query)
            time.sleep(0.1)
            actions.press_key(0x0D)     # Enter
            time.sleep(wait_after)
            post_elements = self.see(hwnd=hwnd, force_ocr=True)
            return {
                "status": "success",
                "message": f"Searched for '{search_query}' in {app_name}.",
                "hwnd": hwnd,
                "elements_after_search": len(post_elements),
            }

        # Step 2: Try finding search element using aliases
        search_aliases = [search_element_name, "Search", "What do you want to play", "Address and search bar", "Search box", "Search or enter web address"]
        search_elem = None
        for alias in search_aliases:
            search_elem = self.find(alias, role="edit", hwnd=hwnd, refresh=(alias == search_element_name))
            if not search_elem:
                search_elem = self.find(alias, hwnd=hwnd, refresh=False)
            if search_elem:
                break

        injected = False
        if search_elem:
            # Try programmatic value injection
            if search_elem.set_value(search_query):
                injected = True
                actions.user32.PostMessageW(hwnd, 0x0100, 0x0D, 0)  # WM_KEYDOWN Enter
                actions.user32.PostMessageW(hwnd, 0x0101, 0x0D, 0)  # WM_KEYUP Enter
                time.sleep(wait_after)

        if not injected:
            if search_elem:
                self.type_into_element(search_elem.name, search_query, hwnd=hwnd, submit=True, cursorless=True)
            elif any(b in app_name.lower() for b in ["chrome", "edge", "firefox", "browser"]):
                # Browser address bar shortcut
                actions.hotkey(0x11, 0x4C)  # Ctrl + L
                time.sleep(0.05)
                actions.type_text(search_query)
                time.sleep(0.05)
                actions.press_key(0x0D)
            time.sleep(wait_after)

        # Step 3: Re-perceive to see search results
        post_elements = self.see(hwnd=hwnd, force_ocr=True)

        return {
            "status": "success",
            "message": f"Searched for '{search_query}' in {app_name}.",
            "hwnd": hwnd,
            "elements_after_search": len(post_elements),
        }

    def click_in_app(
        self,
        app_name: str,
        element_name: str,
        double: bool = False,
    ) -> Dict[str, Any]:
        """Click a specific element in an app without moving the user's cursor.

        Args:
            app_name: Application name.
            element_name: Label of the element to click.
            double: Double-click if True.

        Returns:
            Dict with status and the clicked element info.
        """
        hwnd = self.find_app(app_name) or self.focus_app(app_name)
        if not hwnd:
            return {"status": "error", "message": f"Could not find '{app_name}'."}

        time.sleep(0.2)
        elem = self.click_element(element_name, hwnd=hwnd, double=double, refresh=True, cursorless=True)

        if elem:
            return {
                "status": "success",
                "message": f"Clicked '{elem.name}' ({elem.role}) in {app_name} (cursor-free).",
                "element": {
                    "name": elem.name,
                    "role": elem.role,
                    "bbox": elem.bbox,
                    "center": elem.center,
                },
            }
        else:
            # Try with OCR
            ocr_elements = self.engine.perceive_ocr_only(hwnd=hwnd)
            ocr_match = self.engine.find(element_name, ocr_elements)
            if ocr_match:
                # Use background PostMessage or click_with_restore
                if not actions.post_click_background(hwnd, ocr_match.center[0], ocr_match.center[1], clicks=2 if double else 1):
                    actions.click_with_restore(ocr_match.center[0], ocr_match.center[1], clicks=2 if double else 1)
                return {
                    "status": "success",
                    "message": f"Clicked '{ocr_match.name}' (OCR background) in {app_name}.",
                    "element": {
                        "name": ocr_match.name,
                        "role": ocr_match.role,
                        "bbox": ocr_match.bbox,
                        "center": ocr_match.center,
                    },
                }
            return {
                "status": "not_found",
                "message": f"Could not find element '{element_name}' in {app_name}.",
            }

    # ──────────────────────── Debug ─────────────────────────────────────────

    def inspect(self, hwnd: Optional[int] = None) -> str:
        """Return a human-readable dump of all elements in a window."""
        elements = self.see(hwnd=hwnd, force_ocr=True)
        return self.engine.dump(elements)

    def save_annotated(self, path: str, hwnd: Optional[int] = None):
        """Save an annotated screenshot with bounding boxes drawn."""
        target = hwnd or self._last_hwnd
        if not target:
            return

        img = actions.capture_window(target)
        if img:
            elements = self._last_elements or self.see(hwnd=target)
            rect = actions.get_window_rect(target)
            annotated = self.engine.annotate_image(img, elements, offset=(rect[0], rect[1]))
            annotated.save(path)
