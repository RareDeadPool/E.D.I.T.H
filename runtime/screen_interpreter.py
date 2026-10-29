"""Unified Screen Interpretation Pipeline for E.D.I.T.H. (V7 Priority Two).

Combines:
- Layer A: High-DPI screen and active window capture.
- Layer B: Windows UI Automation accessibility tree extraction.
- Layer C: Local ONNX OCR (RapidOCR) with bounding boxes and text confidence.
- Layer D: Lightweight visual feature and layout analysis.
- Layer E: Unified Grounded Interpretation:
  Answers questions such as:
  - 'What is going on on my screen?'
  - 'What do you think of this?'
  - 'Why isn't this working?'
  - 'Can you fix this error?'
  - 'Explain what I'm looking at.'
  - 'What should I do next?'
- Layer F: Action planning and post-action verification.
"""

import os
import re
import sys
import time
import ctypes
from ctypes import wintypes
from typing import Dict, Any, List, Optional, Tuple

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runtime.screen_engine import ScreenEngine, ScreenElement


class ScreenInterpreter:
    """Combines UIA accessibility metadata, local OCR, and visual cues for unified screen interpretation."""

    def __init__(self, enable_ocr: bool = True):
        self.engine = ScreenEngine(enable_uia=True, enable_ocr=enable_ocr)
        self.observation_paused: bool = False
        self.last_observation: Optional[Dict[str, Any]] = None

    def pause_observation(self, paused: bool = True):
        """Toggle visual observation for user privacy."""
        self.observation_paused = paused

    def get_foreground_window_info(self) -> Dict[str, Any]:
        """Retrieve active window title, process ID, class name, and bounding rect."""
        hwnd = user32.GetForegroundWindow()
        if not hwnd or not user32.IsWindow(hwnd):
            return {
                "hwnd": 0,
                "title": "",
                "app": "Desktop",
                "rect": (0, 0, 0, 0),
                "is_visible": False
            }

        tbuf = ctypes.create_unicode_buffer(512)
        user32.GetWindowTextW(hwnd, tbuf, 512)
        title = tbuf.value.strip()

        cbuf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, cbuf, 256)
        class_name = cbuf.value.strip()

        rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        bounds = (rect.left, rect.top, rect.right, rect.bottom)

        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))

        # Resolve app name from title or class
        app_name = "Unknown App"
        title_lower = title.lower()
        if "visual studio code" in title_lower or class_name == "Chrome_WidgetWin_1" and "code" in title_lower:
            app_name = "Visual Studio Code"
        elif "chrome" in title_lower or class_name == "Chrome_WidgetWin_1" and "google" in title_lower:
            app_name = "Google Chrome"
        elif "spotify" in title_lower:
            app_name = "Spotify"
        elif "notepad" in title_lower:
            app_name = "Notepad"
        elif "command prompt" in title_lower or "cmd.exe" in title_lower or class_name == "ConsoleWindowClass":
            app_name = "Command Prompt"
        elif "windows terminal" in title_lower or "powershell" in title_lower:
            app_name = "Terminal"
        elif title:
            # Guess from title suffix (e.g. "Document - Word")
            parts = title.split(" - ")
            app_name = parts[-1].strip() if len(parts) > 1 else parts[0].strip()

        return {
            "hwnd": hwnd,
            "title": title,
            "class_name": class_name,
            "app": app_name,
            "rect": bounds,
            "pid": pid.value,
            "is_visible": bool(user32.IsWindowVisible(hwnd))
        }

    def observe(self, hwnd: Optional[int] = None, force_ocr: bool = False) -> Dict[str, Any]:
        """
        Perform unified observation: capture window metadata, extract UIA element tree,
        and run local OCR if UIA metadata is sparse or text-heavy.
        """
        if self.observation_paused:
            return {
                "status": "paused",
                "message": "Visual screen observation is currently paused for privacy.",
                "elements": [],
                "confidence": 0.0
            }

        win_info = self.get_foreground_window_info() if hwnd is None else {"hwnd": hwnd}
        target_hwnd = win_info.get("hwnd", 0)

        t_start = time.time()
        elements = self.engine.perceive(hwnd=target_hwnd, use_ocr=force_ocr)
        latency_ms = (time.time() - t_start) * 1000.0

        # Group elements into semantic categories
        buttons = []
        text_fields = []
        labels = []
        dialogs = []

        visible_text_fragments = []
        for el in elements:
            if el.name:
                visible_text_fragments.append(el.name)
            if el.role in ["button", "link"]:
                buttons.append(el.name)
            elif el.role in ["edit", "textbox"]:
                text_fields.append({"name": el.name, "value": el.value})
            elif el.role in ["dialog", "window", "alert"]:
                dialogs.append(el.name)
            else:
                if el.name and len(el.name) > 1:
                    labels.append(el.name)

        combined_visible_text = "\n".join(visible_text_fragments)

        # Detect error patterns in visible text
        error_info = self._detect_errors_in_text(combined_visible_text, win_info.get("title", ""))

        observation = {
            "window": win_info,
            "element_count": len(elements),
            "buttons": list(dict.fromkeys(buttons))[:15],
            "text_fields": text_fields[:8],
            "labels": list(dict.fromkeys(labels))[:20],
            "dialogs": list(dict.fromkeys(dialogs))[:5],
            "error_detected": error_info.get("has_error", False),
            "error_details": error_info,
            "latency_ms": latency_ms,
            "elements_raw": elements
        }
        self.last_observation = observation
        return observation

    def _detect_errors_in_text(self, text: str, window_title: str) -> Dict[str, Any]:
        """Identify code exceptions, terminal tracebacks, and application error messages."""
        full_text = f"{window_title}\n{text}"

        # 1. Python tracebacks & exceptions
        py_match = re.search(r"([A-Za-z]+Error|Exception|AssertionError|NameError|SyntaxError|TypeError|ValueError|KeyError|ImportError|ModuleNotFoundError):\s*(.+)", full_text)
        if py_match:
            err_type = py_match.group(1).strip()
            err_msg = py_match.group(2).strip().split("\n")[0]
            # Look for file and line
            line_match = re.search(r'File "([^"]+)", line (\d+)', full_text)
            file_loc = f"{line_match.group(1)} (line {line_match.group(2)})" if line_match else "unknown line"
            return {
                "has_error": True,
                "category": "code_exception",
                "error_type": err_type,
                "error_message": err_msg,
                "location": file_loc,
                "summary": f"{err_type}: {err_msg} in {file_loc}"
            }

        # 2. General Windows error / crash dialogs
        for phrase in [
            "an error occurred", "fatal error", "access denied", "not responding",
            "could not find file", "permission denied", "failed to connect", "operation failed"
        ]:
            if phrase in full_text.lower():
                return {
                    "has_error": True,
                    "category": "app_error",
                    "error_type": "Application Error",
                    "error_message": phrase.capitalize(),
                    "location": window_title,
                    "summary": f"Detected application warning/error in '{window_title}': '{phrase}'"
                }

        # 3. Compiler / linter errors
        if "error:" in full_text.lower() or "failed:" in full_text.lower():
            lines = [line.strip() for line in full_text.splitlines() if "error" in line.lower()]
            err_line = lines[0] if lines else "Error in active window"
            return {
                "has_error": True,
                "category": "build_error",
                "error_type": "Build/Test Failure",
                "error_message": err_line[:120],
                "location": window_title,
                "summary": f"Build/Command error: {err_line[:120]}"
            }

        return {"has_error": False, "category": None, "error_type": None, "error_message": None}

    def detect_visible_errors(self, ocr_text: str = "", window_title: str = "") -> List[Dict[str, Any]]:
        """Public interface for detecting visible exceptions or application errors from text."""
        err = self._detect_errors_in_text(ocr_text, window_title)
        if err.get("has_error"):
            return [err]
        return []

    def interpret(self, user_question: str, context: Optional[Any] = None) -> Dict[str, Any]:
        """
        Form a grounded semantic interpretation combining screen evidence and conversation.
        Answers questions about errors, active content, charts, and visible tasks.
        """
        obs = self.observe()
        if obs.get("status") == "paused":
            return {
                "status": "paused",
                "message": "Screen observation is currently paused. Please say 'resume observation' or 'look at screen' to enable visual assistance."
            }

        win = obs.get("window", {})
        app = win.get("app", "your desktop")
        title = win.get("title", "")
        err = obs.get("error_details", {})
        q_lower = user_question.lower()

        # Question Type 1: Error troubleshooting ("Why isn't this working?", "Can you fix this error?")
        if any(w in q_lower for w in ["why isn't this working", "fix this error", "why did this fail", "what is this error", "error"]):
            if err.get("has_error"):
                e_type = err.get("error_type")
                e_msg = err.get("error_message")
                loc = err.get("location")
                solution = self._suggest_fix(e_type, e_msg)
                msg = (
                    f"I see the problem on your screen in {app} ({title}).\n\n"
                    f"**Error**: `{e_type}: {e_msg}`\n"
                    f"**Location**: {loc}\n\n"
                    f"**Analysis & Recommended Fix**:\n{solution}"
                )
                return {
                    "status": "success",
                    "interpreted_target": "error_dialog",
                    "error_type": e_type,
                    "confidence": 0.95,
                    "message": msg
                }
            else:
                msg = (
                    f"Looking at your active window in {app} ({title}), I don't see any explicit crash dialogs or tracebacks. "
                    f"What specific behavior or unexpected outcome are you seeing?"
                )
                return {
                    "status": "uncertain",
                    "interpreted_target": "active_window",
                    "confidence": 0.65,
                    "message": msg
                }

        # Question Type 2: Overview / What am I looking at ("What is going on on my screen?", "Explain what I'm looking at")
        if any(w in q_lower for w in ["what is going on", "explain what i'm looking at", "what am i looking at", "what's on my screen"]):
            btn_str = ", ".join(f"'{b}'" for b in obs.get("buttons", [])[:6])
            desc_parts = [f"You are currently working in **{app}** (window title: *{title}*)."]
            if err.get("has_error"):
                desc_parts.append(f"\n⚠️ **Notice**: There is an active {err.get('error_type')} visible: `{err.get('error_message')}`.")
            if btn_str:
                desc_parts.append(f"\nVisible interactive controls include: {btn_str}.")
            lbls = obs.get("labels", [])[:4]
            if lbls:
                desc_parts.append(f"Main visible sections or items: {', '.join(repr(l) for l in lbls)}.")

            return {
                "status": "success",
                "interpreted_target": "screen_overview",
                "app": app,
                "title": title,
                "confidence": 0.92,
                "message": "\n".join(desc_parts)
            }

        # Question Type 3: Subjective / Grounding query ("What do you think of this?")
        if "what do you think of this" in q_lower or "what do you think" in q_lower:
            # Ground 'this' using visible screen:
            if err.get("has_error"):
                msg = (
                    f"Looking at the active screen in {app}, 'this' appears to be the `{err.get('error_type')}` you just encountered. "
                    f"It's complaining that `{err.get('error_message')}`. "
                    f"Would you like me to guide you through fixing it?"
                )
                return {"status": "success", "interpreted_target": "visible_error", "confidence": 0.90, "message": msg}
            else:
                lbls = obs.get("labels", [])[:3]
                focus_desc = f"centered on {', '.join(repr(l) for l in lbls)}" if lbls else "open"
                msg = (
                    f"In your active window ({app} — *{title}*), you have content {focus_desc}. "
                    f"Could you specify what aspect of this you'd like feedback on — the layout, the code logic, or the results?"
                )
                return {"status": "clarification_needed", "interpreted_target": "active_app_content", "confidence": 0.75, "message": msg}

        # Question Type 4: Chart or data question ("What does this chart tell us?")
        if "chart" in q_lower or "graph" in q_lower:
            labels_found = [l for l in obs.get("labels", []) if any(c.isdigit() or "%" in c for c in l)]
            msg = (
                f"Inspecting the visible visualization in {app} ({title}): "
                f"I can observe the window layout and key data values ({', '.join(labels_found[:5]) if labels_found else 'metrics'}). "
                f"For full graphical trend interpretation, let me know which axis or metric you want to zoom into."
            )
            return {"status": "success", "interpreted_target": "chart", "confidence": 0.85, "message": msg}

        # Default grounded response
        return {
            "status": "success",
            "interpreted_target": "active_window",
            "app": app,
            "title": title,
            "confidence": 0.85,
            "message": f"Looking at your active window **{app}** ({title}), I see {obs.get('element_count')} interactive elements. How can I help you with this application?"
        }

    def _suggest_fix(self, error_type: str, error_message: str) -> str:
        """Provide concrete, actionable solutions for common software and system errors."""
        e_low = (error_type or "").lower()
        m_low = (error_message or "").lower()

        if "modulenotfounderror" in e_low or "no module named" in m_low:
            mod_match = re.search(r"no module named ['\"]([^'\"]+)['\"]", m_low)
            mod_name = mod_match.group(1) if mod_match else "the missing package"
            return (
                f"1. Run `pip install {mod_name}` in your project's active virtual environment.\n"
                f"2. If you are using VS Code, check your Python interpreter path (Ctrl+Shift+P -> 'Select Interpreter') to ensure it points to the environment where dependencies are installed."
            )
        elif "syntaxerror" in e_low:
            return (
                f"1. Check the line indicated in the traceback for missing colons (`:`), unmatched parentheses, brackets, or quotes.\n"
                f"2. In Python 3.12+, check that f-strings do not contain nested backslashes or unsupported escape characters."
            )
        elif "indentationerror" in e_low:
            return "Convert all mixed tabs to spaces (4 spaces per indent level) across the file."
        elif "filenotfounderror" in e_low:
            return "Verify that the relative path is computed from `os.path.abspath(__file__)` or that the working directory is set to your project root."
        elif "typeerror" in e_low:
            return "Inspect argument types being passed into the failing function call — verify whether `None` or an unexpected type was supplied."
        else:
            return f"Review the stack trace above the error line and ensure all preconditions and variables are initialized before execution."

    def interpret_screen(self, user_query: str = "") -> Any:
        """Adapter method for executor and test suites returning grounded interpretation."""
        res = self.interpret(user_query)

        class ScreenInterpretationResult:
            def __init__(self, data: Dict[str, Any]):
                self.data = data
                self.grounded_summary = data.get("message", "")

            def to_dict(self) -> Dict[str, Any]:
                return self.data

        return ScreenInterpretationResult(res)
