"""Advanced System Utilities & Extended Task Functions for E.D.I.T.H. V7.

Provides:
1. WindowController: Maximize, minimize, restore, snap left/right, switch/focus, and close active or targeted application windows.
2. TimerManager: Non-blocking background timers, countdowns, and scheduled alerts with audio notifications.
3. NotesManager: Persistent local scratchpad and quick-notes store (~/.edith_notes.json).
4. WeatherClient: Real-time environmental and weather metrics using zero-key local/wttr endpoints.
5. UnitConverter: Precise unit and currency conversion (length, mass, temperature, data storage, time).
6. ProcessInspector: Enumeration of active interactive desktop applications and IP diagnostics.
"""

import os
import sys
import re
import json
import time
import math
import socket
import datetime
import threading
import urllib.request
import urllib.parse
from typing import Dict, Any, List, Optional, Tuple, Union

try:
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.windll.user32
except Exception:
    user32 = None

try:
    import winsound
except Exception:
    winsound = None

try:
    import psutil
except Exception:
    psutil = None

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class WindowController:
    """Precision Win32 Window Management and Placement Controller."""

    SW_HIDE = 0
    SW_SHOWNORMAL = 1
    SW_SHOWMINIMIZED = 2
    SW_MAXIMIZE = 3
    SW_SHOWNOACTIVATE = 4
    SW_SHOW = 5
    SW_MINIMIZE = 6
    SW_SHOWMINNOACTIVE = 7
    SW_SHOWNA = 8
    SW_RESTORE = 9

    @classmethod
    def get_active_window(cls) -> int:
        """Return HWND of foreground window."""
        if not user32:
            return 0
        try:
            return user32.GetForegroundWindow()
        except Exception:
            return 0

    @classmethod
    def get_window_title(cls, hwnd: int) -> str:
        """Return title text of a given window handle."""
        if not user32 or not hwnd:
            return ""
        buf = ctypes.create_unicode_buffer(512)
        try:
            user32.GetWindowTextW(hwnd, buf, 512)
            return buf.value
        except Exception:
            return ""

    @classmethod
    def list_open_windows(cls) -> List[Dict[str, Any]]:
        """Enumerate all interactive top-level desktop application windows."""
        if not user32:
            return []

        results: List[Dict[str, Any]] = []

        def enum_proc(hwnd, lParam):
            try:
                if user32.IsWindowVisible(hwnd):
                    length = user32.GetWindowTextLengthW(hwnd)
                    if length > 0:
                        buf = ctypes.create_unicode_buffer(length + 1)
                        user32.GetWindowTextW(hwnd, buf, length + 1)
                        title = buf.value.strip()

                        # Skip Windows system background overlays
                        if title and title not in [
                            "Program Manager", "Settings", "Windows Input Experience",
                            "Default IME", "MSCTFIME UI"
                        ]:
                            rect = wintypes.RECT()
                            user32.GetWindowRect(hwnd, ctypes.byref(rect))
                            width = rect.right - rect.left
                            height = rect.bottom - rect.top
                            if width > 50 and height > 50:
                                pid = wintypes.DWORD()
                                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                                pname = ""
                                if psutil:
                                    try:
                                        pname = psutil.Process(pid.value).name()
                                    except Exception:
                                        pass
                                results.append({
                                    "hwnd": hwnd,
                                    "title": title,
                                    "pid": pid.value,
                                    "process": pname,
                                    "width": width,
                                    "height": height
                                })
            except Exception:
                pass
            return True

        WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        try:
            user32.EnumWindows(WNDENUMPROC(enum_proc), 0)
        except Exception:
            pass

        return results

    @classmethod
    def find_window_by_name(cls, app_or_title: str) -> Optional[int]:
        """Find window handle by application or window title."""
        target = app_or_title.lower().strip()
        windows = cls.list_open_windows()

        # 1. Exact or substring match in process name
        for w in windows:
            if target in w["process"].lower() or w["process"].lower() in target:
                return w["hwnd"]

        # 2. Match in window title
        for w in windows:
            if target in w["title"].lower():
                return w["hwnd"]

        # 3. Fallback to actions.find_window_by_process if available
        try:
            from runtime import actions
            found = actions.find_window_by_process([target, f"{target}.exe"])
            if found:
                return found
        except Exception:
            pass

        return None

    @classmethod
    def execute_action(cls, action: str, target_app: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute window operation: maximize, minimize, restore, snap_left, snap_right, switch_to, close.
        """
        act = action.lower().strip()
        hwnd = 0

        if target_app:
            hwnd = cls.find_window_by_name(target_app)
            if not hwnd:
                return {
                    "status": "error",
                    "action": f"window_{act}",
                    "message": f"Could not find open application window matching '{target_app}'."
                }
        else:
            hwnd = cls.get_active_window()

        title = cls.get_window_title(hwnd) if hwnd else "Active Window"

        try:
            from runtime import actions

            # 1. Focus / Switch To
            if act in ["switch", "focus", "bring_to_front", "switch_to"]:
                if hwnd:
                    actions.focus_window(hwnd)
                    return {
                        "status": "success",
                        "action": "window_switch",
                        "hwnd": hwnd,
                        "title": title,
                        "message": f"Switched to '{title or target_app}'."
                    }
                return {"status": "error", "message": "No window to switch to."}

            # 2. Maximize
            elif act in ["maximize", "max", "fullscreen"]:
                if hwnd:
                    user32.ShowWindow(hwnd, cls.SW_MAXIMIZE)
                    return {
                        "status": "success",
                        "action": "window_maximize",
                        "hwnd": hwnd,
                        "title": title,
                        "message": f"Maximized '{title or 'window'}'."
                    }
                else:
                    actions.hotkey("win", "up")
                    return {"status": "success", "action": "window_maximize", "message": "Maximized active window."}

            # 3. Minimize
            elif act in ["minimize", "min"]:
                if hwnd:
                    user32.ShowWindow(hwnd, cls.SW_MINIMIZE)
                    return {
                        "status": "success",
                        "action": "window_minimize",
                        "hwnd": hwnd,
                        "title": title,
                        "message": f"Minimized '{title or 'window'}'."
                    }
                else:
                    actions.hotkey("win", "down")
                    return {"status": "success", "action": "window_minimize", "message": "Minimized active window."}

            # 4. Restore
            elif act in ["restore", "unmaximize"]:
                if hwnd:
                    user32.ShowWindow(hwnd, cls.SW_RESTORE)
                    return {
                        "status": "success",
                        "action": "window_restore",
                        "hwnd": hwnd,
                        "title": title,
                        "message": f"Restored '{title or 'window'}' to normal size."
                    }
                else:
                    actions.hotkey("win", "down")
                    return {"status": "success", "action": "window_restore", "message": "Restored active window."}

            # 5. Snap Left
            elif act in ["snap_left", "tile_left", "left"]:
                if hwnd:
                    actions.focus_window(hwnd)
                actions.hotkey("win", "left")
                return {
                    "status": "success",
                    "action": "window_snap_left",
                    "message": f"Snapped '{title or 'window'}' to the left half of the screen."
                }

            # 6. Snap Right
            elif act in ["snap_right", "tile_right", "right"]:
                if hwnd:
                    actions.focus_window(hwnd)
                actions.hotkey("win", "right")
                return {
                    "status": "success",
                    "action": "window_snap_right",
                    "message": f"Snapped '{title or 'window'}' to the right half of the screen."
                }

            # 7. Close Window
            elif act in ["close", "close_window"]:
                if hwnd:
                    WM_CLOSE = 0x0010
                    user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
                    return {
                        "status": "success",
                        "action": "window_close",
                        "message": f"Closed window '{title or 'active window'}'."
                    }
                else:
                    actions.hotkey("alt", "f4")
                    return {"status": "success", "action": "window_close", "message": "Closed active window."}

            return {"status": "error", "message": f"Unrecognized window action: '{action}'."}

        except Exception as e:
            return {"status": "error", "message": f"Window action '{action}' failed: {e}"}

    @classmethod
    def snap_window(cls, direction: str = "left", target_app: Optional[str] = None) -> Dict[str, Any]:
        """Convenience method to snap window left or right."""
        return cls.execute_action(f"snap_{direction}", target_app)

    @classmethod
    def maximize_window(cls, target_app: Optional[str] = None) -> Dict[str, Any]:
        """Convenience method to maximize window."""
        return cls.execute_action("maximize", target_app)

    @classmethod
    def minimize_window(cls, target_app: Optional[str] = None) -> Dict[str, Any]:
        """Convenience method to minimize window."""
        return cls.execute_action("minimize", target_app)

    @classmethod
    def restore_window(cls, target_app: Optional[str] = None) -> Dict[str, Any]:
        """Convenience method to restore window."""
        return cls.execute_action("restore", target_app)

    @classmethod
    def close_window(cls, target_app: Optional[str] = None) -> Dict[str, Any]:
        """Convenience method to close window."""
        return cls.execute_action("close", target_app)


class TimerManager:
    """Background Multi-Timer and Reminder Scheduling Engine."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TimerManager, cls).__new__(cls)
                cls._instance._timers = {}
                cls._instance._timer_counter = 1
            return cls._instance

    @staticmethod
    def parse_duration_string(text: str) -> Optional[float]:
        """
        Extract total duration in seconds from natural speech or text.
        e.g., '5 minutes', '30 seconds', '1 minute 30 seconds', '2 hours', '45 secs'.
        """
        t = text.lower()
        if "half an hour" in t or "half hour" in t:
            return 1800.0
        if "quarter of an hour" in t or "quarter hour" in t:
            return 900.0

        total_seconds = 0.0
        found = False

        # Hours
        m_hr = re.search(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)\b", t)
        if m_hr:
            total_seconds += float(m_hr.group(1)) * 3600.0
            found = True
        elif re.search(r"\b(?:an|one|a)\s+hour\b", t):
            total_seconds += 3600.0
            found = True

        # Minutes
        m_min = re.search(r"(\d+(?:\.\d+)?)\s*(?:minutes?|mins?|m)\b", t)
        if m_min:
            total_seconds += float(m_min.group(1)) * 60.0
            found = True
        elif re.search(r"\b(?:a|one)\s+minute\b", t):
            total_seconds += 60.0
            found = True

        # Seconds
        m_sec = re.search(r"(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b", t)
        if m_sec:
            total_seconds += float(m_sec.group(1))
            found = True

        # Bare number if command is like "set a timer for 10" (default minutes if > 0)
        if not found:
            m_bare = re.search(r"\b(?:for|timer)\s+(\d+(?:\.\d+)?)$", t)
            if m_bare:
                total_seconds = float(m_bare.group(1)) * 60.0
                found = True

        return total_seconds if found and total_seconds > 0 else None

    def set_timer(
        self,
        duration_seconds: float,
        label: str = "Timer",
        on_complete_callback=None
    ) -> Dict[str, Any]:
        """Start an asynchronous daemon countdown timer."""
        timer_id = f"timer_{self._timer_counter}"
        self._timer_counter += 1

        end_time = time.time() + duration_seconds
        timer_entry = {
            "id": timer_id,
            "label": label,
            "duration": duration_seconds,
            "end_time": end_time,
            "active": True
        }

        def _worker():
            time.sleep(duration_seconds)
            with self._lock:
                if timer_id in self._timers and self._timers[timer_id]["active"]:
                    self._timers[timer_id]["active"] = False

            # Audio notification chime
            try:
                if winsound:
                    for _ in range(3):
                        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                        time.sleep(0.3)
            except Exception:
                pass

            # Voice announcement
            try:
                from runtime.stt_engine import EdithTTSEngine
                tts = EdithTTSEngine()
                tts.speak(f"Alert: Your {label} of {int(duration_seconds)} seconds has finished.", async_mode=True)
            except Exception:
                pass

            if on_complete_callback:
                try:
                    on_complete_callback(timer_entry)
                except Exception:
                    pass

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        with self._lock:
            self._timers[timer_id] = timer_entry

        # Format readable string
        m = int(duration_seconds // 60)
        s = int(duration_seconds % 60)
        time_str = f"{m}m {s}s" if m > 0 else f"{s} seconds"

        return {
            "status": "success",
            "action": "system_timer",
            "timer_id": timer_id,
            "duration": duration_seconds,
            "label": label,
            "message": f"Timer set for {time_str} ('{label}'). I'll alert you when it's done."
        }

    def list_timers(self) -> List[Dict[str, Any]]:
        """List all currently active countdown timers."""
        now = time.time()
        active = []
        with self._lock:
            for tid, tinfo in self._timers.items():
                if tinfo["active"] and tinfo["end_time"] > now:
                    remaining = round(tinfo["end_time"] - now, 1)
                    active.append({
                        "id": tid,
                        "label": tinfo["label"],
                        "remaining_seconds": remaining
                    })
        return active

    def cancel_timer(self, timer_id: Optional[str] = None) -> Dict[str, Any]:
        """Cancel active timer by ID or cancel all active timers."""
        with self._lock:
            if timer_id and timer_id in self._timers:
                self._timers[timer_id]["active"] = False
                return {"status": "success", "message": f"Cancelled timer '{timer_id}'."}
            else:
                count = 0
                for tid in self._timers:
                    if self._timers[tid]["active"]:
                        self._timers[tid]["active"] = False
                        count += 1
                return {"status": "success", "message": f"Cancelled {count} active timer(s)."}


class NotesManager:
    """Local Persistent Quick-Notes and Scratchpad Store."""

    NOTES_PATH = os.path.join(os.path.expanduser("~"), ".edith_notes.json")
    NOTES_FILE = NOTES_PATH

    @classmethod
    def _get_path(cls) -> str:
        return getattr(cls, "NOTES_FILE", None) or cls.NOTES_PATH

    @classmethod
    def _load_notes(cls) -> List[Dict[str, Any]]:
        path = cls._get_path()
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    @classmethod
    def _save_notes(cls, notes: List[Dict[str, Any]]) -> bool:
        path = cls._get_path()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(notes, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False

    @classmethod
    def add_note(cls, text: str) -> Dict[str, Any]:
        """Save a new note with timestamp."""
        clean = text.strip()
        if not clean:
            return {"status": "error", "message": "Note content cannot be empty."}

        notes = cls._load_notes()
        new_id = len(notes) + 1
        ts = datetime.datetime.now().strftime("%Y-%m-%d %I:%M %p")

        entry = {
            "id": new_id,
            "text": clean,
            "timestamp": ts
        }
        notes.append(entry)
        cls._save_notes(notes)

        return {
            "status": "success",
            "action": "add_note",
            "note_id": new_id,
            "note": entry,
            "message": f"Saved note #{new_id}: \"{clean}\""
        }

    @classmethod
    def list_notes(cls) -> Dict[str, Any]:
        """Retrieve all saved notes."""
        notes = cls._load_notes()
        if not notes:
            return {
                "status": "success",
                "count": 0,
                "notes": [],
                "message": "You don't have any saved notes right now."
            }

        lines = [f"#{n['id']}. [{n['timestamp']}] {n['text']}" for n in notes]
        return {
            "status": "success",
            "count": len(notes),
            "notes": notes,
            "message": f"You have {len(notes)} saved note(s):\n" + "\n".join(lines)
        }

    @classmethod
    def clear_notes(cls, note_id: Optional[int] = None) -> Dict[str, Any]:
        """Delete specific note or clear all notes."""
        notes = cls._load_notes()
        if not notes:
            return {"status": "success", "cleared_count": 0, "message": "Notes store is already empty."}

        if note_id is not None:
            new_notes = [n for n in notes if n["id"] != note_id]
            if len(new_notes) == len(notes):
                return {"status": "error", "message": f"Note #{note_id} not found."}
            # Reindex
            for idx, n in enumerate(new_notes, 1):
                n["id"] = idx
            cls._save_notes(new_notes)
            return {"status": "success", "remaining_notes": new_notes, "message": f"Deleted note #{note_id}."}
        else:
            cls._save_notes([])
            return {"status": "success", "cleared_count": len(notes), "message": f"Cleared all {len(notes)} notes."}


class WeatherClient:
    """Zero-API-Key Real-time Weather Query Client via wttr.in."""

    @classmethod
    def get_weather(cls, location: str = "") -> Dict[str, Any]:
        """Query current weather and temperature for a city or local area."""
        loc = urllib.parse.quote(location.strip()) if location.strip() else ""
        url = f"https://wttr.in/{loc}?format=j1"

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "curl/7.68.0", "Accept": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            curr = data.get("current_condition", [{}])[0]
            area_info = data.get("nearest_area", [{}])[0]
            city_name = area_info.get("areaName", [{}])[0].get("value", location or "Your Location")
            country = area_info.get("country", [{}])[0].get("value", "")

            temp_c = curr.get("temp_C", "N/A")
            temp_f = curr.get("temp_F", "N/A")
            desc = curr.get("weatherDesc", [{}])[0].get("value", "Clear")
            humidity = curr.get("humidity", "N/A")
            wind_kmph = curr.get("windspeedKmph", "N/A")
            feels_c = curr.get("FeelsLikeC", temp_c)

            msg = (
                f"Weather in {city_name}"
                f"{', ' + country if country else ''}: {temp_c}°C ({temp_f}°F), {desc}. "
                f"Feels like {feels_c}°C | Humidity: {humidity}% | Wind: {wind_kmph} km/h."
            )

            temp_c_num = int(temp_c) if str(temp_c).lstrip("-").isdigit() else temp_c
            temp_f_num = int(temp_f) if str(temp_f).lstrip("-").isdigit() else temp_f

            return {
                "status": "success",
                "city": city_name,
                "temp_c": temp_c_num,
                "temp_f": temp_f_num,
                "temperature_c": temp_c,
                "temperature_f": temp_f,
                "condition": desc,
                "humidity": humidity,
                "wind_kmph": wind_kmph,
                "feels_like_c": feels_c,
                "message": msg
            }

        except Exception as e:
            # Fallback plain text query if JSON parsing fails
            try:
                plain_url = f"https://wttr.in/{loc}?format=%l:+%c+%t,+%w,+Humidity:+%h"
                req2 = urllib.request.Request(plain_url, headers={"User-Agent": "curl/7.68.0"})
                with urllib.request.urlopen(req2, timeout=2.5) as resp2:
                    text_out = resp2.read().decode("utf-8").strip()
                    if text_out and "html" not in text_out.lower():
                        return {
                            "status": "success",
                            "message": f"Weather: {text_out}"
                        }
            except Exception:
                pass

            return {
                "status": "error",
                "message": f"Could not retrieve weather information at this time: {e}"
            }


class UnitConverter:
    """Precision Unit and Currency Conversion Engine."""

    # Fixed base conversion tables to standard SI / metric
    LENGTH_TO_METERS = {
        "m": 1.0, "meter": 1.0, "meters": 1.0,
        "km": 1000.0, "kilometer": 1000.0, "kilometers": 1000.0,
        "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01,
        "mm": 0.001, "millimeter": 0.001, "millimeters": 0.001,
        "mile": 1609.344, "miles": 1609.344, "mi": 1609.344,
        "yard": 0.9144, "yards": 0.9144, "yd": 0.9144,
        "foot": 0.3048, "feet": 0.3048, "ft": 0.3048,
        "inch": 0.0254, "inches": 0.0254, "in": 0.0254,
    }

    MASS_TO_KG = {
        "kg": 1.0, "kilogram": 1.0, "kilograms": 1.0,
        "g": 0.001, "gram": 0.001, "grams": 0.001,
        "mg": 0.000001, "milligram": 0.000001, "milligrams": 0.000001,
        "lb": 0.453592, "pound": 0.453592, "pounds": 0.453592, "lbs": 0.453592,
        "oz": 0.0283495, "ounce": 0.0283495, "ounces": 0.0283495,
        "ton": 1000.0, "tons": 1000.0, "metric ton": 1000.0,
    }

    DATA_TO_BYTES = {
        "b": 1.0, "byte": 1.0, "bytes": 1.0,
        "kb": 1024.0, "kilobyte": 1024.0, "kilobytes": 1024.0,
        "mb": 1024.0 ** 2, "megabyte": 1024.0 ** 2, "megabytes": 1024.0 ** 2,
        "gb": 1024.0 ** 3, "gigabyte": 1024.0 ** 3, "gigabytes": 1024.0 ** 3,
        "tb": 1024.0 ** 4, "terabyte": 1024.0 ** 4, "terabytes": 1024.0 ** 4,
    }

    TIME_TO_SECONDS = {
        "s": 1.0, "second": 1.0, "seconds": 1.0, "sec": 1.0, "secs": 1.0,
        "m": 60.0, "minute": 60.0, "minutes": 60.0, "min": 60.0, "mins": 60.0,
        "h": 3600.0, "hour": 3600.0, "hours": 3600.0, "hr": 3600.0, "hrs": 3600.0,
        "d": 86400.0, "day": 86400.0, "days": 86400.0,
        "week": 604800.0, "weeks": 604800.0,
    }

    # Reference currency exchange rates (against USD base)
    CURRENCY_TO_USD = {
        "usd": 1.0, "dollar": 1.0, "dollars": 1.0,
        "inr": 0.012, "rupee": 0.012, "rupees": 0.012,
        "eur": 1.08, "euro": 1.08, "euros": 1.08,
        "gbp": 1.29, "pound": 1.29, "pounds": 1.29,
        "jpy": 0.0067, "yen": 0.0067,
        "cad": 0.72,
        "aud": 0.65,
    }

    @classmethod
    def convert(cls, amount: float, from_unit: str, to_unit: str) -> Dict[str, Any]:
        """Convert a numerical quantity between unit dimensions."""
        u1 = from_unit.lower().strip()
        u2 = to_unit.lower().strip()

        # 1. Temperature conversions
        if u1 in ["c", "celsius"] and u2 in ["f", "fahrenheit"]:
            res = (amount * 9.0 / 5.0) + 32.0
            return cls._fmt(amount, "°C", round(res, 2), "°F")
        if u1 in ["f", "fahrenheit"] and u2 in ["c", "celsius"]:
            res = (amount - 32.0) * 5.0 / 9.0
            return cls._fmt(amount, "°F", round(res, 2), "°C")
        if u1 in ["c", "celsius"] and u2 in ["k", "kelvin"]:
            res = amount + 273.15
            return cls._fmt(amount, "°C", round(res, 2), "K")

        # 2. Length
        if u1 in cls.LENGTH_TO_METERS and u2 in cls.LENGTH_TO_METERS:
            m = amount * cls.LENGTH_TO_METERS[u1]
            res = m / cls.LENGTH_TO_METERS[u2]
            return cls._fmt(amount, from_unit, round(res, 4), to_unit)

        # 3. Mass
        if u1 in cls.MASS_TO_KG and u2 in cls.MASS_TO_KG:
            kg = amount * cls.MASS_TO_KG[u1]
            res = kg / cls.MASS_TO_KG[u2]
            return cls._fmt(amount, from_unit, round(res, 4), to_unit)

        # 4. Data storage
        if u1 in cls.DATA_TO_BYTES and u2 in cls.DATA_TO_BYTES:
            b = amount * cls.DATA_TO_BYTES[u1]
            res = b / cls.DATA_TO_BYTES[u2]
            return cls._fmt(amount, from_unit, round(res, 4), to_unit)

        # 5. Time
        if u1 in cls.TIME_TO_SECONDS and u2 in cls.TIME_TO_SECONDS:
            sec = amount * cls.TIME_TO_SECONDS[u1]
            res = sec / cls.TIME_TO_SECONDS[u2]
            return cls._fmt(amount, from_unit, round(res, 4), to_unit)

        # 6. Currency
        if u1 in cls.CURRENCY_TO_USD and u2 in cls.CURRENCY_TO_USD:
            usd = amount * cls.CURRENCY_TO_USD[u1]
            res = usd / cls.CURRENCY_TO_USD[u2]
            return cls._fmt(amount, from_unit.upper(), round(res, 2), to_unit.upper())

        return {
            "status": "error",
            "message": f"Cannot convert between incompatible units: '{from_unit}' and '{to_unit}'."
        }

    @staticmethod
    def _fmt(amt: float, u1: str, res: float, u2: str) -> Dict[str, Any]:
        return {
            "status": "success",
            "action": "unit_convert",
            "original_amount": amt,
            "original_unit": u1,
            "converted_amount": res,
            "result": res,
            "converted_unit": u2,
            "message": f"{amt} {u1} = {res} {u2}"
        }


class ProcessInspector:
    """System Application and Network Diagnostics Inspector."""

    @classmethod
    def list_running_apps(cls) -> Dict[str, Any]:
        """List active desktop user applications with window titles."""
        windows = WindowController.list_open_windows()
        if not windows:
            return {"status": "success", "apps": [], "message": "No active user windows found."}

        seen_procs = set()
        apps = []
        for w in windows:
            proc = w["process"].replace(".exe", "") if w["process"] else w["title"]
            if proc and proc.lower() not in seen_procs:
                seen_procs.add(proc.lower())
                apps.append({
                    "process": proc,
                    "title": w["title"],
                    "pid": w["pid"]
                })

        app_names = [a["process"] for a in apps]
        return {
            "status": "success",
            "action": "list_apps",
            "apps": apps,
            "message": f"Active desktop applications ({len(apps)}): {', '.join(app_names)}"
        }

    @classmethod
    def is_app_running(cls, app_name: str) -> Dict[str, Any]:
        """Check if an application is currently running."""
        target = app_name.lower().strip().replace(".exe", "")
        windows = WindowController.list_open_windows()

        matching = [w for w in windows if target in w["process"].lower() or target in w["title"].lower()]
        if matching:
            m = matching[0]
            return {
                "status": "success",
                "running": True,
                "is_running": True,
                "app": app_name,
                "pid": m["pid"],
                "title": m["title"],
                "message": f"Yes, '{app_name}' is currently running (Window: '{m['title']}', PID: {m['pid']})."
            }

        # Check in psutil if installed
        if psutil:
            for p in psutil.process_iter(["pid", "name"]):
                try:
                    if target in p.info["name"].lower():
                        return {
                            "status": "success",
                            "running": True,
                            "is_running": True,
                            "app": app_name,
                            "pid": p.info["pid"],
                            "message": f"Yes, '{app_name}' process is running in the background (PID: {p.info['pid']})."
                        }
                except Exception:
                    pass

        return {
            "status": "success",
            "running": False,
            "is_running": False,
            "app": app_name,
            "message": f"No, '{app_name}' is not currently running."
        }

    @classmethod
    def get_network_info(cls) -> Dict[str, Any]:
        """Query host IP addresses, gateway, and internet connectivity."""
        hostname = socket.gethostname()
        local_ip = "127.0.0.1"
        try:
            local_ip = socket.gethostbyname(hostname)
        except Exception:
            pass

        pub_ip = "Unavailable"
        try:
            req = urllib.request.Request("https://api.ipify.org?format=json", headers={"User-Agent": "curl/7.68.0"})
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                pub_ip = json.loads(resp.read().decode("utf-8")).get("ip", "Unavailable")
        except Exception:
            pass

        return {
            "status": "success",
            "action": "network_info",
            "host": hostname,
            "hostname": hostname,
            "local_ip": local_ip,
            "public_ip": pub_ip,
            "message": f"Host: {hostname} | Local IP: {local_ip} | Public IP: {pub_ip}"
        }
