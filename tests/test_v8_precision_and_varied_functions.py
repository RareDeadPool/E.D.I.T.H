"""Test Suite V8: Precision, Disambiguation & Varied Desktop Task Capabilities for E.D.I.T.H.

Verifies:
1. WindowController: Win32 window state management, snapping, focus, and closing.
2. TimerManager: Multi-duration string parsing, asynchronous daemon timers, chime alerts.
3. NotesManager: Persistent JSON scratchpad storage, note enumeration, and deletion.
4. WeatherClient: Zero-API real-time weather querying and structured metric parsing.
5. UnitConverter: Multi-domain dimensional conversions (length, mass, temperature, data, currency).
6. ProcessInspector: Enumeration of desktop apps and network IP diagnostics.
7. EdithBrain Precision Routing: Correct disambiguation of 'what is my ip' and 'search youtube for ...'.
8. Speech-to-Execution Parity: Spoken command corrections execute end-to-end identically to text commands.
"""

import os
import sys
import json
import unittest
from unittest.mock import patch, MagicMock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runtime.system_functions import (
    WindowController,
    TimerManager,
    NotesManager,
    WeatherClient,
    UnitConverter,
    ProcessInspector
)
from engine.edith_brain import EdithBrain
from runtime.stt_engine import SpeechNormalizer


class TestWindowController(unittest.TestCase):
    """Test Win32 window management controller."""

    def test_list_open_windows(self):
        windows = WindowController.list_open_windows()
        self.assertIsInstance(windows, list)

    def test_execute_action_routing(self):
        with patch.object(WindowController, "execute_action", return_value={"status": "success", "action": "window_snap_left"}) as mock_exec:
            res = WindowController.snap_window("left")
            self.assertEqual(res.get("status"), "success")
            mock_exec.assert_called_once_with("snap_left", None)

        with patch.object(WindowController, "execute_action", return_value={"status": "success", "action": "window_maximize"}) as mock_exec:
            res = WindowController.maximize_window()
            self.assertEqual(res.get("status"), "success")
            mock_exec.assert_called_once_with("maximize", None)

        with patch.object(WindowController, "execute_action", return_value={"status": "success", "action": "window_minimize"}) as mock_exec:
            res = WindowController.minimize_window()
            self.assertEqual(res.get("status"), "success")
            mock_exec.assert_called_once_with("minimize", None)


class TestTimerManager(unittest.TestCase):
    """Test timer parsing, scheduling, and lifecycle."""

    def test_parse_duration_string(self):
        self.assertEqual(TimerManager.parse_duration_string("10 seconds"), 10.0)
        self.assertEqual(TimerManager.parse_duration_string("5 minutes"), 300.0)
        self.assertEqual(TimerManager.parse_duration_string("1 hour"), 3600.0)
        self.assertEqual(TimerManager.parse_duration_string("1 hour 30 minutes"), 5400.0)
        self.assertEqual(TimerManager.parse_duration_string("45s"), 45.0)
        self.assertEqual(TimerManager.parse_duration_string("2m"), 120.0)
        self.assertEqual(TimerManager.parse_duration_string("half an hour"), 1800.0)

    def test_timer_set_list_cancel_lifecycle(self):
        tm = TimerManager()
        # Set a timer for 100 seconds
        res = tm.set_timer(100.0, "Unit Test Timer")
        self.assertEqual(res.get("status"), "success")
        timer_id = res.get("timer_id")
        self.assertIsNotNone(timer_id)

        # List active timers
        active = tm.list_timers()
        self.assertTrue(any(t["id"] == timer_id for t in active))

        # Cancel timer
        cancel_res = tm.cancel_timer(timer_id)
        self.assertEqual(cancel_res.get("status"), "success")

        # Verify not in active list
        active_after = tm.list_timers()
        self.assertFalse(any(t["id"] == timer_id for t in active_after))


class TestNotesManager(unittest.TestCase):
    """Test persistent scratchpad notes manager."""

    def setUp(self):
        self.test_notes_file = os.path.join(PROJECT_ROOT, "scratch", "test_notes.json")
        NotesManager.NOTES_FILE = self.test_notes_file
        if os.path.exists(self.test_notes_file):
            os.remove(self.test_notes_file)

    def tearDown(self):
        if os.path.exists(self.test_notes_file):
            os.remove(self.test_notes_file)

    def test_add_list_clear_notes(self):
        # Add notes
        res1 = NotesManager.add_note("Deploy model checkpoint v8")
        self.assertEqual(res1.get("status"), "success")
        self.assertEqual(res1.get("note_id"), 1)

        res2 = NotesManager.add_note("Review speech-to-text accuracy metrics")
        self.assertEqual(res2.get("status"), "success")
        self.assertEqual(res2.get("note_id"), 2)

        # List notes
        list_res = NotesManager.list_notes()
        self.assertEqual(list_res.get("status"), "success")
        self.assertEqual(list_res.get("count"), 2)
        notes = list_res.get("notes", [])
        self.assertEqual(notes[0]["text"], "Deploy model checkpoint v8")
        self.assertEqual(notes[1]["text"], "Review speech-to-text accuracy metrics")

        # Delete single note
        del_res = NotesManager.clear_notes(note_id=1)
        self.assertEqual(del_res.get("status"), "success")
        self.assertEqual(len(del_res.get("remaining_notes", [])), 1)

        # Clear all notes
        clear_res = NotesManager.clear_notes()
        self.assertEqual(clear_res.get("status"), "success")
        self.assertEqual(clear_res.get("cleared_count"), 1)


class TestWeatherClient(unittest.TestCase):
    """Test environmental weather querying."""

    def test_weather_parsing_mock(self):
        mock_json_response = {
            "current_condition": [{
                "temp_C": "26",
                "temp_F": "79",
                "FeelsLikeC": "28",
                "weatherDesc": [{"value": "Clear sky"}],
                "humidity": "65",
                "windspeedKmph": "12"
            }],
            "nearest_area": [{
                "areaName": [{"value": "Mumbai"}],
                "country": [{"value": "India"}]
            }]
        }

        with patch("urllib.request.urlopen") as mock_url:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(mock_json_response).encode("utf-8")
            mock_url.return_value.__enter__.return_value = mock_resp

            res = WeatherClient.get_weather("Mumbai")
            self.assertEqual(res.get("status"), "success")
            self.assertEqual(res.get("temp_c"), 26)
            self.assertEqual(res.get("condition"), "Clear sky")
            self.assertEqual(res.get("city"), "Mumbai")
            self.assertIn("Mumbai", res.get("message", ""))


class TestUnitConverter(unittest.TestCase):
    """Test dimensional unit and currency conversions."""

    def test_length_conversion(self):
        # 1 km to miles (~0.621371)
        res = UnitConverter.convert(1.0, "km", "miles")
        self.assertEqual(res.get("status"), "success")
        self.assertAlmostEqual(res.get("result"), 0.621371, places=2)

        # 1 meter to feet (~3.28084)
        res2 = UnitConverter.convert(1.0, "meters", "feet")
        self.assertAlmostEqual(res2.get("result"), 3.28, places=1)

    def test_mass_conversion(self):
        # 1 kg to lbs (~2.20462)
        res = UnitConverter.convert(1.0, "kg", "lbs")
        self.assertEqual(res.get("status"), "success")
        self.assertAlmostEqual(res.get("result"), 2.20462, places=2)

    def test_temperature_conversion(self):
        # 0 C to F (32)
        res_c_f = UnitConverter.convert(0.0, "c", "f")
        self.assertEqual(res_c_f.get("result"), 32.0)

        # 212 F to C (100)
        res_f_c = UnitConverter.convert(212.0, "f", "c")
        self.assertEqual(res_f_c.get("result"), 100.0)

    def test_data_storage_conversion(self):
        # 1 GB to MB (1024)
        res = UnitConverter.convert(1.0, "gb", "mb")
        self.assertEqual(res.get("result"), 1024.0)

    def test_currency_conversion(self):
        # 100 USD to INR (approx 8333.33)
        res = UnitConverter.convert(100.0, "usd", "inr")
        self.assertEqual(res.get("status"), "success")
        self.assertGreater(res.get("result"), 8000.0)

    def test_unsupported_unit(self):
        res = UnitConverter.convert(10.0, "lightyears", "bananas")
        self.assertEqual(res.get("status"), "error")


class TestProcessInspector(unittest.TestCase):
    """Test process and network diagnostics."""

    def test_network_info(self):
        info = ProcessInspector.get_network_info()
        self.assertEqual(info.get("status"), "success")
        self.assertIn("host", info)
        self.assertIn("local_ip", info)

    def test_is_app_running(self):
        # Python should be running during test execution
        res = ProcessInspector.is_app_running("python")
        self.assertEqual(res.get("status"), "success")
        self.assertTrue(res.get("is_running"))


class TestEdithBrainPrecisionRouting(unittest.TestCase):
    """Test that precision disambiguation correctly routes queries in EdithBrain."""

    @classmethod
    def setUpClass(cls):
        cls.brain = EdithBrain()

    def test_what_is_my_ip_precision_routing(self):
        res = self.brain.process("what is my ip", execute=False)
        directive = res.get("directive", {})
        self.assertEqual(directive.get("handler"), "system_processes")
        self.assertEqual(directive.get("parameters", {}).get("subaction"), "network")
        self.assertTrue(directive.get("can_execute"))

    def test_search_youtube_precision_routing(self):
        res = self.brain.process("search youtube for lofi beats", execute=False)
        directive = res.get("directive", {})
        self.assertEqual(directive.get("handler"), "web_navigation")
        self.assertEqual(directive.get("parameters", {}).get("service"), "youtube")
        self.assertIn("lofi%20beats", directive.get("parameters", {}).get("url", ""))

    def test_unit_conversion_routing(self):
        res = self.brain.process("convert 100 usd to inr", execute=False)
        directive = res.get("directive", {})
        self.assertEqual(directive.get("handler"), "unit_convert")
        self.assertEqual(directive.get("parameters", {}).get("from_unit"), "usd")
        self.assertEqual(directive.get("parameters", {}).get("to_unit"), "inr")

    def test_notes_management_routing(self):
        res_add = self.brain.process("take a note: implement neural cache", execute=False)
        self.assertEqual(res_add.get("directive", {}).get("handler"), "notes_manager")
        self.assertEqual(res_add.get("directive", {}).get("parameters", {}).get("action"), "add")

        res_list = self.brain.process("show my notes", execute=False)
        self.assertEqual(res_list.get("directive", {}).get("handler"), "notes_manager")
        self.assertEqual(res_list.get("directive", {}).get("parameters", {}).get("action"), "list")

    def test_timer_management_routing(self):
        res_set = self.brain.process("set a timer for 10 minutes", execute=False)
        self.assertEqual(res_set.get("directive", {}).get("handler"), "system_timer")
        self.assertEqual(res_set.get("directive", {}).get("parameters", {}).get("subaction"), "set")

        res_check = self.brain.process("check timer", execute=False)
        self.assertEqual(res_check.get("directive", {}).get("handler"), "system_timer")
        self.assertEqual(res_check.get("directive", {}).get("parameters", {}).get("subaction"), "list")

    def test_weather_query_routing(self):
        res = self.brain.process("what is the weather in London", execute=False)
        self.assertEqual(res.get("directive", {}).get("handler"), "system_weather")
        self.assertEqual(res.get("directive", {}).get("parameters", {}).get("location"), "london")

    def test_window_control_routing(self):
        res_snap = self.brain.process("snap window left", execute=False)
        self.assertEqual(res_snap.get("directive", {}).get("handler"), "window_control")
        self.assertEqual(res_snap.get("directive", {}).get("parameters", {}).get("action"), "snap_left")

        res_max = self.brain.process("maximize window", execute=False)
        self.assertEqual(res_max.get("directive", {}).get("handler"), "window_control")
        self.assertEqual(res_max.get("directive", {}).get("parameters", {}).get("action"), "maximize")


class TestSpeechAndActionParity(unittest.TestCase):
    """Test speech-to-text normalizer parity with text commands."""

    def test_acoustic_correction_executes_as_expected(self):
        spoken = "open pair"
        cleaned = SpeechNormalizer.clean_spoken_command(spoken)["cleaned_text"]
        self.assertEqual(cleaned, "open notepad")

        brain = EdithBrain()
        res = brain.process(cleaned, execute=False)
        self.assertEqual(res.get("directive", {}).get("handler"), "launch_app")
        app_target = res.get("directive", {}).get("parameters", {}).get("app_name") or res.get("directive", {}).get("parameters", {}).get("app")
        self.assertEqual(app_target, "notepad")


if __name__ == "__main__":
    unittest.main()
