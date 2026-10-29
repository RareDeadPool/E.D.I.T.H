"""Unit tests for E.D.I.T.H. V7 Natural Voice Output and Prosody Shaping."""

import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runtime.speech_synthesis import (
    SpeechTextCleaner,
    SpeechProsodyFormatter,
    LocalSpeechSynthesisEngine
)


class TestSpeechProsodyAndSynthesis(unittest.TestCase):
    """Test suite for SpeechTextCleaner, SpeechProsodyFormatter, and SAPI5 Engine."""

    def test_text_cleaner_and_contractions(self):
        """Verify markdown stripping, acronym cleaning, and natural contractions."""
        raw = "```python\nprint(1)\n``` **Hello** `var_name` from E.D.I.T.H. I do not think that is working."
        clean_casual = SpeechTextCleaner.clean(raw, apply_contractions=True)
        self.assertNotIn("```", clean_casual)
        self.assertNotIn("**", clean_casual)
        self.assertIn("Edith", clean_casual)
        self.assertIn("don't", clean_casual)
        self.assertIn("that's", clean_casual)

    def test_phrase_splitting(self):
        """Verify phrase splitting for streaming synthesis."""
        multi_sentence = "Here is the first point. What do you think? Finally, everything looks solid."
        phrases = SpeechTextCleaner.split_into_phrases(multi_sentence)
        self.assertEqual(len(phrases), 3)
        self.assertTrue(phrases[0].endswith("."))
        self.assertTrue(phrases[1].endswith("?"))
        self.assertTrue(phrases[2].endswith("."))

    def test_prosody_formatter_styles(self):
        """Verify XML prosody markup across styles."""
        # 1. Technical Composed
        tech_xml, flags = SpeechProsodyFormatter.format_prosody(
            "CPU load is 15%, memory is 42%.",
            style="technical_composed"
        )
        self.assertEqual(flags, SpeechProsodyFormatter.SVSF_IS_XML)
        self.assertIn("<rate absspeed='0'>", tech_xml)
        self.assertIn("<silence msec='140'/>", tech_xml)

        # 2. Relaxed Casual
        casual_xml, _ = SpeechProsodyFormatter.format_prosody(
            "Sure thing, I'll take care of that right now.",
            style="relaxed_casual"
        )
        self.assertIn("<rate absspeed='1'>", casual_xml)
        self.assertIn("<silence msec='160'/>", casual_xml)

        # 3. Gentle Personal (empathetic / reflective)
        gentle_xml, _ = SpeechProsodyFormatter.format_prosody(
            "I hear you, it makes complete sense that you feel this way.",
            style="gentle_personal"
        )
        self.assertIn("<rate absspeed='-1'>", gentle_xml)
        self.assertIn("<silence msec='220'/>", gentle_xml)

    def test_question_intonation(self):
        """Verify pitch modulation for questions."""
        q_xml, _ = SpeechProsodyFormatter.format_prosody("Should we try that?", style="relaxed_casual")
        self.assertIn("<pitch absmiddle='2'>", q_xml)

    def test_engine_instant_barge_in(self):
        """Verify engine barge-in stop event clearing and state tracking."""
        engine = LocalSpeechSynthesisEngine(volume=0)
        self.assertFalse(engine.is_speaking())
        engine.stop()
        self.assertFalse(engine.is_speaking())


if __name__ == "__main__":
    unittest.main()
