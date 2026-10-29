"""
=============================================================================
             E.D.I.T.H. V9 COGNITIVE ARCHITECTURE & REPAIR SUITE
       Hinglish NLU, Spelled Recipient Assembler, WhatsApp Search Decoupling,
              Pronoun Blacklist & Semantic File Directives Test Suite
=============================================================================
"""

import unittest
from runtime.stt_engine import SpeechNormalizer
from runtime.contact_resolver import ContactResolver
from engine.understanding import UnderstandingNormalizer
from engine.human_dialogue import EdithDialogueEngine
from engine.edith_brain import EdithBrain


class TestV9NLUHinglishAndWhatsAppRepair(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.normalizer = SpeechNormalizer()
        cls.resolver = ContactResolver()
        cls.dialogue = EdithDialogueEngine()
        cls.brain = EdithBrain()

    def test_01_spelled_out_recipient_assembling(self):
        """Verify hyphenated, spaced, and stuttered spelled words assemble into proper names."""
        test_cases = [
            ("s-a-k-s-h-i", "sakshi"),
            ("s - e - k - s - h - i", "sakshi"),
            ("s-s-a-k-s-h-i", "sakshi"),
            ("a person spelled S-S-A-K-S-H-I", "a person spelled sakshi"),
            ("message, S-E-K-S-H-I on whatsapp", "message, sakshi on whatsapp"),
            ("and s-a-k-s-h-i, whatsapp message, what time is it", "and sakshi, whatsapp message, what time is it")
        ]
        for raw, expected in test_cases:
            with self.subTest(raw=raw):
                res = SpeechNormalizer.clean_spoken_command(raw)
                self.assertIn(expected.lower(), res["cleaned_text"].lower())

    def test_02_phonetic_corrections_indian_names(self):
        """Verify acoustic speech-to-text slips map to canonical contact names."""
        slips = [
            ("The message sucks you that I miss her", "sakshi"),
            ("whatsapp message to SAKSI suction", "sakshi"),
            ("message Suxie on whatsapp", "sakshi"),
            ("message Sekshi on whatsapp", "sakshi")
        ]
        for raw, expected_name in slips:
            with self.subTest(raw=raw):
                res = SpeechNormalizer.clean_spoken_command(raw)
                self.assertIn(expected_name.lower(), res["cleaned_text"].lower())

    def test_03_pronoun_blacklist_prevents_our_as_contact(self):
        """Verify that 'our' is never recognized as a contact name."""
        extracted = self.resolver.extract_intent_and_entities("send our whatsapp message")
        self.assertNotEqual(extracted.get("recipient_raw"), "Our")
        self.assertNotEqual(extracted.get("recipient_raw"), "our")

    def test_04_recipient_correction_directive(self):
        """Verify negation and redirection patterns extract the corrected recipient."""
        q = "No I don't want to message any of our, I want to message a person spelled S-S-A-K-S-H-I"
        cleaned = SpeechNormalizer.clean_spoken_command(q)["cleaned_text"]
        extracted = self.resolver.extract_intent_and_entities(cleaned)
        self.assertEqual(extracted.get("recipient_raw", "").lower(), "sakshi")

    def test_05_hinglish_messaging_patterns(self):
        """Verify Hinglish WhatsApp command extraction."""
        test_cases = [
            ("sakshi ko whatsapp message bhejo ki i miss her", "sakshi", "i miss her"),
            ("sakshi ko message bhejo", "sakshi", None),
            ("whatsapp karo rahul ko", "rahul", None),
            ("sakshi, whatsapp message, what time is it", "sakshi", "what time is it"),
        ]
        for query, exp_recip, exp_text in test_cases:
            with self.subTest(query=query):
                extracted = self.resolver.extract_intent_and_entities(query)
                self.assertEqual(extracted.get("recipient_raw", "").lower(), exp_recip.lower())
                if exp_text:
                    self.assertEqual(extracted.get("message_text"), exp_text)

    def test_06_sakshi_contact_resolution(self):
        """Verify Sakshi resolves from contacts.json without defaulting to hardcoded test phone."""
        res = self.resolver.resolve("sakshi")
        self.assertEqual(res.status, "resolved")
        self.assertEqual(res.resolved_contact.name, "Sakshi")

    def test_07_missing_recipient_asks_clarification(self):
        """Verify send_whatsapp without a recipient asks who to send to instead of silently using test phone."""
        res = self.brain.process("send a whatsapp message", execute=False)
        self.assertEqual(res["action"], "send_whatsapp")
        directive = res.get("directive", {})
        self.assertTrue(directive.get("needs_clarification", False))
        self.assertIn("Who would you like to send this WhatsApp message to?", directive.get("message", ""))

    def test_08_hinglish_app_controls(self):
        """Verify Hinglish app launch and termination commands."""
        # Launch
        n_open = UnderstandingNormalizer.normalize("chrome kholo", {})
        self.assertEqual(n_open.goal, "launch_application")
        self.assertEqual(n_open.intent, "app_launch")
        self.assertEqual(n_open.entities.get("APP"), "chrome")

        # Terminate
        n_close = UnderstandingNormalizer.normalize("spotify band karo", {})
        self.assertEqual(n_close.goal, "close_application")
        self.assertEqual(n_close.intent, "app_close")
        self.assertEqual(n_close.entities.get("APP"), "spotify")

    def test_09_hinglish_music_and_spotify(self):
        """Verify Hinglish music commands and Spotify requests."""
        n_music = UnderstandingNormalizer.normalize("spotify pe gaana bajao", {})
        self.assertEqual(n_music.goal, "play_music")
        self.assertEqual(n_music.intent, "music_playback")

        n_slip = UnderstandingNormalizer.normalize("leave much eye on spotify", {})
        self.assertEqual(n_slip.goal, "play_music")
        self.assertEqual(n_slip.intent, "music_playback")

    def test_10_semantic_file_organizer_directives(self):
        """Verify conversational download file sorting directives map to semantic_file_organize."""
        q1 = "load downloading files into folders that are used pretty used normally"
        n1 = UnderstandingNormalizer.normalize(q1, {})
        self.assertEqual(n1.goal, "semantic_file_organize")
        self.assertEqual(n1.intent, "file_organization")

        q2 = "One folder should be named every day use, one folder should be named hardly used and last folder should be named deleted"
        n2 = UnderstandingNormalizer.normalize(q2, {})
        self.assertEqual(n2.goal, "semantic_file_organize")
        self.assertEqual(n2.intent, "file_organization")

    def test_11_hinglish_system_utilities(self):
        """Verify Hinglish datetime, volume, and screen understanding."""
        n_time = UnderstandingNormalizer.normalize("time kya hua", {})
        self.assertEqual(n_time.goal, "check_datetime")

        n_vol = UnderstandingNormalizer.normalize("volume badhao", {})
        self.assertEqual(n_vol.goal, "control_volume")

        n_screen = UnderstandingNormalizer.normalize("screen pe kya chal raha hai", {})
        self.assertEqual(n_screen.goal, "screen_understanding")

    def test_12_dialogue_operational_guard(self):
        """Verify operational keywords are never hijacked by dialogue continuity."""
        # Set dialogue to emotional topic
        self.dialogue.last_problem_category = "love_support"
        self.dialogue.last_dialogue_topic = "love_reflection"

        # Operational queries must return False and clear dialogue state
        self.assertFalse(self.dialogue.is_conversational_or_problem("message Sakshi on whatsapp saying i miss her"))
        self.assertIsNone(self.dialogue.last_problem_category)

        self.dialogue.last_problem_category = "love_support"
        self.assertFalse(self.dialogue.is_conversational_or_problem("load downloading files into folders"))
        self.assertIsNone(self.dialogue.last_problem_category)


if __name__ == "__main__":
    unittest.main()
