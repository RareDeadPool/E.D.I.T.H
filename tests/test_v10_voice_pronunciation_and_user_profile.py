"""V10 Comprehensive Validation Test Suite for E.D.I.T.H.
Covers:
1. Neural Voice Engine & Indian/Hinglish Pronunciation Overhaul.
2. Conversational Normalization & Phonetic Respellings (Sakshi, Aditya, Sawant, Hinglish).
3. Persistent User Psychological Profiling (identity, emotions, heartbreak, late-night habits).
4. 'Describe Me' and 'Judge Me' companion cognition.
5. Turn-by-Turn Dynamic Learning and Context Continuity.
"""

import os
import sys
import json
import time
import unittest
from unittest.mock import patch, MagicMock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runtime.pocket_tts_engine import ConversationalSpeechNormalizer
from runtime.edge_tts_engine import NeuralEdgeTTSEngine, get_edge_tts_engine
from runtime.stt_engine import EdithTTSEngine
from engine.user_profile import UserProfileManager, get_user_profile_manager
from engine.human_dialogue import EdithDialogueEngine
from engine.edith_brain import EdithBrain


class TestV10VoiceAndPronunciation(unittest.TestCase):
    """Test Suite verifying voice quality, female expressive voice selection, and pronunciation phonetics."""

    def test_01_conversational_speech_normalizer_phonetics(self):
        """Verify Indian names and Hinglish verbs are cleanly expanded for auditory pronunciation."""
        # 1. Names
        norm_sakshi = ConversationalSpeechNormalizer.normalize("Please send a message to Sakshi right away.")
        self.assertIn("Saakshi", norm_sakshi)

        norm_aditya = ConversationalSpeechNormalizer.normalize("Hello Aditya, how are you?")
        self.assertIn("Uh-dith-ya", norm_aditya)

        norm_sawant = ConversationalSpeechNormalizer.normalize("Aditya Sawant is here.")
        self.assertIn("Suh-vaant", norm_sawant)

        norm_rahul = ConversationalSpeechNormalizer.normalize("Message Rahul saying hello.")
        self.assertIn("Rah-hool", norm_rahul)

        # 2. Hinglish Conversational Verbs
        norm_hinglish = ConversationalSpeechNormalizer.normalize("Spotify pe gaana bajao aur message bhejo, theek hai?")
        self.assertIn("gaa-na", norm_hinglish)
        self.assertIn("buh-jao", norm_hinglish)
        self.assertIn("bhay-jo", norm_hinglish)
        self.assertIn("theek", norm_hinglish)

        # 3. Technical Terms
        norm_tech = ConversationalSpeechNormalizer.normalize("Open vscode and check the API response.")
        self.assertIn("V S Code", norm_tech)
        self.assertIn("A P I", norm_tech)

    def test_02_spoken_chunking_limits(self):
        """Verify speech chunking splits text into spoken breath groups with proper punctuation."""
        long_text = "I have scanned your entire repository, verified every test suite, updated the speech model, and sent a confirmation message to Sakshi on WhatsApp."
        chunks = ConversationalSpeechNormalizer.split_into_spoken_chunks(long_text, max_words=14)
        self.assertTrue(len(chunks) >= 2)
        for ch in chunks:
            self.assertTrue(len(ch.split()) <= 16)
            self.assertTrue(ch[-1] in ".!?")

    def test_03_neural_edge_tts_engine_initialization(self):
        """Verify NeuralEdgeTTSEngine initializes with the default expressive female voice."""
        engine = NeuralEdgeTTSEngine()
        self.assertEqual(engine.DEFAULT_VOICE, "en-IN-NeerjaExpressiveNeural")
        self.assertEqual(engine.voice, "en-IN-NeerjaExpressiveNeural")
        self.assertTrue(engine.is_available)

        # Test switching voice
        engine.set_voice("en-US-AvaMultilingualNeural")
        self.assertEqual(engine.voice, "en-US-AvaMultilingualNeural")

    def test_04_edith_tts_engine_hierarchy(self):
        """Verify EdithTTSEngine prioritizes NeuralEdgeTTSEngine with fallback to PocketTTS and SAPI."""
        tts = EdithTTSEngine()
        self.assertEqual(tts.voice_name, "en-IN-NeerjaExpressiveNeural")
        self.assertIsNotNone(tts.edge_tts)

        # Verify stop() propagates to all sub-engines
        with patch.object(tts.edge_tts, "stop") as mock_edge_stop:
            with patch.object(tts.pocket_tts, "stop") as mock_pocket_stop:
                tts.stop()
                mock_edge_stop.assert_called_once()
                mock_pocket_stop.assert_called_once()


class TestV10UserPsychologicalProfile(unittest.TestCase):
    """Test Suite verifying persistent psychological user profiling and personal memory."""

    def setUp(self):
        self.test_profile_path = os.path.join(PROJECT_ROOT, "scratch", "test_user_profile.json")
        if os.path.exists(self.test_profile_path):
            os.remove(self.test_profile_path)
        self.profile = UserProfileManager(profile_path=self.test_profile_path)

    def tearDown(self):
        if os.path.exists(self.test_profile_path):
            try:
                os.remove(self.test_profile_path)
            except Exception:
                pass

    def test_05_profile_persistence_and_identity(self):
        """Verify identity, archetype, and baseline emotional values persist across loads."""
        self.assertEqual(self.profile.profile_data["identity"]["full_name"], "Aditya Sawant")
        self.assertIn("Architect", self.profile.profile_data["identity"]["archetype"])
        self.assertTrue(os.path.exists(self.test_profile_path))

        # Reload from disk
        reloaded = UserProfileManager(profile_path=self.test_profile_path)
        self.assertEqual(reloaded.profile_data["identity"]["first_name"], "Aditya")
        self.assertEqual(reloaded.profile_data["identity"]["full_name"], "Aditya Sawant")

    def test_06_describe_me_intent_and_generation(self):
        """Verify 'describe me' query detection and deep personalized output."""
        queries = [
            "describe me",
            "who am i",
            "tell me about myself",
            "what kind of person am i",
            "how would you describe me",
            "do you know me"
        ]
        for q in queries:
            self.assertTrue(self.profile.is_describe_query(q), f"Failed to match describe query: {q}")

        desc = self.profile.describe_user()
        self.assertIn("Aditya", desc)
        self.assertIn("architect", desc.lower())
        self.assertTrue("sakshi" in desc.lower() or "creation" in desc.lower())
        self.assertTrue(len(desc) > 200, "Description should be detailed and observant.")

    def test_07_judge_me_intent_and_generation(self):
        """Verify 'judge me' query detection and candid companion judgment."""
        queries = [
            "judge me",
            "what's your judgment of me",
            "judge who i am",
            "be honest with me",
            "what do you really think of me"
        ]
        for q in queries:
            self.assertTrue(self.profile.is_judge_query(q), f"Failed to match judge query: {q}")

        judg = self.profile.judge_user()
        self.assertIn("Aditya", judg)
        self.assertIn("perfectionism", judg.lower())
        self.assertTrue("sleep" in judg.lower() or "overclocked" in judg.lower())
        self.assertTrue("heart" in judg.lower() or "ex" in judg.lower() or "sakshi" in judg.lower())
        self.assertTrue(len(judg) > 200, "Judgment should be insightful and extensive.")

    def test_08_dynamic_learning_from_turn(self):
        """Verify emotional state and personal disclosures update on turns."""
        # Turn 1: Frustration
        self.profile.update_from_turn("Everything is fucked up and it's driving me crazy", "I'm fixing it now.")
        self.assertIn("frustrated", self.profile.profile_data["emotional_state"]["current_mood"])

        # Turn 2: Vulnerability / Heartbreak
        self.profile.update_from_turn("I miss my ex and feeling sad about the past", "I hear you, Aditya.")
        self.assertIn("vulnerable", self.profile.profile_data["emotional_state"]["current_mood"])

        # Turn 3: WhatsApping Sakshi
        self.profile.update_from_turn("Send a message to Sakshi on WhatsApp", "Message sent.")
        sakshi_info = self.profile.profile_data["relationships"]["Sakshi"]
        self.assertIn("last_action", sakshi_info)
        self.assertIn("WhatsApp", sakshi_info["last_action"])


class TestV10DialogueAndBrainIntegration(unittest.TestCase):
    """Test Suite verifying end-to-end dialogue and brain handling of user profiling."""

    def test_09_dialogue_engine_describe_and_judge_responses(self):
        """Verify EdithDialogueEngine responds with psychological profile when asked to describe or judge."""
        dialogue = EdithDialogueEngine()

        # Describe Me
        res_desc = dialogue.respond("describe me")
        msg_desc = res_desc.get("message", "")
        self.assertIn("Aditya", msg_desc)
        self.assertIn("architect", msg_desc.lower())

        # Judge Me
        res_judg = dialogue.respond("judge me")
        msg_judg = res_judg.get("message", "")
        self.assertIn("Aditya", msg_judg)
        self.assertTrue("perfectionism" in msg_judg.lower() or "sleep" in msg_judg.lower())

    def test_10_brain_process_turn_profile_enrichment(self):
        """Verify EdithBrain.process() enriches user profile across turns."""
        brain = EdithBrain(enable_execution=False)
        self.assertIsNotNone(brain.user_profile)

        # Process a personal describe request through the brain
        result = brain.process("who am i to you", execute=False)
        msg = result.get("directive", {}).get("message", "")
        self.assertIn("Aditya", msg)
        self.assertIn(result.get("intent"), ["dialogue", "chitchat"])

        # Verify timeline was recorded in user profile
        timeline = brain.user_profile.profile_data.get("interaction_timeline", [])
        self.assertTrue(len(timeline) > 0)


if __name__ == "__main__":
    unittest.main()
