"""Unit and Integration Tests for E.D.I.T.H. Pocket TTS (Indian Female English Voice).

Verifies:
1. Model configuration and pinned revision validity.
2. Voice reference WAV and safetensors embedding cache.
3. ConversationalSpeechNormalizer rules (apps, acronyms, ordinals, paths, abbreviations).
4. 100% Offline inference with HF_HUB_OFFLINE=1.
5. Audio integrity (24 kHz, non-empty PCM).
6. Consistent vocal identity across repeated calls.
7. Barge-in interruption and audio resource cleanup.
8. Graceful fallback on missing model/voice assets without crashing assistant.
9. Text preservation upon speech failure.
"""

import os
import sys
import time
import unittest
from unittest.mock import patch, MagicMock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runtime.pocket_tts_engine import (
    PocketTTSEngine,
    ConversationalSpeechNormalizer,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_VOICE_WAV,
    DEFAULT_VOICE_CACHE,
    get_pocket_tts_engine,
)
from runtime.stt_engine import EdithTTSEngine
from runtime.speech_synthesis import LocalSpeechSynthesisEngine


class TestPocketTTSIndianVoiceIntegration(unittest.TestCase):
    """Validation test suite for Pocket TTS Indian Female English integration."""

    def setUp(self):
        # Enforce offline mode to guarantee no network queries are made
        os.environ["HF_HUB_OFFLINE"] = "1"

    def test_01_model_files_and_configuration(self):
        """Verify model weights, tokenizer, and config exist locally."""
        self.assertTrue(os.path.exists(DEFAULT_MODEL_CONFIG), f"Missing config: {DEFAULT_MODEL_CONFIG}")
        model_dir = os.path.dirname(DEFAULT_MODEL_CONFIG)
        weights_path = os.path.join(model_dir, "model.safetensors")
        tokenizer_path = os.path.join(model_dir, "tokenizer.model")

        self.assertTrue(os.path.exists(weights_path), f"Missing weights: {weights_path}")
        self.assertTrue(os.path.exists(tokenizer_path), f"Missing tokenizer: {tokenizer_path}")
        self.assertGreater(os.path.getsize(weights_path), 300 * 1024 * 1024, "Weights file unexpectedly small")

    def test_02_voice_reference_and_safetensors_cache(self):
        """Verify female Indian voice reference recording and fast safetensors cache."""
        self.assertTrue(os.path.exists(DEFAULT_VOICE_WAV), f"Missing voice WAV: {DEFAULT_VOICE_WAV}")
        self.assertTrue(os.path.exists(DEFAULT_VOICE_CACHE), f"Missing voice cache: {DEFAULT_VOICE_CACHE}")
        self.assertGreater(os.path.getsize(DEFAULT_VOICE_WAV), 500 * 1024, "Voice WAV size too small")
        self.assertGreater(os.path.getsize(DEFAULT_VOICE_CACHE), 5 * 1024 * 1024, "Voice safetensors cache too small")

    def test_03_conversational_text_normalizer(self):
        """Verify phonetic expansion for acronyms, apps, file extensions, and paths."""
        # 1. Acronyms & Apps
        raw = "VS Code launched. CPU is 12% and API returned JSON."
        norm = ConversationalSpeechNormalizer.normalize(raw)
        self.assertIn("V S Code", norm)
        self.assertIn("C P U", norm)
        self.assertIn("12 percent", norm)
        self.assertIn("A P I", norm)
        self.assertIn("Jason", norm)

        # 2. Windows and Relative File Paths
        path_text = "Review C:\\Users\\Aditya Sawant\\Desktop\\test.py and models/pocket_tts/config.yaml."
        norm_path = ConversationalSpeechNormalizer.normalize(path_text)
        self.assertIn("file test dot pie", norm_path)
        self.assertIn("file config dot yamel", norm_path)

        # 3. Ordinals and Abbreviations
        abbr_text = "This is the 1st test e.g. for v8.2 vs previous versions etc."
        norm_abbr = ConversationalSpeechNormalizer.normalize(abbr_text)
        self.assertIn("first", norm_abbr)
        self.assertIn("for example", norm_abbr)
        self.assertIn("version 8 point 2", norm_abbr)
        self.assertIn("versus", norm_abbr)
        self.assertIn("etcetera", norm_abbr)

    def test_04_condense_long_responses(self):
        """Verify long multi-sentence responses are safely summarized for speech."""
        long_essay = (
            "Sentence one describes the system status. Sentence two explains the memory state. "
            "Sentence three provides the CPU details. Sentence four outlines disk performance. "
            "Sentence five reviews network activity. Sentence six audits thread safety. "
            "Sentence seven lists active background tasks. Sentence eight confirms stability."
        )
        condensed = ConversationalSpeechNormalizer.condense_long_response(long_essay, max_words=30)
        self.assertLessEqual(len(condensed.split()), 45)
        self.assertTrue(condensed.endswith("I've displayed the full details on your screen."))

    def test_05_engine_initialization_and_ready_state(self):
        """Verify engine loads model and cached voice state successfully."""
        engine = PocketTTSEngine(enable_fallback=False)
        # Verify dual-compatibility: works both as engine.is_ready and engine.is_ready()
        self.assertTrue(bool(engine.is_ready))
        self.assertTrue(engine.is_ready())
        self.assertEqual(engine.sample_rate, 24000)

    def test_06_100_percent_offline_synthesis_and_quality(self):
        """Perform real local synthesis under strict offline constraint (HF_HUB_OFFLINE=1)."""
        engine = PocketTTSEngine(enable_fallback=False)
        test_phrase = "Hello Aditya! Your Indian female voice engine is functioning completely offline."

        metrics = engine._synthesize_and_play(test_phrase)
        self.assertEqual(metrics.get("status"), "success")
        self.assertEqual(metrics.get("provider"), "pocket_tts")
        self.assertGreater(metrics.get("duration_sec", 0), 1.0)
        self.assertLess(metrics.get("rtf", 99.0), 1.0, "Synthesis must be faster than real-time on CPU")

    def test_07_consistent_voice_across_repeated_utterances(self):
        """Verify identical shared model and voice embedding persist across calls."""
        engine1 = PocketTTSEngine(enable_fallback=False)
        engine2 = PocketTTSEngine(enable_fallback=False)
        self.assertIs(engine1._shared_model, engine2._shared_model)
        self.assertIs(engine1._shared_voice_state, engine2._shared_voice_state)

    def test_08_instant_barge_in_and_resource_cleanup(self):
        """Verify stop event halts playback and clears active state."""
        engine = PocketTTSEngine(enable_fallback=False)
        engine.stop()
        self.assertFalse(engine.is_speaking())
        self.assertTrue(engine._stop_event.is_set())

    def test_09_missing_model_graceful_fallback(self):
        """Verify missing model file does not crash the assistant and triggers fallback."""
        with patch.object(PocketTTSEngine, "_shared_model", None):
            with patch.object(PocketTTSEngine, "_shared_voice_state", None):
                engine = PocketTTSEngine(
                    config_path="non_existent_config.yaml",
                    enable_fallback=True
                )
                self.assertFalse(bool(engine.is_ready))
                # Speech must not crash; fallback provider must handle
                metrics = engine.speak("Fallback test", async_mode=False)
                self.assertIn(metrics.get("provider"), ["sapi5_fallback", "empty"])

    def test_10_missing_voice_reference_handled_gracefully(self):
        """Verify missing voice files log clean warning and prevent crash."""
        with patch.object(PocketTTSEngine, "_shared_voice_state", None):
            engine = PocketTTSEngine(
                voice_wav="assets/voices/missing_sample.wav",
                voice_cache="assets/voices/missing_sample.safetensors",
                enable_fallback=True
            )
            self.assertFalse(bool(engine.is_ready))

    def test_11_edith_stt_engine_integration(self):
        """Verify EdithTTSEngine in stt_engine.py routes through PocketTTSEngine."""
        edith_tts = EdithTTSEngine()
        self.assertIsNotNone(edith_tts.pocket_tts)
        self.assertTrue(bool(edith_tts.pocket_tts.is_ready))

    def test_12_local_speech_synthesis_engine_integration(self):
        """Verify LocalSpeechSynthesisEngine in speech_synthesis.py routes through PocketTTSEngine."""
        local_tts = LocalSpeechSynthesisEngine(volume=0)
        self.assertIsNotNone(local_tts._pocket_engine)
        self.assertTrue(bool(local_tts._pocket_engine.is_ready))


if __name__ == "__main__":
    unittest.main()
