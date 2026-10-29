"""Comprehensive Test Suite for E.D.I.T.H. Speech-to-Text (STT) & Voice Interface Engine (V7).

Validates:
1. SpeechNormalizer phonetic corrections (misheard apps, domains, technical terms).
2. Wake-word isolation and conversational filler stripping ("um", "uh", "can you please").
3. Spoken math symbol conversion ("plus" -> "+", "times" -> "*").
4. AudioRecorder float32-to-WAV 16-bit PCM conversion and header integrity.
5. EdithSTTEngine transcription response schema and engine reporting.
6. Faster-whisper model integration and lazy-loading.
7. EdithTTSEngine speech text sanitizer (stripping code blocks, URLs, markdown).
8. ActionRegistry registration of 'voice_listen' and 'voice_speak'.
9. Closed-loop verification of voice step types.
10. End-to-end Voice Assistant pipeline integration.
"""

import os
import sys
import io
import wave
import unittest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runtime.stt_engine import (
    SpeechNormalizer,
    AudioRecorder,
    EdithSTTEngine,
    EdithTTSEngine,
    EdithVoiceAssistant
)
from runtime.action_registry import ActionRegistry
from runtime.closed_loop import ClosedLoopVerifier


class TestEdithSTTEngine(unittest.TestCase):
    """Test suite for Speech-to-Text (STT) & Voice Interaction in E.D.I.T.H."""

    def test_01_speech_normalizer_phonetic_corrections(self):
        """Test correction of commonly misheard phonetic phrases and app names."""
        cases = [
            ("open spot if i", "open spotify", "spotify"),
            ("bring up note pad", "bring up notepad", "notepad"),
            ("launch vs code", "launch vscode", "vscode"),
            ("send text on whats app", "send text on whatsapp", "whatsapp"),
            ("check free space on see drive", "check free space on c drive", None),
            ("check wi fi connection", "check wifi connection", None),
            ("pign 8.8.8.8", "ping 8.8.8.8", None),
        ]
        for raw, expected_substr, expected_app in cases:
            with self.subTest(raw=raw):
                res = SpeechNormalizer.clean_spoken_command(raw)
                self.assertIn(expected_substr, res["cleaned_text"].lower())
                if expected_app:
                    self.assertEqual(res["detected_app"], expected_app)

    def test_02_speech_normalizer_filler_and_wake_words(self):
        """Test wake-word isolation and stripping of filler conversational phrases."""
        # Wake word detection and stripping
        res1 = SpeechNormalizer.clean_spoken_command("edith open chrome")
        self.assertTrue(res1["has_wake_word"])
        self.assertEqual(res1["cleaned_text"].lower(), "open chrome")

        res2 = SpeechNormalizer.clean_spoken_command("hey edith what is the current time")
        self.assertTrue(res2["has_wake_word"])
        self.assertEqual(res2["cleaned_text"].lower(), "what is the current time")

        # Conversational fillers
        res3 = SpeechNormalizer.clean_spoken_command("um can you please launch notepad")
        self.assertIn("notepad", res3["cleaned_text"].lower())
        self.assertFalse(res3["cleaned_text"].lower().startswith("um"))

        res4 = SpeechNormalizer.clean_spoken_command("uh check cpu usage")
        self.assertEqual(res4["cleaned_text"].lower(), "check cpu usage")

    def test_03_speech_normalizer_math_speech(self):
        """Test spoken math terms converted to standard mathematical operators."""
        res1 = SpeechNormalizer.clean_spoken_command("calculate 25 times 40")
        self.assertIn("*", res1["cleaned_text"])

        res2 = SpeechNormalizer.clean_spoken_command("what is 50 plus 30")
        self.assertIn("+", res2["cleaned_text"])

        res3 = SpeechNormalizer.clean_spoken_command("calculate 100 divided by 5")
        self.assertIn("/", res3["cleaned_text"])

    def test_04_audio_recorder_wav_conversion(self):
        """Test converting numpy float32 audio arrays to valid 16kHz 16-bit PCM WAV bytes."""
        sample_rate = 16000
        # Generate 1.0 second 440Hz sine wave
        t = np.linspace(0, 1.0, sample_rate, dtype=np.float32)
        sine_wave = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

        wav_bytes = AudioRecorder.audio_to_wav_bytes(sine_wave, sample_rate=sample_rate)
        self.assertTrue(len(wav_bytes) > 0)

        # Validate WAV header properties using wave module
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getsampwidth(), 2)  # 16-bit = 2 bytes
            self.assertEqual(wf.getframerate(), 16000)
            self.assertEqual(wf.getnframes(), sample_rate)

    def test_05_stt_transcribe_schema_and_array(self):
        """Test transcribing numpy audio array returns expected schema."""
        stt = EdithSTTEngine()
        # 0.5s silence
        silence = np.zeros(8000, dtype=np.float32)
        res = stt.transcribe(silence)

        self.assertIn("success", res)
        self.assertIn("text", res)
        self.assertIn("cleaned_text", res)
        self.assertIn("engine", res)
        self.assertIn("latency_ms", res)
        self.assertTrue(res["latency_ms"] >= 0)

    def test_06_stt_whisper_model_integration(self):
        """Test faster-whisper local neural model loader."""
        stt = EdithSTTEngine(model_size="tiny.en")
        model = stt._get_whisper_model()
        self.assertIsNotNone(model)

    def test_07_tts_clean_for_speech(self):
        """Test speech text sanitizer stripping code blocks, URLs, and markdown."""
        raw_msg = (
            "Here is the code you requested:\n"
            "```python\n"
            "def calculate(a, b):\n"
            "    return a * b\n"
            "```\n"
            "Visit https://python.org for more details. # Heading\n"
            "- Bullet point item"
        )
        cleaned = EdithTTSEngine.clean_for_speech(raw_msg)
        self.assertNotIn("```", cleaned)
        self.assertNotIn("https://", cleaned)
        self.assertNotIn("# Heading", cleaned)
        self.assertIn("Code snippet output omitted", cleaned)
        self.assertIn("Bullet point item", cleaned)

    def test_08_action_registry_voice_capabilities(self):
        """Test that 'voice_listen' and 'voice_speak' are registered in ActionRegistry."""
        reg = ActionRegistry()
        cap_listen = reg.get_capability("voice_listen")
        self.assertIsNotNone(cap_listen)
        self.assertEqual(cap_listen.category, "voice")

        cap_speak = reg.get_capability("voice_speak")
        self.assertIsNotNone(cap_speak)
        self.assertEqual(cap_speak.category, "voice")

        # Test execute_fn on voice_speak
        res, ver = reg.execute("voice_speak", {"text": "Test speech output"})
        self.assertEqual(res["status"], "success")
        self.assertTrue(ver.verified)

    def test_09_closed_loop_verification_voice(self):
        """Test closed-loop verification handles voice actions."""
        verifier = ClosedLoopVerifier()
        v_res1 = verifier.verify(
            step_type="voice_speak",
            target="speech",
            expected_state="success",
            actual_result={"status": "success", "message": "Spoke text."}
        )
        self.assertTrue(v_res1.verified)

        v_res2 = verifier.verify(
            step_type="voice_listen",
            target="transcription",
            expected_state="success",
            actual_result={"status": "success", "cleaned_text": "open notepad"}
        )
        self.assertTrue(v_res2.verified)

    def test_10_voice_assistant_pipeline(self):
        """Test EdithVoiceAssistant initialization and component wiring."""
        va = EdithVoiceAssistant(brain=None, enable_tts=False)
        self.assertIsNotNone(va.stt)
        self.assertIsNone(va.tts)  # TTS disabled for headless test

    def test_11_open_pair_phonetic_normalization(self):
        """Verify that acoustic misrecognitions like 'open pair' are cleanly mapped to 'open notepad'."""
        variants = [
            ("open pair", "open notepad", "notepad"),
            ("launch pair", "launch notepad", "notepad"),
            ("start pair", "start notepad", "notepad"),
            ("open pear", "open notepad", "notepad"),
            ("can you please open pair", "open notepad", "notepad"),
            ("hey edith open pair", "open notepad", "notepad"),
            ("open notepair", "open notepad", "notepad"),
        ]
        for raw, expected_clean, expected_app in variants:
            with self.subTest(raw=raw):
                res = SpeechNormalizer.clean_spoken_command(raw)
                self.assertEqual(res["cleaned_text"].lower(), expected_clean)
                self.assertEqual(res["detected_app"], expected_app)

    def test_12_voice_execution_parity_with_text(self):
        """Verify voice command execution does everything text commands do."""
        from engine.edith_brain import EdithBrain
        brain = EdithBrain(enable_execution=False)
        va = EdithVoiceAssistant(brain=brain, enable_tts=False)

        # 1. Single app command
        res1 = va.listen_and_execute(brain=brain, speak_response=False, cmd_override="open notepad")
        self.assertEqual(res1["status"], "success")
        self.assertEqual(res1["spoken_command"], "open notepad")
        self.assertEqual(res1["brain_result"]["directive"]["handler"], "launch_app")
        self.assertIn("notepad", res1["speech_response"].lower())

        # 2. Multi-step command
        res2 = va.listen_and_execute(brain=brain, speak_response=False, cmd_override="open notepad and type hello")
        self.assertEqual(res2["status"], "success")
        self.assertEqual(res2["brain_result"]["directive"]["handler"], "workflow_script")

    def test_13_voice_empathy_chitchat_parity(self):
        """Verify voice interface handles negative mood and emotional check-in with human empathy."""
        from engine.edith_brain import EdithBrain
        brain = EdithBrain(enable_execution=False)
        va = EdithVoiceAssistant(brain=brain, enable_tts=False)

        res = va.listen_and_execute(brain=brain, speak_response=False, cmd_override="i am in a shitty mood today")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["brain_result"]["intent"], "chitchat")
        self.assertTrue(any(w in res["speech_response"].lower() for w in ["bad days", "tough time", "rough patch", "sorry", "chill"]))

    def test_14_whisper_desktop_prompt_and_model_cascade(self):
        """Verify faster-whisper configuration has domain-biasing prompt and base.en model."""
        stt = EdithSTTEngine()
        self.assertEqual(stt.model_size, "base.en")
        self.assertIn("notepad", stt.DESKTOP_PROMPT)
        self.assertIn("spotify", stt.DESKTOP_PROMPT)
        self.assertIn("calculator", stt.DESKTOP_PROMPT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
