"""Next-Generation Speech-to-Text (STT) & Voice Interface Engine for E.D.I.T.H.

Features:
1. Local Offline Neural STT via faster-whisper (tiny.en/base.en with CTranslate2 INT8 acceleration).
2. Online/Fallback STT via SpeechRecognition (Google Web Speech API).
3. Windows Native SAPI Speech Recognition & SAPI TTS Speech Output (via comtypes).
4. Adaptive Voice Activity Detection (VAD) with dynamic ambient noise floor tracking.
5. Domain-Specific Speech Post-Processing & Phonetic Normalization (correcting app names, filler words, math terms).
6. End-to-end Voice Assistant controller (listen -> transcribe -> understand -> execute -> speak back).
"""

import os
import sys
import re
import io
import wave
import time
import threading
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Optional Imports with Graceful Fallbacks
try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    import comtypes.client
except ImportError:
    comtypes = None

try:
    import edge_tts
except ImportError:
    edge_tts = None

try:
    import pygame
except ImportError:
    pygame = None


class SpeechNormalizer:
    """Domain-specific acoustic and lexical normalization for desktop commands."""

    # Phonetic corrections for terms commonly misheard by microphones
    PHONETIC_CORRECTIONS = {
        r"\b(?:edit|eat it|e\.d\.i\.t\.h\.?|edith)\b": "edith",
        # Notepad and misheard variants (including pair, pear, pad, notepair)
        r"\b(?:note\s*pad|no\s*pad|notepd|notpad|note\s*pair|notepair|no\s*pair|not\s*pad|pair|pear|peyr)\b": "notepad",
        # Spotify and misheard variants
        r"\b(?:spot\s*if\s*i|spot\s*a\s*fly|spotty\s*fly|spotfy|spotfiy|spot\s*high|spot\s*pie|spoty)\b": "spotify",
        # VS Code and misheard variants
        r"\b(?:vs\s*code|v\s*s\s*code|visual\s*studio\s*code|vias\s*code)\b": "vscode",
        # WhatsApp and misheard variants
        r"\b(?:whats\s*app|what\'s\s*app|wat\s*zap|what\s*app|watsapp)\b": "whatsapp",
        r"\b(?:see\s*drive|sea\s*drive|c:?\s*drive)\b": "c drive",
        r"\b(?:wi\s*fi|why\s*fi|y\s*fi|white\s*fi)\b": "wifi",
        r"\b(?:you\s*tube|u\s*tube)\b": "youtube",
        r"\b(?:calc|calulator|calcualte|calci|calcy)\b": "calculator",
        r"\b(?:pign|peeng)\b": "ping",
        r"\b(?:git\s*hub|get\s*hub)\b": "github",
        r"\b(?:task\s*manager|task\s*mgr|task\s*man)\b": "task manager",
        r"\b(?:fire\s*fox|firefx)\b": "firefox",
        r"\b(?:google\s*chrome|chorme|crome)\b": "chrome",
        r"\b(?:power\s*shell|powershl)\b": "powershell",
        # Indian names and misheard phonetic variants
        r"\b(?:suxie|saksi|sucks\s*you|suction|sekshi|shakshi|suxy)\b": "sakshi",
        r"\b(?:adit|adithya|adity)\b": "aditya",
        r"\b(?:rahool|raahul)\b": "rahul",
    }

    @classmethod
    def assemble_spelled_words(cls, text: str) -> str:
        """Reconstruct spelled-out letter sequences (e.g. 's-a-k-s-h-i' -> 'sakshi', 's e k s h i' -> 'sekshi')."""
        if not text:
            return text

        # 1. Hyphenated letters (e.g. s-a-k-s-h-i or s - a - k - s - h - i or S-S-A-K-S-H-I)
        def repl_hyphen(m):
            token = m.group(0)
            letters = re.findall(r"[a-zA-Z]", token)
            word = "".join(letters).lower()
            # Deduplicate consecutive identical leading letters caused by speech stutter (e.g. ssakshi -> sakshi)
            word = re.sub(r"^([a-zA-Z])\1+", r"\1", word)
            return word

        text = re.sub(r"\b[a-zA-Z](?:\s*-\s*[a-zA-Z]){2,}\b", repl_hyphen, text)

        # 2. Spaced single letters after keywords like 'spelled', 'spelt', 'spelling'
        def repl_spelled(m):
            prefix = m.group(1)
            letters = re.findall(r"\b[a-zA-Z]\b", m.group(2))
            word = "".join(letters).lower()
            word = re.sub(r"^([a-zA-Z])\1+", r"\1", word)
            return f"{prefix} {word}"

        text = re.sub(r"\b(spelled|spelt|spelling)\s+((?:[a-zA-Z]\s+){2,}[a-zA-Z])\b", repl_spelled, text, flags=re.IGNORECASE)

        # 3. Trailing or standalone spaced single letters of length >= 3 (e.g. 's a k s h i')
        def repl_spaced_letters(m):
            letters = re.findall(r"\b[a-zA-Z]\b", m.group(0))
            word = "".join(letters).lower()
            word = re.sub(r"^([a-zA-Z])\1+", r"\1", word)
            return word

        text = re.sub(r"(?<![a-zA-Z])([a-zA-Z](?:\s+[a-zA-Z]){2,})(?![a-zA-Z])", repl_spaced_letters, text)
        return text

    # Leading conversational filler phrases
    FILLER_PREFIXES = [
        r"^(?:um|uh|er|ah|like)\s+",
        r"^(?:hey\s+|yo\s+|hello\s+|hi\s+)?edith\s*,?\s*",
        r"^(?:can you\s+|could you\s+|please\s+|would you\s+|will you\s+|kindly\s+)+(?:like\s+)?",
    ]

    # Math spoken words to symbols
    MATH_REPLACEMENTS = [
        (r"\bplus\b", "+"),
        (r"\bminus\b", "-"),
        (r"\btimes\b", "*"),
        (r"\bmultiplied\s+by\b", "*"),
        (r"\bdivided\s+by\b", "/"),
        (r"\bpercent\b", "%"),
    ]

    # Wake word patterns that start the AI
    WAKE_WORD_PATTERNS = [
        r"\b(?:hey\s+|hi\s+|hello\s+|yo\s+|ok\s+|okay\s+)?edith\b",
        r"\bwake\s*up\s*(?:edith)?\b",
        r"\bactivate\s*(?:edith)?\b",
        r"\bstart\s+edith\b",
    ]

    @classmethod
    def extract_wake_word(cls, raw_text: str) -> Tuple[bool, str, str]:
        """Check for wake word and return (has_wake_word, wake_phrase, trailing_command)."""
        if not raw_text:
            return False, "", ""
        t = raw_text.strip()
        for pat in cls.WAKE_WORD_PATTERNS:
            m = re.search(pat, t, re.IGNORECASE)
            if m:
                wake_phrase = m.group(0).strip()
                trailing = t[m.end():].strip(" ,.-!?")
                return True, wake_phrase, trailing
        return False, "", t

    @classmethod
    def clean_spoken_command(cls, raw_text: str) -> Dict[str, Any]:
        """
        Normalize spoken transcription into clean imperative command for EDITH.
        Returns:
            {
                "raw_text": str,
                "cleaned_text": str,
                "has_wake_word": bool,
                "detected_app": Optional[str]
            }
        """
        if not raw_text:
            return {"raw_text": "", "cleaned_text": "", "has_wake_word": False, "detected_app": None}

        text = raw_text.strip()
        has_wake_word, _, _ = cls.extract_wake_word(text)

        # 0. Reconstruct spelled-out words (e.g. 's-a-k-s-h-i' -> 'sakshi', 'spelled s-s-a-k-s-h-i' -> 'spelled sakshi')
        text_assembled = cls.assemble_spelled_words(text)

        # 1. Phonetic entity correction
        text_corrected = text_assembled
        for pat, repl in cls.PHONETIC_CORRECTIONS.items():
            text_corrected = re.sub(pat, repl, text_corrected, flags=re.IGNORECASE)

        # 2. Spoken math normalization (if calculating)
        if any(w in text_corrected.lower() for w in ["calculate", "what is", "how much is", "plus", "times", "minus"]):
            for pat, repl in cls.MATH_REPLACEMENTS:
                text_corrected = re.sub(pat, repl, text_corrected, flags=re.IGNORECASE)

        # 3. Strip conversational filler words and wake words
        cleaned = text_corrected
        for pat in cls.FILLER_PREFIXES:
            cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()

        # Clean trailing punctuation from transcription
        cleaned = re.sub(r"[\.!\?]+$", "", cleaned).strip()

        # 4. Contextual app alias fallback for bare verbs
        # e.g., "open pair", "start pair", "launch pair" -> "open notepad"
        cleaned = re.sub(r"^(open|launch|start|run)\s+(?:a\s+)?(?:pair|pear|peyr|pad)$", r"\1 notepad", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^(close|quit|kill)\s+(?:a\s+)?(?:pair|pear|peyr|pad)$", r"\1 notepad", cleaned, flags=re.IGNORECASE)

        # Extract app name if present
        detected_app = None
        for app in ["notepad", "calculator", "chrome", "spotify", "vscode", "terminal", "whatsapp", "powershell", "firefox"]:
            if app in cleaned.lower():
                detected_app = app
                break

        return {
            "raw_text": raw_text,
            "cleaned_text": cleaned,
            "has_wake_word": has_wake_word,
            "detected_app": detected_app
        }


class AudioRecorder:
    """Microphone audio capture with adaptive energy-based Voice Activity Detection (VAD)."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.is_recording = False

    @staticmethod
    def is_microphone_available() -> bool:
        """Check if at least one audio input device exists."""
        if sd is None:
            return False
        try:
            devices = sd.query_devices()
            for d in devices:
                if d.get("max_input_channels", 0) > 0:
                    return True
            return False
        except Exception:
            return False

    def record_with_vad(
        self,
        silence_duration: float = 0.70,
        max_duration: float = 15.0,
        timeout: float = 8.0,
        energy_threshold: Optional[float] = None,
        chunk_duration: float = 0.03,
        on_voice_start: Optional[Any] = None
    ) -> Optional[np.ndarray]:
        """
        Record audio from default microphone with dynamic Voice Activity Detection and AGC.
        - Automatically begins recording on voice trigger.
        - Triggers `on_voice_start` immediately on speech onset (enabling speech barge-in).
        - Automatically terminates when trailing silence reaches `silence_duration` (default 0.70s).
        - Times out cleanly if no voice is detected within `timeout`.
        - Applies automatic gain peak normalization for optimal speech recognition SNR.
        Returns numpy float32 1D array at 16kHz, or None if timed out / failed.
        """
        if not self.is_microphone_available():
            return None

        chunk_samples = int(self.sample_rate * chunk_duration)
        pre_roll_chunks = int(0.25 / chunk_duration)  # Keep 0.25s pre-roll buffer to avoid clipping speech onset
        pre_roll_buffer: List[np.ndarray] = []

        recorded_chunks: List[np.ndarray] = []
        speech_started = False
        silence_elapsed = 0.0
        start_time = time.time()

        # Calibration for ambient noise floor
        ambient_energies: List[float] = []

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32"
            ) as stream:
                while True:
                    chunk, overflowed = stream.read(chunk_samples)
                    chunk_flat = chunk.flatten()
                    rms = float(np.sqrt(np.mean(chunk_flat ** 2)))

                    now = time.time()
                    elapsed_total = now - start_time

                    # 1. Ambient noise calibration phase (first ~0.15s)
                    if len(ambient_energies) < 5:
                        ambient_energies.append(rms)
                        pre_roll_buffer.append(chunk_flat)
                        continue

                    # Determine voice detection threshold with ambient baseline clamp
                    ambient_baseline = min(float(np.mean(ambient_energies)), 0.005)
                    active_threshold = energy_threshold or max(0.005, ambient_baseline * 1.5)

                    # 2. Waiting for voice onset
                    if not speech_started:
                        pre_roll_buffer.append(chunk_flat)
                        if len(pre_roll_buffer) > pre_roll_chunks:
                            pre_roll_buffer.pop(0)

                        if rms >= active_threshold:
                            speech_started = True
                            if on_voice_start:
                                try:
                                    on_voice_start()
                                except Exception:
                                    pass
                            # Commit pre-roll buffer so initial consonants are preserved
                            recorded_chunks.extend(pre_roll_buffer)
                            silence_elapsed = 0.0
                        elif elapsed_total >= timeout:
                            # Timeout: user did not speak
                            return None

                    # 3. Speech in progress
                    else:
                        recorded_chunks.append(chunk_flat)

                        if rms < active_threshold:
                            silence_elapsed += chunk_duration
                            if silence_elapsed >= silence_duration:
                                # Trailing silence reached: finish recording
                                break
                        else:
                            silence_elapsed = 0.0

                        if elapsed_total >= max_duration:
                            # Max duration limit reached
                            break

            if recorded_chunks:
                full_audio = np.concatenate(recorded_chunks)
                # Automatic Gain Control (AGC) / Peak Normalization to 0.85
                peak = float(np.max(np.abs(full_audio)))
                if 0.001 < peak < 0.90:
                    full_audio = full_audio * (0.85 / peak)
                return full_audio
            return None

        except Exception as e:
            sys.stderr.write(f"[AudioRecorder Error] {e}\n")
            return None

    def monitor_barge_in(
        self,
        is_speaking_fn: Any,
        stop_tts_fn: Any,
        barge_in_threshold: Optional[float] = None,
        poll_interval: float = 0.03
    ) -> bool:
        """Monitor microphone while TTS is speaking. If deliberate user speech is detected,
        immediately stop TTS (barge-in) and return True.
        If speech finishes without interruption, return False.
        """
        if not self.is_microphone_available():
            while is_speaking_fn():
                time.sleep(poll_interval)
            return False

        chunk_samples = int(self.sample_rate * poll_interval)
        consecutive_voice_chunks = 0
        min_consecutive = 2  # Requires ~60ms of continuous voice to avoid false pops/clicks

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32"
            ) as stream:
                # Discard initial stale buffer
                stream.read(chunk_samples)

                while is_speaking_fn():
                    chunk, overflowed = stream.read(chunk_samples)
                    chunk_flat = chunk.flatten()
                    rms = float(np.sqrt(np.mean(chunk_flat ** 2)))

                    # Barge-in threshold: user speaking into mic is typically RMS > 0.025
                    threshold = barge_in_threshold or 0.028
                    if rms >= threshold:
                        consecutive_voice_chunks += 1
                        if consecutive_voice_chunks >= min_consecutive:
                            stop_tts_fn()
                            return True
                    else:
                        consecutive_voice_chunks = 0

            return False
        except Exception:
            while is_speaking_fn():
                time.sleep(poll_interval)
            return False

    @staticmethod
    def audio_to_wav_bytes(audio_array: np.ndarray, sample_rate: int = 16000) -> bytes:
        """Convert float32 numpy array to standard WAV audio bytes (16-bit PCM)."""
        # Clip to prevent distortion
        clipped = np.clip(audio_array, -1.0, 1.0)
        int16_data = (clipped * 32767).astype(np.int16)

        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(int16_data.tobytes())

        return wav_io.getvalue()


class EdithSTTEngine:
    """
    Robust Multi-Engine Speech-to-Text Transcription Engine.
    Cascades: faster-whisper (local offline neural) -> SpeechRecognition (online) -> SAPI / Mock.
    """

    DESKTOP_PROMPT = (
        "E.D.I.T.H., Sakshi, Aditya, Rahul, send whatsapp message to Sakshi, "
        "kholo, bhejo, karo, band karo, chalao, gaana, bajao, screen pe kya hai, "
        "open notepad, open spotify, open calculator, open chrome, "
        "open vscode, open terminal, command prompt, play music, what time is it, "
        "turn volume up, turn volume down, mute, take screenshot, type text, "
        "send whatsapp message, system status, wifi, battery, write notes."
    )

    def __init__(self, model_size: str = "base.en", compute_type: str = "int8"):
        self.model_size = model_size
        self.compute_type = compute_type
        self._whisper_model: Optional[WhisperModel] = None
        self._whisper_lock = threading.Lock()
        self.normalizer = SpeechNormalizer()
        self.recorder = AudioRecorder()

    def _get_whisper_model(self) -> Optional[WhisperModel]:
        """Lazy loader for faster-whisper model with fallback from base.en to tiny.en."""
        if WhisperModel is None:
            return None

        if self._whisper_model is None:
            with self._whisper_lock:
                if self._whisper_model is None:
                    # Attempt primary model (e.g. base.en)
                    try:
                        self._whisper_model = WhisperModel(
                            self.model_size,
                            device="cpu",
                            compute_type=self.compute_type
                        )
                    except Exception as e1:
                        sys.stderr.write(f"[faster-whisper Init Warning: {self.model_size}] {e1}\n")
                        # Fallback to tiny.en if base.en unavailable
                        if self.model_size != "tiny.en":
                            try:
                                self._whisper_model = WhisperModel(
                                    "tiny.en",
                                    device="cpu",
                                    compute_type=self.compute_type
                                )
                                self.model_size = "tiny.en"
                            except Exception as e2:
                                sys.stderr.write(f"[faster-whisper Fallback Init Warning] {e2}\n")
                                self._whisper_model = None
                        else:
                            self._whisper_model = None
        return self._whisper_model

    def transcribe(
        self,
        audio_source: Union[np.ndarray, str, bytes],
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Transcribe audio input from numpy array, .wav file path, or audio bytes.
        Returns:
            {
                "success": bool,
                "text": str,
                "cleaned_text": str,
                "engine": str,
                "confidence": float,
                "latency_ms": float
            }
        """
        t0 = time.time()
        raw_text = ""
        engine_used = "none"
        confidence = 0.0

        # Convert bytes or file path to audio if needed
        audio_data = audio_source
        wav_bytes = None
        if isinstance(audio_source, bytes):
            wav_bytes = audio_source
            try:
                # Read WAV bytes to numpy float32
                with wave.open(io.BytesIO(audio_source), "rb") as wf:
                    frames = wf.readframes(wf.getnframes())
                    int16_arr = np.frombuffer(frames, dtype=np.int16)
                    audio_data = (int16_arr / 32768.0).astype(np.float32)
            except Exception:
                pass
        elif isinstance(audio_source, str) and os.path.exists(audio_source):
            try:
                with wave.open(audio_source, "rb") as wf:
                    frames = wf.readframes(wf.getnframes())
                    int16_arr = np.frombuffer(frames, dtype=np.int16)
                    audio_data = (int16_arr / 32768.0).astype(np.float32)
            except Exception:
                pass

        # 1. Primary Engine: faster-whisper (Local Offline Neural Model)
        model = self._get_whisper_model()
        if model is not None and isinstance(audio_data, np.ndarray) and len(audio_data) > 0:
            try:
                segments, info = model.transcribe(
                    audio_data,
                    language=language,
                    beam_size=5,
                    best_of=5,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=400),
                    initial_prompt=self.DESKTOP_PROMPT,
                    temperature=0.0
                )
                text_parts = [seg.text.strip() for seg in segments if seg.text.strip()]
                raw_text = " ".join(text_parts).strip()
                if raw_text:
                    engine_used = f"faster-whisper-{self.model_size}"
                    confidence = float(info.language_probability) if hasattr(info, "language_probability") else 0.95
            except Exception as e:
                sys.stderr.write(f"[faster-whisper Transcribe Error] {e}\n")

        # 2. Secondary Engine: SpeechRecognition (Google Web Speech API Fallback)
        if not raw_text and sr is not None:
            try:
                if wav_bytes is None and isinstance(audio_data, np.ndarray):
                    wav_bytes = AudioRecorder.audio_to_wav_bytes(audio_data)

                if wav_bytes:
                    recognizer = sr.Recognizer()
                    with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
                        audio = recognizer.record(source)
                        try:
                            raw_text = recognizer.recognize_google(audio, language="en-US")
                        except Exception:
                            raw_text = recognizer.recognize_google(audio, language="en-IN")
                        if raw_text:
                            engine_used = "speech_recognition_google"
                            confidence = 0.90
            except Exception:
                pass

        # 3. Post-Processing & Domain Normalization
        norm_result = self.normalizer.clean_spoken_command(raw_text)
        latency_ms = (time.time() - t0) * 1000.0

        return {
            "success": bool(norm_result["cleaned_text"]),
            "text": raw_text,
            "cleaned_text": norm_result["cleaned_text"],
            "has_wake_word": norm_result["has_wake_word"],
            "detected_app": norm_result["detected_app"],
            "engine": engine_used,
            "confidence": confidence if raw_text else 0.0,
            "latency_ms": round(latency_ms, 2)
        }

    def listen_and_transcribe(
        self,
        timeout: float = 8.0,
        silence_duration: float = 0.70,
        max_duration: float = 15.0,
        on_voice_start: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Record from the active microphone with VAD and transcribe the utterance.
        """
        if not self.recorder.is_microphone_available():
            return {
                "success": False,
                "text": "",
                "cleaned_text": "",
                "error": "No audio input microphone detected.",
                "engine": "none",
                "latency_ms": 0.0
            }

        audio_array = self.recorder.record_with_vad(
            silence_duration=silence_duration,
            max_duration=max_duration,
            timeout=timeout,
            on_voice_start=on_voice_start
        )

        if audio_array is None or len(audio_array) == 0:
            return {
                "success": False,
                "text": "",
                "cleaned_text": "",
                "error": "Silence timeout: no speech detected.",
                "engine": "vad_timeout",
                "latency_ms": 0.0
            }

        return self.transcribe(audio_array)


class EdithTTSEngine:
    """ChatGPT-style Neural Voice Text-to-Speech Engine with natural pacing,
    edge-tts high-fidelity neural voice synthesis, and offline Windows SAPI fallback.
    """

    NEURAL_VOICE = "en-IN-NeerjaExpressiveNeural"  # Expressive, natural Indian English female voice
    FALLBACK_NEURAL_VOICE = "en-US-AvaMultilingualNeural"  # Conversational multilingual female voice

    def __init__(self, rate: int = 0, volume: int = 100, voice: Optional[str] = None):
        self.rate = rate
        self.volume = volume
        self.voice_name = voice or self.NEURAL_VOICE
        self._sapi_voice = None
        self._cache_dir = os.path.join(PROJECT_ROOT, "scratch", "tts_cache")
        os.makedirs(self._cache_dir, exist_ok=True)
        self._pygame_initialized = False
        self._stop_event = threading.Event()
        self._is_speaking = False

        # 1. Primary Neural Voice Engine (Expressive Conversational Female)
        self.edge_tts = None
        try:
            from runtime.edge_tts_engine import NeuralEdgeTTSEngine
            self.edge_tts = NeuralEdgeTTSEngine(voice=self.voice_name)
        except Exception:
            self.edge_tts = None

        # 2. Local Neural Voice Engine (100% Offline Pocket TTS)
        self.pocket_tts = None
        try:
            from runtime.pocket_tts_engine import PocketTTSEngine
            self.pocket_tts = PocketTTSEngine(volume=self.volume / 100.0)
        except Exception:
            self.pocket_tts = None
        self._init_audio()
        self._init_sapi()

    def _init_audio(self):
        """Initialize pygame mixer for ultra-low latency, crisp audio playback."""
        if pygame is not None:
            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                self._pygame_initialized = True
            except Exception:
                self._pygame_initialized = False

    def _init_sapi(self):
        """Initialize native Windows SAPI Voice (preferring female voice Zira)."""
        if comtypes is not None:
            try:
                self._sapi_voice = comtypes.client.CreateObject("SAPI.SpVoice")
                if self._sapi_voice:
                    self._sapi_voice.Rate = self.rate
                    self._sapi_voice.Volume = self.volume
                    # Try to select Zira (female) if available
                    voices = self._sapi_voice.GetVoices()
                    for i in range(voices.Count):
                        v = voices.Item(i)
                        if "zira" in v.GetDescription().lower():
                            self._sapi_voice.Voice = v
                            break
            except Exception:
                self._sapi_voice = None

    def stop(self):
        """Immediately abort speech synthesis and audio playback (barge-in / interrupt)."""
        self._stop_event.set()
        if self.edge_tts:
            try:
                self.edge_tts.stop()
            except Exception:
                pass
        if self.pocket_tts:
            try:
                self.pocket_tts.stop()
            except Exception:
                pass
        if pygame is not None and pygame.mixer.get_init():
            try:
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
            except Exception:
                pass
        if self._sapi_voice:
            try:
                # SVSFPurgeBeforeSpeak = 2 (purges pending speech queue)
                self._sapi_voice.Speak("", 2)
            except Exception:
                pass
        self._is_speaking = False

    def is_speaking(self) -> bool:
        """Check if audio playback or speech is currently active."""
        if self.edge_tts and self.edge_tts.is_speaking():
            return True
        if self.pocket_tts and self.pocket_tts.is_speaking():
            return True
        if pygame is not None and pygame.mixer.get_init():
            try:
                if pygame.mixer.music.get_busy():
                    return True
            except Exception:
                pass
        return self._is_speaking

    @staticmethod
    def play_chime():
        """Play a futuristic 2-tone wake chime when wake word is detected."""
        if pygame is None:
            return
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=2)
            sample_rate = 44100
            t1 = np.linspace(0, 0.07, int(sample_rate * 0.07), False)
            t2 = np.linspace(0, 0.14, int(sample_rate * 0.14), False)
            w1 = 0.22 * np.sin(2 * np.pi * 659.25 * t1)  # E5
            w2 = 0.32 * np.sin(2 * np.pi * 880.00 * t2) * np.linspace(1.0, 0.0, len(t2))  # A5
            audio = np.concatenate([w1, w2])
            audio_int16 = (audio * 32767).astype(np.int16)
            sound = pygame.sndarray.make_sound(np.repeat(audio_int16[:, np.newaxis], 2, axis=1))
            sound.play()
        except Exception:
            pass

    @staticmethod
    def clean_for_speech(text: str) -> str:
        """Strip markdown fences, raw paths, emojis, code blocks, and format for natural speech."""
        t = text.strip()
        # Remove code blocks
        t = re.sub(r"```[\s\S]*?```", " Code snippet output omitted. ", t)
        t = re.sub(r"`[^`]+`", "", t)
        # Remove URLs
        t = re.sub(r"https?://\S+", "", t)
        # Remove markdown headers and hashes
        t = re.sub(r"#+\s*", "", t)
        # Remove markdown bullet points and blockquotes
        t = re.sub(r"^[\*\->]\s*", "", t, flags=re.MULTILINE)
        # Remove markdown bold/italics
        t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
        t = re.sub(r"\*([^*]+)\*", r"\1", t)
        # Format acronym E.D.I.T.H. so it pronounces cleanly as 'Edith'
        t = re.sub(r"\bE\.D\.I\.T\.H\.\b", "Edith", t, flags=re.IGNORECASE)
        # Remove emojis and non-ascii symbols that glitch speech synthesis
        t = re.sub(r"[^\x00-\x7F]+", " ", t)
        # Collapse whitespace
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def speak(self, text: str, async_mode: bool = True):
        """Synthesize and speak text.
        
        Tries high-fidelity Neural TTS (ChatGPT-style voice) first.
        Falls back seamlessly to local Windows SAPI voice if offline.
        """
        clean_text = self.clean_for_speech(text)
        if not clean_text:
            return

        # Reset stop event for new utterance
        self._stop_event.clear()

        if async_mode:
            self._is_speaking = True
            thread = threading.Thread(target=self._synthesize_and_play, args=(clean_text,), daemon=True)
            thread.start()
        else:
            self._synthesize_and_play(clean_text)

    def _synthesize_and_play(self, text: str):
        """Synthesize using expressive female Neural Edge TTS, with fallback to Pocket TTS and SAPI."""
        self._stop_event.clear()
        self._is_speaking = True
        try:
            # 1. Primary: High-fidelity Neural Edge TTS (Natural Indian English female voice)
            if self.edge_tts and self.edge_tts.is_available and not self._stop_event.is_set():
                try:
                    res = self.edge_tts.speak(text, async_mode=False)
                    if res.get("status") == "success":
                        return
                except Exception:
                    pass

            # 2. Local Fallback: 100% Offline Pocket TTS with Indian Female Voice
            if self.pocket_tts and self.pocket_tts.is_ready and not self._stop_event.is_set():
                try:
                    res = self.pocket_tts.speak(text, async_mode=False)
                    if res.get("status") == "success":
                        return
                except Exception:
                    pass

            # 3. Local Fallback via Windows SAPI
            if not self._stop_event.is_set():
                self._speak_sapi(text)
        finally:
            self._is_speaking = False

    def _play_audio_file(self, file_path: str):
        """Play audio file via pygame mixer smoothly with instant barge-in support."""
        try:
            if not self._pygame_initialized:
                self._init_audio()
            if pygame is not None and pygame.mixer.get_init():
                pygame.mixer.music.load(file_path)
                pygame.mixer.music.play()
                self._is_speaking = True
                while pygame.mixer.music.get_busy():
                    if self._stop_event.is_set():
                        pygame.mixer.music.stop()
                        break
                    time.sleep(0.03)
                self._is_speaking = False
                pygame.mixer.music.unload()
        except Exception:
            self._is_speaking = False

    def _speak_sapi(self, text: str):
        """Fallback synthesis using native Windows SAPI."""
        if not self._sapi_voice:
            self._init_sapi()
        if self._sapi_voice and not self._stop_event.is_set():
            try:
                self._is_speaking = True
                self._sapi_voice.Speak(text, 0)
            except Exception:
                pass
            finally:
                self._is_speaking = False


class EdithVoiceAssistant:
    """Unified Voice Controller: Microphone STT -> Cognitive Brain -> OS Actions -> Voice Feedback."""

    def __init__(self, brain: Optional[Any] = None, enable_tts: bool = True):
        self.stt = EdithSTTEngine()
        self.tts = EdithTTSEngine() if enable_tts else None
        self.brain = brain
        self.is_listening_loop = False
        try:
            from runtime.voice_state_machine import VoiceConversationStateMachine
            self.state_machine = VoiceConversationStateMachine(
                tts_engine=self.tts,
                brain=self.brain
            )
        except Exception:
            self.state_machine = None

    def listen_and_execute(
        self,
        brain: Optional[Any] = None,
        speak_response: bool = True,
        timeout: float = 8.0,
        cmd_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Record microphone utterance (or use cmd_override), transcribe, process via EdithBrain,
        execute OS action, and speak the natural response back.
        """
        active_brain = brain or self.brain
        if active_brain is None:
            from engine.edith_brain import EdithBrain
            active_brain = EdithBrain(enable_execution=True)
            self.brain = active_brain

        # 1. Listen & Transcribe (or use cmd_override)
        if cmd_override:
            norm_res = self.stt.normalizer.clean_spoken_command(cmd_override)
            stt_result = {
                "success": True,
                "text": cmd_override,
                "cleaned_text": norm_res["cleaned_text"],
                "has_wake_word": norm_res["has_wake_word"],
                "detected_app": norm_res["detected_app"],
                "engine": "override",
                "confidence": 1.0,
                "latency_ms": 0.0
            }
        else:
            voice_start_cb = self.tts.stop if self.tts else None
            stt_result = self.stt.listen_and_transcribe(timeout=timeout, on_voice_start=voice_start_cb)

        if not stt_result.get("success"):
            return {
                "status": "no_speech",
                "stt": stt_result,
                "brain_result": None,
                "speech_response": None
            }

        cmd = stt_result["cleaned_text"]

        # 2. Process via Cognitive Brain (runs full neural multi-step, actions, dialogue & problem solving)
        brain_res = active_brain.process(cmd, execute=True)

        # 3. Formulate human-friendly spoken response
        ex = brain_res.get("execution")
        directive = brain_res.get("directive", {})
        intent = brain_res.get("intent", "")

        # Prefer execution confirmation message for actions, directive message for dialogues/clarifications
        if ex and isinstance(ex, dict) and ex.get("status") == "success" and ex.get("message") and intent not in ["chitchat", "dialogue"]:
            speech_msg = ex.get("message", "")
        else:
            speech_msg = directive.get("message", "")

        # 4. Voice Output
        if speak_response and self.tts and speech_msg:
            self.tts.speak(speech_msg, async_mode=True)

        return {
            "status": "success",
            "spoken_command": cmd,
            "stt": stt_result,
            "brain_result": brain_res,
            "speech_response": speech_msg
        }

    def continuous_listen_loop(
        self,
        brain: Optional[Any] = None,
        on_command_callback=None,
        on_status_callback=None,
        timeout: float = 8.0
    ):
        """
        Continuous hands-free voice loop.
        Keeps listening, executing, speaking, and continuing until user says 'exit', 'quit', 'stop', or 'text mode'.
        """
        active_brain = brain or self.brain
        if active_brain is None:
            from engine.edith_brain import EdithBrain
            active_brain = EdithBrain(enable_execution=True)
            self.brain = active_brain

        self.is_listening_loop = True
        if on_status_callback:
            on_status_callback("started")

        if self.tts:
            self.tts.speak("E.D.I.T.H. voice mode activated. I'm listening.", async_mode=False)
            time.sleep(0.18)

        while self.is_listening_loop:
            if on_status_callback:
                on_status_callback("listening")

            voice_start_cb = self.tts.stop if self.tts else None
            stt_res = self.stt.listen_and_transcribe(timeout=timeout, on_voice_start=voice_start_cb)
            if not stt_res.get("success"):
                # Timeout or silence, loop back and keep listening
                continue

            cmd = stt_res.get("cleaned_text", "").strip()
            if not cmd:
                continue

            # Check exit phrases
            if cmd.lower() in [
                "exit", "quit", "stop", "exit voice", "stop listening",
                "text mode", "switch to text", "cancel", "standby", "goodbye"
            ]:
                if self.tts:
                    self.tts.speak("Exiting voice mode. Back to text console.", async_mode=False)
                break

            # Execute command
            v_res = self.listen_and_execute(brain=active_brain, speak_response=True, cmd_override=cmd)
            if on_command_callback:
                on_command_callback(cmd, v_res)

            # Turn-Taking & Barge-In Interruption:
            # While E.D.I.T.H. speaks her spoken response, continuously monitor the microphone.
            # If user speaks over her, immediately cut audio playback and listen for the new command.
            if self.tts:
                # Allow a tiny 60ms window for the async playback thread to start
                time.sleep(0.06)
                if self.tts.is_speaking():
                    if on_status_callback:
                        on_status_callback("speaking")
                    interrupted = self.stt.recorder.monitor_barge_in(
                        is_speaking_fn=lambda: self.tts.is_speaking(),
                        stop_tts_fn=self.tts.stop,
                        barge_in_threshold=0.032
                    )
                    if interrupted:
                        if on_status_callback:
                            on_status_callback("interrupted")
                        print("\n[E.D.I.T.H.] (Interrupted)")
                        continue
                # If speech completed naturally without interruption, settle acoustic room reverberation
                time.sleep(0.18)

        self.is_listening_loop = False
        if on_status_callback:
            on_status_callback("stopped")


class EdithWakeWordDetector:
    """Continuous passive wake-word detection engine for E.D.I.T.H.
    
    Listens passively in standby mode for:
    - 'Hey Edith'
    - 'Edith'
    - 'Wake up Edith'
    - 'Yo Edith'
    - 'Hello Edith'

    When detected:
    1. Plays a futuristic wake chime.
    2. Wakes up the AI and speaks a warm acknowledgment.
    3. If a command was attached directly (e.g. 'Hey Edith, open Spotify'), executes immediately.
    4. If just the wake word was spoken, prompts: "I'm here, Aditya! What do you need?",
       listens for your command, executes, and speaks response.
    5. Remains active for continuous conversation, then smoothly returns to standby after silence.
    """

    WAKE_RESPONSES = [
        "I'm here, Aditya! What's on your mind?",
        "Hey Aditya! What do you need?",
        "Online and listening, boss!",
        "Right here, Aditya! How can I help?",
        "Yes, Aditya? Ready when you are!",
    ]

    def __init__(self, brain: Optional[Any] = None, enable_tts: bool = True):
        self.assistant = EdithVoiceAssistant(brain=brain, enable_tts=enable_tts)
        self.brain = self.assistant.brain
        self.stt = self.assistant.stt
        self.tts = self.assistant.tts
        self.is_running = False

    def listen_for_wake_word(
        self,
        on_wake_callback=None,
        on_command_callback=None,
        on_status_callback=None,
        standby_chunk_timeout: float = 3.5,
        session_idle_timeout: float = 12.0
    ):
        """Continuous passive wake-word listener loop.
        
        Runs indefinitely in standby mode until user stops it (Ctrl+C or 'exit' command).
        """
        import random
        self.is_running = True

        if on_status_callback:
            on_status_callback("standby")

        print("\n" + "="*65)
        print("   ⚡ E.D.I.T.H. PASSIVE WAKE-WORD DETECTION ENGINE ONLINE")
        print("   Listening passively in standby for: 'Hey Edith' / 'Edith'")
        print("   Say 'Hey Edith, <command>' or just 'Hey Edith' to wake up!")
        print("   Press Ctrl+C to stop.")
        print("="*65 + "\n")

        # Initial subtle chime
        if self.tts:
            self.tts.play_chime()

        active_session = False
        last_interaction_time = 0.0

        while self.is_running:
            try:
                # -----------------------------------------------------------
                # State 1: Active Session (recently woke up, listening for commands)
                # -----------------------------------------------------------
                if active_session:
                    now = time.time()
                    if now - last_interaction_time > session_idle_timeout:
                        active_session = False
                        print("\n[E.D.I.T.H.] 💤 Returning to standby mode. Say 'Hey Edith' to wake me.")
                        if on_status_callback:
                            on_status_callback("standby")
                        continue

                    if on_status_callback:
                        on_status_callback("listening_active")

                    voice_start_cb = self.tts.stop if self.tts else None
                    stt_res = self.stt.listen_and_transcribe(timeout=6.0, silence_duration=1.2, on_voice_start=voice_start_cb)
                    if not stt_res.get("success"):
                        continue

                    cmd = stt_res.get("cleaned_text", "").strip()
                    if not cmd:
                        continue

                    # Interruption / Hold On Check
                    if any(w in cmd.lower() for w in ["wait", "hold on", "hold up", "pause", "let me speak", "listen to me", "stop talking", "be quiet"]):
                        active_session = True
                        last_interaction_time = time.time()
                        msg = "I'm listening, Aditya! Take your time, I'm all ears. 🎙️"
                        print(f"\n[E.D.I.T.H.] {msg}")
                        if self.tts:
                            self.tts.speak(msg, async_mode=False)
                        continue

                    if cmd.lower() in ["go to sleep", "sleep", "standby", "stop listening", "cancel", "bye", "goodbye"]:
                        active_session = False
                        msg = "Going to standby. Just call my name when you need me! 💤"
                        print(f"\n[E.D.I.T.H.] {msg}")
                        if self.tts:
                            self.tts.speak(msg, async_mode=False)
                        continue

                    if cmd.lower() in ["exit", "quit", "shutdown"]:
                        self.is_running = False
                        msg = "Shutting down voice listener. See you later, Aditya! 💙"
                        print(f"\n[E.D.I.T.H.] {msg}")
                        if self.tts:
                            self.tts.speak(msg, async_mode=False)
                        break

                    # Execute command
                    v_res = self.assistant.listen_and_execute(brain=self.brain, speak_response=True, cmd_override=cmd)
                    last_interaction_time = time.time()
                    if on_command_callback:
                        on_command_callback(cmd, v_res)
                    continue

                # -----------------------------------------------------------
                # State 2: Standby Mode (passively listening for wake word)
                # -----------------------------------------------------------
                stt_res = self.stt.listen_and_transcribe(
                    timeout=standby_chunk_timeout,
                    silence_duration=0.8,
                    max_duration=4.0
                )
                if not stt_res.get("success"):
                    continue

                raw = stt_res.get("raw_text", "")
                has_wake, wake_phrase, trailing_cmd = SpeechNormalizer.extract_wake_word(raw)

                if has_wake:
                    print(f"\n⚡ [WAKE DETECTED] Spoken: \"{raw}\" (Wake Word: \"{wake_phrase}\")")
                    if self.tts:
                        self.tts.play_chime()

                    if on_wake_callback:
                        on_wake_callback(wake_phrase, trailing_cmd)

                    active_session = True
                    last_interaction_time = time.time()

                    # Case A: Wake word had a direct command attached (e.g. 'Hey Edith open spotify')
                    if trailing_cmd.strip():
                        norm = SpeechNormalizer.clean_spoken_command(trailing_cmd)
                        clean_cmd = norm["cleaned_text"]
                        print(f"[Command] > \"{clean_cmd}\"")
                        v_res = self.assistant.listen_and_execute(brain=self.brain, speak_response=True, cmd_override=clean_cmd)
                        if on_command_callback:
                            on_command_callback(clean_cmd, v_res)
                    # Case B: Just the wake word alone (e.g. 'Hey Edith' or 'Wake up')
                    else:
                        greeting = random.choice(self.WAKE_RESPONSES)
                        print(f"[E.D.I.T.H.] {greeting}")
                        if self.tts:
                            self.tts.speak(greeting, async_mode=True)
                        print("🎙️ Listening for your command...")

            except KeyboardInterrupt:
                print("\n[E.D.I.T.H.] Wake-word detection stopped by user.")
                break
            except Exception as e:
                # Keep running resiliently
                time.sleep(0.5)

        self.is_running = False
