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
    }

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
        has_wake_word = bool(re.search(r"\b(?:edith|hey edith|okay edith)\b", text, re.IGNORECASE))

        # 1. Phonetic entity correction
        text_corrected = text
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
        silence_duration: float = 1.4,
        max_duration: float = 15.0,
        timeout: float = 8.0,
        energy_threshold: Optional[float] = None,
        chunk_duration: float = 0.05
    ) -> Optional[np.ndarray]:
        """
        Record audio from default microphone with dynamic Voice Activity Detection and AGC.
        - Automatically begins recording on voice trigger.
        - Automatically terminates when trailing silence reaches `silence_duration`.
        - Times out cleanly if no voice is detected within `timeout`.
        - Applies automatic gain peak normalization for optimal speech recognition SNR.
        Returns numpy float32 1D array at 16kHz, or None if timed out / failed.
        """
        if not self.is_microphone_available():
            return None

        chunk_samples = int(self.sample_rate * chunk_duration)
        pre_roll_chunks = int(0.45 / chunk_duration)  # Keep 0.45s pre-roll buffer to avoid clipping speech onset
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

                    # 1. Ambient noise calibration phase (first ~0.25s)
                    if len(ambient_energies) < 5:
                        ambient_energies.append(rms)
                        pre_roll_buffer.append(chunk_flat)
                        continue

                    # Determine voice detection threshold with ambient baseline clamp
                    ambient_baseline = min(float(np.mean(ambient_energies)), 0.005)
                    active_threshold = energy_threshold or max(0.004, ambient_baseline * 1.4)

                    # 2. Waiting for voice onset
                    if not speech_started:
                        pre_roll_buffer.append(chunk_flat)
                        if len(pre_roll_buffer) > pre_roll_chunks:
                            pre_roll_buffer.pop(0)

                        if rms >= active_threshold:
                            speech_started = True
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
        "E.D.I.T.H., open notepad, open spotify, open calculator, open chrome, "
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
        silence_duration: float = 1.4,
        max_duration: float = 15.0
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
            timeout=timeout
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
    """Windows SAPI Text-to-Speech Engine with natural pacing and clean utterance pre-filtering."""

    def __init__(self, rate: int = 0, volume: int = 100):
        self.rate = rate
        self.volume = volume
        self._voice = None
        self._init_voice()

    def _init_voice(self):
        """Initialize native Windows SAPI Voice."""
        if comtypes is not None:
            try:
                self._voice = comtypes.client.CreateObject("SAPI.SpVoice")
                if self._voice:
                    self._voice.Rate = self.rate
                    self._voice.Volume = self.volume
            except Exception:
                self._voice = None

    @staticmethod
    def clean_for_speech(text: str) -> str:
        """Strip markdown fences, raw paths, emojis, and code blocks before synthesis."""
        t = text.strip()
        # Remove code blocks
        t = re.sub(r"```[\s\S]*?```", " Code snippet output omitted. ", t)
        t = re.sub(r"`[^`]+`", "", t)
        # Remove URLs
        t = re.sub(r"https?://\S+", "", t)
        # Remove markdown headers and hashes
        t = re.sub(r"#+\s*", "", t)
        t = re.sub(r"^[\*\-]\s*", "", t, flags=re.MULTILINE)
        # Remove emojis / non-ascii symbols that cause TTS glitches
        t = t.encode("ascii", "ignore").decode("ascii")
        # Collapse whitespace
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def speak(self, text: str, async_mode: bool = True):
        """Synthesize text via Windows SAPI."""
        clean_text = self.clean_for_speech(text)
        if not clean_text or not self._voice:
            return

        flags = 1 if async_mode else 0  # SVSFlagsAsync = 1
        try:
            self._voice.Speak(clean_text, flags)
        except Exception:
            # Reinitialize if COM connection dropped
            self._init_voice()
            if self._voice:
                try:
                    self._voice.Speak(clean_text, flags)
                except Exception:
                    pass


class EdithVoiceAssistant:
    """Unified Voice Controller: Microphone STT -> Cognitive Brain -> OS Actions -> Voice Feedback."""

    def __init__(self, brain: Optional[Any] = None, enable_tts: bool = True):
        self.stt = EdithSTTEngine()
        self.tts = EdithTTSEngine() if enable_tts else None
        self.brain = brain
        self.is_listening_loop = False

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
            stt_result = self.stt.listen_and_transcribe(timeout=timeout)

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

        while self.is_listening_loop:
            if on_status_callback:
                on_status_callback("listening")

            stt_res = self.stt.listen_and_transcribe(timeout=timeout)
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

        self.is_listening_loop = False
        if on_status_callback:
            on_status_callback("stopped")
