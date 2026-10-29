"""Pocket TTS Local Speech Synthesis Engine for E.D.I.T.H.

Provides:
1. 100% Local, Offline Neural Text-to-Speech via Kyutai Pocket TTS.
2. Multilingual Fine-tune: English with Natural Indian Female Voice.
   - Pinned Model Revision: Thanushttk/pocket-tts-multilingual@80c7f2835138c62750d76a93d4333c5c5bb6a754
   - Reference Recording: assets/voices/edith_indian_english.wav (CSTR VCTK p248, CC BY 4.0)
   - Cached Embedding: assets/voices/edith_indian_english.safetensors (sub-2ms loading)
3. Zero Cloud / Paid API Dependencies: 100% local CPU inference.
4. Natural Conversational Text Normalization & Prosody.
5. Audio Playback with Instant Barge-In Cancellation.
6. Seamless Graceful Fallback to Windows SAPI if assets are missing.
"""

import os
import sys
import time
import threading
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import scipy.io.wavfile
import numpy as np

try:
    import pygame
except ImportError:
    pygame = None

try:
    from pocket_tts import TTSModel
    from pocket_tts.models.model_state import export_model_state
    POCKET_TTS_AVAILABLE = True
except ImportError:
    TTSModel = None
    export_model_state = None
    POCKET_TTS_AVAILABLE = False

from runtime.speech_synthesis import SpeechTextCleaner, SpeechProsodyFormatter

logger = logging.getLogger("EDITH.PocketTTS")

DEFAULT_MODEL_CONFIG = os.path.join(PROJECT_ROOT, "models", "pocket_tts", "config.yaml")
DEFAULT_VOICE_WAV = os.path.join(PROJECT_ROOT, "assets", "voices", "edith_indian_english.wav")
DEFAULT_VOICE_CACHE = os.path.join(PROJECT_ROOT, "assets", "voices", "edith_indian_english.safetensors")
TTS_CACHE_DIR = os.path.join(PROJECT_ROOT, "scratch", "tts_cache")


class ConversationalSpeechNormalizer:
    """Specialized text normalizer for natural, professional Indian-English speech."""

    # Acronyms, application names, Indian names, Hinglish tokens expanded for natural phonetics
    EXPANSIONS = [
        # Assistant & identity
        (r"\bE\.D\.I\.T\.H\.?\b", "Edith"),
        (r"\bEDITH\b", "Edith"),

        # Indian Names & User Identities (Phonetic respelling prevents G2P mumbling/distortion)
        (r"\bSakshi\b", "Saakshi"),
        (r"\bsakshi\b", "Saakshi"),
        (r"\bAditya\b", "Uh-dith-ya"),
        (r"\baditya\b", "Uh-dith-ya"),
        (r"\bSawant\b", "Suh-vaant"),
        (r"\bsawant\b", "Suh-vaant"),
        (r"\bRahul\b", "Rah-hool"),
        (r"\bPooja\b", "Poo-jaa"),
        (r"\bAnanya\b", "Uh-nun-yaa"),
        (r"\bRohan\b", "Ro-hun"),
        (r"\bSneha\b", "Snay-haa"),

        # Conversational Hinglish Spoken Phonetics
        (r"\bbhejo\b", "bhay-jo"),
        (r"\bbhej\b", "bhayj"),
        (r"\bkholo\b", "kho-lo"),
        (r"\bkhol\b", "khol"),
        (r"\bchalao\b", "chuh-lao"),
        (r"\bbajao\b", "buh-jao"),
        (r"\bgaana\b", "gaa-na"),
        (r"\bgana\b", "gaa-na"),
        (r"\bkaro\b", "kuh-ro"),
        (r"\bkar\s+do\b", "kar doh"),
        (r"\btheek\b", "theek"),
        (r"\baccha\b", "uh-chha"),
        (r"\bachha\b", "uh-chha"),
        (r"\bshukriya\b", "shook-ree-ya"),
        (r"\balvida\b", "ul-vi-daa"),
        (r"\bbatao\b", "buh-tao"),
        (r"\bsuno\b", "soo-no"),
        (r"\bbolo\b", "bo-lo"),
        (r"\bsamjhe\b", "sum-jhay"),
        (r"\bsamajh\b", "suh-mujh"),
        (r"\bkaise\b", "kay-say"),
        (r"\bkaisa\b", "kay-saa"),
        (r"\bkaisi\b", "kay-see"),
        (r"\bzindagi\b", "zin-duh-gee"),
        (r"\bwhatsapp\b", "WhatsApp"),

        # Common Applications
        (r"\bvscode\b", "V S Code"),
        (r"\bvs code\b", "V S Code"),
        (r"\bnotepad\+\+\b", "Notepad Plus Plus"),
        (r"\bpowershell\b", "PowerShell"),
        (r"\bgithub\b", "Git Hub"),

        # Technical terms and acronyms
        (r"\bUI\b", "U I"),
        (r"\bUX\b", "U X"),
        (r"\bAPI\b", "A P I"),
        (r"\bAPIs\b", "A P Is"),
        (r"\bCPU\b", "C P U"),
        (r"\bGPU\b", "G P U"),
        (r"\bRAM\b", "Ram"),
        (r"\bVRAM\b", "V Ram"),
        (r"\bOS\b", "O S"),
        (r"\bOCR\b", "O C R"),
        (r"\bTTS\b", "T T S"),
        (r"\bSTT\b", "S T T"),
        (r"\bJSON\b", "Jason"),
        (r"\bHTML\b", "H T M L"),
        (r"\bCSS\b", "C S S"),
        (r"\bURL\b", "U R L"),
        (r"\bURLs\b", "U R Ls"),
        (r"\bCLI\b", "C L I"),
        (r"\bGUI\b", "G U I"),
        (r"\bSDK\b", "S D K"),
        (r"\bIDE\b", "I D E"),
        (r"\bANN\b", "A N N"),
        (r"\bONNX\b", "Onnx"),
        (r"\bHTTP\b", "H T T P"),
        (r"\bHTTPS\b", "H T T P S"),
        (r"\bREST\b", "Rest"),
        (r"\bSQL\b", "Sequel"),
        (r"\bNoSQL\b", "No Sequel"),
        (r"\bLLM\b", "L L M"),
        (r"\bLLMs\b", "L L Ms"),
        (r"\bNLP\b", "N L P"),
        (r"\bAI\b", "A I"),
        (r"\bML\b", "M L"),
        (r"\bRTF\b", "real-time factor"),
        (r"\bPCM\b", "P C M"),
        (r"\bWAV\b", "wave"),

        # Units and measurements
        (r"\bMB\b", "megabytes"),
        (r"\bGB\b", "gigabytes"),
        (r"\bKB\b", "kilobytes"),
        (r"\bms\b", "milliseconds"),
        (r"\bsec\b", "seconds"),
        (r"\bsecs\b", "seconds"),
        (r"\bmin\b", "minutes"),
        (r"\bmins\b", "minutes"),
        (r"\bhr\b", "hours"),
        (r"\bhrs\b", "hours"),
        (r"\bkHz\b", "kilohertz"),

        # Abbreviations
        (r"(?:\b|^)e\.g\.,?\s*", "for example "),
        (r"(?:\b|^)i\.e\.,?\s*", "that is "),
        (r"\betc\.?\b", "etcetera"),
        (r"\bapprox\.?\b", "approximately"),
        (r"\bvs\.?\b", "versus"),

        # Versions and ordinals
        (r"\bv(\d+)\.(\d+)\b", r"version \1 point \2"),
        (r"\bv(\d+)\b", r"version \1"),
        (r"\bV(\d+)\b", r"version \1"),
        (r"\b1st\b", "first"),
        (r"\b2nd\b", "second"),
        (r"\b3rd\b", "third"),
        (r"\b4th\b", "fourth"),
        (r"\b5th\b", "fifth"),
        (r"(\d+)%", r"\1 percent"),
    ]

    # File extensions spoken naturally
    FILE_EXTS = [
        (r"\.py\b", " dot pie"),
        (r"\.wav\b", " dot wave"),
        (r"\.json\b", " dot jason"),
        (r"\.md\b", " dot markdown"),
        (r"\.txt\b", " dot text"),
        (r"\.yaml\b", " dot yamel"),
        (r"\.yml\b", " dot yamel"),
        (r"\.html\b", " dot H T M L"),
        (r"\.css\b", " dot C S S"),
        (r"\.js\b", " dot J S"),
        (r"\.ts\b", " dot T S"),
    ]

    @classmethod
    def condense_long_response(cls, text: str, max_words: int = 65) -> str:
        """Condense long multi-paragraph technical text into a spoken summary clause."""
        words = text.split()
        if len(words) <= max_words:
            return text

        import re
        sentences = re.split(r"(?<=[.!?])\s+", text)
        spoken_sentences = []
        cur_word_count = 0

        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            s_words = len(s_clean.split())
            if cur_word_count + s_words <= max_words or not spoken_sentences:
                spoken_sentences.append(s_clean)
                cur_word_count += s_words
            else:
                break

        summary = " ".join(spoken_sentences).strip()
        if len(sentences) > len(spoken_sentences):
            summary += " I've displayed the full details on your screen."
        return summary

    @classmethod
    def normalize(cls, text: str, max_words: int = 65) -> str:
        """Sanitize text for fluent Indian-English speech delivery."""
        if not text:
            return ""

        import re

        # 1. Simplify Windows paths (including paths with spaces) before backslash stripping
        text = re.sub(
            r"[a-zA-Z]:\\(?:[^\\/:*?\"<>|\r\n]+\\)+([^\\/:*?\"<>|\r\n]+\.\w+)",
            r"file \1",
            text
        )
        # Any remaining Windows paths without extensions
        text = re.sub(
            r"[a-zA-Z]:\\[^ \n\r\t\"'`<>]+",
            lambda m: f"file {os.path.basename(m.group(0))}",
            text
        )

        # 2. Simplify Unix/relative paths (e.g. models/pocket_tts/config.yaml)
        text = re.sub(
            r"(?:[\w\-]+/)+([\w\-]+\.\w+)",
            r"file \1",
            text
        )

        # Strip noisy IDE window parentheticals, e.g. "(E.D.I.T.H - Antigravity IDE - ...)"
        text = re.sub(r"\((?:[^\)]*?(?:IDE|Antigravity|PowerShell|Command Prompt|Visual Studio)[^\)]*?)\)", "", text, flags=re.IGNORECASE)
        # Strip markdown bold/italic/backtick artifacts before cleaner
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"\*([^*]+)\*", r"\1", text)
        text = re.sub(r"`([^`]+)`", r"\1", text)

        # 3. First pass through standard speech text cleaner (markdown/code stripping)
        clean = SpeechTextCleaner.clean(text, apply_contractions=True)

        # 4. Condense lengthy paragraphs to prevent tedious readouts
        clean = cls.condense_long_response(clean, max_words=max_words)

        # 5. Expand acronyms, applications, units, ordinals
        for pat, rep in cls.EXPANSIONS:
            clean = re.sub(pat, rep, clean, flags=re.IGNORECASE)

        # 6. Pronounce file extensions cleanly
        for pat, rep in cls.FILE_EXTS:
            clean = re.sub(pat, rep, clean, flags=re.IGNORECASE)
        # Handle "dot py" created by cleaner
        clean = re.sub(r"\bdot py\b", "dot pie", clean, flags=re.IGNORECASE)

        # Clean multiple punctuation
        clean = re.sub(r"\.{2,}", ".", clean)
        clean = re.sub(r"!+", "!", clean)
        clean = re.sub(r"\?+", "?", clean)

        # Remove stray formatting characters
        clean = re.sub(r"[\{\}\[\]\(\)<>|~_`#*]", " ", clean)
        clean = re.sub(r"\s+([,;.!?])", r"\1", clean)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    @classmethod
    def split_into_spoken_chunks(cls, text: str, max_words: int = 14) -> List[str]:
        """Split text into bite-sized, natural spoken breath groups (<= 14 words each).
        Strictly prevents Pocket TTS internal token overflow (>50 tokens) which causes
        skipped words, clipped phrases, and mumbling/gibberish.
        """
        if not text:
            return []

        import re
        clean = cls.normalize(text)
        if not clean:
            return []

        # Split by sentence terminators
        raw_sentences = re.split(r'(?<=[.!?\n])\s+', clean)
        chunks: List[str] = []

        for sent in raw_sentences:
            sent = sent.strip()
            if not sent:
                continue

            words = sent.split()
            if len(words) <= max_words:
                chunks.append(sent)
                continue

            # Sentence is longer than max_words. Split on clause punctuation (, ; : — -)
            clauses = re.split(r'(?<=[,;:—\-])\s+', sent)
            current_chunk: List[str] = []

            for clause in clauses:
                clause = clause.strip()
                if not clause:
                    continue
                clause_words = clause.split()

                if len(current_chunk) + len(clause_words) <= max_words:
                    current_chunk.extend(clause_words)
                else:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                        current_chunk = []

                    if len(clause_words) <= max_words:
                        current_chunk.extend(clause_words)
                    else:
                        # Clause itself exceeds max_words: split at conjunctions
                        conj_parts = re.split(r'\s+(?=(?:and|but|because|so|while|when|or)\s+)', clause)
                        for part in conj_parts:
                            p_words = part.strip().split()
                            if len(current_chunk) + len(p_words) <= max_words:
                                current_chunk.extend(p_words)
                            else:
                                if current_chunk:
                                    chunks.append(" ".join(current_chunk))
                                    current_chunk = []
                                # Word-level slice fallback
                                for i in range(0, len(p_words), max_words):
                                    slice_w = p_words[i:i + max_words]
                                    if len(slice_w) == max_words:
                                        chunks.append(" ".join(slice_w))
                                    else:
                                        current_chunk.extend(slice_w)

            if current_chunk:
                chunks.append(" ".join(current_chunk))

        # Format chunks with terminal punctuation for optimal prosody
        formatted_chunks = []
        for ch in chunks:
            ch = ch.strip().strip(",;:-— ")
            if not ch:
                continue
            if ch[-1] not in ".!?":
                ch += "."
            formatted_chunks.append(ch)

        return formatted_chunks


class PocketTTSEngine:
    """Production-grade local Pocket TTS Engine with Indian Female voice,
    persistent in-memory model, safetensors voice state caching, and SAPI fallback.
    """

    _shared_model: Optional[Any] = None
    _shared_voice_state: Optional[Dict[str, Any]] = None
    _init_lock = threading.Lock()

    def __init__(
        self,
        config_path: str = DEFAULT_MODEL_CONFIG,
        voice_wav: str = DEFAULT_VOICE_WAV,
        voice_cache: str = DEFAULT_VOICE_CACHE,
        sample_rate: int = 24000,
        volume: float = 1.0,
        enable_fallback: bool = True
    ):
        self.config_path = config_path
        self.voice_wav = voice_wav
        self.voice_cache = voice_cache
        self.sample_rate = sample_rate
        self.volume = max(0.0, min(1.0, volume))
        self.enable_fallback = enable_fallback

        self._stop_event = threading.Event()
        self._is_speaking = False
        self._current_utterance_thread: Optional[threading.Thread] = None
        self._delivered_text_parts: List[str] = []
        self._lock = threading.Lock()

        os.makedirs(TTS_CACHE_DIR, exist_ok=True)
        self._init_audio()
        self._ensure_model_and_voice()

    def _init_audio(self):
        """Initialize pygame mixer for local PCM audio playback."""
        if pygame is not None:
            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.init(frequency=self.sample_rate, size=-16, channels=1)
                pygame.mixer.music.set_volume(self.volume)
            except Exception as e:
                logger.warning(f"Pygame mixer init warning: {e}")

    def _ensure_model_and_voice(self):
        """Load Pocket TTS model and Indian Female voice state into persistent memory."""
        if not POCKET_TTS_AVAILABLE:
            logger.warning("Pocket TTS is not installed in the current environment.")
            return

        with self._init_lock:
            if PocketTTSEngine._shared_model is None:
                if os.path.exists(self.config_path):
                    try:
                        logger.info(f"Loading Pocket TTS model from {self.config_path}...")
                        PocketTTSEngine._shared_model = TTSModel.load_model(config=self.config_path)
                        self.sample_rate = PocketTTSEngine._shared_model.sample_rate
                        logger.info(f"Pocket TTS model loaded successfully (sample_rate: {self.sample_rate} Hz).")
                    except Exception as e:
                        logger.error(f"Failed to load Pocket TTS model: {e}")
                        PocketTTSEngine._shared_model = None

            if PocketTTSEngine._shared_model is not None and PocketTTSEngine._shared_voice_state is None:
                # 1. Prioritize extracting voice state directly from the reference recording WAV
                if os.path.exists(self.voice_wav):
                    try:
                        logger.info(f"Loading voice state directly from reference recording WAV: {self.voice_wav}...")
                        PocketTTSEngine._shared_voice_state = PocketTTSEngine._shared_model.get_state_for_audio_prompt(
                            self.voice_wav
                        )
                        logger.info("Voice state successfully extracted from reference recording WAV.")
                        # Keep safetensors cache synchronized for consistency
                        if export_model_state is not None:
                            try:
                                export_model_state(PocketTTSEngine._shared_voice_state, Path(self.voice_cache))
                            except Exception:
                                pass
                    except Exception as e:
                        logger.warning(f"Could not extract directly from reference WAV {self.voice_wav}: {e}")
                        PocketTTSEngine._shared_voice_state = None

                # 2. Fallback to cached safetensors if WAV is missing
                if PocketTTSEngine._shared_voice_state is None and os.path.exists(self.voice_cache):
                    try:
                        logger.info(f"Loading cached voice state from {self.voice_cache}...")
                        PocketTTSEngine._shared_voice_state = PocketTTSEngine._shared_model.get_state_for_audio_prompt(
                            self.voice_cache
                        )
                    except Exception as e:
                        logger.error(f"Failed loading cached safetensors: {e}")
                        PocketTTSEngine._shared_voice_state = None

    class _ReadyState(int):
        def __call__(self) -> bool:
            return bool(self)

    @property
    def is_ready(self) -> bool:
        """Returns True if local Pocket TTS model and voice state are loaded and ready.
        Supports both property access (engine.is_ready) and function call (engine.is_ready()).
        """
        ready = (
            PocketTTSEngine._shared_model is not None
            and PocketTTSEngine._shared_voice_state is not None
        )
        return self._ReadyState(1 if ready else 0)

    def stop(self):
        """Immediately abort speech synthesis and clear audio playback (barge-in)."""
        self._stop_event.set()
        if pygame is not None and pygame.mixer.get_init():
            try:
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
            except Exception:
                pass
        self._is_speaking = False

    def is_speaking(self) -> bool:
        """Return True if speech synthesis or playback is actively running."""
        if self._stop_event.is_set():
            return False
        if pygame is not None and pygame.mixer.get_init():
            try:
                if pygame.mixer.music.get_busy():
                    return True
            except Exception:
                pass
        return self._is_speaking

    def get_delivered_text(self) -> str:
        """Return the text delivered before interruption."""
        with self._lock:
            return " ".join(self._delivered_text_parts).strip()

    def speak(
        self,
        text: str,
        async_mode: bool = True,
        on_phrase_started: Optional[Callable[[str], None]] = None
    ) -> Dict[str, Any]:
        """Synthesize and speak text using the Indian Female voice.

        Args:
            text: Text to speak.
            async_mode: If True, executes on a background daemon thread.
            on_phrase_started: Optional callback before playback starts.

        Returns:
            Dictionary containing generation status, duration, latency, and provider.
        """
        if not text:
            return {"status": "empty", "text": ""}

        norm_text = ConversationalSpeechNormalizer.normalize(text)
        if not norm_text:
            return {"status": "empty", "text": text}

        self._stop_event.clear()
        self._is_speaking = True
        with self._lock:
            self._delivered_text_parts = []

        if async_mode:
            t = threading.Thread(
                target=self._synthesize_and_play,
                args=(norm_text, on_phrase_started),
                daemon=True
            )
            self._current_utterance_thread = t
            t.start()
            return {"status": "queued_async", "text": norm_text, "provider": "pocket_tts"}
        else:
            return self._synthesize_and_play(norm_text, on_phrase_started)

    def _synthesize_and_play(
        self,
        text: str,
        on_phrase_started: Optional[Callable[[str], None]] = None
    ) -> Dict[str, Any]:
        """Internal execution routine for synthesis and audio playback."""
        self._is_speaking = True
        t_start = time.time()
        metrics: Dict[str, Any] = {
            "text": text,
            "provider": "pocket_tts",
            "status": "success",
            "latency_ms": 0.0,
            "duration_sec": 0.0,
            "rtf": 0.0
        }

        try:
            # 1. Pocket TTS Synthesis (Primary Local Provider)
            if self.is_ready and not self._stop_event.is_set():
                try:
                    model = PocketTTSEngine._shared_model
                    voice = PocketTTSEngine._shared_voice_state

                    # Generate PCM audio tensor chunk-by-chunk (each <= 14 words / ~25 tokens)
                    # This prevents the Pocket TTS 50-token overflow ceiling and eliminates skipped words/gibberish
                    t_gen_start = time.time()
                    chunks = ConversationalSpeechNormalizer.split_into_spoken_chunks(text, max_words=14)
                    if not chunks:
                        chunks = [text]

                    audio_segments = []
                    for ch in chunks:
                        if self._stop_event.is_set():
                            metrics["status"] = "cancelled"
                            return metrics
                        ch_tensor = model.generate_audio(voice, ch)
                        ch_np = ch_tensor.detach().cpu().numpy().squeeze()
                        audio_segments.append(ch_np)
                        # Add a natural micro-pause (60ms) between spoken clauses
                        pause_len = int(self.sample_rate * 0.06)
                        audio_segments.append(np.zeros(pause_len, dtype=ch_np.dtype))

                    if not audio_segments:
                        metrics["status"] = "empty"
                        return metrics

                    audio_np = np.concatenate(audio_segments)
                    t_gen = time.time() - t_gen_start

                    if self._stop_event.is_set():
                        metrics["status"] = "cancelled"
                        return metrics

                    duration = len(audio_np) / self.sample_rate
                    rtf = t_gen / duration if duration > 0 else 1.0

                    metrics["duration_sec"] = round(duration, 2)
                    metrics["latency_ms"] = round(t_gen * 1000.0, 1)
                    metrics["rtf"] = round(rtf, 2)

                    # Save to scratch cache and play
                    out_path = os.path.join(TTS_CACHE_DIR, f"speech_{int(time.time()*1000)}.wav")
                    scipy.io.wavfile.write(out_path, self.sample_rate, (audio_np * 32767).astype(np.int16))

                    if on_phrase_started:
                        try:
                            on_phrase_started(text)
                        except Exception:
                            pass

                    self._play_wav_file(out_path)
                    with self._lock:
                        self._delivered_text_parts.append(text)

                    # Clean up temporary audio file
                    try:
                        if os.path.exists(out_path):
                            os.remove(out_path)
                    except Exception:
                        pass

                    return metrics

                except Exception as p_err:
                    logger.error(f"Pocket TTS generation error: {p_err}. Falling back...")

            # 2. Native Windows SAPI5 Fallback if Pocket TTS is unavailable or failed
            if self.enable_fallback and not self._stop_event.is_set():
                metrics["provider"] = "sapi5_fallback"
                self._speak_sapi_fallback(text)
                with self._lock:
                    self._delivered_text_parts.append(text)
                metrics["latency_ms"] = round((time.time() - t_start) * 1000.0, 1)
                return metrics

        finally:
            self._is_speaking = False

        return metrics

    def _play_wav_file(self, wav_path: str):
        """Stream playback through pygame mixer with responsive barge-in checking."""
        if pygame is None:
            return

        try:
            if not pygame.mixer.get_init():
                self._init_audio()

            pygame.mixer.music.load(wav_path)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                if self._stop_event.is_set():
                    pygame.mixer.music.stop()
                    break
                time.sleep(0.03)

            pygame.mixer.music.unload()
        except Exception as e:
            logger.warning(f"Audio playback error: {e}")

    def _speak_sapi_fallback(self, text: str):
        """Fallback to native Windows SAPI voice."""
        try:
            import comtypes.client
            voice = comtypes.client.CreateObject("SAPI.SpVoice")
            # Select female voice (Zira) if available
            voices = voice.GetVoices()
            for i in range(voices.Count):
                v = voices.Item(i)
                if "zira" in v.GetDescription().lower():
                    voice.Voice = v
                    break
            voice.Speak(text, 0)
        except Exception as e:
            logger.warning(f"SAPI fallback failed: {e}")


_global_pocket_engine: Optional[PocketTTSEngine] = None
_global_pocket_lock = threading.Lock()


def get_pocket_tts_engine(volume: float = 1.0) -> PocketTTSEngine:
    """Return singleton instance of PocketTTSEngine."""
    global _global_pocket_engine
    with _global_pocket_lock:
        if _global_pocket_engine is None:
            _global_pocket_engine = PocketTTSEngine(volume=volume)
        return _global_pocket_engine

