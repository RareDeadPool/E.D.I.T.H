"""Neural Edge TTS Speech Synthesis Engine for E.D.I.T.H.

Provides:
1. Studio-grade neural speech synthesis using Microsoft Edge Neural TTS.
2. Expressive conversational female voice:
   - Primary: 'en-IN-NeerjaExpressiveNeural' (Natural Indian English female voice with expressive, warm conversational tone).
   - Multilingual Alternative: 'en-US-AvaMultilingualNeural' (ChatGPT Advanced Voice style).
3. Native handling of Indian names (Sakshi, Aditya, Sawant), Hinglish vocabulary, and technical terms.
4. Low latency phrase streaming with Pygame audio playback.
5. Instant barge-in cancellation (stop on user interruption).
6. Transparent fallback signals for local offline engines (Pocket TTS / SAPI5).
"""

import os
import sys
import time
import asyncio
import threading
import logging
from typing import Dict, Any, Optional, List, Callable

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    edge_tts = None
    EDGE_TTS_AVAILABLE = False

try:
    import pygame
except ImportError:
    pygame = None

from runtime.pocket_tts_engine import ConversationalSpeechNormalizer
from runtime.speech_synthesis import SpeechTextCleaner

logger = logging.getLogger("EDITH.EdgeTTS")

TTS_CACHE_DIR = os.path.join(PROJECT_ROOT, "scratch", "tts_cache")


class NeuralEdgeTTSEngine:
    """Production-grade neural TTS engine delivering expressive female speech."""

    DEFAULT_VOICE = "en-IN-NeerjaExpressiveNeural"
    ALT_VOICE = "en-IN-NeerjaNeural"
    AVA_VOICE = "en-US-AvaMultilingualNeural"

    def __init__(
        self,
        voice: Optional[str] = None,
        rate: str = "+0%",
        pitch: str = "+0Hz",
        volume: str = "+0%"
    ):
        self.voice = voice or self.DEFAULT_VOICE
        self.rate = rate
        self.pitch = pitch
        self.volume = volume

        self._stop_event = threading.Event()
        self._is_speaking = False
        self._current_thread: Optional[threading.Thread] = None
        self._delivered_text_parts: List[str] = []
        self._lock = threading.Lock()

        os.makedirs(TTS_CACHE_DIR, exist_ok=True)
        self._init_audio()

    def _init_audio(self):
        """Initialize pygame mixer for playback."""
        if pygame is not None:
            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
            except Exception as e:
                logger.warning(f"Pygame mixer initialization warning: {e}")

    @property
    def is_available(self) -> bool:
        """Return True if edge_tts and pygame are available."""
        return EDGE_TTS_AVAILABLE and pygame is not None

    def stop(self):
        """Immediately abort speech synthesis and cancel audio playback (barge-in)."""
        self._stop_event.set()
        if pygame is not None and pygame.mixer.get_init():
            try:
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
            except Exception:
                pass
        self._is_speaking = False

    def is_speaking(self) -> bool:
        """Check if speech synthesis or playback is actively running."""
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
        """Return the text delivered before an interruption."""
        with self._lock:
            return " ".join(self._delivered_text_parts).strip()

    def set_voice(self, voice_name: str):
        """Switch active voice (e.g. 'en-IN-NeerjaExpressiveNeural', 'en-US-AvaMultilingualNeural')."""
        self.voice = voice_name

    def speak(
        self,
        text: str,
        async_mode: bool = True,
        on_phrase_started: Optional[Callable[[str], None]] = None
    ) -> Dict[str, Any]:
        """Synthesize and speak text.
        
        Args:
            text: Text to synthesize.
            async_mode: If True, executes on a background daemon thread.
            on_phrase_started: Callback triggered when phrase playback begins.
        """
        if not text or not self.is_available:
            return {"status": "unavailable", "provider": "edge_tts"}

        clean_text = ConversationalSpeechNormalizer.normalize(text)
        if not clean_text:
            return {"status": "empty", "provider": "edge_tts"}

        self._stop_event.clear()
        with self._lock:
            self._delivered_text_parts = []

        if async_mode:
            self._is_speaking = True
            self._current_thread = threading.Thread(
                target=self._synthesize_and_play,
                args=(clean_text, on_phrase_started),
                daemon=True
            )
            self._current_thread.start()
            return {"status": "queued_async", "text": clean_text, "provider": "edge_tts"}
        else:
            return self._synthesize_and_play(clean_text, on_phrase_started)

    def _synthesize_and_play(
        self,
        text: str,
        on_phrase_started: Optional[Callable[[str], None]] = None
    ) -> Dict[str, Any]:
        """Internal routine generating neural audio and playing it via pygame."""
        self._is_speaking = True
        t_start = time.time()
        temp_audio_file = os.path.join(TTS_CACHE_DIR, f"edge_{int(time.time()*1000)}.mp3")

        metrics: Dict[str, Any] = {
            "text": text,
            "provider": "edge_tts",
            "voice": self.voice,
            "status": "success",
            "latency_ms": 0.0,
            "duration_sec": 0.0
        }

        try:
            # 1. Async Edge TTS Communication runner
            async def _generate_audio():
                communicate = edge_tts.Communicate(
                    text=text,
                    voice=self.voice,
                    rate=self.rate,
                    pitch=self.pitch,
                    volume=self.volume
                )
                await communicate.save(temp_audio_file)

            if self._stop_event.is_set():
                metrics["status"] = "cancelled"
                return metrics

            # Run in a dedicated asyncio loop
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                loop.run_until_complete(_generate_audio())
            finally:
                loop.close()

            t_gen = time.time() - t_start
            metrics["latency_ms"] = round(t_gen * 1000.0, 1)

            if self._stop_event.is_set():
                metrics["status"] = "cancelled"
                return metrics

            if not os.path.exists(temp_audio_file) or os.path.getsize(temp_audio_file) == 0:
                metrics["status"] = "failed_empty_file"
                return metrics

            # 2. Play audio file via pygame mixer
            if on_phrase_started:
                try:
                    on_phrase_started(text)
                except Exception:
                    pass

            self._play_file(temp_audio_file)
            with self._lock:
                self._delivered_text_parts.append(text)

            return metrics

        except Exception as e:
            logger.warning(f"Edge TTS synthesis error: {e}")
            metrics["status"] = "error"
            metrics["error"] = str(e)
            return metrics

        finally:
            self._is_speaking = False
            # Clean up temporary audio file
            try:
                if os.path.exists(temp_audio_file):
                    os.remove(temp_audio_file)
            except Exception:
                pass

    def _play_file(self, file_path: str):
        """Stream playback through pygame mixer with barge-in support."""
        if pygame is None:
            return

        try:
            if not pygame.mixer.get_init():
                self._init_audio()

            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                if self._stop_event.is_set():
                    pygame.mixer.music.stop()
                    break
                time.sleep(0.04)

            pygame.mixer.music.unload()

        except Exception as e:
            logger.warning(f"Pygame playback error: {e}")


_global_edge_engine: Optional[NeuralEdgeTTSEngine] = None

def get_edge_tts_engine(voice: Optional[str] = None) -> NeuralEdgeTTSEngine:
    """Return singleton instance of NeuralEdgeTTSEngine."""
    global _global_edge_engine
    if _global_edge_engine is None:
        _global_edge_engine = NeuralEdgeTTSEngine(voice=voice)
    elif voice and _global_edge_engine.voice != voice:
        _global_edge_engine.set_voice(voice)
    return _global_edge_engine
