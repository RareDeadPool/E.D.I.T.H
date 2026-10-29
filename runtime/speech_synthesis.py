"""Local Speech Synthesis Engine for E.D.I.T.H. V7.

Provides:
1. 100% Offline Windows SAPI5 Speech Output with Voice Selection (Zira, David, Hazel).
2. Advanced Prosody & Clause Boundary Control (XML SSML tags for natural micro-pauses & intonation).
3. Delivery Style Adaptation:
   - 'technical_composed': steady, crisp rate and neutral cadence for metrics/code.
   - 'relaxed_casual': natural rhythm, conversational speed, contractions.
   - 'gentle_personal': warmer, slightly slower cadence and gentler pauses for empathetic turns.
4. Sentence-by-sentence streaming for sub-second first-phrase latency.
5. Instant barge-in cancellation (immediate queue purge on user interruption).
6. Phonetic and lexical text cleaner.
"""

import os
import sys
import re
import time
import threading
from typing import Dict, Any, List, Optional, Tuple, Callable

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import comtypes.client
except ImportError:
    comtypes = None

try:
    import pygame
except ImportError:
    pygame = None


class SpeechTextCleaner:
    """Cleans raw assistant responses for natural auditory pronunciation."""

    # Common contractions for natural casual delivery
    CONTRACTIONS = [
        (r"\bdo not\b", "don't"),
        (r"\bcannot\b", "can't"),
        (r"\bwill not\b", "won't"),
        (r"\bit is\b", "it's"),
        (r"\bthat is\b", "that's"),
        (r"\bthere is\b", "there's"),
        (r"\byou are\b", "you're"),
        (r"\bwe are\b", "we're"),
        (r"\bi have\b", "I've"),
        (r"\bi would\b", "I'd"),
    ]

    @staticmethod
    def clean(text: str, apply_contractions: bool = False) -> str:
        """Strip markdown fences, raw paths, emojis, code blocks, and format acronyms for speech."""
        if not text:
            return ""

        t = text.strip()
        # Remove code blocks
        t = re.sub(r"```[\s\S]*?```", " Code snippet omitted. ", t)
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
        # Pronounce acronym E.D.I.T.H. cleanly as 'Edith'
        t = re.sub(r"\bE\.D\.I\.T\.H\.?(?!\w)", "Edith", t, flags=re.IGNORECASE)
        # Spoken abbreviations
        t = re.sub(r"\bvscode\b", "V S Code", t, flags=re.IGNORECASE)
        t = re.sub(r"\b\.(py|txt|json|md|html|css|js|sh|bat)\b", r" dot \1 ", t, flags=re.IGNORECASE)
        # Natural conversational contractions
        if apply_contractions:
            for pattern, replacement in SpeechTextCleaner.CONTRACTIONS:
                t = re.sub(pattern, replacement, t, flags=re.IGNORECASE)

        # Remove emojis and non-ascii symbols that glitch synthesis
        t = re.sub(r"[^\x00-\x7F]+", " ", t)
        # Clean Windows path backslashes for speech
        t = re.sub(r"\\+", " ", t)
        # Collapse whitespace
        t = re.sub(r"\s+", " ", t).strip()
        return t

    @staticmethod
    def split_into_phrases(text: str) -> List[str]:
        """Split a longer multi-sentence response into discrete phrases for streaming synthesis."""
        clean_text = SpeechTextCleaner.clean(text)
        if not clean_text:
            return []

        # Split on sentence delimiters (., !, ?) while keeping reasonable length
        raw_parts = re.split(r"(?<=[.!?])\s+", clean_text)
        phrases = []
        for p in raw_parts:
            p_strip = p.strip()
            if p_strip:
                phrases.append(p_strip)

        return phrases if phrases else [clean_text]


class SpeechProsodyFormatter:
    """Formats plain speech text with XML prosody tags for natural SAPI5 delivery."""

    # SAPI 5 XML Speak Flags
    SVSF_DEFAULT = 0
    SVSF_IS_XML = 8
    SVSF_PURGE_BEFORE_SPEAK = 2

    @staticmethod
    def format_prosody(phrase: str, style: str = "relaxed_casual") -> Tuple[str, int]:
        """Wrap phrase with natural micro-pauses, pitch modulation, and rate based on delivery style.
        
        Returns: (formatted_xml_string, sapi_speak_flags)
        """
        clean_phrase = phrase.strip()
        if not clean_phrase:
            return "", SpeechProsodyFormatter.SVSF_DEFAULT

        # Clean any raw unescaped XML characters
        escaped = clean_phrase.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        # Style presets: (base_rate, base_pitch, comma_pause_ms, sentence_pause_ms)
        style_map = {
            "technical_composed": (0, 0, 140, 260),
            "relaxed_casual": (1, 0, 160, 300),
            "gentle_personal": (-1, -1, 220, 380),
        }
        rate, pitch, comma_ms, end_ms = style_map.get(style, (1, 0, 160, 300))

        # Insert natural breathing micro-pauses at clause boundaries
        tagged = re.sub(r",\s*", f", <silence msec='{comma_ms}'/> ", escaped)
        tagged = re.sub(r";\s*", f"; <silence msec='{comma_ms + 60}'/> ", tagged)
        tagged = re.sub(r"\s+--?\s+", f" <silence msec='{comma_ms + 40}'/> ", tagged)

        # Question intonation (slight rise in pitch for questions)
        is_question = clean_phrase.endswith("?")
        if is_question:
            q_pitch = pitch + 2
            xml_phrase = f"<rate absspeed='{rate}'><pitch absmiddle='{q_pitch}'>{tagged}</pitch><silence msec='{end_ms}'/></rate>"
        else:
            xml_phrase = f"<rate absspeed='{rate}'><pitch absmiddle='{pitch}'>{tagged}</pitch><silence msec='{end_ms}'/></rate>"

        return xml_phrase, SpeechProsodyFormatter.SVSF_IS_XML


class LocalSpeechSynthesisEngine:
    """100% Offline Speech Synthesizer with Pocket TTS Indian Female Voice and Prosody SAPI Fallback."""

    def __init__(self, voice_preference: str = "zira", rate: int = 1, volume: int = 100, use_pocket_tts: bool = True):
        self.voice_preference = voice_preference.lower()
        self.rate = rate
        self.volume = volume
        self.use_pocket_tts = use_pocket_tts
        self._sapi_voice = None
        self._pocket_engine = None
        self._edge_engine = None
        self._stop_event = threading.Event()
        self._is_speaking = False
        self._delivered_text_parts: List[str] = []
        self._lock = threading.Lock()
        self._cache_dir = os.path.join(PROJECT_ROOT, "scratch", "tts_cache")
        os.makedirs(self._cache_dir, exist_ok=True)

        # 1. Primary Neural Voice Engine (Expressive Indian Female)
        try:
            from runtime.edge_tts_engine import NeuralEdgeTTSEngine
            edge_inst = NeuralEdgeTTSEngine()
            if edge_inst.is_available:
                self._edge_engine = edge_inst
        except Exception:
            self._edge_engine = None

        # 2. Local Neural Engine (Pocket TTS)
        if self.use_pocket_tts:
            try:
                from runtime.pocket_tts_engine import get_pocket_tts_engine
                engine = get_pocket_tts_engine()
                if engine and engine.is_ready():
                    self._pocket_engine = engine
            except Exception:
                self._pocket_engine = None

        self._init_sapi()

    def _init_sapi(self):
        """Initialize native Windows SAPI Voice."""
        if comtypes is not None:
            try:
                self._sapi_voice = comtypes.client.CreateObject("SAPI.SpVoice")
                if self._sapi_voice:
                    self._sapi_voice.Rate = self.rate
                    self._sapi_voice.Volume = self.volume
                    # Select preferred voice (Zira -> Hazel -> David)
                    voices = self._sapi_voice.GetVoices()
                    for i in range(voices.Count):
                        v = voices.Item(i)
                        desc = v.GetDescription().lower()
                        if self.voice_preference in desc:
                            self._sapi_voice.Voice = v
                            break
            except Exception:
                self._sapi_voice = None

    def stop(self):
        """Immediately abort speech synthesis and clear pending queue (instant barge-in)."""
        self._stop_event.set()
        if self._edge_engine:
            try:
                self._edge_engine.stop()
            except Exception:
                pass
        if self._pocket_engine:
            try:
                self._pocket_engine.stop()
            except Exception:
                pass
        if self._sapi_voice:
            try:
                # SVSFPurgeBeforeSpeak = 2 (immediately purges queue and stops current utterance)
                self._sapi_voice.Speak("", SpeechProsodyFormatter.SVSF_PURGE_BEFORE_SPEAK)
            except Exception:
                pass

        if pygame is not None and pygame.mixer.get_init():
            try:
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
            except Exception:
                pass

        self._is_speaking = False

    def is_speaking(self) -> bool:
        return self._is_speaking

    def get_delivered_text(self) -> str:
        """Return the text that was actually delivered to the user before any interruption."""
        with self._lock:
            return " ".join(self._delivered_text_parts).strip()

    def speak(
        self,
        text: str,
        delivery_style: str = "relaxed_casual",
        async_mode: bool = False,
        stream_phrases: bool = True,
        on_phrase_started: Optional[Callable[[str], None]] = None
    ):
        """Synthesize and speak text with prosody shaping.
        
        Args:
            text: Text to synthesize.
            delivery_style: 'technical_composed', 'relaxed_casual', or 'gentle_personal'.
            async_mode: If True, executes on a background daemon thread.
            stream_phrases: If True, streams sentence-by-sentence for low latency.
            on_phrase_started: Callback when a phrase starts playback.
        """
        if not text:
            return

        self._stop_event.clear()
        with self._lock:
            self._delivered_text_parts = []

        if async_mode:
            t = threading.Thread(
                target=self._synthesize_and_play,
                args=(text, delivery_style, stream_phrases, on_phrase_started),
                daemon=True
            )
            t.start()
        else:
            self._synthesize_and_play(text, delivery_style, stream_phrases, on_phrase_started)

    def _synthesize_and_play(
        self,
        text: str,
        delivery_style: str,
        stream_phrases: bool,
        on_phrase_started: Optional[Callable[[str], None]]
    ):
        self._is_speaking = True
        try:
            apply_contractions = (delivery_style in ["relaxed_casual", "gentle_personal"])
            clean_text = SpeechTextCleaner.clean(text, apply_contractions=apply_contractions)

            if stream_phrases:
                phrases = SpeechTextCleaner.split_into_phrases(clean_text)
            else:
                phrases = [clean_text]

            for phrase in phrases:
                if self._stop_event.is_set():
                    break
                if not phrase:
                    continue

                if on_phrase_started:
                    try:
                        on_phrase_started(phrase)
                    except Exception:
                        pass

                self._speak_phrase(phrase, delivery_style=delivery_style)

                with self._lock:
                    self._delivered_text_parts.append(phrase)

        finally:
            self._is_speaking = False

    def _speak_phrase(self, phrase: str, delivery_style: str = "relaxed_casual"):
        """Speak a single phrase via Neural Edge TTS (Expressive Indian Female), Pocket TTS, or SAPI5."""
        if self._stop_event.is_set():
            return

        # 1. Primary: High-fidelity Neural Edge TTS
        if self._edge_engine and self._edge_engine.is_available:
            try:
                res = self._edge_engine.speak(phrase, async_mode=False)
                if res.get("status") == "success":
                    return
            except Exception:
                pass

        # 2. Local Fallback: Pocket TTS
        if self._pocket_engine and self._pocket_engine.is_ready():
            try:
                ok = self._pocket_engine.speak(phrase, async_mode=False)
                if ok:
                    return
            except Exception:
                pass

        # 3. Local OS Fallback: native Windows SAPI
        if not self._sapi_voice:
            self._init_sapi()

        if self._sapi_voice and not self._stop_event.is_set():
            xml_text, flags = SpeechProsodyFormatter.format_prosody(phrase, style=delivery_style)
            try:
                # Attempt prosody-formatted XML speak
                self._sapi_voice.Speak(xml_text, flags)
            except Exception:
                try:
                    # Fallback to plain text speak if XML parsing fails
                    self._sapi_voice.Speak(phrase, SpeechProsodyFormatter.SVSF_DEFAULT)
                except Exception:
                    pass
