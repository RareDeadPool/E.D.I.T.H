"""Contextual Speech Recognition & Uncertainty-Aware Correction for E.D.I.T.H. V6.

Provides:
1. Multi-representation speech container:
   - audio_segment
   - raw_transcript
   - alternative_hypotheses (with logprob & confidence)
   - contextually_interpreted_transcript
   - inferred_semantic_meaning
   - intended_execution_task
2. Local faster-whisper INT8 offline neural transcription.
3. Streaming partial transcript generator.
4. Contextual correction engine integrating active files, contacts, apps, and dev terminology.
5. Four-tier confidence classification: High, Moderate, Low, High-Consequence Confirmation.
"""

import os
import sys
import re
import math
import time
import difflib
import threading
from typing import Dict, Any, List, Optional, Tuple, NamedTuple
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None


class RecognitionHypothesis(NamedTuple):
    text: str
    confidence: float
    avg_logprob: float


class SpeechRepresentation:
    """Preserves distinct, uncorrupted representations across the speech pipeline."""

    def __init__(
        self,
        audio_segment: Optional[np.ndarray],
        raw_transcript: str,
        confidence: float,
        alternative_hypotheses: List[RecognitionHypothesis],
        latency_ms: float
    ):
        self.audio_segment = audio_segment
        self.raw_transcript = raw_transcript
        self.raw_confidence = confidence
        self.alternative_hypotheses = alternative_hypotheses
        self.latency_ms = latency_ms

        # Contextual stages populated downstream
        self.contextually_interpreted_transcript: str = raw_transcript
        self.contextual_confidence: float = confidence
        self.correction_provenance: List[Dict[str, Any]] = []
        self.confidence_tier: str = "high"  # high, moderate, low, confirm_consequential
        self.clarification_question: Optional[str] = None

        # Cognitive stages
        self.inferred_semantic_meaning: Optional[Dict[str, Any]] = None
        self.intended_execution_task: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_transcript": self.raw_transcript,
            "raw_confidence": round(self.raw_confidence, 3),
            "contextually_interpreted_transcript": self.contextually_interpreted_transcript,
            "contextual_confidence": round(self.contextual_confidence, 3),
            "confidence_tier": self.confidence_tier,
            "correction_provenance": self.correction_provenance,
            "clarification_question": self.clarification_question,
            "alternatives": [
                {"text": h.text, "confidence": round(h.confidence, 3)}
                for h in self.alternative_hypotheses
            ],
            "latency_ms": round(self.latency_ms, 2)
        }


class ContextualCorrectionEngine:
    """Uncertainty-aware acoustic and contextual entity resolver without hallucination.
    
    Uses context from active files, contacts, apps, and dev context to correct
    speech-to-text slips (e.g. 'open test screen dot py' -> 'open test_screen.py',
    'message suhana' -> 'message Suhana') without inventing words when acoustic
    evidence is contradictory.
    """

    # Static phonetic normalization map for desktop and common domain terminology
    BASE_PHONETIC_CORRECTIONS = {
        r"\b(?:edit|eat it|e\.d\.i\.t\.h\.?|edith)\b": "edith",
        r"\b(?:note\s*pad|no\s*pad|notepd|notpad|note\s*pair|notepair|no\s*pair|not\s*pad|pair|pear|peyr)\b": "notepad",
        r"\b(?:spot\s*if\s*i|spot\s*a\s*fly|spotty\s*fly|spotfy|spotfiy|spot\s*high|spot\s*pie|spoty)\b": "spotify",
        r"\b(?:vs\s*code|v\s*s\s*code|visual\s*studio\s*code|vias\s*code)\b": "vscode",
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

    FILLER_PREFIXES = [
        r"^(?:um|uh|er|ah|like)\s+",
        r"^(?:hey\s+|yo\s+|hello\s+|hi\s+)?edith\s*,?\s*",
        r"^(?:can you\s+|could you\s+|please\s+|would you\s+|will you\s+|kindly\s+)+(?:like\s+)?",
    ]

    MATH_REPLACEMENTS = [
        (r"\bplus\b", "+"),
        (r"\bminus\b", "-"),
        (r"\btimes\b", "*"),
        (r"\bmultiplied\s+by\b", "*"),
        (r"\bdivided\s+by\b", "/"),
        (r"\bpercent\b", "%"),
    ]

    @staticmethod
    def _compute_string_similarity(a: str, b: str) -> float:
        """Compute character sequence similarity ratio (0.0 to 1.0)."""
        return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

    @classmethod
    def resolve_contextual_transcript(
        cls,
        raw_text: str,
        raw_confidence: float,
        context: Optional[Any] = None
    ) -> Tuple[str, float, str, List[Dict[str, Any]], Optional[str]]:
        """Contextually interpret raw transcription using active task and entity state.
        
        Returns:
            (corrected_text, contextual_confidence, confidence_tier, provenance, clarification_question)
        """
        if not raw_text or not raw_text.strip():
            return "", 0.0, "low", [], "I didn't catch that. Could you repeat?"

        text = raw_text.strip()
        provenance: List[Dict[str, Any]] = []

        # 1. Base phonetic corrections
        corrected = text
        for pat, repl in cls.BASE_PHONETIC_CORRECTIONS.items():
            if re.search(pat, corrected, re.IGNORECASE):
                old_text = corrected
                corrected = re.sub(pat, repl, corrected, flags=re.IGNORECASE)
                if old_text != corrected:
                    provenance.append({
                        "type": "phonetic_normalization",
                        "pattern": pat,
                        "replacement": repl
                    })

        # 2. Math speech conversion (if calculating)
        if any(w in corrected.lower() for w in ["calculate", "what is", "how much is", "+", "-", "*", "/", "plus", "times", "minus"]):
            for pat, repl in cls.MATH_REPLACEMENTS:
                if re.search(pat, corrected, re.IGNORECASE):
                    corrected = re.sub(pat, repl, corrected, flags=re.IGNORECASE)

        # 3. Strip conversational filler prefixes
        for pat in cls.FILLER_PREFIXES:
            if re.search(pat, corrected, re.IGNORECASE):
                corrected = re.sub(pat, "", corrected, flags=re.IGNORECASE).strip()

        # Clean trailing punctuation
        corrected = re.sub(r"[\.!\?]+$", "", corrected).strip()

        # 4. Contextual File & Extension Resolution
        # e.g., 'dot py' -> '.py', 'dot txt' -> '.txt', 'dot md' -> '.md'
        corrected = re.sub(r"\s+dot\s+(py|txt|json|md|html|css|js|sh|bat)\b", r".\1", corrected, flags=re.IGNORECASE)
        # e.g., 'open test screen.py' where context has 'test_screen.py'
        if context:
            candidate_files = []
            if hasattr(context, "search_candidates") and context.search_candidates:
                candidate_files.extend([c.get("name", "") for c in context.search_candidates if isinstance(c, dict)])
            if hasattr(context, "last_file") and context.last_file:
                candidate_files.append(context.last_file)

            # Check if any words in utterance match candidate files with fuzzy spacing
            for cand in candidate_files:
                if not cand:
                    continue
                cand_spoken = cand.replace("_", " ").replace("-", " ")
                cand_clean = cand.replace("\\", "/")
                # Check for phonetic/fuzzy presence
                cand_base = os.path.splitext(cand)[0].replace("_", " ")
                m = re.search(re.escape(cand_spoken), corrected, re.IGNORECASE)
                if m:
                    corrected = corrected[:m.start()] + cand_clean + corrected[m.end():]
                    provenance.append({"type": "contextual_file_match", "source": cand_spoken, "target": cand_clean})
                elif cand_base and re.search(re.escape(cand_base), corrected, re.IGNORECASE):
                    m_base = re.search(re.escape(cand_base), corrected, re.IGNORECASE)
                    if m_base:
                        # Append extension if file has one
                        corrected = corrected[:m_base.start()] + cand_clean + corrected[m_base.end():]
                        provenance.append({"type": "contextual_file_base_match", "source": cand_base, "target": cand_clean})

        # 5. Contextual Contact Name Resolution
        # e.g., 'message suhana' -> match with ContactResolver or context.last_person
        if context:
            known_contacts = []
            if hasattr(context, "last_person") and context.last_person:
                known_contacts.append(context.last_person)
            if hasattr(context, "selected_recipient") and context.selected_recipient:
                known_contacts.append(context.selected_recipient)

            try:
                from runtime.contact_resolver import ContactResolver
                resolver = ContactResolver()
                for c in resolver.list_contacts():
                    known_contacts.append(c["name"])
            except Exception:
                pass

            # Check for close matches to known contacts
            words = corrected.split()
            for i, w in enumerate(words):
                if len(w) >= 3 and w.lower() not in ["send", "text", "message", "to", "on", "whatsapp", "email", "open", "call"]:
                    for kc in known_contacts:
                        sim = cls._compute_string_similarity(w, kc)
                        if (sim >= 0.78 and w != kc) or (w.lower() == kc.lower() and w != kc):
                            words[i] = kc
                            provenance.append({"type": "contextual_contact_resolution", "source": w, "target": kc, "similarity": round(sim, 2)})
                            break
            corrected = " ".join(words)

        # 6. Fallback app alias normalization for bare verbs
        corrected = re.sub(r"^(open|launch|start|run)\s+(?:a\s+)?(?:pair|pear|peyr|pad)$", r"\1 notepad", corrected, flags=re.IGNORECASE)
        corrected = re.sub(r"^(close|quit|kill)\s+(?:a\s+)?(?:pair|pear|peyr|pad)$", r"\1 notepad", corrected, flags=re.IGNORECASE)

        # 7. Confidence Calibration & Evidence Tier Determination
        # Baseline confidence from acoustic model
        calibrated_conf = raw_confidence

        # If high-impact contextual corrections occurred, elevate confidence
        if provenance:
            calibrated_conf = min(1.0, calibrated_conf + 0.10)

        # Check for high-consequence operations (deleting files, sending external messages, shutting down system)
        is_consequential = any(
            re.search(pat, corrected, re.IGNORECASE)
            for pat in [
                r"\b(?:delete|remove|erase|format|wipe|trash)\b",
                r"\b(?:send|submit|transfer|pay)\b",
                r"\b(?:shut\s+down|power\s+off|restart|reboot)\b"
            ]
        )

        clarification_question: Optional[str] = None

        if is_consequential and calibrated_conf < 0.95:
            confidence_tier = "confirm_consequential"
            clarification_question = f"You asked to '{corrected}'. Since this performs a significant action, could you confirm before I proceed?"
        elif calibrated_conf >= 0.82:
            confidence_tier = "high"
        elif calibrated_conf >= 0.55:
            confidence_tier = "moderate"
            # If moderate confidence with ambiguity, formulate subtle clarification
            if len(corrected.split()) <= 2 and not any(a in corrected.lower() for a in ["notepad", "spotify", "chrome", "vscode", "calculator"]):
                clarification_question = f"Did you want me to {corrected}?"
        else:
            confidence_tier = "low"
            clarification_question = "I'm not quite sure I heard that clearly. Could you repeat that portion?"

        return corrected, round(calibrated_conf, 3), confidence_tier, provenance, clarification_question


class ContextualSpeechRecognizer:
    """100% Local Neural Speech Recognizer powered by faster-whisper (CTranslate2 INT8)."""

    DESKTOP_PROMPT = (
        "E.D.I.T.H., open notepad, open spotify, open calculator, open chrome, "
        "open vscode, open terminal, command prompt, play music, what time is it, "
        "turn volume up, turn volume down, mute, take screenshot, type text, "
        "send whatsapp message, system status, wifi, battery, write notes, "
        "do that again, why did you do that, explain that more simply, continue."
    )

    def __init__(self, model_size: str = "base.en", compute_type: str = "int8", device: str = "cpu"):
        self.model_size = model_size
        self.compute_type = compute_type
        self.device = device
        self._model: Optional[WhisperModel] = None
        self._lock = threading.Lock()
        self.correction_engine = ContextualCorrectionEngine()

    def _get_model(self) -> Optional[WhisperModel]:
        """Lazy loader for faster-whisper with automatic fallback to tiny.en."""
        if WhisperModel is None:
            return None

        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        self._model = WhisperModel(
                            self.model_size,
                            device=self.device,
                            compute_type=self.compute_type
                        )
                    except Exception as e:
                        sys.stderr.write(f"[ContextualSpeechRecognizer Load Warning] {e}\n")
                        if self.model_size != "tiny.en":
                            try:
                                self._model = WhisperModel(
                                    "tiny.en",
                                    device=self.device,
                                    compute_type=self.compute_type
                                )
                                self.model_size = "tiny.en"
                            except Exception:
                                self._model = None
                        else:
                            self._model = None
        return self._model

    def transcribe(
        self,
        audio: np.ndarray,
        context: Optional[Any] = None,
        language: str = "en"
    ) -> SpeechRepresentation:
        """Transcribe audio into multi-representation speech object with contextual interpretation."""
        t0 = time.time()
        raw_text = ""
        avg_logprob = -1.0
        confidence = 0.0
        alternatives: List[RecognitionHypothesis] = []

        model = self._get_model()
        if model is not None and len(audio) > 0:
            try:
                segments, info = model.transcribe(
                    audio,
                    language=language,
                    beam_size=5,
                    best_of=5,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=350),
                    initial_prompt=self.DESKTOP_PROMPT,
                    temperature=0.0
                )
                seg_texts = []
                logprobs = []
                for s in segments:
                    t = s.text.strip()
                    if t:
                        seg_texts.append(t)
                        if hasattr(s, "avg_logprob"):
                            logprobs.append(s.avg_logprob)

                raw_text = " ".join(seg_texts).strip()
                if logprobs:
                    avg_logprob = float(np.mean(logprobs))
                    # Calibrate probability: exp(logprob) clamped to [0.1, 0.99]
                    confidence = float(np.clip(np.exp(avg_logprob), 0.1, 0.99))
                else:
                    confidence = 0.88 if raw_text else 0.0

                if raw_text:
                    alternatives.append(RecognitionHypothesis(
                        text=raw_text,
                        confidence=confidence,
                        avg_logprob=avg_logprob
                    ))

            except Exception as e:
                sys.stderr.write(f"[ContextualSpeechRecognizer Transcribe Error] {e}\n")

        latency_ms = (time.time() - t0) * 1000.0

        # Construct multi-representation container
        rep = SpeechRepresentation(
            audio_segment=audio,
            raw_transcript=raw_text,
            confidence=confidence,
            alternative_hypotheses=alternatives,
            latency_ms=latency_ms
        )

        # Contextual correction without hallucination
        if raw_text:
            corr_text, corr_conf, tier, prov, clar_q = self.correction_engine.resolve_contextual_transcript(
                raw_text=raw_text,
                raw_confidence=confidence,
                context=context
            )
            rep.contextually_interpreted_transcript = corr_text
            rep.contextual_confidence = corr_conf
            rep.confidence_tier = tier
            rep.correction_provenance = prov
            rep.clarification_question = clar_q
        else:
            rep.confidence_tier = "low"
            rep.clarification_question = "I didn't detect any spoken input."

        return rep

    def transcribe_partial(self, audio_chunk: np.ndarray) -> str:
        """Fast low-latency partial transcription during speech streaming."""
        model = self._get_model()
        if model is None or len(audio_chunk) < 4000:
            return ""
        try:
            # Greedy 1-beam search for maximum streaming throughput
            segments, _ = model.transcribe(
                audio_chunk,
                language="en",
                beam_size=1,
                best_of=1,
                temperature=0.0
            )
            partial_text = " ".join([s.text.strip() for s in segments if s.text.strip()])
            return partial_text
        except Exception:
            return ""
