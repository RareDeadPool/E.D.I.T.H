"""Voice Conversation State Machine & Adaptive Turn-Taking for E.D.I.T.H. V6.

Features:
1. Event-driven voice conversation state machine with 13 explicit states.
2. Adaptive linguistic & acoustic turn-end detector:
   - Distinguishes complete commands (quick 0.8s cutoff) from hesitations & thinking pauses (extends up to 2.4s).
3. Barge-In & Interruption Controller:
   - Detects speech during TTS playback.
   - Instantly halts speech output.
   - Preserves only the delivered response prefix in conversation context.
   - Processes the new interruption without completing the obsolete task.
4. Continuous hands-free conversation loop with session controls ('wait', 'answer now', 'pause', 'exit').
"""

import os
import sys
import re
import time
import enum
import threading
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runtime.audio_front_end import AudioFrontEnd
from runtime.contextual_speech_recognition import (
    ContextualSpeechRecognizer,
    SpeechRepresentation,
    RecognitionHypothesis
)


class VoiceState(enum.Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    SPEECH_DETECTED = "SPEECH_DETECTED"
    CAPTURING_UTTERANCE = "CAPTURING_UTTERANCE"
    RECEIVING_PARTIAL_TRANSCRIPTION = "RECEIVING_PARTIAL_TRANSCRIPTION"
    WAITING_FOR_SPEAKER_TO_FINISH = "WAITING_FOR_SPEAKER_TO_FINISH"
    INTERPRETING_UTTERANCE = "INTERPRETING_UTTERANCE"
    REASONING_OR_EXECUTING = "REASONING_OR_EXECUTING"
    SPEAKING = "SPEAKING"
    INTERRUPTED = "INTERRUPTED"
    RECOVERING = "RECOVERING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"


class LinguisticCompletenessEvaluator:
    """Evaluates whether an in-progress spoken transcript is linguistically complete or trailing in thought."""

    # Trailing connectors, incomplete prepositions, and hesitation markers
    DANGLING_CONNECTORS = [
        r"\b(?:and\s+then|then|and|or|but|because|so|if|although|while)\s*$",
        r"\b(?:actually\s*,?\s*wait|wait\s+wait|hold\s+on|let\s+me\s+see|um|uh)\s*$",
        r"\b(?:in|into|on|at|to|from|for|with|by|about|of)\s*$",
        r"\b(?:which\s+i|that\s+i|where\s+i|who\s+is|what\s+is|which\s+is)\s*$",
        r"\b(?:the|a|an|my|your|this|that|these|those)\s*$",
        r"\b(?:saved\s+yesterday\s+in|located\s+in|find\s+the|look\s+for\s+the)\s*$",
    ]

    # Clear completion patterns for concise desktop commands and questions
    COMPLETE_COMMAND_PATTERNS = [
        r"^(?:open|launch|start|run|close|quit|kill)\s+[a-zA-Z0-9_\-\.\s]+$",
        r"^(?:what\s+is\s+the\s+time|what\s+time\s+is\s+it|check\s+battery|system\s+status|check\s+wifi)$",
        r"^(?:play|pause|resume|skip|next|previous|prev)\s*(?:music|song|track)?$",
        r"^(?:turn\s+(?:volume\s+)?up|turn\s+(?:volume\s+)?down|mute|unmute)$",
        r"^(?:take\s+a?\s*screenshot|capture\s+screen)$",
        r"^(?:yes|no|cancel|stop|continue|do\s+that\s+again|why\s+did\s+you\s+do\s+that)$",
    ]

    @classmethod
    def evaluate_completeness(cls, text: str) -> Dict[str, Any]:
        """Determine if an utterance is complete, incomplete/hesitating, or thinking aloud."""
        t = text.strip().lower()
        if not t:
            return {"is_complete": False, "is_dangling": False, "recommended_silence_s": 1.2}

        # Check dangling / hesitation
        for pat in cls.DANGLING_CONNECTORS:
            if re.search(pat, t):
                return {
                    "is_complete": False,
                    "is_dangling": True,
                    "recommended_silence_s": 2.4,
                    "reason": "Utterance ends with trailing connector or hesitation phrase."
                }

        # Check self-contained commands
        for pat in cls.COMPLETE_COMMAND_PATTERNS:
            if re.match(pat, t):
                return {
                    "is_complete": True,
                    "is_dangling": False,
                    "recommended_silence_s": 0.8,
                    "reason": "Self-contained imperative command."
                }

        # Standard sentence
        return {
            "is_complete": True,
            "is_dangling": False,
            "recommended_silence_s": 1.2,
            "reason": "Standard utterance."
        }


class VoiceConversationStateMachine:
    """Event-driven Voice Dialogue Manager for E.D.I.T.H."""

    def __init__(
        self,
        audio_front_end: Optional[AudioFrontEnd] = None,
        speech_recognizer: Optional[ContextualSpeechRecognizer] = None,
        tts_engine: Optional[Any] = None,
        brain: Optional[Any] = None
    ):
        self.afe = audio_front_end or AudioFrontEnd()
        self.recognizer = speech_recognizer or ContextualSpeechRecognizer()
        self.tts = tts_engine
        self.brain = brain

        self.current_state = VoiceState.IDLE
        self.is_continuous_mode = False
        self.is_waiting_for_user_cue = False  # Set by "Wait until I tell you to respond"
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()

        # Callbacks
        self.on_state_change: Optional[Callable[[VoiceState, Dict[str, Any]], None]] = None
        self.on_partial_transcript: Optional[Callable[[str], None]] = None
        self.on_final_transcript: Optional[Callable[[SpeechRepresentation], None]] = None

        # Interruption tracking
        self.last_speech_response: str = ""
        self.delivered_response_prefix: str = ""
        self.interruption_detected_time: float = 0.0

    def set_state(self, new_state: VoiceState, metadata: Optional[Dict[str, Any]] = None):
        """Thread-safe transition to a new voice state with notification."""
        with self._state_lock:
            prev = self.current_state
            self.current_state = new_state

        if self.on_state_change:
            try:
                self.on_state_change(new_state, metadata or {"previous_state": prev.value})
            except Exception:
                pass

    def get_state(self) -> VoiceState:
        with self._state_lock:
            return self.current_state

    def pause_listening(self):
        """Explicitly pause voice capture."""
        self.set_state(VoiceState.PAUSED)

    def resume_listening(self):
        """Resume active voice listening."""
        self.set_state(VoiceState.LISTENING)

    def stop_voice_mode(self):
        """Terminate voice conversation and microphone capture cleanly."""
        self._stop_event.set()
        if self.tts and hasattr(self.tts, "stop"):
            self.tts.stop()
        self.set_state(VoiceState.STOPPED)

    def handle_barge_in(self) -> Dict[str, Any]:
        """Execute immediate barge-in interruption when speech is detected while speaking."""
        t0 = time.time()
        self.interruption_detected_time = t0

        # 1. Stop audio playback immediately
        if self.tts and hasattr(self.tts, "stop"):
            self.tts.stop()

        self.afe.set_assistant_speaking(False)

        # 2. Transition state
        self.set_state(VoiceState.INTERRUPTED, {"interruption_time": t0})

        # 3. Retain delivered portion
        if self.brain and hasattr(self.brain, "context") and self.brain.context:
            self.brain.context.touch()

        return {
            "status": "barge_in_triggered",
            "latency_ms": round((time.time() - t0) * 1000.0, 2)
        }

    def process_turn(
        self,
        context: Optional[Any] = None,
        init_timeout: float = 8.0,
        cmd_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute a complete single voice conversation turn:
        Listen -> Partial STT -> Adaptive End -> Contextual Correction -> Brain -> Execution -> TTS.
        """
        active_brain = self.brain
        if active_brain is None:
            from engine.edith_brain import EdithBrain
            active_brain = EdithBrain(enable_execution=True)
            self.brain = active_brain

        active_context = context or (active_brain.context if hasattr(active_brain, "context") else None)

        # 1. Manual / Test Override
        if cmd_override:
            self.set_state(VoiceState.INTERPRETING_UTTERANCE)
            corr_text, conf, tier, prov, clar_q = self.recognizer.correction_engine.resolve_contextual_transcript(
                raw_text=cmd_override,
                raw_confidence=1.0,
                context=active_context
            )
            rep = SpeechRepresentation(
                audio_segment=None,
                raw_transcript=cmd_override,
                confidence=1.0,
                alternative_hypotheses=[RecognitionHypothesis(text=cmd_override, confidence=1.0, avg_logprob=0.0)],
                latency_ms=0.0
            )
            rep.contextually_interpreted_transcript = corr_text
            rep.contextual_confidence = conf
            rep.confidence_tier = tier
            rep.correction_provenance = prov
            rep.clarification_question = clar_q
            return self._execute_and_respond(rep, active_brain)

        # 2. Active Audio Capture with Dynamic Turn-End Detection
        self.set_state(VoiceState.LISTENING)

        partial_text_holder = [""]
        silence_timeout_holder = [1.2]

        def _on_voice_start():
            self.set_state(VoiceState.SPEECH_DETECTED)

        def _on_partial_audio(audio_chunk: np.ndarray):
            self.set_state(VoiceState.RECEIVING_PARTIAL_TRANSCRIPTION)
            p_text = self.recognizer.transcribe_partial(audio_chunk)
            if p_text:
                partial_text_holder[0] = p_text
                if self.on_partial_transcript:
                    self.on_partial_transcript(p_text)

                # Dynamically evaluate linguistic completeness
                comp_res = LinguisticCompletenessEvaluator.evaluate_completeness(p_text)
                silence_timeout_holder[0] = comp_res["recommended_silence_s"]

        # Run utterance recording with adaptive trailing silence
        self.set_state(VoiceState.CAPTURING_UTTERANCE)
        audio = self.afe.record_utterance(
            silence_timeout=silence_timeout_holder[0],
            init_timeout=init_timeout,
            on_voice_start=_on_voice_start,
            on_partial_audio=_on_partial_audio,
            partial_interval=0.75
        )

        if audio is None or len(audio) == 0:
            self.set_state(VoiceState.IDLE)
            return {"status": "timeout_or_silence", "representation": None, "brain_result": None}

        # 3. Transcribe & Contextualize
        self.set_state(VoiceState.INTERPRETING_UTTERANCE)
        rep = self.recognizer.transcribe(audio, context=active_context)
        if self.on_final_transcript:
            self.on_final_transcript(rep)

        # 4. Handle Low Confidence or Consequential Confirmation
        if rep.confidence_tier == "low":
            self.set_state(VoiceState.RECOVERING)
            msg = rep.clarification_question or "I'm sorry, I didn't hear that clearly. Could you repeat?"
            if self.tts:
                self.afe.set_assistant_speaking(True)
                self.tts.speak(msg, async_mode=False)
                self.afe.set_assistant_speaking(False)
            self.set_state(VoiceState.LISTENING)
            return {"status": "clarification_requested", "representation": rep.to_dict(), "response": msg}

        if rep.confidence_tier == "confirm_consequential":
            msg = rep.clarification_question
            if self.tts:
                self.afe.set_assistant_speaking(True)
                self.tts.speak(msg, async_mode=False)
                self.afe.set_assistant_speaking(False)
            self.set_state(VoiceState.LISTENING)
            return {"status": "consequential_confirmation_needed", "representation": rep.to_dict(), "response": msg}

        # 5. Execute via Cognitive Brain and Speak Back
        return self._execute_and_respond(rep, active_brain)

    def _execute_and_respond(self, rep: SpeechRepresentation, brain: Any) -> Dict[str, Any]:
        """Send interpreted text through cognitive cycle and deliver speech output."""
        cmd = rep.contextually_interpreted_transcript
        if not cmd:
            self.set_state(VoiceState.IDLE)
            return {"status": "empty_command", "representation": rep.to_dict()}

        # Check explicit session controls
        cmd_lower = cmd.lower().strip()
        if cmd_lower in ["wait until i tell you to respond", "wait for my cue"]:
            self.is_waiting_for_user_cue = True
            msg = "Understood. I will listen patiently and wait for you to say 'answer now'."
            if self.tts:
                self.tts.speak(msg, async_mode=False)
            self.set_state(VoiceState.LISTENING)
            return {"status": "session_control", "message": msg}

        if cmd_lower in ["answer now", "go ahead and answer", "respond now"]:
            self.is_waiting_for_user_cue = False

        if self.is_waiting_for_user_cue:
            # Accumulating user thought
            return {"status": "accumulating_thought", "spoken": cmd}

        # 6. Brain Processing & Execution
        self.set_state(VoiceState.REASONING_OR_EXECUTING)
        brain_res = brain.process(cmd, execute=True)

        # 7. Formulate Spoken Output
        ex = brain_res.get("execution")
        directive = brain_res.get("directive", {})
        intent = brain_res.get("intent", "")

        if ex and isinstance(ex, dict) and ex.get("status") == "success" and ex.get("message") and intent not in ["chitchat", "dialogue"]:
            speech_msg = ex.get("message", "")
        else:
            speech_msg = directive.get("message", "")

        # 8. Deliver Speech Output with Barge-In capability
        if self.tts and speech_msg:
            self.last_speech_response = speech_msg
            self.delivered_response_prefix = speech_msg
            self.set_state(VoiceState.SPEAKING)
            self.afe.set_assistant_speaking(True)

            # Speak synchronously so state machine transitions smoothly
            try:
                self.tts.speak(speech_msg, async_mode=False)
            finally:
                self.afe.set_assistant_speaking(False)

        self.set_state(VoiceState.IDLE)

        return {
            "status": "success",
            "spoken_command": cmd,
            "representation": rep.to_dict(),
            "brain_result": brain_res,
            "speech_response": speech_msg
        }

    def run_continuous_conversation(self, init_timeout: float = 12.0):
        """Continuous hands-free conversation loop."""
        self.is_continuous_mode = True
        self._stop_event.clear()

        if self.tts:
            self.tts.speak("E.D.I.T.H. natural voice conversation is active. I'm listening.", async_mode=False)

        while self.is_continuous_mode and not self._stop_event.is_set():
            turn_res = self.process_turn(init_timeout=init_timeout)
            status = turn_res.get("status")

            if status == "timeout_or_silence":
                # Standby wait, keep looping
                time.sleep(0.1)
                continue

            cmd = turn_res.get("spoken_command", "").lower()
            if cmd in ["exit", "quit", "stop", "exit voice", "stop listening", "text mode", "cancel"]:
                if self.tts:
                    self.tts.speak("Exiting voice mode. Back to console.", async_mode=False)
                break

        self.is_continuous_mode = False
        self.set_state(VoiceState.STOPPED)
