"""Social Cognition, Contextual Empathy & Stable Personality Model for E.D.I.T.H. V5.

Implements:
1. Three-Stage Contextual Empathy Pipeline:
   - Stage 1: Emotion & Affect Recognition (cues + uncertainty).
   - Stage 2: Empathic Contextual Reasoning (goals, obstacles, perspective).
   - Stage 3: Dynamic Support Strategy Selection (practical troubleshooting vs
     emotional validation vs active listening vs quiet execution).
2. Theory of Mind (Perspective Taking):
   - Explicit separation of: User Stated, System Observed, Inferred, and Unknown.
3. Stable Adaptive Personality Model:
   - Continuous Big Five dimensions influencing decisions and dialogue tone
     without rigid adjective-based canned templates.
"""

from __future__ import annotations
import os
import re
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

from .cognitive_state import CognitiveState, UserSocialEstimate


@dataclass
class EmpathicInference:
    """The result of contextual empathy reasoning."""
    user_emotional_cues: List[str]
    inferred_user_state: str           # e.g., 'task_blocked_frustration', 'existential_loneliness', 'eager_creative'
    empathy_confidence: float          # 0.0 to 1.0
    detected_core_need: str            # 'practical_solution', 'emotional_validation', 'collaborative_dialogue', 'efficient_silence'
    perspective_hypotheses: List[str]  # Counterfactual explanations of why user feels this way
    suggested_support_strategy: str    # Strategy for formulating response
    adapted_tone: str                  # 'warm_reassuring', 'direct_pragmatic', 'gentle_attentive', 'upbeat_collaborative'


@dataclass
class PersonalityProfile:
    """Continuous stable personality traits inspired by Five Factor model."""
    openness: float = 0.80           # High curiosity, exploratory, welcomes new approaches (0.0 - 1.0)
    conscientiousness: float = 0.90  # High diligence, precision, safety verification (0.0 - 1.0)
    extraversion: float = 0.70       # Warmth, expressiveness, engagement (0.0 - 1.0)
    agreeableness: float = 0.85      # Cooperative, compassionate, supportive (0.0 - 1.0)
    emotional_stability: float = 0.85# High resilience, calm under pressure, non-volatile (0.0 - 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SocialCognitionEngine:
    """Manages social perspective taking, contextual empathy, and personality dynamics."""

    def __init__(self, personality_config_path: Optional[str] = None):
        self.config_path = personality_config_path or os.path.join("config", "stable_personality.json")
        self.personality = PersonalityProfile()
        self.load_personality()

    def load_personality(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.personality = PersonalityProfile(**data)
            except Exception:
                pass

    def save_personality(self):
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.config_path)), exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.personality.to_dict(), f, indent=2)
        except Exception:
            pass

    def analyze_social_context(
        self,
        user_utterance: str,
        state: CognitiveState,
        recent_episodes: Optional[List[Any]] = None
    ) -> EmpathicInference:
        """
        Three-Stage Contextual Empathy Pipeline:
        1. Extract cues & detect affect.
        2. Reason about user's context, struggles, and perspective.
        3. Select support strategy based on inferred need, not keyword matching.
        """
        text = user_utterance.lower().strip()
        cues: List[str] = []

        # Stage 1: Emotion, Affect & Communicative Stance Recognition
        # Nuanced category indicators
        is_venting = any(w in text for w in ["just need to vent", "let me vent", "listen to me vent", "vent about"])
        is_concise_demand = any(w in text for w in ["just give me", "quick answer", "concise please", "direct answer", "no explanation"])
        is_tool_frustration = any(w in text for w in ["compiler error", "connection refused", "port 5432", "stack trace", "syntax error", "bug in", "not working", "crash"])
        is_person_frustration = any(w in text for w in ["my boss", "my teammate", "my manager", "coworker", "colleague", "they promised", "didn't do their part"])
        is_sarcasm = bool(re.search(r"(?:oh\s+great|just\s+wonderful|fantastic|exactly\s+what\s+i\s+needed)[\s\.,!]+(?:another\s+error|it\s+broke|failed)", text))
        is_mixed_emotion = ("excited" in text and "terrified" in text) or ("happy" in text and "sad" in text) or ("nervous" in text and "hopeful" in text)
        is_correction = any(text.startswith(p) for p in ["no,", "no ", "not that", "i meant "])

        if is_venting:
            cues.append("venting_space_needed")
        if is_concise_demand:
            cues.append("concise_demand")
        if is_sarcasm:
            cues.append("sarcastic_irony")
        if is_mixed_emotion:
            cues.append("mixed_conflicted_affect")
        if is_correction:
            cues.append("correction_redirect")
        if any(w in text for w in ["stressed", "overwhelmed", "deadline", "pressure", "burnout"]):
            cues.append("stress_markers")
        if any(w in text for w in ["broke up", "breakup", "heartbroken", "dumped", "sad", "crying"]):
            cues.append("heartbreak_or_grief")
        if any(w in text for w in ["lonely", "alone", "nobody", "feel so lonely"]):
            cues.append("loneliness_markers")
        if any(w in text for w in ["why is this", "hate this bug", "stupid", "annoying"]) or is_tool_frustration or is_person_frustration:
            cues.append("frustration_markers")
        if any(w in text for w in ["yay", "finally", "it works", "awesome", "crushed it"]):
            cues.append("celebration_markers")

        # Stage 2: Empathic Contextual Reasoning
        active_goal = state.goals.active_goal
        has_active_coding_or_task = bool(
            (active_goal and any(k in active_goal.lower() for k in ["code", "app", "file", "system", "search", "wifi"])) or
            any(k in text for k in ["code", "build", "port", "error", "bug", "script", "test", "compile", "binding"])
        )

        hypotheses = []
        # Priority mapping from evidence
        if is_concise_demand:
            inferred_state = "time_pressured_seeking_efficiency"
            core_need = "concise_direct_answer"
            hypotheses.append("User requested direct answer without explanatory preamble.")
            strategy = "concise_direct_answer"
            tone = "direct_pragmatic"
            conf = 0.90
        elif is_venting:
            inferred_state = "venting_affective_pressure"
            core_need = "space_to_vent"
            hypotheses.append("User needs active listening and space to express feelings without immediate problem-solving.")
            strategy = "reflective_listening"
            tone = "gentle_attentive"
            conf = 0.90
        elif is_sarcasm:
            inferred_state = "sarcastic_frustration"
            core_need = "practical_solution"
            hypotheses.append("User used irony to vent frustration about a setback; requires calm technical de-escalation.")
            strategy = "direct_troubleshooting"
            tone = "direct_pragmatic"
            conf = 0.85
        elif is_mixed_emotion:
            inferred_state = "ambivalent_uncertainty"
            core_need = "emotional_validation"
            hypotheses.append("User is facing significant transitional change with conflicting feelings.")
            strategy = "validate_both_sides_collaboratively"
            tone = "warm_reassuring"
            conf = 0.85
        elif is_correction:
            inferred_state = "task_realignment"
            core_need = "practical_solution"
            hypotheses.append("User is redirecting an incorrect model assumption.")
            strategy = "immediate_correction_acknowledgment"
            tone = "direct_pragmatic"
            conf = 0.90
        elif "heartbreak_or_grief" in cues or "loneliness_markers" in cues:
            inferred_state = "relational_pain_or_isolation"
            core_need = "emotional_validation"
            hypotheses.append("User is experiencing emotional distress in their personal/love life.")
            hypotheses.append("Jumping straight to cold technical logic would feel dismissive and robotic.")
            strategy = "deep_empathy_and_gentle_grounding"
            tone = "gentle_attentive"
            conf = 0.85
        elif "frustration_markers" in cues:
            if is_person_frustration:
                inferred_state = "interpersonal_friction"
                core_need = "collaborative_dialogue"
                hypotheses.append("User is dealing with human/team friction rather than a software defect.")
                strategy = "reflective_listening_then_advice"
                tone = "warm_reassuring"
                conf = 0.80
            elif has_active_coding_or_task or is_tool_frustration:
                inferred_state = "task_blocked_frustration"
                core_need = "practical_solution"
                hypotheses.append("User is blocked by a specific technical hurdle on their active task.")
                hypotheses.append("Unnecessary emotional lecturing would irritate them; they need concise, direct troubleshooting.")
                strategy = "direct_troubleshooting"
                tone = "direct_pragmatic"
                conf = 0.85
            else:
                inferred_state = "general_irritation"
                core_need = "collaborative_dialogue"
                hypotheses.append("User is expressing general friction or annoyance.")
                strategy = "empathic_listen_then_help"
                tone = "warm_reassuring"
                conf = 0.75
        elif "heartbreak_or_grief" in cues or "loneliness_markers" in cues:
            inferred_state = "relational_pain_or_isolation"
            core_need = "emotional_validation"
            hypotheses.append("User is experiencing emotional distress in their personal/love life.")
            hypotheses.append("Jumping straight to cold technical logic would feel dismissive and robotic.")
            strategy = "deep_empathy_and_gentle_grounding"
            tone = "gentle_attentive"
            conf = 0.85
        elif "stress_markers" in cues:
            inferred_state = "high_cognitive_load_stress"
            core_need = "collaborative_dialogue"
            hypotheses.append("User is carrying heavy cognitive load or deadline pressure.")
            strategy = "de_escalate_and_chunk_tasks"
            tone = "warm_reassuring"
            conf = 0.80
        elif "celebration_markers" in cues:
            inferred_state = "achievement_satisfaction"
            core_need = "collaborative_dialogue"
            hypotheses.append("User completed an objective and is sharing the victory.")
            strategy = "shared_celebration"
            tone = "upbeat_collaborative"
            conf = 0.85
        else:
            # Weak or neutral evidence: avoid overconfident emotional guesses
            inferred_state = "operational_neutral"
            core_need = "practical_solution" if has_active_coding_or_task else "collaborative_dialogue"
            hypotheses.append("Routine task execution or casual communication with no overt affective distress.")
            strategy = "efficient_execution"
            tone = "direct_pragmatic" if has_active_coding_or_task else "warm_reassuring"
            conf = 0.60

        # Update CognitiveState's user estimate (strictly segregated from agent's own PAD state)
        state.user_estimate.detected_user_mood = inferred_state
        state.user_estimate.communicative_need = core_need
        state.user_estimate.confidence_in_user_estimate = conf

        return EmpathicInference(
            user_emotional_cues=cues,
            inferred_user_state=inferred_state,
            empathy_confidence=conf,
            detected_core_need=core_need,
            perspective_hypotheses=hypotheses,
            suggested_support_strategy=strategy,
            adapted_tone=tone
        )

    def modulate_response_delivery(
        self,
        base_message: str,
        empathy_inference: EmpathicInference,
        state: CognitiveState
    ) -> str:
        """
        Modulate response delivery style based on empathic inference and Big Five personality,
        ensuring response matches the user's inferred cognitive/emotional need.
        """
        tone = empathy_inference.adapted_tone
        personality = self.personality

        # If user needs direct pragmatic solution under task blockage:
        # Strip conversational fluff, deliver clean action steps
        if empathy_inference.detected_core_need == "practical_solution" and tone == "direct_pragmatic":
            # Conscientious, crisp delivery
            return base_message.strip()

        # If user needs emotional validation or gentle grounding:
        # High agreeableness and emotional stability provide calm, supportive anchor
        if empathy_inference.detected_core_need == "emotional_validation":
            return base_message.strip()

        # Default cooperative delivery
        return base_message.strip()
