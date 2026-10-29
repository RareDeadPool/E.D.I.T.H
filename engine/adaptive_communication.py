"""Adaptive Communication Engine for E.D.I.T.H. V5.2.

Implements evidence-based communication adaptation:
1. Tracks evolving communication style preferences:
   - Verbosity: 'concise' vs 'detailed_technical'
   - Directness: 'direct_first' vs 'step_by_step'
   - Execution Autonomy: 'autonomous' vs 'confirm_first'
2. Evidence Recency & Decay:
   - Avoids permanent lock-in from a single interaction.
   - Single interactions contribute bounded weight (0.20), decaying over inactive days.
   - Repeated consistent feedback reinforces preferences up to a ceiling (0.95).
3. Dynamic Per-Turn Overrides:
   - Explicit turn instructions (e.g. 'explain in detail this time') take precedence
     for the current turn without overwriting long-term baseline profile.
"""

from typing import Dict, Any, Optional
import os
import time
import json
import re
from dataclasses import dataclass, field, asdict


@dataclass
class CommunicationStylePreferences:
    """User communication preferences with confidence tracking and recency decay."""
    verbosity: str = "balanced"         # 'concise', 'balanced', 'detailed_technical'
    verbosity_confidence: float = 0.5   # 0.0 to 1.0

    directness: str = "direct_first"    # 'direct_first', 'step_by_step'
    directness_confidence: float = 0.5

    autonomy_preference: str = "balanced"  # 'autonomous', 'balanced', 'confirm_first'
    autonomy_confidence: float = 0.5

    last_updated: float = field(default_factory=time.time)
    observation_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AdaptiveCommunicationEngine:
    """Coordinates evidence-based adaptation of response style, verbosity, and format."""

    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = config_dir or os.path.join("config", "communication")
        self.config_file = os.path.join(self.config_dir, "style_preferences.json")
        self.preferences = CommunicationStylePreferences()
        os.makedirs(self.config_dir, exist_ok=True)
        self.load_preferences()

    def load_preferences(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.preferences = CommunicationStylePreferences(**data)
                    self._apply_temporal_decay()
            except Exception:
                pass

    def save_preferences(self):
        try:
            os.makedirs(self.config_dir, exist_ok=True)
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(self.preferences.to_dict(), f, indent=2)
        except Exception:
            pass

    def _apply_temporal_decay(self, decay_rate_per_day: float = 0.05):
        """Gradually decay confidence of stored preferences towards default over inactive days."""
        now = time.time()
        days_passed = max(0.0, (now - self.preferences.last_updated) / 86400.0)
        if days_passed > 1.0:
            factor = max(0.2, 1.0 - (days_passed * decay_rate_per_day))
            self.preferences.verbosity_confidence = round(self.preferences.verbosity_confidence * factor, 3)
            self.preferences.directness_confidence = round(self.preferences.directness_confidence * factor, 3)
            self.preferences.autonomy_confidence = round(self.preferences.autonomy_confidence * factor, 3)
            self.preferences.last_updated = now

    def update_from_feedback(
        self,
        feedback_category: str,
        feedback_details: Dict[str, Any]
    ) -> bool:
        """
        Incrementally update preferences from verified user feedback.
        A single interaction produces a bounded delta (0.20), preventing rigid lock-in.
        """
        pref_code = feedback_details.get("preference", "")
        if not pref_code and feedback_category != "style_preference":
            return False

        changed = False
        step_weight = 0.20
        self.preferences.observation_count += 1
        self.preferences.last_updated = time.time()

        if pref_code in ["concise", "short", "less_talking"]:
            if self.preferences.verbosity == "concise":
                self.preferences.verbosity_confidence = min(0.95, self.preferences.verbosity_confidence + step_weight)
            else:
                self.preferences.verbosity = "concise"
                self.preferences.verbosity_confidence = 0.60
            changed = True

        elif pref_code in ["detailed", "step_by_step", "explain_more"]:
            if self.preferences.verbosity == "detailed_technical":
                self.preferences.verbosity_confidence = min(0.95, self.preferences.verbosity_confidence + step_weight)
            else:
                self.preferences.verbosity = "detailed_technical"
                self.preferences.verbosity_confidence = 0.60
            changed = True

        elif pref_code in ["direct_first", "bottom_line"]:
            self.preferences.directness = "direct_first"
            self.preferences.directness_confidence = min(0.95, self.preferences.directness_confidence + step_weight)
            changed = True

        elif pref_code in ["require_confirmation", "ask_first"]:
            self.preferences.autonomy_preference = "confirm_first"
            self.preferences.autonomy_confidence = min(0.95, self.preferences.autonomy_confidence + step_weight)
            changed = True

        if changed:
            self.save_preferences()
        return changed

    def determine_turn_communication_plan(
        self,
        user_utterance: str,
        task_intent: str = "general"
    ) -> Dict[str, Any]:
        """
        Determine the communication format for the active turn.
        Checks for immediate per-turn overrides (e.g. 'be brief this time' vs 'explain everything').
        """
        u_lower = user_utterance.lower().strip()

        # 1. Check for immediate explicit turn overrides
        turn_verbosity = self.preferences.verbosity
        turn_directness = self.preferences.directness
        override_active = False

        if any(w in u_lower for w in ["in detail this time", "give me the full explanation", "explain fully", "step by step"]):
            turn_verbosity = "detailed_technical"
            turn_directness = "step_by_step"
            override_active = True
        elif any(w in u_lower for w in ["just give me the answer", "be brief this time", "quick answer", "concise please", "short summary"]):
            turn_verbosity = "concise"
            turn_directness = "direct_first"
            override_active = True

        # 2. Intent-conditioned adaptations
        if task_intent in ["system_alert", "safety_confirmation"]:
            turn_directness = "direct_first"
            turn_verbosity = "concise"

        return {
            "verbosity": turn_verbosity,
            "directness": turn_directness,
            "autonomy_preference": self.preferences.autonomy_preference,
            "override_applied": override_active,
            "confidence": self.preferences.verbosity_confidence
        }
