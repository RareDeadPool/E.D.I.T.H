"""E.D.I.T.H. V13 Affect Engine.

Maintains a continuous, persistent emotional state conditioning generation:
- Fast-moving reactions (immediate conversational sentiment)
- Slow-moving mood baseline that drifts within bounded ranges
- Humor suppression during detected user distress
- Generation of <mood>...</mood> conditioning tokens
"""

import json
import os
import sys
from typing import Dict, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class AffectEngine:
    STATE_FILE = os.path.join(PROJECT_ROOT, "config", "affect_state_v13.json")

    def __init__(self, state_file: Optional[str] = None):
        self.state_file = state_file or self.STATE_FILE
        # Valence: -1.0 (very negative) to +1.0 (very positive)
        # Arousal: 0.0 (calm/tired) to 1.0 (excited/energetic)
        # Playfulness: 0.0 (serious/focused) to 1.0 (witty/sarcastic)
        # Confidence: 0.0 (uncertain) to 1.0 (fully assured)
        self.valence: float = 0.65
        self.arousal: float = 0.50
        self.playfulness: float = 0.70
        self.confidence: float = 0.85

        self.user_in_distress: bool = False
        self.load()

    def update_affect(self, user_text: str, is_distress: bool = False):
        """Updates emotional state based on interaction cues."""
        text_lower = user_text.lower()
        self.user_in_distress = is_distress

        distress_keywords = [
            "sad", "depressed", "hopeless", "crying", "lost my job",
            "unemployed", "broken", "exhausted", "pain", "hurt", "grief",
            "upset", "struggling", "3 years", "trying from 3 years"
        ]
        if any(w in text_lower for w in distress_keywords):
            self.user_in_distress = True

        if self.user_in_distress:
            # Empathy mode: High warmth, low playfulness, grounding arousal
            self.playfulness = max(0.0, self.playfulness - 0.35)
            self.valence = 0.40  # Compassionate, attentive
            self.arousal = 0.35  # Calm, slow pacing
        elif any(w in text_lower for w in ["joke", "funny", "laugh", "roast", "bored"]):
            # Banter mode: High playfulness, high arousal
            self.playfulness = max(0.65, min(0.95, self.playfulness + 0.20))
            self.valence = min(0.90, self.valence + 0.10)
            self.arousal = 0.75

        elif any(w in text_lower for w in ["open", "close", "kill", "process", "calculate", "find"]):
            # Focused copilot mode
            self.playfulness = 0.40
            self.confidence = 0.95
        else:
            # Slow drift toward baseline
            self.playfulness = 0.95 * self.playfulness + 0.05 * 0.70
            self.valence = 0.95 * self.valence + 0.05 * 0.65
            self.arousal = 0.95 * self.arousal + 0.05 * 0.50

        self.save()

    def get_mood_token(self) -> str:
        """Returns the appropriate <mood> token conditioning EdithLM generation."""
        if self.user_in_distress:
            return "<mood>warm_empathy</mood>"
        if self.playfulness >= 0.60:
            return "<mood>playful_witty</mood>"
        if self.arousal <= 0.40:
            return "<mood>grounding_support</mood>"
        return "<mood>focused_copilot</mood>"

    def save(self):
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.state_file)), exist_ok=True)
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump({
                    "valence": round(self.valence, 3),
                    "arousal": round(self.arousal, 3),
                    "playfulness": round(self.playfulness, 3),
                    "confidence": round(self.confidence, 3),
                    "user_in_distress": self.user_in_distress,
                }, f, indent=2)
        except Exception:
            pass

    def load(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.valence = data.get("valence", 0.65)
                    self.arousal = data.get("arousal", 0.50)
                    self.playfulness = data.get("playfulness", 0.70)
                    self.confidence = data.get("confidence", 0.85)
                    self.user_in_distress = data.get("user_in_distress", False)
            except Exception:
                pass
