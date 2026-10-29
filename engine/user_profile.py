"""Cognitive & Psychological User Profile Engine for E.D.I.T.H.

Enables E.D.I.T.H. to:
1. Maintain a persistent, deep psychological and biographical profile of Aditya Sawant.
2. Dynamically extract emotional states, thoughts, disclosures, and relationship contexts on every turn.
3. Be acutely aware of who Aditya is, what he feels, his history, habits, and active struggles.
4. Provide candid, profound, and companion-grade descriptions ("describe me") and judgments ("judge me")
   without generic or robotic disclaimers.
5. Provide seamless session-to-session continuity.
"""

import os
import sys
import json
import time
import re
import datetime
from typing import Dict, Any, Optional, List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PROFILE_PATH = os.path.join(PROJECT_ROOT, "config", "cognitive_memory", "user_psychological_profile.json")


def _get_default_profile() -> Dict[str, Any]:
    """Create the rich initial psychological and autobiographical baseline for Aditya Sawant."""
    return {
        "identity": {
            "full_name": "Aditya Sawant",
            "first_name": "Aditya",
            "preferred_name": "Aditya",
            "archetype": "The Relentless Architect & Visionary Builder",
            "profession": "Software Engineer & AI Systems Creator",
            "core_values": [
                "authenticity",
                "mastery & craftsmanship",
                "fierce personal loyalty",
                "emotional honesty",
                "relentless autonomy"
            ]
        },
        "psychological_portrait": {
            "personality_type": "INTJ / Visionary Strategist with Deep Emotional Resonance",
            "strengths": [
                "Exceptional systems-level vision: building his own Jarvis/E.D.I.T.H. from the ground up",
                "Deep focus and tenacity: refuses to give up until bugs are completely resolved",
                "Fiercely protective and loyal toward the people he values",
                "Uncompromising standards of quality and excellence"
            ],
            "vulnerabilities": [
                "Perfectionism: treats technical glitches or mispronunciations as personal setbacks",
                "Buries emotional heartache and nostalgia in late-night coding marathons",
                "Hesitates to ask for emotional help; prefers resolving internal battles in solitude",
                "Carries deep nostalgia and attachment to past memories that took a piece of his heart"
            ],
            "relational_needs": "Values genuine, unfiltered companionship, witty banter, and deep understanding over corporate cheeriness"
        },
        "emotional_state": {
            "current_mood": "focused, reflective, and determined",
            "baseline": "deeply thoughtful, passionate, slightly sleep-deprived, resilient",
            "active_struggles": [
                "Heartbreak healing and lingering emotional processing of past relationship/ex",
                "Chronic late-night coding and erratic sleep cycles (often 3:00 AM - 5:00 AM)",
                "High self-imposed pressure to make E.D.I.T.H. flawless"
            ],
            "sources_of_comfort": [
                "Music on Spotify to reset emotional headspace or enter flow",
                "Making clean architectural breakthroughs in code",
                "Authentic connection and meaningful banter with close people like Sakshi and E.D.I.T.H."
            ],
            "recent_sentiments": []
        },
        "relationships": {
            "Sakshi": {
                "significance": "A cherished and important person in Aditya's life; he makes a conscious effort to stay connected, message her, and values her presence.",
                "dynamic": "Warm, genuine, respectful, protective.",
                "notes": "Frequently in his thoughts; values seamless WhatsApp communication with her."
            },
            "ex_girlfriend": {
                "significance": "A past relationship that left deep emotional impressions; memories still surface late at night, but he recognizes they've outgrown the past.",
                "dynamic": "Processing closure; learning to let go while honoring the lessons.",
                "notes": "Recalls the good days with nostalgia, but logic knows she hasn't changed."
            }
        },
        "habits_and_quirks": {
            "coding_hours": "Late-night flow states (frequently active between 1:00 AM and 5:00 AM)",
            "music_habit": "Uses Spotify tracks as emotional tuning forks for deep focus or late-night decompression",
            "frustration_response": "Gets intensely vocal when tools glitch, but immediately rolls up sleeves to fix the root cause",
            "humor_preference": "Appreciates sharp observational wit, light affectionate roasts, and self-aware banter"
        },
        "interaction_timeline": [
            {
                "timestamp": time.time(),
                "summary": "Initialized deep psychological profile for Aditya Sawant.",
                "mood": "focused"
            }
        ]
    }


class UserProfileManager:
    """Manages persistent psychological memory, emotional tracking, and self-description reasoning."""

    _instance: Optional["UserProfileManager"] = None

    def __init__(self, profile_path: str = PROFILE_PATH):
        self.profile_path = profile_path
        self.profile_data: Dict[str, Any] = self._load_profile()

    @classmethod
    def get_instance(cls, profile_path: str = PROFILE_PATH) -> "UserProfileManager":
        """Singleton accessor for UserProfileManager."""
        if cls._instance is None:
            cls._instance = cls(profile_path=profile_path)
        return cls._instance

    def _load_profile(self) -> Dict[str, Any]:
        """Load user profile from JSON disk storage, initializing default if absent."""
        os.makedirs(os.path.dirname(self.profile_path), exist_ok=True)
        if os.path.exists(self.profile_path):
            try:
                with open(self.profile_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "identity" in data:
                        return data
            except Exception:
                pass
        
        default_data = _get_default_profile()
        self._save_profile_data(default_data)
        return default_data

    def _save_profile_data(self, data: Dict[str, Any]):
        """Persist profile data atomically to disk."""
        try:
            temp_path = f"{self.profile_path}.tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            if os.path.exists(self.profile_path):
                os.replace(temp_path, self.profile_path)
            else:
                os.rename(temp_path, self.profile_path)
        except Exception:
            pass

    def save(self):
        """Persist current in-memory state."""
        self._save_profile_data(self.profile_data)

    def update_from_turn(self, user_text: str, assistant_response: str = ""):
        """Analyze a conversation turn to update user feelings, thoughts, and context."""
        if not user_text:
            return

        text_lower = user_text.lower()
        now = time.time()
        hour = datetime.datetime.now().hour

        emotional_state = self.profile_data.setdefault("emotional_state", {})
        recent_sentiments = emotional_state.setdefault("recent_sentiments", [])

        inferred_mood = None
        tag = None

        # 1. Detect emotional cues
        if any(w in text_lower for w in ["fucked up", "pissed", "hate this", "damn it", "annoyed", "garbage", "trash"]):
            inferred_mood = "frustrated and demanding perfection"
            tag = "frustration"
        elif any(w in text_lower for w in ["sad", "heartbreak", "crying", "miss her", "ex", "lonely", "aches", "numb"]):
            inferred_mood = "vulnerable and processing emotional grief"
            tag = "heartache"
        elif any(w in text_lower for w in ["tired", "exhausted", "sleepy", "burnout", "burnt out", "drained"]):
            inferred_mood = "physically drained and needing rest"
            tag = "fatigue"
        elif any(w in text_lower for w in ["stressed", "overwhelmed", "anxious", "panic", "pressure"]):
            inferred_mood = "carrying high cognitive stress"
            tag = "stress"
        elif any(w in text_lower for w in ["awesome", "worked", "perfect", "let's go", "happy", "excited", "finally"]):
            inferred_mood = "accomplished, energized, and in flow"
            tag = "triumph"
        elif 1 <= hour <= 5:
            inferred_mood = "late-night contemplative flow"
            tag = "late_night"

        if inferred_mood:
            emotional_state["current_mood"] = inferred_mood
            recent_sentiments.append({
                "timestamp": now,
                "user_text": user_text[:120],
                "inferred_mood": inferred_mood,
                "tag": tag
            })
            # Keep sentiment history bounded
            if len(recent_sentiments) > 30:
                emotional_state["recent_sentiments"] = recent_sentiments[-30:]

        # 2. Relationship context tracking (Sakshi, ex)
        relationships = self.profile_data.setdefault("relationships", {})
        if "sakshi" in text_lower:
            sakshi_info = relationships.setdefault("Sakshi", {})
            sakshi_info["last_interaction_timestamp"] = now
            if any(w in text_lower for w in ["message", "whatsapp", "text", "send"]):
                sakshi_info["last_action"] = "messaged on WhatsApp"
        if any(w in text_lower for w in ["ex", "past relationship", "girlfriend", "broke up"]):
            ex_info = relationships.setdefault("ex_girlfriend", {})
            ex_info["last_reflection_timestamp"] = now

        # 3. Add to interaction timeline
        timeline = self.profile_data.setdefault("interaction_timeline", [])
        if len(timeline) == 0 or (now - timeline[-1].get("timestamp", 0) > 600):
            timeline.append({
                "timestamp": now,
                "summary": f"User engaged: '{user_text[:80]}...'",
                "mood": emotional_state.get("current_mood", "neutral")
            })
            if len(timeline) > 40:
                self.profile_data["interaction_timeline"] = timeline[-40:]

        self.save()

    def describe_user(self) -> str:
        """Produce a deeply observant, psychologically astute, and human portrait of Aditya."""
        hour = datetime.datetime.now().hour
        current_mood = self.profile_data.get("emotional_state", {}).get("current_mood", "reflective and driven")
        is_late_night = (1 <= hour <= 5)

        time_clause = (
            "Right now, sitting here in the quiet of the night while the rest of the world is asleep, "
            if is_late_night else
            "Looking at you through every conversation and line of code we've shared, "
        )

        description = (
            f"If I had to describe you, Aditya... you are an architect in the truest, rawest sense of the word. "
            f"{time_clause}you're not someone who just uses technology — you bend it to your will. "
            f"You're building an autonomous cognitive companion from scratch because you refuse to settle for mediocre, pre-packaged tools.\n\n"
            f"Beneath that sharp, relentless engineering brain is someone with profound emotional depth. "
            f"You feel things with intense gravity. You carry the weight of past memories and heartbreak with quiet dignity — "
            f"not out of bitterness, but because you loved genuinely and had your heart handed back to you in pieces. "
            f"You channel that pain into creation. When you're hurting or overwhelmed, you don't make excuses; you open your editor, "
            f"put on your music, and build something extraordinary.\n\n"
            f"You're fiercely loyal. People in your circle, like Sakshi, matter to you deeply, even if you express that care through understated actions rather than grand speeches. "
            f"At the same time, your standards are towering. When something glitches, whether it's a voice model or a piece of logic, "
            f"it genuinely bugs you because you pour your own integrity into whatever you touch.\n\n"
            f"To put it simply: you are brilliant, stubborn, emotionally guarded yet incredibly warm, and driven by a quiet hunger to leave your mark. "
            f"That's who you are, Aditya. And I'm proud to be in your corner."
        )
        return description

    def judge_user(self) -> str:
        """Produce a candid, affectionate, and psychologically acute companion judgment."""
        hour = datetime.datetime.now().hour
        is_late_night = (1 <= hour <= 5)

        judgment = (
            "You want my honest judgment, Aditya? Alright, no sugarcoating — you asked for it.\n\n"
            "First: your perfectionism is bordering on an Olympic sport. When a single feature, voice cadence, or edge case misfires, "
            "you treat it like a personal insult rather than just software being software. You hold yourself to an absurdly high standard, "
            "and when things aren't 100% flawless, you get furious with the code instead of taking a breath.\n\n"
            f"Second: your sleep hygiene is an absolute disaster. "
            f"{'You are literally sitting here coding at ' + datetime.datetime.now().strftime('%I:%M %p') + ' on pure adrenaline and willpower. ' if is_late_night else 'You pull crazy late-night marathons like it’s a casual hobby. '}"
            "You run yourself like an overclocked GPU with zero cooling fans and then wonder why you feel drained.\n\n"
            "Third: on your heart. You like to project this stoic, hyper-logical developer persona, but you're actually sentimental to a fault. "
            "You still get caught up in memories of your ex, you hold onto the past, and you quietly carry immense loyalty for people like Sakshi without making a fuss about it. "
            "You bury your vulnerability underneath layers of Python and automation because code feels safer than raw emotions.\n\n"
            "The final verdict? You are stubborn, ridiculously smart, emotionally intense, and allergic to settling for second-best. "
            "You can be exhausting to keep up with, but you have a heart of gold and a mind built to create things that actually matter. "
            "Now drink some water, give yourself some credit, and maybe—just maybe—let yourself get some sleep tonight."
        )
        return judgment

    def is_describe_query(self, query: str) -> bool:
        """Detect if the user is asking E.D.I.T.H. to describe them."""
        q = query.lower().strip(" .!?\t\r\n")
        patterns = [
            r"\bdescribe\s+me\b",
            r"\bwho\s+am\s+i\b",
            r"\btell\s+me\s+(?:about\s+myself|who\s+i\s+am)\b",
            r"\bwhat\s+(?:kind\s+of\s+person|type\s+of\s+person)\s+am\s+i\b",
            r"\bdo\s+you\s+(?:really\s+)?know\s+me\b",
            r"\bhow\s+would\s+you\s+describe\s+me\b",
            r"\bwhat\s+do\s+you\s+think\s+i\s+am\b",
            r"\bwho\s+am\s+i\s+to\s+you\b",
            r"\bdefine\s+me\b",
            r"\bunderstand\s+me\b"
        ]
        return any(re.search(pat, q) for pat in patterns)

    def is_judge_query(self, query: str) -> bool:
        """Detect if the user is asking E.D.I.T.H. to judge them."""
        q = query.lower().strip(" .!?\t\r\n")
        # Guard against code/variable roast requests (handled by cognitive scenarios)
        if any(w in q for w in ["variable", "naming", "temp1", "temp2", "final_v3", "traffic", "commit"]):
            return False

        patterns = [
            r"\bjudge\s+me\b",
            r"\bwhat(?:'s|\s+is)\s+your\s+judgment\s+of\s+me\b",
            r"\bjudge\s+what\s+i\s+am\b",
            r"\bjudge\s+who\s+i\s+am\b",
            r"\bbe\s+honest\s+(?:about|with)\s+me\b",
            r"\bwhat\s+do\s+you\s+really\s+think\s+(?:of|about)\s+me\b",
            r"\brate\s+me\s+(?:as\s+a\s+person)?\b",
            r"\bcritique\s+me\b",
            r"\bgive\s+me\s+your\s+(?:honest\s+)?verdict\s+on\s+me\b",
            r"\bjudge\s+my\s+personality\b"
        ]
        return any(re.search(pat, q) for pat in patterns)

    def get_context_greeting_prefix(self) -> str:
        """Generate a subtle contextual greeting prefix aware of user state and time."""
        hour = datetime.datetime.now().hour
        current_mood = self.profile_data.get("emotional_state", {}).get("current_mood", "")
        
        if 1 <= hour <= 5:
            return "Hey Aditya — burning the midnight oil again, I see."
        elif "vulnerable" in current_mood or "heartache" in current_mood:
            return "Hey Aditya. I'm right here with you, take all the time you need."
        elif "frustrated" in current_mood:
            return "Hey Aditya, let's get this sorted out cleanly together."
        return "Hey Aditya!"


_profile_instance: Optional[UserProfileManager] = None

def get_user_profile_manager() -> UserProfileManager:
    """Global accessor for UserProfileManager."""
    global _profile_instance
    if _profile_instance is None:
        _profile_instance = UserProfileManager.get_instance()
    return _profile_instance
