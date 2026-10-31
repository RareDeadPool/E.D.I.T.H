"""Conversation State & Context Tracking Subsystem for E.D.I.T.H. V12.

Tracks short-term multi-turn conversational context, topics, goals, people mentioned,
emotional shifts, and explicit topic transitions. Distinguishes temporary conversational
context from persistent memory and active task state.
"""

import time
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class PersonMention:
    """Tracks a person mentioned in the conversation."""
    name: str
    relationship: str = "friend"  # friend, colleague, family, manager, other
    last_mentioned: float = field(default_factory=time.time)
    mention_count: int = 1
    context_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DialogueTurn:
    """A single conversational exchange."""
    user_query: str
    assistant_response: str
    intent: str
    action: str
    topic: Optional[str] = None
    emotion: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConversationStateManager:
    """Manages short-term conversation state, emotional context, and topic transitions."""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.current_topic: Optional[str] = None
        self.current_discussion_goal: Optional[str] = None  # empathy_support, humour, exploration, task_execution, casual
        self.turns: List[DialogueTurn] = []
        self.people_mentioned: Dict[str, PersonMention] = {}
        self.unresolved_questions: List[str] = []
        self.last_assistant_question: Optional[str] = None
        self.active_task_ref: Optional[str] = None
        self.topic_start_time: float = time.time()

    def detect_topic_change(self, text: str) -> bool:
        """Detect whether user is explicitly shifting or abandoning the current topic."""
        t_low = text.lower().strip()
        change_patterns = [
            r"\b(?:actually\s*,?\s*)?(?:let\'?s\s+talk\s+about\s+something\s+else|change\s+the\s+topic|switch\s+topics?)\b",
            r"\b(?:forget\s+(?:about\s+)?that|never\s*mind\s+(?:about\s+)?that|drop\s+it|leave\s+it)\b",
            r"\b(?:different\s+question|another\s+thing|moving\s+on|on\s+another\s+note)\b",
            r"\b(?:actually\s*,?\s*)?(?:tell\s+me\s+a\s+joke|can\s+you\s+tell\s+a\s+joke)\b"
        ]
        return any(bool(re.search(p, t_low)) for p in change_patterns)

    def extract_people(self, text: str) -> List[PersonMention]:
        """Extract social entities and people mentioned in conversational context."""
        t_low = text.lower().strip()
        found: List[PersonMention] = []

        # Pattern: friend named X / friend called X
        m_friend = re.findall(r"\b(?:friend|buddy|pal|colleague|coworker|brother|sister|roommate)\s+(?:named|called|is)?\s+([A-Z][a-z]+|[a-z]{3,})\b", text, flags=re.IGNORECASE)
        for name in m_friend:
            name_clean = name.strip().capitalize()
            # Ignore common non-names
            if name_clean.lower() in ["a", "an", "the", "my", "looking", "very", "sad", "trying", "here", "there", "what", "how"]:
                continue
            rel = "friend"
            if "colleague" in t_low or "coworker" in t_low:
                rel = "colleague"
            elif "brother" in t_low or "sister" in t_low or "mom" in t_low or "dad" in t_low:
                rel = "family"

            if name_clean in self.people_mentioned:
                self.people_mentioned[name_clean].last_mentioned = time.time()
                self.people_mentioned[name_clean].mention_count += 1
            else:
                self.people_mentioned[name_clean] = PersonMention(name=name_clean, relationship=rel)
            found.append(self.people_mentioned[name_clean])

        return found

    def infer_topic_and_goal(self, text: str, intent: str, action: str) -> Tuple[str, str]:
        """Infer current topic and user discussion goal from utterance."""
        t_low = text.lower()

        # 1. Job search / Career struggles
        if any(w in t_low for w in ["job", "looking for jobs", "interviews", "unemployed", "career", "applying"]):
            return "job_search_struggles", "empathy_support"

        # 2. Emotional sadness / distress
        if any(w in t_low for w in ["sad", "depressed", "heartbroken", "down", "upset", "crying", "unhappy"]):
            return "emotional_wellbeing", "empathy_support"

        # 3. Humour / Comedy
        if any(w in t_low for w in ["joke", "funny", "laugh", "humour", "humor", "comedy"]):
            return "humour_entertainment", "humour"

        # 4. Social Introductions / Friends
        if any(w in t_low for w in ["friend named", "friend called", "have a friend", "my friend"]):
            return "friend_introduction", "social_sharing"

        # 5. Technology / OS task
        if intent in ["app_launch", "app_close", "file_operation", "system_control", "process_manage"]:
            return "os_task_execution", "task_execution"

        return "general_conversation", "casual"

    def record_turn(
        self,
        user_query: str,
        assistant_response: str,
        intent: str,
        action: str
    ):
        """Update conversation state with new turn."""
        # 1. Check for explicit topic change
        if self.detect_topic_change(user_query):
            self.current_topic = None
            self.current_discussion_goal = None
            self.topic_start_time = time.time()

        # 2. Extract people mentioned
        self.extract_people(user_query)

        # 3. Infer topic and discussion goal if not already set or changed
        topic, goal = self.infer_topic_and_goal(user_query, intent, action)
        if self.current_topic is None or topic != "general_conversation":
            self.current_topic = topic
            self.current_discussion_goal = goal

        # 4. Check if assistant asked a question in response
        if "?" in assistant_response:
            sentences = re.split(r"(?<=[.!?])\s+", assistant_response)
            questions = [s.strip() for s in sentences if s.strip().endswith("?")]
            if questions:
                self.last_assistant_question = questions[-1]
                self.unresolved_questions = questions[-2:]

        # 5. Append turn
        turn = DialogueTurn(
            user_query=user_query,
            assistant_response=assistant_response,
            intent=intent,
            action=action,
            topic=self.current_topic,
            emotion=goal
        )
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def get_recent_turns_for_context(self, n: int = 4) -> List[Dict[str, str]]:
        """Return formatted recent dialogue turns for LLM chat context."""
        recent = self.turns[-n:] if self.turns else []
        return [{"user": t.user_query, "bot": t.assistant_response} for t in recent]

    def to_context_dict(self) -> Dict[str, Any]:
        """Convert state into dictionary for integration with StructuredTaskContext."""
        return {
            "current_topic": self.current_topic,
            "discussion_goal": self.current_discussion_goal,
            "people_mentioned": {k: v.to_dict() for k, v in self.people_mentioned.items()},
            "unresolved_questions": list(self.unresolved_questions),
            "last_assistant_question": self.last_assistant_question,
            "recent_dialogue_turns": self.get_recent_turns_for_context()
        }
