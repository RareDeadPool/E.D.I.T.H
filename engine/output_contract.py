"""
E.D.I.T.H. V14 Output Contract: Single-Spoken-Interface, Leak Linter, TTS Sanitizer & Result Verbalizer
Strictly prevents internal state, intent names, action policies, or DB record dumps from reaching the user.
"""

import re
import random
from typing import Dict, Any, List, Optional, Tuple


class LeakLinter:
    """
    Linters user-visible dialogue against internal terminology leaks.
    Rejects and flags anything containing classifier labels, execution internals,
    database notation, or telemetry format strings.
    """

    FORBIDDEN_PATTERNS = [
        # Classifier / intent / action label names
        r"exec_[a-zA-Z0-9_]+\b",
        r"\bsystem_utility\b",
        r"\bnetwork_task\b",
        r"\brespond_dialogue\b",
        r"\bproject_manage\b",
        r"\bcontext_control\b",
        r"\bautomation_workflow\b",
        r"\bfile_operation\b",
        r"\bapp_launch\b",
        r"\bmedia_control\b",
        r"\bcompute_math\b",
        r"\bgui_action\b",
        r"\bscreen_perception\b",
        r"\bunsupported_ood\b",
        r"\brequest_clarification\b",
        r"\bdraft_content\b",
        # Internal architectural keywords
        r"\bhandler\b",
        r"\bclassified\b",
        r"\bpredicted\b",
        r"\bANN\b",
        r"\bsemantic memory\b",
        r"\bnot yet bound\b",
        r"\bContextual control\b",
        r"\boperational confidence\b",
        r"\bcalibrated confidence\b",
        r"\bneural status\b",
        r"\bAction Policy\b",
        r"\bExtracted Slots\b",
        # Raw DB records and JSON/KV formats
        r"\bis relationship \(",
        r"\{\s*[\"'][a-zA-Z0-9_]+[\"']\s*:\s*",
        r"\b[a-zA-Z0-9_]+:\s*\[.*?\]",
        r"\[EXEC-OK\]",
        r"\[EXEC-INFO\]",
        r"\[EXEC-ERR\]",
        r"\[SUGGESTIONS\]",
        # Incomplete contextual strings
        r"Contextual control resolved:\s*\.?",
    ]

    @classmethod
    def lint(cls, text: str) -> Tuple[bool, List[str]]:
        """
        Inspects text for leaks.
        Returns:
            (is_clean, list_of_violations)
        """
        if not text:
            return True, []

        violations = []
        for pattern in cls.FORBIDDEN_PATTERNS:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            if matches:
                violations.extend(matches)

        is_clean = len(violations) == 0
        return is_clean, list(set(violations))

    @classmethod
    def sanitize_or_reject(cls, text: str, fallback: Optional[str] = None) -> str:
        """
        Lints text. If violations are present, sanitizes or replaces with fallback.
        """
        is_clean, violations = cls.lint(text)
        if is_clean:
            return text

        if fallback:
            return fallback

        # Clean stripped fallback
        clean_text = text
        for pattern in cls.FORBIDDEN_PATTERNS:
            clean_text = re.sub(pattern, "", clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()

        if len(clean_text) < 10:
            return "I'm right here with you. What can I help you with?"
        return clean_text


class TTSSanitizer:
    """
    Sanitizes spoken dialogue for Natural Text-To-Speech execution:
    - Removes emojis
    - Removes markdown formatting (*, **, #, `, ~)
    - Removes parentheses, ASCII arrows, brackets, bullets
    - Normalizes contractions and spoken numbers
    """

    # Regex for unicode emoji sequences
    EMOJI_PATTERN = re.compile(
        r"[\U00010000-\U0010ffff]|[\u2600-\u27bf]|[\u2300-\u23ff]|[\u2b50-\u2b55]|[\u200d\ufe0f]",
        flags=re.UNICODE
    )

    @classmethod
    def sanitize(cls, text: str) -> str:
        if not text:
            return ""

        # 1. Remove emojis
        s = cls.EMOJI_PATTERN.sub("", text)

        # 2. Remove markdown formatting
        s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)  # Bold
        s = re.sub(r"\*([^*]+)\*", r"\1", s)      # Italics
        s = re.sub(r"__([^_]+)__", r"\1", s)
        s = re.sub(r"_([^_]+)_", r"\1", s)
        s = re.sub(r"`([^`]+)`", r"\1", s)        # Code
        s = re.sub(r"#{1,6}\s*", "", s)           # Headers
        s = re.sub(r"~~([^~]+)~~", r"\1", s)      # Strikethrough

        # 3. Remove bullets, dashes at start of lines, ASCII arrows
        s = re.sub(r"^\s*[-*•>]\s*", "", s, flags=re.MULTILINE)
        s = re.sub(r"\s*[-=]>\s*", " to ", s)
        s = re.sub(r"\s*<[-=]\s*", " from ", s)

        # 4. Remove parentheses and square brackets while preserving inner text
        s = re.sub(r"\(([^)]*)\)", r" \1 ", s)
        s = re.sub(r"\[([^\]]*)\]", r" \1 ", s)

        # 5. Remove special symbols that cause TTS glitches
        s = re.sub(r"[#%^*~|<>{}\\_]", " ", s)

        # 6. Normalize whitespace
        s = re.sub(r"\s+", " ", s).strip()

        # 7. Ensure clean sentence termination
        if s and s[-1] not in ".!?":
            s += "."

        return s


class ResultVerbalizer:
    """
    Translates structured facts (tool outcomes, memory records, capability limits, errors)
    into fluid, human-like speech. Guarantees zero raw templates or DB schemas reach the user.
    """

    @classmethod
    def verbalize_memory(cls, facts: List[Dict[str, str]], user_name: str = "Aditya", streak: int = 3, query_type: str = "general") -> str:
        """
        Translates raw memory facts into a conversational reflection of what E.D.I.T.H. knows.
        """
        friends = []
        preferences = []
        projects = []

        for f in facts:
            subj = f.get("subject", "")
            pred = f.get("predicate", "")
            obj = f.get("object", "")

            if pred in ["relationship", "friend"] or obj in ["friend", "relationship"]:
                name = subj if obj in ["friend", "relationship"] else obj
                if name.lower() not in ["user", "aditya", "friend"]:
                    friends.append(name.capitalize())
            elif "like" in pred or "prefer" in pred or "preference" in pred:
                preferences.append(f"{pred} {obj}")
            elif "project" in pred:
                projects.append(obj)

        parts = []
        name_str = user_name if user_name else "there"
        parts.append(f"You're {name_str}")

        if streak > 1:
            parts.append(f"you've been working on this for {streak} days straight")

        if friends:
            friend_str = ", ".join(friends)
            parts.append(f"you've got a friend named {friend_str}")

        if preferences:
            parts.append(f"you enjoy {', '.join(preferences[:2])}")

        if len(parts) <= 1:
            return (
                f"I know your name is {name_str}, but honestly, my working memory on you is pretty thin right now. "
                "Fill me in on what you're working on today."
            )

        core_knowledge = ", ".join(parts)
        closing = (
            "Beyond that, I'm working from scraps so far. Fill me in whenever you're ready."
            if len(friends) + len(preferences) < 3 else
            "That's my primary picture of you so far."
        )

        return f"{core_knowledge}. {closing}"

    @classmethod
    def verbalize_capability_gap(cls, action_intent: str, query: str = "", repeat_count: int = 0) -> str:
        """
        Honest, in-character acknowledgement of an unimplemented capability or missing handler.
        """
        q_lower = query.lower()

        if "subscribe" in q_lower or "subscription" in q_lower:
            if repeat_count > 0:
                return (
                    "Still stuck on that one. Give me the channel or newsletter name and I'll pull up the page for you."
                )
            return (
                "Subscribe you to what, a YouTube channel, a newsletter, or something else? "
                "Let me know what you want to follow."
            )

        if repeat_count > 0:
            return (
                "I hear you, but I don't have the tools to handle that directly yet. "
                "Tell me the specific app or page you need and I'll open it."
            )

        return (
            "I don't have a direct handler set up for that task yet. "
            "If you want, I can open the relevant site or application for you."
        )

    @classmethod
    def verbalize_ambiguity_or_mishear(cls, query: str, stt_confidence: float = 1.0) -> str:
        """
        Human response to an ambiguous or likely misheard utterance.
        """
        q = query.strip()
        q_low = q.lower()
        if len(q) < 3 or q_low in ["i'll allow you", "allow you", "i allow you"]:
            return f"Sorry, I think I caught '{q}', did you mean something else?"
        if "crow" in q_low:
            return "Caught 'open crow'. Did you mean launch Google Chrome?"
        if "spot a fly" in q_low:
            return "Sounded like 'spot a fly'. Did you mean launch Spotify?"

        return f"I caught '{q}', but I'm not completely sure what you'd like me to do with that. Could you clarify?"

    @classmethod
    def verbalize_action_execution(cls, handler: str, params: Dict[str, Any]) -> str:
        """
        Short, conversational confirmation of an executed action.
        """
        app = params.get("app_name") or params.get("app") or ""
        if app:
            return f"Opening {app}."
        if handler == "play_music":
            song = params.get("song") or "music"
            return f"Playing {song}."
        if handler == "compute_math":
            return "Calculated."
        return "Done."


def speak(
    text: str,
    tts_engine: Optional[Any] = None,
    voice_mode: bool = False,
    debug_mode: bool = False,
    strict_raise: bool = False
) -> str:
    """
    The NON-NEGOTIABLE single bottleneck for every string the user sees or hears.
    Lints against leaks, sanitizes for voice, and directs to UI/TTS.

    Args:
        text: The candidate string to speak.
        tts_engine: Optional TTS engine instance.
        voice_mode: Whether voice mode is active.
        debug_mode: If True, logs lint violations to debug output.
        strict_raise: If True, raises ValueError if leak linter fails (used in test suites).

    Returns:
        The sanitized, user-ready spoken string.
    """
    is_clean, violations = LeakLinter.lint(text)

    if not is_clean:
        err_msg = f"[LeakLinter Error] User-visible text contained internal leaks: {violations}. Text: '{text}'"
        if strict_raise:
            raise ValueError(err_msg)
        if debug_mode:
            print(f"[DEBUG LINT] {err_msg}")
        text = LeakLinter.sanitize_or_reject(text, fallback="I'm right here. What can I help you with?")

    if voice_mode:
        text = TTSSanitizer.sanitize(text)

    if voice_mode and tts_engine and hasattr(tts_engine, "speak"):
        try:
            tts_engine.speak(text, async_mode=True)
        except Exception as e:
            if debug_mode:
                print(f"[DEBUG TTS Error] {e}")

    return text
