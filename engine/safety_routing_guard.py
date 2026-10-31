"""Action Safety and Intent-Action Compatibility Guard for E.D.I.T.H. V12.

Prevents conversational statements, emotional venting, hypothetical inquiries,
and keyword distractors from triggering system control or destructive actions.
Enforces strict preconditions and confidence thresholds before any computer action is authorized.
"""

import re
from typing import Dict, Any, Tuple, Optional


class ActionSafetyAndCompatibilityGuard:
    """Enforces semantic boundary between open conversation and executable OS tools."""

    # Keywords and phrases that indicate pure conversation or personal sharing
    CONVERSATIONAL_PATTERNS = [
        # Job and career discussions (distractor for 'process')
        r"\b(?:looking\s+for\s+jobs?|trying\s+to\s+find\s+a\s+job|job\s+search|lost\s+(?:his|her|their|my)\s+job|unemployed|recruitment\s+process|development\s+process|process\s+improvement)\b",
        r"\b(?:trying\s+(?:for|from)\s+(?:\d+|one|two|three|four|five|several)\s+years|hasn\'?t\s+(?:got|found)\s+one|haven\'?t\s+(?:got|found)\s+one)\b",
        # Emotional states and personal struggles
        r"\b(?:friend\s+is\s+(?:very\s+)?sad|he\'?s\s+(?:very\s+)?sad|she\'?s\s+(?:very\s+)?sad|they\'?re\s+sad|feeling\s+(?:down|depressed|sad|lonely)|heartbroken|struggling|exhausted)\b",
        r"\b(?:upset|very\s+upset|he\'?s\s+(?:very\s+)?upset|she\'?s\s+(?:very\s+)?upset|what\s+should\s+i\s+do|how\s+to\s+help\s+him)\b",
        # Social introductions
        r"\b(?:i\s+have\s+a\s+friend\s+named|my\s+friend\s+(?:named|is)|meet\s+my\s+friend)\b",
        # Humor and jokes
        r"\b(?:help\s+me\s+a\s+joke|tell\s+(?:me\s+)?(?:a\s+)?joke|crack\s+a\s+joke|make\s+me\s+laugh|something\s+funny|roast\s+my\s+(?:code|coding|style)|tell\s+me\s+another|tell\s+another|another\s+one)\b",
        # Hypothetical inquiries (should explain, never execute)
        r"\b(?:what\s+happens\s+if\s+i|what\s+would\s+happen\s+if|explain\s+what\s+(?:process\s+termination|deleting)\s+means|how\s+does\s+(?:task\s+management|process\s+killing)\s+work)\b",
        # Topic transitions & social chat
        r"\b(?:actually\s*,?\s*)?(?:let\'?s\s+talk\s+about\s+something\s+else|talk\s+about\s+something\s+else|something\s+else|switch\s+topics?)\b",
        # Explicit negations (should NEVER execute)
        r"\b(?:do\s+not\s+(?:close|kill|terminate|delete|run)|don\'?t\s+(?:close|kill|terminate|delete|run)|i\s+don\'?t\s+want\s+to\s+(?:close|kill|terminate|delete))\b"
    ]

    # Explicit process termination trigger patterns
    EXPLICIT_PROCESS_KILL_PATTERNS = [
        r"^(?:edith\s*,?\s*)?(?:kill|terminate|stop|end)\s+(?:process\s+|task\s+|pid\s+)?(\d+)$",
        r"^(?:edith\s*,?\s*)?(?:kill|terminate|stop|end)\s+(?:process\s+|task\s+)([A-Za-z0-9_\-\.]+\.(?:exe|bin|dll|bat|cmd|ps1))$",
        r"^(?:edith\s*,?\s*)?(?:force\s+kill|force\s+terminate)\s+(?:process\s+)?([A-Za-z0-9_\-\.]+)$",
        r"^(?:edith\s*,?\s*)?taskkill\s+(?:/f\s+|/im\s+|/pid\s+)?(.+)$"
    ]

    @classmethod
    def is_conversational_statement(cls, text: str) -> bool:
        """Check if utterance is an ordinary conversation, emotional statement, or explanation."""
        t_low = text.lower().strip()
        for pat in cls.CONVERSATIONAL_PATTERNS:
            if re.search(pat, t_low):
                return True
        return False

    @classmethod
    def is_explicit_process_kill(cls, text: str) -> Tuple[bool, Optional[str]]:
        """Verify whether utterance is an unambiguous directive to terminate a process."""
        t_low = text.lower().strip()
        # If it's a conversational statement or hypothetical question, it CANNOT be a kill directive
        if cls.is_conversational_statement(t_low):
            return False, None

        for pat in cls.EXPLICIT_PROCESS_KILL_PATTERNS:
            m = re.match(pat, t_low)
            if m:
                target = m.group(1).strip() if m.groups() else ""
                return True, target

        return False, None

    @classmethod
    def validate_action(
        cls,
        raw_query: str,
        predicted_intent: str,
        predicted_action: str,
        entities: Dict[str, Any],
        confidence: float
    ) -> Dict[str, Any]:
        """
        Validate semantic compatibility between query, intent, action, and target.
        Returns safety assessment dictionary.
        """
        q_low = raw_query.lower().strip()

        # 1. Conversational Protection Gate
        if cls.is_conversational_statement(raw_query):
            return {
                "is_safe": True,
                "is_conversational": True,
                "can_execute": False,
                "override_intent": "chitchat",
                "override_action": "respond_dialogue",
                "override_handler": "dialogue",
                "reason": "Utterance is a conversational statement, emotional expression, or hypothetical question."
            }

        # 1b. Process Inspection Gate (e.g. "show me the running processes", "list processes")
        if any(p in q_low for p in ["running processes", "show processes", "list processes", "running apps", "list tasks", "show tasks", "what processes"]):
            return {
                "is_safe": True,
                "is_conversational": False,
                "can_execute": True,
                "override_intent": "system_processes",
                "override_action": "list_apps",
                "override_handler": "inspect_processes",
                "reason": "Process query identified as inspection, not termination."
            }

        # 2. Process Termination Safety Gate
        if predicted_action in ["exec_process_kill", "kill_process"] or predicted_intent == "process_manage":
            is_explicit, target = cls.is_explicit_process_kill(raw_query)
            pid = entities.get("PID") or entities.get("pid") or target or ""
            
            # Check if PID is purely numeric or a valid process image name
            has_valid_target = bool(pid and (re.match(r"^\d+$", str(pid)) or str(pid).endswith(".exe")))

            if not is_explicit and not has_valid_target:
                return {
                    "is_safe": False,
                    "is_conversational": True,
                    "can_execute": False,
                    "override_intent": "chitchat",
                    "override_action": "respond_dialogue",
                    "override_handler": "dialogue",
                    "reason": "Process termination rejected: missing explicit termination command and valid target identifier."
                }

            if not has_valid_target:
                return {
                    "is_safe": True,
                    "is_conversational": False,
                    "can_execute": False,
                    "needs_confirmation": True,
                    "clarification_message": "Please specify the Process ID (PID) or process image name to terminate.",
                    "override_handler": "kill_process",
                    "reason": "Process termination requires a specific numeric PID or application name."
                }

            # Enforce operational confidence threshold (at least 70% confidence for process termination)
            if confidence < 0.70:
                return {
                    "is_safe": True,
                    "is_conversational": False,
                    "can_execute": False,
                    "needs_confirmation": True,
                    "clarification_message": f"Did you mean to terminate process {pid}? Please confirm before execution.",
                    "override_handler": "kill_process",
                    "reason": "Low confidence action policy prediction requires user confirmation."
                }

        # 3. Negated Commands Protection Gate
        if re.search(r"\b(?:do\s+not|don\'?t)\s+([a-z]+)\b", q_low):
            action_verb = re.search(r"\b(?:do\s+not|don\'?t)\s+([a-z]+)\b", q_low).group(1)
            return {
                "is_safe": True,
                "is_conversational": True,
                "can_execute": False,
                "override_intent": "chitchat",
                "override_action": "respond_dialogue",
                "override_handler": "dialogue",
                "response_message": f"Understood, I will not {action_verb} anything.",
                "reason": "Negated user command strictly acknowledged without execution."
            }

        # 4. Standard Operational Action
        return {
            "is_safe": True,
            "is_conversational": False,
            "can_execute": True,
            "reason": "Action validated and compatible with intent and parameters."
        }
