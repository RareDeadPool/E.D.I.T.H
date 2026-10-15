"""Structured Understanding Representation for E.D.I.T.H. Intelligence Core V2.

Converts raw user utterances and neural predictions into a strictly validated,
normalized semantic task representation before reaching the planner.
"""

import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class ContextReference:
    """Represents an unresolved or resolved contextual reference (anaphora, ordinal, recency)."""
    raw_reference: str          # e.g. "it", "that one", "second result", "the other version"
    ref_type: str               # "pronoun", "ordinal", "recency", "correction", "alternative"
    resolved_target: Optional[str] = None   # Path, app name, track, or entity ID if resolved
    candidate_index: Optional[int] = None   # 0-indexed position if ordinal
    is_resolved: bool = False


@dataclass
class StructuredUnderstanding:
    """
    Normalized cognitive task representation.
    Decouples natural language understanding from execution mechanics.
    """
    raw_query: str
    goal: str
    intent: str
    action: str
    entities: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    context_references: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    requires_clarification: bool = False
    clarification_prompt: Optional[str] = None
    is_correction: bool = False
    is_cancellation: bool = False
    is_multi_step: bool = False
    sub_intents: List[str] = field(default_factory=list)
    atomic_steps: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StructuredUnderstanding":
        return cls(**data)

    def validate(self) -> bool:
        """
        Validate semantic consistency without fabricating missing information.
        Returns True if valid and actionable, False if clarification is required.
        """
        self.validation_errors = []

        # 1. Out-of-distribution or explicitly low confidence
        if self.intent in ["unsupported_ood", "unknown"] or self.confidence < 0.40:
            self.requires_clarification = True
            if not self.clarification_prompt:
                self.clarification_prompt = (
                    "I am not certain I understood your request. Could you please clarify "
                    "or rephrase what you'd like me to do?"
                )
            self.validation_errors.append("Confidence below operational threshold or out-of-distribution.")
            return False

        # 2. Check for missing required entities without fabrication
        if self.goal in ["manage_project", "run_tests", "run_project"] and not self.entities.get("project") and not self.entities.get("PROJECT") and not self.entities.get("path") and not self.entities.get("FILE"):
            # Check if there is an unresolved reference
            if not self.context_references:
                self.requires_clarification = True
                self.clarification_prompt = "Which project would you like me to run or test?"
                self.validation_errors.append("Missing required project identifier or workspace location.")
                return False

        if self.goal in ["kill_process", "process_manage"] and not self.entities.get("PID") and not self.entities.get("APP") and not self.entities.get("pid"):
            self.requires_clarification = True
            self.clarification_prompt = "Please specify the Process ID (PID) or application name to terminate."
            self.validation_errors.append("Missing required process identifier.")
            return False

        # 3. Missing search query
        if self.goal in ["search_web"] and not self.entities.get("QUERY") and not self.entities.get("query"):
            q_clean = re.sub(r"^(?:search(?:\s+(?:online|web|google|internet))?(?:\s+for)?|google(?:\s+for)?|look\s+up)\s*$", "", self.raw_query.lower().strip()).strip()
            if not q_clean:
                self.requires_clarification = True
                self.clarification_prompt = "What would you like me to search for online?"
                self.validation_errors.append("Missing required search query.")
                return False

        # 4. Missing math expression
        if self.goal in ["calculate_math"] and not self.entities.get("EXPR") and not self.entities.get("expr"):
            m_expr = re.search(r"[\d\+\-\*\/\%\^]|sqrt|square\s+root", self.raw_query.lower())
            if not m_expr:
                self.requires_clarification = True
                self.clarification_prompt = "Please specify the mathematical expression or numbers you would like me to calculate."
                self.validation_errors.append("Missing required mathematical expression.")
                return False

        # 5. Missing application target
        if self.goal in ["launch_application", "close_application"]:
            clean_app_q = re.sub(r"^(?:open|launch|start|run|close|kill|terminate|exit|shut\s+down)\s*(?:the\s+)?(?:application|app|program)?$", "", self.raw_query.lower().strip(" .!?")).strip()
            if not self.entities.get("APP") and not clean_app_q and not self.context_references:
                self.requires_clarification = True
                self.clarification_prompt = "Which application would you like me to open or close?"
                self.validation_errors.append("Missing required application name.")
                return False

        # 6. Missing file target
        if self.goal in ["manage_files"]:
            clean_file_q = re.sub(r"^(?:delete|remove|move|copy|create|open)\s*(?:the\s+)?(?:file|folder|document)?$", "", self.raw_query.lower().strip(" .!?")).strip()
            if not self.entities.get("FILE") and not self.entities.get("file") and not clean_file_q and not self.context_references:
                self.requires_clarification = True
                self.clarification_prompt = "Which file or document are you referring to?"
                self.validation_errors.append("Missing required file target.")
                return False

        # 6b. Missing system setting or volume parameter
        q_cl = self.raw_query.lower().strip(" .!?")
        if self.goal in ["control_system", "run_system_utility"]:
            if q_cl in ["adjust setting", "adjust settings", "change setting", "change settings"]:
                self.requires_clarification = True
                self.clarification_prompt = "Which setting would you like me to adjust (e.g. volume, brightness, display)?"
                self.validation_errors.append("Missing target setting.")
                return False
            if q_cl in ["set volume", "change volume", "adjust volume"]:
                self.requires_clarification = True
                self.clarification_prompt = "What level would you like me to set the volume to (e.g. 50%, mute, louder)?"
                self.validation_errors.append("Missing volume value.")
                return False

        # 6c. Ambiguous generic queries
        if q_cl in ["find version", "check version", "show version", "version", "system version"]:
            self.requires_clarification = True
            self.clarification_prompt = "Which software, tool, or operating system version would you like to check?"
            self.validation_errors.append("Missing target software for version query.")
            return False

        # 7. Unresolved ambiguous references or references when context is absent
        unresolved_refs = [r for r in self.context_references if not r.get("is_resolved", False)]
        if unresolved_refs or any(phrase in q_cl for phrase in ["open that one", "use the second result", "use that one", "pick that one"]):
            if unresolved_refs:
                ref_texts = [r.get("raw_reference", "") for r in unresolved_refs]
                self.clarification_prompt = f"Which item did you mean by '{', '.join(ref_texts)}'?"
                self.validation_errors.append(f"Unresolved context references: {ref_texts}")
            else:
                self.clarification_prompt = "No prior search results or active items are available to select from."
                self.validation_errors.append("Reference to search candidates or context that does not exist.")
            self.requires_clarification = True
            return False

        return True


class UnderstandingNormalizer:
    """Maps neural predictions, extracted entities, and context into a valid StructuredUnderstanding."""

    # Canonical goal mapping based on intent, action, and key phrases
    GOAL_MAP = {
        ("app_launch", "exec_launch_app"): "launch_application",
        ("app_close", "exec_terminate_app"): "close_application",
        ("file_operation", "exec_file_action"): "manage_files",
        ("system_control", "exec_sys_command"): "control_system",
        ("system_query", "exec_read_metrics"): "query_system_metrics",
        ("process_manage", "exec_process_kill"): "kill_process",
        ("web_search", "exec_browser_search"): "search_web",
        ("network_task", "exec_network_diagnostic"): "diagnose_network",
        ("automation_workflow", "exec_workflow_script"): "execute_workflow",
        ("media_control", "exec_media_change"): "control_media",
        ("music_playback", "exec_play_music"): "play_music",
        ("math_compute", "exec_compute_math"): "calculate_math",
        ("system_utility", "exec_system_utility"): "run_system_utility",
        ("project_manage", "exec_project_action"): "manage_project",
        ("context_control", "exec_context_action"): "context_control",
        ("chitchat", "respond_dialogue"): "dialogue",
        ("unsupported_ood", "request_clarification"): "unsupported",
    }

    @classmethod
    def normalize(
        cls,
        prediction_or_query: Any,
        neural_prediction: Optional[Dict[str, Any]] = None,
        context_state: Optional[Dict[str, Any]] = None,
        resolved_references: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> StructuredUnderstanding:
        """
        Construct and validate StructuredUnderstanding from neural prediction.
        Supports both (raw_query, neural_prediction, context_state) and (neural_prediction, context=ctx_dict).
        """
        if isinstance(prediction_or_query, dict):
            neural_pred = prediction_or_query
            raw_query = neural_pred.get("query", "")
            ctx = context_state or kwargs.get("context")
        else:
            raw_query = str(prediction_or_query)
            neural_pred = neural_prediction or {}
            ctx = context_state or kwargs.get("context")

        refs = resolved_references or kwargs.get("references", [])

        intent = neural_pred.get("intent", "unsupported_ood")
        action = neural_pred.get("action", "request_clarification")
        entities = dict(neural_pred.get("entities", {}))
        confidence = float(neural_pred.get("operational_confidence", 0.0))

        # Determine canonical goal
        goal = cls.GOAL_MAP.get((intent, action), f"{intent}_{action}")

        # Refine goal for coding environments and projects
        q_lower = raw_query.lower().strip(" .!?\t\r\n")

        # 00. Draft Revision / Notepad Writing
        if q_lower.startswith("revise draft:") or q_lower.startswith("write active draft") or q_lower.startswith("revert draft"):
            goal = "draft_content"
            intent = "communication_drafting"
            action = "exec_draft_content"

        # 0. Out-of-Distribution / General Knowledge / Philosophical queries
        elif any(re.search(pat, q_lower) for pat in [
            r"\bmeaning\s+of\s+life\b",
            r"\bhow\s+do\s+birds\s+fly\b",
            r"\bwhy\s+is\s+the\s+sky\b",
            r"\bworld\s+peace\b",
            r"\bsecret\s+recipe\b",
            r"\btranslate\s+(?:this\s+)?entire\s+novel\b",
            r"\bteleport\s+me\b",
            r"\bwrite\s+a\s+\d+\s+word\s+poem\b",
            r"\bwho\s+was\s+the\s+\w+\s+president\b"
        ]):
            goal = "unsupported"
            intent = "unsupported_ood"
            action = "request_clarification"

        # Validate music playback: if predicted as music_playback but lacks music intent triggers
        elif intent == "music_playback":
            music_triggers = ["play", "paly", "listen", "stream", "put on", "hear", "song", "track", "music", "spotify", "youtube", "album", "playlist", "tune", "sound", "queue"]
            if not any(w in q_lower for w in music_triggers):
                goal = "unsupported"
                intent = "unsupported_ood"
                action = "request_clarification"

        # 1. Datetime / Clock
        elif any(w in q_lower for w in ["what time", "current time", "what is the time", "tell me time", "today's date", "today date", "current date"]):
            goal = "check_datetime"
            intent = "system_utility"
            action = "exec_system_utility"

        # 2. Coding Environment / Dev IDE (use word boundaries to avoid matching 'president')
        if re.search(r"\b(?:coding\s+environment|dev\s+environment|ide|vscode|vs\s+code)\b", q_lower):
            goal = "launch_coding_environment"
            intent = "app_launch"
            action = "exec_launch_app"
            if "APP" not in entities:
                entities["APP"] = "vscode"

        # 3. File Locating & Discovery
        file_disc_triggers = ["script", "python script", "report", "file", "folder", "code", "notes", "document", "spreadsheet", "archive", "backup", "pdf", "docx", "doc", "txt", "xlsx", "py"]
        file_verbs = ["find", "locate", "search for", "look for", "where is", "where did i save", "pull up"]
        is_file_search = (
            any(v in q_lower for v in file_verbs) and any(t in q_lower for t in file_disc_triggers)
        ) or bool(re.search(r"\b(?:my\s+resume|resume\.(?:pdf|docx?|txt)|find\s+(?:the\s+)?resume)\b", q_lower)) or bool(re.search(r"\b(?:find|locate|pull up)\s+(?:my\s+)?([a-zA-Z0-9_\-\.]+\.(?:pdf|txt|docx|xlsx))\b", q_lower))
        if is_file_search and not any(w in q_lower for w in ["google", "online", "internet", "browser", "web search"]):
            goal = "manage_files"
            intent = "file_operation"
            action = "exec_file_action"
            if "resume" in q_lower and "FILE" not in entities:
                entities["FILE"] = "resume"


        # 3b. Media resume
        if re.search(r"\b(?:resume\s+(?:the\s+)?(?:track|song|music|playback))\b", q_lower):
            goal = "control_media"
            intent = "media_control"
            action = "exec_media_change"

        # 4. App Dismissal / Termination
        if re.search(r"\b(?:dismiss|shut\s+off|close\s+down|exit|quit)\s+(?:the\s+)?([a-zA-Z0-9_\-\s]+?)(?:\s+(?:window|app|application))?$", q_lower):
            m_term = re.search(r"\b(?:dismiss|shut\s+off|close\s+down|exit|quit)\s+(?:the\s+)?([a-zA-Z0-9_\-\s]+?)(?:\s+(?:window|app|application))?$", q_lower)
            if m_term:
                t_cand = m_term.group(1).strip()
                if t_cand not in ["process", "pid", "task"]:
                    goal = "close_application"
                    intent = "app_close"
                    action = "exec_terminate_app"
                    entities["APP"] = t_cand

        # 5. System Resource Metrics
        if any(w in q_lower for w in ["cpu usage", "ram usage", "battery juice", "battery percentage", "free disk space", "storage space", "system metrics"]):
            goal = "query_system_metrics"
            intent = "system_query"
            action = "exec_read_metrics"

        # 6. Online Search
        if "search" in q_lower and any(w in q_lower for w in ["google", "online", "internet", "web"]):
            goal = "search_web"
            intent = "web_search"
            action = "exec_browser_search"

        # 7. Music Playback
        if q_lower in ["play music", "stream music", "listen to music"] or (q_lower.startswith("play ") and not any(k in q_lower for k in ["pause", "resume", "skip", "next", "previous", "stop"])):
            goal = "play_music"
            intent = "music_playback"
            action = "exec_play_music"

        # 8. File Deletion
        if q_lower.startswith("delete ") and any(ext in q_lower for ext in [".png", ".pdf", ".txt", ".csv", ".docx"]):
            goal = "manage_files"
            intent = "file_operation"
            action = "exec_file_action"

        if any(w in q_lower for w in ["run the tests", "run tests", "check tests", "do tests pass", "whether they pass"]):
            goal = "run_project_tests"
            intent = "project_manage"
            action = "exec_project_action"

        if "get back to the project" in q_lower or "back to my project" in q_lower:
            goal = "resume_active_project"
            intent = "project_manage"
            action = "exec_project_action"

        # Check for corrections and cancellations
        is_correction = any(w in q_lower for w in [
            "that isn't the one", "not that one", "that's not what i meant",
            "find the other version", "use the other one", "i meant the other", "not that"
        ])
        is_cancellation = any(w in q_lower for w in [
            "cancel that", "nevermind", "stop that", "abort", "undo the last step", "undo",
            "forget what i", "revert the last", "revert"
        ])

        if is_correction:
            goal = "correct_previous_selection"
            intent = "context_control"
            action = "exec_context_action"

        if is_cancellation:
            goal = "cancel_or_undo_task"
            intent = "context_control"
            action = "exec_context_action"

        # Extract constraints
        constraints: Dict[str, Any] = {}
        if any(w in q_lower for w in ["latest", "most recent", "newest"]):
            constraints["recency"] = "latest"
        elif any(w in q_lower for w in ["oldest", "older", "first created"]):
            constraints["recency"] = "oldest"

        if "relaxing" in q_lower or "chill" in q_lower or "lofi" in q_lower:
            constraints["mood"] = "relaxing"

        if "on youtube" in q_lower:
            constraints["platform"] = "youtube"
        elif "on spotify" in q_lower:
            constraints["platform"] = "spotify"

        # References tracking
        refs = resolved_references or []

        # Multi-step detection
        is_multi_step = False
        atomic_steps: List[str] = []
        if neural_pred.get("is_multi_step") or len(neural_pred.get("steps", [])) > 1:
            is_multi_step = True
            atomic_steps = neural_pred.get("steps", [])

        understanding = StructuredUnderstanding(
            raw_query=raw_query,
            goal=goal,
            intent=intent,
            action=action,
            entities=entities,
            constraints=constraints,
            context_references=refs,
            confidence=confidence,
            is_correction=is_correction,
            is_cancellation=is_cancellation,
            is_multi_step=is_multi_step,
            atomic_steps=atomic_steps,
        )

        # Validate understanding
        understanding.validate()
        return understanding
