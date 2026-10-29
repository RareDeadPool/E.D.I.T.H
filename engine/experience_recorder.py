"""Privacy-Preserving Local Experience Recorder & Feedback Interpretation System for E.D.I.T.H. V5.2.

Implements:
1. Structured Interaction Event Logging:
   Captures operational events with strict separation between observable facts
   and inferred model hypotheses.
2. Privacy & Consent Controls:
   - Local-only storage (default: config/experience_data/).
   - Explicit opt-in (disabled by default).
   - Sensitive credential/token redaction.
   - Configurable retention (max 1000 events, 30-day window).
   - Audit, export, clear, and individual deletion controls.
   - Per-interaction exclusion (/do-not-learn).
3. Feedback Interpretation Engine:
   Hierarchy of evidence distinguishing:
   - Explicit Corrections (entity/action redirection)
   - Negative Outcomes (goal missed)
   - Positive Feedback (solution confirmed)
   - Repeated Execution Failures (procedural unreliability)
   - Explicit Style Preferences (communication adaptation)
   - Ambiguous Acknowledgments (weak conversational proxy, not ground truth)
"""

from typing import Dict, Any, List, Optional, Tuple, Set
import os
import re
import time
import json
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum


class FeedbackCategory(str, Enum):
    EXPLICIT_CORRECTION = "explicit_correction"
    NEGATIVE_OUTCOME = "negative_outcome"
    POSITIVE_FEEDBACK = "positive_feedback"
    REPEATED_FAILURE = "repeated_failure"
    STYLE_PREFERENCE = "style_preference"
    AMBIGUOUS = "ambiguous"
    NONE = "none"


@dataclass
class FeedbackSignal:
    """Interpreted user feedback with confidence and evidence hierarchy weight."""
    category: FeedbackCategory
    evidence_weight: float             # 0.0 (weak/ambiguous) to 1.0 (verified closed-loop / explicit)
    learning_directive: str            # e.g., 'update_entity_binding', 'downgrade_procedure', 'record_style_preference'
    details: Dict[str, Any] = field(default_factory=dict)
    is_ambiguous: bool = False
    requires_confirmation: bool = False


@dataclass
class InteractionEvent:
    """Schema-versioned, privacy-scrubbed interaction event record."""
    schema_version: str = "5.2.0"
    event_id: str = field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)
    task_id: str = "task_default"
    session_id: str = "session_default"

    # Input and Plan
    user_request: str = ""             # Redacted / sanitized
    resolved_intent: str = "unknown"
    resolved_action: str = "unknown"
    entities: Dict[str, Any] = field(default_factory=dict)
    contextual_references: Dict[str, Any] = field(default_factory=dict)

    # Predictions & Confidences
    prior_confidence: float = 0.5
    calibrated_confidence: float = 0.5
    predicted_plan: List[str] = field(default_factory=list)

    # Execution & Verification Outcomes
    execution_status: str = "pending"  # 'success', 'failure', 'needs_confirmation', 'needs_clarification'
    verified: bool = False
    error_message: Optional[str] = None
    latency_ms: float = 0.0

    # User Feedback & Corrections
    user_correction: Optional[str] = None
    feedback_category: str = "none"
    feedback_weight: float = 0.0
    retry_occurred: bool = False
    retry_reason: Optional[str] = None
    task_completed: bool = False
    explicit_feedback: Optional[str] = None

    # Provenance & Influence Tracing
    influencing_memory_ids: List[str] = field(default_factory=list)
    influencing_procedure: Optional[str] = None

    # Strict separation between facts and model interpretations
    observed_facts: Dict[str, Any] = field(default_factory=dict)
    inferred_explanations: Dict[str, Any] = field(default_factory=dict)

    # Privacy & Consent
    consent_state: str = "opt_in_local"
    exclude_from_learning: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------------------------------------------------------------
# Feedback Interpretation Engine (Evidence Hierarchy)
# -----------------------------------------------------------------------------
class FeedbackInterpreter:
    """
    Parses dialogue and interaction results into structured learning signals
    using an evidence hierarchy that prevents treating casual remarks as ground truth.
    """

    CORRECTION_PATTERNS = [
        (r"^(?:no|nope|not\s+that|wrong)\s*,?\s*(?:i\s+meant|i\s+said|use|open|choose|target)\s+(.+)$", "target_correction"),
        (r"^(?:no|not)\s+(?:the\s+)?([a-zA-Z0-9_\-]+(?:\s+[a-zA-Z0-9_\-]+)?)\s*,?\s*(?:use|open|the)\s+([a-zA-Z0-9_\-]+(?:\s+[a-zA-Z0-9_\-]+)?)$", "replacement_correction"),
        (r"^no\s*,?\s*i\s+meant\s+the\s+([a-zA-Z0-9_\-]+)\s*,?\s*not\s+(?:the\s+)?([a-zA-Z0-9_\-]+)$", "entity_swap"),
        (r"^(?:actually|wait)\s*,?\s*(?:do\s+not|don't)\s+(.+)$", "plan_abort")
    ]

    NEGATIVE_OUTCOME_PATTERNS = [
        r"you\s+opened\s+the\s+wrong\s+(?:file|window|app|tab)",
        r"that\s+is\s+the\s+wrong\s+(?:file|folder|document|person|contact|number)",
        r"this\s+is\s+not\s+what\s+i\s+(?:asked|wanted|requested)",
        r"that\s+didn't\s+work",
        r"(?:it|this|that|the\s+same\s+approach|same\s+method)\s+failed(?:\s+again)?",
        r"same\s+error\s+(?:happened|returned)",
        r"failed\s+again",
        r"that\s+broke"
    ]

    POSITIVE_FEEDBACK_PATTERNS = [
        r"yes\s*,?\s*that(?:'s|\s+is)\s+exactly\s+what\s+i\s+wanted",
        r"perfect\s*,?\s*thank\s+you",
        r"that\s+worked\s*(!|\.)?",
        r"great\s+job",
        r"awesome\s*,?\s*thanks",
        r"that\s+fixed\s+it"
    ]

    STYLE_PREFERENCE_PATTERNS = [
        (r"(?:please\s+)?give\s+me\s+(?:the\s+)?direct\s+answer\s+first(?:\s*,?\s*then\s+explain)?", "direct_first"),
        (r"(?:please\s+)?be\s+(?:more\s+)?concise|keep\s+it\s+short|less\s+talking", "concise"),
        (r"(?:please\s+)?explain\s+(?:in\s+detail|step\s+by\s+step|the\s+details)", "detailed"),
        (r"(?:always\s+)?ask\s+me\s+before\s+(?:deleting|closing|sending|executing)", "require_confirmation")
    ]

    AMBIGUOUS_PATTERNS = [
        r"^(?:ok|okay|fine|sure|alright|cool|k|got\s+it|mhm|uh\s*huh)[\.!\?]*$"
    ]

    def interpret_feedback(
        self,
        user_utterance: str,
        prior_action: Optional[str] = None,
        prior_verified: bool = True
    ) -> FeedbackSignal:
        """
        Classifies user utterance according to evidence hierarchy:
        1. Explicit entity / action correction -> 0.95 weight
        2. Negative outcome assertion -> 0.90 weight
        3. Explicit positive confirmation -> 0.75 weight
        4. Style preference directive -> 0.80 weight
        5. Ambiguous acknowledgment -> 0.15 weight (weak proxy)
        """
        text = user_utterance.strip().lower()

        # 1. Explicit Corrections
        for pattern, corr_type in self.CORRECTION_PATTERNS:
            m = re.match(pattern, text, re.IGNORECASE)
            if m:
                groups = m.groups()
                return FeedbackSignal(
                    category=FeedbackCategory.EXPLICIT_CORRECTION,
                    evidence_weight=0.95,
                    learning_directive="update_entity_or_intent_binding",
                    details={
                        "correction_type": corr_type,
                        "extracted": groups[0] if groups else text,
                        "raw": user_utterance
                    }
                )

        # 2. Negative Outcome Declarations
        if any(re.search(p, text) for p in self.NEGATIVE_OUTCOME_PATTERNS):
            is_repeated = "again" in text or "same error" in text
            return FeedbackSignal(
                category=FeedbackCategory.REPEATED_FAILURE if is_repeated else FeedbackCategory.NEGATIVE_OUTCOME,
                evidence_weight=0.90,
                learning_directive="downgrade_procedure_reliability" if is_repeated else "mark_goal_unfulfilled",
                details={"prior_action": prior_action, "prior_verified": prior_verified}
            )

        # 3. Explicit Style Preferences
        for pat, pref_code in self.STYLE_PREFERENCE_PATTERNS:
            if re.search(pat, text):
                return FeedbackSignal(
                    category=FeedbackCategory.STYLE_PREFERENCE,
                    evidence_weight=0.80,
                    learning_directive="record_communication_preference",
                    details={"preference": pref_code}
                )

        # 4. Explicit Positive Feedback
        if any(re.search(p, text) for p in self.POSITIVE_FEEDBACK_PATTERNS):
            return FeedbackSignal(
                category=FeedbackCategory.POSITIVE_FEEDBACK,
                evidence_weight=0.75,
                learning_directive="reinforce_successful_strategy",
                details={"prior_action": prior_action}
            )

        # 5. Ambiguous Conversational Acknowledgments (Weak proxy)
        if any(re.match(p, text) for p in self.AMBIGUOUS_PATTERNS):
            return FeedbackSignal(
                category=FeedbackCategory.AMBIGUOUS,
                evidence_weight=0.15,
                learning_directive="no_update_ambiguous_proxy",
                details={"raw": user_utterance},
                is_ambiguous=True
            )

        return FeedbackSignal(
            category=FeedbackCategory.NONE,
            evidence_weight=0.0,
            learning_directive="noop"
        )


# -----------------------------------------------------------------------------
# Privacy-Preserving Local Experience Recorder
# -----------------------------------------------------------------------------
class ExperienceRecorder:
    """
    Manages consent-gated, locally persisted interaction event collection
    with redaction, retention pruning, and administrative controls.
    """

    REDACTION_PATTERNS = [
        # Passwords and secrets
        re.compile(r"(?i)\b(password|passwd|secret|token|api[_-]?key|bearer\s+[a-zA-Z0-9_\-\.]+)\s*[:=]\s*['\"]?([^\s'\"]+)", re.IGNORECASE),
        # Credit cards / SSNs
        re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        # Long hex or base64 tokens
        re.compile(r"\b[a-fA-F0-9]{32,64}\b"),
    ]

    def __init__(
        self,
        storage_dir: Optional[str] = None,
        enabled: bool = False,
        collect_raw_transcripts: bool = False,
        max_events: int = 1000,
        retention_days: int = 30
    ):
        self.storage_dir = storage_dir or os.path.join("config", "experience_data")
        self.enabled = enabled
        self.collect_raw_transcripts = collect_raw_transcripts
        self.max_events = max_events
        self.retention_days = retention_days

        self.interpreter = FeedbackInterpreter()
        self.events_file = os.path.join(self.storage_dir, "interaction_events.jsonl")
        self.settings_file = os.path.join(self.storage_dir, "recorder_settings.json")

        self.in_memory_events: List[InteractionEvent] = []
        os.makedirs(self.storage_dir, exist_ok=True)
        self._load_settings()
        self._load_recent_events()

    def _load_settings(self):
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.enabled = data.get("enabled", self.enabled)
                    self.collect_raw_transcripts = data.get("collect_raw_transcripts", self.collect_raw_transcripts)
                    self.max_events = data.get("max_events", self.max_events)
                    self.retention_days = data.get("retention_days", self.retention_days)
            except Exception:
                pass

    def save_settings(self):
        os.makedirs(self.storage_dir, exist_ok=True)
        with open(self.settings_file, "w", encoding="utf-8") as f:
            json.dump({
                "enabled": self.enabled,
                "collect_raw_transcripts": self.collect_raw_transcripts,
                "max_events": self.max_events,
                "retention_days": self.retention_days
            }, f, indent=2)

    def redact_sensitive_text(self, text: str) -> str:
        """Sanitize secrets, passwords, and tokens before storage."""
        redacted = text
        for pat in self.REDACTION_PATTERNS:
            redacted = pat.sub("[REDACTED_SECRET]", redacted)
        return redacted

    def _load_recent_events(self):
        if not os.path.exists(self.events_file):
            return
        loaded = []
        try:
            with open(self.events_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        loaded.append(InteractionEvent(**data))
            self.in_memory_events = loaded[-self.max_events:]
        except Exception:
            pass

    def record_interaction(
        self,
        user_request: str,
        resolved_intent: str,
        resolved_action: str,
        entities: Dict[str, Any],
        contextual_references: Dict[str, Any],
        prior_confidence: float,
        calibrated_confidence: float,
        predicted_plan: List[str],
        execution_status: str,
        verified: bool,
        error_message: Optional[str] = None,
        user_correction: Optional[str] = None,
        task_completed: bool = True,
        influencing_memory_ids: Optional[List[str]] = None,
        influencing_procedure: Optional[str] = None,
        observed_facts: Optional[Dict[str, Any]] = None,
        inferred_explanations: Optional[Dict[str, Any]] = None,
        exclude_from_learning: bool = False,
        task_id: str = "task_default",
        session_id: str = "session_default",
        latency_ms: float = 0.0
    ) -> Optional[InteractionEvent]:
        """
        Record an interaction event if experience collection is enabled.
        Applies redaction and interprets feedback provenance.
        """
        # If disabled by global setting or flagged to exclude, do not persist
        if not self.enabled or exclude_from_learning:
            return None

        # Redact raw request unless raw transcripts explicitly allowed
        clean_request = self.redact_sensitive_text(user_request)
        if not self.collect_raw_transcripts:
            # Store structural intent summary rather than full raw text
            clean_request = f"[{resolved_intent}] {clean_request[:120]}"

        # Interpret user correction or feedback
        feedback_sig = FeedbackSignal(category=FeedbackCategory.NONE, evidence_weight=0.0, learning_directive="noop")
        if user_correction:
            feedback_sig = self.interpreter.interpret_feedback(
                user_correction,
                prior_action=resolved_action,
                prior_verified=verified
            )

        event = InteractionEvent(
            task_id=task_id,
            session_id=session_id,
            user_request=clean_request,
            resolved_intent=resolved_intent,
            resolved_action=resolved_action,
            entities=entities or {},
            contextual_references=contextual_references or {},
            prior_confidence=round(prior_confidence, 3),
            calibrated_confidence=round(calibrated_confidence, 3),
            predicted_plan=predicted_plan or [],
            execution_status=execution_status,
            verified=verified,
            error_message=self.redact_sensitive_text(error_message) if error_message else None,
            latency_ms=round(latency_ms, 2),
            user_correction=self.redact_sensitive_text(user_correction) if user_correction else None,
            feedback_category=feedback_sig.category.value,
            feedback_weight=feedback_sig.evidence_weight,
            retry_occurred=bool(error_message),
            retry_reason=error_message,
            task_completed=task_completed,
            explicit_feedback=user_correction,
            influencing_memory_ids=influencing_memory_ids or [],
            influencing_procedure=influencing_procedure,
            observed_facts=observed_facts or {},
            inferred_explanations=inferred_explanations or {},
            consent_state="opt_in_local",
            exclude_from_learning=exclude_from_learning
        )

        self.in_memory_events.append(event)
        self._append_to_file(event)
        self._prune_retention()
        return event

    def _append_to_file(self, event: InteractionEvent):
        try:
            with open(self.events_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
        except Exception:
            pass

    def _prune_retention(self):
        """Enforce max_events limit and 30-day time window."""
        cutoff_time = time.time() - (self.retention_days * 86400.0)
        valid = [e for e in self.in_memory_events if e.timestamp >= cutoff_time]
        if len(valid) > self.max_events:
            valid = valid[-self.max_events:]

        if len(valid) != len(self.in_memory_events):
            self.in_memory_events = valid
            try:
                with open(self.events_file, "w", encoding="utf-8") as f:
                    for e in self.in_memory_events:
                        f.write(json.dumps(e.to_dict()) + "\n")
            except Exception:
                pass

    # Administrative Controls
    def inspect_events(self, limit: int = 50, filter_category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve recent events for auditing."""
        events = self.in_memory_events
        if filter_category:
            events = [e for e in events if e.feedback_category == filter_category]
        return [e.to_dict() for e in events[-limit:]]

    def export_events(self, export_path: str, anonymize: bool = True) -> int:
        """Export events to an external JSON file with optional extra anonymization."""
        os.makedirs(os.path.dirname(os.path.abspath(export_path)), exist_ok=True)
        exported = []
        for e in self.in_memory_events:
            d = e.to_dict()
            if anonymize:
                d["session_id"] = "anon_session"
                d["task_id"] = "anon_task"
            exported.append(d)
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(exported, f, indent=2)
        return len(exported)

    def delete_event(self, event_id: str) -> bool:
        """Delete an individual event by ID."""
        initial_len = len(self.in_memory_events)
        self.in_memory_events = [e for e in self.in_memory_events if e.event_id != event_id]
        if len(self.in_memory_events) < initial_len:
            try:
                with open(self.events_file, "w", encoding="utf-8") as f:
                    for e in self.in_memory_events:
                        f.write(json.dumps(e.to_dict()) + "\n")
            except Exception:
                pass
            return True
        return False

    def clear_all_events(self) -> int:
        """Wipe all recorded interaction events."""
        count = len(self.in_memory_events)
        self.in_memory_events = []
        if os.path.exists(self.events_file):
            try:
                os.remove(self.events_file)
            except Exception:
                pass
        return count
