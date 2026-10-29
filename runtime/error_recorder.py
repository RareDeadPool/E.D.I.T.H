"""Error-Driven Learning and Structured Diagnostics for E.D.I.T.H. V8.

Provides privacy-conscious diagnostic tracking of operational failures,
user corrections, misclassifications, and unexpected tool outcomes.
Enables controlled conversion of real-world failures into reproducible regression tests
and versioned dataset additions without uncontrolled online learning.
"""

import os
import re
import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict, field

DIAGNOSTICS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "diagnostics"))
FAILURES_FILE = os.path.join(DIAGNOSTICS_DIR, "operational_failures.jsonl")
CANDIDATE_DATASET_FILE = os.path.join(DIAGNOSTICS_DIR, "reviewed_corrections.json")

# Privacy redact patterns (passwords, tokens, keys, card numbers)
REDACT_PATTERNS = [
    (re.compile(r"(?i)(password|secret|api[_\s]?key|bearer\s+[a-zA-Z0-9_\-\.]|\bkey)\s*[:=]\s*[^\s,]+"), "[REDACTED_CREDENTIAL]"),
    (re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b"), "[REDACTED_CARD]"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"), "[REDACTED_EMAIL]"),
]


def redact_sensitive_text(text: str) -> str:
    """Sanitize diagnostic text against inadvertent secrets leakage."""
    sanitized = text
    for pattern, replacement in REDACT_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


@dataclass
class FailureTrace:
    """Structured record of a single conversational or execution failure."""
    trace_id: str
    timestamp: str
    utterance: str
    predicted_intent: str
    expected_intent: Optional[str] = None
    confidence: float = 0.0
    selected_tool: str = "none"
    outcome_status: str = "failed"
    failure_category: str = "unclassified"  # classification_error, slot_mismatch, precondition_failed, execution_error, user_correction, timeout, ambiguity_unresolved
    recovery_action: Optional[str] = None
    user_corrected: bool = False
    model_version: str = "edith_v8"
    state_summary: Dict[str, Any] = field(default_factory=dict)
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailureTrace":
        return cls(**data)


class ErrorRecorder:
    """
    Controlled failure recorder and regression loop manager.
    Adheres to local-first privacy: no cloud upload, local JSONL persistence,
    explicit opt-out, and manual review gating.
    """

    def __init__(self, enabled: bool = True, max_records: int = 1000):
        self.enabled = enabled
        self.max_records = max_records
        os.makedirs(DIAGNOSTICS_DIR, exist_ok=True)

    def record_failure(
        self,
        utterance: str,
        predicted_intent: str,
        expected_intent: Optional[str] = None,
        confidence: float = 0.0,
        selected_tool: str = "none",
        outcome_status: str = "failed",
        failure_category: str = "unclassified",
        recovery_action: Optional[str] = None,
        user_corrected: bool = False,
        state_summary: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
    ) -> Optional[FailureTrace]:
        """Record an operational failure or correction if telemetry is active."""
        if not self.enabled:
            return None

        trace_id = f"fail_{int(time.time() * 1000)}_{os.urandom(2).hex()}"
        clean_text = redact_sensitive_text(utterance)

        trace = FailureTrace(
            trace_id=trace_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            utterance=clean_text,
            predicted_intent=predicted_intent,
            expected_intent=expected_intent,
            confidence=round(confidence, 4),
            selected_tool=selected_tool,
            outcome_status=outcome_status,
            failure_category=failure_category,
            recovery_action=recovery_action,
            user_corrected=user_corrected,
            state_summary=state_summary or {},
            notes=notes,
        )

        try:
            with open(FAILURES_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
        except Exception:
            pass

        return trace

    def list_failures(self, category: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """List recorded failure traces with optional category filtering."""
        if not os.path.exists(FAILURES_FILE):
            return []

        results = []
        try:
            with open(FAILURES_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if category is None or record.get("failure_category") == category:
                            results.append(record)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            return []

        return results[-limit:]

    def clear_failures(self) -> bool:
        """Purge all recorded failure traces."""
        if os.path.exists(FAILURES_FILE):
            try:
                os.remove(FAILURES_FILE)
                return True
            except Exception:
                return False
        return True

    def propose_dataset_example(
        self,
        trace_id: str,
        verified_intent: str,
        verified_slots: Optional[List[Dict[str, str]]] = None,
        calibrated_conf: float = 1.0
    ) -> Optional[Dict[str, Any]]:
        """
        Human-in-the-loop review tool: Promotes a verified failure record
        into candidate training data with validated intent and slot annotations.
        """
        failures = self.list_failures(limit=500)
        target = next((f for f in failures if f["trace_id"] == trace_id), None)
        if not target:
            return None

        sample = {
            "source_trace_id": trace_id,
            "text": target["utterance"],
            "intent": verified_intent,
            "slots": verified_slots or [],
            "calibrated_confidence": calibrated_conf,
            "reviewed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "human_verified",
        }

        # Append to reviewed candidate dataset
        existing = []
        if os.path.exists(CANDIDATE_DATASET_FILE):
            try:
                with open(CANDIDATE_DATASET_FILE, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = []

        existing.append(sample)
        with open(CANDIDATE_DATASET_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)

        return sample

    def generate_regression_test(self, trace_id: str) -> Optional[str]:
        """Generate a Python test method stub for a recorded failure."""
        failures = self.list_failures(limit=500)
        target = next((f for f in failures if f["trace_id"] == trace_id), None)
        if not target:
            return None

        q = target["utterance"].replace('"', '\\"')
        expected = target.get("expected_intent") or target.get("predicted_intent")
        fn_name = f"test_regression_{target['trace_id']}"

        stub = f"""
    def {fn_name}(self):
        \"\"\"Regression test for failure {trace_id}: {target.get('failure_category', 'general')}\"\"\"
        result = self.brain.process(\"{q}\", execute=False)
        self.assertEqual(result.get("intent"), \"{expected}\")
        self.assertGreaterEqual(result.get("operational_confidence", 0.0), 0.40)
"""
        return stub.strip()
