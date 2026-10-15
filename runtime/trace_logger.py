"""Execution Trace & Reliability Logger for E.D.I.T.H. V4.

Records end-to-end telemetry for every user command:
- Original user message
- Resolved intent and slot entities
- Selected handler and task plan
- Resolved file / draft / entity identifier
- Execution outcome and return values
- Closed-loop verification results
- Final user-facing response
- Explicit failure reasons (if any)

Includes automatic redaction of credentials, API tokens, and sensitive file contents.
"""

import os
import re
import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict


SENSITIVE_PATTERNS = [
    (r"(?i)(bearer\s+)[a-zA-Z0-9_\-\.]{15,}", r"\1[REDACTED_TOKEN]"),
    (r"(?i)(api[_-]?key\s*[:=]\s*['\"]?)[a-zA-Z0-9_\-\.]{15,}['\"]?", r"\1[REDACTED_API_KEY]"),
    (r"(?i)(password\s*[:=]\s*['\"]?)[^'\"\s]{4,}['\"]?", r"\1[REDACTED_PASSWORD]"),
    (r"(?i)(secret\s*[:=]\s*['\"]?)[a-zA-Z0-9_\-\.]{10,}['\"]?", r"\1[REDACTED_SECRET]"),
]


def redact_sensitive(obj: Any) -> Any:
    """Recursively redact passwords, tokens, and secrets from trace payloads."""
    if isinstance(obj, str):
        redacted = obj
        for pat, rep in SENSITIVE_PATTERNS:
            redacted = re.sub(pat, rep, redacted)
        return redacted
    elif isinstance(obj, dict):
        return {k: redact_sensitive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [redact_sensitive(item) for item in obj]
    return obj


@dataclass
class ExecutionTrace:
    """Structured execution trace record for an atomic or multi-step command."""
    trace_id: str
    original_message: str
    resolved_query: str
    intent: str
    action: str
    entities: Dict[str, Any]
    handler: Optional[str]
    plan: Optional[Dict[str, Any]]
    resolved_target: Optional[str]
    execution_result: Optional[Dict[str, Any]]
    verification_result: Optional[Dict[str, Any]]
    final_response: str
    status: str
    failure_reason: Optional[str] = None
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return redact_sensitive(data)


class ExecutionTraceLogger:
    """Manages local in-memory and file-backed execution traces for debugging."""

    def __init__(self, log_path: Optional[str] = None, max_memory_traces: int = 100):
        self.max_memory_traces = max_memory_traces
        self.traces: List[ExecutionTrace] = []
        if log_path is None:
            log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
            os.makedirs(log_dir, exist_ok=True)
            self.log_path = os.path.join(log_dir, "execution_traces.jsonl")
        else:
            self.log_path = log_path
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

    def log_trace(self, trace: ExecutionTrace) -> ExecutionTrace:
        """Store trace in memory and append to local JSONL log."""
        self.traces.append(trace)
        if len(self.traces) > self.max_memory_traces:
            self.traces = self.traces[-self.max_memory_traces:]

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
        except Exception:
            pass
        return trace

    def get_last_trace(self) -> Optional[ExecutionTrace]:
        """Return the most recent execution trace."""
        return self.traces[-1] if self.traces else None

    def get_all_traces(self) -> List[Dict[str, Any]]:
        """Return all traces currently held in memory."""
        return [t.to_dict() for t in self.traces]

    def clear(self):
        """Clear in-memory traces."""
        self.traces.clear()
