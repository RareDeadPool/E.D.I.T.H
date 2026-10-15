"""Clear-Boundary Memory and Preferences Subsystem for E.D.I.T.H. V3.

Strictly separates:
1. Short-term Conversational Context: Ephemeral turn state, active references, and disambiguation.
2. Current Task State: Active goal, step dependencies, current draft and revisions.
3. Persistent User Preferences: Explicit user-approved settings saved to config/user_preferences.json.
4. Search & Action History: Audit trail of recent searches and verified outcomes.

Provides clear inspection, clearing, and reset APIs without leaking sensitive data.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional


class UserPreferences:
    """Manages persistent, user-approved configuration and behavioral preferences."""

    DEFAULT_PREFS = {
        "preferred_music_platform": "spotify",
        "preferred_editor": "notepad",
        "default_workspace_dir": os.path.abspath("."),
        "confirmation_required_for_deletions": True,
        "max_search_results": 5,
        "auto_focus_application": True,
    }

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.path.join("config", "user_preferences.json")
        self.prefs: Dict[str, Any] = dict(self.DEFAULT_PREFS)
        self.load()

    def load(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.prefs.update(data)
            except Exception:
                pass

    def save(self):
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.config_path)), exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.prefs, f, indent=2)
        except Exception:
            pass

    def get(self, key: str, default: Any = None) -> Any:
        return self.prefs.get(key, default if default is not None else self.DEFAULT_PREFS.get(key))

    def set(self, key: str, value: Any):
        self.prefs[key] = value
        self.save()

    def reset_defaults(self):
        self.prefs = dict(self.DEFAULT_PREFS)
        self.save()


class SearchActionHistory:
    """Bounded, privacy-safe chronological log of recent searches and actions."""

    def __init__(self, max_entries: int = 50):
        self.max_entries = max_entries
        self.searches: List[Dict[str, Any]] = []
        self.actions: List[Dict[str, Any]] = []

    def record_search(self, query: str, top_result: Optional[str], candidate_count: int):
        self.searches.append({
            "timestamp": time.time(),
            "query": query,
            "top_result": top_result,
            "count": candidate_count
        })
        if len(self.searches) > self.max_entries:
            self.searches.pop(0)

    def record_action(self, handler: str, parameters: Dict[str, Any], status: str, verified: bool):
        # Sanitize sensitive fields before logging
        safe_params = {k: v for k, v in parameters.items() if k not in ["password", "token", "secret", "auth"]}
        self.actions.append({
            "timestamp": time.time(),
            "handler": handler,
            "parameters": safe_params,
            "status": status,
            "verified": verified
        })
        if len(self.actions) > self.max_entries:
            self.actions.pop(0)

    def clear(self):
        self.searches.clear()
        self.actions.clear()


class EdithMemoryManager:
    """Unified memory coordinator enforcing clear separation of memory tiers."""

    def __init__(self, config_dir: Optional[str] = None, storage_path: Optional[str] = None):
        cfg_path = storage_path or (os.path.join(config_dir, "user_preferences.json") if config_dir else None)
        self.preferences = UserPreferences(config_path=cfg_path)
        self.history = SearchActionHistory()

    def record_search(self, query: str, results: Optional[List[Dict[str, Any]]] = None, top_result: Optional[str] = None, candidate_count: Optional[int] = None):
        count = candidate_count if candidate_count is not None else (len(results) if results else 0)
        top = top_result or (results[0].get("path") if results else None)
        self.history.record_search(query=query, top_result=top, candidate_count=count)

    def record_action(self, handler: str, parameters: Dict[str, Any], status: str = "success", verified: bool = True):
        self.history.record_action(handler=handler, parameters=parameters, status=status, verified=verified)

    def get_summary(self, active_context: Optional[Any] = None) -> Dict[str, Any]:
        """Inspect all tiers of memory clearly separated."""
        summary = {
            "tier_1_persistent_preferences": dict(self.preferences.prefs),
            "tier_2_search_history_count": len(self.history.searches),
            "tier_2_action_history_count": len(self.history.actions),
            "tier_3_task_and_ephemeral_state": {}
        }
        if active_context and hasattr(active_context, "to_dict"):
            summary["tier_3_task_and_ephemeral_state"] = active_context.to_dict()
        return summary

    def clear_ephemeral(self, active_context: Optional[Any] = None):
        """Clear task and conversation memory without erasing user preferences."""
        if active_context and hasattr(active_context, "reset"):
            active_context.reset()
        self.history.clear()

    def factory_reset(self, active_context: Optional[Any] = None):
        """Full reset: restore default preferences and clear all context/history."""
        self.preferences.reset_defaults()
        self.clear_ephemeral(active_context)
