"""Stateful Conversation and Structured Task Context Management for E.D.I.T.H. Intelligence Core V2.

Tracks multi-turn context, active workflows, candidate files/search results,
active projects, process outcomes, and handles contextual reference resolution
(anaphora, ordinals, relative recency, user corrections, reversals, and unambiguous clarification).
"""

import os
import re
import time
from typing import Dict, Any, List, Optional, Tuple

from runtime.text_writer import DraftManager


class StructuredTaskContext:
    """
    Persistent, structured task context separate from raw conversational history.
    Maintains domain entities, candidate result pools, active workflows, and undo stacks.
    """

    def __init__(self, ttl_seconds: float = 900.0):
        self.ttl_seconds = ttl_seconds
        self.last_activity_time: float = time.time()

        # 1. Goal and Workflow State
        self.active_goal: Optional[str] = None
        self.active_workflow: Optional[Dict[str, Any]] = None
        self.active_step_index: int = 0
        self.workflow_status: str = "idle"  # idle, in_progress, paused, completed, cancelled
        self.active_plan: Optional[Any] = None

        # 2. Project and Development State
        self.active_project: Optional[str] = "E.D.I.T.H."
        self.active_project_path: Optional[str] = os.path.abspath(".")
        self.last_command_executed: Optional[str] = None
        self.last_test_output: Optional[Dict[str, Any]] = None

        # 3. Environment & Application State
        self.last_app: Optional[str] = None
        self.last_window: Optional[str] = None
        self.last_service: Optional[str] = None
        self.last_channel: Optional[str] = None

        # 4. File and Search Candidate State
        self.last_project: Optional[str] = None
        self.last_file: Optional[str] = None
        self.last_path: Optional[str] = None
        self.last_destination: Optional[str] = None
        self.search_candidates: List[Dict[str, Any]] = []
        self.candidate_cursor: int = 0  # Points to currently inspected candidate

        # 5. Entity & Media State
        self.last_person: Optional[str] = None
        self.last_song: Optional[str] = None
        self.last_artist: Optional[str] = None
        self.last_platform: str = "spotify"
        self.last_media_action: Optional[str] = None

        # 6. Action Execution Results & History
        self.last_action_result: Optional[Dict[str, Any]] = None
        self.turn_history: List[Dict[str, Any]] = []
        self.action_history: List[Dict[str, Any]] = []
        self.undo_stack: List[Dict[str, Any]] = []

        # 7. Decision Gates & Clarifications
        self.pending_clarification: Optional[Dict[str, Any]] = None
        self.pending_decisions: List[Dict[str, Any]] = []

        # 8. Text Drafting & Editing Pipeline State
        self.draft_manager = DraftManager()

        # 9. Cross-Application V4 Task State
        self.selected_recipient: Optional[str] = None
        self.selected_phone: Optional[str] = None
        self.is_recipient_group: bool = False
        self.active_conversation: Optional[str] = None
        self.selected_file_path: Optional[str] = None
        self.pending_attachment_path: Optional[str] = None
        self.pending_confirmation: Optional[Dict[str, Any]] = None
        self.user_approved_send: bool = False
        self.completed_actions: List[Dict[str, Any]] = []
        self.failed_actions: List[Dict[str, Any]] = []
        self.verification_evidence: List[Dict[str, Any]] = []
        self.is_cancelled: bool = False

    def has_draft(self) -> bool:
        """Check if an active draft exists in context."""
        return self.draft_manager.has_draft()

    def set_draft(self, body: str, topic: str = "Untitled Document", subject: str = "", recipient: str = "", document_type: str = "general"):
        """Store or update active draft in context."""
        self.touch()
        return self.draft_manager.create_draft(body=body, topic=topic, subject=subject, recipient=recipient, document_type=document_type)

    def get_current_draft(self):
        """Retrieve current active draft revision."""
        return self.draft_manager.get_current_draft()

    def revise_draft(self, instruction: str):
        """Apply in-place revision to the current draft."""
        self.touch()
        return self.draft_manager.revise_draft(instruction)

    def revert_draft(self):
        """Revert draft to the previous revision."""
        self.touch()
        return self.draft_manager.revert_draft()

    def set_selected_recipient(self, name: str, phone: Optional[str] = None, is_group: bool = False):
        """Record the active recipient or group for the current task."""
        self.touch()
        self.selected_recipient = name
        self.selected_phone = phone
        self.is_recipient_group = is_group
        self.last_person = name

    def set_selected_file(self, file_path: str):
        """Record the selected file and verified full path."""
        self.touch()
        self.selected_file_path = file_path
        self.last_file = os.path.basename(file_path)
        self.last_path = file_path

    def cancel_current_task(self) -> Dict[str, Any]:
        """Cancel pending actions immediately. Guarantees no pending send proceeds."""
        self.touch()
        self.is_cancelled = True
        self.pending_confirmation = None
        self.user_approved_send = False
        self.workflow_status = "cancelled"
        return {
            "status": "cancelled",
            "message": "Task cancelled. Pending actions and draft sends were halted."
        }

    def approve_pending(self):
        """Mark pending action as approved by the user."""
        self.touch()
        self.user_approved_send = True
        self.is_cancelled = False

    def record_action_outcome(self, handler: str, success: bool, details: Dict[str, Any], evidence: Optional[str] = None):
        """Record observable verification evidence and outcome."""
        self.touch()
        entry = {
            "handler": handler,
            "success": success,
            "details": details,
            "evidence": evidence,
            "timestamp": time.time()
        }
        if success:
            self.completed_actions.append(entry)
        else:
            self.failed_actions.append(entry)
        if evidence:
            self.verification_evidence.append(entry)

    def touch(self):
        """Update last activity timestamp."""
        self.last_activity_time = time.time()

    def is_expired(self) -> bool:
        """Check if session context has expired due to inactivity."""
        return (time.time() - self.last_activity_time) > self.ttl_seconds

    def reset_if_expired(self):
        """Reset state if context TTL has elapsed."""
        if self.is_expired():
            self.reset()

    def reset(self):
        """Clear all ephemeral context memory while preserving defaults."""
        self.active_goal = None
        self.active_workflow = None
        self.active_step_index = 0
        self.workflow_status = "idle"
        self.last_app = None
        self.last_window = None
        self.last_service = None
        self.last_channel = None
        self.last_file = None
        self.last_path = None
        self.last_destination = None
        self.search_candidates = []
        self.candidate_cursor = 0
        self.last_person = None
        self.last_song = None
        self.last_artist = None
        self.last_platform = "spotify"
        self.last_media_action = None
        self.last_command_executed = None
        self.selected_recipient = None
        self.selected_phone = None
        self.is_recipient_group = False
        self.active_conversation = None
        self.selected_file_path = None
        self.pending_attachment_path = None
        self.pending_confirmation = None
        self.user_approved_send = False
        self.is_cancelled = False
        self.completed_actions = []
        self.failed_actions = []
        self.verification_evidence = []
        self.last_test_output = None
        self.last_action_result = None
        self.pending_clarification = None
        self.pending_decisions = []
        self.turn_history = []
        self.action_history = []
        self.undo_stack = []
        self.touch()

    def record_turn(self, query: str, intent: str, action: str, result: Dict[str, Any]):
        """Record turn outcome into history."""
        self.touch()
        self.turn_history.append({
            "query": query,
            "intent": intent,
            "action": action,
            "timestamp": time.time(),
            "status": result.get("status")
        })
        if len(self.turn_history) > 30:
            self.turn_history = self.turn_history[-30:]

    def record_action_result(self, action_type: str, result: Dict[str, Any]):
        """Record specific action result with return codes, paths, and undo data."""
        self.touch()
        self.last_action_result = result
        entry = {
            "action": action_type,
            "result": result,
            "timestamp": time.time(),
            "status": result.get("status", "unknown")
        }
        self.action_history.append(entry)

        # Update entity pointers from results
        if result.get("path"):
            self.last_path = result["path"]
            self.last_file = os.path.basename(result["path"])
        if result.get("app_name") or result.get("app"):
            self.last_app = result.get("app_name") or result.get("app")
        if result.get("destination"):
            self.last_destination = result["destination"]

        # Track undo capability for reversible operations
        if action_type in ["file_copy", "file_move", "file_create", "create_file"]:
            self.undo_stack.append({
                "action": action_type,
                "path": result.get("path") or result.get("created_path"),
                "source": result.get("source"),
                "destination": result.get("destination")
            })

    def set_search_results(self, items: List[Dict[str, Any]]):
        """Store candidate file or item search results for follow-up ordinal resolution."""
        self.touch()
        self.search_candidates = items
        self.candidate_cursor = 0
        if items:
            top = items[0]
            self.last_path = top.get("path")
            self.last_file = top.get("name") or (os.path.basename(top["path"]) if top.get("path") else None)

    # ──────────────────────── Text Drafting & Revision Helpers ────────────────────────

    def set_draft(self, body: str, topic: str = "", subject: str = "", recipient: str = "", document_type: str = "document"):
        """Store or update the current active draft in context."""
        self.touch()
        return self.draft_manager.create_draft(body=body, topic=topic, subject=subject, recipient=recipient, document_type=document_type)

    def get_current_draft(self):
        """Retrieve active draft."""
        return self.draft_manager.get_current_draft()

    def has_draft(self) -> bool:
        """Check if context holds an active draft."""
        return self.draft_manager.has_draft()

    def revise_draft(self, instruction: str):
        """Revise the active draft with an editing instruction."""
        self.touch()
        return self.draft_manager.revise_draft(instruction)

    def revert_draft(self):
        """Revert the active draft to previous revision."""
        self.touch()
        return self.draft_manager.revert_draft()

    def clear_all(self):
        """Clear all context including drafts, active project, and search candidates."""
        self.reset()
        self.active_project = None
        self.active_project_path = None
        self.last_project = None
        if hasattr(self, "draft_manager"):
            self.draft_manager.clear()

    # ──────────────────────── Reference & Anaphora Resolution ────────────────────────

    def resolve_ordinal(self, text: str) -> Optional[Dict[str, Any]]:
        """Resolve ordinal references ('the second PDF', 'first one', 'the latest', 'newer') against search candidates."""
        if not self.search_candidates:
            return None

        t = text.strip().lower()
        candidates = list(self.search_candidates)

        # Filter by file extension if specified (e.g. 'the second pdf')
        ext_match = re.search(r"\b(pdf|txt|docx?|xlsx?|pptx?|csv|json|py|png|jpe?g)\b", t)
        if ext_match:
            req_ext = "." + ext_match.group(1).lower()
            filtered = [c for c in candidates if str(c.get("path", "")).lower().endswith(req_ext) or str(c.get("name", "")).lower().endswith(req_ext)]
            if filtered:
                candidates = filtered

        # Check recency words: 'latest', 'newest', 'newer', 'most recent'
        if any(w in t for w in ["latest", "newest", "newer", "most recent", "recent"]):
            sorted_by_time = sorted(candidates, key=lambda c: c.get("timestamp", c.get("modified_time", 0)), reverse=True)
            if sorted_by_time:
                chosen = sorted_by_time[0]
                self.candidate_cursor = self.search_candidates.index(chosen) if chosen in self.search_candidates else 0
                self._update_selected_file(chosen)
                return chosen

        # Check older words
        if any(w in t for w in ["older", "oldest", "first one created"]):
            sorted_by_time = sorted(candidates, key=lambda c: c.get("timestamp", c.get("modified_time", 0)))
            if sorted_by_time:
                chosen = sorted_by_time[0]
                self.candidate_cursor = self.search_candidates.index(chosen) if chosen in self.search_candidates else 0
                self._update_selected_file(chosen)
                return chosen

        # Numeric and word ordinals
        ordinal_map = {
            "first": 0, "1st": 0,
            "second": 1, "2nd": 1,
            "third": 2, "3rd": 2,
            "fourth": 3, "4th": 3,
            "fifth": 4, "5th": 4,
            "last": -1, "final": -1,
        }

        # Check latest / newest with timestamp
        if any(w in t for w in ["latest", "newest", "most recent"]):
            if any("timestamp" in c for c in candidates):
                chosen = max(candidates, key=lambda c: c.get("timestamp", 0))
                self.candidate_cursor = self.search_candidates.index(chosen) if chosen in self.search_candidates else len(candidates) - 1
                self._update_selected_file(chosen)
                return chosen
            else:
                chosen = candidates[-1]
                self._update_selected_file(chosen)
                return chosen

        m_num = re.search(r"(?:#|number\s+)(\d+)", t)
        if m_num:
            idx = int(m_num.group(1)) - 1
            if 0 <= idx < len(candidates):
                chosen = candidates[idx]
                self.candidate_cursor = self.search_candidates.index(chosen) if chosen in self.search_candidates else idx
                self._update_selected_file(chosen)
                return chosen

        for ord_word, idx in ordinal_map.items():
            if re.search(rf"\b{re.escape(ord_word)}\b", t):
                if idx == -1:
                    chosen = candidates[-1]
                elif 0 <= idx < len(candidates):
                    chosen = candidates[idx]
                else:
                    continue
                self.candidate_cursor = self.search_candidates.index(chosen) if chosen in self.search_candidates else 0
                self._update_selected_file(chosen)
                return chosen

        return None

    def resolve_alternative(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Resolve alternative requests or corrections:
        e.g. "That isn't the one I meant; find the other version."
        e.g. "Not that one, the other one."
        """
        if not self.search_candidates or len(self.search_candidates) < 2:
            return None

        t = text.strip().lower()
        if any(phrase in t for phrase in [
            "that isn't the one", "not that one", "other version", "the other one",
            "different version", "not this", "wrong one", "find the other"
        ]):
            # Advance candidate cursor to next candidate
            self.candidate_cursor = (self.candidate_cursor + 1) % len(self.search_candidates)
            chosen = self.search_candidates[self.candidate_cursor]
            self._update_selected_file(chosen)
            return chosen

        return None

    def _update_selected_file(self, chosen: Dict[str, Any]):
        """Helper to update context pointers to newly selected file."""
        self.last_path = chosen.get("path")
        self.last_file = chosen.get("name") or (os.path.basename(chosen["path"]) if chosen.get("path") else None)

    def resolve_anaphora(self, text: str) -> str:
        """Resolve pronouns ('it', 'that', 'the file', 'there') to known context entities."""
        self.reset_if_expired()
        t = text.strip()
        t_lower = t.lower()

        # Discourse marker stripping ("Actually, ...", "No, ...")
        m_disc = re.match(r"^(?:actually|no|wait|instead|on second thought)\s*,\s*", t, re.IGNORECASE)
        if m_disc:
            t = t[m_disc.end():].strip()
            t_lower = t.lower()

        # 0. Check for alternative resolution ("find the other version", "not that one")
        if self.search_candidates and len(self.search_candidates) > 1:
            alt = self.resolve_alternative(t)
            if alt:
                alt_path = alt.get("path") or alt.get("name")
                if alt_path:
                    clean_alt = str(alt_path).replace("\\", "/")
                    return f"open '{clean_alt}'"

        # 1. First check ordinal resolution if we have prior search results
        if self.search_candidates:
            # Check format / extension disambiguation from candidate pool (supporting negation: "not the PDF, the Word file")
            negated_fmts = set(re.findall(r"\bnot\s+(?:the\s+)?(pdf|docx?|word|python|txt|excel|sheet)\b", t_lower))
            all_fmts = re.findall(r"\b(?:the\s+)?(pdf|docx?|word|python|txt|excel|sheet|presentation|pptx)\b", t_lower)
            positive_fmts = [f for f in all_fmts if f.lower() not in negated_fmts]
            if positive_fmts:
                req_type = positive_fmts[0].lower()
                target_ext = ".pdf" if "pdf" in req_type else (".docx" if ("word" in req_type or "doc" in req_type) else (".py" if "python" in req_type else ".txt"))
                matching = [c for c in self.search_candidates if c.get("extension", "").lower() == target_ext or c.get("path", "").lower().endswith(target_ext)]
                if matching:
                    chosen = matching[0]
                    self._update_selected_file(chosen)
                    clean_path = str(chosen.get("path") or chosen.get("name")).replace("\\", "/")
                    return f"open '{clean_path}'"

            resolved = self.resolve_ordinal(t)
            if resolved:
                target_path = resolved.get("path") or resolved.get("name")
                if target_path:
                    clean_path = str(target_path).replace("\\", "/")
                    if any(phrase in t_lower for phrase in ["not the old one", "not that one", "instead"]):
                        return f"open '{clean_path}'"

                    m_ord = re.search(
                        r"\b(?:the\s+)?(first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th|latest|newest|newer|last)(?:\s+(?:one|pdf|file|item|document|result|track))?\b",
                        t,
                        re.IGNORECASE
                    )
                    if m_ord:
                        t = t[:m_ord.start()] + f"'{clean_path}'" + t[m_ord.end():]
                        return t.strip()

        # 1b. Draft References & In-place Editing ("write that in notepad", "make that shorter", "go back to previous version")
        if hasattr(self, "draft_manager") and self.draft_manager.has_draft():
            if any(p in t_lower for p in ["write that in notepad", "put that in notepad", "insert that in notepad", "type that in notepad", "write that into notepad", "put that into notepad", "type that into notepad", "write it in notepad", "put it in notepad"]):
                return "write active draft into notepad"

            if any(p in t_lower for p in ["make that shorter", "make it shorter", "shorten it", "make it more formal", "make it professional", "make it more polite", "make it polite", "add this point", "add a sentence", "add this sentence", "fix the grammar"]):
                return f"revise draft: {t}"

            if any(p in t_lower for p in ["undo the last change", "undo last change", "go back to the previous version", "go back to previous version", "revert to previous version", "previous draft", "rollback"]):
                return "revert draft to previous version"

        # 1c. Plan continuation ("continue where we stopped", "finish the rest")
        if any(p in t_lower for p in ["continue where we stopped", "continue where we left off", "finish the rest", "continue the plan"]):
            return "continue active plan"

        # 2. Check destination references like 'there' (e.g. "move it there instead")
        if " there " in f" {t_lower} " or t_lower.endswith(" there") or t_lower.endswith(" there instead"):
            if self.last_destination:
                t = re.sub(r"\bthere(?:\s+instead)?\b", f"to '{self.last_destination}'", t, flags=re.IGNORECASE)

        # 3. Check App pronoun resolution (e.g. "close it")
        if self.last_app:
            m_close_it = re.match(r"^(?:close|quit|exit|kill)\s+(?:it|that|this)(?:\s+app|\s+window)?$", t_lower)
            if m_close_it:
                return f"close '{self.last_app}'"

        # 4. Check Song pronoun resolution (e.g. "play it on spotify")
        if self.last_song:
            if re.search(r"\bplay\s+(?:it|that|this)\b", t_lower):
                t = re.sub(r"\bplay\s+(?:it|that|this)\b", f"play '{self.last_song}'", t, flags=re.IGNORECASE)

        # 5. Check Project pronoun resolution ("now run it", "run tests on it")
        proj = self.last_project or self.active_project
        if proj:
            if "run tests on it" in t_lower or "run tests on that" in t_lower:
                return f"run tests on project '{proj}'"
            elif re.search(r"^(?:now\s+)?run\s+(?:it|that)\b", t_lower):
                return f"run project '{proj}'"

        # 6. Check File pronouns like 'it', 'that', 'this', 'that file'
        ref_path = self.last_path or self.last_file
        if ref_path:
            clean_ref = str(ref_path).replace("\\", "/")
            idiom_pattern = r"\b(?:what time is it|is it\b|who are you|how is it|make it (?:louder|quieter|brighter|dimmer|quick)|cut it out|leave it|step on it|take it easy|forget it)\b"
            if not re.search(idiom_pattern, t, re.IGNORECASE):
                # Pattern A: Action verb + pronoun e.g. "open it", "delete that", "save it in Documents", "open that one"
                verb_pronoun_pattern = re.compile(
                    r"\b(open|delete|remove|erase|trash|view|show|read|edit|modify|copy|move|rename|send|email|mail|execute|launch|print|save|put|place)\s+(?:up\s+)?(it|that|this)(?:\s+one)?\b",
                    re.IGNORECASE
                )
                if verb_pronoun_pattern.search(t):
                    t = verb_pronoun_pattern.sub(lambda m: f"{m.group(1)} '{clean_ref}'", t, count=1)
                else:
                    noun_phrase_pattern = re.compile(
                        r"\b(the\s+file|that\s+file|this\s+file|the\s+document|that\s+document)\b",
                        re.IGNORECASE
                    )
                    if noun_phrase_pattern.search(t):
                        t = noun_phrase_pattern.sub(lambda m: f"'{clean_ref}'", t, count=1)

        return t

    def resolve_references_structured(self, query: str) -> Tuple[str, List[Dict[str, Any]], bool]:
        """
        Comprehensive reference resolution returning:
        (resolved_query, list_of_detected_references, is_ambiguous)
        """
        references: List[Dict[str, Any]] = []
        is_ambiguous = False
        resolved_query = self.resolve_anaphora(query)

        # Check if query was resolved to an active draft operation
        if resolved_query.startswith("revise draft:") or resolved_query.startswith("write active draft") or resolved_query.startswith("revert draft"):
            return resolved_query, [{"raw_reference": "draft", "ref_type": "draft", "resolved_target": "active_draft", "is_resolved": True}], False

        # Search / Find commands describing items are local searches, not unresolved dangling references
        if re.match(r"^(?:find\s+file|find|search\s+for\s+file|search\s+file|search\s+for|locate\s+file|locate|look\s+for|where\s+did\s+i\s+save|where\s+is)\b", query.strip(), re.IGNORECASE):
            return resolved_query, [], False

        # Check if query contains non-referential dummy pronoun idioms
        idiom_pattern = r"\b(?:what time is it|is it\b|who are you|how is it|make it (?:louder|quieter|brighter|dimmer|quick)|cut it out|leave it|step on it|take it easy|forget it)\b"
        if re.search(idiom_pattern, resolved_query, re.IGNORECASE):
            return resolved_query, [], False

        # Check if query still contains unresolved references without prior context
        unresolved_tokens = re.findall(r"\b(it|that|the other one|the second one|that one|the file|the document|second result|first result)\b", resolved_query, re.IGNORECASE)
        for tok in unresolved_tokens:
            t_low = tok.lower()
            if t_low == "that":
                # Filter out 'that' when used as a subordinating conjunction in communication dictation
                if re.search(r"\bthat\s+(?:i\b|we\b|you\b|he\b|she\b|they\b|the\b|our\b|my\b|there\b|it\b|a\b)", resolved_query, re.IGNORECASE) or re.search(r"\b(?:draft|send|tell|message|notify|saying|state|stating)\b.+?\bthat\b", resolved_query, re.IGNORECASE):
                    continue

            if t_low in ["it", "that", "that one", "the file", "the document"] and not self.last_file and not self.last_path and not self.last_app and not (hasattr(self, "has_draft") and self.has_draft()):
                references.append({
                    "raw_reference": tok,
                    "ref_type": "pronoun",
                    "resolved_target": None,
                    "is_resolved": False
                })
                is_ambiguous = True
            elif ("other" in t_low or "second" in t_low or "first" in t_low or "result" in t_low) and not self.search_candidates:
                references.append({
                    "raw_reference": tok,
                    "ref_type": "ordinal",
                    "resolved_target": None,
                    "is_resolved": False
                })
                is_ambiguous = True
            elif "other" in t_low and len(self.search_candidates) < 2:
                references.append({
                    "raw_reference": tok,
                    "ref_type": "alternative",
                    "resolved_target": None,
                    "is_resolved": False
                })
                is_ambiguous = True

        return resolved_query, references, is_ambiguous


    def push_undo_action(self, action: str, details: Optional[Dict[str, Any]] = None):
        """Push an action into the undo stack for reversal."""
        self.touch()
        payload = {"action": action, "timestamp": time.time()}
        if details:
            payload.update(details)
        self.undo_stack.append(payload)
        if len(self.undo_stack) > 20:
            self.undo_stack = self.undo_stack[-20:]

    def undo_last_step(self) -> Optional[Dict[str, Any]]:
        """Pop and return the last undoable action directive from the undo stack."""
        if not self.undo_stack:
            return None
        last_action = self.undo_stack.pop()
        return last_action

    def to_dict(self) -> Dict[str, Any]:
        """Serialize context for debugging and inspection."""
        return {
            "active_goal": self.active_goal,
            "active_workflow": self.active_workflow,
            "active_step_index": self.active_step_index,
            "workflow_status": self.workflow_status,
            "active_project": self.active_project,
            "active_project_path": self.active_project_path,
            "last_app": self.last_app,
            "last_window": self.last_window,
            "last_service": self.last_service,
            "last_channel": self.last_channel,
            "last_file": self.last_file,
            "last_path": self.last_path,
            "last_destination": self.last_destination,
            "search_candidates_count": len(self.search_candidates),
            "candidate_cursor": self.candidate_cursor,
            "last_song": self.last_song,
            "last_artist": self.last_artist,
            "last_platform": self.last_platform,
            "last_media_action": self.last_media_action,
            "turn_count": len(self.turn_history),
            "undo_available": len(self.undo_stack) > 0,
        }


# Maintain backward-compatible class alias
ConversationContext = StructuredTaskContext
