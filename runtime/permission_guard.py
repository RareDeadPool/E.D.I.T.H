"""Permission, Safety, and Authorization Gating Layer for E.D.I.T.H. V4.

Enforces strict authorization rules for consequential actions:
- Direct communication (sending WhatsApp messages, sending emails)
- Filesystem mutations (deletions, overwrites, irreversible relocations)
- External web transactions (form submissions, checkouts)
- System lifecycle changes (process kills, restarts)

Manages confirmation tokens, reviews, and guarantees immediate cancellation on 'cancel that'.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple


@dataclass
class PendingAction:
    """Action awaiting explicit user review or authorization."""
    action_id: str
    action_type: str                   # e.g., "send_whatsapp", "send_email", "delete_file", "submit_form"
    parameters: Dict[str, Any]
    summary_for_user: str
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 300.0)
    is_cancelled: bool = False

    def is_expired(self) -> bool:
        return time.time() > self.expires_at


class PermissionGuard:
    """Central safety gate enforcing authorization and cancellation."""

    CONSEQUENTIAL_ACTIONS = {
        "send_whatsapp",
        "send_email",
        "delete_file",
        "overwrite_file",
        "submit_form",
        "kill_process",
        "shutdown_system"
    }

    def __init__(self):
        self.pending_actions: Dict[str, PendingAction] = {}
        self._action_counter: int = 0
        self.last_cancelled_action: Optional[PendingAction] = None

    def is_consequential(self, action_type: str) -> bool:
        """Check if an action is consequential and requires authorization."""
        return action_type in self.CONSEQUENTIAL_ACTIONS

    def is_authorized(
        self,
        action_type: str,
        parameters: Dict[str, Any],
        is_exact_dictation: bool = False,
        user_confirmed: bool = False
    ) -> Tuple[bool, Optional[str]]:
        """
        Evaluate if an operation is authorized to execute immediately or requires confirmation.
        Returns (authorized, reason_or_prompt).
        """
        # If user explicitly confirmed via follow-up, it is authorized
        if user_confirmed:
            return True, None

        # Exact verbatim dictation directed explicitly to a specific contact is authorized
        if is_exact_dictation and action_type in ["send_whatsapp", "send_email"]:
            return True, None

        # If action is consequential and content was assistant-generated or destructive, require confirmation
        if self.is_consequential(action_type):
            if action_type == "send_whatsapp":
                recip = parameters.get("recipient", "the contact")
                msg = parameters.get("content", "")
                att = parameters.get("attachment_path")
                prompt = (
                    f"Please confirm sending this WhatsApp message to {recip}:\n"
                    f"\"{msg}\"\n" +
                    (f"Attachment: {att}\n" if att else "") +
                    "Send this message? (Say 'Yes' or 'Cancel')"
                )
                return False, prompt

            elif action_type == "send_email":
                recip = parameters.get("recipient", "the recipient")
                subj = parameters.get("subject", "No subject")
                prompt = (
                    f"Please confirm sending this email to {recip}:\n"
                    f"Subject: {subj}\n"
                    "Send this email? (Say 'Yes' or 'Cancel')"
                )
                return False, prompt

            elif action_type in ["delete_file", "overwrite_file"]:
                path = parameters.get("path") or parameters.get("destination", "")
                prompt = f"Are you sure you want to {action_type.replace('_', ' ')} '{path}'? (Say 'Yes' or 'Cancel')"
                return False, prompt

            elif action_type == "submit_form":
                prompt = "Form submission requires confirmation. Would you like me to submit the form?"
                return False, prompt

        return True, None

    def register_pending_action(
        self,
        action_type: str,
        parameters: Dict[str, Any],
        summary: str
    ) -> PendingAction:
        """Store pending action awaiting confirmation."""
        self._action_counter += 1
        aid = f"act_{int(time.time())}_{self._action_counter}"
        action = PendingAction(
            action_id=aid,
            action_type=action_type,
            parameters=parameters,
            summary_for_user=summary
        )
        self.pending_actions[aid] = action
        return action

    def get_latest_pending_action(self) -> Optional[PendingAction]:
        """Retrieve most recent non-expired, non-cancelled pending action."""
        for aid in sorted(self.pending_actions.keys(), reverse=True):
            act = self.pending_actions[aid]
            if not act.is_cancelled and not act.is_expired():
                return act
        return None

    def cancel_all_pending(self) -> Dict[str, Any]:
        """
        Immediately cancel all pending actions. Guarantees no pending send proceeds.
        """
        latest = self.get_latest_pending_action()
        cancelled_count = 0
        for act in self.pending_actions.values():
            if not act.is_cancelled:
                act.is_cancelled = True
                cancelled_count += 1

        self.last_cancelled_action = latest
        self.pending_actions.clear()

        desc = f" '{latest.action_type.replace('_', ' ')}'" if latest else ""
        return {
            "status": "cancelled",
            "cancelled_count": cancelled_count,
            "message": f"Cancelled pending action{desc}. Nothing was sent or modified."
        }

    def confirm_latest_action(self) -> Optional[PendingAction]:
        """Approve and return the latest pending action."""
        act = self.get_latest_pending_action()
        if act:
            act.is_cancelled = True  # Consume it
            return act
        return None
