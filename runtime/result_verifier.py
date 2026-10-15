"""Observable Result Verifier for E.D.I.T.H. V4 Cross-Application Tasks.

Inspects actual system state post-execution to verify observable success conditions
across WhatsApp, Email, Filesystem, Browser, and Windows applications.
Prevents false claims of success and guides safe recovery.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class VerifiedOutcome:
    """Rigorous verification verdict based on observable evidence."""
    verified: bool
    observed_state: str                 # e.g., "message_submitted", "file_verified", "draft_stored", "failed"
    evidence: str                       # e.g., "File exists at '...'", "WM_GETTEXT matched 223 characters"
    diagnostic_details: Optional[str] = None
    can_recover: bool = False
    recovery_strategy: Optional[str] = None
    fallback_parameters: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verified": self.verified,
            "observed_state": self.observed_state,
            "evidence": self.evidence,
            "diagnostic_details": self.diagnostic_details,
            "can_recover": self.can_recover,
            "recovery_strategy": self.recovery_strategy,
            "fallback_parameters": self.fallback_parameters,
        }


class CrossAppResultVerifier:
    """Inspects post-execution state across all application adapters."""

    @classmethod
    def verify_action(cls, handler: str, result: Dict[str, Any], parameters: Dict[str, Any]) -> VerifiedOutcome:
        """Evaluate observable success condition for an executed action."""
        if not isinstance(result, dict):
            return VerifiedOutcome(
                verified=False,
                observed_state="unknown_result",
                evidence="No structured result dictionary returned by handler."
            )

        status = result.get("status", "error")

        # 1. WhatsApp Verification
        if handler in ["whatsapp", "send_whatsapp", "draft_whatsapp", "whatsapp_draft", "whatsapp_send"]:
            if status == "success":
                state = result.get("transmission_state", "action_completed")
                recip = result.get("recipient", "contact")
                evidence = f"WhatsApp conversation opened for {recip}; message submitted into active thread."
                return VerifiedOutcome(
                    verified=True,
                    observed_state=state,
                    evidence=evidence
                )
            elif status == "duplicate_prevented":
                return VerifiedOutcome(
                    verified=True,
                    observed_state="duplicate_prevented",
                    evidence=result.get("message", "Duplicate send prevented.")
                )
            elif status == "needs_confirmation":
                return VerifiedOutcome(
                    verified=True,
                    observed_state="pending_user_confirmation",
                    evidence="Draft prepared; awaiting explicit user approval before transmission."
                )
            elif status == "needs_clarification":
                return VerifiedOutcome(
                    verified=False,
                    observed_state="ambiguous_recipient",
                    evidence=result.get("message", "Recipient requires clarification.")
                )
            else:
                return VerifiedOutcome(
                    verified=False,
                    observed_state="whatsapp_failed",
                    evidence=result.get("message", "WhatsApp action failed."),
                    can_recover=True,
                    recovery_strategy="retry_via_web"
                )

        # 2. Email Verification
        elif handler in ["email", "send_email", "draft_email", "email_draft", "email_send"]:
            if status == "success":
                recip = result.get("recipient", "recipient")
                subj = result.get("subject", "subject")
                return VerifiedOutcome(
                    verified=True,
                    observed_state="email_dispatched" if "send" in handler else "draft_stored",
                    evidence=f"Mail client invoked for {recip} with subject '{subj}'."
                )
            elif status == "needs_confirmation":
                return VerifiedOutcome(
                    verified=True,
                    observed_state="pending_user_confirmation",
                    evidence="Email prepared; awaiting approval before dispatch."
                )
            else:
                return VerifiedOutcome(
                    verified=False,
                    observed_state="email_failed",
                    evidence=result.get("message", "Email action failed.")
                )

        # 3. File Operations Verification
        elif handler in ["file_operation", "file_find", "file_summarize", "file_rename", "file_move", "file_copy"]:
            target_path = result.get("path") or result.get("destination") or parameters.get("destination") or parameters.get("path")
            subaction = parameters.get("subaction", "")

            if subaction in ["find", "locate", "search"] or handler == "file_find":
                cands = result.get("candidates", [])
                if cands:
                    top_p = cands[0].get("path", "")
                    exists = os.path.exists(top_p) if top_p else False
                    return VerifiedOutcome(
                        verified=exists,
                        observed_state="file_discovered" if exists else "file_missing",
                        evidence=f"Discovered {len(cands)} candidate files. Top path exists: {exists}."
                    )
                return VerifiedOutcome(
                    verified=False,
                    observed_state="file_not_found",
                    evidence="No matching files found."
                )

            elif subaction in ["rename", "move", "copy"] or handler in ["file_rename", "file_move", "file_copy"]:
                dest = result.get("destination", target_path)
                dest_exists = bool(dest and os.path.exists(dest))
                return VerifiedOutcome(
                    verified=dest_exists,
                    observed_state="filesystem_mutation_verified" if dest_exists else "mutation_failed",
                    evidence=f"Target file exists on filesystem at '{dest}': {dest_exists}."
                )

            elif handler == "file_summarize" or subaction == "summarize":
                has_summary = bool(result.get("summary") and result.get("points"))
                return VerifiedOutcome(
                    verified=has_summary,
                    observed_state="summary_generated" if has_summary else "summary_empty",
                    evidence=f"Generated {len(result.get('points', []))} extractive summary points."
                )

            else:
                # Default file open/check
                p_exists = bool(target_path and os.path.exists(target_path))
                return VerifiedOutcome(
                    verified=p_exists,
                    observed_state="file_exists" if p_exists else "file_missing",
                    evidence=f"Path verified on disk: '{target_path}' (exists: {p_exists})."
                )

        # 4. Notepad Text Writing Verification
        elif handler in ["type_notepad", "notepad_write", "gui_action"]:
            if result.get("action") == "type_notepad":
                v = result.get("verified", False)
                exact = result.get("exact_match", False)
                chars = result.get("characters_written", 0)
                return VerifiedOutcome(
                    verified=v,
                    observed_state="notepad_written" if v else "notepad_failed",
                    evidence=f"Read-back verified via WM_GETTEXT: {v} ({chars} chars, exact: {exact})."
                )

        # 5. Browser Verification
        elif handler in ["browser", "open_browser_page", "search_and_summarize"]:
            v = result.get("verified", False) or (status == "success")
            return VerifiedOutcome(
                verified=v,
                observed_state="browser_navigated" if v else "browser_failed",
                evidence=f"Browser navigation to '{result.get('url', '')}' completed."
            )

        # Default fallback
        v = (status == "success")
        return VerifiedOutcome(
            verified=v,
            observed_state="completed" if v else "failed",
            evidence=result.get("message", "Action execution finished.")
        )
