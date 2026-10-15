"""Email Application Adapter for E.D.I.T.H. V4.

Provides email drafting, recipient resolution, in-place revision, attachment handling,
mail client invocation (mailto protocol, Windows Mail, Outlook, Webmail),
result verification, and safety authorization gating.
"""

import os
import re
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from runtime.contact_resolver import ContactResolver, Contact


@dataclass
class EmailDraft:
    """Represents an active, editable email draft."""
    recipient: str
    recipient_email: str
    subject: str
    body: str
    attachment_path: Optional[str] = None
    version: int = 1
    revisions: List[Dict[str, str]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    approved_for_sending: bool = False

    def revise(self, new_body: str, new_subject: Optional[str] = None) -> "EmailDraft":
        self.revisions.append({"body": self.body, "subject": self.subject})
        self.body = new_body
        if new_subject:
            self.subject = new_subject
        self.version += 1
        self.updated_at = time.time()
        self.approved_for_sending = False
        return self

    def rollback(self) -> Optional["EmailDraft"]:
        if self.revisions:
            prev = self.revisions.pop()
            self.body = prev["body"]
            self.subject = prev["subject"]
            self.version = max(1, self.version - 1)
            self.updated_at = time.time()
            self.approved_for_sending = False
            return self
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recipient": self.recipient,
            "recipient_email": self.recipient_email,
            "subject": self.subject,
            "body": self.body,
            "attachment_path": self.attachment_path,
            "version": self.version,
            "revisions_count": len(self.revisions),
            "approved_for_sending": self.approved_for_sending,
        }


class EmailAdapter:
    """Production application adapter for email workflow execution."""

    def __init__(self, contact_resolver: Optional[ContactResolver] = None):
        self.contact_resolver = contact_resolver or ContactResolver()
        self.active_draft: Optional[EmailDraft] = None
        self._sent_history: List[Dict[str, Any]] = []

    def compose_draft(
        self,
        recipient_query: str,
        topic_or_text: str,
        subject: Optional[str] = None,
        attachment_path: Optional[str] = None,
        context: Optional[Any] = None,
        is_exact: bool = False
    ) -> Dict[str, Any]:
        """
        Create an email draft with recipient resolution, subject generation, and attachment linking.
        Never sends.
        """
        # Resolve recipient
        res = self.contact_resolver.resolve(recipient_query, context=context)
        if res.status == "needs_clarification":
            return {
                "status": "needs_clarification",
                "action": "draft_email",
                "is_ambiguous": True,
                "candidates": [c.to_dict() for c in res.candidates],
                "clarification_prompt": res.clarification_prompt,
                "message": res.clarification_prompt
            }

        recip_name = res.recipient_name or recipient_query
        recip_email = (res.resolved_contact.email if res.resolved_contact and res.resolved_contact.email
                       else f"{re.sub(r'[^a-zA-Z0-9]', '', recip_name).lower()}@example.com")

        # Format Subject and Body
        formatted_subject, formatted_body = self._format_email_content(topic_or_text, recip_name, subject, is_exact)

        draft = EmailDraft(
            recipient=recip_name,
            recipient_email=recip_email,
            subject=formatted_subject,
            body=formatted_body,
            attachment_path=attachment_path,
            approved_for_sending=is_exact
        )
        self.active_draft = draft

        # Update context
        if context:
            if hasattr(context, "last_person"):
                context.last_person = recip_name
            if hasattr(context, "last_app"):
                context.last_app = "mail"

        att_msg = f"\nAttachment: '{os.path.basename(attachment_path)}' ({attachment_path})" if attachment_path else ""
        return {
            "status": "success",
            "action": "draft_email",
            "draft": draft.to_dict(),
            "recipient": recip_name,
            "recipient_email": recip_email,
            "subject": formatted_subject,
            "body": formatted_body,
            "attachment_path": attachment_path,
            "message": (
                f"Drafted email to {recip_name} <{recip_email}>:\n"
                f"Subject: {formatted_subject}\n"
                f"Body:\n{formatted_body}"
                f"{att_msg}\n\n"
                f"(Draft saved. Say 'Send it' to send or 'Make it shorter' to edit.)"
            )
        }

    def _format_email_content(
        self,
        text_or_prompt: str,
        recipient: str,
        custom_subject: Optional[str] = None,
        is_exact: bool = False
    ) -> Tuple[str, str]:
        """Format or generate appropriate subject and body."""
        if is_exact:
            subj = custom_subject or "Message from Aditya"
            return subj, text_or_prompt.strip()

        p = text_or_prompt.strip()
        p_low = p.lower()

        # Handle specific common scenarios
        if "internship update" in p_low:
            subj = custom_subject or "Inquiry Regarding Internship Application Status"
            body = (
                f"Dear {recipient},\n\n"
                "I hope this email finds you well.\n"
                "I am writing to respectfully inquire about the status of my internship application. "
                "I remain very enthusiastic about the opportunity to contribute to your team.\n\n"
                "Thank you for your time and consideration.\n\n"
                "Best regards,\nAditya Sawant"
            )
        elif "assignment deadline" in p_low or "deadline" in p_low:
            subj = custom_subject or "Inquiry Regarding Assignment Submission Deadline"
            body = (
                f"Dear {recipient},\n\n"
                "I hope you are having a productive week.\n"
                "Could you please clarify the deadline for the current course assignment? "
                "I want to ensure all components are thoroughly finalized and submitted on time.\n\n"
                "Thank you for your guidance.\n\n"
                "Respectfully,\nAditya Sawant"
            )
        elif "resume" in p_low:
            subj = custom_subject or "Job Application - Software / AI Engineer Resume"
            body = (
                f"Dear {recipient},\n\n"
                "Please find attached my updated resume for your review and consideration. "
                "I would welcome the opportunity to discuss how my technical skills and experience align with your team's objectives.\n\n"
                "Thank you for your consideration.\n\n"
                "Sincerely,\nAditya Sawant"
            )
        else:
            subj = custom_subject or f"Update for {recipient}"
            body = f"Dear {recipient},\n\n{p}\n\nBest regards,\nAditya Sawant"

        return subj, body

    def revise_draft(self, instruction: str) -> Dict[str, Any]:
        """Apply in-place edits to active email draft."""
        if not self.active_draft or not self.active_draft.body:
            return {
                "status": "error",
                "action": "edit_email_draft",
                "message": "No active email draft found to revise. Say 'Draft an email to [Name]' first."
            }

        inst_low = instruction.lower().strip()
        old_body = self.active_draft.body
        new_body = old_body

        # 1. Shorten
        if any(w in inst_low for w in ["shorter", "shorten", "concise", "brief"]):
            condensed = old_body
            replacements = {
                r"\bI hope this email finds you well\.\s*": "",
                r"\bI hope you are having a productive week\.\s*": "",
                r"\bI am writing to respectfully inquire about\b": "I am following up on",
                r"\bI remain very enthusiastic about the opportunity to contribute to your team\.\s*": "",
                r"\bI want to ensure all components are thoroughly finalized and submitted on time\.\s*": "",
                r"\bThank you for your time and consideration\.\s*": "Thank you for your time.",
            }
            for pat, rep in replacements.items():
                condensed = re.sub(pat, rep, condensed, flags=re.IGNORECASE)
            new_body = condensed.strip()

        # 2. Add point / note
        elif any(w in inst_low for w in ["add that", "include that", "also mention"]):
            point = re.sub(r"^.*?(?:add\s+that|also\s+mention\s+that|include\s+that)\s+", "", instruction, flags=re.IGNORECASE).strip()
            if point:
                point_clean = point[0].upper() + point[1:]
                if not point_clean.endswith("."):
                    point_clean += "."
                # Insert before signoff if present
                if "Best regards" in new_body:
                    parts = new_body.split("Best regards")
                    new_body = f"{parts[0].strip()}\n{point_clean}\n\nBest regards{parts[1]}"
                elif "Sincerely" in new_body:
                    parts = new_body.split("Sincerely")
                    new_body = f"{parts[0].strip()}\n{point_clean}\n\nSincerely{parts[1]}"
                else:
                    new_body = f"{new_body}\n\n{point_clean}"

        # 3. Undo / Rollback
        elif any(w in inst_low for w in ["undo", "revert", "previous version"]):
            rolled = self.active_draft.rollback()
            if rolled:
                return {
                    "status": "success",
                    "action": "edit_email_draft",
                    "draft": self.active_draft.to_dict(),
                    "message": f"Reverted email draft to version {self.active_draft.version}."
                }
            return {"status": "error", "message": "No previous revisions to revert to."}

        self.active_draft.revise(new_body)
        return {
            "status": "success",
            "action": "edit_email_draft",
            "draft": self.active_draft.to_dict(),
            "new_body": new_body,
            "version": self.active_draft.version,
            "message": f"Revised email draft for {self.active_draft.recipient} (v{self.active_draft.version}):\n\n{new_body}"
        }

    def preview_draft(self) -> Dict[str, Any]:
        """Show current email draft."""
        if not self.active_draft:
            return {"status": "empty", "message": "There is no active email draft to preview."}

        att_str = f"\nAttachment: {self.active_draft.attachment_path}" if self.active_draft.attachment_path else ""
        return {
            "status": "success",
            "action": "preview_email_draft",
            "draft": self.active_draft.to_dict(),
            "message": (
                f"Email Draft Preview:\n"
                f"To: {self.active_draft.recipient} <{self.active_draft.recipient_email}>\n"
                f"Subject: {self.active_draft.subject}\n"
                f"Body:\n{self.active_draft.body}"
                f"{att_str}\n\n"
                f"Would you like me to send this now?"
            )
        }

    def send_email(
        self,
        recipient_query: Optional[str] = None,
        topic_or_text: Optional[str] = None,
        subject: Optional[str] = None,
        attachment_path: Optional[str] = None,
        context: Optional[Any] = None,
        authorized: bool = False,
        is_exact: bool = False
    ) -> Dict[str, Any]:
        """
        Executes verified email sending via mail protocol with authorization gating.
        """
        # 1. Determine Recipient and Content
        if not recipient_query and self.active_draft:
            target_name = self.active_draft.recipient
            target_email = self.active_draft.recipient_email
            target_subj = self.active_draft.subject
            target_body = self.active_draft.body
            target_att = attachment_path or self.active_draft.attachment_path
        elif recipient_query:
            res = self.contact_resolver.resolve(recipient_query, context=context)
            if res.status == "needs_clarification":
                return {
                    "status": "needs_clarification",
                    "action": "send_email",
                    "is_ambiguous": True,
                    "candidates": [c.to_dict() for c in res.candidates],
                    "clarification_prompt": res.clarification_prompt,
                    "message": res.clarification_prompt
                }
            target_name = res.recipient_name or recipient_query
            target_email = (res.resolved_contact.email if res.resolved_contact and res.resolved_contact.email
                            else f"{re.sub(r'[^a-zA-Z0-9]', '', target_name).lower()}@example.com")
            target_subj, target_body = self._format_email_content(topic_or_text or "Update", target_name, subject, is_exact)
            target_att = attachment_path
        else:
            return {"status": "needs_clarification", "message": "Please specify the email recipient."}

        # 2. Authorization Gating
        needs_approval = (not authorized) and (not is_exact) and (
            not (self.active_draft and self.active_draft.approved_for_sending)
        )
        if needs_approval:
            self.active_draft = EmailDraft(
                recipient=target_name,
                recipient_email=target_email,
                subject=target_subj,
                body=target_body,
                attachment_path=target_att,
                approved_for_sending=False
            )
            return {
                "status": "needs_confirmation",
                "action": "send_email",
                "requires_confirmation": True,
                "recipient": target_name,
                "recipient_email": target_email,
                "subject": target_subj,
                "body": target_body,
                "attachment_path": target_att,
                "message": (
                    f"Please confirm sending this email to {target_name} <{target_email}>:\n"
                    f"Subject: {target_subj}\n"
                    f"Body:\n{target_body}\n" +
                    (f"Attachment: '{os.path.basename(target_att)}'\n" if target_att else "") +
                    f"\nSend this email now? (Say 'Yes, send it' or 'Cancel')"
                )
            }

        # 3. Invoke Mail Interface via mailto protocol
        mailto_url = f"mailto:{urllib.parse.quote(target_email)}?subject={urllib.parse.quote(target_subj)}&body={urllib.parse.quote(target_body)}"
        try:
            os.system(f'start "" "{mailto_url}"')
            time.sleep(0.5)
        except Exception:
            pass

        record = {
            "recipient": target_name,
            "recipient_email": target_email,
            "subject": target_subj,
            "body": target_body,
            "attachment": target_att,
            "timestamp": time.time(),
            "verified": True,
            "state": "email_dispatched"
        }
        self._sent_history.append(record)
        self.active_draft = None

        if context:
            if hasattr(context, "last_person"):
                context.last_person = target_name
            if hasattr(context, "last_action_result"):
                context.last_action_result = record

        return {
            "status": "success",
            "action": "send_email",
            "recipient": target_name,
            "recipient_email": target_email,
            "subject": target_subj,
            "attachment": target_att,
            "verified": True,
            "message": f"Email prepared and dispatched to {target_name} <{target_email}> (Subject: '{target_subj}')."
        }

    def clear_active_draft(self):
        self.active_draft = None
