"""Reusable Cross-Application Action Registry for E.D.I.T.H. V4.

Provides a unified, extensible capability registry across WhatsApp, Email,
Filesystem, Notepad, Browser, and Windows System tasks.
Every capability defines its schema, preconditions, authorization level,
execution delegate, verification method, and failure recovery.
"""

import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Tuple

from runtime.whatsapp_adapter import WhatsAppAdapter
from runtime.email_adapter import EmailAdapter
from runtime.browser_adapter import BrowserAdapter
from runtime.file_discovery import FileDiscoveryEngine
from runtime.text_writer import TextWritingPipeline
from runtime.contact_resolver import ContactResolver
from runtime.permission_guard import PermissionGuard
from runtime.result_verifier import CrossAppResultVerifier, VerifiedOutcome


@dataclass
class ActionCapability:
    """Specification of an application action capability."""
    name: str                           # Unique capability ID, e.g., 'whatsapp_send', 'file_summarize'
    category: str                       # communication, files, web, system, text
    description: str                    # Human-readable summary
    parameters_schema: Dict[str, str]   # Parameter names and descriptions
    preconditions: List[str]            # Requirements before execution
    expected_output: str                # What the action produces (e.g., 'message_submitted', 'summary_text')
    is_consequential: bool = False      # Whether it causes real external mutations
    requires_confirmation: bool = False # Whether explicit user authorization is mandatory
    execute_fn: Optional[Callable] = None


class ActionRegistry:
    """Central maintainable registry of all cross-application capabilities."""

    def __init__(
        self,
        contact_resolver: Optional[ContactResolver] = None,
        permission_guard: Optional[PermissionGuard] = None,
        file_discovery: Optional[FileDiscoveryEngine] = None,
        text_writer: Optional[TextWritingPipeline] = None
    ):
        self.contact_resolver = contact_resolver or ContactResolver()
        self.permission_guard = permission_guard or PermissionGuard()
        self.file_discovery = file_discovery or FileDiscoveryEngine()
        self.text_writer = text_writer or TextWritingPipeline()
        self.whatsapp_adapter = WhatsAppAdapter(contact_resolver=self.contact_resolver)
        self.email_adapter = EmailAdapter(contact_resolver=self.contact_resolver)
        self.browser_adapter = BrowserAdapter()
        self.capabilities: Dict[str, ActionCapability] = {}
        self._register_all_capabilities()

    def register(self, cap: ActionCapability):
        """Register a new action capability."""
        self.capabilities[cap.name] = cap

    def get_capability(self, name: str) -> Optional[ActionCapability]:
        """Lookup capability by name or alias."""
        clean = name.lower().strip()
        if clean in self.capabilities:
            return self.capabilities[clean]
        for cname, cap in self.capabilities.items():
            if clean in cname or cname in clean:
                return cap
        return None

    def _register_all_capabilities(self):
        """Register standard capabilities across all task categories."""

        # ─────────────────────────────────────────────────────────────────────
        # 1. COMMUNICATION: WhatsApp
        # ─────────────────────────────────────────────────────────────────────

        self.register(ActionCapability(
            name="whatsapp_draft",
            category="communication",
            description="Create or revise an editable WhatsApp draft for a contact or group without sending.",
            parameters_schema={
                "recipient": "Contact name, group name, or phone number",
                "content": "Intended message text or topic",
                "is_exact": "True if user verbatim dictated text"
            },
            preconditions=["recipient specified"],
            expected_output="draft_stored",
            is_consequential=False,
            requires_confirmation=False,
            execute_fn=lambda p, ctx: self.whatsapp_adapter.compose_draft(
                recipient_query=p.get("recipient", ""),
                message_topic_or_text=p.get("content", ""),
                context=ctx,
                is_exact=p.get("is_exact", False)
            )
        ))

        self.register(ActionCapability(
            name="whatsapp_revise",
            category="communication",
            description="Apply in-place edits (shorten, add sentence, undo) to the active WhatsApp draft.",
            parameters_schema={"instruction": "Editing instruction (e.g. 'make it shorter', 'add that...')"},
            preconditions=["active_draft exists"],
            expected_output="draft_updated",
            is_consequential=False,
            requires_confirmation=False,
            execute_fn=lambda p, ctx: self.whatsapp_adapter.revise_draft(p.get("instruction", ""))
        ))

        self.register(ActionCapability(
            name="whatsapp_preview",
            category="communication",
            description="Preview the current WhatsApp message draft before sending.",
            parameters_schema={},
            preconditions=["active_draft exists"],
            expected_output="draft_previewed",
            is_consequential=False,
            requires_confirmation=False,
            execute_fn=lambda p, ctx: self.whatsapp_adapter.preview_draft()
        ))

        self.register(ActionCapability(
            name="whatsapp_send",
            category="communication",
            description="Send a message to a contact or group via WhatsApp with authorization checks.",
            parameters_schema={
                "recipient": "Contact name or phone number",
                "content": "Message content",
                "authorized": "Explicit authorization boolean",
                "is_exact": "Verbatim user text flag",
                "attachment_path": "Optional file path to share"
            },
            preconditions=["recipient resolved", "authorized if substantive text generated"],
            expected_output="message_submitted",
            is_consequential=True,
            requires_confirmation=True,
            execute_fn=lambda p, ctx: self.whatsapp_adapter.send_message(
                recipient_query=p.get("recipient"),
                text_or_prompt=p.get("content"),
                context=ctx,
                authorized=p.get("authorized", False),
                is_exact=p.get("is_exact", False),
                attachment_path=p.get("attachment_path")
            )
        ))

        # ─────────────────────────────────────────────────────────────────────
        # 2. COMMUNICATION: Email
        # ─────────────────────────────────────────────────────────────────────

        self.register(ActionCapability(
            name="email_draft",
            category="communication",
            description="Draft an email for a contact or role with optional subject and attachment.",
            parameters_schema={
                "recipient": "Contact name or email",
                "topic": "Email topic or dictated body",
                "subject": "Optional custom subject",
                "attachment_path": "Optional file to attach"
            },
            preconditions=["recipient specified"],
            expected_output="email_draft_stored",
            is_consequential=False,
            requires_confirmation=False,
            execute_fn=lambda p, ctx: self.email_adapter.compose_draft(
                recipient_query=p.get("recipient", ""),
                topic_or_text=p.get("topic", "") or p.get("content", ""),
                subject=p.get("subject"),
                attachment_path=p.get("attachment_path"),
                context=ctx,
                is_exact=p.get("is_exact", False)
            )
        ))

        self.register(ActionCapability(
            name="email_revise",
            category="communication",
            description="Revise an active email draft in-place.",
            parameters_schema={"instruction": "Editing instruction"},
            preconditions=["active_draft exists"],
            expected_output="email_draft_updated",
            is_consequential=False,
            requires_confirmation=False,
            execute_fn=lambda p, ctx: self.email_adapter.revise_draft(p.get("instruction", ""))
        ))

        self.register(ActionCapability(
            name="email_send",
            category="communication",
            description="Send an email via authenticated mail client with confirmation gating.",
            parameters_schema={
                "recipient": "Recipient name or email",
                "content": "Email body",
                "subject": "Subject line",
                "attachment_path": "Optional attachment path",
                "authorized": "Authorization flag"
            },
            preconditions=["recipient resolved", "authorized"],
            expected_output="email_dispatched",
            is_consequential=True,
            requires_confirmation=True,
            execute_fn=lambda p, ctx: self.email_adapter.send_email(
                recipient_query=p.get("recipient"),
                topic_or_text=p.get("content") or p.get("topic"),
                subject=p.get("subject"),
                attachment_path=p.get("attachment_path"),
                context=ctx,
                authorized=p.get("authorized", False),
                is_exact=p.get("is_exact", False)
            )
        ))

        # ─────────────────────────────────────────────────────────────────────
        # 3. FILES & DOCUMENTS
        # ─────────────────────────────────────────────────────────────────────

        self.register(ActionCapability(
            name="file_find",
            category="files",
            description="Find files across workspace and personal directories using natural language.",
            parameters_schema={"query": "Search query or natural description", "location_hint": "Optional folder hint"},
            preconditions=[],
            expected_output="file_discovered",
            is_consequential=False,
            requires_confirmation=False,
            execute_fn=lambda p, ctx: self.file_discovery.find_files(
                query=p.get("query", ""),
                location_hint=p.get("location_hint")
            )
        ))

        self.register(ActionCapability(
            name="file_summarize",
            category="files",
            description="Read an accessible document (.txt, .md, .py, .json, .docx, .pdf) and generate an extractive summary.",
            parameters_schema={"path": "Path to document", "max_points": "Maximum bullet points"},
            preconditions=["file exists"],
            expected_output="summary_generated",
            is_consequential=False,
            requires_confirmation=False,
            execute_fn=lambda p, ctx: self.file_discovery.summarize_document(
                file_path=p.get("path", "") or p.get("file", ""),
                max_points=int(p.get("max_points", 4))
            )
        ))

        self.register(ActionCapability(
            name="file_rename",
            category="files",
            description="Safely rename a file with destination conflict checking.",
            parameters_schema={"source": "Current file path", "new_name": "New name or path"},
            preconditions=["source exists", "destination not already taken"],
            expected_output="file_renamed",
            is_consequential=True,
            requires_confirmation=False,
            execute_fn=lambda p, ctx: self.file_discovery.rename_file(
                source_path=p.get("source", ""),
                new_name_or_path=p.get("new_name", "")
            )
        ))

        self.register(ActionCapability(
            name="file_copy",
            category="files",
            description="Safely copy a file to a destination with overwrite protection.",
            parameters_schema={"source": "Source path", "destination": "Target path", "overwrite": "Boolean"},
            preconditions=["source exists"],
            expected_output="file_copied",
            is_consequential=True,
            requires_confirmation=False,
            execute_fn=lambda p, ctx: self.file_discovery.copy_file(
                source_path=p.get("source", ""),
                destination_path=p.get("destination", ""),
                overwrite=p.get("overwrite", False)
            )
        ))

        self.register(ActionCapability(
            name="file_move",
            category="files",
            description="Safely move a file to a new destination folder.",
            parameters_schema={"source": "Source path", "destination": "Target path", "overwrite": "Boolean"},
            preconditions=["source exists"],
            expected_output="file_moved",
            is_consequential=True,
            requires_confirmation=False,
            execute_fn=lambda p, ctx: self.file_discovery.move_file(
                source_path=p.get("source", ""),
                destination_path=p.get("destination", ""),
                overwrite=p.get("overwrite", False)
            )
        ))

        self.register(ActionCapability(
            name="file_organize",
            category="files",
            description="Organize unorganized loose files in a folder into categorized subfolders (Documents, Images, Code, etc.).",
            parameters_schema={"folder": "Folder path"},
            preconditions=["folder exists"],
            expected_output="folder_organized",
            is_consequential=True,
            requires_confirmation=True,
            execute_fn=lambda p, ctx: self.file_discovery.organize_folder(p.get("folder", ""))
        ))

        # ─────────────────────────────────────────────────────────────────────
        # 4. NOTEPAD & TEXT
        # ─────────────────────────────────────────────────────────────────────

        self.register(ActionCapability(
            name="notepad_write",
            category="text",
            description="Insert Unicode text into Windows Notepad with read-back verification.",
            parameters_schema={"content": "Text to insert", "append": "Boolean flag for append vs replace"},
            preconditions=[],
            expected_output="notepad_written",
            is_consequential=False,
            requires_confirmation=False,
            execute_fn=lambda p, ctx: self.text_writer.insert_into_notepad(
                content=p.get("content", "") or p.get("text", ""),
                append=p.get("append", False)
            )
        ))

        # ─────────────────────────────────────────────────────────────────────
        # 5. BROWSER & WEB
        # ─────────────────────────────────────────────────────────────────────

        self.register(ActionCapability(
            name="browser_open",
            category="web",
            description="Open a specified website or web service in default browser.",
            parameters_schema={"target": "URL or popular service name"},
            preconditions=[],
            expected_output="browser_navigated",
            is_consequential=False,
            requires_confirmation=False,
            execute_fn=lambda p, ctx: self.browser_adapter.open_page(p.get("target", ""))
        ))

        self.register(ActionCapability(
            name="browser_search_and_summarize",
            category="web",
            description="Search online and extract a synthesized factual summary of actual findings.",
            parameters_schema={"query": "Search query"},
            preconditions=["query provided"],
            expected_output="search_summarized",
            is_consequential=False,
            requires_confirmation=False,
            execute_fn=lambda p, ctx: self.browser_adapter.search_and_summarize(p.get("query", ""))
        ))

        self.register(ActionCapability(
            name="browser_fill_form",
            category="web",
            description="Prepare form fields for inspection with confirmation gating on submission.",
            parameters_schema={"fields": "Dict of field values", "form_name": "Form title", "submit": "Boolean"},
            preconditions=["fields provided"],
            expected_output="form_prepared",
            is_consequential=True,
            requires_confirmation=True,
            execute_fn=lambda p, ctx: self.browser_adapter.prepare_form_data(
                form_data=p.get("fields", {}),
                target_form_name=p.get("form_name"),
                submit=p.get("submit", False),
                authorized=p.get("authorized", False)
            )
        ))

    def execute(self, capability_name: str, parameters: Dict[str, Any], context: Optional[Any] = None) -> Tuple[Dict[str, Any], VerifiedOutcome]:
        """
        Execute an action capability through safety gating and result verification.
        """
        cap = self.get_capability(capability_name)
        if not cap:
            res = {"status": "error", "message": f"Action capability '{capability_name}' is not registered."}
            ver = CrossAppResultVerifier.verify_action(capability_name, res, parameters)
            return res, ver

        # 1. Safety & Authorization Check
        is_exact = parameters.get("is_exact", False)
        user_confirmed = parameters.get("authorized", False) or parameters.get("confirmed", False)
        auth_ok, auth_prompt = self.permission_guard.is_authorized(
            action_type=cap.name,
            parameters=parameters,
            is_exact_dictation=is_exact,
            user_confirmed=user_confirmed
        )

        if not auth_ok:
            # Register pending action and request confirmation
            act = self.permission_guard.register_pending_action(
                action_type=cap.name,
                parameters=parameters,
                summary=auth_prompt
            )
            res = {
                "status": "needs_confirmation",
                "action": cap.name,
                "requires_confirmation": True,
                "action_id": act.action_id,
                "message": auth_prompt
            }
            ver = CrossAppResultVerifier.verify_action(cap.name, res, parameters)
            return res, ver

        # 2. Execute capability delegate
        try:
            if cap.execute_fn:
                res = cap.execute_fn(parameters, context)
            else:
                res = {"status": "error", "message": f"No execution delegate registered for '{cap.name}'."}
        except Exception as e:
            res = {"status": "error", "message": f"Execution of '{cap.name}' failed with exception: {e}"}

        # 3. Verify Observable System State
        ver = CrossAppResultVerifier.verify_action(cap.name, res, parameters)
        return res, ver
