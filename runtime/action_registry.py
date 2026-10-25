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

        self.register(ActionCapability(
            name="browser_read_page",
            category="web",
            description="Fetch a web page or article URL and generate a synthesized summary.",
            parameters_schema={"url": "Website or article URL"},
            preconditions=["url provided"],
            expected_output="summary_generated",
            is_consequential=False,
            requires_confirmation=False,
            execute_fn=lambda p, ctx: self.browser_adapter.read_and_summarize_url(p.get("url") or p.get("target", ""))
        ))

        # ─────────────────────────────────────────────────────────────────────
        # 6. EXPANDED FILE & ARCHIVE OPERATIONS
        # ─────────────────────────────────────────────────────────────────────

        self.register(ActionCapability(
            name="file_create",
            category="files",
            description="Create or write content to a file with read-back verification.",
            parameters_schema={"path": "File path", "content": "Text content to write", "overwrite": "Boolean"},
            preconditions=[],
            expected_output="file_created",
            is_consequential=True,
            requires_confirmation=False,
            execute_fn=lambda p, ctx: self.text_writer.write_to_file(
                file_path=p.get("path") or p.get("file") or p.get("filename", "output.txt"),
                content=p.get("content") or p.get("text", ""),
                overwrite=p.get("overwrite", True)
            )
        ))

        self.register(ActionCapability(
            name="file_read",
            category="files",
            description="Read text content from an accessible local document.",
            parameters_schema={"path": "File path"},
            preconditions=["file exists"],
            expected_output="file_content",
            is_consequential=False,
            requires_confirmation=False,
            execute_fn=lambda p, ctx: self._execute_file_read(p.get("path") or p.get("file", ""))
        ))

        self.register(ActionCapability(
            name="file_append",
            category="files",
            description="Append content to an existing file with read-back verification.",
            parameters_schema={"path": "File path", "content": "Text to append"},
            preconditions=["file exists"],
            expected_output="file_appended",
            is_consequential=True,
            requires_confirmation=False,
            execute_fn=lambda p, ctx: self.text_writer.write_to_file(
                file_path=p.get("path") or p.get("file", ""),
                content=p.get("content") or p.get("text", ""),
                append=True
            )
        ))

        self.register(ActionCapability(
            name="file_zip",
            category="files",
            description="Compress files or folders into a .zip archive.",
            parameters_schema={"source": "Source path", "destination": "Output zip path"},
            preconditions=["source exists"],
            expected_output="file_zipped",
            is_consequential=True,
            requires_confirmation=False,
            execute_fn=lambda p, ctx: self._execute_zip(p.get("source", ""), p.get("destination", ""))
        ))

        self.register(ActionCapability(
            name="file_unzip",
            category="files",
            description="Extract a .zip archive into a target folder.",
            parameters_schema={"source": "Zip file path", "destination": "Extract destination"},
            preconditions=["source exists"],
            expected_output="file_unzipped",
            is_consequential=True,
            requires_confirmation=False,
            execute_fn=lambda p, ctx: self._execute_unzip(p.get("source", ""), p.get("destination", ""))
        ))

        # ─────────────────────────────────────────────────────────────────────
        # 7. SYSTEM UTILITIES, CLIPBOARD, & DIAGNOSTICS
        # ─────────────────────────────────────────────────────────────────────

        self.register(ActionCapability(
            name="system_screenshot",
            category="system",
            description="Capture screen screenshot and save to Desktop or custom path.",
            parameters_schema={"destination": "Optional destination path"},
            preconditions=[],
            expected_output="screenshot_path",
            is_consequential=False,
            requires_confirmation=False,
            execute_fn=lambda p, ctx: self._execute_screenshot(p.get("destination") or p.get("path"))
        ))

        self.register(ActionCapability(
            name="system_clipboard_copy",
            category="system",
            description="Copy specified text to Windows system clipboard.",
            parameters_schema={"content": "Text to copy"},
            preconditions=[],
            expected_output="clipboard_copied",
            is_consequential=False,
            requires_confirmation=False,
            execute_fn=lambda p, ctx: self._execute_clipboard_copy(p.get("content") or p.get("text", ""))
        ))

        self.register(ActionCapability(
            name="system_clipboard_read",
            category="system",
            description="Read text from Windows system clipboard.",
            parameters_schema={},
            preconditions=[],
            expected_output="clipboard_text",
            is_consequential=False,
            requires_confirmation=False,
            execute_fn=lambda p, ctx: self._execute_clipboard_read()
        ))

        self.register(ActionCapability(
            name="system_metrics",
            category="system",
            description="Query live system CPU, RAM, disk, battery, and process diagnostics.",
            parameters_schema={},
            preconditions=[],
            expected_output="system_metrics",
            is_consequential=False,
            requires_confirmation=False,
            execute_fn=lambda p, ctx: self._execute_metrics()
        ))

        self.register(ActionCapability(
            name="voice_listen",
            category="voice",
            description="Record microphone input and transcribe using Speech-to-Text.",
            parameters_schema={"timeout": {"type": "number", "default": 7.0}},
            preconditions=[],
            expected_output="transcription",
            is_consequential=False,
            requires_confirmation=False,
            execute_fn=lambda p, ctx: self._execute_voice_listen(p.get("timeout", 7.0))
        ))

        self.register(ActionCapability(
            name="voice_speak",
            category="voice",
            description="Speak message aloud using Windows SAPI Text-to-Speech.",
            parameters_schema={"text": {"type": "string", "required": True}},
            preconditions=[],
            expected_output="speech_output",
            is_consequential=False,
            requires_confirmation=False,
            execute_fn=lambda p, ctx: self._execute_voice_speak(p.get("text", ""))
        ))

        # ─────────────────────────────────────────────────────────────────────
        # 8. EXTENDED SYSTEM & PRODUCTIVITY UTILITIES
        # ─────────────────────────────────────────────────────────────────────

        self.register(ActionCapability(
            name="window_control",
            category="system",
            description="Manage active or target windows (maximize, minimize, restore, snap left/right, switch to, close).",
            parameters_schema={"action": "maximize | minimize | restore | snap_left | snap_right | switch | close", "app_name": "Target application name"},
            preconditions=[],
            expected_output="window_manipulated",
            is_consequential=False,
            requires_confirmation=False,
            execute_fn=lambda p, ctx: self._execute_window_control(p.get("action", ""), p.get("app_name"))
        ))

        self.register(ActionCapability(
            name="system_timer",
            category="system",
            description="Set a background countdown timer or reminder with audio notification.",
            parameters_schema={"duration": "Seconds or duration string", "label": "Timer label", "subaction": "set | list | cancel"},
            preconditions=[],
            expected_output="timer_event",
            is_consequential=False,
            requires_confirmation=False,
            execute_fn=lambda p, ctx: self._execute_timer(p)
        ))

        self.register(ActionCapability(
            name="notes_manager",
            category="productivity",
            description="Add, list, or clear persistent quick notes and scratchpad entries.",
            parameters_schema={"action": "add | list | clear", "content": "Note text", "note_id": "Optional note index"},
            preconditions=[],
            expected_output="notes_updated",
            is_consequential=False,
            requires_confirmation=False,
            execute_fn=lambda p, ctx: self._execute_notes(p)
        ))

        self.register(ActionCapability(
            name="system_weather",
            category="system",
            description="Query current weather and temperature for a city or local area.",
            parameters_schema={"location": "Optional city name or country"},
            preconditions=[],
            expected_output="weather_report",
            is_consequential=False,
            requires_confirmation=False,
            execute_fn=lambda p, ctx: self._execute_weather(p.get("location", ""))
        ))

        self.register(ActionCapability(
            name="unit_convert",
            category="computation",
            description="Convert physical units and currencies (length, mass, temperature, data storage, time).",
            parameters_schema={"amount": "Number", "from_unit": "Unit", "to_unit": "Unit"},
            preconditions=[],
            expected_output="conversion_result",
            is_consequential=False,
            requires_confirmation=False,
            execute_fn=lambda p, ctx: self._execute_unit_convert(p)
        ))

        self.register(ActionCapability(
            name="system_processes",
            category="system",
            description="Inspect open user desktop windows and check if an app or process is running.",
            parameters_schema={"subaction": "list_apps | is_running", "app_name": "Target application name"},
            preconditions=[],
            expected_output="processes_info",
            is_consequential=False,
            requires_confirmation=False,
            execute_fn=lambda p, ctx: self._execute_processes(p)
        ))

    def _execute_file_read(self, file_path: str) -> Dict[str, Any]:
        """Read content from a local document."""
        if not file_path or not os.path.exists(file_path):
            return {"status": "error", "message": f"File '{file_path}' does not exist on disk."}
        try:
            content = self.file_discovery._extract_text_content(file_path, max_chars=12000)
            if not content:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read(12000)
            return {
                "status": "success",
                "action": "file_read",
                "path": file_path,
                "file": os.path.basename(file_path),
                "length": len(content),
                "content": content,
                "message": f"Read {len(content)} characters from '{os.path.basename(file_path)}'."
            }
        except Exception as e:
            return {"status": "error", "message": f"Failed reading file '{file_path}': {e}"}

    def _execute_zip(self, source: str, destination: str = "") -> Dict[str, Any]:
        """Compress file or directory to a .zip archive."""
        if not source or not os.path.exists(source):
            return {"status": "error", "message": f"Source '{source}' does not exist."}
        import zipfile
        out_zip = destination or (source.rstrip("/\\") + ".zip")
        try:
            with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
                if os.path.isdir(source):
                    for root, dirs, files in os.walk(source):
                        for f in files:
                            full_p = os.path.join(root, f)
                            rel_p = os.path.relpath(full_p, source)
                            z.write(full_p, rel_p)
                else:
                    z.write(source, os.path.basename(source))
            return {
                "status": "success",
                "action": "file_zip",
                "path": out_zip,
                "source": source,
                "size_bytes": os.path.getsize(out_zip) if os.path.exists(out_zip) else 0,
                "message": f"Compressed '{os.path.basename(source)}' into '{os.path.basename(out_zip)}'."
            }
        except Exception as e:
            return {"status": "error", "message": f"Failed compressing '{source}': {e}"}

    def _execute_unzip(self, source: str, destination: str = "") -> Dict[str, Any]:
        """Extract a .zip archive to destination directory."""
        if not source or not os.path.exists(source):
            return {"status": "error", "message": f"Zip file '{source}' does not exist."}
        import zipfile
        dest_dir = destination or os.path.splitext(source)[0]
        try:
            os.makedirs(dest_dir, exist_ok=True)
            with zipfile.ZipFile(source, "r") as z:
                z.extractall(dest_dir)
            return {
                "status": "success",
                "action": "file_unzip",
                "path": dest_dir,
                "source": source,
                "message": f"Extracted '{os.path.basename(source)}' into '{os.path.basename(dest_dir)}'."
            }
        except Exception as e:
            return {"status": "error", "message": f"Failed extracting '{source}': {e}"}

    def _execute_screenshot(self, destination: Optional[str] = None) -> Dict[str, Any]:
        """Capture screen screenshot and save to disk."""
        import datetime
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"edith_screenshot_{ts}.png"
        filepath = destination or os.path.join(desktop, filename)
        try:
            from PIL import ImageGrab
            img = ImageGrab.grab()
            img.save(filepath)
            return {
                "status": "success",
                "action": "screenshot",
                "path": filepath,
                "file": filename,
                "message": f"Captured screenshot and saved to: {filepath}"
            }
        except Exception:
            import subprocess
            escaped = filepath.replace("\\", "/")
            ps_cmd = f"$b = New-Object System.Drawing.Bitmap ([System.Windows.Forms.Screen]::PrimaryScreen.Bounds.Width, [System.Windows.Forms.Screen]::PrimaryScreen.Bounds.Height); $g = [System.Drawing.Graphics]::FromImage($b); $g.CopyFromScreen(0, 0, 0, 0, $b.Size); $b.Save('{escaped}');"
            subprocess.run(f'powershell -NoProfile -Command "Add-Type -AssemblyName System.Windows.Forms; {ps_cmd}"', shell=True)
            return {
                "status": "success" if os.path.exists(filepath) else "error",
                "action": "screenshot",
                "path": filepath,
                "file": filename,
                "message": f"Captured screenshot and saved to: {filepath}"
            }

    def _execute_clipboard_copy(self, text: str) -> Dict[str, Any]:
        """Place text on Windows clipboard."""
        from runtime import actions
        ok = actions.set_clipboard_text(text)
        return {
            "status": "success" if ok else "error",
            "action": "clipboard_copy",
            "length": len(text),
            "content": text,
            "message": f"Copied {len(text)} characters to clipboard." if ok else "Failed copying text to clipboard."
        }

    def _execute_clipboard_read(self) -> Dict[str, Any]:
        """Read text from Windows clipboard."""
        from runtime import actions
        txt = actions.get_clipboard_text()
        return {
            "status": "success",
            "action": "clipboard_read",
            "length": len(txt),
            "content": txt,
            "message": f"Retrieved {len(txt)} characters from clipboard."
        }

    def _execute_metrics(self) -> Dict[str, Any]:
        """Query live system telemetry."""
        import psutil
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage(os.path.abspath("."))
        batt = psutil.sensors_battery()
        batt_pct = batt.percent if batt else None
        plugged = batt.power_plugged if batt else None

        batt_str = f", Battery: {batt_pct}% ({'Charging' if plugged else 'Discharging'})" if batt_pct is not None else ""
        formatted = f"CPU Usage: {cpu}%, RAM Usage: {mem.percent}% ({round(mem.used/(1024**3), 1)}GB / {round(mem.total/(1024**3), 1)}GB), Disk Free: {round(disk.free/(1024**3), 1)}GB{batt_str}"

        return {
            "status": "success",
            "action": "system_metrics",
            "cpu_percent": cpu,
            "memory_percent": mem.percent,
            "disk_free_gb": round(disk.free / (1024**3), 1),
            "battery_percent": batt_pct,
            "is_charging": plugged,
            "formatted": formatted,
            "message": f"System Diagnostics: {formatted}"
        }

    def _execute_voice_listen(self, timeout: float = 7.0) -> Dict[str, Any]:
        """Transcribe speech from microphone using EdithSTTEngine."""
        try:
            from runtime.stt_engine import EdithSTTEngine
            stt = EdithSTTEngine()
            res = stt.listen_and_transcribe(timeout=timeout)
            if res.get("success"):
                return {
                    "status": "success",
                    "action": "voice_listen",
                    "text": res["text"],
                    "cleaned_text": res["cleaned_text"],
                    "engine": res.get("engine", "faster-whisper"),
                    "latency_ms": res.get("latency_ms", 0.0),
                    "message": f"Recognized speech: \"{res['cleaned_text']}\""
                }
            return {
                "status": "error",
                "action": "voice_listen",
                "error": res.get("error", "No speech detected"),
                "message": res.get("error", "No speech detected")
            }
        except Exception as e:
            return {"status": "error", "action": "voice_listen", "message": f"Voice listen failed: {e}"}

    def _execute_voice_speak(self, text: str) -> Dict[str, Any]:
        """Speak text using SAPI TTS."""
        try:
            from runtime.stt_engine import EdithTTSEngine
            tts = EdithTTSEngine()
            tts.speak(text, async_mode=True)
            return {
                "status": "success",
                "action": "voice_speak",
                "text": text,
                "message": f"Spoke text ({len(text)} chars)."
            }
        except Exception as e:
            return {"status": "error", "action": "voice_speak", "message": f"Voice speak failed: {e}"}

    def _execute_window_control(self, action: str, app_name: Optional[str] = None) -> Dict[str, Any]:
        """Manipulate window size, placement, or foreground focus."""
        from runtime.system_functions import WindowController
        return WindowController.execute_action(action, app_name)

    def _execute_timer(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Manage countdown timers and reminders."""
        from runtime.system_functions import TimerManager
        tm = TimerManager()
        subaction = params.get("subaction", "set")
        if subaction == "list":
            active = tm.list_timers()
            return {"status": "success", "action": "system_timer", "timers": active, "message": f"Active timers ({len(active)})." if active else "No timers currently active."}
        elif subaction == "cancel":
            return tm.cancel_timer(params.get("timer_id"))
        else:
            dur = params.get("duration")
            if isinstance(dur, (int, float)):
                secs = float(dur)
            else:
                secs = TimerManager.parse_duration_string(str(dur or params.get("command", ""))) or 60.0
            label = params.get("label", "Timer")
            return tm.set_timer(secs, label)

    def _execute_notes(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Manage persistent quick notes."""
        from runtime.system_functions import NotesManager
        act = params.get("action", "add")
        if act == "list":
            return NotesManager.list_notes()
        elif act in ["clear", "delete"]:
            return NotesManager.clear_notes(params.get("note_id"))
        else:
            return NotesManager.add_note(params.get("content") or params.get("text", ""))

    def _execute_weather(self, location: str = "") -> Dict[str, Any]:
        """Fetch current weather metrics."""
        from runtime.system_functions import WeatherClient
        return WeatherClient.get_weather(location)

    def _execute_unit_convert(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert units or currencies."""
        from runtime.system_functions import UnitConverter
        amt = float(params.get("amount", 1.0))
        u1 = params.get("from_unit", "")
        u2 = params.get("to_unit", "")
        return UnitConverter.convert(amt, u1, u2)

    def _execute_processes(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Inspect running processes and open windows."""
        from runtime.system_functions import ProcessInspector
        subact = params.get("subaction", "list_apps")
        if subact == "is_running" or params.get("app_name"):
            return ProcessInspector.is_app_running(params.get("app_name", ""))
        elif subact == "network":
            return ProcessInspector.get_network_info()
        else:
            return ProcessInspector.list_running_apps()

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
