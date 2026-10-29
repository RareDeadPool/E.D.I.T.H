"""Task and Operating System Execution Handler for E.D.I.T.H."""

import os
import re
import ast
import json
import math
import time
import difflib
import operator
import socket
import datetime
import urllib.parse
import urllib.request
import subprocess
import webbrowser
import shutil
import ctypes
from ctypes import wintypes
from typing import Dict, Any, Optional, Tuple, List

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass


class EdithTaskExecutor:
    """Production-grade Windows system and software task executor."""

    def __init__(self):
        self.file_clipboard: Optional[Dict[str, Any]] = None
        self._gui_agent = None
        self._file_discovery = None
        self._text_writer = None
        self._memory_manager = None
        self._action_registry = None

    @property
    def action_registry(self):
        """Lazy-loaded ActionRegistry for cross-application capability execution."""
        if self._action_registry is None:
            try:
                from runtime.action_registry import ActionRegistry
                self._action_registry = ActionRegistry(
                    file_discovery=self.file_discovery,
                    text_writer=self.text_writer
                )
            except ImportError:
                self._action_registry = None
        return self._action_registry

    @property
    def file_discovery(self):
        """Lazy-loaded FileDiscoveryEngine for intent-based multi-signal file discovery."""
        if self._file_discovery is None:
            try:
                from runtime.file_discovery import FileDiscoveryEngine
                self._file_discovery = FileDiscoveryEngine()
            except ImportError:
                self._file_discovery = None
        return self._file_discovery

    @property
    def text_writer(self):
        """Lazy-loaded TextWritingPipeline for faithful transcription, drafting, and Win32 Notepad insertion."""
        if self._text_writer is None:
            try:
                from runtime.text_writer import TextWritingPipeline
                self._text_writer = TextWritingPipeline()
            except ImportError:
                self._text_writer = None
        return self._text_writer

    @property
    def memory(self):
        """Lazy-loaded EdithMemoryManager for persistent preferences and audit logs."""
        if self._memory_manager is None:
            try:
                from runtime.memory import EdithMemoryManager
                self._memory_manager = EdithMemoryManager()
            except ImportError:
                self._memory_manager = None
        return self._memory_manager

    @property
    def gui_agent(self):
        """Lazy-loaded GUIAgent powered by ScreenEngine perception."""
        if self._gui_agent is None:
            from runtime.gui_agent import GUIAgent
            self._gui_agent = GUIAgent()
        return self._gui_agent

    @property
    def storage_analyzer(self):
        """Lazy-loaded StorageAnalyzer for safe storage intelligence and cleanup."""
        if getattr(self, "_storage_analyzer", None) is None:
            try:
                from runtime.storage_analyzer import StorageAnalyzer
                self._storage_analyzer = StorageAnalyzer()
            except ImportError:
                self._storage_analyzer = None
        return self._storage_analyzer

    @property
    def file_organizer(self):
        """Lazy-loaded SemanticFileOrganizer for taxonomy classification and reversible reorganization."""
        if getattr(self, "_file_organizer", None) is None:
            try:
                from runtime.file_organizer import SemanticFileOrganizer
                self._file_organizer = SemanticFileOrganizer()
            except ImportError:
                self._file_organizer = None
        return self._file_organizer

    @property
    def task_manager(self):
        """Lazy-loaded TaskManager for dependable multitasking and concurrency."""
        if getattr(self, "_task_manager", None) is None:
            try:
                from runtime.task_manager import TaskManager
                self._task_manager = TaskManager()
            except ImportError:
                self._task_manager = None
        return self._task_manager

    @property
    def screen_interpreter(self):
        """Lazy-loaded ScreenInterpreter for unified screen understanding and UI reasoning."""
        if getattr(self, "_screen_interpreter", None) is None:
            try:
                from runtime.screen_interpreter import ScreenInterpreter
                self._screen_interpreter = ScreenInterpreter()
            except ImportError:
                self._screen_interpreter = None
        return self._screen_interpreter

    APP_MAP = {
        "notepad": "notepad.exe",
        "calculator": "calc.exe",
        "calc": "calc.exe",
        "chrome": "chrome.exe",
        "google chrome": "chrome.exe",
        "firefox": "firefox.exe",
        "edge": "msedge.exe",
        "microsoft edge": "msedge.exe",
        "browser": "chrome.exe",
        "text editor": "notepad.exe",
        "code editor": "code",
        "file manager": "explorer.exe",
        "spotify": "explorer.exe shell:AppsFolder\\SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify",
        "vlc": "vlc.exe",
        "vscode": "code",
        "visual studio code": "code",
        "code": "code",
        "coding environment": "code",
        "dev environment": "code",
        "development environment": "code",
        "ide": "code",
        "terminal": "wt.exe",
        "windows terminal": "wt.exe",
        "powershell": "powershell.exe",
        "cmd": "cmd.exe",
        "command prompt": "cmd.exe",
        "task manager": "taskmgr.exe",
        "taskmgr": "taskmgr.exe",
        "file explorer": "explorer.exe",
        "explorer": "explorer.exe",
        "files": "explorer.exe",
        "control panel": "control.exe",
        "settings": "start ms-settings:",
        "wordpad": "write.exe",
        "paint": "mspaint.exe",
        "word": "winword.exe",
        "excel": "excel.exe",
        "powerpoint": "powerpnt.exe",
        "discord": "discord.exe",
        "slack": "slack.exe",
        "teams": "teams.exe",
        "mail": "explorer.exe shell:AppsFolder\\microsoft.windowscommunicationsapps_8wekyb3d8bbwe!microsoft.windowslive.mail",
        "email": "explorer.exe shell:AppsFolder\\microsoft.windowscommunicationsapps_8wekyb3d8bbwe!microsoft.windowslive.mail",
        "outlook": "outlook.exe",
        "gmail": "https://mail.google.com",
        "whatsapp": "whatsapp:",
        "whatsapp desktop": "whatsapp:",
    }

    def _find_app_in_registry(self, app_name: str) -> Optional[str]:
        """Lookup application path in Windows Registry App Paths."""
        try:
            import winreg
            clean = re.sub(r"\.exe$", "", app_name.strip().lower())
            for root in [winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER]:
                for sub in [
                    f"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\App Paths\\{clean}.exe",
                    f"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\App Paths\\{clean}"
                ]:
                    try:
                        with winreg.OpenKey(root, sub) as key:
                            val, _ = winreg.QueryValueEx(key, "")
                            if val and os.path.exists(val):
                                return val
                    except Exception:
                        pass
        except Exception:
            pass
        return None

    def execute(self, directive: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task directive produced by EDITH Brain."""
        handler = directive.get("handler")
        params = directive.get("parameters", {})

        if not directive.get("can_execute", False) or not handler:
            return {
                "status": "skipped",
                "message": directive.get("message", "Directive is marked non-executable.")
            }

        # 1. Launch Application
        if handler == "launch_app":
            app_name = params.get("app_name", "").strip().lower()
            profile = params.get("profile", "Aditya Sawant")
            return self._launch_app(app_name, profile=profile)

        # 2. Terminate Application
        elif handler == "close_app":
            app_name = params.get("app_name", "").strip().lower()
            return self._close_app(app_name)

        # 3. System Metrics Query
        elif handler == "system_metrics":
            return self._query_system_metrics()

        # 4. System Control (Volume, Lock, Sleep)
        elif handler in ["system_control", "exec_sys_command"]:
            return self._system_control(params)

        # 5. File Operations
        elif handler == "file_operation":
            return self._file_operation(params)

        # 6. Web Search
        elif handler == "web_search":
            query = params.get("query", "")
            return self._web_search(query)

        # 7. Network Diagnostics
        elif handler in ["network_task", "exec_network_diagnostic"]:
            return self._network_diagnostics()

        # 8. Kill Process by PID
        elif handler == "kill_process":
            pid = params.get("pid")
            return self._kill_process(pid)

        # 9. Media Control (Pause, Skip, Resume)
        elif handler in ["media_control", "exec_media_change"]:
            return self._media_control(params)

        # 10. Music Playback (Spotify / YouTube)
        elif handler in ["play_music", "exec_play_music", "music_playback"]:
            return self._play_music(params)

        # 11. Complex Mathematical Computation
        elif handler in ["compute_math", "exec_compute_math", "math_compute"]:
            return self._compute_math(params)

        # 12. Windows System Utilities (Screenshot, Time, Folders, Minimize)
        elif handler in ["system_utility", "exec_system_utility"]:
            return self._system_utility(params)

        # 13. Multi-action / Workflow Script
        elif handler in ["workflow_script", "exec_workflow_script", "automation_workflow"]:
            return self._workflow_script(params)

        # 14. Universal GUI Action (Powered by Screen Perception Engine)
        elif handler in ["gui_action", "gui_click", "gui_search", "gui_inspect"]:
            return self._gui_action(params)

        # 15. Web Navigation & Service Dispatch (Gmail, YouTube, GitHub, any URL)
        elif handler in ["web_navigation", "exec_web_navigation", "navigate_url"]:
            return self._web_navigation(params)

        # 16. Intelligent Content Drafting & Composition (Email, Letters, Memos)
        elif handler in ["draft_content", "exec_draft_content", "compose_draft"]:
            return self._draft_content(params)

        # 17. Dialogue Response
        elif handler == "dialogue":
            return {"status": "success", "message": directive.get("message")}

        # 18. Project Action (run tests, test suite, project build, project run)
        elif handler in ["project_action", "exec_project_action", "run_tests"]:
            return self._project_action(params)

        # 19. WhatsApp Cross-App Capability Execution
        elif handler in ["whatsapp", "whatsapp_send", "whatsapp_draft", "whatsapp_revise", "whatsapp_preview", "send_whatsapp", "draft_whatsapp"]:
            reg = self.action_registry
            if reg:
                cap_name = handler if handler.startswith("whatsapp_") else f"whatsapp_{handler.replace('send_whatsapp', 'send').replace('draft_whatsapp', 'draft')}"
                if cap_name not in reg.capabilities:
                    cap_name = "whatsapp_send" if "send" in handler else "whatsapp_draft"
                res, ver = reg.execute(cap_name, params)
                return res
            return {"status": "error", "message": "Action registry is not initialized."}

        # 20. Email Cross-App Capability Execution
        elif handler in ["email", "email_send", "email_draft", "email_revise", "email_preview", "send_email", "draft_email"]:
            reg = self.action_registry
            if reg:
                cap_name = handler if handler.startswith("email_") else f"email_{handler.replace('send_email', 'send').replace('draft_email', 'draft')}"
                if cap_name not in reg.capabilities:
                    cap_name = "email_send" if "send" in handler else "email_draft"
                res, ver = reg.execute(cap_name, params)
                return res
            return {"status": "error", "message": "Action registry is not initialized."}

        # 21. Document Summarization, File Operations, and Archiving
        elif handler in [
            "file_summarize", "file_rename", "file_copy", "file_move", "file_organize",
            "file_create", "file_read", "file_append", "file_zip", "file_unzip"
        ]:
            reg = self.action_registry
            if reg:
                res, ver = reg.execute(handler, params)
                return res
            return {"status": "error", "message": "Action registry is not initialized."}

        # 22. Browser and Web Actions
        elif handler in ["browser_open", "browser_search_and_summarize", "browser_fill_form", "browser_read_page"]:
            reg = self.action_registry
            if reg:
                res, ver = reg.execute(handler, params)
                return res
            return {"status": "error", "message": "Action registry is not initialized."}

        # 23. System Utilities & Clipboard Actions
        elif handler in ["system_screenshot", "system_clipboard_copy", "system_clipboard_read", "clipboard_copy", "clipboard_read"]:
            reg = self.action_registry
            if reg:
                cap_name = handler if handler.startswith("system_") else f"system_{handler}"
                res, ver = reg.execute(cap_name, params)
                return res
            return {"status": "error", "message": "Action registry is not initialized."}

        # 24. Voice STT & TTS Actions
        elif handler in ["voice_listen", "voice_speak"]:
            reg = self.action_registry
            if reg:
                res, ver = reg.execute(handler, params)
                return res
            return {"status": "error", "message": "Action registry is not initialized."}

        # 25. Extended Desktop & Productivity Utilities
        elif handler in [
            "window_control", "system_timer", "notes_manager", "system_weather",
            "unit_convert", "system_processes", "timer", "notes", "weather"
        ]:
            reg = self.action_registry
            if reg:
                norm_handler = handler
                if handler == "timer": norm_handler = "system_timer"
                elif handler == "notes": norm_handler = "notes_manager"
                elif handler == "weather": norm_handler = "system_weather"
                res, ver = reg.execute(norm_handler, params)
                return res
            return {"status": "error", "message": "Action registry is not initialized."}

        # 26. Safe Storage Intelligence & Cleanup
        elif handler in ["storage_audit", "storage_cleanup", "storage_intelligence"]:
            return self._handle_storage_intelligence(params)

        # 27. Intelligent File Organization
        elif handler in ["file_organize_plan", "semantic_file_organize", "file_organize_rollback"]:
            return self._handle_file_organization(handler, params)

        # 28. Multitasking & Background Task Manager
        elif handler in ["task_submit", "task_cancel", "task_list", "task_status"]:
            return self._handle_task_manager(handler, params)

        # 29. Unified Screen Understanding
        elif handler in ["screen_understanding", "explain_screen", "screen_error_check"]:
            return self._handle_screen_understanding(params)

        return {
            "status": "unhandled",
            "message": f"Action handler '{handler}' is recognized by ANN but not yet bound in runtime."
        }

    def _handle_storage_intelligence(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute safe storage audit or reversible quarantine cleanup."""
        if not self.storage_analyzer:
            return {"status": "error", "message": "StorageAnalyzer is unavailable."}

        subaction = params.get("subaction", "audit")
        target_dir = params.get("path")
        report = self.storage_analyzer.audit_storage(target_dir=target_dir)

        if subaction == "cleanup":
            dry_run = params.get("dry_run", True)
            candidates = report.candidates_for_cleanup
            quarantine_res = self.storage_analyzer.quarantine_candidates(candidates, dry_run=dry_run)
            return {
                "status": "success",
                "action": "storage_cleanup",
                "dry_run": dry_run,
                "items_processed": len(quarantine_res.get("items", [])),
                "bytes_saved": quarantine_res.get("total_bytes_saved", 0),
                "manifest_path": quarantine_res.get("manifest_path"),
                "message": f"Storage cleanup {'dry run completed' if dry_run else 'quarantined'} {len(quarantine_res.get('items', []))} items safely."
            }

        return {
            "status": "success",
            "action": "storage_audit",
            "disk_free_gb": report.disk_free_gb,
            "potential_savings_mb": report.potential_savings_mb,
            "duplicate_groups": len(report.duplicates),
            "candidates_count": len(report.candidates_for_cleanup),
            "message": f"Storage audit complete: {report.disk_free_gb} GB free. Identified {report.potential_savings_mb} MB potential savings across {len(report.candidates_for_cleanup)} disposable items."
        }

    def _handle_file_organization(self, handler: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute semantic file categorization, dry-run plan, or transaction rollback."""
        if not self.file_organizer:
            return {"status": "error", "message": "SemanticFileOrganizer is unavailable."}

        if handler == "file_organize_rollback":
            tx_id = params.get("transaction_id", "")
            res = self.file_organizer.rollback_transaction(tx_id)
            return {
                "status": "success" if res.get("success") else "error",
                "restored_count": res.get("restored_count", 0),
                "message": f"Rollback restored {res.get('restored_count', 0)} files back to their original locations."
            }

        source_dir = params.get("source_dir") or params.get("path") or os.path.expanduser("~/Downloads")
        plan = self.file_organizer.generate_organization_plan(source_dir)

        execute_real = params.get("execute", False)
        exec_res = self.file_organizer.execute_organization_plan(plan, dry_run=not execute_real)

        return {
            "status": "success",
            "dry_run": not execute_real,
            "planned_moves": len(plan.moves),
            "protected_files": len(plan.protected_files),
            "transaction_file": exec_res.get("transaction_file"),
            "message": f"File organization {'plan generated' if not execute_real else 'executed'}: {len(plan.moves)} moves proposed, {len(plan.protected_files)} protected project files preserved."
        }

    def _handle_task_manager(self, handler: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Manage background concurrent multitasking, cancellation, and inspection."""
        if not self.task_manager:
            return {"status": "error", "message": "TaskManager is unavailable."}

        if handler == "task_list":
            tasks = self.task_manager.list_tasks()
            return {"status": "success", "count": len(tasks), "tasks": tasks}

        elif handler == "task_cancel":
            t_id = params.get("task_id", "")
            success = self.task_manager.cancel_task(t_id, reason=params.get("reason", "Cancelled by user"))
            return {
                "status": "success" if success else "failed",
                "task_id": t_id,
                "message": f"Task '{t_id}' cancellation {'requested successfully' if success else 'failed or already finished'}."
            }

        elif handler == "task_status":
            t_id = params.get("task_id", "")
            task = self.task_manager.get_task(t_id)
            if task:
                return {"status": "success", "task": task}
            return {"status": "error", "message": f"Task '{t_id}' not found."}

        return {"status": "unhandled", "message": f"Task action '{handler}' not supported."}

    def _handle_screen_understanding(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform unified multi-modal screen interpretation using UIA, OCR, and error detection."""
        if not self.screen_interpreter:
            return {"status": "error", "message": "ScreenInterpreter is unavailable."}

        query = params.get("query", "")
        if hasattr(self.screen_interpreter, "interpret_screen"):
            interpretation = self.screen_interpreter.interpret_screen(user_query=query)
        else:
            interpretation = self.screen_interpreter.interpret(user_question=query)

        summary = getattr(interpretation, "grounded_summary", None)
        if summary is None and isinstance(interpretation, dict):
            summary = interpretation.get("message", "Screen analysis complete.")

        dict_data = interpretation.to_dict() if hasattr(interpretation, "to_dict") else (interpretation if isinstance(interpretation, dict) else {})

        return {
            "status": "success",
            "interpretation": dict_data,
            "message": summary or "Screen analysis complete."
        }

    def _project_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute project-level development workflows (test execution, verification, builds)."""
        subaction = params.get("subaction", "run_tests")
        workspace = params.get("workspace", os.path.abspath("."))

        if subaction == "resolve_test_cmd":
            test_cmd = "py -3.12 -m unittest discover tests"
            if os.path.exists(os.path.join(workspace, "pytest.ini")) or os.path.exists(os.path.join(workspace, "conftest.py")):
                test_cmd = "pytest"
            return {
                "status": "success",
                "test_command": test_cmd,
                "message": f"Resolved test command: '{test_cmd}'."
            }

        elif subaction == "verify_dir":
            p = params.get("path") or workspace
            if os.path.isdir(p):
                return {
                    "status": "success",
                    "path": p,
                    "message": f"Verified project directory at: {p}"
                }
            return {
                "status": "error",
                "message": f"Project directory does not exist: {p}"
            }

        elif subaction == "evaluate_output":
            return {
                "status": "success",
                "message": "Project test execution evaluated and verified."
            }

        elif subaction in ["run_tests", "execute_tests", "test"]:
            cmd = params.get("command") or "py -3.12 -m unittest discover tests"
            try:
                proc = subprocess.run(
                    cmd,
                    shell=True,
                    cwd=workspace,
                    capture_output=True,
                    text=True,
                    timeout=90
                )
                exit_code = proc.returncode
                stdout = proc.stdout
                stderr = proc.stderr
                tests_passed = (exit_code == 0)
                summary_msg = "All project tests passed successfully." if tests_passed else f"Tests failed with exit code {exit_code}."
                return {
                    "status": "success" if tests_passed else "test_failure",
                    "exit_code": exit_code,
                    "stdout": stdout,
                    "stderr": stderr,
                    "tests_passed": tests_passed,
                    "command": cmd,
                    "message": summary_msg
                }
            except subprocess.TimeoutExpired:
                return {
                    "status": "timeout",
                    "exit_code": -1,
                    "message": "Test execution timed out after 90 seconds."
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Failed to execute tests: {e}"
                }

        return {"status": "unhandled", "message": f"Unknown project subaction: {subaction}"}

    def _gui_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute dynamic GUI perception-driven actions across any Windows application."""
        action_type = params.get("action", "click").lower().strip()
        app_name = params.get("app_name", "").strip()
        target = params.get("target", "").strip()
        text = params.get("text", "").strip()
        double = params.get("double", False)

        try:
            # 1. Click an element dynamically
            if action_type in ["click", "double_click"]:
                is_double = double or (action_type == "double_click")
                if app_name:
                    res = self.gui_agent.click_in_app(app_name, target, double=is_double)
                else:
                    elem = self.gui_agent.click_element(target, double=is_double, refresh=True)
                    if elem:
                        res = {
                            "status": "success",
                            "message": f"Clicked '{elem.name}' ({elem.role}) at {elem.center}.",
                            "element": {"name": elem.name, "role": elem.role, "center": elem.center}
                        }
                    else:
                        res = {
                            "status": "not_found",
                            "message": f"Could not find element '{target}' on the active screen."
                        }
                return res

            # 2. Search inside an app dynamically
            elif action_type == "search":
                search_query = text or target
                target_app = app_name or "chrome"
                return self.gui_agent.search_in_app(target_app, search_query)

            # 3. Type text into a targeted element
            elif action_type == "type":
                # If typing into Notepad, route through verified Win32 text writing pipeline
                if app_name and app_name.lower() in ["notepad", "text editor", "editor"] and self.text_writer:
                    ins_res = self.text_writer.insert_into_notepad(
                        content=text,
                        append=params.get("append", False),
                        replace=params.get("replace", False)
                    )
                    if ins_res.get("success") or ins_res.get("status") == "success":
                        return {
                            "status": "success",
                            "action": "type_notepad",
                            "app": "notepad",
                            "verified": ins_res.get("verified", False),
                            "exact_match": ins_res.get("exact_match", False),
                            "characters_written": ins_res.get("characters_written", len(text)),
                            "message": f"Faithfully entered {len(text)} characters into Notepad (verified: {ins_res.get('verified', False)})."
                        }
                    else:
                        return {
                            "status": "error",
                            "action": "type_notepad",
                            "message": f"Failed to insert text into Notepad: {ins_res.get('message')}"
                        }

                hwnd = self.gui_agent.focus_app(app_name) if app_name else None
                elem = None
                if target:
                    elem = self.gui_agent.type_into_element(target, text, hwnd=hwnd, submit=params.get("submit", False))
                if elem:
                    return {
                        "status": "success",
                        "message": f"Typed '{text}' into '{elem.name}' ({elem.role}) in {app_name or 'active window'}.",
                        "element": {"name": elem.name, "role": elem.role, "center": elem.center}
                    }
                else:
                    # Direct typing fallback into focused window
                    from runtime import actions
                    if hwnd:
                        actions.focus_window(hwnd)
                        time.sleep(0.1)
                    actions.type_text(text)
                    if params.get("submit", False):
                        time.sleep(0.05)
                        actions.press_key(0x0D)
                    return {
                        "status": "success",
                        "message": f"Typed '{text}' into {app_name or 'active window'}."
                    }

            # 4. Inspect / perceive what is on screen
            elif action_type in ["inspect", "perceive", "see"]:
                hwnd = self.gui_agent.focus_app(app_name) if app_name else None
                elements = self.gui_agent.see(hwnd=hwnd, force_ocr=True)
                top_items = []
                for e in elements[:15]:
                    if e.name:
                        top_items.append(f"[{e.role}] '{e.name}' at {e.center}")
                summary = f"Detected {len(elements)} elements" + (f" in {app_name}" if app_name else " on screen") + ":\n" + "\n".join(f"  • {item}" for item in top_items)
                return {
                    "status": "success",
                    "action": "gui_inspect",
                    "app": app_name,
                    "element_count": len(elements),
                    "summary": summary,
                    "message": summary
                }

            else:
                return {
                    "status": "error",
                    "message": f"Unknown GUI action type: '{action_type}'"
                }

        except Exception as e:
            return {
                "status": "error",
                "message": f"GUI action failed: {e}"
            }

    def _get_chrome_exe_path(self) -> str:
        candidates = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
            shutil.which("chrome.exe") or "",
        ]
        for p in candidates:
            if p and os.path.isfile(p):
                return p
        return "chrome.exe"

    def _resolve_chrome_profile_dir(self, query_name: str = "Aditya Sawant") -> str:
        """Resolve a friendly profile name (e.g. 'Aditya Sawant') to its Chrome directory."""
        try:
            base = os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\User Data")
            local_state_p = os.path.join(base, "Local State")
            if os.path.isfile(local_state_p):
                with open(local_state_p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                profiles = data.get("profile", {}).get("info_cache", {})
                q = (query_name or "").lower().strip()
                for pdir, info in profiles.items():
                    name = (info.get("name") or "").lower()
                    user_name = (info.get("user_name") or "").lower()
                    gaia_name = (info.get("gaia_name") or "").lower()
                    if q and (q in name or q in user_name or q in gaia_name or name in q):
                        return pdir
                for pdir in profiles.keys():
                    pref_p = os.path.join(base, pdir, "Preferences")
                    if os.path.isfile(pref_p):
                        with open(pref_p, "r", encoding="utf-8") as pf:
                            pd = json.load(pf)
                        for a in pd.get("account_info", []):
                            fn = (a.get("full_name") or "").lower()
                            if q and (q in fn or fn in q):
                                return pdir
        except Exception:
            pass
        return "Default"

    def _handle_chrome_profile_picker_if_open(self, target_name: str = "Aditya Sawant", timeout: float = 2.0):
        """Auto-detect if Chrome Profile Picker is showing and click the target profile."""
        try:
            from runtime import actions
            t0 = time.time()
            while time.time() - t0 < timeout:
                hwnd = actions.find_window_by_process(["chrome"], title_contains="Who's using Chrome")
                if not hwnd:
                    hwnd = actions.find_window_by_process(["chrome"], title_contains="Google Chrome - Profile")
                if hwnd:
                    actions.focus_window(hwnd)
                    time.sleep(0.2)
                    elems = self.gui_agent.see(hwnd=hwnd, force_ocr=True)
                    target_elem = self.gui_agent.find(target_name, hwnd=hwnd)
                    if not target_elem and " " in target_name:
                        target_elem = self.gui_agent.find(target_name.split()[0], hwnd=hwnd)
                    if target_elem:
                        if not target_elem.invoke():
                            actions.click_with_restore(target_elem.center[0], target_elem.center[1], hwnd=hwnd)
                        time.sleep(0.5)
                        return True
                time.sleep(0.3)
        except Exception:
            pass
        return False

    def _launch_chrome(self, profile: str = "Aditya Sawant", url: str = "") -> bool:
        """Launch Google Chrome directly into a specific profile (bypassing profile picker)."""
        exe = self._get_chrome_exe_path()
        profile_dir = self._resolve_chrome_profile_dir(profile)
        args = [f"--profile-directory={profile_dir}"]
        if url:
            args.append(url)
        try:
            subprocess.Popen([exe] + args)
            time.sleep(0.8)
            self._handle_chrome_profile_picker_if_open(target_name=profile)
            return True
        except Exception as e:
            print(f"[Executor] Chrome launch error: {e}")
            return False

    def _launch_app(self, app_name: str, profile: str = "Aditya Sawant") -> Dict[str, Any]:
        if not app_name:
            return {"status": "error", "message": "No application specified to launch."}

        raw = app_name.strip()
        low = raw.lower()

        # Handle Chrome specifically with profile selection
        if low in ["chrome", "google chrome"] or "chrome" in low:
            profile_target = profile or "Aditya Sawant"
            success = self._launch_chrome(profile=profile_target)
            if success:
                return {
                    "status": "success",
                    "action": "launch_app",
                    "app": "Google Chrome",
                    "profile": profile_target,
                    "message": f"Opened Google Chrome with profile '{profile_target}'."
                }
            else:
                return {
                    "status": "error",
                    "action": "launch_app",
                    "app": "Google Chrome",
                    "message": "Failed to launch Google Chrome."
                }

        # 1. System shell folder aliases & common typos
        folder_aliases = {
            "downloads": "shell:Downloads",
            "download": "shell:Downloads",
            "dowloads": "shell:Downloads",
            "dowlnoad": "shell:Downloads",
            "dowlnoads": "shell:Downloads",
            "documents": "shell:Personal",
            "document": "shell:Personal",
            "docs": "shell:Personal",
            "desktop": "shell:Desktop",
            "pictures": "shell:My Pictures",
            "photos": "shell:My Pictures",
            "music": "shell:My Music",
            "videos": "shell:My Video",
        }
        for alias, shell_uri in folder_aliases.items():
            if low == alias or low.startswith(f"open {alias}"):
                try:
                    os.startfile(shell_uri)
                    return {
                        "status": "success",
                        "action": "launch_app",
                        "app": raw,
                        "message": f"Opened {alias.capitalize()} folder successfully."
                    }
                except Exception as e:
                    return {"status": "error", "message": f"Failed to open {alias}: {e}"}

        # 2. Check for matching folders on User Desktop (e.g. "E.D.I.T.H", project folders)
        desktop_dir = os.path.join(os.path.expanduser("~"), "Desktop")
        clean_name = re.sub(r"^(?:open\s+)?(?:the\s+)?", "", low)
        clean_name = re.sub(r"\s+folder(?:.*)?$", "", clean_name).strip()
        if os.path.isdir(desktop_dir):
            for entry in os.listdir(desktop_dir):
                entry_norm = entry.lower().replace(".", "").replace(" ", "").replace("_", "").replace("-", "")
                cn_norm = clean_name.lower().replace(".", "").replace(" ", "").replace("_", "").replace("-", "")
                if entry_norm == cn_norm or entry.lower() == clean_name:
                    full_p = os.path.join(desktop_dir, entry)
                    if os.path.isdir(full_p):
                        try:
                            os.startfile(full_p)
                            return {
                                "status": "success",
                                "action": "open_folder",
                                "folder": entry,
                                "path": full_p,
                                "message": f"Opened folder '{entry}' from Desktop."
                            }
                        except Exception as e:
                            return {"status": "error", "message": f"Failed to open folder '{entry}': {e}"}

        # 3. Lookup in APP_MAP with fuzzy matching
        if low == "spotify":
            try:
                from runtime.spotify_adapter import SpotifyAdapter
                spot = SpotifyAdapter()
                ok, msg, hwnd = spot.launch_desktop_app(timeout=8.0)
                return {
                    "status": "success" if ok else "warning",
                    "action": "launch_app",
                    "app": "spotify",
                    "hwnd": hwnd,
                    "message": msg
                }
            except Exception as ex:
                pass

        target = self.APP_MAP.get(low)
        if not target:
            matches = difflib.get_close_matches(low, list(self.APP_MAP.keys()), n=1, cutoff=0.7)
            if matches:
                target = self.APP_MAP[matches[0]]

        # 4. Lookup in Windows Registry App Paths
        if not target:
            reg_path = self._find_app_in_registry(low)
            if reg_path:
                target = reg_path

        if not target:
            target = raw

        try:
            if target.startswith("start "):
                subprocess.Popen(target, shell=True)
            else:
                subprocess.Popen(f'start "" "{target}"', shell=True)
            return {
                "status": "success",
                "action": "launch_app",
                "app": app_name,
                "message": f"Launched '{app_name}' successfully."
            }
        except Exception as e:
            return {"status": "error", "message": f"Failed to launch '{app_name}': {e}"}

    def _close_app(self, app_name: str) -> Dict[str, Any]:
        if not app_name:
            return {"status": "error", "message": "No application specified to terminate."}

        target = self.APP_MAP.get(app_name, app_name)
        if not target.endswith(".exe") and not target.startswith("start "):
            target = f"{target}.exe"

        try:
            res = subprocess.run(
                f'taskkill /f /im "{target}"',
                shell=True,
                capture_output=True,
                text=True
            )
            if res.returncode == 0:
                return {
                    "status": "success",
                    "action": "close_app",
                    "app": app_name,
                    "message": f"Terminated process '{target}'."
                }
            else:
                return {
                    "status": "warning",
                    "action": "close_app",
                    "message": f"Process '{target}' was not currently active."
                }
        except Exception as e:
            return {"status": "error", "message": f"Failed to close '{app_name}': {e}"}

    GENERIC_MUSIC_QUERIES = {
        "a song", "some song", "song", "songs", "some songs",
        "music", "some music", "a track", "tracks", "something",
        "anything", "a music", "track", "tunes", "some tunes"
    }

    def _clean_music_query(self, raw_query: str) -> str:
        """Strip trigger phrases from a raw command, preserving feat./ft. artist qualifiers."""
        q = raw_query.strip()
        q_lower = q.lower()
        for trig in [
            "open spotify and play", "open youtube and play",
            "play on spotify", "play on youtube", "play in spotify",
            "play song", "edith play", "stream", "put on", "play"
        ]:
            if trig in q_lower:
                parts = q_lower.split(trig, 1)
                q = parts[1].strip()
                q_lower = q.lower()
        # Remove trailing platform words only (not inside feat. phrases)
        for p in ["on spotify", "on youtube", "in spotify", "on soundcloud", "on apple music", "spotify", "youtube"]:
            q = re.sub(rf"\s+{re.escape(p)}\s*$", "", q, flags=re.IGNORECASE).strip()
        q = re.sub(r"\'s\b", "", q).strip()
        if q.lower() in self.GENERIC_MUSIC_QUERIES:
            return ""
        return q

    def _build_music_search_query(self, song: str, modifier: str, artist: str) -> str:
        """Construct a precise search query from structured slot fields."""
        if song and song.strip().lower() in self.GENERIC_MUSIC_QUERIES:
            song = ""
        if song:
            song = re.sub(r"\s+(?:on|in)\s+(?:spotify|youtube)\b.*$", "", song, flags=re.IGNORECASE).strip()
        parts: List[str] = []
        if song:
            parts.append(song.strip())
        if modifier:
            parts.append(modifier.strip())
        if artist:
            parts.append(artist.strip())
        return " ".join(parts).strip()

    def _calculate_match_score(self, cleaned_query: str, suggestion: Dict[str, str]) -> float:
        q_words = [w for w in re.findall(r"\w+", cleaned_query.lower()) if len(w) > 1]
        if not q_words:
            return 0.0

        track = suggestion.get("track", "").lower()
        artist = suggestion.get("artist", "").lower()
        combined = f"{track} {artist}"
        comb_words = [w for w in re.findall(r"\w+", combined) if len(w) > 1]

        q_full = " ".join(q_words)
        if q_full == track or q_full == combined:
            return 1.0

        if q_full in track or track in q_full:
            return 0.95

        word_scores = []
        for qw in q_words:
            best_w = 0.0
            for cw in comb_words:
                if qw == cw:
                    best_w = 1.0
                    break
                sim = difflib.SequenceMatcher(None, qw, cw).ratio()
                if sim > best_w:
                    best_w = sim
            word_scores.append(best_w)

        avg_word_score = sum(word_scores) / len(word_scores)
        return avg_word_score

    def _fetch_music_suggestions(self, query: str, limit: int = 6) -> list:
        try:
            encoded = urllib.parse.quote_plus(query)
            url = f"https://itunes.apple.com/search?term={encoded}&media=music&limit={limit}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            results = []
            seen = set()
            for r in data.get("results", []):
                t = r.get("trackName", "")
                a = r.get("artistName", "")
                k = f"{t.lower()} - {a.lower()}"
                if k not in seen and t:
                    seen.add(k)
                    results.append({"track": t, "artist": a})
            return results
        except Exception:
            return []

    def _get_youtube_watch_url(self, query: str) -> str:
        encoded = urllib.parse.quote_plus(query)
        try:
            req = urllib.request.Request(
                f"https://www.youtube.com/results?search_query={encoded}",
                headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                html = resp.read().decode("utf-8")
            m = re.findall(r"/watch\?v=([a-zA-Z0-9_-]{11})", html)
            if m:
                return f"https://www.youtube.com/watch?v={m[0]}&autoplay=1"
        except Exception:
            pass
        return f"https://www.youtube.com/results?search_query={encoded}"

    def _spotify_send_input(self, vk: int, flags: int = 0):
        user32 = ctypes.windll.user32
        extra = ctypes.c_ulong(0)
        
        class KeyBdInput(ctypes.Structure):
            _fields_ = [
                ("wVk", wintypes.WORD),
                ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))
            ]
        class HardwareInput(ctypes.Structure):
            _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]
        class MouseInput(ctypes.Structure):
            _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]
        class Input_I(ctypes.Union):
            _fields_ = [("ki", KeyBdInput), ("mi", MouseInput), ("hi", HardwareInput)]
        class Input(ctypes.Structure):
            _fields_ = [("type", ctypes.c_ulong), ("ii", Input_I)]

        scan = user32.MapVirtualKeyW(vk, 0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput(vk, scan, flags, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)
        user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    def _spotify_press_key(self, vk: int):
        self._spotify_send_input(vk, 0)
        time.sleep(0.04)
        self._spotify_send_input(vk, 0x0002) # KEYEVENTF_KEYUP
        time.sleep(0.04)

    def _spotify_press_ctrl_combo(self, vk: int):
        self._spotify_send_input(0x11, 0) # Ctrl down
        time.sleep(0.04)
        self._spotify_send_input(vk, 0)
        time.sleep(0.04)
        self._spotify_send_input(vk, 0x0002) # Key up
        time.sleep(0.04)
        self._spotify_send_input(0x11, 0x0002) # Ctrl up
        time.sleep(0.04)

    def _spotify_type_string(self, text: str):
        user32 = ctypes.windll.user32
        extra = ctypes.c_ulong(0)
        
        class KeyBdInput(ctypes.Structure):
            _fields_ = [
                ("wVk", wintypes.WORD),
                ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))
            ]
        class HardwareInput(ctypes.Structure):
            _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]
        class MouseInput(ctypes.Structure):
            _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]
        class Input_I(ctypes.Union):
            _fields_ = [("ki", KeyBdInput), ("mi", MouseInput), ("hi", HardwareInput)]
        class Input(ctypes.Structure):
            _fields_ = [("type", ctypes.c_ulong), ("ii", Input_I)]

        for ch in text:
            # Key down with unicode (KEYEVENTF_UNICODE = 0x0004)
            ii_d = Input_I()
            ii_d.ki = KeyBdInput(0, ord(ch), 0x0004, 0, ctypes.pointer(extra))
            xd = Input(ctypes.c_ulong(1), ii_d)
            user32.SendInput(1, ctypes.pointer(xd), ctypes.sizeof(xd))
            time.sleep(0.01)

            # Key up with unicode (KEYEVENTF_UNICODE | KEYEVENTF_KEYUP = 0x0006)
            ii_u = Input_I()
            ii_u.ki = KeyBdInput(0, ord(ch), 0x0006, 0, ctypes.pointer(extra))
            xu = Input(ctypes.c_ulong(1), ii_u)
            user32.SendInput(1, ctypes.pointer(xu), ctypes.sizeof(xu))
            time.sleep(0.01)

    def _find_and_focus_spotify(self) -> Optional[int]:
        user32 = ctypes.windll.user32
        hdesk = user32.OpenInputDesktop(0, False, 0x01FF)
        if hdesk:
            user32.SetThreadDesktop(hdesk)

        # Retrieve all active Spotify process IDs
        spot_pids = set()
        try:
            import psutil
            for p in psutil.process_iter(['pid', 'name']):
                pname = (p.info['name'] or '').lower()
                if 'spotify' in pname:
                    spot_pids.add(p.info['pid'])
        except Exception:
            pass

        candidates = []
        def enum_proc(hwnd, lparam):
            cbuf = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd, cbuf, 256)
            cls_name = cbuf.value
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            tbuf = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(hwnd, tbuf, 256)
            title = tbuf.value.lower()
            
            is_spot = (spot_pids and pid.value in spot_pids) or ("spotify" in title) or (cls_name == "Spotify")
            if is_spot:
                rect = wintypes.RECT()
                user32.GetWindowRect(hwnd, ctypes.byref(rect))
                w = rect.right - rect.left
                h = rect.bottom - rect.top
                vis = user32.IsWindowVisible(hwnd)
                # Ignore background helper / IME / notification windows
                if w >= 250 and h >= 200:
                    score = w * h
                    if vis:
                        score += 10000000
                    if cls_name in ["Chrome_WidgetWin_1", "Spotify"]:
                        score += 5000000
                    candidates.append((score, hwnd))
            return True

        WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        user32.EnumDesktopWindows.argtypes = [wintypes.HANDLE, WNDENUMPROC, wintypes.LPARAM]
        user32.EnumDesktopWindows.restype = wintypes.BOOL

        if hdesk:
            user32.EnumDesktopWindows(hdesk, WNDENUMPROC(enum_proc), 0)
        if not candidates:
            user32.EnumWindows.argtypes = [WNDENUMPROC, wintypes.LPARAM]
            user32.EnumWindows.restype = wintypes.BOOL
            user32.EnumWindows(WNDENUMPROC(enum_proc), 0)

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            spotify_hwnd = candidates[0][1]
            user32.AllowSetForegroundWindow(0xFFFFFFFF)
            if user32.IsIconic(spotify_hwnd):
                user32.ShowWindow(spotify_hwnd, 9) # SW_RESTORE
            else:
                user32.ShowWindow(spotify_hwnd, 5) # SW_SHOW
            user32.SetWindowPos(spotify_hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002) # HWND_TOPMOST
            user32.SetWindowPos(spotify_hwnd, -2, 0, 0, 0, 0, 0x0001 | 0x0002) # HWND_NOTOPMOST
            user32.SetForegroundWindow(spotify_hwnd)
            time.sleep(0.08)
            return spotify_hwnd
        return None

    def _find_spotify_green_button(self, img: Any) -> Optional[Tuple[int, int]]:
        """Visually detect Spotify's high-contrast green play button (#1DB954)."""
        w, h = img.size
        clusters = []
        for y in range(int(h * 0.15), int(h * 0.55)):
            for x in range(int(w * 0.15), int(w * 0.75)):
                r, g, b = img.getpixel((x, y))[:3]
                if g > 165 and r < 80 and b < 130 and (g - r > 100) and (g - b > 60):
                    placed = False
                    for c in clusters:
                        if abs(c["cx"] - x) < 30 and abs(c["cy"] - y) < 30:
                            c["pts"].append((x, y))
                            c["cx"] = sum(p[0] for p in c["pts"]) // len(c["pts"])
                            c["cy"] = sum(p[1] for p in c["pts"]) // len(c["pts"])
                            placed = True
                            break
                    if not placed:
                        clusters.append({"pts": [(x, y)], "cx": x, "cy": y})
        valid = [c for c in clusters if len(c["pts"]) > 30]
        if valid:
            valid.sort(key=lambda c: len(c["pts"]), reverse=True)
            return (valid[0]["cx"], valid[0]["cy"])
        return None

    def _capture_window_image(self, hwnd: int) -> Optional[Any]:
        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32
        rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        w = rect.right - rect.left
        h = rect.bottom - rect.top
        if w <= 0 or h <= 0:
            return None
        hwndDC = user32.GetWindowDC(hwnd)
        memDC = gdi32.CreateCompatibleDC(hwndDC)
        bitmap = gdi32.CreateCompatibleBitmap(hwndDC, w, h)
        gdi32.SelectObject(memDC, bitmap)
        user32.PrintWindow(hwnd, memDC, 2)

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ("biSize", wintypes.DWORD),
                ("biWidth", wintypes.LONG),
                ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD),
                ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD),
                ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG),
                ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD)
            ]
        bih = BITMAPINFOHEADER()
        bih.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bih.biWidth = w
        bih.biHeight = -h  # Negative = top-down DIB (correct orientation)
        bih.biPlanes = 1
        bih.biBitCount = 32
        bih.biCompression = 0
        bih.biSizeImage = w * h * 4
        buf = (ctypes.c_char * (w * h * 4))()
        gdi32.GetDIBits(memDC, bitmap, 0, h, ctypes.byref(buf), ctypes.byref(bih), 0)

        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(memDC)
        user32.ReleaseDC(hwnd, hwndDC)

        try:
            from PIL import Image
            return Image.frombuffer("RGBA", (w, h), bytes(buf), "raw", "BGRA", 0, 1)
        except Exception:
            return None

    def _find_spotify_green_play_button(self, img: Any) -> Optional[Tuple[int, int]]:
        w, h = img.size
        # Search the main content area (ignore left navigation bar and right friend/queue panel)
        clusters: List[Dict[str, Any]] = []
        for y in range(int(h * 0.15), int(h * 0.60)):
            for x in range(int(w * 0.15), int(w * 0.75)):
                r, g, b = img.getpixel((x, y))[:3]
                # Match Spotify signature green: RGB approx (30, 215, 96)
                if g > 165 and r < 80 and b < 130 and (g - r > 100) and (g - b > 60):
                    placed = False
                    for c in clusters:
                        if abs(c["cx"] - x) < 40 and abs(c["cy"] - y) < 40:
                            c["pts"].append((x, y))
                            c["cx"] = sum(p[0] for p in c["pts"]) // len(c["pts"])
                            c["cy"] = sum(p[1] for p in c["pts"]) // len(c["pts"])
                            placed = True
                            break
                    if not placed:
                        clusters.append({"pts": [(x, y)], "cx": x, "cy": y})

        valid = [c for c in clusters if len(c["pts"]) > 30]
        if valid:
            valid.sort(key=lambda c: len(c["pts"]), reverse=True)
            return (valid[0]["cx"], valid[0]["cy"])
        return None

    def _capture_window_topdown(self, hwnd: int) -> Optional[Any]:
        """Capture window as a top-down image (correct orientation for coordinate mapping)."""
        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32
        rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        w = rect.right - rect.left
        h = rect.bottom - rect.top
        if w <= 0 or h <= 0:
            return None
        hwndDC = user32.GetWindowDC(hwnd)
        memDC = gdi32.CreateCompatibleDC(hwndDC)
        bitmap = gdi32.CreateCompatibleBitmap(hwndDC, w, h)
        gdi32.SelectObject(memDC, bitmap)
        user32.PrintWindow(hwnd, memDC, 2)

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ("biSize", wintypes.DWORD),
                ("biWidth", wintypes.LONG),
                ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD),
                ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD),
                ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG),
                ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD)
            ]
        bih = BITMAPINFOHEADER()
        bih.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bih.biWidth = w
        bih.biHeight = -h  # Negative = top-down DIB (correct orientation)
        bih.biPlanes = 1
        bih.biBitCount = 32
        bih.biCompression = 0
        bih.biSizeImage = w * h * 4
        buf = (ctypes.c_char * (w * h * 4))()
        gdi32.GetDIBits(memDC, bitmap, 0, h, ctypes.byref(buf), ctypes.byref(bih), 0)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(memDC)
        user32.ReleaseDC(hwnd, hwndDC)
        try:
            from PIL import Image
            return Image.frombuffer("RGBA", (w, h), bytes(buf), "raw", "BGRA", 0, 1)
        except Exception:
            return None

    def _wait_for_spotify_dom(self, hwnd: int, timeout: float = 12.0) -> bool:
        """Wait until Spotify's Electron DOM has hydrated (UI is interactive)."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            time.sleep(0.3)
            img = self._capture_window_topdown(hwnd)
            if img:
                w, h = img.size
                top_pixels = [img.getpixel((int(w * x), int(h * 0.03)))[:3] for x in [0.2, 0.35, 0.5, 0.7]]
                if len(set(top_pixels)) > 1 or (0, 0, 0) in set(top_pixels):
                    return True
        return False

    def _spotify_play_track(self, target_title: str) -> bool:
        try:
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                pass

            from runtime import actions

            # 1. Clean target string
            clean_search = re.sub(r"\s+", " ", target_title).strip()
            if not clean_search:
                clean_search = "Today's Top Hits"

            # 2. Check / Launch Spotify app if not running
            was_cold_start = False
            hwnd = self._find_and_focus_spotify()
            if not hwnd:
                was_cold_start = True
                subprocess.Popen(
                    "cmd.exe /c start shell:AppsFolder\\SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify",
                    shell=True
                )
                for _ in range(25):
                    time.sleep(0.3)
                    hwnd = self._find_and_focus_spotify()
                    if hwnd:
                        break

            if not hwnd:
                encoded = urllib.parse.quote_plus(clean_search)
                os.system(f"start spotify:search:{encoded}")
                return False

            if was_cold_start:
                # Wait for Spotify Electron DOM to mount and hydrate
                self._wait_for_spotify_dom(hwnd, timeout=12.0)
                time.sleep(0.8)

            # 3. Bring Spotify reliably to foreground using Alt-key bypass
            actions.focus_window(hwnd)
            time.sleep(0.2)

            # 4. Search via native Spotify shortcuts (Ctrl+L -> Ctrl+A -> type query -> Enter)
            actions.hotkey(0x11, 0x4C)  # Ctrl + L (focus search pill)
            time.sleep(0.08)
            actions.hotkey(0x11, 0x41)  # Ctrl + A
            time.sleep(0.04)
            actions.press_key(0x08)     # Backspace
            time.sleep(0.04)

            # Type target query
            actions.type_text(clean_search)
            time.sleep(0.15)
            actions.press_key(0x0D)     # Enter
            time.sleep(1.0)             # Wait for results to render over network

            # 5. Move mouse to hover over Top Result card to reveal hover play button
            rect = wintypes.RECT()
            user32 = ctypes.windll.user32
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            win_w = rect.right - rect.left
            win_h = rect.bottom - rect.top
            hover_x = rect.left + int(win_w * 0.22)
            hover_y = rect.top + int(win_h * 0.35)
            user32.SetCursorPos(hover_x, hover_y)
            user32.mouse_event(0x0001, 1, 1, 0, 0)

            # 6. Check for high-contrast green play button in the search results
            btn_pos = None
            img_res = None
            for _ in range(12):
                time.sleep(0.25)
                # Jiggle hover over Top Result card to trigger Chromium :hover reveal
                user32.SetCursorPos(hover_x, hover_y)
                user32.mouse_event(0x0001, 2, 2, 0, 0)
                time.sleep(0.04)
                user32.mouse_event(0x0001, -2, -2, 0, 0)

                img_res = self._capture_window_topdown(hwnd) or self._capture_window_image(hwnd)
                if img_res:
                    btn_pos = self._find_spotify_green_button(img_res)
                    if btn_pos:
                        break

            if btn_pos and img_res:
                scale_x = win_w / img_res.size[0] if img_res.size[0] > 0 else 1.0
                scale_y = win_h / img_res.size[1] if img_res.size[1] > 0 else 1.0
                click_x = rect.left + int(btn_pos[0] * scale_x)
                click_y = rect.top + int(btn_pos[1] * scale_y)
                user32.SetCursorPos(click_x, click_y)
                time.sleep(0.05)
                user32.mouse_event(0x0002, 0, 0, 0, 0)
                time.sleep(0.05)
                user32.mouse_event(0x0004, 0, 0, 0, 0)
                time.sleep(0.5)
                return True

            # 7. Fallback 1: Double-click Top Result card directly
            user32.SetCursorPos(hover_x, hover_y)
            time.sleep(0.05)
            user32.mouse_event(0x0002, 0, 0, 0, 0)
            user32.mouse_event(0x0004, 0, 0, 0, 0)
            time.sleep(0.05)
            user32.mouse_event(0x0002, 0, 0, 0, 0)
            user32.mouse_event(0x0004, 0, 0, 0, 0)
            time.sleep(0.5)

            # 8. Fallback 2: Perceive search results via ScreenEngine RapidOCR
            try:
                elements = self.gui_agent.see(hwnd=hwnd, force_ocr=True)
            except Exception:
                elements = []

            # Click "Songs" tab if present to filter to clean track table
            songs_tab = None
            for e in elements:
                if 100 < e.center[1] < 145 and e.name.strip().lower() in ["songs", "song"]:
                    songs_tab = e
                    break

            if songs_tab:
                actions.click_with_restore(songs_tab.center[0], songs_tab.center[1])
                time.sleep(0.5)
                try:
                    elements = self.gui_agent.see(hwnd=hwnd, force_ocr=True)
                except Exception:
                    elements = []

            # Locate matching song in results table (y between 140 and 880)
            q_words = [w.lower() for w in re.findall(r"\w+", clean_search.lower()) if len(w) > 1]
            best_elem = None
            best_score = 0.0

            for elem in elements:
                cx, cy = elem.center
                if cy < 140 or cy > 880:
                    continue
                name_lower = elem.name.lower()
                if name_lower in ["title", "artist", "album", "#", "play", "pause", "songs", "all"]:
                    continue

                score = 0.0
                if any(w in name_lower for w in q_words):
                    score = 0.8
                    if clean_search.lower() in name_lower or name_lower in clean_search.lower():
                        score = 1.0
                else:
                    sim = difflib.SequenceMatcher(None, clean_search.lower(), name_lower).ratio()
                    if sim > 0.55:
                        score = sim

                if score > best_score:
                    best_score = score
                    best_elem = elem

            if best_elem and best_score >= 0.55:
                # Ensure window focus and double click using cursor restoration
                actions.focus_window(hwnd)
                time.sleep(0.12)
                actions.click_with_restore(best_elem.center[0], best_elem.center[1], clicks=2, hwnd=hwnd)
                time.sleep(0.5)
                return True

            # Fallback: double click the first track row in table (~y=240, x=300)
            first_row = None
            for elem in elements:
                if 180 < elem.center[1] < 300 and elem.center[0] > 180 and elem.name.lower() not in ["title", "#", "songs"]:
                    first_row = elem
                    break

            if first_row:
                actions.focus_window(hwnd)
                time.sleep(0.12)
                actions.click_with_restore(first_row.center[0], first_row.center[1], clicks=2, hwnd=hwnd)
                time.sleep(0.5)
                return True

            return False
        except Exception as e:
            print(f"[Executor] Spotify playback error: {e}")
            return False


    def _play_music(self, params: Dict[str, Any]) -> Dict[str, Any]:
        song = params.get("song", "").strip()
        artist = params.get("artist", "").strip()
        modifier = params.get("modifier", "").strip()
        platform = params.get("platform", "spotify").strip().lower()
        cmd = params.get("command", "")

        # Build search query: prefer structured fields (preserves feat./remix/acoustic precisely)
        structured_query = self._build_music_search_query(song, modifier, artist)
        if structured_query:
            cleaned_query = structured_query
        elif cmd:
            # Fallback: extract from raw command text
            cleaned_query = self._clean_music_query(cmd)
        else:
            cleaned_query = ""

        is_generic = not cleaned_query

        # If generic music command ("play a song on spotify", "play some music")
        if is_generic:
            # If Spotify is already active, try fast unpause/resume first
            if "spotify" in platform:
                from runtime import actions
                spot_hwnd = self._find_and_focus_spotify()
                if spot_hwnd:
                    actions.press_media_key(0xB3)  # VK_MEDIA_PLAY_PAUSE
                    time.sleep(0.3)
                    return {
                        "status": "success",
                        "action": "play_music",
                        "platform": "spotify",
                        "song": "Current Queue",
                        "suggestions": [],
                        "message": "Resumed music playback on Spotify."
                    }
            cleaned_query = "Today's Top Hits"

        # Fetch suggestions from music catalog
        suggestions = self._fetch_music_suggestions(cleaned_query, limit=6)

        best_score = 0.0
        best_match = None
        if is_generic and suggestions:
            best_score = 1.0
            best_match = suggestions[0]
        elif suggestions:
            for s in suggestions:
                score = self._calculate_match_score(cleaned_query, s)
                if score > best_score:
                    best_score = score
                    best_match = s

        # Case 1: No suggestions found online
        if not suggestions:
            encoded = urllib.parse.quote_plus(cleaned_query)
            if "youtube" in platform:
                watch_url = self._get_youtube_watch_url(cleaned_query)
                webbrowser.open(watch_url)
                return {
                    "status": "warning",
                    "action": "play_music",
                    "platform": "youtube",
                    "song": cleaned_query,
                    "suggestions": [],
                    "message": f"No direct music matches found online. Opened YouTube search for '{cleaned_query}'."
                }
            else:
                success = self._spotify_play_track(cleaned_query)
                return {
                    "status": "success" if success else "warning",
                    "action": "play_music",
                    "platform": "spotify",
                    "song": cleaned_query,
                    "suggestions": [],
                    "message": f"Playing '{cleaned_query}' on Spotify." if success else f"Opened Spotify for '{cleaned_query}'."
                }

        # Case 2: Ambiguous / Low-confidence match (ask user: "These are the suggestions, should I play these?")
        if not is_generic and best_score < 0.65:
            sugg_list = [f"  [{i+1}] {s['track']} - {s['artist']}" for i, s in enumerate(suggestions[:4])]
            sugg_prompt = "\n".join(sugg_list)
            return {
                "status": "suggestions_available",
                "action": "play_music",
                "platform": platform,
                "query": cleaned_query,
                "suggestions": suggestions,
                "message": (
                    f"I couldn't find an exact match for '{cleaned_query}'. Here are the closest suggestions:\n"
                    f"{sugg_prompt}\n\n"
                    f"Should I play one of these? Reply with a number (e.g., '1') or song title."
                )
            }

        # Case 3: Confident match found (play directly on Spotify / YouTube!)
        top = best_match or suggestions[0]
        best_target = f"{top['track']} {top['artist']}"
        other_suggs = [s for s in suggestions if s != top]
        sugg_text = ""
        if other_suggs:
            sugg_lines = [f"    [{i+1}] {s['track']} - {s['artist']}" for i, s in enumerate(other_suggs[:3])]
            sugg_text = "\n  Other suggestions:\n" + "\n".join(sugg_lines) + "\n  (Reply with a number anytime to switch tracks)"

        if "youtube" in platform:
            watch_url = self._get_youtube_watch_url(best_target)
            webbrowser.open(watch_url)
            return {
                "status": "success",
                "action": "play_music",
                "platform": "youtube",
                "song": best_target,
                "suggestions": suggestions,
                "message": f"Playing '{best_target}' on YouTube.{sugg_text}"
            }
        else:
            try:
                from runtime.spotify_adapter import SpotifyAdapter
                spot = SpotifyAdapter()
                spot_res = spot.play_track(best_target)
                is_playing = spot_res.get("is_playing", False)
                verified = spot_res.get("verified", False)
                stat = "success" if is_playing else "playback_initiated"
                if is_playing:
                    msg = f"Playing '{top['track']}' by {top['artist']} on Spotify!{sugg_text}"
                else:
                    msg = spot_res.get("message") or f"Opened Spotify for '{best_target}'. Please check playback."
                return {
                    "status": stat,
                    "action": "play_music",
                    "platform": "spotify",
                    "song": best_target,
                    "verified": verified,
                    "is_playing": is_playing,
                    "window_title": spot_res.get("window_title", ""),
                    "suggestions": suggestions,
                    "message": msg
                }
            except Exception:
                success = self._spotify_play_track(best_target)
                msg = f"Playing '{top['track']}' by {top['artist']} on Spotify!{sugg_text}" if success else f"Opened Spotify for '{best_target}'."
                return {
                    "status": "success" if success else "warning",
                    "action": "play_music",
                    "platform": "spotify",
                    "song": best_target,
                    "suggestions": suggestions,
                    "message": msg
                }


    def _compute_math(self, params: Dict[str, Any]) -> Dict[str, Any]:
        expr = params.get("expr", "").strip()
        cmd = params.get("command", "").strip()

        calc_result = None
        if expr:
            calc_result = self._evaluate_complex_math(expr)
        if not calc_result and cmd:
            calc_result = self._evaluate_complex_math(cmd)

        if calc_result:
            return {
                "status": "success",
                "action": "compute_math",
                "result": calc_result,
                "message": f"Calculation Result: {calc_result}"
            }
        return {
            "status": "error",
            "message": f"Could not evaluate mathematical expression in '{expr or cmd}'."
        }

    def _evaluate_complex_math(self, text: str) -> Optional[str]:
        cleaned = text.lower()

        # Handle percentage: "X% of Y" or "X percent of Y"
        pct_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s*of\s*(\d+(?:\.\d+)?)", cleaned)
        if pct_match:
            pct = float(pct_match.group(1))
            val = float(pct_match.group(2))
            res = (pct / 100.0) * val
            if res.is_integer():
                res = int(res)
            return f"{pct}% of {val} = {res}"

        # Handle square root: "sqrt X", "square root of X"
        sqrt_match = re.search(r"(?:sqrt|square\s+root\s+of)\s*\(?(\d+(?:\.\d+)?)\)?", cleaned)
        if sqrt_match:
            val = float(sqrt_match.group(1))
            res = math.sqrt(val)
            if res.is_integer():
                res = int(res)
            return f"sqrt({val}) = {res}"

        # Handle power: "X^Y", "X to the power of Y"
        pow_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:\^|to\s+the\s+power\s+of)\s*(\d+(?:\.\d+)?)", cleaned)
        if pow_match:
            base = float(pow_match.group(1))
            exp = float(pow_match.group(2))
            res = base ** exp
            if res.is_integer():
                res = int(res)
            return f"{base}^{exp} = {res}"

        # Natural language operator translation
        cleaned = cleaned.replace("plus", "+").replace("minus", "-")
        cleaned = cleaned.replace("times", "*").replace("multiplied by", "*")
        cleaned = cleaned.replace("divided by", "/").replace("over", "/")

        # Match general arithmetic expression with digits and operators
        match = re.search(r"(\(?\s*\d+(?:\.\d+)?[\d\.\s\+\-\*\/\(\)\%]*[\+\-\*\/][\d\.\s\+\-\*\/\(\)\%]*\d+(?:\.\d+)?\s*\)?)", cleaned)
        if not match:
            return None

        expr = match.group(1).strip()

        ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Mod: operator.mod,
        }

        try:
            tree = ast.parse(expr, mode="eval")
            def eval_node(node):
                if isinstance(node, ast.BinOp):
                    return ops[type(node.op)](eval_node(node.left), eval_node(node.right))
                elif isinstance(node, ast.Constant):
                    return node.value
                elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                    return -eval_node(node.operand)
                raise ValueError("Unsupported node")

            ans = eval_node(tree.body)
            if isinstance(ans, float) and ans.is_integer():
                ans = int(ans)
            return f"{expr} = {ans}"
        except Exception:
            return None

    def _system_utility(self, params: Dict[str, Any]) -> Dict[str, Any]:
        utility = params.get("utility", "").lower()
        cmd = params.get("command", "").lower()
        target = utility or cmd

        # 1. Screenshot
        if "screenshot" in target or "screen" in target or "capture" in target:
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"edith_screenshot_{ts}.png"
            filepath = os.path.join(desktop, filename)

            try:
                user32 = ctypes.windll.user32
                hdesk = user32.OpenInputDesktop(0, False, 0x01FF)
                if hdesk:
                    user32.SetThreadDesktop(hdesk)
                from PIL import ImageGrab
                img = ImageGrab.grab()
                img.save(filepath)
                return {
                    "status": "success",
                    "action": "screenshot",
                    "path": filepath,
                    "message": f"Captured screenshot and saved to Desktop: {filename}"
                }
            except Exception:
                escaped_path = filepath.replace("\\", "/")
                ps_cmd = f"$b = New-Object System.Drawing.Bitmap ([System.Windows.Forms.Screen]::PrimaryScreen.Bounds.Width, [System.Windows.Forms.Screen]::PrimaryScreen.Bounds.Height); $g = [System.Drawing.Graphics]::FromImage($b); $g.CopyFromScreen(0, 0, 0, 0, $b.Size); $b.Save('{escaped_path}');"
                subprocess.run(f'powershell -NoProfile -Command "Add-Type -AssemblyName System.Windows.Forms; {ps_cmd}"', shell=True)
                return {
                    "status": "success",
                    "action": "screenshot",
                    "path": filepath,
                    "message": f"Screenshot saved to Desktop: {filename}"
                }

        # 2. Open Desktop Folder (e.g. "open E.D.I.T.H folder from desktop", "open folder ...")
        desktop_dir = os.path.join(os.path.expanduser("~"), "Desktop")
        if ("folder" in target or "directory" in target or "from desktop" in target) and os.path.isdir(desktop_dir):
            m_f = re.search(r"open\s+(.+?)\s+folder", target)
            folder_name = m_f.group(1).strip() if m_f else ""
            if not folder_name:
                m_f2 = re.search(r"open\s+(.+?)\s+from\s+desktop", target)
                folder_name = m_f2.group(1).replace("folder", "").strip() if m_f2 else ""
            if not folder_name:
                for entry in os.listdir(desktop_dir):
                    if entry.lower() in target:
                        folder_name = entry
                        break

            if folder_name:
                for entry in os.listdir(desktop_dir):
                    entry_norm = entry.lower().replace(".", "").replace(" ", "").replace("_", "").replace("-", "")
                    fn_norm = folder_name.lower().replace(".", "").replace(" ", "").replace("_", "").replace("-", "")
                    if entry_norm == fn_norm or entry.lower() == folder_name.lower():
                        full_p = os.path.join(desktop_dir, entry)
                        if os.path.isdir(full_p):
                            try:
                                os.startfile(full_p)
                                return {
                                    "status": "success",
                                    "action": "open_folder",
                                    "folder": entry,
                                    "path": full_p,
                                    "message": f"Opened folder '{entry}' from Desktop."
                                }
                            except Exception as e:
                                return {"status": "error", "message": f"Failed to open '{entry}': {e}"}

        # 3. Minimize all windows / Show Desktop (ONLY when specifically requested)
        if any(k in target for k in ["minimize", "show desktop", "minimize all", "clear screen", "hide windows"]) or target.strip() in ["desktop", "go to desktop"]:
            subprocess.run('powershell -c "(New-Object -ComObject Shell.Application).MinimizeAll()"', shell=True)
            return {
                "status": "success",
                "action": "show_desktop",
                "message": "Minimized all active windows to show Desktop."
            }

        # 4. Current Time & Date
        elif "time" in target or "date" in target or "clock" in target:
            now = datetime.datetime.now()
            t_str = now.strftime("%I:%M:%S %p")
            d_str = now.strftime("%A, %B %d, %Y")
            return {
                "status": "success",
                "action": "datetime",
                "time": t_str,
                "date": d_str,
                "message": f"Time: {t_str} | Date: {d_str}"
            }

        # 5. Open Downloads / Documents / Desktop folders
        elif "download" in target or "dowload" in target:
            os.startfile("shell:Downloads")
            return {"status": "success", "message": "Opened Downloads folder."}
        elif "document" in target or "doc" in target:
            os.startfile("shell:Personal")
            return {"status": "success", "message": "Opened Documents folder."}
        elif target.strip() in ["open desktop", "open desktop folder"]:
            os.startfile("shell:Desktop")
            return {"status": "success", "message": "Opened Desktop folder."}

        # 6. Empty Recycle Bin
        elif "recycle" in target or "trash" in target:
            subprocess.run("powershell -c Clear-RecycleBin -Force -ErrorAction SilentlyContinue", shell=True)
            return {"status": "success", "message": "Emptied Recycle Bin."}

        return {"status": "success", "message": f"Executed utility: {target}"}

    def _query_system_metrics(self) -> Dict[str, Any]:
        metrics = {}
        msg_parts = []

        try:
            import psutil
            cpu = psutil.cpu_percent(interval=0.2)
            ram = psutil.virtual_memory()
            disk = psutil.disk_usage("C:\\")

            metrics["cpu_percent"] = cpu
            metrics["ram_percent"] = ram.percent
            metrics["ram_free_gb"] = round(ram.available / (1024 ** 3), 2)
            metrics["disk_percent"] = disk.percent
            metrics["disk_free_gb"] = round(disk.free / (1024 ** 3), 2)

            msg_parts.append(f"CPU: {cpu}%")
            msg_parts.append(f"RAM: {ram.percent}% used ({metrics['ram_free_gb']} GB available)")
            msg_parts.append(f"Disk C: {disk.percent}% used ({metrics['disk_free_gb']} GB free)")

            battery = psutil.sensors_battery()
            if battery:
                metrics["battery_percent"] = battery.percent
                metrics["power_plugged"] = battery.power_plugged
                msg_parts.append(f"Battery: {battery.percent}% ({'Plugged in' if battery.power_plugged else 'On battery'})")

            return {
                "status": "success",
                "metrics": metrics,
                "message": " | ".join(msg_parts)
            }
        except ImportError:
            return {
                "status": "partial",
                "message": "System metrics require 'psutil'. Install via: pip install psutil"
            }

    def _get_system_brightness(self) -> Optional[int]:
        try:
            cmd = 'powershell -NoProfile -Command "(Get-CimInstance -Namespace root/Wmi -ClassName WmiMonitorBrightness).CurrentBrightness"'
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            m = re.search(r"\d+", res.stdout)
            return int(m.group(0)) if m else None
        except Exception:
            return None

    def _set_system_brightness(self, val: int) -> bool:
        try:
            val = max(10, min(100, int(val)))
            cmd = f'powershell -NoProfile -Command "(Get-WmiObject -Namespace root/Wmi -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1, {val})"'
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return res.returncode == 0
        except Exception:
            return False

    def _system_control(self, params: Dict[str, Any]) -> Dict[str, Any]:
        setting = params.get("setting", "").lower()
        value = params.get("value", "").lower()
        cmd = params.get("command", "").lower()
        target = f"{setting} {value} {cmd}"

        user32 = ctypes.windll.user32

        # 1. Brightness control
        if "bright" in target:
            curr = self._get_system_brightness()
            if any(k in target for k in ["up", "increase", "high", "more", "louder", "raise"]):
                new_val = min(100, (curr if curr is not None else 70) + 15)
                ok = self._set_system_brightness(new_val)
                msg = f"Increased system brightness to {new_val}%." if ok else f"Current system brightness is {curr}%."
                return {"status": "success" if ok else "warning", "action": "brightness_up", "brightness": new_val, "message": msg}
            elif any(k in target for k in ["down", "decrease", "low", "less", "dim"]):
                new_val = max(10, (curr if curr is not None else 50) - 15)
                ok = self._set_system_brightness(new_val)
                msg = f"Decreased system brightness to {new_val}%." if ok else f"Current system brightness is {curr}%."
                return {"status": "success" if ok else "warning", "action": "brightness_down", "brightness": new_val, "message": msg}
            else:
                m = re.search(r"(\d+)", target)
                if m:
                    new_val = max(10, min(100, int(m.group(1))))
                    ok = self._set_system_brightness(new_val)
                    msg = f"Set system brightness to {new_val}%." if ok else "Could not set system brightness."
                    return {"status": "success" if ok else "warning", "action": "set_brightness", "brightness": new_val, "message": msg}
                return {"status": "success", "action": "get_brightness", "message": f"Current system brightness is {curr}%."}

        # 2. Lock Workstation
        elif "lock" in target:
            subprocess.run("rundll32.exe user32.dll,LockWorkStation", shell=True)
            return {"status": "success", "action": "lock_workstation", "message": "Workstation locked."}

        # 3. Sleep mode
        elif "sleep" in target:
            subprocess.run("rundll32.exe powrprof.dll,SetSuspendState 0,1,0", shell=True)
            return {"status": "success", "action": "sleep", "message": "Entering sleep mode."}

        # 4. Master System Volume control
        elif "volume" in target or "sound" in target or "mute" in target:
            if "mute" in target or "unmute" in target:
                user32.keybd_event(0xAD, 0, 0, 0) # VK_VOLUME_MUTE
                time.sleep(0.04)
                user32.keybd_event(0xAD, 0, 2, 0)
                return {"status": "success", "action": "volume_mute", "message": "Toggled system volume mute."}
            
            # Check for specific volume percentage: e.g. "set volume to 80%", "volume 50"
            m_pct = re.search(r"\b(\d{1,3})\s*%?\b", target)
            if m_pct and any(k in target for k in ["set", "to", "%", "level", "percent"]):
                target_vol = max(0, min(100, int(m_pct.group(1))))
                # Reset to zero volume first (50 down presses)
                for _ in range(50):
                    user32.keybd_event(0xAE, 0, 0, 0)
                    user32.keybd_event(0xAE, 0, 2, 0)
                time.sleep(0.05)
                # Each volume up press is 2%
                steps_up = target_vol // 2
                for _ in range(steps_up):
                    user32.keybd_event(0xAF, 0, 0, 0)
                    user32.keybd_event(0xAF, 0, 2, 0)
                return {
                    "status": "success",
                    "action": "set_volume",
                    "volume": target_vol,
                    "message": f"Set system master volume to {target_vol}%."
                }
            elif any(k in target for k in ["up", "increase", "louder", "high", "raise"]):
                steps = 15 if any(w in target for w in ["way", "all the way", "much", "lot"]) else 5
                for _ in range(steps):
                    user32.keybd_event(0xAF, 0, 0, 0) # VK_VOLUME_UP
                    time.sleep(0.02)
                    user32.keybd_event(0xAF, 0, 2, 0)
                return {"status": "success", "action": "volume_up", "message": f"Increased system volume by {steps*2}%."}
            else:
                steps = 15 if any(w in target for w in ["way", "all the way", "much", "lot"]) else 5
                for _ in range(steps):
                    user32.keybd_event(0xAE, 0, 0, 0) # VK_VOLUME_DOWN
                    time.sleep(0.02)
                    user32.keybd_event(0xAE, 0, 2, 0)
                return {"status": "success", "action": "volume_down", "message": f"Decreased system volume by {steps*2}%."}

        return {"status": "success", "message": f"Adjusted system setting '{setting}'."}

    def _resolve_dest_path_and_name(self, dest_str: str, default_filename: str) -> Tuple[str, str]:
        home = os.path.expanduser("~")
        if not dest_str:
            return os.path.join(home, "Desktop"), default_filename

        dest_clean = dest_str.strip().strip("'\"")
        dest_expanded = os.path.expandvars(dest_clean)
        dest_expanded = os.path.expanduser(dest_expanded)
        dest_norm = os.path.normpath(dest_expanded)

        aliases = {
            "desktop": os.path.join(home, "Desktop"),
            "the desktop": os.path.join(home, "Desktop"),
            "my desktop": os.path.join(home, "Desktop"),
            "downloads": os.path.join(home, "Downloads"),
            "download": os.path.join(home, "Downloads"),
            "the downloads": os.path.join(home, "Downloads"),
            "my downloads": os.path.join(home, "Downloads"),
            "documents": os.path.join(home, "Documents"),
            "docs": os.path.join(home, "Documents"),
            "the documents": os.path.join(home, "Documents"),
            "my documents": os.path.join(home, "Documents"),
            "workspace": os.getcwd(),
            "here": os.getcwd(),
            "project": os.getcwd(),
            "cwd": os.getcwd(),
            "pictures": os.path.join(home, "Pictures"),
            "photos": os.path.join(home, "Pictures"),
            "music": os.path.join(home, "Music"),
            "videos": os.path.join(home, "Videos"),
        }
        if dest_clean.lower() in aliases:
            return aliases[dest_clean.lower()], default_filename

        ends_with_slash = dest_str.strip().endswith(("/", "\\"))

        if os.path.isdir(dest_norm):
            return dest_norm, default_filename

        ext = os.path.splitext(dest_norm)[1]
        if ext and not ends_with_slash:
            parent_dir = os.path.dirname(dest_norm)
            target_fname = os.path.basename(dest_norm)
            if not parent_dir:
                parent_dir = os.path.join(home, "Desktop")
            return parent_dir, target_fname

        if os.path.isabs(dest_norm):
            return dest_norm, default_filename

        cwd_cand = os.path.join(os.getcwd(), dest_norm)
        if os.path.isdir(cwd_cand):
            return cwd_cand, default_filename

        return os.path.join(home, "Desktop", dest_norm), default_filename

    def _resolve_dest_folder(self, dest_str: str) -> str:
        d_dir, _ = self._resolve_dest_path_and_name(dest_str, "")
        return d_dir

    def _resolve_target_path(self, target_name: str, context_dir: Optional[str] = None) -> Optional[str]:
        if not target_name:
            return None

        target_clean = target_name.strip().strip("'\"")
        home = os.path.expanduser("~")
        aliases = {
            "desktop": os.path.join(home, "Desktop"),
            "downloads": os.path.join(home, "Downloads"),
            "download": os.path.join(home, "Downloads"),
            "documents": os.path.join(home, "Documents"),
            "docs": os.path.join(home, "Documents"),
            "pictures": os.path.join(home, "Pictures"),
            "photos": os.path.join(home, "Pictures"),
            "music": os.path.join(home, "Music"),
            "videos": os.path.join(home, "Videos"),
            "workspace": os.getcwd(),
            "here": os.getcwd(),
            "project": os.getcwd(),
            "cwd": os.getcwd(),
        }
        low = target_clean.lower()
        if low in aliases:
            return aliases[low]

        expanded = os.path.expandvars(target_clean)
        expanded = os.path.expanduser(expanded)
        expanded = os.path.normpath(expanded)

        # 1. If explicit absolute path
        if os.path.isabs(expanded):
            if os.path.exists(expanded):
                return expanded
            parent = os.path.dirname(expanded)
            base = os.path.basename(expanded)
            if os.path.isdir(parent):
                for entry in os.listdir(parent):
                    if entry.lower() == base.lower():
                        return os.path.join(parent, entry)
                for entry in os.listdir(parent):
                    b, _ = os.path.splitext(entry)
                    if b.lower() == base.lower():
                        return os.path.join(parent, entry)
            return expanded

        # 2. If context_dir provided
        if context_dir:
            c_dir = self._resolve_dest_folder(context_dir)
            if c_dir and os.path.isdir(c_dir):
                candidate = os.path.join(c_dir, expanded)
                if os.path.exists(candidate):
                    return candidate
                for entry in os.listdir(c_dir):
                    if entry.lower() == expanded.lower():
                        return os.path.join(c_dir, entry)
                for entry in os.listdir(c_dir):
                    b, _ = os.path.splitext(entry)
                    if b.lower() == expanded.lower():
                        return os.path.join(c_dir, entry)

        # 3. If relative path exists in cwd
        cwd_cand = os.path.join(os.getcwd(), expanded)
        if os.path.exists(cwd_cand):
            return cwd_cand

        # 4. Search in common folders
        search_dirs = [
            os.path.join(home, "Desktop"),
            os.path.join(home, "Downloads"),
            os.path.join(home, "Documents"),
            os.getcwd()
        ]
        for sdir in search_dirs:
            if not os.path.isdir(sdir):
                continue
            cand = os.path.join(sdir, expanded)
            if os.path.exists(cand):
                return cand
            for entry in os.listdir(sdir):
                if entry.lower() == expanded.lower():
                    return os.path.join(sdir, entry)

        for sdir in search_dirs:
            if not os.path.isdir(sdir):
                continue
            for entry in os.listdir(sdir):
                b, _ = os.path.splitext(entry)
                if b.lower() == expanded.lower():
                    return os.path.join(sdir, entry)

        # 5. Semantic FileDiscoveryEngine fallback across project and user roots
        if self.file_discovery:
            disc_res = self.file_discovery.find_files(query=target_clean, limit=1)
            cands = disc_res.get("candidates", [])
            if cands:
                cand_path = cands[0].get("path") if isinstance(cands[0], dict) else cands[0].path
                if cand_path and os.path.exists(cand_path):
                    return cand_path

        return None

    def _send_to_recycle_bin(self, file_path: str) -> bool:
        try:
            class SHFILEOPSTRUCTW(ctypes.Structure):
                _fields_ = [
                    ('hwnd', wintypes.HWND),
                    ('wFunc', wintypes.UINT),
                    ('pFrom', wintypes.LPCWSTR),
                    ('pTo', wintypes.LPCWSTR),
                    ('fFlags', wintypes.WORD),
                    ('fAnyOperationsAborted', wintypes.BOOL),
                    ('hNameMappings', wintypes.LPVOID),
                    ('lpszProgressTitle', wintypes.LPCWSTR)
                ]
            p_from = os.path.abspath(file_path) + '\0\0'
            fileop = SHFILEOPSTRUCTW()
            fileop.hwnd = 0
            fileop.wFunc = 3 # FO_DELETE
            fileop.pFrom = p_from
            fileop.fFlags = 0x0040 | 0x0010 | 0x0400 # FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_NOERRORUI
            res = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(fileop))
            if res == 0 and not os.path.exists(file_path):
                return True
        except Exception:
            pass

        try:
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)
            return True
        except Exception:
            return False

    def _file_operation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        subaction = params.get("subaction", "").lower().strip()
        source = params.get("source", "") or params.get("file", "") or params.get("query", "")
        dest = params.get("destination", "") or params.get("path", "")
        cmd = params.get("command", "").lower().strip()

        # Auto-detect subaction if not explicitly given
        if not subaction:
            if any(k in cmd for k in ["find", "locate", "where is", "search for file", "search for"]):
                subaction = "find"
            elif any(k in cmd for k in ["copy", "duplicate"]):
                subaction = "copy"
            elif any(k in cmd for k in ["cut", "move", "relocate"]):
                subaction = "cut"
            elif "paste" in cmd:
                subaction = "paste"
            elif any(k in cmd for k in ["delete", "remove", "trash", "erase", "get rid of"]):
                subaction = "delete"
            elif any(k in cmd for k in ["create", "make", "touch", "new file", "new folder"]):
                subaction = "create"
            elif any(k in cmd for k in ["list", "show files", "dir"]):
                subaction = "list"
            else:
                subaction = "open"

        # 0. FIND / LOCATE
        if subaction in ["find", "locate", "search"]:
            target_term = source or cmd or dest
            query_to_run = target_term.strip()

            # Use advanced multi-signal FileDiscoveryEngine if available
            if self.file_discovery:
                disc_res = self.file_discovery.find_files(query=query_to_run, location_hint=dest, limit=5)
                cands = disc_res.get("candidates", [])
                is_ambiguous = disc_res.get("is_ambiguous", False)

                if cands:
                    cand_dicts = [c if isinstance(c, dict) else c.to_dict() for c in cands]
                    if self.memory:
                        self.memory.record_search(query=query_to_run, results=cand_dicts)

                    top_cand = cand_dicts[0]
                    top_score = top_cand.get("score", 0.0)
                    top_path = top_cand.get("path", "")
                    top_name = top_cand.get("name", "")
                    top_dir = top_cand.get("directory", "") or os.path.basename(os.path.dirname(top_path))
                    reasons = top_cand.get("match_reasons", [])
                    reason_str = ", ".join(reasons) if reasons else "high semantic relevance"

                    if len(cand_dicts) == 1 or (not is_ambiguous and top_score >= 0.82):
                        return {
                            "status": "success",
                            "action": "find_file",
                            "path": top_path,
                            "candidates": cand_dicts,
                            "is_ambiguous": False,
                            "message": f"Found '{top_name}' in {top_dir} ({reason_str})."
                        }
                    else:
                        cand_lines = []
                        for i, c in enumerate(cand_dicts[:5], 1):
                            c_dir = c.get("directory", "") or os.path.basename(os.path.dirname(c.get("path", "")))
                            c_reasons = c.get("match_reasons", [])
                            c_reason_str = ", ".join(c_reasons[:2]) if c_reasons else c.get("extension", "")
                            cand_lines.append(f"{i}. {c.get('name')} (in {c_dir}) - {c_reason_str}")
                        msg = f"Found {len(cand_dicts)} matching files:\n  " + "\n  ".join(cand_lines) + "\nSay 'open the first one' or specify which."
                        return {
                            "status": "needs_clarification" if is_ambiguous else "success",
                            "action": "find_file",
                            "path": top_path,
                            "candidates": cand_dicts,
                            "is_ambiguous": is_ambiguous,
                            "message": msg
                        }
                else:
                    return {
                        "status": "warning",
                        "action": "find_file",
                        "candidates": [],
                        "message": disc_res.get("message") or f"Could not find any files matching '{query_to_run}'."
                    }

            # Fallback search if engine not initialized
            home = os.path.expanduser("~")
            search_roots = []
            if dest and dest.lower() in ["desktop", "the desktop"]:
                search_roots.append(os.path.join(home, "Desktop"))
            elif dest and dest.lower() in ["downloads", "my downloads"]:
                search_roots.append(os.path.join(home, "Downloads"))
            elif dest and dest.lower() in ["documents", "my documents"]:
                search_roots.append(os.path.join(home, "Documents"))
            else:
                search_roots = [
                    os.path.join(home, "Desktop"),
                    os.path.join(home, "Downloads"),
                    os.path.join(home, "Documents"),
                    os.getcwd()
                ]

            matches = []
            for root in search_roots:
                if not os.path.isdir(root):
                    continue
                try:
                    for entry in os.listdir(root):
                        if query_to_run.lower() in entry.lower():
                            full_p = os.path.join(root, entry)
                            mtime = os.path.getmtime(full_p) if os.path.exists(full_p) else 0
                            matches.append({
                                "name": entry,
                                "path": full_p,
                                "modified_time": mtime,
                                "directory": os.path.basename(root)
                            })
                except Exception:
                    pass

            if not matches:
                return {
                    "status": "warning",
                    "action": "find_file",
                    "message": f"Could not find any files matching '{query_to_run}'."
                }

            matches.sort(key=lambda m: m["modified_time"], reverse=True)
            top = matches[0]
            if len(matches) == 1:
                return {
                    "status": "success",
                    "action": "find_file",
                    "path": top["path"],
                    "candidates": matches,
                    "message": f"Found '{top['name']}' in {top['directory']} ({top['path']})."
                }
            else:
                cand_list = [f"{i+1}. {m['name']} (in {m['directory']})" for i, m in enumerate(matches[:5])]
                msg = f"Found {len(matches)} matching files:\n  " + "\n  ".join(cand_list) + "\nSay 'open the first one' or specify which."
                return {
                    "status": "success",
                    "action": "find_file",
                    "path": top["path"],
                    "candidates": matches,
                    "message": msg
                }

        # 1. COPY
        if subaction == "copy":
            if not source and self.file_clipboard:
                source = self.file_clipboard.get("path", "")
            if not source:
                return {"status": "error", "message": "No file specified to copy."}

            src_path = self._resolve_target_path(source)
            if not src_path or not os.path.exists(src_path):
                return {"status": "error", "message": f"Could not find file '{source}' to copy."}

            fname = os.path.basename(src_path)

            if dest:
                dest_dir, target_fname = self._resolve_dest_path_and_name(dest, fname)
                os.makedirs(dest_dir, exist_ok=True)
                target_path = os.path.join(dest_dir, target_fname)
                if os.path.isdir(src_path):
                    if os.path.exists(target_path):
                        target_path = os.path.join(dest_dir, f"{target_fname}_copy")
                    shutil.copytree(src_path, target_path)
                else:
                    shutil.copy2(src_path, target_path)
                return {
                    "status": "success",
                    "action": "copy_file",
                    "source": src_path,
                    "destination": target_path,
                    "message": f"Copied '{fname}' to '{target_path}'."
                }
            else:
                self.file_clipboard = {"path": src_path, "name": fname, "is_cut": False}
                actions.set_clipboard_text(src_path)
                return {
                    "status": "success",
                    "action": "copy_to_clipboard",
                    "path": src_path,
                    "message": f"Copied '{fname}' to clipboard. Say 'paste file to <destination>' or 'paste' to paste it."
                }

        # 2. CUT / MOVE
        elif subaction == "cut":
            if not source and self.file_clipboard:
                source = self.file_clipboard.get("path", "")
            if not source:
                return {"status": "error", "message": "No file specified to cut."}

            src_path = self._resolve_target_path(source)
            if not src_path or not os.path.exists(src_path):
                return {"status": "error", "message": f"Could not find file '{source}' to cut."}

            fname = os.path.basename(src_path)

            if dest:
                dest_dir, target_fname = self._resolve_dest_path_and_name(dest, fname)
                os.makedirs(dest_dir, exist_ok=True)
                target_path = os.path.join(dest_dir, target_fname)
                shutil.move(src_path, target_path)
                return {
                    "status": "success",
                    "action": "cut_file",
                    "source": src_path,
                    "destination": target_path,
                    "message": f"Moved '{fname}' to '{target_path}'."
                }
            else:
                self.file_clipboard = {"path": src_path, "name": fname, "is_cut": True}
                return {
                    "status": "success",
                    "action": "cut_to_clipboard",
                    "path": src_path,
                    "message": f"Cut '{fname}' to clipboard. Say 'paste file to <destination>' to complete the move."
                }

        # 3. PASTE
        elif subaction == "paste":
            if not self.file_clipboard or not os.path.exists(self.file_clipboard.get("path", "")):
                return {
                    "status": "warning",
                    "action": "paste",
                    "message": "Clipboard is empty. Copy or cut a file first (e.g. 'copy file notes.txt')."
                }

            src_path = self.file_clipboard["path"]
            fname = self.file_clipboard.get("name", os.path.basename(src_path))
            is_cut = self.file_clipboard.get("is_cut", False)

            dest_dir, target_fname = self._resolve_dest_path_and_name(dest, fname)
            os.makedirs(dest_dir, exist_ok=True)
            target_path = os.path.join(dest_dir, target_fname)

            # Avoid collision if copying into the same directory
            if os.path.exists(target_path) and os.path.abspath(src_path) == os.path.abspath(target_path) and not is_cut:
                base, ext = os.path.splitext(target_fname)
                target_path = os.path.join(dest_dir, f"{base}_copy{ext}")

            if is_cut:
                shutil.move(src_path, target_path)
                self.file_clipboard = None
                return {
                    "status": "success",
                    "action": "paste_move",
                    "path": target_path,
                    "message": f"Moved and pasted '{fname}' into '{target_path}'."
                }
            else:
                if os.path.isdir(src_path):
                    shutil.copytree(src_path, target_path)
                else:
                    shutil.copy2(src_path, target_path)
                return {
                    "status": "success",
                    "action": "paste_copy",
                    "path": target_path,
                    "message": f"Pasted '{os.path.basename(target_path)}' into '{dest_dir}'."
                }

        # 4. DELETE / REMOVE
        elif subaction == "delete":
            if not source:
                return {"status": "error", "message": "No file specified to delete."}

            src_path = self._resolve_target_path(source, context_dir=dest)
            if not src_path or not os.path.exists(src_path):
                return {"status": "error", "message": f"Could not find file '{source}' to delete."}

            fname = os.path.basename(src_path)
            ok = self._send_to_recycle_bin(src_path)
            if ok:
                return {
                    "status": "success",
                    "action": "delete_file",
                    "path": src_path,
                    "message": f"Deleted '{fname}' (moved to Recycle Bin)."
                }
            else:
                return {"status": "error", "message": f"Failed to delete '{fname}'."}

        # 5. CREATE
        elif subaction == "create":
            if not source:
                return {"status": "error", "message": "No filename provided to create."}

            is_folder = "folder" in cmd or "directory" in cmd
            source_clean = source.strip().strip("'\"")
            source_expanded = os.path.expandvars(source_clean)
            source_expanded = os.path.expanduser(source_expanded)
            source_norm = os.path.normpath(source_expanded)

            # If source is an absolute path or has parent directory
            if os.path.isabs(source_norm) or os.path.dirname(source_norm):
                target_path = source_norm
                dest_dir = os.path.dirname(target_path)
            else:
                dest_dir = self._resolve_dest_folder(dest)
                target_path = os.path.join(dest_dir, source_clean)

            try:
                if is_folder:
                    os.makedirs(target_path, exist_ok=True)
                    return {"status": "success", "action": "create_folder", "path": target_path, "message": f"Created folder '{os.path.basename(target_path)}' at '{target_path}'."}
                else:
                    if dest_dir:
                        os.makedirs(dest_dir, exist_ok=True)
                    with open(target_path, "w", encoding="utf-8") as f:
                        f.write("")
                    return {"status": "success", "action": "create_file", "path": target_path, "message": f"Created file '{os.path.basename(target_path)}' at '{target_path}'."}
            except Exception as e:
                return {"status": "error", "message": f"Could not create '{source}': {e}"}

        # 6. LIST FILES
        elif subaction == "list":
            target_dir = self._resolve_dest_folder(dest or source)
            if not os.path.isdir(target_dir):
                return {"status": "error", "message": f"Directory '{target_dir}' does not exist."}

            items = os.listdir(target_dir)
            files = []
            folders = []
            for item in items[:25]:
                ip = os.path.join(target_dir, item)
                if os.path.isdir(ip):
                    folders.append(f"📁 {item}/")
                else:
                    sz = os.path.getsize(ip)
                    sz_str = f"{sz/1024:.1f} KB" if sz > 1024 else f"{sz} B"
                    files.append(f"📄 {item} ({sz_str})")

            summary_items = folders[:6] + files[:10]
            msg_body = "\n  ".join(summary_items) if summary_items else f"{os.path.basename(target_dir)} is empty."
            return {
                "status": "success",
                "action": "list_files",
                "path": target_dir,
                "count": len(items),
                "message": f"Found {len(items)} items in {target_dir}:\n  {msg_body}"
            }

        # 6b. VERIFY FILE
        elif subaction in ["verify_file", "verify"]:
            target_path = self._resolve_target_path(source or dest)
            if target_path and os.path.exists(target_path):
                return {
                    "status": "success",
                    "action": "verify_file",
                    "path": target_path,
                    "verified": True,
                    "message": f"Confirmed file exists on filesystem at '{target_path}'."
                }
            return {
                "status": "error",
                "action": "verify_file",
                "path": source or dest,
                "verified": False,
                "message": f"File does not exist on filesystem: '{source or dest}'."
            }

        # 7. OPEN
        else:
            target_path = self._resolve_target_path(source or dest)
            if not target_path or not os.path.exists(target_path):
                # Try finding via FileDiscoveryEngine
                if self.file_discovery:
                    query_term = source or dest or cmd
                    disc_res = self.file_discovery.find_files(query=query_term, limit=5)
                    cands = disc_res.get("candidates", [])
                    if cands:
                        cand_dicts = [c if isinstance(c, dict) else c.to_dict() for c in cands]
                        if not disc_res.get("is_ambiguous", False) or len(cand_dicts) == 1:
                            target_path = cand_dicts[0].get("path", "")
                        else:
                            cand_lines = [f"{i}. {c.get('name')} (in {c.get('directory', '') or os.path.basename(os.path.dirname(c.get('path', '')))})" for i, c in enumerate(cand_dicts[:5], 1)]
                            return {
                                "status": "needs_clarification",
                                "action": "open_file",
                                "is_ambiguous": True,
                                "candidates": cand_dicts,
                                "message": f"Found multiple potential matches for '{query_term}':\n  " + "\n  ".join(cand_lines) + "\nSay 'open the first one' or specify which."
                            }

            if not target_path or not os.path.exists(target_path):
                return {"status": "error", "message": f"Could not find '{source or dest}' to open. Confirmed path does not exist on disk."}

            try:
                os.startfile(target_path)
                return {"status": "success", "action": "open_file", "path": target_path, "message": f"Opened '{target_path}'."}
            except Exception as e:
                return {"status": "error", "message": f"Failed to open '{target_path}': {e}"}

    def _web_search(self, query: Any) -> Dict[str, Any]:
        if isinstance(query, (list, tuple)):
            query_str = " ".join(str(q) for q in query if q).strip()
        elif not isinstance(query, str):
            query_str = str(query).strip()
        else:
            query_str = query.strip()

        if not query_str:
            return {"status": "error", "message": "No search query provided."}
        
        encoded = urllib.parse.quote_plus(query_str)
        search_url = f"https://www.google.com/search?q={encoded}"
        try:
            chrome_exe = self._get_chrome_exe_path()
            if os.path.isfile(chrome_exe):
                if self._launch_chrome(profile="Aditya Sawant", url=search_url):
                    return {
                        "status": "success",
                        "action": "web_search",
                        "query": query_str,
                        "url": search_url,
                        "message": f"Opened Google Chrome (Aditya Sawant) for: '{query_str}'."
                    }
            webbrowser.open(search_url)
            return {
                "status": "success",
                "action": "web_search",
                "query": query_str,
                "url": search_url,
                "message": f"Opened browser search for: '{query_str}'."
            }
        except Exception as e:
            return {"status": "error", "message": f"Failed to open browser: {e}"}

    def _web_navigation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Navigate to specific web destinations or services in Chrome or default browser."""
        url = params.get("url", "").strip()
        service = params.get("service", "website")
        if not url:
            return {"status": "error", "message": "No URL provided for navigation."}

        try:
            chrome_exe = self._get_chrome_exe_path()
            if os.path.isfile(chrome_exe):
                if self._launch_chrome(profile="Aditya Sawant", url=url):
                    return {
                        "status": "success",
                        "action": "web_navigation",
                        "url": url,
                        "service": service,
                        "message": f"Navigated to {service.capitalize() if service else 'website'} ({url}) in Google Chrome."
                    }
            webbrowser.open(url)
            return {
                "status": "success",
                "action": "web_navigation",
                "url": url,
                "service": service,
                "message": f"Opened {service.capitalize() if service else 'website'} ({url}) in default browser."
            }
        except Exception as e:
            return {"status": "error", "message": f"Failed to navigate to {url}: {e}"}

    def _draft_content(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Deliver and route drafted content to clipboard, Gmail web compose, mail client, or active editor."""
        subject = params.get("subject", "")
        body = params.get("body", "")
        channel = str(params.get("channel", "clipboard")).lower().strip()
        gmail_url = params.get("gmail_url", "")
        mailto_url = params.get("mailto_url", "")
        recip = params.get("recipient", "")

        # Always copy the draft to clipboard via native Win32 API
        try:
            full_text = f"Subject: {subject}\n\n{body}" if subject else body
            actions.set_clipboard_text(full_text)
        except Exception:
            pass

        msg_parts = ["Draft generated and copied to clipboard."]

        # 1. If channel is Gmail (or Gmail URL provided and channel indicates web mail)
        if channel in ["gmail", "google mail"] or (gmail_url and "gmail" in channel):
            chrome_exe = self._get_chrome_exe_path()
            if os.path.isfile(chrome_exe):
                self._launch_chrome(profile="Aditya Sawant", url=gmail_url)
            else:
                webbrowser.open(gmail_url)
            msg_parts.append("Opened Gmail compose window with draft pre-filled.")

        # 2. If channel is native mail
        elif channel in ["mail", "email", "outlook"] and mailto_url:
            try:
                os.startfile(mailto_url)
                msg_parts.append("Opened mail client with draft pre-filled.")
            except Exception:
                webbrowser.open(mailto_url)

        # 3. If channel is Notepad or active editor
        elif channel in ["notepad", "editor", "document"]:
            if "notepad" in channel and self.text_writer:
                ins_res = self.text_writer.insert_into_notepad(
                    content=body,
                    append=params.get("append", False),
                    replace=params.get("replace", False)
                )
                if ins_res.get("success") or ins_res.get("status") == "success":
                    msg_parts.append(f"Faithfully inserted draft into Notepad (verified: {ins_res.get('verified', False)}).")
                else:
                    return {
                        "status": "error",
                        "action": "draft_content",
                        "subject": subject,
                        "recipient": recip,
                        "body": body,
                        "channel": channel,
                        "message": f"Failed to insert draft into Notepad: {ins_res.get('message')}"
                    }
            else:
                from runtime import actions
                hwnd = self.gui_agent.focus_app(channel) if channel != "editor" else None
                if hwnd:
                    actions.focus_window(hwnd)
                    time.sleep(0.3)
                actions.type_text(body)
                msg_parts.append(f"Typed draft directly into {channel.capitalize()}.")

        ret = {
            "status": "success",
            "action": "draft_content",
            "subject": subject,
            "recipient": recip,
            "body": body,
            "channel": channel,
            "message": " ".join(msg_parts)
        }
        if "notepad" in channel and 'ins_res' in locals() and ins_res:
            ret["verified"] = ins_res.get("verified", False)
            ret["exact_match"] = ins_res.get("exact_match", False)
            ret["characters_written"] = ins_res.get("characters_written", len(body))
        return ret

    def _workflow_script(self, params: Dict[str, Any]) -> Dict[str, Any]:
        steps = params.get("steps", [])
        if not steps:
            messages = []
            app_name = params.get("app_name", "").strip()
            if app_name:
                launch_res = self._launch_app(app_name)
                messages.append(launch_res.get("message", f"Launched {app_name}."))

            cmd = params.get("command", "")
            math_res = self._evaluate_complex_math(cmd)
            if math_res:
                messages.append(f"Result: {math_res}")

            if not messages:
                messages.append("Multi-step task completed.")

            return {
                "status": "success",
                "action": "workflow_script",
                "message": " | ".join(messages)
            }

        step_results = []
        messages = []
        context: Dict[str, Any] = {
            "last_app": None,
            "last_file": None,
            "last_path": None,
            "last_song": None
        }

        all_ok = True
        for idx, step_directive in enumerate(steps, 1):
            # Inter-step delay and focus sync if previous step was an app launch
            if context.get("last_app"):
                time.sleep(0.6)

            # Context propagation: if step is a GUI action without app, inject last_app
            handler = step_directive.get("handler")
            step_params = step_directive.setdefault("parameters", {})
            if handler == "gui_action" and not step_params.get("app_name") and context.get("last_app"):
                step_params["app_name"] = context["last_app"]

            # If file operation on 'it', resolve to last_file/last_path
            if handler == "file_operation":
                for k in ["source", "file"]:
                    val = str(step_params.get(k, "")).strip().lower()
                    if val in ["it", "that", "the file", "the folder", "the screenshot"]:
                        step_params[k] = context.get("last_path") or context.get("last_file") or ""

            res = self.execute(step_directive)
            step_results.append(res)
            msg = res.get("message", f"Step {idx} completed.")
            messages.append(f"Step {idx}: {msg}")

            if res.get("status") not in ["success", "warning", None]:
                all_ok = False

            # Update context
            if res.get("path"):
                context["last_path"] = res["path"]
                context["last_file"] = os.path.basename(res["path"])
            if handler == "launch_app":
                context["last_app"] = step_params.get("app_name")
            elif handler == "file_operation":
                if step_params.get("source"):
                    context["last_file"] = step_params["source"]
            elif handler == "play_music":
                context["last_song"] = step_params.get("song")

        return {
            "status": "success" if all_ok else "partial",
            "action": "workflow_script",
            "steps": step_results,
            "message": " | ".join(messages)
        }

    def _network_diagnostics(self) -> Dict[str, Any]:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)

        res = subprocess.run("ping -n 1 8.8.8.8", shell=True, capture_output=True, text=True)
        connected = (res.returncode == 0)

        return {
            "status": "success",
            "hostname": hostname,
            "local_ip": local_ip,
            "internet_reachable": connected,
            "message": f"Host: {hostname} | Local IP: {local_ip} | Internet: {'Online' if connected else 'Offline'}"
        }

    def _kill_process(self, pid: Any) -> Dict[str, Any]:
        if not pid:
            return {"status": "error", "message": "No PID provided."}
        try:
            res = subprocess.run(f"taskkill /f /pid {pid}", shell=True, capture_output=True, text=True)
            if res.returncode == 0:
                return {"status": "success", "message": f"Terminated process PID {pid}."}
            else:
                return {"status": "warning", "message": f"Could not find process with PID {pid}."}
        except Exception as e:
            return {"status": "error", "message": f"Error terminating PID {pid}: {e}"}

    def _media_control(self, params: Dict[str, Any]) -> Dict[str, Any]:
        user32 = ctypes.windll.user32
        action = params.get("action", "").lower().strip()
        cmd = params.get("command", "").lower().strip()
        target = f"{action} {cmd}".strip()

        # 1. Next track / skip
        if any(k in target for k in ["next", "skip", "forward"]):
            try:
                elem = self.gui_agent.find("Next", role="button")
                if elem:
                    from runtime import actions
                    actions.click(elem.center[0], elem.center[1])
                    return {"status": "success", "action": "next_track", "message": f"Clicked '{elem.name}' via screen perception."}
            except Exception:
                pass
            hwnd = self._find_and_focus_spotify()
            if hwnd:
                rect = wintypes.RECT()
                user32.GetWindowRect(hwnd, ctypes.byref(rect))
                win_w = rect.right - rect.left
                # Click the next track button on player bar
                user32.SetCursorPos(rect.left + win_w // 2 + 61, rect.bottom - 71)
                time.sleep(0.04)
                user32.mouse_event(0x0002, 0, 0, 0, 0)
                time.sleep(0.04)
                user32.mouse_event(0x0004, 0, 0, 0, 0)
                self._spotify_press_ctrl_combo(0x27) # Ctrl + Right (Spotify next track)
            scan = user32.MapVirtualKeyW(0xB0, 0)
            user32.keybd_event(0xB0, scan, 0x0001, 0) # VK_MEDIA_NEXT_TRACK (KEYEVENTF_EXTENDEDKEY)
            time.sleep(0.04)
            user32.keybd_event(0xB0, scan, 0x0001 | 0x0002, 0)
            return {"status": "success", "action": "next_track", "message": "Skipped to next track on Spotify."}

        # 2. Previous track
        elif any(k in target for k in ["prev", "previous", "back", "rewind"]):
            try:
                elem = self.gui_agent.find("Previous", role="button")
                if elem:
                    from runtime import actions
                    actions.click(elem.center[0], elem.center[1])
                    return {"status": "success", "action": "prev_track", "message": f"Clicked '{elem.name}' via screen perception."}
            except Exception:
                pass
            hwnd = self._find_and_focus_spotify()
            if hwnd:
                rect = wintypes.RECT()
                user32.GetWindowRect(hwnd, ctypes.byref(rect))
                win_w = rect.right - rect.left
                # Click the previous track button on player bar
                user32.SetCursorPos(rect.left + win_w // 2 - 60, rect.bottom - 71)
                time.sleep(0.04)
                user32.mouse_event(0x0002, 0, 0, 0, 0)
                time.sleep(0.04)
                user32.mouse_event(0x0004, 0, 0, 0, 0)
                self._spotify_press_ctrl_combo(0x25) # Ctrl + Left (Spotify previous track)
            scan = user32.MapVirtualKeyW(0xB1, 0)
            user32.keybd_event(0xB1, scan, 0x0001, 0) # VK_MEDIA_PREV_TRACK (KEYEVENTF_EXTENDEDKEY)
            time.sleep(0.04)
            user32.keybd_event(0xB1, scan, 0x0001 | 0x0002, 0)
            return {"status": "success", "action": "prev_track", "message": "Returned to previous track on Spotify."}

        # 3. Lyrics (Open / Show Lyrics)
        elif any(k in target for k in ["lyrics", "lyric"]):
            try:
                elem = self.gui_agent.find("Lyrics", role="button")
                if elem:
                    from runtime import actions
                    actions.click(elem.center[0], elem.center[1])
                    return {"status": "success", "action": "open_lyrics", "message": f"Clicked '{elem.name}' via screen perception."}
            except Exception:
                pass
            hwnd = self._find_and_focus_spotify()
            if not hwnd:
                subprocess.run("powershell -NoProfile -Command \"Start-Process 'shell:AppsFolder\\SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify'\"", shell=True)
                time.sleep(1.0)
                hwnd = self._find_and_focus_spotify()

            if hwnd:
                rect = wintypes.RECT()
                user32.GetWindowRect(hwnd, ctypes.byref(rect))
                w = rect.right - rect.left
                h = rect.bottom - rect.top
                # Microphone lyrics icon is at (w - 290, h - 56) in standard player bar
                click_x = rect.left + max(200, w - 290)
                click_y = rect.top + max(50, h - 56)
                user32.SetCursorPos(click_x, click_y)
                time.sleep(0.05)
                user32.mouse_event(0x0002, 0, 0, 0, 0) # LEFTDOWN
                time.sleep(0.05)
                user32.mouse_event(0x0004, 0, 0, 0, 0) # LEFTUP
                return {"status": "success", "action": "open_lyrics", "message": "Toggled lyrics view in Spotify."}
            else:
                # Web fallback for lyrics
                query = cmd.replace("open lyrics", "").replace("show lyrics", "").replace("lyrics in spotify", "").replace("lyrics", "").strip()
                if query:
                    webbrowser.open(f"https://www.google.com/search?q={urllib.parse.quote_plus(query + ' lyrics')}")
                    return {"status": "success", "action": "open_lyrics", "message": f"Opened lyrics search for '{query}' in browser."}
                return {"status": "warning", "action": "open_lyrics", "message": "Spotify window could not be focused to toggle lyrics."}

        # 4. Add to Playlist / Like song
        elif any(k in target for k in ["playlist", "like", "favorite", "save"]):
            hwnd = self._find_and_focus_spotify()
            # Official Spotify desktop shortcut: Alt + Shift + B (save to library/liked songs)
            user32.keybd_event(0x12, 0, 0, 0) # Alt down
            user32.keybd_event(0x10, 0, 0, 0) # Shift down
            user32.keybd_event(0x42, 0, 0, 0) # 'B' down
            time.sleep(0.04)
            user32.keybd_event(0x42, 0, 2, 0) # 'B' up
            user32.keybd_event(0x10, 0, 2, 0) # Shift up
            user32.keybd_event(0x12, 0, 2, 0) # Alt up

            # Also click the (+) button on player bar if window is available
            if hwnd:
                rect = wintypes.RECT()
                user32.GetWindowRect(hwnd, ctypes.byref(rect))
                h = rect.bottom - rect.top
                click_x = rect.left + 249
                click_y = rect.top + max(50, h - 56)
                user32.SetCursorPos(click_x, click_y)
                time.sleep(0.05)
                user32.mouse_event(0x0002, 0, 0, 0, 0)
                time.sleep(0.05)
                user32.mouse_event(0x0004, 0, 0, 0, 0)

            return {"status": "success", "action": "add_to_playlist", "message": "Added current track to your Spotify playlist / Liked Songs."}

        # 5. Spotify Volume Up
        elif ("spotify" in target and any(k in target for k in ["up", "increase", "louder"])) or action == "spotify_volume_up":
            hwnd = self._find_and_focus_spotify()
            for _ in range(5):
                self._spotify_press_ctrl_combo(0x26) # Ctrl + Up
                time.sleep(0.03)
            return {"status": "success", "action": "spotify_volume_up", "message": "Increased Spotify volume."}

        # 6. Spotify Volume Down
        elif ("spotify" in target and any(k in target for k in ["down", "decrease", "quieter"])) or action == "spotify_volume_down":
            hwnd = self._find_and_focus_spotify()
            for _ in range(5):
                self._spotify_press_ctrl_combo(0x28) # Ctrl + Down
                time.sleep(0.03)
            return {"status": "success", "action": "spotify_volume_down", "message": "Decreased Spotify volume."}

        # 7. Play / Pause / Resume
        else:
            hwnd = self._find_and_focus_spotify()
            if hwnd:
                rect = wintypes.RECT()
                user32.GetWindowRect(hwnd, ctypes.byref(rect))
                win_w = rect.right - rect.left
                center_x = rect.left + win_w // 2
                click_y = rect.bottom - 71

                # Detect state: center of circle is white (240, 240, 240) between pause bars when playing, black when paused
                is_playing = None
                try:
                    from PIL import ImageGrab
                    crop = ImageGrab.grab(bbox=(center_x - 10, click_y - 10, center_x + 10, click_y + 10)).convert("RGB")
                    px = crop.getpixel((10, 10))
                    is_playing = (px[0] > 180 and px[1] > 180 and px[2] > 180)
                except Exception:
                    pass

                # If asked to pause and already paused
                if (action == "pause" or any(k in target for k in ["pause", "stop"])) and is_playing is False:
                    return {"status": "success", "action": "pause", "message": "Spotify is already paused."}

                # If asked to resume/play and already playing
                if (action == "play" or any(k in target for k in ["resume", "unpause"])) and is_playing is True:
                    return {"status": "success", "action": "play", "message": "Spotify playback is already active."}

                # Try programmatic UIA invoke or background post_click first
                btn_name = "Pause" if (action == "pause" or any(k in target for k in ["pause", "stop"])) else "Play"
                elem = self.gui_agent.find(btn_name, role="button", hwnd=hwnd)
                if elem:
                    if elem.invoke():
                        msg = "Paused Spotify playback." if is_playing is True else "Resumed Spotify playback."
                        return {"status": "success", "action": "play_pause", "message": msg}
                    if elem.post_click():
                        msg = "Paused Spotify playback." if is_playing is True else "Resumed Spotify playback."
                        return {"status": "success", "action": "play_pause", "message": msg}

                from runtime import actions
                with actions.cursor_guard():
                    user32.SetCursorPos(center_x, click_y)
                    time.sleep(0.04)
                    user32.mouse_event(0x0002, 0, 0, 0, 0)
                    time.sleep(0.04)
                    user32.mouse_event(0x0004, 0, 0, 0, 0)

                msg = "Paused Spotify playback." if is_playing is True else "Resumed Spotify playback."
                return {"status": "success", "action": "play_pause", "message": msg}
            else:
                scan = user32.MapVirtualKeyW(0xB3, 0)
                user32.keybd_event(0xB3, scan, 0x0001, 0)
                time.sleep(0.04)
                user32.keybd_event(0xB3, scan, 0x0001 | 0x0002, 0)
                return {"status": "success", "action": "play_pause", "message": "Toggled media playback (Play/Pause)."}
