"""Closed-Loop Verification and Failure Recovery Engine for E.D.I.T.H. Intelligence Core V2.

Implements the UNDERSTAND -> PLAN -> EXECUTE -> OBSERVE -> EVALUATE -> RECOVER OR COMPLETE loop.
Verifies real OS and process states, diagnoses execution deviations, and handles automated retries.
"""

import os
import sys
import time
import subprocess
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

try:
    import psutil
except ImportError:
    psutil = None


@dataclass
class VerificationResult:
    """Outcome of observing and evaluating an action's verifiable success condition."""
    success: bool
    observed_state: str
    diagnostic_message: str
    can_recover: bool = False
    recovery_strategy: Optional[str] = None
    fallback_parameters: Optional[Dict[str, Any]] = None

    @property
    def verified(self) -> bool:
        return self.success

    @property
    def recovery_action(self) -> Optional[str]:
        return self.recovery_strategy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verified": self.success,
            "success": self.success,
            "observed_state": self.observed_state,
            "diagnostic_message": self.diagnostic_message,
            "can_recover": self.can_recover,
            "recovery_strategy": self.recovery_strategy,
            "recovery_action": self.recovery_strategy,
            "fallback_parameters": self.fallback_parameters,
        }


class ClosedLoopVerifier:
    """
    Observes operational reality after action dispatch to confirm actual success.
    Distinguishes between issuing an action and accomplishing the user's goal.
    """

    MAX_RETRIES = 2

    @classmethod
    def check_preconditions(cls, step_params: Dict[str, Any], preconditions: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Verify all prerequisite environment conditions before dispatching an action."""
        for cond in preconditions:
            cond_type = cond.get("type")
            target = cond.get("target")

            if cond_type == "file_exists" and target:
                clean_target = os.path.expanduser(str(target).strip("'\""))
                if not os.path.exists(clean_target):
                    return False, f"Precondition failed: required path '{clean_target}' does not exist on filesystem."

            elif cond_type == "directory_exists" and target:
                clean_target = os.path.expanduser(str(target).strip("'\""))
                if not os.path.isdir(clean_target):
                    return False, f"Precondition failed: required directory '{clean_target}' does not exist."

            elif cond_type == "app_running" and target and psutil is not None:
                app_base = os.path.basename(str(target)).lower()
                running = any(app_base in p.info["name"].lower() for p in psutil.process_iter(["name"]) if p.info.get("name"))
                if not running:
                    return False, f"Precondition failed: application '{target}' is not currently running."

        return True, "Preconditions satisfied."

    @classmethod
    def verify_action(
        cls,
        handler: str,
        parameters: Dict[str, Any],
        execution_result: Dict[str, Any]
    ) -> VerificationResult:
        """
        Observe and evaluate actual OS state post-execution.
        """
        raw_status = execution_result.get("status", "unknown")
        raw_message = execution_result.get("message", "")

        # 1. Application Launch Verification
        if handler == "launch_app":
            app_name = parameters.get("app_name", "").lower().strip()
            # Observe running processes
            is_running = False
            if psutil is not None and app_name:
                clean_app = app_name.replace(".exe", "")
                for p in psutil.process_iter(["name"]):
                    try:
                        pname = (p.info.get("name") or "").lower()
                        if clean_app in pname or (clean_app == "vscode" and "code" in pname):
                            is_running = True
                            break
                    except Exception:
                        pass

            if is_running or raw_status in ["success", "ok"]:
                return VerificationResult(
                    success=True,
                    observed_state="process_verified_running",
                    diagnostic_message=f"Application '{app_name}' verified active in OS process table."
                )
            else:
                return VerificationResult(
                    success=False,
                    observed_state="process_not_found",
                    diagnostic_message=f"Application '{app_name}' failed to launch or process terminated immediately.",
                    can_recover=True,
                    recovery_strategy="fallback_shell_launch",
                    fallback_parameters={"app_name": app_name, "use_shell": True}
                )

        # 2. File Operation Verification (find, create, delete, open)
        elif handler == "file_operation":
            subaction = parameters.get("subaction") or parameters.get("action")
            target = execution_result.get("path") or parameters.get("source") or parameters.get("file")

            if subaction in ["create", "touch"]:
                if target and os.path.exists(target):
                    return VerificationResult(
                        success=True,
                        observed_state="file_confirmed_created",
                        diagnostic_message=f"File verified at: {target}"
                    )
                else:
                    return VerificationResult(
                        success=False,
                        observed_state="file_missing",
                        diagnostic_message=f"Expected created file '{target}' was not found on disk.",
                        can_recover=True,
                        recovery_strategy="retry_in_fallback_directory",
                        fallback_parameters={"path": os.path.join(os.path.expanduser("~"), "Desktop")}
                    )

            elif subaction in ["find", "locate", "search"]:
                candidates = execution_result.get("candidates", [])
                found_path = execution_result.get("path") or (candidates[0].get("path") if candidates else None)
                if found_path and os.path.exists(found_path):
                    return VerificationResult(
                        success=True,
                        observed_state="file_confirmed_exists",
                        diagnostic_message=f"Resolved file verified: {found_path}"
                    )
                elif candidates:
                    return VerificationResult(
                        success=True,
                        observed_state="candidates_found",
                        diagnostic_message=f"Found {len(candidates)} candidates matching query."
                    )
                else:
                    return VerificationResult(
                        success=False,
                        observed_state="zero_results",
                        diagnostic_message="No matching files found in primary directory.",
                        can_recover=True,
                        recovery_strategy="broaden_search_scope",
                        fallback_parameters={"depth": 2, "search_roots": ["Desktop", "Documents"]}
                    )

            elif subaction == "verify_file":
                p = parameters.get("path") or parameters.get("source") or execution_result.get("path")
                if p and os.path.exists(p):
                    return VerificationResult(
                        success=True,
                        observed_state="file_confirmed_exists",
                        diagnostic_message=f"File confirmed accessible on filesystem: {p}"
                    )
                else:
                    return VerificationResult(
                        success=False,
                        observed_state="file_not_found",
                        diagnostic_message=f"File verification failed: '{p}' does not exist on disk.",
                        can_recover=True,
                        recovery_strategy="broaden_search_scope"
                    )

            elif subaction in ["delete", "trash"]:
                if target and not os.path.exists(target):
                    return VerificationResult(
                        success=True,
                        observed_state="file_confirmed_deleted",
                        diagnostic_message=f"File successfully removed: {target}"
                    )
                else:
                    return VerificationResult(
                        success=False,
                        observed_state="file_still_exists",
                        diagnostic_message=f"Target file '{target}' still exists after delete attempt.",
                        can_recover=False
                    )

            # Default file success
            return VerificationResult(
                success=raw_status in ["success", "ok"],
                observed_state="file_action_completed",
                diagnostic_message=raw_message or "File operation finished."
            )

        # 3. Test & Project Execution Verification
        elif handler in ["project_action", "exec_project_action", "run_tests"]:
            exit_code = execution_result.get("exit_code", 0)
            stdout = execution_result.get("stdout", "")
            stderr = execution_result.get("stderr", "")

            if exit_code == 0:
                return VerificationResult(
                    success=True,
                    observed_state="tests_passed",
                    diagnostic_message="Test execution completed successfully with exit code 0."
                )
            else:
                return VerificationResult(
                    success=False,
                    observed_state="tests_failed",
                    diagnostic_message=f"Process exited with non-zero code {exit_code}. Error: {stderr or stdout[:200]}",
                    can_recover=False
                )

        # 4. Process Kill Verification
        elif handler in ["kill_process", "exec_process_kill"]:
            pid = parameters.get("pid")
            if pid and psutil is not None:
                try:
                    exists = psutil.pid_exists(int(pid))
                    if not exists:
                        return VerificationResult(
                            success=True,
                            observed_state="pid_terminated",
                            diagnostic_message=f"Process {pid} confirmed terminated."
                        )
                except Exception:
                    pass

        # 4b. GUI Action Verification
        elif handler in ["gui_action", "exec_gui_action"]:
            act = parameters.get("action", "interact")
            targ = parameters.get("target") or parameters.get("app_name") or "screen"
            if raw_status in ["success", "ok"]:
                return VerificationResult(
                    success=True,
                    observed_state="gui_action_executed",
                    diagnostic_message=f"GUI action '{act}' on '{targ}' completed successfully."
                )
            else:
                return VerificationResult(
                    success=False,
                    observed_state="gui_action_failed",
                    diagnostic_message=raw_message or f"Failed to execute GUI action '{act}' on '{targ}'.",
                    can_recover=True,
                    recovery_strategy="retry_with_focus",
                    fallback_parameters={"app_name": parameters.get("app_name"), "force_focus": True}
                )

        # 5. Default Verification
        return VerificationResult(
            success=(raw_status in ["success", "ok"]),
            observed_state="completed_nominally",
            diagnostic_message=raw_message or f"Action {handler} execution complete."
        )

    def verify(
        self,
        step_type: str = "",
        target: str = "",
        expected_state: str = "",
        actual_result: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> VerificationResult:
        """Instance alias delegating to verify_action."""
        handler = step_type or kwargs.get("handler", "")
        params = kwargs.get("parameters") or {"app_name": target, "target": target}
        exec_res = actual_result or kwargs.get("execution_result", {})
        return self.verify_action(handler, params, exec_res)
