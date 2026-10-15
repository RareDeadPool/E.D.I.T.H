"""Structured Multi-Step Task Planner, Dependency Engine, and Safety Gate for E.D.I.T.H. Intelligence Core V2."""

import os
import re
import time
from typing import Dict, Any, List, Optional, Tuple

from runtime.closed_loop import ClosedLoopVerifier, VerificationResult


class PlanStep:
    """Discrete step within a structured task execution plan with full closed-loop metadata."""

    def __init__(
        self,
        step_id: int,
        query: str,
        handler: str,
        parameters: Dict[str, Any],
        dependencies: Optional[List[int]] = None,
        expected_output: Optional[str] = None,
        preconditions: Optional[List[Dict[str, Any]]] = None,
        required_entities: Optional[List[str]] = None,
        is_consequential: bool = False,
        requires_confirmation: bool = False,
        retry_policy: Optional[Dict[str, Any]] = None,
        failure_recovery: Optional[str] = None,
    ):
        self.step_id = step_id
        self.query = query
        self.handler = handler
        self.parameters = parameters
        self.dependencies = dependencies or []
        self.expected_output = expected_output  # e.g., 'path', 'app', 'test_results', 'text'
        self.preconditions = preconditions or []
        self.required_entities = required_entities or []
        self.is_consequential = is_consequential
        self.requires_confirmation = requires_confirmation
        self.status = "pending"  # pending, running, completed, failed, skipped
        self.execution_result: Optional[Dict[str, Any]] = None
        self.retry_policy = retry_policy or {"max_retries": 2, "retries_taken": 0}
        self.failure_recovery = failure_recovery
        self.verified = False
        self.verification_details: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "query": self.query,
            "handler": self.handler,
            "parameters": self.parameters,
            "dependencies": self.dependencies,
            "expected_output": self.expected_output,
            "preconditions": self.preconditions,
            "required_entities": self.required_entities,
            "is_consequential": self.is_consequential,
            "requires_confirmation": self.requires_confirmation,
            "status": self.status,
            "verified": self.verified,
            "verification_details": self.verification_details,
            "retry_policy": self.retry_policy,
            "failure_recovery": self.failure_recovery,
            "result": self.execution_result,
        }


class TaskPlan:
    """Validated structured action plan with dependency tracking, goal outcome tracking, and dynamic plan revision."""

    def __init__(
        self,
        goal: str,
        steps: List[PlanStep],
        intended_outcome: Optional[str] = None,
        required_entities: Optional[Dict[str, Any]] = None
    ):
        self.goal = goal
        self.intended_outcome = intended_outcome or goal
        self.required_entities = required_entities or {}
        self.steps = steps
        self.status = "planned"  # planned, running, completed, partial, failed, needs_clarification, needs_confirmation
        self.clarification_needed: Optional[str] = None
        self.confirmation_needed: Optional[str] = None
        self.revisions: List[Dict[str, Any]] = []
        self.created_at = time.time()

    @property
    def remaining_steps(self) -> List[PlanStep]:
        return [s for s in self.steps if s.status in ["pending", "planned"]]

    @property
    def completed_steps(self) -> List[PlanStep]:
        return [s for s in self.steps if s.status == "completed"]

    def revise_on_failure(self, failed_step: PlanStep, failure_details: str, context: Optional[Any] = None) -> bool:
        """
        Dynamically revise remaining steps of the plan when a step fails verification or execution.
        Returns True if plan was revised and can continue; False if halted.
        """
        revision_entry = {
            "failed_step_id": failed_step.step_id,
            "failed_handler": failed_step.handler,
            "failure_details": failure_details,
            "timestamp": time.time(),
        }

        # Case 1: File discovery returned empty or not found in primary directory
        if failed_step.handler == "file_operation" and failed_step.parameters.get("subaction") in ["find", "locate", "search"]:
            query = failed_step.parameters.get("query") or failed_step.parameters.get("source") or failed_step.query
            new_step = PlanStep(
                step_id=len(self.steps) + 1,
                query=f"Search broader system paths for '{query}'",
                handler="file_operation",
                parameters={"subaction": "find", "query": query, "broaden": True},
                expected_output="path",
                dependencies=[]
            )
            self.steps.append(new_step)
            for s in self.remaining_steps:
                if failed_step.step_id in s.dependencies:
                    s.dependencies.remove(failed_step.step_id)
                    s.dependencies.append(new_step.step_id)
            revision_entry["action"] = "appended_broad_search"
            self.revisions.append(revision_entry)
            return True

        # Case 2: Opening in preferred app failed -> Fallback to default system open
        if failed_step.handler == "launch_app" and failed_step.parameters.get("path"):
            path = failed_step.parameters.get("path")
            new_step = PlanStep(
                step_id=len(self.steps) + 1,
                query=f"Open '{os.path.basename(path)}' with default system application",
                handler="file_operation",
                parameters={"subaction": "open", "source": path},
                expected_output="file_open",
                dependencies=[]
            )
            self.steps.append(new_step)
            for s in self.remaining_steps:
                if failed_step.step_id in s.dependencies:
                    s.dependencies.remove(failed_step.step_id)
                    s.dependencies.append(new_step.step_id)
            revision_entry["action"] = "fallback_default_open"
            self.revisions.append(revision_entry)
            return True

        # Case 3: Test command failed to resolve -> fallback to python unittest discovery
        if failed_step.handler == "project_action" and failed_step.parameters.get("subaction") == "resolve_test_cmd":
            for s in self.remaining_steps:
                if s.handler == "project_action" and s.parameters.get("subaction") == "run_tests":
                    s.parameters["command"] = "py -3.12 -m unittest discover tests"
            revision_entry["action"] = "fallback_unittest_command"
            self.revisions.append(revision_entry)
            return True

        self.revisions.append(revision_entry)
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "intended_outcome": self.intended_outcome,
            "required_entities": self.required_entities,
            "status": self.status,
            "clarification_needed": self.clarification_needed,
            "confirmation_needed": self.confirmation_needed,
            "completed_steps_count": len(self.completed_steps),
            "remaining_steps_count": len(self.remaining_steps),
            "revisions": self.revisions,
            "steps": [s.to_dict() for s in self.steps],
        }


class TaskPlanner:
    """Synthesizes, validates, and coordinates multi-step agent plans with closed-loop verification."""

    CONSEQUENTIAL_HANDLERS = {"delete_file", "empty_recycle_bin", "kill_process", "send_email"}

    @classmethod
    def create_plan(
        cls,
        goal: str,
        raw_steps: List[str],
        step_directives: List[Dict[str, Any]],
        context: Optional[Any] = None,
        structured_understanding: Optional[Any] = None,
    ) -> TaskPlan:
        """Create and validate a structured TaskPlan from atomic step directives or complex goals."""
        goal_lower = goal.lower()

        # Check for compound project report finding, opening, and summarizing goal:
        if ("report" in goal_lower and any(w in goal_lower for w in ["find", "look", "submit"]) and any(w in goal_lower for w in ["summary", "summarize", "prepare"])):
            return cls._create_report_summary_plan(goal, context)

        # Check for compound project test & run goal:
        if ("project" in goal_lower and re.search(r"\btests?\b", goal_lower) and any(w in goal_lower for w in ["pass", "fail", "check", "run"])):
            return cls._create_project_test_plan(goal, context)

        plan_steps: List[PlanStep] = []

        for idx, (raw_query, directive) in enumerate(zip(raw_steps, step_directives), 1):
            handler = directive.get("handler", "noop")
            params = dict(directive.get("parameters", {}))
            subaction = params.get("subaction") or params.get("action")

            # Check if consequential
            is_consequential = False
            requires_confirmation = False

            if handler == "file_operation" and subaction in ["delete", "trash"]:
                is_consequential = True
                target = params.get("source") or params.get("target") or params.get("path") or params.get("file")
                # Consequential operations without explicit named target need confirmation
                if not target or str(target).lower() in ["it", "that", "old"]:
                    requires_confirmation = True
            elif handler in ["kill_process", "exec_process_kill"]:
                is_consequential = True
                requires_confirmation = True
            elif handler in ["system_utility", "exec_system_utility"] and "empty_recycle_bin" in str(params.get("utility", "")):
                is_consequential = True
                requires_confirmation = True

            # Determine dependencies on previous steps
            dependencies = []
            if idx > 1:
                # If current step references anaphora ('it', 'that', 'the file', 'into it')
                if re.search(r"\b(it|that|the file|the screenshot|the result|there)\b", raw_query, re.IGNORECASE):
                    dependencies.append(idx - 1)
                elif handler in ["file_operation", "draft_content"] and not params.get("source") and not params.get("body") and not params.get("file"):
                    dependencies.append(idx - 1)
                elif handler in ["launch_app"] and not params.get("app_name") and context and context.last_app:
                    params["app_name"] = context.last_app

            expected_output = None
            preconditions = []
            required_entities = []

            if handler == "file_operation":
                if subaction in ["find", "locate", "search"]:
                    expected_output = "path"
                elif subaction in ["create", "touch"]:
                    expected_output = "created_file"
                    required_entities.append("file")
                elif subaction in ["delete", "remove"]:
                    expected_output = "deleted_status"
            elif handler in ["launch_app", "exec_launch_app"]:
                expected_output = "running_app"
                required_entities.append("app_name")
            elif handler in ["system_utility", "exec_system_utility"] and "screenshot" in str(params.get("utility", "")):
                expected_output = "screenshot_path"
            elif handler in ["project_action", "exec_project_action", "run_tests"]:
                expected_output = "test_results"

            step = PlanStep(
                step_id=idx,
                query=raw_query,
                handler=handler,
                parameters=params,
                dependencies=dependencies,
                expected_output=expected_output,
                preconditions=preconditions,
                required_entities=required_entities,
                is_consequential=is_consequential,
                requires_confirmation=requires_confirmation,
            )
            plan_steps.append(step)

        plan = TaskPlan(goal=goal, steps=plan_steps)

        # Safety gate: check if any step needs confirmation or clarification
        for s in plan.steps:
            if s.requires_confirmation:
                plan.status = "needs_confirmation"
                plan.confirmation_needed = f"Please confirm {s.handler} operation for '{s.query}'."
                break

        return plan

    @classmethod
    def _create_project_test_plan(cls, goal: str, context: Optional[Any] = None) -> TaskPlan:
        """
        Synthesize the verified 7-stage project test & verification plan:
        1. Resolve the EDITH project location.
        2. Verify the selected project.
        3. Open it in VS Code.
        4. Determine the correct existing test command from project configuration.
        5. Execute the test command.
        6. Inspect the actual process result and test output.
        7. Summarize verified results.
        """
        workspace_path = os.path.abspath(".")

        steps = [
            PlanStep(
                step_id=1,
                query="Resolve the EDITH project location",
                handler="file_operation",
                parameters={"subaction": "find", "query": "EDITH", "source": workspace_path},
                expected_output="project_path",
                dependencies=[],
            ),
            PlanStep(
                step_id=2,
                query="Verify the selected project exists on filesystem",
                handler="file_operation",
                parameters={"subaction": "verify_dir", "path": workspace_path},
                expected_output="verified_project",
                preconditions=[{"type": "directory_exists", "target": workspace_path}],
                dependencies=[1],
            ),
            PlanStep(
                step_id=3,
                query="Open project in VS Code",
                handler="launch_app",
                parameters={"app_name": "vscode", "path": workspace_path},
                expected_output="editor_open",
                dependencies=[2],
            ),
            PlanStep(
                step_id=4,
                query="Determine the correct test command from project configuration",
                handler="project_action",
                parameters={"subaction": "resolve_test_cmd", "workspace": workspace_path},
                expected_output="test_command",
                dependencies=[2],
            ),
            PlanStep(
                step_id=5,
                query="Execute the test suite",
                handler="project_action",
                parameters={"subaction": "run_tests", "command": "py -3.12 -m unittest discover tests"},
                expected_output="test_results",
                dependencies=[4],
                retry_policy={"max_retries": 1, "retries_taken": 0},
            ),
            PlanStep(
                step_id=6,
                query="Inspect the process result and test output",
                handler="project_action",
                parameters={"subaction": "evaluate_output"},
                expected_output="test_verdict",
                dependencies=[5],
            ),
            PlanStep(
                step_id=7,
                query="Summarize verified test results",
                handler="dialogue",
                parameters={"message": "Project tests evaluated."},
                expected_output="summary",
                dependencies=[6],
            ),
        ]

        return TaskPlan(goal=goal, steps=steps)

    @classmethod
    def _create_report_summary_plan(cls, goal: str, context: Optional[Any] = None) -> TaskPlan:
        """
        Synthesize compound goal plan:
        "I need to submit my project tomorrow. Find the latest project report, check that it exists, open it, and help me prepare a professional summary."
        """
        steps = [
            PlanStep(
                step_id=1,
                query="Find the latest project report",
                handler="file_operation",
                parameters={"subaction": "find", "query": "project report", "recency": "latest"},
                expected_output="path",
                dependencies=[],
            ),
            PlanStep(
                step_id=2,
                query="Verify the project report exists and is accessible",
                handler="file_operation",
                parameters={"subaction": "verify_file"},
                expected_output="verified_file",
                dependencies=[1],
            ),
            PlanStep(
                step_id=3,
                query="Open the project report",
                handler="file_operation",
                parameters={"subaction": "open"},
                expected_output="file_open",
                dependencies=[2],
            ),
            PlanStep(
                step_id=4,
                query="Prepare a professional summary of the project report",
                handler="draft_content",
                parameters={"topic": "Project Report Executive Summary", "channel": "clipboard"},
                expected_output="draft_summary",
                dependencies=[2],
            ),
        ]
        return TaskPlan(
            goal=goal,
            steps=steps,
            intended_outcome="Project report located, verified, opened, and professional summary prepared.",
            required_entities={"FILE": "project report", "TOPIC": "summary"}
        )

    @classmethod
    def pipe_step_dependencies(cls, step: PlanStep, step_outputs: Dict[int, Any]):
        """Inject resolved outputs from prerequisite steps into downstream step parameters."""
        for dep_id in step.dependencies:
            dep_output = step_outputs.get(dep_id)
            if not dep_output:
                continue

            if isinstance(dep_output, dict):
                actual_path = dep_output.get("path") or dep_output.get("file")
                if actual_path:
                    if not step.parameters.get("source") or step.parameters.get("source") in ["it", "that", "the file"]:
                        step.parameters["source"] = actual_path
                    if not step.parameters.get("path"):
                        step.parameters["path"] = actual_path
                    if not step.parameters.get("file"):
                        step.parameters["file"] = os.path.basename(actual_path)

                if dep_output.get("test_command") and step.handler in ["project_action", "exec_project_action"]:
                    step.parameters["command"] = dep_output["test_command"]

            elif isinstance(dep_output, str):
                if step.handler in ["file_operation"]:
                    step.parameters["source"] = dep_output
                elif step.handler in ["launch_app"] and not step.parameters.get("app_name"):
                    step.parameters["app_name"] = dep_output

    @classmethod
    def execute_plan(
        cls,
        plan: TaskPlan,
        executor: Any,
        context: Any,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Execute structured plan with closed-loop verification passing real artifacts between steps."""
        if plan.status == "needs_confirmation" and not dry_run:
            return {
                "status": "needs_confirmation",
                "message": plan.confirmation_needed or "Operation requires explicit user confirmation.",
                "plan": plan.to_dict()
            }

        step_outputs: Dict[int, Any] = {}
        all_success = True
        executed_messages: List[str] = []

        step_idx = 0
        while step_idx < len(plan.steps):
            step = plan.steps[step_idx]
            step_idx += 1

            # 1. Pipe outputs from dependencies
            cls.pipe_step_dependencies(step, step_outputs)

            if dry_run or not executor:
                step.status = "planned"
                executed_messages.append(f"Step {step.step_id}: {step.query} [planned]")
                continue

            # 2. Check Preconditions
            precond_ok, precond_msg = ClosedLoopVerifier.check_preconditions(step.parameters, step.preconditions)
            if not precond_ok:
                step.status = "failed"
                step.verification_details = precond_msg
                all_success = False
                executed_messages.append(f"Step {step.step_id} aborted: {precond_msg}")
                break

            # 3. Live Execution with closed-loop observation
            step.status = "running"
            directive_payload = {
                "can_execute": True,
                "handler": step.handler,
                "parameters": step.parameters,
                "message": step.query
            }

            try:
                res = executor.execute(directive_payload)
                step.execution_result = res

                # 4. Closed-loop verification
                verification: VerificationResult = ClosedLoopVerifier.verify_action(
                    step.handler, step.parameters, res
                )
                step.verified = verification.success
                step.verification_details = verification.diagnostic_message

                # 5. Recovery attempt if verification failed
                if not verification.success and verification.can_recover:
                    max_retries = step.retry_policy.get("max_retries", 2)
                    retries = step.retry_policy.get("retries_taken", 0)
                    if retries < max_retries:
                        step.retry_policy["retries_taken"] = retries + 1
                        # Apply fallback parameters
                        if verification.fallback_parameters:
                            step.parameters.update(verification.fallback_parameters)
                            directive_payload["parameters"] = step.parameters
                        res = executor.execute(directive_payload)
                        step.execution_result = res
                        verification = ClosedLoopVerifier.verify_action(step.handler, step.parameters, res)
                        step.verified = verification.success
                        step.verification_details = f"After retry: {verification.diagnostic_message}"

                if step.verified:
                    step.status = "completed"
                    step_outputs[step.step_id] = res

                    if res.get("path"):
                        context.last_path = res["path"]
                        context.last_file = os.path.basename(res["path"])
                    if res.get("app_name"):
                        context.last_app = res.get("app_name")

                    executed_messages.append(f"Step {step.step_id}: {res.get('message', 'Completed')}")
                else:
                    step.status = "failed"
                    # Attempt dynamic plan revision based on failure details
                    revised = plan.revise_on_failure(step, step.verification_details or "Verification failed", context)
                    if revised:
                        executed_messages.append(f"Step {step.step_id} failed: {step.verification_details}. Plan revised dynamically to recover.")
                        continue
                    else:
                        all_success = False
                        executed_messages.append(f"Step {step.step_id} failed verification: {step.verification_details}")
                        break

            except Exception as e:
                step.status = "failed"
                step.execution_result = {"status": "error", "message": str(e)}
                revised = plan.revise_on_failure(step, str(e), context)
                if revised:
                    executed_messages.append(f"Step {step.step_id} error: {e}. Plan revised dynamically to recover.")
                    continue
                else:
                    all_success = False
                    executed_messages.append(f"Step {step.step_id} error: {e}")
                    break

        plan.status = "completed" if all_success else "partial"
        return {
            "status": "success" if all_success else "partial",
            "message": " | ".join(executed_messages) if executed_messages else "Plan formulated.",
            "plan": plan.to_dict()
        }
