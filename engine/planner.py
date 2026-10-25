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
        if ("report" in goal_lower and any(w in goal_lower for w in ["find", "look", "submit"]) and any(w in goal_lower for w in ["summary", "summarize", "prepare"]) and "notepad" not in goal_lower and "whatsapp" not in goal_lower):
            return cls._create_report_summary_plan(goal, context)

        # Check for compound file discovery and WhatsApp sharing goal:
        if any(w in goal_lower for w in ["find", "locate", "get", "share"]) and any(w in goal_lower for w in ["report", "file", "document", "notes"]) and "whatsapp" in goal_lower:
            return cls._create_file_and_whatsapp_plan(goal, context)

        # Check for compound file discovery, resume attach and email draft goal:
        if any(w in goal_lower for w in ["find", "locate", "attach"]) and any(w in goal_lower for w in ["resume", "cv", "report", "file"]) and any(w in goal_lower for w in ["email", "mail"]):
            return cls._create_file_and_email_plan(goal, context)

        # Check for compound open notes, summarize, and put in Notepad goal:
        if any(w in goal_lower for w in ["open", "find", "read", "take"]) and any(w in goal_lower for w in ["notes", "report", "file", "document", "text"]) and any(w in goal_lower for w in ["summary", "summarize", "short summary"]) and "notepad" in goal_lower:
            return cls._create_file_summary_notepad_plan(goal, context)

        # Check for compound screenshot and WhatsApp sharing goal:
        if ("screenshot" in goal_lower or "screen capture" in goal_lower) and "whatsapp" in goal_lower:
            return cls._create_screenshot_and_whatsapp_plan(goal, context)

        # Check for compound calculate and Notepad goal:
        if any(w in goal_lower for w in ["calculate", "compute", "what is"]) and "notepad" in goal_lower:
            return cls._create_calc_and_notepad_plan(goal, context)

        # Check for compound metrics / battery and Notepad goal:
        if any(w in goal_lower for w in ["metrics", "stats", "battery", "cpu usage", "system status"]) and "notepad" in goal_lower and not any(w in goal_lower for w in ["calculate", "report", "notes"]):
            return cls._create_metrics_and_notepad_plan(goal, context)

        # Check for compound web search, summarize, and Notepad goal:
        if any(w in goal_lower for w in ["search", "google", "look up"]) and any(w in goal_lower for w in ["summary", "summarize"]) and "notepad" in goal_lower:
            return cls._create_web_summary_notepad_plan(goal, context)

        # Check for compound create file, write content, and open goal:
        if any(w in goal_lower for w in ["create a file", "create file", "make a file", "make file"]) and any(w in goal_lower for w in ["write", "with content", "saying"]) and "open" in goal_lower:
            return cls._create_file_create_write_open_plan(goal, context)

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
                # If current step references anaphora ('it', 'that', 'the file', 'the screenshot', 'the result', etc.)
                if re.search(r"\b(it|that|the file|the screenshot|the result|the answer|that number|the stats|the metrics|the summary|there|into it)\b", raw_query, re.IGNORECASE):
                    dependencies.append(idx - 1)
                elif handler in ["gui_action", "type_notepad", "notepad_write"] and not params.get("text") and not params.get("content"):
                    dependencies.append(idx - 1)
                elif handler in ["file_operation", "draft_content", "file_create", "file_append", "file_read"] and not params.get("source") and not params.get("body") and not params.get("file") and not params.get("path"):
                    dependencies.append(idx - 1)
                elif handler in ["whatsapp_send", "whatsapp_draft", "email_draft", "email_send"] and (not params.get("attachment_path") and not params.get("content")):
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
    def _create_file_and_whatsapp_plan(cls, goal: str, context: Optional[Any] = None) -> TaskPlan:
        """
        Synthesize compound goal plan:
        "Find the latest project report and send it to Rahul on WhatsApp"
        """
        # Extract recipient if present
        m_recip = re.search(r"(?:to|message|send)\s+([a-zA-Z0-9_\-\s]+?)(?:\s+on|\s+via)?\s+whatsapp", goal, re.IGNORECASE)
        recip_name = m_recip.group(1).strip() if m_recip else (getattr(context, "last_person", None) or "Rahul")
        recip_name = re.sub(r"^(?:it\s+to|it\s+on)\s+", "", recip_name, flags=re.IGNORECASE).strip()

        # Determine target file keyword
        m_file = re.search(r"(?:find|get|locate)\s+(?:the\s+)?([a-zA-Z0-9_\-\s]+?)(?:\s+and|\s+to|\s+report)", goal, re.IGNORECASE)
        file_term = "project report" if "report" in goal.lower() else (m_file.group(1).strip() if m_file else "document")

        steps = [
            PlanStep(
                step_id=1,
                query=f"Find the latest {file_term}",
                handler="file_operation",
                parameters={"subaction": "find", "query": file_term, "recency": "latest"},
                expected_output="path",
                dependencies=[],
            ),
            PlanStep(
                step_id=2,
                query=f"Verify {file_term} exists on filesystem",
                handler="file_operation",
                parameters={"subaction": "verify_file"},
                expected_output="verified_file",
                dependencies=[1],
            ),
            PlanStep(
                step_id=3,
                query=f"Prepare and share {file_term} with {recip_name} on WhatsApp",
                handler="whatsapp_send",
                parameters={
                    "recipient": recip_name,
                    "content": f"Hi {recip_name}, here is the latest {file_term}.",
                    "attachment_path": "",
                    "authorized": False  # Confirmation required before real transmission
                },
                expected_output="message_submitted",
                dependencies=[1, 2],
                is_consequential=True,
                requires_confirmation=True,
            ),
        ]
        return TaskPlan(
            goal=goal,
            steps=steps,
            intended_outcome=f"Located latest {file_term} and prepared WhatsApp message with attachment for {recip_name}."
        )

    @classmethod
    def _create_file_and_email_plan(cls, goal: str, context: Optional[Any] = None) -> TaskPlan:
        """
        Synthesize compound goal plan:
        "Find my resume, attach it to an email to the recruiter, and save the email as a draft."
        """
        m_recip = re.search(r"(?:to|email)\s+(?:the\s+)?([a-zA-Z0-9_\-]+)(?:\s+and|\s+as|\s+with|$)", goal, re.IGNORECASE)
        recip_name = m_recip.group(1).strip() if m_recip else "recruiter"
        if recip_name.lower() in ["an", "email", "draft", "a"]:
            m_recip2 = re.search(r"\bto\s+(?:the\s+)?([a-zA-Z0-9_\-]+)\b", goal, re.IGNORECASE)
            if m_recip2 and m_recip2.group(1).lower() not in ["an", "email"]:
                recip_name = m_recip2.group(1).strip()
            else:
                recip_name = "recruiter"

        steps = [
            PlanStep(
                step_id=1,
                query="Find my latest resume",
                handler="file_operation",
                parameters={"subaction": "find", "query": "resume", "recency": "latest"},
                expected_output="path",
                dependencies=[],
            ),
            PlanStep(
                step_id=2,
                query="Verify resume file exists and is accessible",
                handler="file_operation",
                parameters={"subaction": "verify_file"},
                expected_output="verified_file",
                dependencies=[1],
            ),
            PlanStep(
                step_id=3,
                query=f"Draft email to {recip_name} with resume attached",
                handler="email_draft",
                parameters={
                    "recipient": recip_name,
                    "topic": "Job Application Resume Submission",
                    "attachment_path": ""
                },
                expected_output="email_draft_stored",
                dependencies=[1, 2],
            ),
        ]
        return TaskPlan(
            goal=goal,
            steps=steps,
            intended_outcome=f"Discovered latest resume and saved email draft with attachment for {recip_name}."
        )

    @classmethod
    def _create_file_summary_notepad_plan(cls, goal: str, context: Optional[Any] = None) -> TaskPlan:
        """
        Synthesize compound goal plan:
        "Open the project notes, create a short summary, and put it in Notepad."
        """
        m_doc = re.search(r"(?:open|find|read)\s+(?:the\s+)?([a-zA-Z0-9_\-\s]+?)(?:,|\s+create|\s+and)", goal, re.IGNORECASE)
        doc_term = m_doc.group(1).strip() if m_doc else "project notes"

        steps = [
            PlanStep(
                step_id=1,
                query=f"Find {doc_term}",
                handler="file_operation",
                parameters={"subaction": "find", "query": doc_term, "recency": "latest"},
                expected_output="path",
                dependencies=[],
            ),
            PlanStep(
                step_id=2,
                query=f"Read and create extractive summary of {doc_term}",
                handler="file_summarize",
                parameters={"max_points": 4},
                expected_output="summary",
                dependencies=[1],
            ),
            PlanStep(
                step_id=3,
                query="Insert verified summary into Windows Notepad",
                handler="gui_action",
                parameters={
                    "action": "type",
                    "app_name": "notepad",
                    "text": "",
                    "content": "",
                    "force_focus": True
                },
                expected_output="notepad_written",
                dependencies=[2],
            ),
        ]
        return TaskPlan(
            goal=goal,
            steps=steps,
            intended_outcome=f"Discovered {doc_term}, generated factual summary, and faithfully inserted into Notepad."
        )

    @classmethod
    def _create_screenshot_and_whatsapp_plan(cls, goal: str, context: Optional[Any] = None) -> TaskPlan:
        """
        Synthesize compound goal plan:
        "Take a screenshot and send it to Rahul on WhatsApp"
        """
        m_recip = re.search(r"(?:to|message|send)\s+([a-zA-Z0-9_\-\s]+?)(?:\s+on|\s+via)?\s+whatsapp", goal, re.IGNORECASE)
        recip_name = m_recip.group(1).strip() if m_recip else (getattr(context, "last_person", None) or "8591211342")
        recip_name = re.sub(r"^(?:it\s+to|it\s+on)\s+", "", recip_name, flags=re.IGNORECASE).strip()

        steps = [
            PlanStep(
                step_id=1,
                query="Capture full screen snapshot",
                handler="system_utility",
                parameters={"utility": "screenshot", "command": "screenshot"},
                expected_output="screenshot_path",
                dependencies=[],
            ),
            PlanStep(
                step_id=2,
                query=f"Send captured screenshot to {recip_name} on WhatsApp",
                handler="whatsapp_send",
                parameters={
                    "recipient": recip_name,
                    "content": f"Here is the screenshot requested.",
                    "attachment_path": "",
                    "authorized": False
                },
                expected_output="message_submitted",
                dependencies=[1],
                is_consequential=True,
                requires_confirmation=True,
            ),
        ]
        return TaskPlan(
            goal=goal,
            steps=steps,
            intended_outcome=f"Captured screenshot and prepared WhatsApp transmission for {recip_name}."
        )

    @classmethod
    def _create_calc_and_notepad_plan(cls, goal: str, context: Optional[Any] = None) -> TaskPlan:
        """
        Synthesize compound goal plan:
        "Calculate 15 * 40, write the result in notepad, and save the file"
        """
        m_calc = re.search(r"(?:calculate|compute|what\s+is)\s+([0-9\+\-\*\/\^\.\s\%]+)", goal, re.IGNORECASE)
        expr = m_calc.group(1).strip() if m_calc else "15 * 40"
        expr = re.sub(r"\s+(?:and|then|write|put).*$", "", expr, flags=re.IGNORECASE).strip()

        steps = [
            PlanStep(
                step_id=1,
                query=f"Calculate {expr}",
                handler="compute_math",
                parameters={"expr": expr, "command": expr},
                expected_output="calc_result",
                dependencies=[],
            ),
            PlanStep(
                step_id=2,
                query="Insert calculation result into Windows Notepad",
                handler="gui_action",
                parameters={
                    "action": "type",
                    "app_name": "notepad",
                    "text": "",
                    "content": "",
                    "force_focus": True
                },
                expected_output="notepad_written",
                dependencies=[1],
            ),
        ]
        if "save" in goal.lower():
            steps.append(PlanStep(
                step_id=3,
                query="Save document in Notepad",
                handler="file_operation",
                parameters={"subaction": "save"},
                expected_output="file_saved",
                dependencies=[2],
            ))
        return TaskPlan(
            goal=goal,
            steps=steps,
            intended_outcome=f"Computed '{expr}', entered result into Notepad{', and saved file' if 'save' in goal.lower() else ''}."
        )

    @classmethod
    def _create_metrics_and_notepad_plan(cls, goal: str, context: Optional[Any] = None) -> TaskPlan:
        """
        Synthesize compound goal plan:
        "Check battery and write the stats in notepad"
        """
        steps = [
            PlanStep(
                step_id=1,
                query="Query system performance metrics",
                handler="system_metrics",
                parameters={},
                expected_output="system_metrics",
                dependencies=[],
            ),
            PlanStep(
                step_id=2,
                query="Insert verified system metrics into Windows Notepad",
                handler="gui_action",
                parameters={
                    "action": "type",
                    "app_name": "notepad",
                    "text": "",
                    "content": "",
                    "force_focus": True
                },
                expected_output="notepad_written",
                dependencies=[1],
            ),
        ]
        return TaskPlan(
            goal=goal,
            steps=steps,
            intended_outcome="Queried system metrics and inserted telemetry into Notepad."
        )

    @classmethod
    def _create_web_summary_notepad_plan(cls, goal: str, context: Optional[Any] = None) -> TaskPlan:
        """
        Synthesize compound goal plan:
        "Search for Python tutorials, summarize it, and put it in Notepad"
        """
        m_query = re.search(r"(?:search\s+for|search|google|look\s+up)\s+(.+?)(?:,|\s+summarize|\s+and)", goal, re.IGNORECASE)
        q_term = m_query.group(1).strip() if m_query else "information"

        steps = [
            PlanStep(
                step_id=1,
                query=f"Search online for '{q_term}'",
                handler="web_search",
                parameters={"query": q_term},
                expected_output="search_results",
                dependencies=[],
            ),
            PlanStep(
                step_id=2,
                query=f"Synthesize extractive summary of '{q_term}' findings",
                handler="browser_search_and_summarize",
                parameters={"query": q_term},
                expected_output="summary",
                dependencies=[1],
            ),
            PlanStep(
                step_id=3,
                query="Insert factual summary into Windows Notepad",
                handler="gui_action",
                parameters={
                    "action": "type",
                    "app_name": "notepad",
                    "text": "",
                    "content": "",
                    "force_focus": True
                },
                expected_output="notepad_written",
                dependencies=[2],
            ),
        ]
        return TaskPlan(
            goal=goal,
            steps=steps,
            intended_outcome=f"Discovered online data for '{q_term}', synthesized summary, and typed into Notepad."
        )

    @classmethod
    def _create_file_create_write_open_plan(cls, goal: str, context: Optional[Any] = None) -> TaskPlan:
        """
        Synthesize compound goal plan:
        "Create a file named notes.txt with content hello and open it"
        """
        m_file = re.search(r"(?:file\s+named|file\s+called|file)\s+['\"]?([a-zA-Z0-9_\-\.]+)['\"]?", goal, re.IGNORECASE)
        filename = m_file.group(1).strip() if m_file else "output.txt"

        m_content = re.search(r"(?:with\s+content|with\s+text|saying|containing|write)\s+['\"]?(.+?)['\"]?(?:\s+and\s+open|\s*,\s*open|$)", goal, re.IGNORECASE)
        content_text = m_content.group(1).strip() if m_content else "Hello from EDITH."

        steps = [
            PlanStep(
                step_id=1,
                query=f"Create file '{filename}' with content",
                handler="file_create",
                parameters={"path": filename, "content": content_text},
                expected_output="file_created",
                dependencies=[],
                is_consequential=True,
            ),
            PlanStep(
                step_id=2,
                query=f"Open file '{filename}'",
                handler="file_operation",
                parameters={"subaction": "open", "source": filename},
                expected_output="file_open",
                dependencies=[1],
            ),
        ]
        return TaskPlan(
            goal=goal,
            steps=steps,
            intended_outcome=f"Created '{filename}' with verified content and opened it."
        )

    @classmethod
    def pipe_step_dependencies(cls, step: PlanStep, step_outputs: Dict[int, Any]):
        """Inject resolved outputs from prerequisite steps into downstream step parameters."""
        for dep_id in step.dependencies:
            dep_output = step_outputs.get(dep_id)
            if not dep_output:
                continue

            if isinstance(dep_output, dict):
                actual_path = dep_output.get("path") or dep_output.get("file") or dep_output.get("created_file")
                if actual_path:
                    if not step.parameters.get("source") or step.parameters.get("source") in ["it", "that", "the file", "the screenshot"]:
                        step.parameters["source"] = actual_path
                    if not step.parameters.get("path"):
                        step.parameters["path"] = actual_path
                    if not step.parameters.get("file"):
                        step.parameters["file"] = os.path.basename(actual_path)
                    # Pipe attachment path for communication actions
                    if step.handler in ["whatsapp_send", "whatsapp_draft", "email_draft", "email_send"]:
                        step.parameters["attachment_path"] = actual_path

                # Pipe summary content into Notepad typing or message drafting
                if dep_output.get("summary"):
                    sum_text = dep_output["summary"]
                    if step.handler in ["gui_action", "type_notepad", "notepad_write", "file_create", "file_append"]:
                        step.parameters["text"] = sum_text
                        step.parameters["content"] = sum_text
                    elif step.handler in ["whatsapp_send", "whatsapp_draft", "email_draft", "email_send"]:
                        step.parameters["content"] = sum_text

                # Pipe calculation results
                calc_val = dep_output.get("result") or dep_output.get("calc_result") or dep_output.get("answer")
                if calc_val is not None:
                    calc_str = str(calc_val)
                    if step.handler in ["gui_action", "type_notepad", "notepad_write", "file_create", "file_append"]:
                        if not step.parameters.get("text") and not step.parameters.get("content"):
                            step.parameters["text"] = calc_str
                            step.parameters["content"] = calc_str
                    elif step.handler in ["whatsapp_send", "whatsapp_draft", "email_draft", "email_send"]:
                        if not step.parameters.get("content") or step.parameters.get("content") in ["the result", "the answer", "that number", "it", "that"]:
                            step.parameters["content"] = f"The result is {calc_str}"

                # Pipe clipboard content
                clip_text = dep_output.get("content") if dep_output.get("action") == "clipboard_read" else (dep_output.get("clipboard") or dep_output.get("clipboard_text"))
                if clip_text:
                    if step.handler in ["gui_action", "type_notepad", "notepad_write", "file_create", "file_append"]:
                        if not step.parameters.get("text") and not step.parameters.get("content"):
                            step.parameters["text"] = clip_text
                            step.parameters["content"] = clip_text
                    elif step.handler in ["whatsapp_send", "whatsapp_draft", "email_draft", "email_send"]:
                        if not step.parameters.get("content") or step.parameters.get("content") in ["the clipboard", "it", "that"]:
                            step.parameters["content"] = clip_text

                # Pipe system metrics
                metrics_data = dep_output.get("metrics") or (dep_output if dep_output.get("action") == "system_metrics" else None)
                if metrics_data:
                    m_formatted = dep_output.get("formatted") or dep_output.get("message") or str(metrics_data)
                    if step.handler in ["gui_action", "type_notepad", "notepad_write", "file_create", "file_append"]:
                        if not step.parameters.get("text") and not step.parameters.get("content"):
                            step.parameters["text"] = m_formatted
                            step.parameters["content"] = m_formatted
                    elif step.handler in ["whatsapp_send", "whatsapp_draft", "email_draft", "email_send"]:
                        if not step.parameters.get("content") or step.parameters.get("content") in ["the stats", "the metrics", "system stats"]:
                            step.parameters["content"] = f"System Stats: {m_formatted}"

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

                    if context:
                        if res.get("path"):
                            context.last_path = res["path"]
                            context.last_file = os.path.basename(res["path"])
                        if res.get("app_name"):
                            context.last_app = res.get("app_name")
                        if res.get("result") or res.get("calc_result"):
                            if hasattr(context, "set_last_calculation"):
                                context.set_last_calculation(step.query, res.get("result") or res.get("calc_result"))
                        if res.get("summary"):
                            if hasattr(context, "set_last_summary"):
                                context.set_last_summary(res.get("summary"), res.get("source", "step"))
                        if res.get("screenshot_path") or res.get("action") == "screenshot" or (res.get("utility") == "screenshot" and res.get("path")):
                            if hasattr(context, "set_last_screenshot"):
                                context.set_last_screenshot(res.get("screenshot_path") or res.get("path"))
                        if res.get("metrics") or res.get("action") == "system_metrics":
                            if hasattr(context, "set_last_metrics"):
                                context.set_last_metrics(res.get("metrics") or res)
                        if res.get("clipboard") or res.get("action") in ["clipboard_read", "clipboard_copy"]:
                            if hasattr(context, "set_last_clipboard"):
                                context.set_last_clipboard(res.get("content") or res.get("clipboard") or "")
                        if res.get("created_file") or (step.handler == "file_create" and res.get("path")):
                            if hasattr(context, "set_last_created_file"):
                                context.set_last_created_file(res.get("created_file") or res.get("path"))
                        if res.get("url"):
                            if hasattr(context, "set_last_url"):
                                context.set_last_url(res.get("url"))

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
