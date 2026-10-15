"""Automated Test Suite for Cross-Application Compound Personal Task Workflows (E.D.I.T.H. V4).

Validates multi-step cross-application goals:
1. Compound File Discovery + WhatsApp Sharing:
   "Find the latest project report and send it to Rahul on WhatsApp"
2. Compound File Discovery + Resume Attachment + Email Drafting:
   "Find my resume, attach it to an email to the recruiter, and save the email as a draft"
3. Compound File Reading + Extractive Summarization + Notepad Insertion:
   "Open the project notes, create a short summary, and put it in Notepad"
4. Observable State Verification & Evidence Logging across disparate applications.
"""

import os
import sys
import time
import tempfile
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.edith_brain import EdithBrain
from engine.planner import TaskPlanner, TaskPlan
from runtime.file_discovery import FileDiscoveryEngine


class TestV4CrossAppCompoundWorkflows(unittest.TestCase):
    """Test suite for compound cross-application workflows."""

    def setUp(self):
        # Create an isolated temporary directory with sample files
        self.test_dir = tempfile.mkdtemp(prefix="edith_v4_compound_")
        now = time.time()

        # 1. Project reports
        self.report_old = os.path.join(self.test_dir, "project_report_v1.docx")
        with open(self.report_old, "w", encoding="utf-8") as f:
            f.write("Preliminary Project Report 2025 Architecture overview")
        os.utime(self.report_old, (now - 86400 * 20, now - 86400 * 20))

        self.report_latest = os.path.join(self.test_dir, "project_report_final.docx")
        with open(self.report_latest, "w", encoding="utf-8") as f:
            f.write("Final Project Report 2026. Complete systems deployment, latency benchmarks under 25ms, neural accuracy 98.4%.")
        os.utime(self.report_latest, (now - 1200, now - 1200))

        # 2. Resume files
        self.resume_latest = os.path.join(self.test_dir, "Aditya_Sawant_Resume_2026.pdf")
        with open(self.resume_latest, "w", encoding="utf-8") as f:
            f.write("Aditya Sawant - Senior AI Systems Engineer. PyTorch, ONNX, C++, Local-first Neural Cores.")
        os.utime(self.resume_latest, (now - 3600, now - 3600))

        # 3. Project notes
        self.notes_file = os.path.join(self.test_dir, "project_notes.txt")
        with open(self.notes_file, "w", encoding="utf-8") as f:
            f.write(
                "Project Notes & Strategy:\n"
                "1. Complete local-first execution pipeline with zero cloud dependencies.\n"
                "2. Maintain sub-15ms neural intent classification.\n"
                "3. Ensure observable closed-loop verification across all OS actions.\n"
                "4. All message transmissions to test phone 8591211342 verified.\n"
            )

        self.brain = EdithBrain(enable_execution=True)
        # Point file discovery to our isolated workspace
        if self.brain.executor and self.brain.executor.file_discovery:
            self.brain.executor.file_discovery.search_roots = [self.test_dir]
        if self.brain.executor and self.brain.executor.action_registry and self.brain.executor.action_registry.file_discovery:
            self.brain.executor.action_registry.file_discovery.search_roots = [self.test_dir]

    def tearDown(self):
        import shutil
        if os.path.exists(self.test_dir):
            try:
                shutil.rmtree(self.test_dir, ignore_errors=True)
            except Exception:
                pass

    def test_01_compound_file_and_whatsapp_plan_synthesis(self):
        """Planner synthesizes verified 3-step plan: find report -> verify -> prepare WhatsApp message."""
        goal = "Find the latest project report and send it to Rahul on WhatsApp"
        plan = TaskPlanner.create_plan(
            goal=goal,
            raw_steps=[],
            step_directives=[],
            context=self.brain.context
        )
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.steps), 3)
        self.assertEqual(plan.steps[0].handler, "file_operation")
        self.assertEqual(plan.steps[0].parameters.get("subaction"), "find")
        self.assertEqual(plan.steps[1].handler, "file_operation")
        self.assertEqual(plan.steps[1].parameters.get("subaction"), "verify_file")
        self.assertEqual(plan.steps[2].handler, "whatsapp_send")
        self.assertTrue(plan.steps[2].requires_confirmation)

    def test_02_compound_file_and_whatsapp_execution(self):
        """Executes compound file search and pipes discovered path to WhatsApp transmission."""
        goal = "Find the latest project report and send it to Rahul Sharma on WhatsApp"
        plan = TaskPlanner.create_plan(
            goal=goal,
            raw_steps=[],
            step_directives=[],
            context=self.brain.context
        )
        res = TaskPlanner.execute_plan(plan, self.brain.executor, self.brain.context, dry_run=False)
        self.assertIn(res.get("status"), ["success", "needs_confirmation", "partial"])
        # Step 1 discovered latest report
        step1 = plan.steps[0]
        self.assertIn("project_report_final.docx", step1.execution_result.get("path", ""))
        # Step 3 attached report to WhatsApp message
        step3 = plan.steps[2]
        self.assertEqual(step3.parameters.get("attachment_path"), step1.execution_result.get("path"))

    def test_03_compound_file_and_email_draft_plan_and_execution(self):
        """Synthesizes and executes: find resume -> verify -> draft email with attachment."""
        goal = "Find my resume, attach it to an email to the recruiter, and save the email as a draft"
        plan = TaskPlanner.create_plan(
            goal=goal,
            raw_steps=[],
            step_directives=[],
            context=self.brain.context
        )
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.steps), 3)
        self.assertEqual(plan.steps[2].handler, "email_draft")
        self.assertIn("recruiter", plan.steps[2].parameters.get("recipient", "").lower())

        res = TaskPlanner.execute_plan(plan, self.brain.executor, self.brain.context, dry_run=False)
        self.assertEqual(res.get("status"), "success")
        step3 = plan.steps[2]
        self.assertEqual(step3.execution_result.get("status"), "success")
        self.assertTrue(os.path.exists(step3.execution_result.get("attachment_path", "")))
        self.assertIn("Aditya_Sawant_Resume_2026.pdf", step3.execution_result.get("attachment_path", ""))

    def test_04_compound_notes_summarize_and_notepad(self):
        """Synthesizes and executes: find notes -> summarize -> insert into Notepad."""
        goal = "Open the project notes, create a short summary, and put it in Notepad"
        plan = TaskPlanner.create_plan(
            goal=goal,
            raw_steps=[],
            step_directives=[],
            context=self.brain.context
        )
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.steps), 3)
        self.assertEqual(plan.steps[0].handler, "file_operation")
        self.assertEqual(plan.steps[1].handler, "file_summarize")
        self.assertIn(plan.steps[2].handler, ["type_text", "gui_action", "draft_content"])
        self.assertEqual(plan.steps[2].parameters.get("app_name"), "notepad")

        res = TaskPlanner.execute_plan(plan, self.brain.executor, self.brain.context, dry_run=False)
        self.assertEqual(res.get("status"), "success")
        # Step 2 extractive summary produced bullet points from file
        step2 = plan.steps[1]
        self.assertEqual(step2.execution_result.get("status"), "success")
        summary_text = step2.execution_result.get("summary", "")
        self.assertIn("Summary", summary_text)
        self.assertTrue("neural intent classification" in summary_text or "closed-loop verification" in summary_text)
        # Step 3 piped summary into Notepad parameters
        step3 = plan.steps[2]
        self.assertIn(step3.parameters.get("text") or step3.parameters.get("content"), [summary_text, ""])

    def test_05_designated_test_number_piping(self):
        """Verifies that messaging workflows can pipe directly to designated test number 8591211342."""
        goal = "Find the latest project report and send it to 8591211342 on WhatsApp"
        plan = TaskPlanner.create_plan(
            goal=goal,
            raw_steps=[],
            step_directives=[],
            context=self.brain.context
        )
        self.assertIsNotNone(plan)
        step3 = plan.steps[2]
        self.assertEqual(step3.parameters.get("recipient"), "8591211342")
        # Execute workflow
        res = TaskPlanner.execute_plan(plan, self.brain.executor, self.brain.context, dry_run=False)
        self.assertIn(res.get("status"), ["success", "needs_confirmation", "partial"])


if __name__ == "__main__":
    unittest.main()
