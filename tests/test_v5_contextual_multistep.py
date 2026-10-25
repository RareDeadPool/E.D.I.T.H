"""Comprehensive Automated Test Suite for Contextual Entity Tracking,
Multi-Step Workflows, Output Dependency Piping, and Expanded Tasks (E.D.I.T.H. V5).

Validates:
1. Multi-turn contextual entity tracking across domains:
   - Calculations (expr, result, formatted string) and resolution in downstream turns.
   - File operations (create, read, append) and anaphoric file references.
   - Screenshots and downstream WhatsApp/Email attachment routing.
   - System metrics (CPU, RAM, Disk, Battery) and downstream Notepad insertion.
   - Clipboard operations (copy, read) and round-trip verification.
   - Web page reading and extractive summarization.
2. Multi-step command decomposition:
   - Numbered list structures ("1. Check battery 2. Write the stats in notepad", "Step 1: ... Step 2: ...").
   - Sequential transition markers ("First ..., then ..., finally ...").
   - Output dependency piping across discrete steps (calc result -> notepad, screenshot -> whatsapp).
3. Compound autonomous plan generation and closed-loop verification:
   - Screenshot + WhatsApp plan.
   - Math solve + Notepad plan (+ save).
   - Metrics + Notepad plan.
   - File create + open plan.
   - Zip and Unzip archive operations.
"""

import os
import sys
import unittest
import tempfile
import shutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.edith_brain import EdithBrain, split_multi_step_command
from engine.planner import TaskPlanner, PlanStep, TaskPlan
from runtime.context import StructuredTaskContext
from runtime.action_registry import ActionRegistry
from runtime.closed_loop import ClosedLoopVerifier


class TestV5ContextualMultiStep(unittest.TestCase):
    """Production test suite for contextual tracking and multi-step tasks."""

    def setUp(self):
        self.brain = EdithBrain(enable_execution=True)
        self.test_dir = tempfile.mkdtemp(prefix="edith_test_v5_")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            try:
                shutil.rmtree(self.test_dir)
            except Exception:
                pass

    # =========================================================================
    # PART 1: Multi-Step Command Decomposition
    # =========================================================================

    def test_01_split_numbered_steps(self):
        """Numbered lists (1. ... 2. ...) decompose into discrete executable steps."""
        cmd = "1. Check battery 2. Write the stats in notepad"
        steps = split_multi_step_command(cmd)
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0], "Check battery")
        self.assertEqual(steps[1], "Write the stats in notepad")

    def test_02_split_step_prefixed_commands(self):
        """Commands with 'Step 1: ... Step 2: ...' decompose accurately."""
        cmd = "Step 1: Calculate 25 * 4 Step 2: Write the result in notepad"
        steps = split_multi_step_command(cmd)
        self.assertEqual(len(steps), 2)
        self.assertIn("Calculate 25 * 4", steps[0])
        self.assertIn("Write the result in notepad", steps[1])

    def test_03_split_sequential_transitions(self):
        """Sequential markers 'First ..., then ..., finally ...' decompose cleanly."""
        cmd = "First calculate 15 * 40, then write the result in notepad, finally save the file"
        steps = split_multi_step_command(cmd)
        self.assertEqual(len(steps), 3)
        self.assertEqual(steps[0], "calculate 15 * 40")
        self.assertEqual(steps[1], "write the result in notepad")
        self.assertEqual(steps[2], "save the file")

    # =========================================================================
    # PART 2: Multi-Turn Contextual Entity Tracking
    # =========================================================================

    def test_04_calculation_context_tracking_and_piping(self):
        """Math computation stores result in context; subsequent turns resolve 'the result'."""
        # Turn 1: Compute
        res1 = self.brain.process("calculate 15 * 40", execute=True)
        self.assertIn(res1["status"], ["OPERATIONAL", "success"])
        self.assertIsNotNone(self.brain.context.last_calculation)
        self.assertEqual(self.brain.context.last_calculation["result"], "15 * 40 = 600")

        # Turn 2: Reference 'the result' in Notepad typing
        res2 = self.brain.process("write the result in notepad", execute=False)
        self.assertEqual(res2.get("directive", {}).get("handler"), "gui_action")
        self.assertIn("600", str(res2.get("directive", {}).get("parameters", {}).get("text", "")))

        # Turn 3: Reference 'that answer' for WhatsApp to designated test number
        res3 = self.brain.process("send that answer to 8591211342 on WhatsApp", execute=False)
        self.assertEqual(res3.get("directive", {}).get("handler"), "whatsapp_send")
        pars = res3.get("directive", {}).get("parameters", {})
        self.assertIn(pars.get("recipient"), ["8591211342", "Test Recipient"])
        self.assertIn("600", str(pars.get("content", "")))

    def test_05_screenshot_context_tracking_and_whatsapp_piping(self):
        """Screenshot capture stores path; subsequent turn references 'the screenshot'."""
        # Turn 1: Take screenshot
        res1 = self.brain.process("take a screenshot", execute=True)
        self.assertIn(res1["status"], ["OPERATIONAL", "success"])
        self.assertIsNotNone(self.brain.context.last_screenshot)
        ss_path = self.brain.context.last_screenshot
        self.assertTrue(os.path.exists(ss_path))

        # Turn 2: Reference 'the screenshot' to share on WhatsApp
        res2 = self.brain.process("send the screenshot to 8591211342 on WhatsApp", execute=False)
        self.assertEqual(res2.get("directive", {}).get("handler"), "whatsapp_send")
        pars = res2.get("directive", {}).get("parameters", {})
        self.assertIn(pars.get("recipient"), ["8591211342", "Test Recipient"])
        self.assertEqual(os.path.normpath(pars.get("attachment_path")), os.path.normpath(ss_path))

        # Cleanup generated screenshot
        try:
            os.remove(ss_path)
        except Exception:
            pass

    def test_06_system_metrics_context_tracking(self):
        """System metrics query stores telemetry; subsequent turn puts stats in notepad."""
        # Turn 1: Check metrics
        res1 = self.brain.process("check battery", execute=True)
        self.assertIn(res1["status"], ["OPERATIONAL", "success"])
        self.assertIsNotNone(self.brain.context.last_metrics)

        # Turn 2: Put stats in notepad
        res2 = self.brain.process("write the stats in notepad", execute=False)
        self.assertEqual(res2.get("directive", {}).get("handler"), "gui_action")
        text_param = res2.get("directive", {}).get("parameters", {}).get("text", "")
        self.assertTrue(len(text_param) > 0)
        self.assertTrue(any(k in text_param for k in ["CPU", "RAM", "Battery", "%"]))

    def test_07_clipboard_copy_and_read(self):
        """Copies text to Windows clipboard and reads it back faithfully."""
        test_payload = "E.D.I.T.H. Autonomous Mission 2026 Verified"

        # Turn 1: Copy to clipboard
        res1 = self.brain.process(f"copy '{test_payload}' to clipboard", execute=True)
        self.assertIn(res1["status"], ["OPERATIONAL", "success"])
        self.assertEqual(self.brain.context.last_clipboard, test_payload)

        # Turn 2: Read clipboard
        res2 = self.brain.process("read clipboard", execute=True)
        self.assertIn(res2["status"], ["OPERATIONAL", "success"])
        exec_data = res2.get("execution", {})
        self.assertEqual(exec_data.get("content"), test_payload)

    def test_08_file_create_read_and_append_cycle(self):
        """Creates a file with content, reads it back, and appends additional lines."""
        test_file = os.path.join(self.test_dir, "mission_log.txt")
        test_file_clean = test_file.replace("\\", "/")

        # 1. Create file with content
        res1 = self.brain.process(f"create file '{test_file_clean}' with content: Initial mission briefing.", execute=True)
        self.assertIn(res1["status"], ["OPERATIONAL", "success"])
        self.assertTrue(os.path.exists(test_file))
        self.assertEqual(os.path.normpath(self.brain.context.last_created_file), os.path.normpath(test_file))

        # 2. Read file
        res2 = self.brain.process(f"read file '{test_file_clean}'", execute=True)
        self.assertIn(res2["status"], ["OPERATIONAL", "success"])
        self.assertIn("Initial mission briefing", res2.get("execution", {}).get("content", ""))

        # 3. Append to file
        res3 = self.brain.process(f"append 'Phase 2 underway.' to '{test_file_clean}'", execute=True)
        self.assertIn(res3["status"], ["OPERATIONAL", "success"])

        # Verify combined content on filesystem
        with open(test_file, "r", encoding="utf-8") as f:
            full_content = f.read()
        self.assertIn("Initial mission briefing", full_content)
        self.assertIn("Phase 2 underway", full_content)

    def test_09_archive_zip_and_unzip(self):
        """Compresses directory into zip and extracts it."""
        sub_dir = os.path.join(self.test_dir, "data_to_zip")
        os.makedirs(sub_dir, exist_ok=True)
        with open(os.path.join(sub_dir, "sample.txt"), "w", encoding="utf-8") as f:
            f.write("Payload inside zipped archive.")

        zip_dst = os.path.join(self.test_dir, "archive.zip")
        unzip_dst = os.path.join(self.test_dir, "extracted")

        sub_dir_clean = sub_dir.replace("\\", "/")
        zip_dst_clean = zip_dst.replace("\\", "/")
        unzip_dst_clean = unzip_dst.replace("\\", "/")

        # 1. Zip
        res1 = self.brain.process(f"zip '{sub_dir_clean}' to '{zip_dst_clean}'", execute=True)
        self.assertIn(res1["status"], ["OPERATIONAL", "success"])
        self.assertTrue(os.path.exists(zip_dst))

        # 2. Unzip
        res2 = self.brain.process(f"unzip '{zip_dst_clean}' to '{unzip_dst_clean}'", execute=True)
        self.assertIn(res2["status"], ["OPERATIONAL", "success"])
        self.assertTrue(os.path.exists(os.path.join(unzip_dst, "sample.txt")))

    # =========================================================================
    # PART 3: Compound Autonomous Plans & Output Piping
    # =========================================================================

    def test_10_compound_screenshot_and_whatsapp_plan(self):
        """Synthesizes and validates compound screenshot + WhatsApp plan."""
        goal = "Take a screenshot and send it to 8591211342 on WhatsApp"
        plan = TaskPlanner.create_plan(goal, [goal], [], context=self.brain.context)

        self.assertIsInstance(plan, TaskPlan)
        self.assertEqual(len(plan.steps), 2)
        self.assertEqual(plan.steps[0].handler, "system_utility")
        self.assertEqual(plan.steps[0].parameters.get("utility"), "screenshot")
        self.assertEqual(plan.steps[1].handler, "whatsapp_send")
        self.assertEqual(plan.steps[1].parameters.get("recipient"), "8591211342")
        self.assertEqual(plan.steps[1].dependencies, [1])

    def test_11_compound_calculate_and_notepad_plan(self):
        """Synthesizes compound math computation + Notepad insertion + save plan."""
        goal = "Calculate 15 * 40, write the result in notepad, and save the file"
        plan = TaskPlanner.create_plan(goal, [goal], [], context=self.brain.context)

        self.assertIsInstance(plan, TaskPlan)
        self.assertEqual(len(plan.steps), 3)
        self.assertEqual(plan.steps[0].handler, "compute_math")
        self.assertEqual(plan.steps[0].parameters.get("expr"), "15 * 40")
        self.assertEqual(plan.steps[1].handler, "gui_action")
        self.assertEqual(plan.steps[1].parameters.get("app_name"), "notepad")
        self.assertEqual(plan.steps[1].dependencies, [1])
        self.assertEqual(plan.steps[2].handler, "file_operation")
        self.assertEqual(plan.steps[2].parameters.get("subaction"), "save")
        self.assertEqual(plan.steps[2].dependencies, [2])

    def test_12_compound_metrics_and_notepad_plan(self):
        """Synthesizes compound metrics query + Notepad insertion plan."""
        goal = "Check battery and write the stats in notepad"
        plan = TaskPlanner.create_plan(goal, [goal], [], context=self.brain.context)

        self.assertIsInstance(plan, TaskPlan)
        self.assertEqual(len(plan.steps), 2)
        self.assertEqual(plan.steps[0].handler, "system_metrics")
        self.assertEqual(plan.steps[1].handler, "gui_action")
        self.assertEqual(plan.steps[1].parameters.get("app_name"), "notepad")
        self.assertEqual(plan.steps[1].dependencies, [1])

    def test_13_compound_create_file_write_and_open_plan(self):
        """Synthesizes compound file creation with content + open plan."""
        goal = "Create a file named notes.txt with content hello world and open it"
        plan = TaskPlanner.create_plan(goal, [goal], [], context=self.brain.context)

        self.assertIsInstance(plan, TaskPlan)
        self.assertEqual(len(plan.steps), 2)
        self.assertEqual(plan.steps[0].handler, "file_create")
        self.assertEqual(plan.steps[0].parameters.get("path"), "notes.txt")
        self.assertEqual(plan.steps[0].parameters.get("content"), "hello world")
        self.assertEqual(plan.steps[1].handler, "file_operation")
        self.assertEqual(plan.steps[1].parameters.get("subaction"), "open")
        self.assertEqual(plan.steps[1].dependencies, [1])

    def test_14_live_execution_pipe_step_dependencies(self):
        """Verifies that pipe_step_dependencies correctly transfers artifacts between steps."""
        # Step 1: Simulated math output
        step1_res = {"status": "success", "result": "1200", "action": "compute_math"}

        # Step 2: Notepad writing step depending on step 1
        step2 = PlanStep(
            step_id=2,
            query="Write result into Notepad",
            handler="gui_action",
            parameters={"action": "type", "app_name": "notepad", "text": ""},
            dependencies=[1]
        )

        TaskPlanner.pipe_step_dependencies(step2, {1: step1_res})
        self.assertEqual(step2.parameters["text"], "1200")
        self.assertEqual(step2.parameters["content"], "1200")

        # Step 3: Screenshot output piped to WhatsApp attachment
        step3_res = {"status": "success", "path": "C:/fake/screenshot.png", "action": "screenshot"}
        step4 = PlanStep(
            step_id=4,
            query="Send screenshot to Rahul on WhatsApp",
            handler="whatsapp_send",
            parameters={"recipient": "8591211342", "content": "Here is screenshot", "attachment_path": ""},
            dependencies=[3]
        )
        TaskPlanner.pipe_step_dependencies(step4, {3: step3_res})
        self.assertEqual(step4.parameters["attachment_path"], "C:/fake/screenshot.png")


if __name__ == "__main__":
    unittest.main()
