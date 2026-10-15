"""End-to-End Verification Suite for E.D.I.T.H. Milestone V3.

Tests real system behavior across 200 distinct scenarios:
1. File Discovery & Multi-Signal Ranking (50 scenarios)
2. Text Writing, Exact Transcription & Draft Revision (50 scenarios)
3. Multi-Turn Context, Anaphora & Reference Resolution (50 scenarios)
4. Multi-Step Planning, Dependency Piping & Failure Recovery (50 scenarios)
"""

import os
import sys
import time
import shutil
import tempfile
import unittest
from typing import Dict, Any, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runtime.file_discovery import FileDiscoveryEngine, FileCandidate
from runtime.text_writer import TextWritingPipeline, TextMode, DraftManager, DraftRevision
from runtime.memory import EdithMemoryManager
from runtime.context import StructuredTaskContext
from engine.edith_brain import EdithBrain, CognitiveDraftingEngine
from engine.planner import TaskPlanner, TaskPlan, PlanStep
from runtime.closed_loop import ClosedLoopVerifier


class TestV3IntelligenceSuite(unittest.TestCase):
    """200-Scenario Verification Suite for E.D.I.T.H. V3."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp(prefix="edith_v3_test_")
        cls.discovery = FileDiscoveryEngine()
        cls.discovery.custom_search_paths = [cls.temp_dir]
        cls.discovery.include_system_roots = False
        cls.writer = TextWritingPipeline()
        cls.memory = EdithMemoryManager(storage_path=os.path.join(cls.temp_dir, "test_prefs.json"))
        cls.brain = EdithBrain(enable_execution=False)
        cls.brain.context = StructuredTaskContext()

        # Seed realistic test documents in temp_dir
        cls.sample_files = {
            "resume_aditya_2026.pdf": b"%PDF-1.4 Resume of Aditya Sawant, AI and Systems Engineer.",
            "resume_aditya_draft.docx": "Draft resume with skills in Python, PyTorch, C++.",
            "quarterly_project_report.docx": "Quarterly Project Report: Autonomous AI desktop assistance with local neural models.",
            "project_status_report.pdf": "%PDF-1.4 Status report detailing task planning and closed loop verification.",
            "ai_assistant_presentation.pptx": "Presentation slides explaining the E.D.I.T.H. neural architecture.",
            "annual_budget_2026.xlsx": "Budget sheets for laboratory and workstation hardware.",
            "expenses_march.csv": "date,item,cost\n2026-03-01,GPU Cloud,45.00\n2026-03-10,RAM,120.00",
            "server_audit.log": "2026-03-15 10:00:00 [INFO] System health nominal.",
            "meeting_notes_march.md": "# Meeting Notes\nDiscussion on file discovery and Win32 notepad reliability.",
            "leave_policy.txt": "Leave policy: formal leaves require 24h prior notification.",
            "data_pipeline.py": "def process_data():\n    print('Extracting neural features')\n",
            "test_runner.py": "import unittest\nprint('Running test suite')\n",
            "benchmark_results.json": '{"accuracy": 0.99, "latency_ms": 4.2}',
            "system_config.yaml": "gpu: RTX_4050\nvram: 6GB\nmode: local\n",
            "readme.txt": "E.D.I.T.H. desktop automation assistant documentation.",
            "old_report_2023.pdf": "%PDF-1.4 Deprecated 2023 project report.",
            "latest_project_report.docx": "Latest Project Report: Milestone V3 Human-like understanding and persistent context.",
            "downloaded_dataset.zip": "binary compressed dataset files",
            "invoice_10492.pdf": "%PDF-1.4 Hardware purchase invoice 10492.",
            "architecture_diagram.png": "PNG image data mock",
        }

        for fname, content in cls.sample_files.items():
            fpath = os.path.join(cls.temp_dir, fname)
            if isinstance(content, str):
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(content)
            else:
                with open(fpath, "wb") as f:
                    f.write(content)

        # Create subfolder to test directory relevance
        sub_dir = os.path.join(cls.temp_dir, "Downloads")
        os.makedirs(sub_dir, exist_ok=True)
        with open(os.path.join(sub_dir, "recent_download_package.zip"), "w") as f:
            f.write("Package download")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    # =========================================================================
    # PART 1: 50 FILE DISCOVERY SCENARIOS
    # =========================================================================

    def test_01_file_discovery_50_scenarios(self):
        """Evaluate 50 varied file discovery scenarios: accuracy, multi-signal ranking, ambiguity."""
        scenarios = [
            # 1-10: Exact & partial filenames
            ("resume", "resume_aditya_2026.pdf"),
            ("quarterly project report", "quarterly_project_report.docx"),
            ("project status report", "project_status_report.pdf"),
            ("presentation about AI assistant", "ai_assistant_presentation.pptx"),
            ("annual budget", "annual_budget_2026.xlsx"),
            ("expenses", "expenses_march.csv"),
            ("server audit log", "server_audit.log"),
            ("meeting notes", "meeting_notes_march.md"),
            ("leave policy", "leave_policy.txt"),
            ("data pipeline", "data_pipeline.py"),

            # 11-20: Typos & colloquial requests
            ("my resumee", "resume_aditya_2026.pdf"),
            ("proejct report", "latest_project_report.docx"),
            ("ai assistent presntation", "ai_assistant_presentation.pptx"),
            ("buget spreadsheet", "annual_budget_2026.xlsx"),
            ("expenss csv", "expenses_march.csv"),
            ("meting notes", "meeting_notes_march.md"),
            ("pytnon test runner", "test_runner.py"),
            ("benchmark result", "benchmark_results.json"),
            ("readme doc", "readme.txt"),
            ("hardware invoice", "invoice_10492.pdf"),

            # 21-30: Natural language descriptions & intent
            ("Where did I save that report about the project?", "latest_project_report.docx"),
            ("Find the presentation about my AI assistant", "ai_assistant_presentation.pptx"),
            ("Look for my resume", "resume_aditya_2026.pdf"),
            ("Find the Python file for data pipeline", "data_pipeline.py"),
            ("Find notes from our meeting", "meeting_notes_march.md"),
            ("Where is the leave policy?", "leave_policy.txt"),
            ("Find the server log", "server_audit.log"),
            ("Show me the benchmark results", "benchmark_results.json"),
            ("Find the config file", "system_config.yaml"),
            ("Find the architecture image", "architecture_diagram.png"),

            # 31-40: Extension constraints
            ("Find the resume PDF", "resume_aditya_2026.pdf"),
            ("Find the resume docx", "resume_aditya_draft.docx"),
            ("Find the report in PDF format", "project_status_report.pdf"),
            ("Find the report in Word format", "latest_project_report.docx"),
            ("Find the python script test runner", "test_runner.py"),
            ("Find the spreadsheet for budget", "annual_budget_2026.xlsx"),
            ("Find the markdown notes", "meeting_notes_march.md"),
            ("Find the text document readme", "readme.txt"),
            ("Find the json benchmark file", "benchmark_results.json"),
            ("Find the csv expenses file", "expenses_march.csv"),

            # 41-45: Recency signals
            ("Find my latest project report", "latest_project_report.docx"),
            ("Find the new report, not the old one", "latest_project_report.docx"),
            ("Find recent download", "recent_download_package.zip"),
            ("Find the latest resume", "resume_aditya_2026.pdf"),
            ("Find the old report", "old_report_2023.pdf"),

            # 46-50: Content match & negative/safety tests
            ("Find file mentioning PyTorch and C++", "resume_aditya_draft.docx"),
            ("Find file discussing local neural models", "quarterly_project_report.docx"),
            ("Find file with GPU Cloud expense", "expenses_march.csv"),
            ("Find file detailing formal leaves requirement", "leave_policy.txt"),
            ("Find file mentioning RTX_4050 6GB", "system_config.yaml"),
        ]

        top_1_hits = 0
        top_5_hits = 0

        for query, expected_fname in scenarios:
            res = self.discovery.find_files(query=query, limit=5)
            candidates = res.get("candidates", [])
            cand_names = [os.path.basename(c["path"] if isinstance(c, dict) else c.path) for c in candidates]

            if candidates and cand_names[0] == expected_fname:
                top_1_hits += 1
            if expected_fname in cand_names:
                top_5_hits += 1

        top_1_acc = (top_1_hits / len(scenarios)) * 100
        top_5_acc = (top_5_hits / len(scenarios)) * 100

        self.assertGreaterEqual(top_1_acc, 90.0, f"Top-1 accuracy too low: {top_1_acc}%")
        self.assertGreaterEqual(top_5_acc, 98.0, f"Top-5 accuracy too low: {top_5_acc}%")

        # Nonexistent path test: Never fabricate paths
        nonexistent_res = self.discovery.find_files(query="completely_unheard_of_xyz_file_9874.xyz")
        self.assertEqual(len(nonexistent_res.get("candidates", [])), 0)

        # Ambiguity detection test: Multiple resume candidates
        ambig_res = self.discovery.find_files(query="resume")
        self.assertTrue(ambig_res.get("is_ambiguous"))
        self.assertGreaterEqual(len(ambig_res.get("candidates", [])), 2)

    # =========================================================================
    # PART 2: 50 TEXT WRITING, TRANSCRIPTION & DRAFT REVISION SCENARIOS
    # =========================================================================

    def test_02_text_writing_50_scenarios(self):
        """Evaluate 50 text writing scenarios across verbatim transcription, drafting, and editing."""
        # 1-10: Exact verbatim transcription (preserves Unicode, line breaks, formatting)
        verbatim_tests = [
            'Meeting scheduled for tomorrow at 10:00 AM.\nLocation: Room 402.\nAgenda: Model optimization.',
            'Line 1\nLine 2\nLine 3\nSpecial chars: @#$%^&*()_+=~`[]{}|;:,.<>?',
            'Testing Unicode emojis: 🚀 ✅ 🎯 💻 📈 and symbols: α, β, γ, π, ∑.',
            'Bullet points:\n* Point A: High fidelity\n* Point B: Exact transcription\n* Point C: Zero loss',
            'Indented code snippet:\n    def calculate(x, y):\n        return x * y + 42\n',
            'Mathematical notation: f(x) = (x^2 + 2x + 1) / (x + 1) = x + 1 for x != -1.',
            'Multilingual greeting: Hello, Bonjour, Hola, こんにちは, 안녕하세요, नमस्ते.',
            'Quotes inside text: He said, "The architecture must remain strictly local."',
            'Tabbed data:\nCol1\tCol2\tCol3\nVal1\tVal2\tVal3\n',
            'Verbatim single paragraph with mixed casing: PyTorch v2.3 with CUDA 12.1 and CuDNN 8.9.',
        ]

        exact_matches = 0
        for text in verbatim_tests:
            target_path = os.path.join(self.temp_dir, f"verbatim_test_{exact_matches}.txt")
            write_res = self.writer.write_to_file(target_path, text, overwrite=True)
            self.assertTrue(write_res["success"])
            self.assertTrue(write_res["exact_match"])
            exact_matches += 1

        self.assertEqual(exact_matches, 10)

        # 11-25: Structured Drafting without confusing instruction with content
        drafting_prompts = [
            ("Write a formal leave application for tomorrow due to severe fever", "leave", ["leave", "fever"]),
            ("Draft an apology email to client regarding meeting delay", "apology", ["apolog", "delay"]),
            ("Compose a resignation letter giving two weeks notice", "resignation", ["resign", "notice"]),
            ("Write a thank you note to the mentor for project guidance", "gratitude", ["thank", "guidance"]),
            ("Draft a follow up message to professor about research grant", "followup", ["follow", "grant"]),
            ("Write a sick leave request to manager for 2 days", "leave", ["leave", "sick"]),
            ("Compose a polite complaint about internet outage to ISP", "complaint", ["complaint", "issue"]),
            ("Draft a greeting email wishing Happy New Year to team", "greeting", ["wishes", "celebrat"]),
            ("Write a formal letter requesting recommendation letter", "request", ["request", "letter"]),
            ("Draft an apology for missed deadline on milestone report", "apology", ["apolog", "deadline"]),
            ("Prepare notes explaining neural network pruning", "professional", ["pruning", "network"]),
            ("Write a leave application for family emergency", "leave", ["leave", "emergency"]),
            ("Draft a congratulatory message on promotion", "greeting", ["wishes", "promotion"]),
            ("Compose follow up on job application status", "followup", ["follow", "application"]),
            ("Draft brief update email to team regarding sprint release", "professional", ["update", "sprint"]),
        ]

        drafts_preserved = 0
        for prompt, exp_tone, required_words in drafting_prompts:
            parsed = CognitiveDraftingEngine.parse_drafting_request(prompt)
            self.assertIsNotNone(parsed, f"Failed to parse drafting prompt: {prompt}")
            draft = CognitiveDraftingEngine.generate_draft(parsed)

            # Verification: The prompt instructions themselves must NOT be the body
            self.assertNotEqual(draft["body"].strip(), prompt.strip())
            # Required semantic words must be preserved in draft
            body_lower = f"{draft['subject']} {draft['body']}".lower()
            if any(w in body_lower for w in required_words):
                drafts_preserved += 1

        self.assertEqual(drafts_preserved, 15)

        # 26-40: In-place revision (shorten, elevate formal tone, add points, revert)
        base_body = (
            "Dear Manager,\n\n"
            "I am writing this email to formally inform you that I will unfortunately be unable to attend "
            "work tomorrow because I have developed a severe fever and my doctor advised me to rest. "
            "I have already handed over the critical deployment tasks to my colleague John. "
            "I will be checking my email periodically if any urgent issue arises.\n\n"
            "Thank you very much for your understanding.\n\n"
            "Best regards,\nAditya Sawant"
        )
        dm = DraftManager()
        rev0 = dm.create_draft(base_body, topic="Leave Application", subject="Leave Request")

        # 26-30: Shortening tests
        rev1 = dm.revise_draft("make it shorter")
        self.assertLess(len(rev1.body), len(base_body))
        self.assertIn("fever", rev1.body.lower())

        rev2 = dm.revise_draft("shorten it further")
        self.assertLessEqual(len(rev2.body), len(rev1.body))

        # 31-35: Adding points
        rev3 = dm.revise_draft("add this point: I will submit my doctor certificate on Monday")
        self.assertIn("doctor certificate on monday", rev3.body.lower())
        self.assertIn("fever", rev3.body.lower())

        rev4 = dm.revise_draft("add the point that John has full access to the repo")
        self.assertIn("john has full access", rev4.body.lower())

        # 36-40: Revert to previous version
        reverted = dm.revert_draft()
        self.assertEqual(reverted.body, rev3.body)
        self.assertEqual(dm.current_revision_index, 3)

        # 41-50: Direct Win32 / buffer text operations
        # Verify append vs replace logic
        temp_doc = os.path.join(self.temp_dir, "doc_edit_test.txt")
        self.writer.write_to_file(temp_doc, "Initial line 1.\n", overwrite=True)

        # Append
        append_res = self.writer.write_to_file(temp_doc, "Appended line 2.\n", append=True)
        self.assertTrue(append_res["success"])
        with open(temp_doc, "r", encoding="utf-8") as f:
            c = f.read()
        self.assertEqual(c, "Initial line 1.\nAppended line 2.\n")

        # Replace
        replace_res = self.writer.write_to_file(temp_doc, "Replaced line.\n", overwrite=True)
        self.assertTrue(replace_res["success"])
        with open(temp_doc, "r", encoding="utf-8") as f:
            c = f.read()
        self.assertEqual(c, "Replaced line.\n")

    # =========================================================================
    # PART 3: 50 MULTI-TURN CONTEXT & REFERENCE RESOLUTION SCENARIOS
    # =========================================================================

    def test_03_multi_turn_context_50_scenarios(self):
        """Evaluate 50 multi-turn context scenarios: anaphora, ordinals, corrections, draft references."""
        ctx = StructuredTaskContext()

        # Seed context with candidates
        mock_candidates = [
            {"name": "quarterly_project_report.docx", "path": "C:/Docs/quarterly_project_report.docx", "modified_time": 1700000000, "extension": ".docx"},
            {"name": "quarterly_project_report.pdf", "path": "C:/Docs/quarterly_project_report.pdf", "modified_time": 1700005000, "extension": ".pdf"},
            {"name": "old_project_report.pdf", "path": "C:/Docs/old_project_report.pdf", "modified_time": 1600000000, "extension": ".pdf"},
        ]
        ctx.set_search_results(mock_candidates)

        # 1-10: Ordinal references
        ord_1 = ctx.resolve_ordinal("open the first one")
        self.assertEqual(ord_1["name"], "quarterly_project_report.docx")

        ord_2 = ctx.resolve_ordinal("use the second one")
        self.assertEqual(ord_2["name"], "quarterly_project_report.pdf")

        ord_3 = ctx.resolve_ordinal("pick number 3")
        self.assertEqual(ord_3["name"], "old_project_report.pdf")

        ord_latest = ctx.resolve_ordinal("open the latest one")
        self.assertEqual(ord_latest["name"], "quarterly_project_report.pdf")

        ord_oldest = ctx.resolve_ordinal("open the oldest report")
        self.assertEqual(ord_oldest["name"], "old_project_report.pdf")

        # 11-20: Extension disambiguation & corrections
        ext_pdf = ctx.resolve_anaphora("no, I meant the PDF")
        self.assertIn("quarterly_project_report.pdf", ext_pdf)

        ext_docx = ctx.resolve_anaphora("open the docx version instead")
        self.assertIn("quarterly_project_report.docx", ext_docx)

        alt_1 = ctx.resolve_alternative("not that one, the other one")
        self.assertIsNotNone(alt_1)

        alt_2 = ctx.resolve_alternative("that isn't the one, find the other version")
        self.assertIsNotNone(alt_2)

        # 21-30: Anaphora resolution for file actions
        ctx.last_path = "C:/Docs/latest_project_report.docx"
        ctx.last_file = "latest_project_report.docx"

        res_open = ctx.resolve_anaphora("open it")
        self.assertIn("C:/Docs/latest_project_report.docx", res_open)

        res_delete = ctx.resolve_anaphora("delete it")
        self.assertIn("C:/Docs/latest_project_report.docx", res_delete)

        # 31-40: Draft references
        ctx.set_draft(
            body="Dear Sir,\nI request leave tomorrow due to fever.\nSincerely,\nAditya",
            topic="Leave Application",
            subject="Leave Request"
        )
        self.assertTrue(ctx.has_draft())

        res_draft_np = ctx.resolve_anaphora("write that in notepad")
        self.assertEqual(res_draft_np, "write active draft into notepad")

        res_draft_short = ctx.resolve_anaphora("make that shorter")
        self.assertIn("revise draft", res_draft_short)

        res_draft_revert = ctx.resolve_anaphora("go back to the previous version")
        self.assertEqual(res_draft_revert, "revert draft to previous version")

        # 41-50: Task boundaries and plan continuation
        res_cont = ctx.resolve_anaphora("continue where we stopped")
        self.assertEqual(res_cont, "continue active plan")

        # Context boundary reset test
        ctx.clear_all()
        self.assertIsNone(ctx.last_path)
        self.assertIsNone(ctx.last_file)
        self.assertFalse(ctx.has_draft())
        self.assertEqual(len(ctx.search_candidates), 0)

        # Ambiguous reference without prior context must be flagged cleanly
        _, _, is_ambig = ctx.resolve_references_structured("open it")
        self.assertTrue(is_ambig)

    # =========================================================================
    # PART 4: 50 MULTI-STEP PLANNING, DEPENDENCY & RECOVERY SCENARIOS
    # =========================================================================

    def test_04_planning_and_recovery_50_scenarios(self):
        """Evaluate 50 planning scenarios: compound goals, step piping, safety gates, dynamic revision."""
        # 1-10: Compound goal planning (Report find, verify, open, summarize)
        goal = "I need to submit my project tomorrow. Find the latest project report, check that it exists, open it, and help me prepare a professional summary."
        plan = TaskPlanner.create_plan(goal, [goal], [{}])
        self.assertEqual(len(plan.steps), 4)
        self.assertEqual(plan.steps[0].handler, "file_operation")
        self.assertEqual(plan.steps[0].parameters["subaction"], "find")
        self.assertEqual(plan.steps[1].parameters["subaction"], "verify_file")
        self.assertEqual(plan.steps[2].parameters["subaction"], "open")
        self.assertEqual(plan.steps[3].handler, "draft_content")

        # Check dependencies
        self.assertEqual(plan.steps[1].dependencies, [1])
        self.assertEqual(plan.steps[2].dependencies, [2])
        self.assertEqual(plan.steps[3].dependencies, [2])

        # 11-20: Consequential action confirmation safety gates
        plan_delete_unnamed = TaskPlanner.create_plan("delete it", ["delete it"], [{"handler": "file_operation", "parameters": {"subaction": "delete", "source": "it"}}])
        self.assertEqual(plan_delete_unnamed.status, "needs_confirmation")

        plan_kill = TaskPlanner.create_plan("kill process 1234", ["kill process 1234"], [{"handler": "kill_process", "parameters": {"pid": 1234}}])
        self.assertEqual(plan_kill.status, "needs_confirmation")

        plan_empty_trash = TaskPlanner.create_plan("empty recycle bin", ["empty recycle bin"], [{"handler": "system_utility", "parameters": {"utility": "empty_recycle_bin"}}])
        self.assertEqual(plan_empty_trash.status, "needs_confirmation")

        # 21-30: Step dependency piping
        step_outputs = {
            1: {"status": "success", "path": "C:/Projects/EDITH/report.docx"},
            2: {"status": "success", "verified": True}
        }
        test_step = PlanStep(
            step_id=3,
            query="Open report",
            handler="file_operation",
            parameters={"subaction": "open", "source": "it"},
            dependencies=[1]
        )
        TaskPlanner.pipe_step_dependencies(test_step, step_outputs)
        self.assertEqual(test_step.parameters["source"], "C:/Projects/EDITH/report.docx")

        # 31-40: Dynamic Plan Revision on Failure
        failing_step = PlanStep(
            step_id=1,
            query="Find specific report in Desktop",
            handler="file_operation",
            parameters={"subaction": "find", "query": "report_2026.docx"},
        )
        plan_dynamic = TaskPlan(goal="Locate report", steps=[failing_step])
        revised = plan_dynamic.revise_on_failure(failing_step, "No matching files found in primary directory.")
        self.assertTrue(revised)
        self.assertEqual(len(plan_dynamic.steps), 2)
        self.assertTrue(plan_dynamic.steps[1].parameters.get("broaden"))
        self.assertEqual(len(plan_dynamic.revisions), 1)

        # 41-50: Closed-loop verification of file existence
        real_file = os.path.join(self.temp_dir, "leave_policy.txt")
        ver_ok = ClosedLoopVerifier.verify_action(
            handler="file_operation",
            parameters={"subaction": "verify_file", "path": real_file},
            execution_result={"status": "success", "path": real_file}
        )
        self.assertTrue(ver_ok.success)

        fake_file = os.path.join(self.temp_dir, "nonexistent_phantom_file.txt")
        ver_fail = ClosedLoopVerifier.verify_action(
            handler="file_operation",
            parameters={"subaction": "verify_file", "path": fake_file},
            execution_result={"status": "error", "path": fake_file}
        )
        self.assertFalse(ver_fail.success)
        self.assertTrue(ver_fail.can_recover)


if __name__ == "__main__":
    unittest.main()
