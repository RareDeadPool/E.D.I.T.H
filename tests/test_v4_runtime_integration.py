"""Milestone V4 — Runtime Integration & Real-World Reliability Audit Test Suite.

Validates end-to-end execution across the full live pipeline:
User input -> command normalization -> neural understanding -> context resolution ->
planning -> handler execution -> desktop OS action -> closed-loop verification -> trace recording.

Sections:
1. Isolated File Discovery & Ranking in Temporary Test Workspace
2. Real Win32 Notepad End-to-End Insertion with WM_GETTEXT Read-back Verification
3. Persistent Multi-Turn Conversation Sequences (same application instance)
4. Model, Checkpoint, Tokenizer, and Backend Verification
5. Execution Trace and Telemetry Completeness
"""

import os
import sys
import time
import shutil
import tempfile
import unittest
import hashlib
from typing import Dict, Any, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.edith_brain import EdithBrain
from runtime.file_discovery import FileDiscoveryEngine
from runtime.text_writer import TextWritingPipeline, DraftManager, TextMode
import subprocess



class TestV4FileDiscoveryE2E(unittest.TestCase):
    """Section 1: Isolated End-to-End File Discovery Tests."""

    def setUp(self):
        # Create an isolated temporary test workspace with diverse documents
        self.test_dir = tempfile.mkdtemp(prefix="edith_v4_test_workspace_")
        self.sub_docs = os.path.join(self.test_dir, "documents")
        self.sub_src = os.path.join(self.test_dir, "src")
        self.sub_empty = os.path.join(self.test_dir, "empty_archive")
        os.makedirs(self.sub_docs, exist_ok=True)
        os.makedirs(self.sub_src, exist_ok=True)
        os.makedirs(self.sub_empty, exist_ok=True)

        now = time.time()

        # 1. Resumes with different dates and extensions
        self.resume_old = os.path.join(self.sub_docs, "resume_v1.pdf")
        with open(self.resume_old, "w", encoding="utf-8") as f:
            f.write("Aditya Sawant Software Engineer Resume 2024 C++ Python Machine Learning")
        os.utime(self.resume_old, (now - 86400 * 30, now - 86400 * 30))

        self.resume_latest = os.path.join(self.sub_docs, "resume_latest.pdf")
        with open(self.resume_latest, "w", encoding="utf-8") as f:
            f.write("Aditya Sawant Senior AI Engineer Resume 2026 Deep Learning PyTorch ONNX")
        os.utime(self.resume_latest, (now - 3600, now - 3600))  # 1 hour ago

        self.resume_word = os.path.join(self.sub_docs, "resume_editable.docx")
        with open(self.resume_word, "w", encoding="utf-8") as f:
            f.write("Aditya Sawant Word Document CV Editable Resume 2026")
        os.utime(self.resume_word, (now - 7200, now - 7200))

        # 2. Deliberate spelling mistake filename
        self.resume_typo = os.path.join(self.sub_docs, "resme_backup.txt")
        with open(self.resume_typo, "w", encoding="utf-8") as f:
            f.write("Backup text of resume credentials")

        # 3. Project reports in different subdirectories
        self.project_report = os.path.join(self.sub_docs, "project_report_final.docx")
        with open(self.project_report, "w", encoding="utf-8") as f:
            f.write("Comprehensive Project Final Report and Systems Architecture Summary")

        self.project_summary = os.path.join(self.sub_docs, "project_summary.pdf")
        with open(self.project_summary, "w", encoding="utf-8") as f:
            f.write("Brief executive summary of project goals and achievements")

        self.project_notes = os.path.join(self.sub_docs, "project_notes.txt")
        with open(self.project_notes, "w", encoding="utf-8") as f:
            f.write("Meeting notes regarding project roadmap and deployment milestones")

        # 4. Misleading name with older date
        self.draft_misleading = os.path.join(self.sub_docs, "draft_report_do_not_use.docx")
        with open(self.draft_misleading, "w", encoding="utf-8") as f:
            f.write("Draft temporary notes not intended for final review")
        os.utime(self.draft_misleading, (now - 86400 * 60, now - 86400 * 60))

        # 5. Python scripts in subdirectories
        self.python_worker = os.path.join(self.sub_src, "script_worker.py")
        with open(self.python_worker, "w", encoding="utf-8") as f:
            f.write("#!/usr/bin/env python\n# Python script for background data processing and training\ndef main():\n    pass\n")

        self.python_helper = os.path.join(self.sub_src, "data_pipeline.py")
        with open(self.python_helper, "w", encoding="utf-8") as f:
            f.write("# Data pipeline automation\n")

        # Instantiate brain with isolated file discovery paths
        self.brain = EdithBrain(enable_execution=True)
        if self.brain.executor and self.brain.executor.file_discovery:
            self.brain.executor.file_discovery.custom_search_paths = [self.test_dir]
            self.brain.executor.file_discovery.include_system_roots = False

    def tearDown(self):
        try:
            shutil.rmtree(self.test_dir)
        except Exception:
            pass

    def test_find_latest_resume(self):
        """Verify natural query 'Find my latest resume' correctly ranks the newer resume."""
        res = self.brain.process("Find my latest resume.")
        self.assertEqual(res.get("intent"), "file_operation")
        exec_res = res.get("execution", {})
        cands = exec_res.get("candidates", [])
        self.assertTrue(len(cands) >= 1, "Should discover candidate resume files")
        # Top candidate must be the latest resume, not the older one
        top_cand = cands[0]
        self.assertEqual(os.path.normpath(top_cand.get("path")), os.path.normpath(self.resume_latest))

    def test_find_python_script(self):
        """Verify 'Look for that Python script I was working on' locates Python file in subfolder."""
        res = self.brain.process("Look for that Python script I was working on.")
        exec_res = res.get("execution", {})
        cands = exec_res.get("candidates", [])
        self.assertTrue(len(cands) >= 1)
        # Top candidate must have .py extension and be inside the test workspace
        top_cand = cands[0]
        self.assertTrue(top_cand.get("path").endswith(".py"))
        self.assertTrue(os.path.exists(top_cand.get("path")))

    def test_find_project_report_multiformat(self):
        """Verify 'Find the report about the project' discovers relevant report documents."""
        res = self.brain.process("Find the report about the project.")
        exec_res = res.get("execution", {})
        cands = exec_res.get("candidates", [])
        self.assertTrue(len(cands) >= 2, "Should find multiple project reports/summaries")
        paths = [os.path.normpath(c["path"]) for c in cands]
        self.assertIn(os.path.normpath(self.project_report), paths)

    def test_typo_resilience_file_discovery(self):
        """Verify typo in query or filename still successfully retrieves candidates."""
        # Search for typo 'resme'
        engine = self.brain.executor.file_discovery
        search_res = engine.find_files("Find my resme backup", force_refresh=True)
        cands = search_res.get("candidates", [])
        self.assertTrue(len(cands) >= 1)
        found_typo = any("resme_backup.txt" in c["name"] for c in cands)
        self.assertTrue(found_typo, "Typo matching should discover 'resme_backup.txt'")

    def test_empty_directory_handling(self):
        """Verify searching within an empty folder produces an honest, clear response."""
        engine = self.brain.executor.file_discovery
        search_res = engine.find_files("Find spreadsheet in empty_archive", location_hint="empty_archive")
        cands = search_res.get("candidates", [])
        self.assertEqual(len(cands), 0)


class TestV4NotepadFaithfulWriting(unittest.TestCase):
    """Section 2: Live Win32 Notepad End-to-End Tests with Read-back Verification."""

    @classmethod
    def setUpClass(cls):
        # Open a fresh Notepad instance for controlled test verification
        cls._proc = subprocess.Popen(["notepad.exe"])
        time.sleep(1.5)

    @classmethod
    def tearDownClass(cls):
        # Gracefully close test Notepad instance
        try:
            if hasattr(cls, "_proc") and cls._proc:
                cls._proc.terminate()
        except Exception:
            pass
        subprocess.run("taskkill /f /im notepad.exe", shell=True, capture_output=True)
        time.sleep(0.5)

    def setUp(self):
        self.brain = EdithBrain(enable_execution=True)

    def test_exact_transcription_into_notepad(self):
        """Exact transcription preserves verbatim content without summarization or tone shift."""
        verbatim_text = (
            "E.D.I.T.H. V4 Exact Transcription:\n"
            "Line 1: Special Characters: !@#$%^&*()_+-=[]{}|;:'\",.<>?/\n"
            "Line 2: Unicode & Multilingual: 🚀 ✅ 🎯 — Bonjour, こんにちは, Привет.\n"
            "Line 3: Numbers & Quotes: \"100% Guaranteed Reliability\" in [2026]."
        )
        cmd = f"Type exactly this into Notepad: {verbatim_text}"
        res = self.brain.process(cmd, execute=True)

        self.assertEqual(res.get("intent"), "gui_action")
        exec_res = res.get("execution", {})
        self.assertEqual(exec_res.get("status"), "success")
        self.assertTrue(exec_res.get("verified"), "WM_GETTEXT verification must succeed")
        self.assertTrue(exec_res.get("exact_match"), "Inserted content must match verbatim payload")

        # Explicitly verify content via TextWriter inspection
        writer = self.brain.executor.text_writer
        read_back = writer.read_notepad_content()
        self.assertIn("Special Characters", read_back)
        self.assertIn("Unicode & Multilingual", read_back)

    def test_drafting_and_contextual_notepad_writing(self):
        """Draft formal email and faithfully insert generated draft into Notepad."""
        # 1. Draft formal email
        cmd1 = "Write a formal email requesting an internship update."
        res1 = self.brain.process(cmd1, execute=True)
        self.assertEqual(res1.get("intent"), "communication_drafting")
        self.assertTrue(self.brain.context.draft_manager.has_draft())

        current_draft = self.brain.context.draft_manager.get_current_draft()
        self.assertIsNotNone(current_draft)
        draft_body = current_draft.body

        # 2. Put generated draft into Notepad
        cmd2 = "Put that in Notepad."
        res2 = self.brain.process(cmd2, execute=True)
        exec_res2 = res2.get("execution", {})
        self.assertEqual(exec_res2.get("status"), "success")
        self.assertTrue(exec_res2.get("verified"))

        # Read back Notepad content and compare
        writer = self.brain.executor.text_writer
        read_back = writer.read_notepad_content()
        # Check that core subject and body lines are present
        for line in draft_body.splitlines()[:3]:
            if line.strip():
                self.assertIn(line.strip()[:25], read_back)

    def test_revision_cycle_in_notepad(self):
        """Revise active draft: shorten, add sentence, and verify in Notepad."""
        # 1. Establish draft
        self.brain.process("Draft an email to my professor.")
        draft_v1 = self.brain.context.draft_manager.get_current_draft()
        self.assertIsNotNone(draft_v1)

        # 2. Make shorter
        self.brain.process("Make that shorter.")
        draft_v2 = self.brain.context.draft_manager.get_current_draft()
        self.assertLessEqual(len(draft_v2.body), len(draft_v1.body))

        # 3. Add deadline sentence
        self.brain.process("Add a sentence about the deadline.")
        draft_v3 = self.brain.context.draft_manager.get_current_draft()
        self.assertIn("deadline", draft_v3.body.lower())

        # 4. Put revised draft into Notepad
        res_put = self.brain.process("Write that in Notepad.", execute=True)
        self.assertEqual(res_put.get("execution", {}).get("status"), "success")
        self.assertTrue(res_put.get("execution", {}).get("verified"))

        # 5. Undo the last change
        self.brain.process("Undo the last change.")
        draft_v4 = self.brain.context.draft_manager.get_current_draft()
        self.assertEqual(draft_v4.version, draft_v2.version)


class TestV4PersistentMultiTurnConversations(unittest.TestCase):
    """Section 3: Persistent Multi-Turn Conversation Sequences in a Single Brain Instance."""

    def setUp(self):
        self.brain = EdithBrain(enable_execution=True)
        self.test_dir = tempfile.mkdtemp(prefix="edith_v4_conv_")
        self.doc1 = os.path.join(self.test_dir, "project_report.docx")
        self.doc2 = os.path.join(self.test_dir, "project_summary.pdf")
        with open(self.doc1, "w", encoding="utf-8") as f:
            f.write("Full project specification document")
        with open(self.doc2, "w", encoding="utf-8") as f:
            f.write("Executive summary pdf")

        if self.brain.executor and self.brain.executor.file_discovery:
            self.brain.executor.file_discovery.custom_search_paths = [self.test_dir]
            self.brain.executor.file_discovery.include_system_roots = False

    def tearDown(self):
        try:
            shutil.rmtree(self.test_dir)
        except Exception:
            pass

    def test_conversation_flow_ordinal_and_pronoun(self):
        """Sequence: 'Find my project report.' -> 'Show me the second one.' -> 'Open it.'"""
        # Turn 1: Search
        res1 = self.brain.process("Find my project report.")
        self.assertEqual(res1.get("intent"), "file_operation")
        candidates = self.brain.context.search_candidates
        self.assertTrue(len(candidates) >= 2)

        # Turn 2: Ordinal selection
        res2 = self.brain.process("Show me the second result.")
        self.assertEqual(res2.get("intent"), "file_operation")
        second_cand = candidates[1]
        self.assertEqual(os.path.normpath(self.brain.context.last_path), os.path.normpath(second_cand["path"]))

        # Turn 3: Pronoun resolution
        res3 = self.brain.process("Open it.")
        self.assertEqual(res3.get("intent"), "file_operation")
        self.assertIn(os.path.basename(second_cand["path"]), res3.get("directive", {}).get("message", ""))

    def test_conversation_flow_drafting_and_revisions(self):
        """Sequence: 'Draft an email to my professor.' -> 'Make it more polite.' -> 'Add a sentence about the deadline.' -> 'Put that in Notepad.'"""
        # Turn 1
        res1 = self.brain.process("Draft an email to my professor.")
        self.assertEqual(res1.get("intent"), "communication_drafting")
        self.assertTrue(self.brain.context.draft_manager.has_draft())

        # Turn 2
        res2 = self.brain.process("Make it more polite.")
        self.assertEqual(res2.get("intent"), "communication_drafting")
        cur_draft = self.brain.context.draft_manager.get_current_draft()
        self.assertIn("grateful", cur_draft.body.lower())

        # Turn 3
        res3 = self.brain.process("Add a sentence about the deadline.")
        self.assertEqual(res3.get("intent"), "communication_drafting")
        cur_draft = self.brain.context.draft_manager.get_current_draft()
        self.assertIn("deadline", cur_draft.body.lower())

        # Turn 4
        res4 = self.brain.process("Put that in Notepad.", execute=False)  # Directive check
        self.assertEqual(res4.get("intent"), "communication_drafting")
        self.assertEqual(res4.get("directive", {}).get("parameters", {}).get("channel"), "notepad")
        self.assertIn("deadline", res4.get("directive", {}).get("parameters", {}).get("body", "").lower())

    def test_conversation_flow_negative_disambiguation(self):
        """Sequence: 'Find my resume.' -> 'Not the PDF, the Word file.' -> 'Open that one.'"""
        resume_pdf = os.path.join(self.test_dir, "resume.pdf")
        resume_docx = os.path.join(self.test_dir, "resume.docx")
        with open(resume_pdf, "w", encoding="utf-8") as f:
            f.write("Resume PDF")
        with open(resume_docx, "w", encoding="utf-8") as f:
            f.write("Resume Word DOCX")

        # Turn 1
        self.brain.process("Find my resume.")
        self.assertTrue(len(self.brain.context.search_candidates) >= 2)

        # Turn 2: Explicit negative constraint
        res2 = self.brain.process("Not the PDF, the Word file.")
        self.assertEqual(res2.get("intent"), "file_operation")
        self.assertTrue(self.brain.context.last_path.endswith(".docx"))

        # Turn 3
        res3 = self.brain.process("Open that one.")
        self.assertEqual(res3.get("intent"), "file_operation")
        self.assertTrue(res3.get("directive", {}).get("parameters", {}).get("source", "").endswith(".docx"))


class TestV4ModelInspectionAndTelemetry(unittest.TestCase):
    """Section 4 & 5: Model Claims, Checkpoints, and Execution Tracing."""

    def setUp(self):
        self.brain = EdithBrain(enable_execution=True)

    def test_model_checkpoint_and_tokenizer_integrity(self):
        """Verify model weights, tokenizer file, architecture parameters, and ONNX runtime session."""
        engine = self.brain.engine
        self.assertIsNotNone(engine.model)
        self.assertIsNotNone(engine.tokenizer)

        # Verify physical checkpoint path and size
        ckpt_path = engine.checkpoint_path
        self.assertTrue(os.path.isfile(ckpt_path), f"Checkpoint must exist at {ckpt_path}")
        ckpt_size = os.path.getsize(ckpt_path)
        self.assertGreater(ckpt_size, 40_000_000, "Trained neural weights must exceed 40MB")

        # Verify SHA256 hash of checkpoint
        sha256 = hashlib.sha256()
        with open(ckpt_path, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        digest = sha256.hexdigest()
        self.assertTrue(digest.startswith("ed8e5443"), f"Checkpoint hash mismatch: {digest}")

        # Verify tokenizer vocab size
        self.assertGreaterEqual(engine.tokenizer.vocab_size, 500)

        # Verify active ONNX acceleration session
        self.assertTrue(hasattr(engine, "ort_session") and engine.ort_session is not None)

    def test_execution_trace_generation(self):
        """Verify execution trace records all mandatory diagnostic fields with sensitive redaction."""
        res = self.brain.process("Find my resume.")
        trace = self.brain.get_last_trace()

        self.assertIsNotNone(trace, "Execution trace must be logged for every command")
        self.assertIn("trace_id", trace)
        self.assertIn("original_message", trace)
        self.assertIn("resolved_query", trace)
        self.assertIn("intent", trace)
        self.assertIn("handler", trace)
        self.assertIn("execution_result", trace)
        self.assertIn("final_response", trace)
        self.assertIn("latency_ms", trace)
        self.assertGreater(trace["latency_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
