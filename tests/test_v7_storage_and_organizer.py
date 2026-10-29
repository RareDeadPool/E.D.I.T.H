"""Unit tests for E.D.I.T.H. V7 Safe Storage Intelligence & Semantic File Organizer."""

import os
import sys
import shutil
import tempfile
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runtime.storage_analyzer import StorageAnalyzer
from runtime.file_organizer import SemanticFileOrganizer


class TestStorageAnalyzer(unittest.TestCase):
    """Test suite for safe storage analysis, risk categorization, and reversible quarantine."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="edith_test_storage_")
        self.quarantine_dir = os.path.join(self.test_dir, "quarantine_vault")
        self.analyzer = StorageAnalyzer(
            preferences_path=os.path.join(self.test_dir, "test_prefs.json")
        )

        # Create simulated files
        # 1. Low risk cache (Category A)
        self.cache_dir = os.path.join(self.test_dir, "__pycache__")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, "module.cpython-312.pyc")
        with open(self.cache_file, "wb") as f:
            f.write(b"bytecode_data" * 1000)

        # 2. Temp log (Category A)
        self.temp_log = os.path.join(self.test_dir, "debug.log")
        with open(self.temp_log, "w") as f:
            f.write("temporary log lines\n" * 500)

        # 3. Old installer (Category B)
        self.installer = os.path.join(self.test_dir, "setup_tool.msi")
        with open(self.installer, "wb") as f:
            f.write(b"installer_payload" * 5000)

        # 4. Personal photograph (Category C) - NEVER auto-deleted
        self.photo = os.path.join(self.test_dir, "family_vacation.jpg")
        with open(self.photo, "wb") as f:
            f.write(b"jpeg_image_bytes" * 5000)

        # 5. Duplicate personal photos (Category C)
        self.photo_copy = os.path.join(self.test_dir, "family_vacation_copy.jpg")
        with open(self.photo_copy, "wb") as f:
            f.write(b"jpeg_image_bytes" * 5000)

        # 6. Active project file (Category D)
        self.proj_dir = os.path.join(self.test_dir, "my_active_project")
        os.makedirs(self.proj_dir, exist_ok=True)
        self.proj_code = os.path.join(self.proj_dir, "app.py")
        with open(self.proj_code, "w") as f:
            f.write("print('hello world')\n")
        self.analyzer.mark_active_project(self.proj_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_risk_categorization(self):
        """Verify Category A, B, C, D assignment."""
        cat_cache, risk_cache, _, action_cache = self.analyzer.classify_file(self.cache_file, 1000, 0)
        self.assertEqual(cat_cache, "A")
        self.assertEqual(action_cache, "quarantine")

        cat_inst, risk_inst, _, action_inst = self.analyzer.classify_file(self.installer, 5000, 0)
        self.assertEqual(cat_inst, "B")
        self.assertEqual(action_inst, "review")

        cat_photo, risk_photo, _, action_photo = self.analyzer.classify_file(self.photo, 5000, 0)
        self.assertEqual(cat_photo, "C")
        self.assertEqual(action_photo, "keep")  # Personal files are kept!

        cat_proj, risk_proj, _, action_proj = self.analyzer.classify_file(self.proj_code, 100, 0)
        self.assertEqual(cat_proj, "D")
        self.assertEqual(action_proj, "keep")

    def test_duplicate_detection(self):
        """Verify duplicate detection by content hash (SHA-256)."""
        report = self.analyzer.audit_storage(self.test_dir, max_depth=3, check_duplicates=True)
        self.assertGreaterEqual(len(report.duplicates), 1)
        dupe_files = report.duplicates[0]["files"]
        self.assertTrue(any("family_vacation.jpg" in f for f in dupe_files))
        self.assertTrue(any("family_vacation_copy.jpg" in f for f in dupe_files))

    def test_reversible_quarantine_and_restore(self):
        """Verify that Category A files can be quarantined and restored via manifest."""
        report = self.analyzer.audit_storage(self.test_dir, max_depth=3)
        candidates = report.candidates_for_cleanup
        self.assertTrue(len(candidates) >= 1)

        # 1. Dry run
        dry_manifest = self.analyzer.quarantine_candidates(
            candidates, dry_run=True, quarantine_root=self.quarantine_dir
        )
        self.assertEqual(dry_manifest["status"], "dry_run_completed")
        self.assertTrue(os.path.exists(self.cache_file))  # Still exists

        # 2. Execution
        real_manifest = self.analyzer.quarantine_candidates(
            candidates, dry_run=False, quarantine_root=self.quarantine_dir
        )
        self.assertEqual(real_manifest["status"], "quarantine_executed")
        self.assertFalse(os.path.exists(self.cache_file))  # Moved to quarantine

        # Personal photo was NOT quarantined
        self.assertTrue(os.path.exists(self.photo))

        # 3. Rollback / Restore
        restore_res = self.analyzer.restore_quarantine(real_manifest["manifest_path"])
        self.assertTrue(restore_res["success"])
        self.assertTrue(os.path.exists(self.cache_file))  # Successfully restored!


class TestSemanticFileOrganizer(unittest.TestCase):
    """Test suite for semantic taxonomy classification, Git protection, and reversible transactions."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="edith_test_organizer_")
        self.logs_dir = os.path.join(self.test_dir, "tx_logs")
        self.organizer = SemanticFileOrganizer(log_directory=self.logs_dir)

        # Create files for testing
        # 1. Education - Lecture notes
        self.notes_file = os.path.join(self.test_dir, "lecture_04_neural_nets.pdf")
        with open(self.notes_file, "w") as f:
            f.write("lecture notes content")

        # 2. Professional - Resume
        self.resume_file = os.path.join(self.test_dir, "aditya_sawant_resume_2026.docx")
        with open(self.resume_file, "w") as f:
            f.write("resume content")

        # 3. Media - Screenshot
        self.shot_file = os.path.join(self.test_dir, "screenshot_2026-10-29.png")
        with open(self.shot_file, "w") as f:
            f.write("png screenshot content")

        # 4. Games - Save file
        self.save_file = os.path.join(self.test_dir, "quicksave_slot01.sav")
        with open(self.save_file, "w") as f:
            f.write("game save bytes")

        # 5. Git repo simulation
        self.repo_dir = os.path.join(self.test_dir, "my_git_repo")
        os.makedirs(os.path.join(self.repo_dir, ".git"), exist_ok=True)
        self.repo_code = os.path.join(self.repo_dir, "server.py")
        with open(self.repo_code, "w") as f:
            f.write("import socket\n")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_semantic_classification(self):
        """Verify semantic taxonomy classification across categories."""
        cat, sub, conf, _ = self.organizer.classify_file(self.notes_file)
        self.assertEqual(cat, "Education")
        self.assertEqual(sub, "Lecture Notes")
        self.assertGreaterEqual(conf, 0.6)

        cat, sub, conf, _ = self.organizer.classify_file(self.resume_file)
        self.assertEqual(cat, "Professional")
        self.assertEqual(sub, "Resume & CV")

        cat, sub, conf, _ = self.organizer.classify_file(self.shot_file)
        self.assertEqual(cat, "Media")
        self.assertEqual(sub, "Screenshots")

        cat, sub, conf, _ = self.organizer.classify_file(self.save_file)
        self.assertEqual(cat, "Games")
        self.assertEqual(sub, "Save Files")

    def test_git_project_protection(self):
        """Verify files in Git repositories are protected from restructuring."""
        self.assertTrue(self.organizer.is_inside_project_repository(self.repo_code))
        plan = self.organizer.generate_organization_plan(self.test_dir)
        prot_paths = [p["source_path"] for p in plan.protected_files]
        self.assertIn(self.repo_code, prot_paths)

    def test_dry_run_and_reversible_execution(self):
        """Verify dry-run plan, execution, and transaction rollback."""
        dest_root = os.path.join(self.test_dir, "SortedFiles")
        plan = self.organizer.generate_organization_plan(self.test_dir, destination_root=dest_root)
        self.assertGreaterEqual(len(plan.moves), 3)

        # 1. Dry run
        dry_res = self.organizer.execute_organization_plan(plan, dry_run=True)
        self.assertEqual(dry_res["status"], "dry_run_success")
        self.assertTrue(os.path.exists(self.notes_file))

        # 2. Real execution
        exec_res = self.organizer.execute_organization_plan(plan, dry_run=False)
        self.assertEqual(exec_res["status"], "completed")
        self.assertFalse(os.path.exists(self.notes_file))  # Moved to Organized/Education/Lecture Notes/

        # 3. Rollback
        rollback_res = self.organizer.rollback_transaction(exec_res["transaction_file"])
        self.assertTrue(rollback_res["success"])
        self.assertTrue(os.path.exists(self.notes_file))  # Restored to original position!
        self.assertTrue(os.path.exists(self.resume_file))


if __name__ == "__main__":
    unittest.main()
