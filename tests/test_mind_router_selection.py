"""
Tests for MindRouter Checkpoint Selection and Rollback Mechanism
Validates:
1. Production default is V14.
2. V15.3 is selectable as a research candidate.
3. Fallback and rollback execute safely when an invalid or missing checkpoint is requested.
4. Diagnostics report accurate checkpoint, tokenizer, and device metadata.
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.mind_router import MindRouter


class TestMindRouterSelection(unittest.TestCase):
    def test_default_production_is_v14(self):
        router = MindRouter()
        diag = router.get_diagnostics()
        self.assertEqual(diag.get("active_version"), "v14")
        self.assertEqual(diag.get("tier"), "production")
        self.assertFalse(diag.get("rollback_occurred"))
        self.assertEqual(diag.get("status"), "operational")

    def test_v15_3_research_candidate_selection(self):
        v15_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_lm_v15_sft", "edith_lm_v15_3_sft_best.pt")
        if not os.path.exists(v15_path):
            self.skipTest("V15.3 checkpoint not present")

        router = MindRouter(model_version="v15_3")
        diag = router.get_diagnostics()
        self.assertEqual(diag.get("active_version"), "v15_3")
        self.assertEqual(diag.get("tier"), "research")
        self.assertEqual(diag.get("vocab_size"), 16384)
        self.assertFalse(diag.get("rollback_occurred"))

    def test_invalid_version_triggers_safe_rollback_to_v14(self):
        # Request non-existent version
        router = MindRouter(model_version="non_existent_future_v99")
        diag = router.get_diagnostics()
        # Should cleanly roll back to production v14
        self.assertEqual(diag.get("active_version"), "v14")
        self.assertEqual(diag.get("status"), "rolled_back")
        self.assertTrue(diag.get("rollback_occurred"))
        self.assertIn("non_existent_future_v99", diag.get("rollback_reason"))
        self.assertIsNotNone(router._edith_lm_inference)


if __name__ == "__main__":
    unittest.main()
