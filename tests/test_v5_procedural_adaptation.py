"""Procedural Adaptation & Prediction-Error Recovery Verification for E.D.I.T.H. V5.1.

Validates that:
1. An 8-step failure-adaptation-recovery loop operates reliably:
   Attempt -> Verifiable Failure -> Diagnosis -> Procedure Update -> Alternative Selection -> Verification -> Transfer.
2. The agent avoids repeating unconditioned failed actions without learning permanent prohibitions.
3. Procedural memory recovery allows previously failed tools to become valid again when conditions change.
4. Procedural adaptation is strictly decoupled from neural weight updates (no false claims of weight retraining).
"""

import os
import sys
import unittest
import time
import tempfile
import shutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.cognitive_cycle import EdithCognitiveCycle
from engine.multi_memory import MultiSystemMemory


class TestV5ProceduralAdaptation(unittest.TestCase):
    """Test procedural adaptation and prediction-error recovery loop."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_eight_step_adaptation_recovery_loop(self):
        """Execute and verify the full 8-step failure diagnosis and procedural adaptation cycle."""
        session_id = f"test_adapt_{int(time.time()*1000)}"
        cycle = EdithCognitiveCycle(session_id=session_id, memory_storage_dir=self.test_dir)

        # 1. E.D.I.T.H. attempts an authorized task using a valid tool
        task_cmd = "fetch sales analytics from postgres"
        pre1 = cycle.pre_action_cycle(
            user_command=task_cmd,
            neural_prediction={"intent": "query_db", "action": "postgres_query", "confidence": 0.88, "entities": {}}
        )

        # 2. The task fails with a verifiable error
        error_result = {
            "status": "failure",
            "message": "ConnectionRefusedError: [Errno 111] Connection refused on port 5432",
            "target": "localhost:5432"
        }

        # 3. E.D.I.T.H. records actual outcome and triggers metacognition
        # 4. It identifies plausible failure causes and how uncertain they are
        # 5. It updates its record of the relevant procedure
        post1 = cycle.post_action_cycle(
            user_command=task_cmd,
            action="postgres_query",
            execution_result=error_result,
            verified=False,
            latency_ms=120.0
        )

        self.assertFalse(post1["verified"])
        self.assertGreater(post1["prediction_error"], 0.4, "Prediction error must be substantial on unexpected failure")
        self.assertIsNotNone(post1["lesson_learned"], "Metacognition must produce a failure diagnosis and recovery lesson")

        # Verify procedural memory update
        proc_record = cycle.memory.get_procedure("postgres_query")
        self.assertIsNotNone(proc_record)
        self.assertEqual(proc_record.failure_count, 1)
        self.assertLess(proc_record.reliability, 0.5)
        self.assertIn("ConnectionRefusedError", str(proc_record.common_failure_reasons))

        # 6. On next attempt, procedural memory signals low reliability -> selects alternative recovery step
        known_proc = cycle.memory.get_reliable_procedure("postgres_query")
        self.assertIsNone(known_proc, "Unreliable procedure must be filtered from default reliable procedure lookup")

        # Select justified recovery action: start service before querying
        recovery_cmd = "start postgres database service"
        cycle.pre_action_cycle(
            recovery_cmd,
            {"intent": "system_service", "action": "start_service", "confidence": 0.9, "entities": {"service": "postgres"}}
        )
        post_recovery = cycle.post_action_cycle(
            recovery_cmd, "start_service", {"status": "success", "message": "postgres active"}, verified=True
        )
        self.assertTrue(post_recovery["verified"])

        # 7. Re-attempt query and verify the next outcome
        retry_query_cmd = "query sales analytics after service start"
        cycle.pre_action_cycle(
            retry_query_cmd,
            {"intent": "query_db", "action": "postgres_query", "confidence": 0.85, "entities": {}}
        )
        post2 = cycle.post_action_cycle(
            retry_query_cmd, "postgres_query", {"status": "success", "message": "returned 42 rows"}, verified=True
        )
        self.assertTrue(post2["verified"])

        # 8. On later comparable task, uses updated learned experience (reliability improved)
        proc_updated = cycle.memory.get_procedure("postgres_query")
        self.assertGreater(proc_updated.reliability, 0.0, "Procedural reliability must reflect verified recovery")
        self.assertEqual(proc_updated.success_count, 1)
        self.assertEqual(proc_updated.failure_count, 1)

    def test_no_permanent_prohibition_when_conditions_change(self):
        """Verify that a single failure does not result in a permanent prohibition."""
        session_id = f"test_no_perm_ban_{int(time.time()*1000)}"
        cycle = EdithCognitiveCycle(session_id=session_id, memory_storage_dir=self.test_dir)

        # Tool fails under bad conditions (no wifi)
        cycle.post_action_cycle(
            "fetch remote git tags",
            "git_fetch_tags",
            {"status": "failure", "message": "Network is unreachable"},
            verified=False
        )

        proc = cycle.memory.get_procedure("git_fetch_tags")
        self.assertEqual(proc.reliability, 0.0)

        # When network is restored, tool is run again and succeeds
        cycle.post_action_cycle(
            "fetch remote git tags after network restored",
            "git_fetch_tags",
            {"status": "success", "message": "fetched 12 tags"},
            verified=True
        )
        cycle.post_action_cycle(
            "fetch remote git tags periodic sync",
            "git_fetch_tags",
            {"status": "success", "message": "up to date"},
            verified=True
        )

        # Tool reliability must recover to 2/3 = 0.67 >= 0.5
        proc_recovered = cycle.memory.get_reliable_procedure("git_fetch_tags")
        self.assertIsNotNone(proc_recovered, "Tool must be re-admitted to reliable procedures once verified successful")
        self.assertGreaterEqual(proc_recovered.reliability, 0.6)

    def test_procedural_adaptation_decoupled_from_neural_weights(self):
        """Explicitly verify that experience-based procedural adaptation does NOT modify neural weights."""
        session_id = f"test_weight_invariance_{int(time.time()*1000)}"
        cycle = EdithCognitiveCycle(session_id=session_id, memory_storage_dir=self.test_dir)

        if cycle.neural_model is None:
            self.skipTest("Neural model checkpoint not available")

        # Snapshot neural parameter values before adaptation
        initial_params = [p.clone().detach() for p in cycle.neural_model.parameters()]

        # Perform multiple cognitive cycles with failures and recoveries
        for i in range(5):
            cycle.pre_action_cycle("task", {"intent": "t", "action": f"act_{i}", "confidence": 0.8, "entities": {}})
            cycle.post_action_cycle(
                "task", f"act_{i}", {"status": "failure" if i % 2 == 0 else "success"}, verified=(i % 2 != 0)
            )

        # Verify that neural parameter tensors are BITWISE IDENTICAL
        post_params = [p.clone().detach() for p in cycle.neural_model.parameters()]
        for p_init, p_post in zip(initial_params, post_params):
            diff = (p_init - p_post).abs().max().item()
            self.assertEqual(diff, 0.0, "Procedural adaptation must not alter neural network weights without explicit retraining")


if __name__ == "__main__":
    unittest.main()
