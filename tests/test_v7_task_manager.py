"""Unit tests for E.D.I.T.H. V7 TaskManager and Concurrency Control."""

import os
import sys
import time
import threading
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runtime.task_manager import TaskManager, TaskStatus


class TestTaskManager(unittest.TestCase):
    """Test suite for concurrent execution, resource conflict serialization, and cancellation."""

    def setUp(self):
        self.tm = TaskManager(max_concurrent_workers=4)

    def tearDown(self):
        self.tm.shutdown(wait=False)

    def test_independent_tasks_run_concurrently(self):
        """Verify two independent tasks run simultaneously."""
        start_barrier = threading.Barrier(2)

        def worker(worker_id):
            start_barrier.wait(timeout=2.0)
            time.sleep(0.05)
            return f"worker_{worker_id}_done"

        t1 = self.tm.submit_task("t1", "worker 1", worker, args=(1,), required_resources=["cpu1"])
        t2 = self.tm.submit_task("t2", "worker 2", worker, args=(2,), required_resources=["cpu2"])

        # Wait for both to complete
        for _ in range(30):
            task1 = self.tm.get_task(t1)
            task2 = self.tm.get_task(t2)
            if task1["status"] == "completed" and task2["status"] == "completed":
                break
            time.sleep(0.05)

        self.assertEqual(task1["status"], "completed")
        self.assertEqual(task2["status"], "completed")
        self.assertEqual(task1["result"], "worker_1_done")
        self.assertEqual(task2["result"], "worker_2_done")

    def test_conflicting_resources_are_serialized(self):
        """Verify two tasks requiring the same resource (e.g. 'gui') are serialized."""
        execution_order = []

        def worker_gui(name):
            execution_order.append(f"{name}_start")
            time.sleep(0.1)
            execution_order.append(f"{name}_end")
            return name

        t1 = self.tm.submit_task("gui_task1", "gui 1", worker_gui, args=("task1",), required_resources=["gui"])
        t2 = self.tm.submit_task("gui_task2", "gui 2", worker_gui, args=("task2",), required_resources=["gui"])

        # Wait for both to complete
        for _ in range(40):
            task1 = self.tm.get_task(t1)
            task2 = self.tm.get_task(t2)
            if task1["status"] == "completed" and task2["status"] == "completed":
                break
            time.sleep(0.05)

        self.assertEqual(task1["status"], "completed")
        self.assertEqual(task2["status"], "completed")
        # Ensure task1 finished before task2 started
        self.assertEqual(execution_order, ["task1_start", "task1_end", "task2_start", "task2_end"])

    def test_task_cancellation(self):
        """Verify task cancellation sets cancellation token and cancels task."""
        def long_running_task(cancel_event):
            while not cancel_event.is_set():
                time.sleep(0.02)
            return "interrupted"

        t_id = self.tm.submit_task("long_task", "test cancel", long_running_task)
        time.sleep(0.15)  # Let it start running

        task_info = self.tm.get_task(t_id)
        self.assertEqual(task_info["status"], "running")

        # Cancel it
        cancelled = self.tm.cancel_task(t_id, reason="Testing user abort")
        self.assertTrue(cancelled)

        time.sleep(0.1)
        final_info = self.tm.get_task(t_id)
        self.assertEqual(final_info["status"], "cancelled")

    def test_verification_callback(self):
        """Verify task verification callback verifies the resulting output."""
        def produces_dict():
            return {"files_indexed": 42}

        def verify_at_least_40(res):
            return res.get("files_indexed", 0) >= 40

        t_id = self.tm.submit_task(
            "index_task", "index files", produces_dict, verifier_fn=verify_at_least_40
        )

        for _ in range(20):
            task = self.tm.get_task(t_id)
            if task["status"] == "completed":
                break
            time.sleep(0.05)

        self.assertEqual(task["status"], "completed")
        self.assertTrue(task["verified_success"])


if __name__ == "__main__":
    unittest.main()
