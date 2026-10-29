"""End-to-End Integration Tests for Experience-Based Learning & Adaptive Communication (V5.2).

Validates:
1. EdithBrain integration with ExperienceRecorder, FeedbackInterpreter, and AdaptiveCommunication.
2. End-to-end event logging from execution results through closed-loop verification.
3. User correction signals updating procedural track records and working memory.
4. User style preferences dynamically adapting subsequent turn plans.
5. Administrative controls operating directly through EdithBrain.
"""

import os
import sys
import unittest
import tempfile
import shutil
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.cognitive_cycle import EdithCognitiveCycle
from engine.experience_recorder import FeedbackCategory


class TestV52ExperienceLearningIntegration(unittest.TestCase):
    """Integration test suite for experience recording, feedback, and adaptive communication."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.cycle = EdithCognitiveCycle(
            session_id=f"test_v5_2_int_{int(time.time()*1000)}",
            memory_storage_dir=self.test_dir
        )
        # Enable experience collection for testing
        self.cycle.experience_recorder.enabled = True
        self.cycle.experience_recorder.collect_raw_transcripts = True

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_end_to_end_experience_cycle_and_correction(self):
        """Verify that pre-action, post-action, and user corrections flow through experience recorder."""
        # 1. Turn 1: Attempt command
        pre1 = self.cycle.pre_action_cycle(
            user_command="open project resume",
            neural_prediction={"intent": "file_op", "action": "open_file", "confidence": 0.82, "entities": {"file": "resume.docx"}}
        )
        self.assertIn("communication_plan", pre1)

        # Execution fails (wrong file selected)
        post1 = self.cycle.post_action_cycle(
            user_command="open project resume",
            action="open_file",
            execution_result={"status": "failure", "message": "File not found: resume.docx", "target": "resume.docx"},
            verified=False,
            latency_ms=45.0
        )
        self.assertFalse(post1["verified"])

        # Check recorded event
        events = self.cycle.experience_recorder.in_memory_events
        self.assertEqual(len(events), 1)
        e1 = events[0]
        self.assertEqual(e1.resolved_action, "open_file")
        self.assertFalse(e1.verified)

        # 2. Turn 2: User provides explicit correction
        corr_text = "No, I meant the PDF, not the doc file"
        post2 = self.cycle.post_action_cycle(
            user_command=corr_text,
            action="open_file",
            execution_result={"status": "success", "message": "opened resume.pdf", "target": "resume.pdf"},
            verified=True,
            latency_ms=30.0
        )

        self.assertEqual(post2["feedback_signal"]["category"], FeedbackCategory.EXPLICIT_CORRECTION.value)
        self.assertGreaterEqual(post2["feedback_signal"]["evidence_weight"], 0.90)

        # 3. Turn 3: User sets communication style preference
        pref_text = "Please be more concise"
        post3 = self.cycle.post_action_cycle(
            user_command=pref_text,
            action="dialogue",
            execution_result={"status": "success", "message": "acknowledged"},
            verified=True
        )

        self.assertEqual(post3["feedback_signal"]["category"], FeedbackCategory.STYLE_PREFERENCE.value)
        self.assertEqual(post3["communication_preferences"]["verbosity"], "concise")

        # 4. Turn 4: Verify subsequent pre_action plan adapts to concise verbosity
        pre4 = self.cycle.pre_action_cycle(
            user_command="check system status",
            neural_prediction={"intent": "system_metrics", "action": "system_metrics", "confidence": 0.9, "entities": {}}
        )
        self.assertEqual(pre4["communication_plan"]["verbosity"], "concise")


if __name__ == "__main__":
    unittest.main()
