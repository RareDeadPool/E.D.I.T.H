"""Privacy-Preserving Experience Recorder and Feedback Hierarchy Tests (V5.2).

Validates:
1. Default disabled state (strict opt-in privacy guarantee).
2. Credential, token, and secret redaction.
3. Configurable retention bounds and eviction pruning.
4. Administrative audit, export, clear, and deletion controls.
5. Per-interaction exclusion (/do-not-learn).
6. Objective observed facts vs subjective inferred explanations segregation.
7. Feedback evidence hierarchy (explicit corrections vs negative outcomes vs ambiguous proxies).
"""

import os
import sys
import unittest
import tempfile
import shutil
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.experience_recorder import (
    ExperienceRecorder,
    FeedbackInterpreter,
    FeedbackCategory,
    InteractionEvent
)


class TestV52ExperienceRecorder(unittest.TestCase):
    """Test suite for privacy-preserving experience recording and feedback interpretation."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.recorder = ExperienceRecorder(
            storage_dir=self.test_dir,
            enabled=False,
            max_events=5,
            retention_days=7
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_default_disabled_privacy_protection(self):
        """Verify that experience recording is strictly disabled by default."""
        self.assertFalse(self.recorder.enabled)
        res = self.recorder.record_interaction(
            user_request="open research papers",
            resolved_intent="app_control",
            resolved_action="launch_app",
            entities={"app": "chrome"},
            contextual_references={},
            prior_confidence=0.9,
            calibrated_confidence=0.85,
            predicted_plan=["launch_app"],
            execution_status="success",
            verified=True
        )
        self.assertIsNone(res, "When disabled, record_interaction must return None and store nothing")
        self.assertEqual(len(self.recorder.in_memory_events), 0)
        self.assertFalse(os.path.exists(self.recorder.events_file))

    def test_opt_in_enablement_and_credential_redaction(self):
        """Verify that enabling collection records events and scrubs all passwords and tokens."""
        self.recorder.enabled = True
        self.recorder.collect_raw_transcripts = True

        sensitive_request = "login to database with password: MySuperSecretPassword123 and token: a1b2c3d4e5f60718293a4b5c6d7e8f90"
        event = self.recorder.record_interaction(
            user_request=sensitive_request,
            resolved_intent="db_auth",
            resolved_action="connect_db",
            entities={"user": "admin"},
            contextual_references={},
            prior_confidence=0.8,
            calibrated_confidence=0.75,
            predicted_plan=["connect_db"],
            execution_status="failure",
            verified=False,
            error_message="Authentication failed for bearer a1b2c3d4e5f60718293a4b5c6d7e8f90"
        )

        self.assertIsNotNone(event)
        self.assertNotIn("MySuperSecretPassword123", event.user_request)
        self.assertIn("[REDACTED_SECRET]", event.user_request)
        self.assertNotIn("a1b2c3d4e5f60718293a4b5c6d7e8f90", str(event.error_message))
        self.assertTrue(os.path.exists(self.recorder.events_file))

    def test_retention_limit_pruning(self):
        """Verify that stored events strictly respect max_events bound."""
        self.recorder.enabled = True
        self.recorder.max_events = 3

        for i in range(6):
            self.recorder.record_interaction(
                user_request=f"task query #{i}",
                resolved_intent="task",
                resolved_action=f"action_{i}",
                entities={},
                contextual_references={},
                prior_confidence=0.8,
                calibrated_confidence=0.8,
                predicted_plan=[f"action_{i}"],
                execution_status="success",
                verified=True
            )

        self.assertEqual(len(self.recorder.in_memory_events), 3)
        # Verify oldest events were evicted
        last_req = self.recorder.in_memory_events[-1].user_request
        self.assertIn("task query #5", last_req)

    def test_administrative_controls(self):
        """Verify inspect, export, delete, and clear controls."""
        self.recorder.enabled = True
        e1 = self.recorder.record_interaction(
            user_request="req1", resolved_intent="i1", resolved_action="a1",
            entities={}, contextual_references={}, prior_confidence=0.8, calibrated_confidence=0.8,
            predicted_plan=[], execution_status="success", verified=True
        )
        e2 = self.recorder.record_interaction(
            user_request="req2", resolved_intent="i2", resolved_action="a2",
            entities={}, contextual_references={}, prior_confidence=0.8, calibrated_confidence=0.8,
            predicted_plan=[], execution_status="success", verified=True
        )

        # Inspect
        inspected = self.recorder.inspect_events(limit=10)
        self.assertEqual(len(inspected), 2)

        # Export with anonymization
        export_file = os.path.join(self.test_dir, "export.json")
        exported_count = self.recorder.export_events(export_file, anonymize=True)
        self.assertEqual(exported_count, 2)
        with open(export_file, "r") as f:
            data = json.load(f)
            self.assertEqual(data[0]["session_id"], "anon_session")

        # Delete single event
        deleted = self.recorder.delete_event(e1.event_id)
        self.assertTrue(deleted)
        self.assertEqual(len(self.recorder.in_memory_events), 1)

        # Clear all
        cleared_count = self.recorder.clear_all_events()
        self.assertEqual(cleared_count, 1)
        self.assertEqual(len(self.recorder.in_memory_events), 0)

    def test_per_interaction_exclusion(self):
        """Verify that an interaction can be explicitly excluded from learning."""
        self.recorder.enabled = True
        res = self.recorder.record_interaction(
            user_request="private confidential search",
            resolved_intent="web_search",
            resolved_action="search",
            entities={},
            contextual_references={},
            prior_confidence=0.9,
            calibrated_confidence=0.9,
            predicted_plan=["search"],
            execution_status="success",
            verified=True,
            exclude_from_learning=True
        )
        self.assertIsNone(res, "Excluded interaction must not be recorded")
        self.assertEqual(len(self.recorder.in_memory_events), 0)

    def test_observed_facts_vs_inferred_explanations(self):
        """Verify strict separation between observable interaction events and model hypotheses."""
        self.recorder.enabled = True
        event = self.recorder.record_interaction(
            user_request="open sales spreadsheet",
            resolved_intent="file_op",
            resolved_action="open_file",
            entities={"file": "sales.xlsx"},
            contextual_references={},
            prior_confidence=0.85,
            calibrated_confidence=0.60,
            predicted_plan=["open_file"],
            execution_status="failure",
            verified=False,
            user_correction="No, I meant the PDF, not the Excel file",
            observed_facts={"file_selected": "sales.xlsx", "user_replacement": "sales.pdf"},
            inferred_explanations={"hypothesis": "ambiguous_extension_resolution", "inferred_user_frustration": 0.45}
        )

        self.assertIsNotNone(event)
        self.assertEqual(event.observed_facts["user_replacement"], "sales.pdf")
        self.assertIn("inferred_user_frustration", event.inferred_explanations)
        self.assertNotIn("inferred_user_frustration", event.observed_facts, "Inferences must never be recorded as facts")

    def test_feedback_interpretation_hierarchy(self):
        """Test evidence hierarchy across explicit corrections, negative outcomes, and weak ambiguous proxies."""
        interpreter = FeedbackInterpreter()

        # 1. Explicit Correction -> High Weight (0.95)
        sig_corr = interpreter.interpret_feedback("No, I meant the PDF, not the Word document")
        self.assertEqual(sig_corr.category, FeedbackCategory.EXPLICIT_CORRECTION)
        self.assertGreaterEqual(sig_corr.evidence_weight, 0.90)
        self.assertFalse(sig_corr.is_ambiguous)

        # 2. Negative Outcome -> High Weight (0.90)
        sig_neg = interpreter.interpret_feedback("You opened the wrong file")
        self.assertEqual(sig_neg.category, FeedbackCategory.NEGATIVE_OUTCOME)
        self.assertGreaterEqual(sig_neg.evidence_weight, 0.85)

        # 3. Repeated Failure -> Procedural Unreliability (0.90)
        sig_rep = interpreter.interpret_feedback("The same approach failed again")
        self.assertEqual(sig_rep.category, FeedbackCategory.REPEATED_FAILURE)
        self.assertEqual(sig_rep.learning_directive, "downgrade_procedure_reliability")

        # 4. Explicit Style Preference (0.80)
        sig_style = interpreter.interpret_feedback("Please give me the direct answer first, then explain")
        self.assertEqual(sig_style.category, FeedbackCategory.STYLE_PREFERENCE)
        self.assertEqual(sig_style.details.get("preference"), "direct_first")

        # 5. Positive Confirmation (0.75)
        sig_pos = interpreter.interpret_feedback("Yes, that's exactly what I wanted")
        self.assertEqual(sig_pos.category, FeedbackCategory.POSITIVE_FEEDBACK)
        self.assertGreaterEqual(sig_pos.evidence_weight, 0.70)

        # 6. Ambiguous Conversational Proxy -> Weak Weight (0.15)
        sig_amb = interpreter.interpret_feedback("Okay.")
        self.assertEqual(sig_amb.category, FeedbackCategory.AMBIGUOUS)
        self.assertLessEqual(sig_amb.evidence_weight, 0.20)
        self.assertTrue(sig_amb.is_ambiguous)
        self.assertEqual(sig_amb.learning_directive, "no_update_ambiguous_proxy")


if __name__ == "__main__":
    unittest.main()
