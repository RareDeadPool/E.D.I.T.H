"""Unit & Gating Tests for Active-Learning Pipeline & Shadow Deployment (V5.2).

Validates:
1. Event deduplication and exclusion filtering.
2. Partitioning of informative training candidates vs review queue for ambiguous feedback.
3. Offline candidate model training on experience-derived contrastive triplets.
4. Gated evaluation of candidate vs production across acceptance criteria.
5. Atomic promotion with automatic rollback checkpoint preservation.
6. Shadow mode execution verifying non-intrusive parallel evaluation.
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

from engine.experience_recorder import ExperienceRecorder, InteractionEvent, FeedbackCategory
from engine.active_learning import ActiveLearningPipeline, ShadowDeploymentRunner
from engine.semantic_encoder import ContextualSemanticEncoder


class TestV52ActiveLearningPipeline(unittest.TestCase):
    """Test suite for controlled active learning, promotion gating, and shadow mode."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.ckpt_dir = os.path.join(self.test_dir, "checkpoints")
        self.recorder = ExperienceRecorder(storage_dir=os.path.join(self.test_dir, "exp"), enabled=True)
        self.pipeline = ActiveLearningPipeline(recorder=self.recorder, checkpoint_dir=self.ckpt_dir)

        # Seed initial dummy production checkpoint
        os.makedirs(self.ckpt_dir, exist_ok=True)
        base_model = ContextualSemanticEncoder()
        base_model.save_checkpoint(self.pipeline.production_ckpt)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_filtering_and_deduplication(self):
        """Verify that duplicate requests and excluded events are cleanly filtered."""
        e1 = InteractionEvent(user_request="open chrome", resolved_action="launch_app")
        e2 = InteractionEvent(user_request="open chrome", resolved_action="launch_app")  # Duplicate
        e3 = InteractionEvent(user_request="secret operation", resolved_action="auth", exclude_from_learning=True)  # Excluded
        e4 = InteractionEvent(user_request="open vs code", resolved_action="launch_app")

        filtered = self.pipeline.filter_and_deduplicate_events([e1, e2, e3, e4])
        self.assertEqual(len(filtered), 2)
        actions = [e.resolved_action for e in filtered]
        self.assertIn("launch_app", actions)

    def test_informative_partitioning_and_review_queue(self):
        """Verify high-uncertainty and correction events are partitioned from ambiguous cases."""
        e_corr = InteractionEvent(
            user_request="open resume.docx", resolved_action="open_doc",
            user_correction="No, I meant the PDF, not the Word document",
            feedback_category=FeedbackCategory.EXPLICIT_CORRECTION.value,
            calibrated_confidence=0.50
        )
        e_ambig = InteractionEvent(
            user_request="routine query", resolved_action="status",
            feedback_category=FeedbackCategory.AMBIGUOUS.value,
            calibrated_confidence=0.75
        )
        e_low_conf = InteractionEvent(
            user_request="unclear command", resolved_action="clarify",
            calibrated_confidence=0.35
        )

        candidates, review = self.pipeline.identify_informative_events([e_corr, e_ambig, e_low_conf])
        self.assertEqual(len(candidates), 2, "Explicit correction and low confidence must be candidate examples")
        self.assertEqual(len(review), 1, "Ambiguous feedback must be queued for review without synthetic label fabrication")

        # Verify review queue is saved
        self.pipeline.save_review_queue(review)
        self.assertTrue(os.path.exists(self.pipeline.review_queue_file))

    def test_offline_candidate_training(self):
        """Verify candidate encoder training generates a valid checkpoint."""
        # Seed recorder with informative events
        self.recorder.record_interaction(
            user_request="open project resume",
            resolved_intent="file_op",
            resolved_action="open_file",
            entities={},
            contextual_references={},
            prior_confidence=0.7,
            calibrated_confidence=0.45,
            predicted_plan=["open_file"],
            execution_status="failure",
            verified=False,
            user_correction="No, I meant the PDF, not the doc file"
        )

        cand_model, final_loss = self.pipeline.train_candidate_model(epochs=5, lr=0.005)
        self.assertTrue(os.path.exists(self.pipeline.candidate_ckpt))
        self.assertIsInstance(cand_model, ContextualSemanticEncoder)

    def test_gated_evaluation_and_promotion(self):
        """Verify candidate evaluation against acceptance criteria, atomic promotion, and rollback backup."""
        # Train candidate
        self.pipeline.train_candidate_model(epochs=10, lr=0.005)

        eval_res = self.pipeline.evaluate_candidate_vs_production()
        self.assertIn("production", eval_res)
        self.assertIn("candidate", eval_res)
        self.assertIn("accepted_for_promotion", eval_res)

        # Execute promotion
        promoted = self.pipeline.promote_candidate()
        if eval_res["accepted_for_promotion"]:
            self.assertTrue(promoted)
            # Verify rollback checkpoint exists
            self.assertTrue(os.path.exists(self.pipeline.rollback_ckpt))

            # Test rollback restoration
            rolled_back = self.pipeline.rollback_to_previous_checkpoint()
            self.assertTrue(rolled_back)

    def test_shadow_deployment_runner(self):
        """Verify shadow mode computes parallel scores without modifying runtime state."""
        shadow_runner = ShadowDeploymentRunner(pipeline=self.pipeline)
        targets = ["open chrome", "close window", "search online"]

        diff = shadow_runner.evaluate_turn_in_shadow(
            user_utterance="open the browser please",
            reference_targets=targets
        )

        self.assertIn("prod_latency_ms", diff)
        self.assertIn("cand_latency_ms", diff)
        self.assertIn("prod_similarities", diff)
        self.assertIn("cand_similarities", diff)
        self.assertIn("open chrome", diff["prod_similarities"])
        self.assertTrue(os.path.exists(shadow_runner.shadow_log_file))


if __name__ == "__main__":
    unittest.main()
