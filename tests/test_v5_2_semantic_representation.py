"""Unit and Generalization Tests for Semantic Representation (V5.2).

Validates:
1. DeterministicTokenHashEncoder vs SubwordCharacterNGramEncoder vs ContextualSemanticEncoder.
2. Controlled semantically equivalent requests receive high similarity under the neural encoder.
3. Similar words with opposite contextual meanings are distinguished (positive discrimination gap).
4. Typo and spelling variations maintain semantic proximity.
5. Post-restart representation invariance across separate model instantiations.
"""

import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.semantic_encoder import (
    DeterministicTokenHashEncoder,
    SubwordCharacterNGramEncoder,
    ContextualSemanticEncoder,
    benchmark_semantic_representations
)


class TestV52SemanticRepresentation(unittest.TestCase):
    """Test suite for semantic representations, subword encoding, and contextual embeddings."""

    @classmethod
    def setUpClass(cls):
        cls.baseline = DeterministicTokenHashEncoder()
        cls.subword = SubwordCharacterNGramEncoder()
        cls.neural = ContextualSemanticEncoder()
        ckpt_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "semantic_encoder_v1.pt")
        if os.path.exists(ckpt_path):
            cls.neural.load_checkpoint(ckpt_path)

    def test_semantically_equivalent_paraphrases(self):
        """Verify that semantically equivalent requests receive high similarity under the neural encoder."""
        equivalent_pairs = [
            ("I'm annoyed because it failed again.", "This keeps breaking and it's getting frustrating."),
            ("It failed another time. Help me figure out why.", "Can you explain why this execution broke again?"),
            ("Can you fix this? It's the third time it has broken.", "Please repair this issue, it broke repeatedly."),
            ("This is still not working, what am I missing?", "I am stuck on this error, what did I overlook?"),
        ]

        for a, b in equivalent_pairs:
            sim_baseline = float(torch.dot(self.baseline.encode(a), self.baseline.encode(b)).item())
            sim_neural = self.neural.compute_similarity(a, b)
            # Neural encoder must dramatically outperform baseline on paraphrase alignment
            self.assertGreater(sim_neural, 0.50, f"Expected high neural similarity for paraphrases: '{a}' vs '{b}'")
            self.assertGreater(sim_neural, sim_baseline, f"Neural similarity must exceed baseline for: '{a}' vs '{b}'")

    def test_context_dependent_meaning_shifts(self):
        """
        Verify that sentences sharing vocabulary but with opposite contextual meanings
        are separated by the neural encoder (hard negative discrimination).
        """
        # "That failure was useful..." is positive/constructive; "...cost us an important deadline" is detrimental
        a = "That failure was useful; now I understand the bug."
        b = "The failure cost us an important deadline."
        sim_shift = self.neural.compute_similarity(a, b)

        # "returned success, but result is wrong" vs "returned success and result was verified"
        c = "The command returned success, but the actual result is wrong."
        d = "The command returned success and the result was verified."
        sim_outcome = self.neural.compute_similarity(c, d)

        # Baseline gets confused by shared words "failure" and "success"
        base_shift = float(torch.dot(self.baseline.encode(a), self.baseline.encode(b)).item())
        base_outcome = float(torch.dot(self.baseline.encode(c), self.baseline.encode(d)).item())

        # Neural model must assign substantially lower similarity to contextual opposites than baseline
        self.assertLess(sim_shift, 0.40, "Contextual opposites sharing words must not receive high similarity")
        self.assertLess(sim_shift, base_shift, "Neural model should distinguish hard negatives better than bag-of-words")
        self.assertLess(sim_outcome, 0.35, "Unverified failure must be distinguished from verified success")

    def test_typo_and_morphological_robustness(self):
        """Verify that character n-gram and neural encoders gracefully handle spelling errors."""
        typo_cases = [
            ("open notepad", "opn notepd"),
            ("execution failed with timeout", "executon faild with timout"),
            ("terminate background process", "termnate backgroud proces")
        ]

        for orig, typo in typo_cases:
            base_sim = float(torch.dot(self.baseline.encode(orig), self.baseline.encode(typo)).item())
            subword_sim = float(torch.dot(self.subword.encode(orig), self.subword.encode(typo)).item())
            neural_sim = self.neural.compute_similarity(orig, typo)

            # Baseline collapses because MD5 hashes of typos are completely uncorrelated
            self.assertLess(base_sim, 0.30, f"Baseline should be fragile to typos on '{orig}'")
            # Subword and neural encoders must maintain high proximity
            self.assertGreater(subword_sim, 0.35, f"Subword encoder must withstand typos on '{orig}'")
            self.assertGreater(neural_sim, 0.85, f"Neural encoder must maintain high similarity for '{orig}' vs '{typo}'")

    def test_post_restart_representation_invariance(self):
        """Verify that reloading checkpoint in a fresh model produces bitwise identical representations."""
        ckpt_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "semantic_encoder_v1.pt")
        if not os.path.exists(ckpt_path):
            self.skipTest("semantic_encoder_v1.pt not found")

        model_1 = ContextualSemanticEncoder()
        model_1.load_checkpoint(ckpt_path)

        model_2 = ContextualSemanticEncoder()
        model_2.load_checkpoint(ckpt_path)

        test_sentences = [
            "Open Google Chrome and search for scientific papers.",
            "Connection refused on port 5432 after 3 retries.",
            "No, I meant the PDF document, not the Word file."
        ]

        for s in test_sentences:
            vec_1 = model_1.encode(s)
            vec_2 = model_2.encode(s)
            diff = (vec_1 - vec_2).abs().max().item()
            self.assertEqual(diff, 0.0, f"Representations across model reloads must be bitwise identical for: '{s}'")


if __name__ == "__main__":
    unittest.main()
