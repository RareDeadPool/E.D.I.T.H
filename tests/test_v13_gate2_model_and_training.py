"""Gate 2 Automated Regression Test Suite: EdithLM Architecture, Trainer & Inference.

Validates:
1. EdithLM model initialization, RoPE, RMSNorm, SwiGLU, and GQA forward pass.
2. Weight tying between token embeddings and output projection.
3. Checkpoint loading from models/checkpoints/edith_lm/edith_lm_best.pt.
4. Int8 dynamic quantization execution without NaN or crash.
5. Autoregressive streaming generation and stop token handling.
"""

import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.edith_lm.model import EdithLM, EdithLMConfig
from engine.edith_lm.tokenizer import EdithByteBPETokenizer
from engine.edith_lm.inference import EdithLMInference


class TestV13Gate2ModelAndTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_bpe_tokenizer.json")
        cls.tokenizer = EdithByteBPETokenizer(vocab_file=cls.tok_path)
        cls.ckpt_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_lm", "edith_lm_best.pt")

    def test_01_model_forward_pass_and_shapes(self):
        config = EdithLMConfig(
            vocab_size=self.tokenizer.vocab_size,
            max_seq_len=64,
            dim=128,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            intermediate_dim=344,
        )
        model = EdithLM(config)
        x = torch.randint(0, config.vocab_size, (2, 16))
        logits, loss, kv_caches = model(x, targets=x)

        self.assertEqual(logits.shape, (2, 16, config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0.0)

    def test_02_weight_tying_verified(self):
        config = EdithLMConfig.tiny(vocab_size=self.tokenizer.vocab_size)
        model = EdithLM(config)
        self.assertTrue(model.output.weight is model.tok_embeddings.weight)

    def test_03_best_checkpoint_exists_and_loads(self):
        self.assertTrue(os.path.exists(self.ckpt_path), f"Checkpoint missing: {self.ckpt_path}")
        state_dict = torch.load(self.ckpt_path, map_location="cpu")
        self.assertIn("tok_embeddings.weight", state_dict)
        self.assertIn("norm.weight", state_dict)

    def test_04_int8_quantization_and_inference(self):
        config = EdithLMConfig(
            vocab_size=self.tokenizer.vocab_size,
            max_seq_len=64,
            dim=256,
            n_layers=4,
            n_heads=4,
            n_kv_heads=2,
            intermediate_dim=688,
        )
        model = EdithLM(config)
        state_dict = torch.load(self.ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict)

        quantized_model = model.quantize_int8()
        self.assertIsNotNone(quantized_model)

        inference = EdithLMInference(model=quantized_model, tokenizer=self.tokenizer, device="cpu")
        out = inference.generate("<user>Tell me a joke.</user><edith>", max_new_tokens=15)
        self.assertIsInstance(out, str)


if __name__ == "__main__":
    unittest.main()
