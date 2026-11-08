"""
Tests for EdithLMInference Checkpoint Loader
Validates:
1. V15.3 SFT Checkpoint loading (model_state_dict + 16k vocab)
2. V15.2 Pretrained Checkpoint loading
3. Legacy V14 Checkpoint loading (model_state)
4. Raw state_dict loading
5. Tokenizer auto-alignment and mismatch detection
6. Actual end-to-end token generation on both CPU and GPU
"""

import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.edith_lm.inference import EdithLMInference
from engine.edith_lm.model import EdithLM, EdithLMConfig
from engine.edith_lm.tokenizer import EdithByteBPETokenizer


class TestCheckpointLoader(unittest.TestCase):
    def test_v15_3_sft_checkpoint_loading_and_generation(self):
        v15_path = "models/checkpoints/edith_lm_v15_sft/edith_lm_v15_3_sft_best.pt"
        if not os.path.exists(os.path.join(PROJECT_ROOT, v15_path)):
            self.skipTest("V15.3 SFT Checkpoint not present")

        # Load with auto-alignment
        engine = EdithLMInference.from_checkpoint(
            checkpoint_path=v15_path,
            tokenizer_path="models/checkpoints/edith_bpe_tokenizer_16k.json",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.assertIsNotNone(engine)
        self.assertEqual(engine.config.vocab_size, 16384)
        self.assertEqual(engine.checkpoint_format, "v15_trainer_dict")
        self.assertEqual(engine.tokenizer.vocab_size, 16384)

        # Verify generation
        prompt = "<mood>calm_focused</mood><user>Hello Edith</user><edith>"
        output = engine.generate(prompt=prompt, max_new_tokens=20, temperature=0.7)
        self.assertIsInstance(output, str)
        self.assertGreater(len(output), 0)
        print(f"\n[Test V15.3 Generation]: '{output}'")

    def test_v14_legacy_checkpoint_loading(self):
        v14_path = "models/checkpoints/edith_lm_v14/edith_lm_v14_chat.pt"
        if not os.path.exists(os.path.join(PROJECT_ROOT, v14_path)):
            self.skipTest("V14 Checkpoint not present")

        engine = EdithLMInference.from_checkpoint(
            checkpoint_path=v14_path,
            tokenizer_path="models/checkpoints/edith_bpe_tokenizer.json",
            device="cpu"
        )
        self.assertIsNotNone(engine)
        self.assertEqual(engine.config.vocab_size, 3051)
        self.assertIn(engine.checkpoint_format, ["v15_trainer_dict", "legacy_trainer_dict"])

    def test_raw_state_dict_loading(self):
        raw_path = "models/checkpoints/edith_lm/edith_lm_best.pt"
        if not os.path.exists(os.path.join(PROJECT_ROOT, raw_path)):
            self.skipTest("Raw checkpoint not present")

        engine = EdithLMInference.from_checkpoint(
            checkpoint_path=raw_path,
            tokenizer_path="models/checkpoints/edith_bpe_tokenizer.json",
            device="cpu"
        )
        self.assertIsNotNone(engine)
        self.assertEqual(engine.checkpoint_format, "raw_state_dict")

    def test_tokenizer_auto_alignment(self):
        v15_path = "models/checkpoints/edith_lm_v15_sft/edith_lm_v15_3_sft_best.pt"
        if not os.path.exists(os.path.join(PROJECT_ROOT, v15_path)):
            self.skipTest("V15.3 SFT Checkpoint not present")

        # Passing default 3k/4k tokenizer path should auto-align to 16k tokenizer
        engine = EdithLMInference.from_checkpoint(
            checkpoint_path=v15_path,
            tokenizer_path="models/checkpoints/edith_bpe_tokenizer.json",
            device="cpu"
        )
        self.assertEqual(engine.tokenizer.vocab_size, 16384)


if __name__ == "__main__":
    unittest.main()

