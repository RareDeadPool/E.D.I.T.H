"""Gate 1 Automated Regression Test Suite: Custom Tokenizer & Data Pipeline.

Validates:
1. Byte-level BPE tokenizer initialization, encode, decode, and serialization.
2. 100% exact round-trip reconstruction on diverse text, emojis, and Hinglish.
3. Strict 0.00% <unk> guarantee on arbitrary out-of-domain input.
4. Processed dataset existence and non-emptiness (train.jsonl, val.jsonl, test.jsonl).
5. Presence of all 4 real voice failure holdouts in test set.
"""

import json
import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.edith_lm.tokenizer import EdithByteBPETokenizer


class TestV13Gate1TokenizerAndData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_bpe_tokenizer.json")
        cls.tokenizer = EdithByteBPETokenizer(vocab_file=cls.tok_path)

    def test_01_tokenizer_vocabulary_and_special_tokens(self):
        self.assertGreater(self.tokenizer.vocab_size, 256)
        # Check all required special tokens exist
        for tok in ["<pad>", "<bos>", "<eos>", "<user>", "<edith>", "<think>", "</think>", "<act>", "</act>", "<mood>", "<mem>"]:
            self.assertIn(tok, self.tokenizer.special_token_to_id)
            self.assertEqual(self.tokenizer.decode([self.tokenizer.special_token_to_id[tok]]), tok)

    def test_02_zero_unk_guarantee_on_arbitrary_inputs(self):
        challenging_inputs = [
            "My friend is looking for jobs, he's trying from 3 years but he hasn't got one, he's very sad.",
            "Come on, help me a joke.",
            "My friend is very sad.",
            "I have a friend named Harshan.",
            "Bhai mera dost bohot upset hai, 3 saal se try kar raha hai.",
            "Random gibberish: aslkdjasldkfj!@#$%^&*()_+ 129384091283",
            "Emojis: 💡 🚀 🤖 ❤️ 🎉 🔥 ✨ 🌟",
            "<think>Need to check processes</think><act>{\"action\": \"inspect_processes\"}</act>",
        ]

        for text in challenging_inputs:
            encoded = self.tokenizer.encode(text)
            self.assertNotIn(self.tokenizer.unk_id, encoded, f"Tokenizer emitted <unk> for: {text}")
            decoded = self.tokenizer.decode(encoded)
            self.assertEqual(decoded, text, f"Decoded mismatch for: {text}")

    def test_03_processed_datasets_exist_and_populated(self):
        processed_dir = os.path.join(PROJECT_ROOT, "data_pipeline", "processed")
        for split in ["train.jsonl", "val.jsonl", "test.jsonl"]:
            split_file = os.path.join(processed_dir, split)
            self.assertTrue(os.path.exists(split_file), f"Missing split file: {split_file}")
            with open(split_file, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            self.assertGreater(len(lines), 30, f"Split {split} is too small: {len(lines)}")

    def test_04_permanent_regression_holdouts_in_test_set(self):
        test_file = os.path.join(PROJECT_ROOT, "data_pipeline", "processed", "test.jsonl")
        with open(test_file, "r", encoding="utf-8") as f:
            test_records = [json.loads(line) for line in f if line.strip()]

        raw_texts = [r.get("text", "") for r in test_records]
        # Check all 4 real failure cases are preserved
        self.assertTrue(any("help me a joke" in t for t in raw_texts), "Case A missing from test set")
        self.assertTrue(any("looking for jobs" in t and "3 years" in t for t in raw_texts), "Case B missing from test set")
        self.assertTrue(any("My friend is very sad." in t for t in raw_texts), "Case C missing from test set")
        self.assertTrue(any("Harshan" in t for t in raw_texts), "Case D missing from test set")


if __name__ == "__main__":
    unittest.main()
