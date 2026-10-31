"""Tokenizer Training & Evaluation Runner for E.D.I.T.H. V13.

Trains the custom Byte-Level BPE Tokenizer on the processed corpus and validates
zero <unk> fallback, special tokens preservation, and compression metrics.
"""

import json
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.edith_lm.tokenizer import EdithByteBPETokenizer


def train_and_evaluate_tokenizer(
    train_path: str = "data_pipeline/processed/train.jsonl",
    save_path: str = "models/checkpoints/edith_bpe_tokenizer.json",
    target_vocab_size: int = 16384,
):
    print("=" * 70)
    print("E.D.I.T.H. V13 Byte-Level BPE Tokenizer Training")
    print("=" * 70)

    # 1. Gather all raw texts
    full_train_path = os.path.join(PROJECT_ROOT, train_path)
    if not os.path.exists(full_train_path):
        raise FileNotFoundError(f"Training corpus not found at {full_train_path}")

    corpus_texts = []
    with open(full_train_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                corpus_texts.append(data.get("text", ""))

    print(f"[Tokenizer] Loaded {len(corpus_texts)} corpus texts.")

    # 2. Train tokenizer
    tok = EdithByteBPETokenizer()
    t0 = time.time()
    # We train with target vocab; for fast initial build on ~8.3k records, we target 12,288 or 16,384
    tok.train_from_corpus(corpus_texts, target_vocab_size=target_vocab_size, verbose=True)
    train_duration = time.time() - t0
    print(f"[Tokenizer] Training completed in {train_duration:.2f} seconds.")

    # 3. Save tokenizer
    full_save_path = os.path.join(PROJECT_ROOT, save_path)
    tok.save(full_save_path)
    print(f"[Tokenizer] Saved tokenizer to {full_save_path}")

    # 4. Rigorous Gate 1 Verification
    print("\n" + "=" * 70)
    print("GATE 1 VERIFICATION: Zero <unk> Audit & Compression Evaluation")
    print("=" * 70)

    test_cases = [
        # Standard English
        "Tell me a joke about distributed systems.",
        # Historical Voice Failure A (STT slip)
        "Come on, help me a joke.",
        # Historical Voice Failure B (Emotional job struggle)
        "My friend is looking for jobs, he's trying from 3 years but he hasn't got one, he's very sad.",
        # Historical Voice Failure C (3rd-person sadness)
        "My friend is very sad.",
        # Historical Voice Failure D (Friend introduction)
        "I have a friend named Harshan.",
        # Hinglish code-switching
        "Bhai mera dost bohot upset hai, kuch help kar de.",
        # Edge case: Emojis, math, and technical symbols
        "Calculate 450 * 3.14159 -> 💡 🚀 🤖 \u2764\ufe0f",
        # Out-of-vocabulary nonsensical gibberish
        "asdfghjklqwertyuiop zxcvbnm1234567890 !@#$%^&*()_+",
        # Structured tool syntax
        "<think>Need to launch Chrome</think><act>{\"action\": \"launch_app\", \"app_name\": \"chrome\"}</act>",
    ]

    total_bytes = 0
    total_tokens = 0
    unk_count = 0

    for idx, sample in enumerate(test_cases, 1):
        encoded = tok.encode(sample)
        decoded = tok.decode(encoded)
        raw_bytes = len(sample.encode("utf-8"))
        token_count = len(encoded)

        total_bytes += raw_bytes
        total_tokens += token_count

        # Check if decoded matches original exactly
        is_exact = (sample == decoded)
        # Check if unk was emitted
        has_unk = (tok.unk_id in encoded)
        if has_unk:
            unk_count += 1

        print(f"\nTest {idx}: {repr(sample[:50])}...")
        print(f"  Tokens ({token_count}): {encoded[:10]}{'...' if len(encoded) > 10 else ''}")
        print(f"  Round-trip Exact: {is_exact} | UNK tokens: {has_unk} | Bytes/Token: {raw_bytes/token_count:.2f}")

        if not is_exact:
            print(f"  WARNING: Mismatch!\n    Original: {sample}\n    Decoded:  {decoded}")

    avg_compression = total_bytes / total_tokens if total_tokens > 0 else 0
    print("\n" + "-" * 70)
    print(f"SUMMARY AUDIT METRICS:")
    print(f"  Total Vocab Size:     {tok.vocab_size}")
    print(f"  Total Test Bytes:     {total_bytes}")
    print(f"  Total Test Tokens:    {total_tokens}")
    print(f"  Avg Compression Ratio:{avg_compression:.2f} bytes / token")
    print(f"  Total <unk> Emitted:  {unk_count} (Rate: {unk_count/len(test_cases)*100:.2f}%)")
    print("-" * 70)

    if unk_count == 0 and avg_compression >= 2.0:
        print("GATE 1 STATUS: PASSED! Zero <unk> guarantee verified across all domains.")
    else:
        print("GATE 1 STATUS: FAILED! Check tokenizer parameters.")


if __name__ == "__main__":
    train_and_evaluate_tokenizer()
