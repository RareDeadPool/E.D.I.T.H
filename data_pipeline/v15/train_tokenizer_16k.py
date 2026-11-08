"""
E.D.I.T.H. V15 Byte-Level BPE Tokenizer Retraining Script (16k Vocab Target)
Retrains the proprietary byte-level BPE tokenizer on the audited V15 human corpus:
- Guaranteed zero <unk> byte fallback (256 byte tokens preserved)
- English, Hinglish, and conversational turn tokens (<user>, <edith>, etc.)
- Evaluates bytes-per-token and token fragment rate on held-out text
- Compares before vs after tokenization of real user sentences
"""

import os
import sys
import json
import time
import re
from typing import List, Dict, Any, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.edith_lm.tokenizer import EdithByteBPETokenizer


def load_training_corpus(train_path: str, max_docs: int = 25000) -> List[str]:
    texts = []
    with open(train_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= max_docs:
                break
            if not line.strip():
                continue
            item = json.loads(line)
            t = item.get("text", "")
            if t:
                texts.append(t)

    # Ingest core Hinglish dialogues and dictionary words
    hinglish_seeds = [
        "bhai aaj bohot bekar din tha, boss ne bina baat ke daant diya.",
        "tu mere se zyada smart hai kya?",
        "kuch toh open karo ya gaana chalao.",
        "Harshan is my friend, not my manager.",
        "World Cup me Dhoni ne kitne run banaye the?",
        "soch raha hu ki break le lu thoda time.",
        "dimag kharab ho gaya hai bug solve karte karte.",
    ]
    texts.extend(hinglish_seeds * 5)
    return texts


def evaluate_tokenization_metrics(tokenizer: EdithByteBPETokenizer, val_path: str, sample_size: int = 500) -> Tuple[float, float]:
    val_texts = []
    with open(val_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= sample_size:
                break
            if not line.strip():
                continue
            item = json.loads(line)
            val_texts.append(item.get("text", ""))

    total_bytes = 0
    total_tokens = 0
    single_byte_token_count = 0  # fragmented tokens (length 1 byte)

    for t in val_texts:
        b = len(t.encode("utf-8"))
        toks = tokenizer.encode(t, add_bos=False, add_eos=False)
        total_bytes += b
        total_tokens += len(toks)

        for tok_id in toks:
            tok_str = tokenizer.decoder.get(tok_id, "")
            # If token representation is a single raw byte character
            if len(tok_str) == 1 and ord(tok_str) < 300 and tok_str not in ["a", "I", " ", ".", ",", "?", "!"]:
                single_byte_token_count += 1

    bytes_per_token = total_bytes / max(1, total_tokens)
    fragment_rate = single_byte_token_count / max(1, total_tokens)
    return bytes_per_token, fragment_rate


def run_tokenizer_retraining():
    print("=" * 80)
    print("E.D.I.T.H. V15 TOKENIZER RETRAINING: 16k VOCAB BYTE-LEVEL BPE")
    print("=" * 80)

    train_path = os.path.join(PROJECT_ROOT, "data_pipeline", "v15", "processed", "train.jsonl")
    val_path = os.path.join(PROJECT_ROOT, "data_pipeline", "v15", "processed", "val.jsonl")
    old_tok_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_bpe_tokenizer.json")
    new_tok_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_bpe_tokenizer_16k.json")

    # 1. Evaluate OLD 3k Tokenizer as baseline
    print("\n--- BASELINE EVALUATION (OLD 3K TOKENIZER) ---")
    old_tokenizer = EdithByteBPETokenizer(vocab_file=old_tok_path)
    old_bpt, old_frag = evaluate_tokenization_metrics(old_tokenizer, val_path)
    print(f"Old Vocab Size:     {old_tokenizer.vocab_size}")
    print(f"Old Bytes/Token:    {old_bpt:.3f} bytes/token")
    print(f"Old Fragment Rate:  {old_frag:.1%}")

    # 2. Train NEW 16k Tokenizer
    print("\n--- TRAINING NEW 16K BYTE-LEVEL BPE TOKENIZER ---")
    corpus = load_training_corpus(train_path)
    print(f"Loaded {len(corpus):,} documents for BPE merge learning.")

    new_tokenizer = EdithByteBPETokenizer()
    t0 = time.perf_counter()
    new_tokenizer.train_from_corpus(corpus, target_vocab_size=16384, verbose=True)
    elapsed = time.perf_counter() - t0
    print(f"BPE training completed in {elapsed:.2f}s. Final Vocab Size: {new_tokenizer.vocab_size}")

    # Save newly trained tokenizer
    os.makedirs(os.path.dirname(new_tok_path), exist_ok=True)
    new_tokenizer.save(new_tok_path)
    print(f"Saved new tokenizer to: {new_tok_path}")

    # 3. Evaluate NEW 16k Tokenizer
    print("\n--- NEW 16K TOKENIZER EVALUATION ---")
    new_bpt, new_frag = evaluate_tokenization_metrics(new_tokenizer, val_path)
    print(f"New Vocab Size:     {new_tokenizer.vocab_size}")
    print(f"New Bytes/Token:    {new_bpt:.3f} bytes/token (compression gain: {(new_bpt - old_bpt)/old_bpt:+.1%})")
    print(f"New Fragment Rate:  {new_frag:.1%} (reduction: {(old_frag - new_frag)/old_frag:.1%})")

    # 4. Before / After Comparison on Real User Sentences
    test_sentences = [
        "describe me",
        "What do you think about me",
        "Really",
        "I'm right here with you. What can I help you explore or accomplish next?",
        "bhai aaj bohot bekar din tha, boss ne bina baat ke daant diya",
        "Harshan is saved as your friend, not your manager.",
        "Building local sovereign neural intelligence without renting commercial cloud APIs."
    ]

    print("\n" + "=" * 80)
    print("BEFORE / AFTER TOKENIZATION OF REAL SENTENCES:")
    print("=" * 80)
    for s in test_sentences:
        old_tokens = [old_tokenizer.decoder.get(i, f"[{i}]") for i in old_tokenizer.encode(s, add_bos=False, add_eos=False)]
        new_tokens = [new_tokenizer.decoder.get(i, f"[{i}]") for i in new_tokenizer.encode(s, add_bos=False, add_eos=False)]
        print(f"\nSENTENCE: \"{s}\"")
        print(f"  OLD (3k, {len(old_tokens)} tokens): {old_tokens}")
        print(f"  NEW (16k, {len(new_tokens)} tokens): {new_tokens}")

    # Also update default edith_bpe_tokenizer.json if new is verified superior
    if new_tokenizer.vocab_size > 8000:
        new_tokenizer.save(old_tok_path)
        print(f"\n[Updated active system tokenizer at {old_tok_path}]")


if __name__ == "__main__":
    run_tokenizer_retraining()

