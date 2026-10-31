"""
E.D.I.T.H. V14 Binary Shard Tokenizer & Auditor.
Packs tokenized dialogues into memory-mapped uint16 binary files (.bin) and document index tables (.idx).
Validates zero <unk> token rate across all tokens.
"""

import os
import sys
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.edith_lm.tokenizer import EdithByteBPETokenizer


class ShardTokenizer:
    def __init__(self, tokenizer_path: Optional[str] = None, output_dir: Optional[str] = None):
        self.tokenizer_path = tokenizer_path or os.path.join(
            PROJECT_ROOT, "models", "checkpoints", "edith_bpe_tokenizer.json"
        )
        self.output_dir = output_dir or os.path.join(
            PROJECT_ROOT, "data_pipeline", "v14", "shards"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self.tokenizer = EdithByteBPETokenizer.from_file(self.tokenizer_path)

    def tokenize_jsonl_to_shard(
        self,
        jsonl_path: str,
        shard_prefix: str,
        max_tokens_per_shard: int = 50_000_000
    ) -> List[Dict[str, Any]]:
        """Tokenizes JSONL corpus into memory-mapped uint16 binary shards and index files."""
        print(f"[ShardTokenizer] Processing {jsonl_path} -> {shard_prefix} shards...")

        shard_idx = 0
        current_tokens: List[int] = []
        doc_offsets: List[int] = [0]
        shard_summaries = []

        total_unk_count = 0
        total_token_count = 0

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if not line.strip():
                    continue
                item = json.loads(line)
                text = item.get("text", "")
                if not text:
                    continue

                # Encode with <bos> and <eos>
                token_ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
                
                # Check for <unk> tokens
                unk_id = self.tokenizer.unk_id
                unk_count = token_ids.count(unk_id)
                total_unk_count += unk_count
                total_token_count += len(token_ids)

                current_tokens.extend(token_ids)
                doc_offsets.append(len(current_tokens))

                # Flush shard if limit reached
                if len(current_tokens) >= max_tokens_per_shard:
                    summary = self._flush_shard(current_tokens, doc_offsets, shard_prefix, shard_idx)
                    shard_summaries.append(summary)
                    shard_idx += 1
                    current_tokens = []
                    doc_offsets = [0]

        if current_tokens:
            summary = self._flush_shard(current_tokens, doc_offsets, shard_prefix, shard_idx)
            shard_summaries.append(summary)

        unk_rate = (total_unk_count / max(total_token_count, 1)) * 100.0
        print(f"[ShardTokenizer] Completed {shard_prefix}!")
        print(f"  Total tokens: {total_token_count:,}")
        print(f"  <unk> tokens: {total_unk_count} (Rate: {unk_rate:.4f}%)")
        assert total_unk_count == 0, f"Violation: Found {total_unk_count} <unk> tokens in shard!"

        return shard_summaries

    def _flush_shard(
        self,
        tokens: List[int],
        doc_offsets: List[int],
        shard_prefix: str,
        shard_idx: int
    ) -> Dict[str, Any]:
        """Writes binary array and index offset array to disk."""
        bin_filename = f"{shard_prefix}_{shard_idx:03d}.bin"
        idx_filename = f"{shard_prefix}_{shard_idx:03d}.idx"

        bin_path = os.path.join(self.output_dir, bin_filename)
        idx_path = os.path.join(self.output_dir, idx_filename)

        # Convert to numpy uint16 array (vocab size is ~3,051 to 8,192, comfortably < 65,535)
        arr = np.array(tokens, dtype=np.uint16)
        arr.tofile(bin_path)

        # Save document boundary offsets
        idx_arr = np.array(doc_offsets, dtype=np.uint32)
        idx_arr.tofile(idx_path)

        summary = {
            "shard": bin_filename,
            "tokens": len(tokens),
            "documents": len(doc_offsets) - 1,
            "size_bytes": os.path.getsize(bin_path),
            "bin_path": bin_path,
            "idx_path": idx_path
        }
        print(f"  [Shard Wrote] {bin_filename}: {len(tokens):,} tokens ({summary['size_bytes'] / (1024*1024):.2f} MB)")
        return summary


class ShardDataset:
    """Fast memory-mapped dataset loader for training batches from .bin shards."""

    def __init__(self, bin_path: str, context_len: int = 512):
        self.bin_path = bin_path
        self.context_len = context_len
        # Memory-map the binary shard (zero memory overhead)
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.total_tokens = len(self.data)
        self.num_chunks = max(1, self.total_tokens // context_len)

    def __len__(self) -> int:
        return self.num_chunks

    def get_batch(self, batch_size: int = 16) -> Tuple[np.ndarray, np.ndarray]:
        """Samples random continuous chunks of length context_len."""
        max_start = self.total_tokens - self.context_len - 1
        starts = np.random.randint(0, max_start, size=batch_size)

        x = np.stack([self.data[s:s + self.context_len] for s in starts])
        y = np.stack([self.data[s + 1:s + self.context_len + 1] for s in starts])
        return x.astype(np.int64), y.astype(np.int64)


if __name__ == "__main__":
    tokenizer_obj = ShardTokenizer()
    processed_dir = os.path.join(PROJECT_ROOT, "data_pipeline", "v14", "processed")
    train_jsonl = os.path.join(processed_dir, "train.jsonl")
    val_jsonl = os.path.join(processed_dir, "val.jsonl")

    if os.path.exists(train_jsonl):
        print(f"\n--- Tokenizing Train Split ---")
        tokenizer_obj.tokenize_jsonl_to_shard(train_jsonl, "train_shard")
    if os.path.exists(val_jsonl):
        print(f"\n--- Tokenizing Val Split ---")
        tokenizer_obj.tokenize_jsonl_to_shard(val_jsonl, "val_shard")

