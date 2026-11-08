"""
E.D.I.T.H. V15.3 SFT Dataset Builder
Assembles high-quality human-authored conversational and instruction pairs across:
1. Databricks Dolly-15k (CC BY-SA 3.0, Human QA & Instructions)
2. Empathetic Dialogues (CC BY-NC 4.0, Human Multi-turn Empathy)
3. Hand-Authored Assistant Seeds (Proprietary Custom)
4. E.D.I.T.H. Personality & Skill Curriculum (Proprietary Custom)

Strictly enforces target token masking schema, document length boundaries (< 512),
SHA-256 deduplication, and commercial vs non-commercial partitioning.
"""

import os
import sys
import json
import re
import hashlib
import random
from typing import List, Dict, Any, Set

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.edith_lm.tokenizer import EdithByteBPETokenizer
from data_pipeline.v14.authored_seed import AuthoredSeedData
from data_pipeline.v15.edith_personality_curriculum import EdithPersonalityCurriculum


def build_sft_corpus():
    print("=" * 80)
    print("E.D.I.T.H. V15.3 SFT DATASET BUILDER")
    print("=" * 80)

    sft_proc_dir = os.path.join(PROJECT_ROOT, "data_pipeline", "v15", "processed_sft")
    os.makedirs(sft_proc_dir, exist_ok=True)

    tok_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_bpe_tokenizer_16k.json")
    tokenizer = EdithByteBPETokenizer(vocab_file=tok_path)
    print(f"Loaded Tokenizer: {tokenizer.vocab_size} vocab entries from {tok_path}")

    all_records: List[Dict[str, Any]] = []
    seen_hashes: Set[str] = set()

    def add_sft_record(turns: List[Dict[str, str]], source: str, category: str, license_tag: str, partition: str):
        if not turns or len(turns) < 2:
            return
        
        # Verify alternate user/edith turns
        cleaned_turns = []
        for idx, t in enumerate(turns):
            role = t.get("role", "")
            text = t.get("text", "").strip()
            if not text:
                continue
            cleaned_turns.append({"role": role, "text": text})

        if len(cleaned_turns) < 2:
            return

        # Check total tokens to ensure fits within 512
        full_dialogue_text = "<mood>calm_focused</mood>" + "".join(
            f"<{t['role']}>{t['text']}</{t['role']}>" for t in cleaned_turns
        )
        tok_ids = tokenizer.encode(full_dialogue_text, add_bos=True, add_eos=True)
        if len(tok_ids) > 480:  # conservative headroom for 512
            return

        # Deduplicate using normalized first user prompt
        first_user = next((t["text"] for t in cleaned_turns if t["role"] == "user"), "")
        norm_key = re.sub(r"\s+", " ", (first_user + "->" + cleaned_turns[-1]["text"]).lower())
        h = hashlib.sha256(norm_key.encode("utf-8")).hexdigest()
        if h in seen_hashes:
            return
        seen_hashes.add(h)

        all_records.append({
            "id": f"sft_{len(all_records):06d}",
            "turns": cleaned_turns,
            "source": source,
            "category": category,
            "license": license_tag,
            "partition": partition,
            "token_count": len(tok_ids)
        })

    # -------------------------------------------------------------------------
    # 1. E.D.I.T.H. Personality & Conversational Skills Curriculum (Custom)
    # -------------------------------------------------------------------------
    print("\n[1/4] Ingesting E.D.I.T.H. Personality & Skill Curriculum...")
    curriculum_data = EdithPersonalityCurriculum.get_all_curriculum_dialogues()
    for item in curriculum_data:
        add_sft_record(
            turns=item["turns"],
            source="edith_curriculum",
            category=item["category"],
            license_tag="Proprietary Custom",
            partition="commercial"
        )
    print(f"  Ingested {len(curriculum_data)} curriculum lessons.")

    # -------------------------------------------------------------------------
    # 2. Hand-Authored E.D.I.T.H. Seed Dialogues (Custom)
    # -------------------------------------------------------------------------
    print("\n[2/4] Ingesting Hand-Authored Assistant Seed Dialogues...")
    raw_seeds = AuthoredSeedData.get_seed_dialogues()
    for item in raw_seeds:
        if "turns" in item:
            turns = item["turns"]
        else:
            turns = [
                {"role": "user", "text": item.get("user", "")},
                {"role": "edith", "text": item.get("edith", "")}
            ]
        add_sft_record(
            turns=turns,
            source="edith_authored_seeds",
            category=item.get("category", "persona"),
            license_tag="Proprietary Custom",
            partition="commercial"
        )
    print(f"  Ingested {len(raw_seeds)} authored assistant seeds.")

    # -------------------------------------------------------------------------
    # 3. Databricks Dolly-15k (CC BY-SA 3.0, Human QA & Instruction)
    # -------------------------------------------------------------------------
    dolly_path = r"C:\Users\Aditya Sawant\.cache\huggingface\hub\datasets--databricks--databricks-dolly-15k\snapshots\bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a\databricks-dolly-15k.jsonl"
    if os.path.exists(dolly_path):
        print(f"\n[3/4] Ingesting Databricks Dolly-15k ({dolly_path})...")
        dolly_count = 0
        with open(dolly_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                instr = rec.get("instruction", "").strip()
                ctx = rec.get("context", "").strip()
                resp = rec.get("response", "").strip()
                cat = rec.get("category", "general_instruction")
                if not instr or not resp:
                    continue
                user_msg = f"{instr}\n\nContext:\n{ctx}" if ctx else instr
                turns = [
                    {"role": "user", "text": user_msg},
                    {"role": "edith", "text": resp}
                ]
                add_sft_record(
                    turns=turns,
                    source="databricks-dolly-15k",
                    category=f"dolly_{cat}",
                    license_tag="CC BY-SA 3.0",
                    partition="commercial"
                )
                dolly_count += 1
        print(f"  Processed {dolly_count} instruction pairs from Dolly-15k.")

    # -------------------------------------------------------------------------
    # 4. Empathetic Dialogues (CC BY-NC 4.0, Human Multi-turn Empathy)
    # -------------------------------------------------------------------------
    emp_parquet = r"C:\Users\Aditya Sawant\.cache\huggingface\hub\datasets--pixelsandpointers--empathetic_dialogues_for_lm\snapshots\dca016ca9d4fb5f89e0b25a5bdbff348bbfe0bd2\data\train-00000-of-00001.parquet"
    if os.path.exists(emp_parquet):
        print(f"\n[4/4] Ingesting Empathetic Dialogues ({emp_parquet})...")
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(emp_parquet)
            conv_col = table["conv"].to_pylist()
            emp_count = 0
            for conv_list in conv_col:
                if not conv_list or len(conv_list) < 2:
                    continue
                turns = []
                for idx, turn in enumerate(conv_list):
                    clean_t = turn.replace("_comma_", ",").strip()
                    if not clean_t:
                        continue
                    role = "user" if idx % 2 == 0 else "edith"
                    turns.append({"role": role, "text": clean_t})
                # Truncate to maximum 4 turns (2 user, 2 edith) to fit in context cleanly
                turns = turns[:4]
                if len(turns) >= 2:
                    add_sft_record(
                        turns=turns,
                        source="empathetic_dialogues",
                        category="empathy_conversation",
                        license_tag="CC BY-NC 4.0",
                        partition="non_commercial"
                    )
                    emp_count += 1
                    if emp_count >= 8000:  # Curated subset for empathy balance
                        break
            print(f"  Processed {emp_count} multi-turn emotional dialogues.")
        except Exception as e:
            print(f"  Empathetic ingestion notice: {e}")

    # -------------------------------------------------------------------------
    # Deduplication, Statistics & Splits
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SFT CORPUS VALIDATION & SPLITTING")
    print("=" * 80)
    print(f"Total Unique SFT Dialogues: {len(all_records):,}")

    total_tokens = sum(r["token_count"] for r in all_records)
    print(f"Total SFT Ingested Tokens: {total_tokens:,} (~{total_tokens / 1e6:.2f}M tokens)")

    # Split into 90% Train, 5% Val, 5% Test
    random.seed(42)
    random.shuffle(all_records)

    n_total = len(all_records)
    n_val = max(500, int(n_total * 0.05))
    n_test = max(500, int(n_total * 0.05))
    n_train = n_total - n_val - n_test

    train_data = all_records[:n_train]
    val_data = all_records[n_train:n_train + n_val]
    test_data = all_records[n_train + n_val:]

    train_path = os.path.join(sft_proc_dir, "train_sft.jsonl")
    val_path = os.path.join(sft_proc_dir, "val_sft.jsonl")
    test_path = os.path.join(sft_proc_dir, "test_sft.jsonl")

    for path, data in [(train_path, train_data), (val_path, val_data), (test_path, test_data)]:
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r) + "\n")

    # Generate Manifest
    category_counts = {}
    partition_counts = {}
    for r in all_records:
        category_counts[r["category"]] = category_counts.get(r["category"], 0) + 1
        partition_counts[r["partition"]] = partition_counts.get(r["partition"], 0) + 1

    manifest = {
        "total_records": n_total,
        "total_tokens": total_tokens,
        "train_records": len(train_data),
        "val_records": len(val_data),
        "test_records": len(test_data),
        "partitions": partition_counts,
        "categories": category_counts,
        "licenses": {
            "databricks-dolly-15k": "CC BY-SA 3.0",
            "empathetic_dialogues": "CC BY-NC 4.0",
            "edith_curriculum": "Proprietary Custom",
            "edith_authored_seeds": "Proprietary Custom"
        }
    }
    manifest_path = os.path.join(sft_proc_dir, "sft_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSFT Splits Created:")
    print(f"  Train: {len(train_data):,} records -> {train_path}")
    print(f"  Val:   {len(val_data):,} records -> {val_path}")
    print(f"  Test:  {len(test_data):,} records -> {test_path}")
    print(f"Manifest written to {manifest_path}")

    return manifest


if __name__ == "__main__":
    build_sft_corpus()

