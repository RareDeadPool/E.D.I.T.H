"""
E.D.I.T.H. V15 25M-Token Proof-of-Concept Training Pipeline
Model: EdithLM 55.90M Medium Class (12 layers, dim 576, 16k vocab, tied embeddings)
Dataset: 100% Audited Human Corpus (data_pipeline/v15/processed/train.jsonl)
Tokenizer: New 16,384-Vocab Byte-Level BPE (edith_bpe_tokenizer_16k.json)
Hardware: NVIDIA GeForce RTX 4050 6GB Laptop GPU (cu124)
"""

import os
import sys
import time
import json
import math
from typing import List, Dict, Any, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from engine.edith_lm.model import EdithLM, EdithLMConfig
from engine.edith_lm.tokenizer import EdithByteBPETokenizer
from engine.fluency_guard import FluencyGuard


class PackedJsonlDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer: EdithByteBPETokenizer, max_seq_len: int = 512):
        cache_path = jsonl_path.replace(".jsonl", f"_cache_tensor_{max_seq_len}.pt")
        if os.path.exists(cache_path):
            print(f"Loading pre-tokenized tensor cache from {cache_path}...")
            self.tensor = torch.load(cache_path, weights_only=True)
            return

        print(f"Tokenizing and caching {jsonl_path}...")
        samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                text = record.get("text", "")
                if not text:
                    continue
                ids = tokenizer.encode(text, add_bos=True, add_eos=True)
                if len(ids) > max_seq_len:
                    ids = ids[:max_seq_len]
                elif len(ids) < max_seq_len:
                    ids = ids + [tokenizer.pad_id] * (max_seq_len - len(ids))
                samples.append(torch.tensor(ids, dtype=torch.int32))
        self.tensor = torch.stack(samples)
        torch.save(self.tensor, cache_path)
        print(f"Saved {len(self.tensor):,} tokenized tensor rows ({self.tensor.numel() * 4 / (1024*1024):.1f} MB) to {cache_path}")

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        return self.tensor[idx].long()


def evaluate_val_loss(model: EdithLM, val_loader: DataLoader, device: torch.device, pad_id: int = 0) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            x = batch[:, :-1]
            y = batch[:, 1:].clone()
            y[y == pad_id] = -100
            n_valid = (y != -100).sum().item()
            if n_valid == 0:
                continue
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, loss, _ = model(x, targets=y)
            total_loss += loss.item() * n_valid
            total_tokens += n_valid

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(20.0, avg_loss))
    model.train()
    return avg_loss, ppl


def sample_generation(model: EdithLM, tokenizer: EdithByteBPETokenizer, prompt: str, device: torch.device, max_new_tokens: int = 40) -> str:
    model.eval()
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    generated = list(prompt_ids)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            curr_input = torch.tensor([generated[-512:]], dtype=torch.long, device=device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, _, _ = model(curr_input)
            next_token_logits = logits[0, -1, :] / 0.7
            topk_logits, topk_indices = torch.topk(next_token_logits, k=40)
            probs = F.softmax(topk_logits, dim=-1)
            next_token_idx = torch.multinomial(probs, num_samples=1)
            next_token = topk_indices[next_token_idx].item()

            if next_token == tokenizer.eos_id:
                break
            generated.append(next_token)

    model.train()
    full_text = tokenizer.decode(generated)
    if full_text.startswith("<bos>"):
        full_text = full_text[5:]
    if full_text.startswith(prompt):
        return full_text[len(prompt):].strip()
    return full_text.strip()


def run_training_25m():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    print("=" * 80)
    print("E.D.I.T.H. V15 25M-TOKEN PROOF-OF-CONCEPT TRAINING PIPELINE")
    print("Device: NVIDIA GeForce RTX 4050 6GB Laptop GPU")
    print("Architecture: 55.90M Medium Class (12 layers, dim 576, 16k vocab)")
    print("=" * 80)

    device = torch.device("cuda:0")
    total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)

    # 1. Load 16k Tokenizer
    tok_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_bpe_tokenizer_16k.json")
    tokenizer = EdithByteBPETokenizer(vocab_file=tok_path)
    print(f"Loaded Tokenizer: {tokenizer.vocab_size} vocab entries from {tok_path}")

    # 2. Instantiate 55.90M Model
    cfg = EdithLMConfig(
        vocab_size=tokenizer.vocab_size,
        dim=576,
        n_layers=12,
        n_heads=6,
        n_kv_heads=2,
        intermediate_dim=1728,
        max_seq_len=512,
        tie_word_embeddings=True
    )
    model = EdithLM(cfg).to(device)
    actual_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {actual_params / 1e6:.2f}M actual parameters (tied)")

    # 3. Load Datasets
    train_path = os.path.join(PROJECT_ROOT, "data_pipeline", "v15", "processed", "train.jsonl")
    val_path = os.path.join(PROJECT_ROOT, "data_pipeline", "v15", "processed", "val.jsonl")

    train_ds = PackedJsonlDataset(train_path, tokenizer, max_seq_len=512)
    val_ds = PackedJsonlDataset(val_path, tokenizer, max_seq_len=512)

    batch_size = 8
    grad_accum_steps = 4
    tokens_per_step = batch_size * 512 * grad_accum_steps  # 16,384 tokens/step

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    total_target_tokens = 25_000_000
    n_train_steps = int(total_target_tokens / tokens_per_step)
    if "MAX_STEPS" in os.environ:
        n_train_steps = int(os.environ["MAX_STEPS"])

    print(f"Training Dataset: {len(train_ds):,} samples | Validation: {len(val_ds):,} samples")
    print(f"Tokens/Step: {tokens_per_step:,} | Total Steps: {n_train_steps:,} -> Target: {total_target_tokens:,} tokens")

    # 4. Optimizer & Cosine Schedule
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda")

    warmup_steps = 150
    min_lr = 6e-5

    def get_lr(step: int) -> float:
        if step < warmup_steps:
            return 6e-4 * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, n_train_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (6e-4 - min_lr) * cosine_decay

    # 5. Fixed Probe Prompts for Fluency Journaling
    probe_prompts = [
        "<mood>calm_focused</mood><user>describe me</user><edith>",
        "<mood>calm_focused</mood><user>What do you think about me?</user><edith>",
        "<mood>calm_focused</mood><user>Really?</user><edith>",
        "<mood>calm_focused</mood><user>How are you doing today?</user><edith>",
        "<mood>calm_focused</mood><user>bhai aaj bohot bekar din tha, boss ne daant diya</user><edith>",
        "<mood>calm_focused</mood><user>Tell me something about yourself.</user><edith>",
        "<mood>calm_focused</mood><user>What are you working on right now?</user><edith>",
    ]

    journal_path = os.path.join(PROJECT_ROOT, "data_pipeline", "v15", "generation_journal.jsonl")
    ckpt_dir = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_lm_v15")
    os.makedirs(ckpt_dir, exist_ok=True)

    print("\nInitial Validation Baseline...")
    val_loss_init, val_ppl_init = evaluate_val_loss(model, val_loader, device, pad_id=tokenizer.pad_id)
    print(f"Initial Val Loss: {val_loss_init:.4f} | Initial Perplexity: {val_ppl_init:.2f}")

    best_ckpt_path = os.path.join(ckpt_dir, "edith_lm_v15_poc_25m_best.pt")
    best_val_loss = float("inf")
    tokens_seen = 0
    t0 = time.perf_counter()

    data_iter = iter(train_loader)
    print("\n" + "=" * 80)
    print(f"LAUNCHING 25M-TOKEN TRAINING RUN ON RTX 4050 ({n_train_steps} steps)...")
    print("=" * 80)

    for step in range(1, n_train_steps + 1):
        lr_now = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_now

        step_loss = 0.0
        for _ in range(grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            batch = batch.to(device)
            x = batch[:, :-1]
            y = batch[:, 1:].clone()
            y[y == tokenizer.pad_id] = -100

            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, loss, _ = model(x, targets=y)
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()
            step_loss += loss.item() * grad_accum_steps
            tokens_seen += (batch != tokenizer.pad_id).sum().item()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Logging every 50 steps
        if step % 50 == 0 or step == 1:
            elapsed = time.perf_counter() - t0
            tok_s = tokens_seen / elapsed
            peak_vram = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            print(f"Step {step:4d}/{n_train_steps} | TOKENS SEEN: {tokens_seen:>10,d} | Loss: {step_loss:6.4f} | LR: {lr_now:.2e} | Grad: {grad_norm:.2f} | Tok/s: {tok_s:7.1f} | VRAM: {peak_vram:5.0f}MB")

        # Evaluation & Generation Journal every 300 steps (~5M tokens)
        if step % 300 == 0 or step == n_train_steps:
            val_loss, val_ppl = evaluate_val_loss(model, val_loader, device, pad_id=tokenizer.pad_id)
            print(f"\n>>> [EVALUATION @ {tokens_seen:,} TOKENS] Val Loss: {val_loss:.4f} | Perplexity: {val_ppl:.2f}")

            # Sample generations
            journal_entries = []
            for p in probe_prompts:
                gen = sample_generation(model, tokenizer, p, device)
                is_fluent, diag = FluencyGuard.inspect(gen)
                journal_entries.append({
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "prompt": p,
                    "generation": gen,
                    "word_validity": diag.get("word_validity_ratio", 0.0),
                    "is_fluent": is_fluent
                })
                print(f"    PROMPT: \"{p}\"")
                print(f"    OUTPUT: \"{gen}\"")
                print(f"    VALIDITY: {diag.get('word_validity_ratio', 0.0):.1%} | FLUENT: {is_fluent}\n")

            with open(journal_path, "a", encoding="utf-8") as f:
                for entry in journal_entries:
                    f.write(json.dumps(entry) + "\n")

            # Checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt_path = os.path.join(ckpt_dir, "edith_lm_v15_poc_25m_best.pt")
                torch.save({
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                    "config": cfg
                }, best_ckpt_path)
                print(f"    * Saved best checkpoint: {best_ckpt_path} (Val Loss: {val_loss:.4f})")

    # Save latest checkpoint
    latest_ckpt_path = os.path.join(ckpt_dir, "edith_lm_v15_poc_25m_latest.pt")
    torch.save({
        "step": n_train_steps,
        "tokens_seen": tokens_seen,
        "model_state_dict": model.state_dict(),
        "config": cfg
    }, latest_ckpt_path)

    total_time = time.perf_counter() - t0
    peak_vram_final = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    print("\n" + "=" * 80)
    print("25M-TOKEN TRAINING COMPLETED:")
    print(f"  Total Tokens Seen:    {tokens_seen:,}")
    print(f"  Total Time Elapsed:   {total_time:.2f} s ({total_time / 60:.2f} mins)")
    print(f"  Average Throughput:   {tokens_seen / total_time:.1f} tokens/second")
    print(f"  Peak VRAM Used:       {peak_vram_final:.1f} MB ({peak_vram_final / total_vram_mb:.1%} of 6GB)")
    print(f"  Initial PPL -> Final PPL: {val_ppl_init:.2f} -> {math.exp(best_val_loss):.2f}")
    print(f"  Checkpoints:          {best_ckpt_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_training_25m()
