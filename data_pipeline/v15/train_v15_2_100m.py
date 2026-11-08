"""
E.D.I.T.H. V15.2 100M Valid-Token Pretraining Pipeline
Optimized Sequence Packing (0.0% padding compute waste)
Model: EdithLM 55.90M Medium Class (12 layers, dim 576, 16k vocab, tied embeddings)
Dataset: 100% Organic Expanded Human Corpus (15.59M unique tokens)
Hardware: NVIDIA GeForce RTX 4050 6GB Laptop GPU (cu124)
Target: 100,000,000 Valid Next-Token Prediction Targets
"""

import os
import sys
import time
import json
import math
import argparse
from typing import List, Dict, Any, Tuple, Optional

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


class PackedStreamDataset(Dataset):
    """
    High-Throughput Sequence Packing Dataset.
    Packs multiple documents separated by <bos> and <eos> into contiguous 513-token chunks.
    Eliminates 100% of padding compute waste.
    """
    def __init__(self, jsonl_path: str, tokenizer: EdithByteBPETokenizer, seq_len: int = 512):
        self.chunk_len = seq_len + 1  # 513 tokens -> 512 for x, 512 for y
        cache_path = jsonl_path.replace(".jsonl", f"_stream_cache_{seq_len}.pt")

        if os.path.exists(cache_path):
            print(f"Loading packed stream cache from {cache_path}...")
            self.tensor = torch.load(cache_path, weights_only=True)
            print(f"Loaded {len(self.tensor):,} packed sequence rows ({self.tensor.numel() * 4 / (1024*1024):.1f} MB)")
            return

        print(f"Packing and caching stream from {jsonl_path}...")
        all_ids = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                text = record.get("text", "")
                if not text:
                    continue
                ids = tokenizer.encode(text, add_bos=True, add_eos=True)
                all_ids.extend(ids)

        flat_tensor = torch.tensor(all_ids, dtype=torch.int32)
        n_chunks = len(flat_tensor) // self.chunk_len
        self.tensor = flat_tensor[:n_chunks * self.chunk_len].reshape(n_chunks, self.chunk_len)
        torch.save(self.tensor, cache_path)
        print(f"Saved {len(self.tensor):,} packed sequence rows ({self.tensor.numel() * 4 / (1024*1024):.1f} MB) to {cache_path}")

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        return self.tensor[idx].long()


def evaluate_val_loss(model: EdithLM, val_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            x = batch[:, :-1]
            y = batch[:, 1:]
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, loss, _ = model(x, targets=y)
            n_tokens = y.numel()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(20.0, avg_loss))
    model.train()
    return avg_loss, ppl


def sample_generation(model: EdithLM, tokenizer: EdithByteBPETokenizer, prompt: str, device: torch.device, max_new_tokens: int = 50) -> str:
    model.eval()
    if not prompt.startswith("<mood>"):
        formatted = f"<mood>calm_focused</mood><user>{prompt}</user><edith>"
    else:
        formatted = prompt

    prompt_ids = tokenizer.encode(formatted, add_bos=True, add_eos=False)
    generated = list(prompt_ids)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            curr_input = torch.tensor([generated[-512:]], dtype=torch.long, device=device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, _, _ = model(curr_input)
            
            # Apply recommended decoding config from Phase 5 (temp=0.7, top_p=0.85, rep_pen=1.20)
            next_logits = logits[0, -1, :].clone()
            for token_id in set(generated[len(prompt_ids):]):
                if next_logits[token_id] > 0:
                    next_logits[token_id] /= 1.20
                else:
                    next_logits[token_id] *= 1.20

            next_logits = next_logits / 0.70
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > 0.85
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_logits[indices_to_remove] = -float("Inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            if next_token == tokenizer.eos_id:
                break
            generated.append(next_token)

    model.train()
    full_text = tokenizer.decode(generated)
    if full_text.startswith("<bos>"):
        full_text = full_text[5:]
    if full_text.startswith(formatted):
        return full_text[len(formatted):].strip()
    return full_text.strip()


def run_100m_training(resume_path: Optional[str] = None):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    print("=" * 80)
    print("E.D.I.T.H. V15.2: 100M VALID-TOKEN TRAINING PIPELINE")
    print("Device: NVIDIA GeForce RTX 4050 6GB Laptop GPU")
    print("Architecture: EdithLM 55.90M Medium Class (12 layers, dim 576, 16k vocab)")
    print("Pipeline: 100% Sequence Packed (Zero Padding Compute Waste)")
    print("=" * 80)

    device = torch.device("cuda:0")
    total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)

    # 1. Load Tokenizer
    tok_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_bpe_tokenizer_16k.json")
    tokenizer = EdithByteBPETokenizer(vocab_file=tok_path)
    print(f"Loaded Tokenizer: {tokenizer.vocab_size} vocab entries from {tok_path}")

    # 2. Instantiate Model
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

    # 3. Load Packed Datasets
    train_file = os.path.join(PROJECT_ROOT, "data_pipeline", "v15", "processed_expanded", "train_expanded.jsonl")
    val_file = os.path.join(PROJECT_ROOT, "data_pipeline", "v15", "processed_expanded", "val_expanded.jsonl")

    train_ds = PackedStreamDataset(train_file, tokenizer, seq_len=512)
    val_ds = PackedStreamDataset(val_file, tokenizer, seq_len=512)

    batch_size = 8
    grad_accum_steps = 4
    valid_targets_per_step = batch_size * 512 * grad_accum_steps  # exactly 16,384 valid next-token targets

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    target_valid_tokens = 100_000_000
    n_train_steps = int(target_valid_tokens / valid_targets_per_step)  # 6,103 steps
    if "MAX_STEPS" in os.environ:
        n_train_steps = int(os.environ["MAX_STEPS"])

    print(f"Training Packed Sequences: {len(train_ds):,} (512-token chunks)")
    print(f"Validation Packed Sequences: {len(val_ds):,} (512-token chunks)")
    print(f"Valid Targets per Step: {valid_targets_per_step:,}")
    print(f"Total Steps: {n_train_steps:,} -> Target: {target_valid_tokens:,} valid next-token targets")

    # 4. Optimizer & Cosine Schedule
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda")

    warmup_steps = 300
    min_lr = 5e-5

    def get_lr(step: int) -> float:
        if step < warmup_steps:
            return 5e-4 * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, n_train_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (5e-4 - min_lr) * cosine_decay

    # 5. Resuming Checkpoint (if provided)
    start_step = 1
    valid_targets_seen = 0
    if resume_path and os.path.exists(resume_path):
        print(f"\nResuming model weights from previous checkpoint: {resume_path}...")
        ckpt_data = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_data["model_state_dict"])
        print("  Model state dict successfully restored.")

    # 6. Evaluation Setup
    ckpt_dir = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_lm_v15")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_dir, "edith_lm_v15_2_100m_best.pt")
    latest_ckpt_path = os.path.join(ckpt_dir, "edith_lm_v15_2_100m_latest.pt")
    journal_path = os.path.join(PROJECT_ROOT, "data_pipeline", "v15", "generation_journal_v15_2.jsonl")

    probe_prompts = [
        "Hey Edith, how are you?",
        "describe me",
        "Really?",
        "Why would someone carry an umbrella when the sky is dark and cloudy?",
        "Explain recursion to someone who is learning programming for the first time.",
        "Tell me a genuinely funny programming joke.",
        "bhai aaj bohot bekar din tha, boss ne daant diya"
    ]

    print("\nRunning Baseline Evaluation...")
    val_loss_init, val_ppl_init = evaluate_val_loss(model, val_loader, device)
    print(f"Initial Val Loss: {val_loss_init:.4f} | Initial Perplexity: {val_ppl_init:.2f}")

    best_val_loss = float("inf")
    t0 = time.perf_counter()
    data_iter = iter(train_loader)

    print("\n" + "=" * 80)
    print(f"LAUNCHING 100M VALID-TARGET TRAINING RUN ({n_train_steps:,} steps)...")
    print("=" * 80)

    for step in range(start_step, n_train_steps + 1):
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
            y = batch[:, 1:]

            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, loss, _ = model(x, targets=y)
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()
            step_loss += loss.item() * grad_accum_steps
            valid_targets_seen += y.numel()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Logging every 50 steps
        if step % 50 == 0 or step == 1:
            elapsed = time.perf_counter() - t0
            tok_s = valid_targets_seen / max(0.001, elapsed)
            peak_vram = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            print(f"Step {step:4d}/{n_train_steps} | TARGETS: {valid_targets_seen:>10,d} | Loss: {step_loss:6.4f} | LR: {lr_now:.2e} | Grad: {grad_norm:.2f} | Tok/s: {tok_s:7.1f} | VRAM: {peak_vram:5.0f}MB")

        # Evaluation & Generation Journal every 600 steps (~10M tokens) or final step
        if step % 600 == 0 or step == n_train_steps:
            val_loss, val_ppl = evaluate_val_loss(model, val_loader, device)
            print(f"\n>>> [EVALUATION @ {valid_targets_seen:,} VALID TARGETS] Val Loss: {val_loss:.4f} | Perplexity: {val_ppl:.2f}")

            journal_entries = []
            for p in probe_prompts:
                gen = sample_generation(model, tokenizer, p, device)
                is_fluent, diag = FluencyGuard.inspect(gen, query=p)
                journal_entries.append({
                    "step": step,
                    "targets_seen": valid_targets_seen,
                    "prompt": p,
                    "generation": gen,
                    "word_validity": diag.get("word_validity_ratio", 0.0),
                    "trigram_score": diag.get("char_trigram_logprob", 0.0),
                    "is_fluent": is_fluent
                })
                print(f"    PROMPT: \"{p}\"")
                print(f"    OUTPUT: \"{gen}\"")
                print(f"    VALIDITY: {diag.get('word_validity_ratio', 0.0):.1%} | TRIGRAM: {diag.get('char_trigram_logprob', 0.0):.2f} | FLUENT: {is_fluent}\n")

            with open(journal_path, "a", encoding="utf-8") as f:
                for entry in journal_entries:
                    f.write(json.dumps(entry) + "\n")

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "step": step,
                    "valid_targets_seen": valid_targets_seen,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                    "config": cfg
                }, best_ckpt_path)
                print(f"    * Saved best checkpoint: {best_ckpt_path} (Val Loss: {val_loss:.4f})")

    # Save latest checkpoint
    torch.save({
        "step": n_train_steps,
        "valid_targets_seen": valid_targets_seen,
        "model_state_dict": model.state_dict(),
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        "config": cfg
    }, latest_ckpt_path)

    total_time = time.perf_counter() - t0
    peak_vram_final = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    print("\n" + "=" * 80)
    print("100M VALID-TARGET TRAINING COMPLETED:")
    print(f"  Total Valid Targets Seen: {valid_targets_seen:,}")
    print(f"  Total Time Elapsed:       {total_time:.2f} s ({total_time / 60:.2f} mins / {total_time / 3600:.2f} hrs)")
    print(f"  Average Throughput:       {valid_targets_seen / total_time:.1f} valid targets/second")
    print(f"  Peak VRAM Used:           {peak_vram_final:.1f} MB ({peak_vram_final / total_vram_mb:.1%} of 6GB)")
    print(f"  Initial PPL -> Final PPL: {val_ppl_init:.2f} -> {math.exp(best_val_loss):.2f}")
    print(f"  Best Checkpoint:          {best_ckpt_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume weights from")
    args = parser.parse_args()

    default_resume = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_lm_v15", "edith_lm_v15_poc_25m_best.pt")
    resume_target = args.resume if args.resume is not None else (default_resume if os.path.exists(default_resume) else None)

    run_100m_training(resume_path=resume_target)

