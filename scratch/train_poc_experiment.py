"""
E.D.I.T.H. V15 Proof-of-Concept (POC) Experimental Training Pipeline
Runs a real CUDA training loop on the RTX 4050 Laptop GPU to verify:
1. Loss convergence and numerical stability with FP16 AMP and AdamW
2. Real-world training throughput and peak VRAM under continuous training
3. Held-out validation loss and perplexity evaluation
4. Raw generation sampling on 5 probe prompts to observe linguistic structure emergence
"""

import os
import sys
import time
import json
import math
from typing import List, Dict, Any, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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
        self.samples = []
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
                self.samples.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


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
            # count valid tokens
            valid_mask = (y != 0)
            n_valid = valid_mask.sum().item()
            if n_valid > 0:
                total_loss += loss.item() * n_valid
                total_tokens += n_valid

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(20.0, avg_loss))
    model.train()
    return avg_loss, ppl


def sample_generation(model: EdithLM, tokenizer: EdithByteBPETokenizer, prompt: str, device: torch.device, max_new_tokens: int = 35) -> str:
    model.eval()
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    generated = list(prompt_ids)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            curr_input = torch.tensor([generated[-512:]], dtype=torch.long, device=device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, _, _ = model(curr_input)
            next_token_logits = logits[0, -1, :] / 0.7
            # Top-k sampling (k=40)
            topk_logits, topk_indices = torch.topk(next_token_logits, k=40)
            probs = F.softmax(topk_logits, dim=-1)
            next_token_idx = torch.multinomial(probs, num_samples=1)
            next_token = topk_indices[next_token_idx].item()

            if next_token == tokenizer.eos_id:
                break
            generated.append(next_token)

    model.train()
    full_text = tokenizer.decode(generated)
    # Return continuation after prompt
    if full_text.startswith(prompt):
        return full_text[len(prompt):].strip()
    return full_text


def run_poc():
    print("=" * 80)
    print("E.D.I.T.H. V15 PROOF-OF-CONCEPT TRAINING EXPERIMENT")
    print("Device: NVIDIA GeForce RTX 4050 6GB Laptop GPU")
    print("Environment: Isolated .venv_train (PyTorch CUDA cu124)")
    print("=" * 80)

    device = torch.device("cuda:0")
    total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)

    # 1. Load Tokenizer
    tok_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_bpe_tokenizer.json")
    tokenizer = EdithByteBPETokenizer(vocab_file=tok_path)
    print(f"Loaded tokenizer with vocab size: {tokenizer.vocab_size}")

    # 2. Instantiate Model Config (37M Baseline Class: 10 layers, dim 512, tied embeddings)
    cfg = EdithLMConfig(
        vocab_size=tokenizer.vocab_size,
        dim=512,
        n_layers=10,
        n_heads=8,
        n_kv_heads=2,
        intermediate_dim=1536,
        max_seq_len=512,
        tie_word_embeddings=True
    )
    model = EdithLM(cfg).to(device)
    actual_params = sum(p.numel() for p in model.parameters())
    print(f"Model instantiated: {actual_params / 1e6:.2f}M actual parameters (tied)")

    # 3. Load Datasets
    train_path = os.path.join(PROJECT_ROOT, "data_pipeline", "v14", "processed", "train.jsonl")
    val_path = os.path.join(PROJECT_ROOT, "data_pipeline", "v14", "processed", "val.jsonl")

    train_ds = PackedJsonlDataset(train_path, tokenizer, max_seq_len=512)
    val_ds = PackedJsonlDataset(val_path, tokenizer, max_seq_len=512)

    batch_size = 8
    grad_accum_steps = 4
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Loaded {len(train_ds):,} training samples, {len(val_ds):,} validation samples.")
    print(f"Micro-batch: {batch_size}, Grad Accum: {grad_accum_steps} -> Effective batch: {batch_size * 512 * grad_accum_steps:,} tokens/step")

    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda")

    # 4. Measure Initial Un-trained Baseline
    print("\nEvaluating initial random initialization baseline...")
    val_loss_init, val_ppl_init = evaluate_val_loss(model, val_loader, device)
    print(f"Initial Validation Loss: {val_loss_init:.4f} | Initial Perplexity: {val_ppl_init:.2f}")

    probe_prompts = [
        "<mood>calm_focused</mood><user>How are you?</user><edith>",
        "<mood>calm_focused</mood><user>What are you working on?</user><edith>",
        "<mood>calm_focused</mood><user>Tell me something interesting.</user><edith>",
    ]

    print("\nInitial Raw Model Generation (Random Weights):")
    for p in probe_prompts:
        gen = sample_generation(model, tokenizer, p, device)
        print(f"  Prompt: \"{p}\"\n  Output: \"{gen}\"")

    # 5. Execute 200 Training Steps (POC Run)
    n_train_steps = 200
    print(f"\nLaunching {n_train_steps} POC training steps on RTX 4050...")
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()

    data_iter = iter(train_loader)
    tokens_seen = 0
    step_losses = []

    for step in range(1, n_train_steps + 1):
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
            tokens_seen += batch.numel()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        step_losses.append(step_loss)

        if step % 25 == 0 or step == 1:
            elapsed = time.perf_counter() - t0
            tok_s = tokens_seen / elapsed
            print(f"Step {step:3d}/{n_train_steps} | Tokens Seen: {tokens_seen:>8,d} | Loss: {step_loss:6.4f} | Grad Norm: {grad_norm:.2f} | Throughput: {tok_s:8.1f} tok/s")

    train_time = time.perf_counter() - t0
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    # 6. Final Evaluation
    val_loss_final, val_ppl_final = evaluate_val_loss(model, val_loader, device)
    print("\n" + "=" * 80)
    print("POC TRAINING EXPERIMENT COMPLETED:")
    print(f"  Total Tokens Processed:      {tokens_seen:,}")
    print(f"  Total Elapsed Time:          {train_time:.2f} s ({train_time / 60:.2f} mins)")
    print(f"  Average Training Throughput: {tokens_seen / train_time:.1f} tokens/second")
    print(f"  Peak VRAM Used:              {peak_vram_mb:.1f} MB ({peak_vram_mb / total_vram_mb:.1%} of 6GB)")
    print(f"  Initial Val Loss / PPL:      {val_loss_init:.4f} / {val_ppl_init:.2f}")
    print(f"  Final Val Loss / PPL:        {val_loss_final:.4f} / {val_ppl_final:.2f}")
    print(f"  Perplexity Reduction:        {(val_ppl_init - val_ppl_final) / val_ppl_init:.1%}")
    print("=" * 80)

    print("\nPost-POC Model Raw Generations (After 3.3M Tokens):")
    for p in probe_prompts:
        gen = sample_generation(model, tokenizer, p, device)
        is_fluent, diag = FluencyGuard.inspect(gen)
        print(f"  Prompt: \"{p}\"")
        print(f"  Output: \"{gen}\"")
        print(f"  Word Validity: {diag.get('word_validity_ratio', 0.0):.1%} | Fluent: {is_fluent}")
        print("-" * 60)


if __name__ == "__main__":
    run_poc()
