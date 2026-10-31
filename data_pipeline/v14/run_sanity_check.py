"""
E.D.I.T.H. V14 50-Step Sanity Training Run & Generation Journal Verification.
Trains for 50 steps on the real tokenized memory-mapped shards, logs metrics,
and writes generation probe completions to runs/<run_id>/samples.md.
"""

import os
import sys
import time
import math
import torch
import torch.nn as nn
from datetime import datetime
from typing import Dict, Any, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.edith_lm.model import EdithLM, EdithLMConfig
from engine.edith_lm.tokenizer import EdithByteBPETokenizer
from data_pipeline.v14.shard_tokenizer import ShardDataset
from data_pipeline.v14.generation_journal import GenerationJournal


def run_sanity_training(
    run_id: Optional[str] = None,
    total_steps: int = 50,
    batch_size: int = 4,
    context_len: int = 256,
    lr: float = 3e-4
) -> Dict[str, Any]:
    run_id = run_id or f"v14_sanity_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"[SanityTrain] Starting 50-step run `{run_id}`...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SanityTrain] Compute device: {device}")

    # 1. Load Tokenizer
    tok_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_bpe_tokenizer.json")
    tokenizer = EdithByteBPETokenizer.from_file(tok_path)
    vocab_size = len(tokenizer)
    print(f"[SanityTrain] Loaded tokenizer with vocab size {vocab_size}.")

    # 2. Locate Shards
    shard_dir = os.path.join(PROJECT_ROOT, "data_pipeline", "v14", "shards")
    train_bin = os.path.join(shard_dir, "train_shard_000.bin")
    val_bin = os.path.join(shard_dir, "val_shard_000.bin")

    if not os.path.exists(train_bin):
        raise FileNotFoundError(f"Missing train shard at {train_bin}. Run shard_tokenizer.py first!")

    train_ds = ShardDataset(train_bin, context_len=context_len)
    val_ds = ShardDataset(val_bin, context_len=context_len) if os.path.exists(val_bin) else train_ds
    print(f"[SanityTrain] Shard loaded: {train_ds.total_tokens:,} tokens ({len(train_ds)} chunks).")

    # 3. Model Configuration (~48M params: dim=576, 10 layers, 6 heads, intermediate_dim=1536)
    config = EdithLMConfig(
        vocab_size=vocab_size,
        dim=576,
        n_layers=10,
        n_heads=6,
        n_kv_heads=2,
        intermediate_dim=1536,
        max_seq_len=context_len,
        rope_theta=10000.0,
        norm_eps=1e-5
    )
    model = EdithLM(config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[SanityTrain] Model initialized: {param_count:,} parameters ({param_count / 1e6:.2f}M).")

    # 4. Optimizer & Journal Setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    journal = GenerationJournal(run_id=run_id)

    step_losses = []
    t_start = time.time()

    # Step 0 initial generation journal entry
    print("[SanityTrain] Evaluating probe prompts at step 0...")
    with torch.no_grad():
        x_val, y_val = val_ds.get_batch(batch_size=batch_size)
        x_val_t = torch.tensor(x_val, dtype=torch.long, device=device)
        y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)
        _, init_loss, _ = model(x_val_t, targets=y_val_t)
        init_val_loss = init_loss.item()
        init_val_ppl = math.exp(min(init_val_loss, 20.0))

    journal.record_step(
        step=0,
        model=model,
        tokenizer=tokenizer,
        train_loss=init_val_loss,
        val_loss=init_val_loss,
        val_ppl=init_val_ppl
    )

    # 5. Execute 50 Training Steps
    print("[SanityTrain] Running 50 optimization steps...")
    for step in range(1, total_steps + 1):
        x, y = train_ds.get_batch(batch_size=batch_size)
        x_t = torch.tensor(x, dtype=torch.long, device=device)
        y_t = torch.tensor(y, dtype=torch.long, device=device)

        # Cosine learning rate with linear warmup
        warmup_steps = 10
        if step <= warmup_steps:
            cur_lr = lr * (step / warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cur_lr = lr * 0.5 * (1.0 + math.cos(math.pi * progress))

        for param_group in optimizer.param_groups:
            param_group["lr"] = cur_lr

        optimizer.zero_grad()
        _, loss, _ = model(x_t, targets=y_t)

        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError(f"Numerical instability detected at step {step}: loss is {loss.item()}!")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_val = loss.item()
        step_losses.append(loss_val)

        if step % 10 == 0 or step == total_steps:
            elapsed = time.time() - t_start
            print(f"  Step {step:02d}/{total_steps} | Loss: {loss_val:.4f} | LR: {cur_lr:.2e} | Elapsed: {elapsed:.1f}s")

        # Record probe generation at midpoint (step 25) and finish (step 50)
        if step in [25, total_steps]:
            with torch.no_grad():
                x_v, y_v = val_ds.get_batch(batch_size=batch_size)
                x_vt = torch.tensor(x_v, dtype=torch.long, device=device)
                y_vt = torch.tensor(y_v, dtype=torch.long, device=device)
                _, v_loss_t, _ = model(x_vt, targets=y_vt)
                v_loss = v_loss_t.item()
                v_ppl = math.exp(min(v_loss, 20.0))

            journal.record_step(
                step=step,
                model=model,
                tokenizer=tokenizer,
                train_loss=loss_val,
                val_loss=v_loss,
                val_ppl=v_ppl
            )

    # 6. Save Sanity Checkpoint
    ckpt_dir = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_lm_v14")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "edith_lm_v14_sanity.pt")

    torch.save({
        "step": total_steps,
        "config": config,
        "model_state": model.state_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": step_losses[-1],
        "val_loss": v_loss,
        "val_ppl": v_ppl,
        "run_id": run_id
    }, ckpt_path)
    print(f"[SanityTrain] Checkpoint saved to: {ckpt_path}")

    # Verify loss decreased
    first_window = sum(step_losses[:5]) / 5
    last_window = sum(step_losses[-5:]) / 5
    print(f"[SanityTrain] Initial loss window: {first_window:.4f} -> Final loss window: {last_window:.4f}")
    assert last_window < first_window, f"Sanity failure: loss did not decrease ({first_window:.4f} -> {last_window:.4f})"

    return {
        "run_id": run_id,
        "status": "SUCCESS",
        "total_steps": total_steps,
        "initial_loss": first_window,
        "final_loss": last_window,
        "val_loss": v_loss,
        "val_ppl": v_ppl,
        "samples_path": journal.samples_file,
        "checkpoint_path": ckpt_path
    }


if __name__ == "__main__":
    from typing import Optional
    res = run_sanity_training()
    print("Sanity Run Results:", res)
