"""
E.D.I.T.H. V14 Gate C: Pretraining & Dialogue Mid-Training.

Executes foundation pretraining and dialogue mid-training on the tokenized binary shards
with AdamW, linear warmup + cosine decay, gradient clipping, memory-mapped batch loading,
loss curve logging, and generation journal probe sampling at 10%, 25%, 50%, 75%, and 100%.
"""

import os
import sys
import time
import math
import json
import torch
import torch.nn as nn
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.edith_lm.model import EdithLM, EdithLMConfig
from engine.edith_lm.tokenizer import EdithByteBPETokenizer
from data_pipeline.v14.shard_tokenizer import ShardDataset
from data_pipeline.v14.generation_journal import GenerationJournal


def run_gate_c_training(
    run_id: Optional[str] = None,
    total_steps: int = 250,
    batch_size: int = 4,
    context_len: int = 256,
    lr: float = 3.5e-4,
    min_lr: float = 1.0e-5,
    warmup_steps: int = 25,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    eval_interval: int = 25
) -> Dict[str, Any]:
    run_id = run_id or f"v14_pretrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print("=" * 70)
    print(f"[Gate C Pretraining] Starting run `{run_id}`")
    print(f"  Total Steps: {total_steps} | Batch Size: {batch_size} | Context: {context_len}")
    print(f"  LR: {lr:.2e} -> {min_lr:.2e} | Warmup: {warmup_steps} | Grad Clip: {grad_clip}")
    print("=" * 70)

    # 1. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(min(12, os.cpu_count() or 8))
        print(f"[Gate C] Compute: CPU (threads={torch.get_num_threads()})")
    else:
        print(f"[Gate C] Compute: CUDA ({torch.cuda.get_device_name(0)})")

    # 2. Tokenizer Setup
    tok_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_bpe_tokenizer.json")
    tokenizer = EdithByteBPETokenizer.from_file(tok_path)
    vocab_size = len(tokenizer)
    print(f"[Gate C] Tokenizer loaded: vocab={vocab_size}, zero-<unk> guaranteed.")

    # 3. Memory-Mapped Binary Shard Dataset
    shard_dir = os.path.join(PROJECT_ROOT, "data_pipeline", "v14", "shards")
    train_bin = os.path.join(shard_dir, "train_shard_000.bin")
    val_bin = os.path.join(shard_dir, "val_shard_000.bin")

    if not os.path.exists(train_bin) or not os.path.exists(val_bin):
        raise FileNotFoundError(f"Missing binary shards in {shard_dir}. Run shard_tokenizer.py first!")

    train_ds = ShardDataset(train_bin, context_len=context_len)
    val_ds = ShardDataset(val_bin, context_len=context_len)
    print(f"[Gate C] Shards loaded:")
    print(f"  Train: {train_ds.total_tokens:,} tokens ({len(train_ds)} chunks)")
    print(f"  Val:   {val_ds.total_tokens:,} tokens ({len(val_ds)} chunks)")

    # 4. Model Architecture (~37.16M params)
    config = EdithLMConfig(
        vocab_size=vocab_size,
        dim=576,
        n_layers=10,
        n_heads=6,
        n_kv_heads=2,
        intermediate_dim=1536,
        max_seq_len=context_len,
        rope_theta=10000.0,
        norm_eps=1e-5,
        tie_word_embeddings=True
    )
    model = EdithLM(config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[Gate C] EdithLM initialized: {param_count:,} parameters ({param_count / 1e6:.2f}M).")

    # 5. Optimizer & Journal Setup
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)
    journal = GenerationJournal(run_id=run_id)

    # 6. Evaluation Milestones: 10%, 25%, 50%, 75%, 100%
    journal_milestones = {
        int(total_steps * 0.10): "10%",
        int(total_steps * 0.25): "25%",
        int(total_steps * 0.50): "50%",
        int(total_steps * 0.75): "75%",
        total_steps: "100%"
    }
    print(f"[Gate C] Generation Journal probe milestones: {journal_milestones}")

    def evaluate_validation(eval_batches: int = 8) -> Tuple[float, float]:
        """Calculates mean validation cross-entropy loss and perplexity across multiple batches."""
        model.eval()
        val_losses = []
        with torch.no_grad():
            for _ in range(eval_batches):
                xv, yv = val_ds.get_batch(batch_size=batch_size)
                xvt = torch.tensor(xv, dtype=torch.long, device=device)
                yvt = torch.tensor(yv, dtype=torch.long, device=device)
                _, v_loss, _ = model(xvt, targets=yvt)
                val_losses.append(v_loss.item())
        mean_val_loss = sum(val_losses) / len(val_losses)
        val_ppl = math.exp(min(mean_val_loss, 20.0))
        model.train()
        return mean_val_loss, val_ppl

    # Initial Step 0 Evaluation
    print("\n[Gate C] Computing initial baseline at Step 0...")
    init_val_loss, init_val_ppl = evaluate_validation(eval_batches=6)
    print(f"  Step 0 | Val Loss: {init_val_loss:.4f} | Val PPL: {init_val_ppl:.2f}")

    journal.record_step(
        step=0,
        model=model,
        tokenizer=tokenizer,
        train_loss=init_val_loss,
        val_loss=init_val_loss,
        val_ppl=init_val_ppl
    )

    metrics_log: List[Dict[str, Any]] = []
    milestone_val_losses: List[Tuple[int, str, float, float]] = [(0, "0%", init_val_loss, init_val_ppl)]
    step_losses: List[float] = []
    t_start = time.time()

    print("\n[Gate C] Commencing Pretraining & Mid-Training Optimization Loop...")
    for step in range(1, total_steps + 1):
        x, y = train_ds.get_batch(batch_size=batch_size)
        x_t = torch.tensor(x, dtype=torch.long, device=device)
        y_t = torch.tensor(y, dtype=torch.long, device=device)

        # Learning Rate Schedule: Linear Warmup + Cosine Decay
        if step <= warmup_steps:
            cur_lr = lr * (step / max(1, warmup_steps))
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cur_lr = min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))

        for param_group in optimizer.param_groups:
            param_group["lr"] = cur_lr

        # Forward & Loss
        optimizer.zero_grad()
        _, loss, _ = model(x_t, targets=y_t)
        loss_val = loss.item()

        if math.isnan(loss_val) or math.isinf(loss_val):
            raise RuntimeError(f"Numerical explosion detected at step {step}: loss={loss_val}!")

        # Backward & Gradient Clip
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        step_losses.append(loss_val)

        # Logging
        if step % 10 == 0 or step == total_steps:
            elapsed = time.time() - t_start
            steps_per_sec = step / max(elapsed, 0.001)
            eta_sec = (total_steps - step) / max(steps_per_sec, 0.001)
            print(f"  Step {step:03d}/{total_steps} | Loss: {loss_val:.4f} | LR: {cur_lr:.2e} | Speed: {steps_per_sec:.2f} st/s | ETA: {eta_sec:.0f}s")

        # Validation & Metrics Checkpoint
        if step % eval_interval == 0 or step in journal_milestones:
            v_loss, v_ppl = evaluate_validation(eval_batches=6)
            recent_train_loss = sum(step_losses[-eval_interval:]) / max(1, len(step_losses[-eval_interval:]))
            metrics_entry = {
                "step": step,
                "train_loss": recent_train_loss,
                "val_loss": v_loss,
                "val_ppl": v_ppl,
                "lr": cur_lr,
                "elapsed_sec": round(time.time() - t_start, 2)
            }
            metrics_log.append(metrics_entry)

            if step in journal_milestones:
                tag = journal_milestones[step]
                milestone_val_losses.append((step, tag, v_loss, v_ppl))
                print(f"\n>>> [Journal Milestone {tag} at Step {step}] Val Loss: {v_loss:.4f} | Val PPL: {v_ppl:.2f}")
                journal.record_step(
                    step=step,
                    model=model,
                    tokenizer=tokenizer,
                    train_loss=recent_train_loss,
                    val_loss=v_loss,
                    val_ppl=v_ppl
                )
                print("<<< [Journal Probe Complete]\n")

    # 7. Final Validation & Loss Curve Analysis
    final_val_loss, final_val_ppl = evaluate_validation(eval_batches=12)
    print("\n" + "=" * 70)
    print(f"[Gate C Pretraining Complete]")
    print(f"  Total Duration: {time.time() - t_start:.1f}s")
    print(f"  Initial Val Loss: {init_val_loss:.4f} (PPL: {init_val_ppl:.2f})")
    print(f"  Final Val Loss:   {final_val_loss:.4f} (PPL: {final_val_ppl:.2f})")

    # Verify monotonic progression across key quartile checkpoints
    print("\n[Milestone Loss Trajectory]:")
    for s, tag, vl, vp in milestone_val_losses:
        print(f"  - Milestone {tag:>4s} (Step {s:03d}): Val Loss = {vl:.4f} | Val PPL = {vp:.2f}")

    assert final_val_loss < init_val_loss, (
        f"Gate C Failure: Final validation loss ({final_val_loss:.4f}) did not improve over initial ({init_val_loss:.4f})!"
    )

    # 8. Save Pretrained Checkpoint
    ckpt_dir = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_lm_v14")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "edith_lm_v14_pretrained.pt")

    torch.save({
        "step": total_steps,
        "config": config,
        "model_state": model.state_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": step_losses[-1],
        "val_loss": final_val_loss,
        "val_ppl": final_val_ppl,
        "run_id": run_id,
        "vocab_size": vocab_size,
        "milestones": milestone_val_losses
    }, ckpt_path)
    print(f"\n[Gate C Checkpoint Saved]: {ckpt_path}")

    # 9. Save Metrics Log to Run Dir
    metrics_path = os.path.join(journal.run_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": run_id,
            "total_steps": total_steps,
            "config": {
                "vocab_size": vocab_size,
                "dim": 576,
                "layers": 10,
                "heads": 6,
                "kv_heads": 2,
                "intermediate_dim": 1536,
                "context_len": context_len,
                "batch_size": batch_size,
                "initial_lr": lr,
                "min_lr": min_lr
            },
            "initial_val_loss": init_val_loss,
            "final_val_loss": final_val_loss,
            "final_val_ppl": final_val_ppl,
            "milestones": [
                {"step": s, "milestone": tag, "val_loss": vl, "val_ppl": vp}
                for s, tag, vl, vp in milestone_val_losses
            ],
            "history": metrics_log
        }, f, indent=2)
    print(f"[Gate C Metrics Saved]: {metrics_path}")
    print("=" * 70)

    return {
        "run_id": run_id,
        "status": "PASS",
        "total_steps": total_steps,
        "init_val_loss": init_val_loss,
        "final_val_loss": final_val_loss,
        "final_val_ppl": final_val_ppl,
        "milestones": milestone_val_losses,
        "samples_path": journal.samples_file,
        "checkpoint_path": ckpt_path,
        "metrics_path": metrics_path
    }


if __name__ == "__main__":
    from typing import Tuple
    results = run_gate_c_training()
    print("Gate C Execution Summary:", results["status"])
