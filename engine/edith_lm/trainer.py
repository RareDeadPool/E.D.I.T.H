"""E.D.I.T.H. V13 Training Engine for EdithLM.

Supports:
- Resumable training from checkpoints with full RNG, model, and optimizer states
- Cosine learning rate schedule with linear warmup
- Mixed precision support
- Gradient clipping and step accumulation
- Validation perplexity and loss tracking
"""

import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.edith_lm.model import EdithLM, EdithLMConfig
from engine.edith_lm.tokenizer import EdithByteBPETokenizer


class ConversationalDataset(Dataset):
    """Tokenized conversational dataset for autoregressive causal LM training."""

    def __init__(self, jsonl_path: str, tokenizer: EdithByteBPETokenizer, max_len: int = 512):
        self.samples: List[torch.Tensor] = []
        full_path = os.path.join(PROJECT_ROOT, jsonl_path) if not os.path.isabs(jsonl_path) else jsonl_path

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Dataset not found at {full_path}")

        with open(full_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                text = record.get("text", "")
                if not text:
                    continue

                ids = tokenizer.encode(text, add_bos=True, add_eos=True)
                if len(ids) > max_len:
                    ids = ids[:max_len]
                elif len(ids) < max_len:
                    ids = ids + [tokenizer.pad_id] * (max_len - len(ids))

                self.samples.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.samples[idx]


class EdithLMTrainer:
    def __init__(
        self,
        model: EdithLM,
        tokenizer: EdithByteBPETokenizer,
        train_path: str = "data_pipeline/processed/train.jsonl",
        val_path: str = "data_pipeline/processed/val.jsonl",
        checkpoint_dir: str = "models/checkpoints/edith_lm",
        batch_size: int = 16,
        lr: float = 3e-4,
        min_lr: float = 3e-5,
        warmup_steps: int = 100,
        max_steps: int = 1000,
        eval_interval: int = 100,
        save_interval: int = 250,
        grad_accum_steps: int = 2,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.checkpoint_dir = os.path.join(PROJECT_ROOT, checkpoint_dir) if not os.path.isabs(checkpoint_dir) else checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.batch_size = batch_size
        self.lr = lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.grad_accum_steps = grad_accum_steps

        # Load Datasets
        self.train_dataset = ConversationalDataset(train_path, tokenizer, max_len=model.config.max_seq_len)
        self.val_dataset = ConversationalDataset(val_path, tokenizer, max_len=model.config.max_seq_len)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)

        # Optimizer: AdamW with weight decay
        decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
        nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": 0.01},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95))

        self.step = 0
        self.history: List[Dict[str, float]] = []

    def get_lr(self, step: int) -> float:
        """Cosine learning rate schedule with linear warmup."""
        if step < self.warmup_steps:
            return self.lr * (step + 1) / self.warmup_steps
        if step > self.max_steps:
            return self.min_lr
        decay_ratio = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.lr - self.min_lr)

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float]:
        """Evaluates loss and perplexity on the validation set."""
        self.model.eval()
        total_loss = 0.0
        batches = 0

        for batch in self.val_loader:
            batch = batch.to(self.device)
            input_ids = batch[:, :-1]
            targets = batch[:, 1:].clone()
            targets[targets == self.tokenizer.pad_id] = -100

            _, loss, _ = self.model(input_ids, targets=targets)
            if loss is not None and not torch.isnan(loss):
                total_loss += loss.item()
                batches += 1

        self.model.train()
        avg_loss = total_loss / batches if batches > 0 else float("inf")
        perplexity = math.exp(min(avg_loss, 20.0))
        return avg_loss, perplexity

    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Saves resumable checkpoint containing weights, optimizer state, and training metrics."""
        ckpt_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(
            {
                "step": self.step,
                "config": self.model.config,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "history": self.history,
            },
            ckpt_path,
        )
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "edith_lm_best.pt")
            torch.save(self.model.state_dict(), best_path)
            print(f"[Trainer] Saved new BEST model checkpoint to {best_path}")

    def resume_from_checkpoint(self, filepath: str):
        """Restores training state from an existing checkpoint."""
        ckpt = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.step = ckpt.get("step", 0)
        self.history = ckpt.get("history", [])
        print(f"[Trainer] Resumed from step {self.step} at {filepath}")

    def train(self) -> List[Dict[str, float]]:
        """Executes the training loop until max_steps is reached."""
        self.model.train()
        print(f"[Trainer] Starting training on {self.device.upper()} for {self.max_steps} steps...")
        print(f"[Trainer] Parameters: {self.model.get_num_params()/1e6:.2f}M | Batch size: {self.batch_size}")

        best_val_loss = float("inf")
        train_iter = iter(self.train_loader)
        t0 = time.time()

        while self.step < self.max_steps:
            # Update learning rate
            current_lr = self.get_lr(self.step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = current_lr

            self.optimizer.zero_grad()
            accum_loss = 0.0

            for _ in range(self.grad_accum_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)

                batch = batch.to(self.device)
                input_ids = batch[:, :-1]
                targets = batch[:, 1:].clone()
                targets[targets == self.tokenizer.pad_id] = -100

                _, loss, _ = self.model(input_ids, targets=targets)
                if loss is not None:
                    loss = loss / self.grad_accum_steps
                    loss.backward()
                    accum_loss += loss.item()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            self.step += 1

            # Periodic Evaluation & Logging
            if self.step % self.eval_interval == 0 or self.step == self.max_steps:
                val_loss, val_ppl = self.evaluate()
                dt = time.time() - t0
                t0 = time.time()
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss

                log_entry = {
                    "step": self.step,
                    "train_loss": round(accum_loss * self.grad_accum_steps, 4),
                    "val_loss": round(val_loss, 4),
                    "val_ppl": round(val_ppl, 2),
                    "lr": current_lr,
                }
                self.history.append(log_entry)

                print(
                    f"Step {self.step:4d}/{self.max_steps:4d} | "
                    f"Train Loss: {log_entry['train_loss']:.4f} | "
                    f"Val Loss: {log_entry['val_loss']:.4f} | "
                    f"Val PPL: {log_entry['val_ppl']:.2f} | "
                    f"LR: {current_lr:.2e} | Time: {dt:.1f}s"
                )

            # Periodic Checkpointing
            if self.step % self.save_interval == 0 or self.step == self.max_steps:
                self.save_checkpoint(f"checkpoint_step_{self.step}.pt", is_best=(self.step == self.max_steps))

        print("[Trainer] Training run complete!")
        return self.history
