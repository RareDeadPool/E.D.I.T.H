"""
E.D.I.T.H. V15.3 Supervised Fine-Tuning (SFT) Training Pipeline
Fine-tunes the pretrained EdithLM model (resumed from edith_lm_v15_2_100m_best.pt)
using causal supervised fine-tuning with strict target token masking.
User prompt positions are masked with -100; loss is computed ONLY on assistant responses.
Preserves base model weights; saves exclusively to models/checkpoints/edith_lm_v15_sft/.
"""

import os
import sys
import time
import math
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.edith_lm.model import EdithLM, EdithLMConfig
from engine.edith_lm.tokenizer import EdithByteBPETokenizer
from engine.fluency_guard import FluencyGuard


class SFTDataset(Dataset):
    def __init__(self, cache_file: str):
        data = torch.load(cache_file, map_location="cpu", weights_only=True)
        self.x = data["x"]
        self.y = data["y"]

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def evaluate_sft_loss(model: EdithLM, val_loader: DataLoader, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                _, loss, _ = model(x, targets=y)
            num_valid = (y != -100).sum().item()
            if num_valid > 0:
                total_loss += loss.item() * num_valid
                total_tokens += num_valid

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(20.0, avg_loss))
    model.train()
    return avg_loss, ppl


def evaluate_general_ppl(model: EdithLM, stream_val_loader: DataLoader, device: torch.device):
    """Monitors general language perplexity to ensure zero catastrophic forgetting."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in stream_val_loader:
            batch = batch.to(device).long()
            x = batch[:, :-1]
            y = batch[:, 1:]
            with torch.amp.autocast("cuda", dtype=torch.float16):
                _, loss, _ = model(x, targets=y)
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(20.0, avg_loss))
    model.train()
    return avg_loss, ppl


def sample_generation(model: EdithLM, tokenizer: EdithByteBPETokenizer, prompt: str, device: torch.device, max_new_tokens: int = 55) -> str:
    model.eval()
    formatted = f"<mood>calm_focused</mood><user>{prompt}</user><edith>"
    prompt_ids = tokenizer.encode(formatted, add_bos=True, add_eos=False)
    generated = list(prompt_ids)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            curr_input = torch.tensor([generated[-512:]], dtype=torch.long, device=device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, _, _ = model(curr_input)
            next_logits = logits[0, -1, :].clone()
            for token_id in set(generated[len(prompt_ids):]):
                if next_logits[token_id] > 0:
                    next_logits[token_id] /= 1.2
                else:
                    next_logits[token_id] *= 1.2
            next_logits = next_logits / 0.7
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > 0.85
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_logits[indices_to_remove] = -float("Inf")
            probs = F.softmax(next_logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1).item()
            if next_tok == tokenizer.eos_id:
                break
            generated.append(next_tok)

    full_text = tokenizer.decode(generated)
    if full_text.startswith("<bos>"):
        full_text = full_text[5:]
    if full_text.startswith(formatted):
        response = full_text[len(formatted):].strip()
    else:
        response = full_text.strip()
    model.train()
    return response


def run_sft_training(total_steps: int = 1200, peak_lr: float = 8e-5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("E.D.I.T.H. V15.3 SUPERVISED FINE-TUNING PIPELINE")
    print(f"Device: {device}")
    print(f"Target SFT Steps: {total_steps:,} | Peak LR: {peak_lr:.2e}")
    print("=" * 80)

    tok_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_bpe_tokenizer_16k.json")
    tokenizer = EdithByteBPETokenizer(vocab_file=tok_path)
    print(f"Loaded Tokenizer: {tokenizer.vocab_size} vocab entries")

    # Resume from V15.2 Best Pretrained Checkpoint
    pretrained_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_lm_v15", "edith_lm_v15_2_100m_best.pt")
    print(f"Loading pretrained model weights from: {pretrained_path}...")
    ckpt_data = torch.load(pretrained_path, map_location=device, weights_only=False)
    cfg = ckpt_data["config"]
    model = EdithLM(cfg).to(device)
    model.load_state_dict(ckpt_data["model_state_dict"])
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M (tied)")

    # DataLoaders
    train_cache = os.path.join(PROJECT_ROOT, "data_pipeline", "v15", "processed_sft", "train_sft_cache_512.pt")
    val_cache = os.path.join(PROJECT_ROOT, "data_pipeline", "v15", "processed_sft", "val_sft_cache_512.pt")
    stream_val_cache = os.path.join(PROJECT_ROOT, "data_pipeline", "v15", "processed_expanded", "val_expanded_stream_cache_512.pt")

    train_ds = SFTDataset(train_cache)
    val_ds = SFTDataset(val_cache)

    batch_size = 8
    grad_accum_steps = 4

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    stream_val_tensors = torch.load(stream_val_cache, map_location="cpu", weights_only=True)
    stream_val_loader = DataLoader(stream_val_tensors, batch_size=batch_size, shuffle=False)

    print(f"Train SFT Dialogues: {len(train_ds):,} | Val SFT Dialogues: {len(val_ds):,}")
    print(f"Effective batch size: {batch_size * grad_accum_steps} dialogues per optimizer step")

    # Optimizer & Cosine Schedule
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.95), weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda")

    warmup_steps = 100
    min_lr = 1e-5

    def get_lr(step: int) -> float:
        if step < warmup_steps:
            return peak_lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (peak_lr - min_lr) * cosine_decay

    # Directory for SFT candidates
    sft_ckpt_dir = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_lm_v15_sft")
    os.makedirs(sft_ckpt_dir, exist_ok=True)
    best_sft_path = os.path.join(sft_ckpt_dir, "edith_lm_v15_3_sft_best.pt")
    latest_sft_path = os.path.join(sft_ckpt_dir, "edith_lm_v15_3_sft_latest.pt")
    journal_path = os.path.join(PROJECT_ROOT, "data_pipeline", "v15", "sft_training_journal.jsonl")

    # Baseline validation evaluation
    print("\nRunning initial baseline evaluation before SFT...")
    init_sft_loss, init_sft_ppl = evaluate_sft_loss(model, val_loader, device)
    init_gen_loss, init_gen_ppl = evaluate_general_ppl(model, stream_val_loader, device)
    print(f"Initial SFT Val Loss: {init_sft_loss:.4f} (PPL: {init_sft_ppl:.2f})")
    print(f"Initial Base General PPL: {init_gen_ppl:.2f} (Loss: {init_gen_loss:.4f})")

    phase6_probes = [
        "Tell me a joke.",
        "That's not funny.",
        "Roast my coding style.",
        "My friend has been unemployed for three years.",
        "My friend is very sad.",
        "I have a friend named Harshan.",
        "I'm stressed about college.",
        "What do you think I should do?",
        "I disagree with you.",
        "Explain recursion to me.",
        "Why does this code fail?",
        "Let's talk about something else."
    ]

    best_val_loss = float("inf")
    t0 = time.perf_counter()
    data_iter = iter(train_loader)
    targets_seen = 0

    print("\n" + "=" * 80)
    print(f"LAUNCHING SFT TRAINING RUN ({total_steps:,} steps)...")
    print("=" * 80)

    for step in range(1, total_steps + 1):
        lr_now = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        step_loss = 0.0
        step_targets = 0
        for _ in range(grad_accum_steps):
            try:
                x_b, y_b = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x_b, y_b = next(data_iter)

            x_b = x_b.to(device)
            y_b = y_b.to(device)
            valid_targets = (y_b != -100).sum().item()
            step_targets += valid_targets
            targets_seen += valid_targets

            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, loss, _ = model(x_b, targets=y_b)
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()
            step_loss += loss.item() * grad_accum_steps

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if step % 25 == 0 or step == 1:
            elapsed = time.perf_counter() - t0
            tok_s = targets_seen / max(0.001, elapsed)
            peak_vram = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            print(f"SFT Step {step:4d}/{total_steps} | Targets: {targets_seen:>8,d} | Loss: {step_loss:6.4f} | LR: {lr_now:.2e} | Grad: {grad_norm:.2f} | Speed: {tok_s:6.1f} tok/s | VRAM: {peak_vram:5.0f}MB")

        # Evaluation every 200 steps or at the end
        if step % 200 == 0 or step == total_steps:
            val_loss, val_ppl = evaluate_sft_loss(model, val_loader, device)
            gen_loss, gen_ppl = evaluate_general_ppl(model, stream_val_loader, device)
            print(f"\n>>> [EVALUATION @ STEP {step}] SFT Val Loss: {val_loss:.4f} (PPL: {val_ppl:.2f}) | General PPL: {gen_ppl:.2f}")

            journal_entries = []
            for p in phase6_probes:
                gen = sample_generation(model, tokenizer, p, device)
                clean_resp = gen.split("</edith>")[0].strip()
                is_fluent, diag = FluencyGuard.inspect(clean_resp, query=p)
                journal_entries.append({
                    "step": step,
                    "prompt": p,
                    "clean_response": clean_resp,
                    "raw_generation": gen,
                    "word_validity": diag.get("word_validity_ratio", 0.0),
                    "trigram_score": diag.get("char_trigram_logprob", 0.0),
                    "is_fluent": is_fluent
                })
                print(f"    P: \"{p}\"")
                print(f"    R: \"{clean_resp}\" (Fluent: {is_fluent})\n")

            with open(journal_path, "a", encoding="utf-8") as f:
                for entry in journal_entries:
                    f.write(json.dumps(entry) + "\n")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "step": step,
                    "targets_seen": targets_seen,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "sft_val_loss": val_loss,
                    "sft_val_ppl": val_ppl,
                    "general_val_ppl": gen_ppl,
                    "config": cfg
                }, best_sft_path)
                print(f"    * Saved best SFT candidate: {best_sft_path} (SFT Val Loss: {val_loss:.4f})")

    # Save latest
    torch.save({
        "step": total_steps,
        "targets_seen": targets_seen,
        "model_state_dict": model.state_dict(),
        "sft_val_loss": val_loss,
        "sft_val_ppl": val_ppl,
        "general_val_ppl": gen_ppl,
        "config": cfg
    }, latest_sft_path)

    total_time = time.perf_counter() - t0
    print("\n" + "=" * 80)
    print("SFT TRAINING COMPLETED:")
    print(f"  Total SFT Target Tokens Supervised: {targets_seen:,}")
    print(f"  Total Duration:                     {total_time:.1f} s ({total_time / 60:.2f} mins)")
    print(f"  Best SFT Val Loss:                  {best_val_loss:.4f}")
    print(f"  Best Checkpoint:                    {best_sft_path}")
    print("=" * 80)

if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
    lr = float(sys.argv[2]) if len(sys.argv) > 2 else 8e-5
    run_sft_training(total_steps=steps, peak_lr=lr)

