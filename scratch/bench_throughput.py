import time
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from engine.edith_lm.model import EdithLM, EdithLMConfig

def benchmark(n_layers=10, dim=576, n_heads=6, n_kv_heads=2, intermediate_dim=1536, name="37M"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n--- Benchmarking {name} on {device} ---")
    cfg = EdithLMConfig(
        vocab_size=16384,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        intermediate_dim=intermediate_dim,
        max_seq_len=512
    )
    model = EdithLM(cfg).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    batch_size = 4
    seq_len = 512
    tokens_per_step = batch_size * seq_len  # 2048 tokens/step

    # Warmup
    for _ in range(5):
        x = torch.randint(0, 16384, (batch_size, seq_len), device=device)
        y = torch.randint(0, 16384, (batch_size, seq_len), device=device)
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        loss = criterion(logits.view(-1, 16384), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 100 steps
    n_steps = 20  # run 20 steps to get accurate tokens/sec
    t0 = time.perf_counter()
    for _ in range(n_steps):
        x = torch.randint(0, 16384, (batch_size, seq_len), device=device)
        y = torch.randint(0, 16384, (batch_size, seq_len), device=device)
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        loss = criterion(logits.view(-1, 16384), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    elapsed = time.perf_counter() - t0

    tokens_processed = n_steps * tokens_per_step
    tok_sec = tokens_processed / elapsed
    hours_per_billion = (1e9 / tok_sec) / 3600
    print(f"Tokens/sec: {tok_sec:.2f} tok/s")
    print(f"Hours per 1B tokens: {hours_per_billion:.1f} hours ({hours_per_billion / 24:.1f} days)")
    return tok_sec, hours_per_billion

if __name__ == "__main__":
    benchmark(10, 576, 6, 2, 1536, "37M")
    benchmark(14, 640, 8, 2, 1792, "60M")
