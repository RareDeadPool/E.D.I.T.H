"""
E.D.I.T.H. V15 Real GPU Benchmark Suite (RTX 4050 6GB Laptop GPU)
Executes a 100-step training benchmark on actual hardware across:
1. Baseline ~37M configuration (actual 37.23M tied)
2. Medium ~60M configuration (actual 59.77M tied)
3. Heavy ~85M configuration (actual 83.88M tied)

Measures:
- Actual tokens/second on real CUDA hardware
- Step latency (ms/step)
- Peak VRAM allocation (MB)
- Correct tokens-per-parameter scaling calculations
"""

import os
import sys
import time
import math
from typing import Dict, Any, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from engine.edith_lm.model import EdithLM, EdithLMConfig


def run_gpu_benchmark(
    name: str,
    n_layers: int,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    intermediate_dim: int,
    vocab_size: int = 16384,
    seq_len: int = 512,
    batch_size: int = 16,
    grad_accum_steps: int = 8,
    n_steps: int = 100
) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! Benchmark requires NVIDIA GPU.")

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)

    # 1. Instantiate configuration and model
    cfg = EdithLMConfig(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        intermediate_dim=intermediate_dim,
        max_seq_len=seq_len,
        tie_word_embeddings=True
    )
    model = EdithLM(cfg).to(device)

    # Count exact parameters
    total_params = sum(p.numel() for p in model.parameters())
    embed_params = model.tok_embeddings.weight.numel()
    backbone_params = total_params - embed_params

    print("\n" + "=" * 80)
    print(f"BENCHMARK: {name} on {gpu_name} (Total VRAM: {total_vram_mb:.0f} MB)")
    print(f"  Architecture: layers={n_layers}, dim={dim}, heads={n_heads}/{n_kv_heads}, inter={intermediate_dim}")
    print(f"  Embedding Parameters (tied): {embed_params / 1e6:.2f}M")
    print(f"  Backbone Parameters:         {backbone_params / 1e6:.2f}M")
    print(f"  ACTUAL TOTAL PARAMETERS:     {total_params / 1e6:.2f}M")
    print(f"  Batch Config: micro_batch={batch_size}, seq_len={seq_len}, accum={grad_accum_steps}")
    print(f"  Effective Batch Size:        {batch_size * seq_len * grad_accum_steps:,} tokens/optimizer step")
    print("=" * 80)

    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda")

    # Reset VRAM tracking
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    # 2. Warmup (10 micro-steps)
    print("Running 10 warmup steps...")
    for _ in range(10):
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits, loss, _ = model(x, targets=y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()

    # 3. Timed benchmark (100 steps)
    print(f"Executing {n_steps} timed benchmark steps...")
    tokens_per_micro_step = batch_size * seq_len
    total_tokens = n_steps * tokens_per_micro_step

    t0 = time.perf_counter()
    for step in range(n_steps):
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits, loss, _ = model(x, targets=y)
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # Metrics
    tok_sec = total_tokens / elapsed
    ms_per_step = (elapsed / n_steps) * 1000
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    vram_headroom_mb = total_vram_mb - peak_vram_mb

    hours_per_10m = (10e6 / tok_sec) / 3600
    hours_per_100m = (100e6 / tok_sec) / 3600
    hours_per_1b = (1e9 / tok_sec) / 3600

    # Chinchilla token-to-parameter ratio at 1B tokens
    tokens_per_param_1b = 1e9 / total_params

    print("\n--- MEASURED RESULTS ---")
    print(f"  Execution Time ({n_steps} steps): {elapsed:.2f} s")
    print(f"  Step Latency:                 {ms_per_step:.2f} ms/step")
    print(f"  MEASURED THROUGHPUT:          {tok_sec:.2f} tokens/second")
    print(f"  Peak VRAM Used:               {peak_vram_mb:.1f} MB ({peak_vram_mb / total_vram_mb:.1%} of 6GB)")
    print(f"  VRAM Headroom:                {vram_headroom_mb:.1f} MB")
    print(f"  10M Token POC Time:           {hours_per_10m * 60:.1f} minutes")
    print(f"  100M Token Milestone:         {hours_per_100m:.2f} hours")
    print(f"  1.0B Token Target:            {hours_per_1b:.2f} hours ({hours_per_1b / 24:.2f} days)")
    print(f"  Tokens-per-Parameter at 1B:   {tokens_per_param_1b:.1f} tok/param (Chinchilla min: 20.0)")

    # Clean up model
    del model, optimizer, scaler
    torch.cuda.empty_cache()

    return {
        "name": name,
        "actual_params_m": total_params / 1e6,
        "tok_sec": tok_sec,
        "ms_per_step": ms_per_step,
        "peak_vram_mb": peak_vram_mb,
        "vram_headroom_mb": vram_headroom_mb,
        "hours_per_10m": hours_per_10m,
        "hours_per_100m": hours_per_100m,
        "hours_per_1b": hours_per_1b,
        "tokens_per_param_1b": tokens_per_param_1b
    }


def main():
    print("=" * 80)
    print("E.D.I.T.H. V15 HARDWARE BENCHMARK PROTOCOL")
    print("Host Device: NVIDIA GeForce RTX 4050 6GB Laptop GPU")
    print("Environment: Isolated .venv_train (PyTorch CUDA cu124)")
    print("=" * 80)

    results = []

    # 1. 37M Target Class (10 layers, dim 512, inter 1536) -> ~37.2M actual
    res_37 = run_gpu_benchmark(
        name="37M Baseline Class",
        n_layers=10,
        dim=512,
        n_heads=8,
        n_kv_heads=2,
        intermediate_dim=1536,
        batch_size=16,
        grad_accum_steps=8,
        n_steps=100
    )
    results.append(res_37)

    # 2. 60M Target Class (12 layers, dim 576, inter 1728) -> ~59.8M actual
    res_60 = run_gpu_benchmark(
        name="60M Medium Class",
        n_layers=12,
        dim=576,
        n_heads=6,
        n_kv_heads=2,
        intermediate_dim=1728,
        batch_size=16,
        grad_accum_steps=8,
        n_steps=100
    )
    results.append(res_60)

    # 3. 85M Target Class (14 layers, dim 640, inter 2048) -> ~83.9M actual
    res_85 = run_gpu_benchmark(
        name="85M Heavy Class",
        n_layers=14,
        dim=640,
        n_heads=10,
        n_kv_heads=2,
        intermediate_dim=2048,
        batch_size=16,
        grad_accum_steps=8,
        n_steps=100
    )
    results.append(res_85)

    print("\n" + "=" * 80)
    print("EXECUTIVE COMPARISON SUMMARY (MEASURED ON REAL RTX 4050 GPU):")
    print(f"{'Config':<20} | {'Actual Params':<14} | {'Throughput':<15} | {'Peak VRAM':<12} | {'100M Time':<10} | {'1B Time':<10}")
    print("-" * 90)
    for r in results:
        print(f"{r['name']:<20} | {r['actual_params_m']:>6.2f}M       | {r['tok_sec']:>7.1f} tok/s   | {r['peak_vram_mb']:>5.0f} MB     | {r['hours_per_100m']:>5.2f} hrs  | {r['hours_per_1b']:>5.2f} hrs")
    print("=" * 80)


if __name__ == "__main__":
    main()

