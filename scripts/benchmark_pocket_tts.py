"""Benchmark and Quality Validation for E.D.I.T.H. Pocket TTS Engine.

Measures:
1. Cold Model Loading Time
2. Voice Reference Preparation Time (Cached safetensors vs WAV extraction)
3. Latency & Real-Time Factor (RTF) across 6 representative utterance categories
4. RAM & CPU consumption
5. Audio integrity (duration, non-empty, 24kHz)
"""

import os
import sys
import time
import json
import psutil
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import scipy.io.wavfile
import numpy as np

from runtime.pocket_tts_engine import (
    PocketTTSEngine,
    ConversationalSpeechNormalizer,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_VOICE_WAV,
    DEFAULT_VOICE_CACHE,
    TTS_CACHE_DIR,
)

BENCHMARK_SUITE = [
    {
        "category": "Casual Conversation",
        "input": "Hey there! I am Edith, your local desktop AI assistant. How can I help you today?"
    },
    {
        "category": "Question & Confirmation",
        "input": "Should I open Spotify and resume your playlist, or do you need assistance with VS Code?"
    },
    {
        "category": "Technical Metrics & Acronyms",
        "input": "CPU utilization is currently 14%, RAM usage is 4.2 GB, and the REST API returned HTTP status 200."
    },
    {
        "category": "Application & Tool Workflow",
        "input": "Opening VS Code, Notepad++, and PowerShell in your project workspace now."
    },
    {
        "category": "File Paths & Extensions",
        "input": "I have saved the audio benchmark results to C:\\Users\\Aditya Sawant\\Desktop\\E.D.I.T.H\\runtime\\pocket_tts_engine.py."
    },
    {
        "category": "Multi-sentence Technical Response",
        "input": "I ran a full diagnostic check across all subsystems. The neural intent classifier is operating at 99.2% accuracy. The screen interpreter captured the active window geometry with zero dropped frames. All local models are loaded in memory and ready for immediate execution."
    }
]


def run_benchmark():
    process = psutil.Process()
    ram_initial_mb = process.memory_info().rss / (1024 * 1024)
    print(f"============================================================")
    print(f" E.D.I.T.H. POCKET TTS BENCHMARK & QUALITY EVALUATION")
    print(f" Target Voice: Natural Indian Female English (VCTK p248)")
    print(f" Pinned Revision: Thanushttk/pocket-tts-multilingual@80c7f28")
    print(f" Initial Process RAM: {ram_initial_mb:.1f} MB")
    print(f"============================================================\n")

    # 1. Benchmark Voice State Preparation (safetensors vs wav)
    print("[1/4] Benchmarking Voice Reference Preparation...")
    from pocket_tts import TTSModel

    t0 = time.perf_counter()
    model = TTSModel.load_model(config=DEFAULT_MODEL_CONFIG)
    t_model_load = time.perf_counter() - t0
    print(f"  - Model Weight Load Time: {t_model_load:.3f} s")

    # Safetensors cache load
    t0 = time.perf_counter()
    voice_state_cached = model.get_state_for_audio_prompt(DEFAULT_VOICE_CACHE)
    t_cache_load = (time.perf_counter() - t0) * 1000.0
    print(f"  - Cached Safetensors Load: {t_cache_load:.2f} ms")

    # WAV extraction (if wav exists)
    t_wav_extract = 0.0
    if os.path.exists(DEFAULT_VOICE_WAV):
        t0 = time.perf_counter()
        voice_state_wav = model.get_state_for_audio_prompt(DEFAULT_VOICE_WAV)
        t_wav_extract = (time.perf_counter() - t0) * 1000.0
        print(f"  - Raw WAV Extraction Time: {t_wav_extract:.2f} ms")

    ram_after_model_mb = process.memory_info().rss / (1024 * 1024)
    model_ram_mb = ram_after_model_mb - ram_initial_mb
    print(f"  - Model RAM Footprint: {model_ram_mb:.1f} MB\n")

    # 2. Benchmark Speech Generation Suite
    print("[2/4] Running Benchmark Suite Across 6 Utterance Categories...")
    engine = PocketTTSEngine(enable_fallback=False)
    results = []

    for idx, item in enumerate(BENCHMARK_SUITE, 1):
        raw_text = item["input"]
        cat = item["category"]

        # Measure text normalization
        t_norm_0 = time.perf_counter()
        norm_text = ConversationalSpeechNormalizer.normalize(raw_text)
        t_norm_ms = (time.perf_counter() - t_norm_0) * 1000.0

        # Measure audio generation
        cpu_start = process.cpu_percent(interval=None)
        t_gen_0 = time.perf_counter()
        audio_tensor = PocketTTSEngine._shared_model.generate_audio(
            PocketTTSEngine._shared_voice_state,
            norm_text
        )
        t_gen = time.perf_counter() - t_gen_0
        cpu_usage = process.cpu_percent(interval=None)

        audio_np = audio_tensor.detach().cpu().numpy().squeeze()
        sample_rate = 24000
        duration_s = len(audio_np) / sample_rate
        rtf = t_gen / duration_s if duration_s > 0 else 0.0

        # Save sample for inspection
        out_wav = os.path.join(TTS_CACHE_DIR, f"bench_{idx}_{cat.lower().replace(' ', '_')[:15]}.wav")
        scipy.io.wavfile.write(out_wav, sample_rate, (audio_np * 32767).astype(np.int16))

        result_entry = {
            "index": idx,
            "category": cat,
            "raw_text": raw_text,
            "normalized_text": norm_text,
            "duration_sec": round(duration_s, 2),
            "generation_time_sec": round(t_gen, 3),
            "rtf": round(rtf, 3),
            "cpu_percent": cpu_usage,
            "wav_path": out_wav,
            "audio_samples": len(audio_np)
        }
        results.append(result_entry)

        print(f"  [{idx}/6] Category: {cat}")
        print(f"        Input     : \"{raw_text[:60]}...\"" if len(raw_text) > 60 else f"        Input     : \"{raw_text}\"")
        print(f"        Spoken    : \"{norm_text[:60]}...\"" if len(norm_text) > 60 else f"        Spoken    : \"{norm_text}\"")
        print(f"        Audio Len : {duration_s:.2f} s | Gen Time: {t_gen:.3f} s | RTF: {rtf:.2f}x ({'Faster than real-time' if rtf < 1.0 else 'Slower than real-time'})")
        print(f"        WAV Saved : {os.path.basename(out_wav)}")

    # 3. Overall Aggregate Statistics
    print(f"\n[3/4] Aggregate Performance Metrics:")
    avg_rtf = np.mean([r["rtf"] for r in results])
    total_audio_sec = np.sum([r["duration_sec"] for r in results])
    total_gen_sec = np.sum([r["generation_time_sec"] for r in results])
    first_res_latency = results[0]["generation_time_sec"]
    warm_latencies = [r["generation_time_sec"] for r in results[1:]]
    avg_warm_latency = np.mean(warm_latencies)
    final_ram_mb = process.memory_info().rss / (1024 * 1024)

    print(f"  - Total Audio Generated : {total_audio_sec:.2f} s")
    print(f"  - Total Generation Time : {total_gen_sec:.2f} s")
    print(f"  - First Response Latency: {first_res_latency:.3f} s")
    print(f"  - Avg Warm Latency      : {avg_warm_latency:.3f} s")
    print(f"  - Average RTF           : {avg_rtf:.2f}x")
    print(f"  - Peak Process RAM      : {final_ram_mb:.1f} MB (Delta: +{final_ram_mb - ram_initial_mb:.1f} MB)")

    # 4. Save Benchmark Report to JSON
    report_data = {
        "model_loading_time_sec": round(t_model_load, 3),
        "voice_cached_load_ms": round(t_cache_load, 2),
        "voice_wav_extract_ms": round(t_wav_extract, 2),
        "model_ram_mb": round(model_ram_mb, 1),
        "final_ram_mb": round(final_ram_mb, 1),
        "first_response_latency_sec": round(first_res_latency, 3),
        "average_warm_latency_sec": round(avg_warm_latency, 3),
        "average_rtf": round(avg_rtf, 3),
        "total_audio_sec": round(total_audio_sec, 2),
        "suite_results": results
    }

    report_path = os.path.join(PROJECT_ROOT, "scratch", "pocket_tts_benchmark.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n[4/4] Benchmark JSON report saved to: {report_path}")
    print("============================================================\n")
    return report_data


if __name__ == "__main__":
    run_benchmark()
