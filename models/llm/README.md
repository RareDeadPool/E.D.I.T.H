# E.D.I.T.H. V12 Local LLM Weights Directory

This directory stores open-weights quantized models for local offline generative reasoning, dialogue, empathy, and conversational intelligence.

## Active Model
- **Model Architecture**: Qwen2.5-1.5B-Instruct
- **Format**: GGUF (4-bit Medium quantization: `q4_k_m`)
- **Filename**: `qwen2.5-1.5b-instruct-q4_k_m.gguf`
- **File Size**: ~1.06 GB (1,117,328,096 bytes)
- **Hugging Face Source**: `Qwen/Qwen2.5-1.5B-Instruct-GGUF`
- **License**: Apache 2.0 (Open Weights, Commercial-friendly, 100% Offline)

## Automated Downloader
If this file is missing on a fresh clone, run:
```bash
python scratch/download_local_llm.py
```
Or initialize `get_generative_reasoner()` in Python, which gracefully falls back to deterministic rule-based dialogue when the GGUF weights are not yet downloaded.
