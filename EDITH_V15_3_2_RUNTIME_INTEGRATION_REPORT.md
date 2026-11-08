# E.D.I.T.H. V15.3.2 — RUNTIME INTEGRATION REPORT
**Author**: ML Engineering Team  
**Date**: November 8, 2026  
**Status**: VERIFIED (PRODUCTION STABLE ON V14, RESEARCH READY ON V15.3)  

---

## 1. Runtime Architecture & Checkpoint Selection

`engine/mind_router.py` was enhanced to support dynamic checkpoint configuration, multi-tier fallback, and diagnostic telemetry:

### Checkpoint Catalog:
1. `"v14"` (**Production Default**):
   - Checkpoint: `models/checkpoints/edith_lm_v14/edith_lm_v14_chat.pt`
   - Tokenizer: `models/checkpoints/edith_bpe_tokenizer_v14_3k.json` (3,051 vocab)
   - Tier: Production (proven stability, low latency)
2. `"v15_3"` (**Research Candidate**):
   - Checkpoint: `models/checkpoints/edith_lm_v15_sft/edith_lm_v15_3_sft_best.pt`
   - Tokenizer: `models/checkpoints/edith_bpe_tokenizer_16k.json` (16,384 vocab)
   - Tier: Research (higher fluency, larger vocabulary)
3. `"v15_2"` (**Research Baseline**):
   - Checkpoint: `models/checkpoints/edith_lm_v15/edith_lm_v15_2_100m_best.pt`
   - Tokenizer: `models/checkpoints/edith_bpe_tokenizer_16k.json`
4. `"legacy_base"` (**Emergency Fallback**):
   - Checkpoint: `models/checkpoints/edith_lm/edith_lm_best.pt`

### Selection Precedence:
1. Constructor argument: `MindRouter(model_version="v15_3")`.
2. Environment variable: `EDITH_LM_MODEL_VERSION="v15_3"`.
3. Configuration file: `config/model_config.json` -> `"edith_lm_active_version"`.
4. Production Default: Strictly `"v14"`.

### Safe Rollback Mechanism:
If an invalid, missing, or corrupted model version is requested:
- The system logs a warning detailing the missing/corrupt asset.
- The router automatically falls back to `"v14"`.
- If `"v14"` fails, it falls back to `"legacy_base"`.
- If all neural models fail, it seamlessly falls back to Bridge Mode and deterministic handlers.
- `router.get_diagnostics()` returns complete rollback audit records.

---

## 2. End-to-End Pipeline Verification

Pipeline: `wake_edith.py` $\to$ `STT (stt_engine.py)` $\to$ `EdithBrain` $\to$ `MindRouter` $\to$ `EdithLMInference` $\to$ `MemoryStore` $\to$ `FluencyGuard` $\to$ `TTSSanitizer / TTS`

### Verified Capabilities (`tests/test_end_to_end_runtime_v15.py`):
1. **Normal Greetings**: Warm, non-repetitive greetings without template leaks.
2. **Natural Multi-Turn Chat**: Multi-turn dialogue maintained.
3. **Technical Inquiries**: Concepts routed cleanly without execution errors.
4. **Delayed Entity Recall**: Probed across 4 turns through SQLite memory; entity "Harshan" recalled accurately.
5. **Hinglish Empathy**: Conversational Hindi-English mixture processed naturally.
6. **Topic Switching**: Context cleanly switches when prompted.
7. **Device Execution**: GPU execution verified on NVIDIA RTX 4050 Laptop GPU (with CPU fallback).
8. **TTS Sanitization**: All XML tags (`<think>`, `<mood>`, `<act>`, `<memory>`) stripped before speech synthesis.
9. **Rollback Resilience**: Tested with non-existent version keys; rolled back cleanly to production V14.
10. **Tool Decoupling**: Conversational queries mentioning applications ("I love using Spotify", "WhatsApp encryption") never trigger tool actions (`can_execute == False`). Verified commands ("open notepad") execute only with $\ge 0.75$ operational confidence and safety firewall clearance.

### Results:
10/10 test cases PASSED in 101.2s.
