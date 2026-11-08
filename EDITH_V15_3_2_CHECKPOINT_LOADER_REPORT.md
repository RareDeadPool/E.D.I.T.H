# E.D.I.T.H. V15.3.2 — CHECKPOINT LOADER REPORT
**Author**: ML Engineering Team  
**Date**: November 8, 2026  
**Status**: VERIFIED & TESTED  

---

## 1. Problem Diagnosis

The V15.3.1 runtime audit discovered that calling `EdithLMInference.from_checkpoint()` crashed when loading V15 checkpoints.

### Root Cause:
- `engine/edith_lm/inference.py` line 62 executed:
  `state_dict = ckpt.get("model_state", ckpt)`
- V15 training checkpoints saved state dictionaries under the key `"model_state_dict"`, whereas legacy V14 trainers saved under `"model_state"`.
- When `"model_state"` was absent, `.get()` returned the entire `ckpt` dictionary (including outer metadata keys: `step`, `targets_seen`, `optimizer_state_dict`, `sft_val_loss`).
- Passing the full dictionary into `model.load_state_dict()` threw `RuntimeError: Unexpected key(s) in state_dict`.

---

## 2. Implementation of Robust Multi-Format Loader

`EdithLMInference.from_checkpoint()` in `engine/edith_lm/inference.py` was completely overhauled with strict validation:

1. **Format Detection**:
   ```python
   if "model_state_dict" in ckpt:
       state_dict = ckpt["model_state_dict"]
       ckpt_format = "v15_trainer_dict"
   elif "model_state" in ckpt:
       state_dict = ckpt["model_state"]
       ckpt_format = "legacy_trainer_dict"
   elif "tok_embeddings.weight" in ckpt:
       state_dict = ckpt
       ckpt_format = "raw_state_dict"
   ```
2. **Architecture Configuration Resolution**:
   - Inspects `ckpt.get("config")`.
   - If missing, inspects weight shapes (`tok_embeddings.weight`, `q_proj.weight`, `gate_proj.weight`, layer count) and builds an exact matching `EdithLMConfig`.
3. **Tokenizer Auto-Alignment & Mismatch Detection**:
   - If checkpoint has `vocab_size == 16384` and caller passes 3k/4k tokenizer, auto-aligns to `models/checkpoints/edith_bpe_tokenizer_16k.json`.
   - If checkpoint has `vocab_size == 3051`, auto-aligns to `models/checkpoints/edith_bpe_tokenizer_v14_3k.json`.
   - If tokenizer vocab size does not match checkpoint vocab size and cannot be aligned, raises an explanatory `ValueError`.
4. **Parameter Key Verification**:
   - Executes `model.load_state_dict(state_dict, strict=False)`.
   - Allows missing `output.weight` only when `tie_word_embeddings=True`.
   - Raises explicit error on any unexpected or missing weight tensors.

---

## 3. Verification Test Suite Results

Test suite: `tests/test_checkpoint_loader_v15.py`
Execution environment: CUDA PyTorch 2.6.0+cu124 on RTX 4050 Laptop GPU.

```
Ran 4 tests in 6.859s

OK

Test Cases:
1. test_v15_3_sft_checkpoint_loading_and_generation: PASSED (GPU generation verified)
2. test_v14_legacy_checkpoint_loading:              PASSED (3,051 vocab compatibility verified)
3. test_raw_state_dict_loading:                     PASSED (Raw weights verified)
4. test_tokenizer_auto_alignment:                   PASSED (Auto-alignment verified)
```

The V15.3 SFT best checkpoint (`models/checkpoints/edith_lm_v15_sft/edith_lm_v15_3_sft_best.pt`) loads cleanly and generates text without error. Checkpoint files remain strictly untouched and non-overwritten.
