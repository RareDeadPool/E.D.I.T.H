# E.D.I.T.H. V15.3.1 — Checkpoint and Artifact Integrity Report
**Project**: E.D.I.T.H. Language Model Independent Audit  
**Phase**: Phase 1 — Checkpoint and Artifact Integrity  
**Date**: November 8, 2026  
**Auditor**: Senior ML Engineer (Independent Audit)  
**Status**: **VERIFIED — ALL CHECKS PASSED**  

---

## 1. Executive Summary

An exhaustive artifact integrity verification was conducted across all model checkpoints, configuration dictionaries, weights state dictionaries, and tokenizers. All findings are derived directly from executable script inspection (`data_pipeline/v15/audit_checkpoints.py`) and recorded in the audit ledger (`data_pipeline/v15/audit_outputs/checkpoint_audit_ledger.json`).

### Key Audit Findings:
1. **Pretrained Baseline (`v15_2`) Is Unmodified**: The V15.2 pretrained checkpoint (`edith_lm_v15_2_100m_best.pt`) has SHA-256 `11c2818b9d122d9ecf64e27881188d5880020f21026c999ade5116d4f9b46d74`, perfectly matching its pre-SFT backup. Zero bits have changed.
2. **V15.3 SFT Best Checkpoint Corresponds Strictly to Step 200**: The checkpoint metadata, saved state, and validation metrics (`sft_val_loss: 2.2737`, `general_val_ppl: 30.11`) confirm that `edith_lm_v15_3_sft_best.pt` was generated at Step 200 after supervising 406,300 targets.
3. **V15.3 Latest Checkpoint Corresponds Strictly to Step 1200**: The latest checkpoint (`edith_lm_v15_3_sft_latest.pt`) preserves the final state at Step 1200 after 2,412,480 supervised targets.
4. **Architectural Parity**: All checkpoints share the identical 55,900,224-parameter architecture with tied word embeddings and 16,384-token Byte-Level BPE vocabulary.
5. **Independent Load & Forward Compatibility**: Every checkpoint successfully loaded into memory on CPU and GPU and passed dummy forward tensor inference without shape mismatch or nan values.

---

## 2. Checkpoint Ledger

| Checkpoint Name | File Path | File Size | SHA-256 Hash | Step | Targets Seen | Loss | Perplexity | Loadable? | Forward Test? |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **V15.2 Pretrain Best** | `models/checkpoints/edith_lm_v15/edith_lm_v15_2_100m_best.pt` | 639.86 MB | `11c2818b9d122d9ecf64e27881188d5880020f21026c999ade5116d4f9b46d74` | 4,200 | 68,800,512 | 3.3520 | 28.56 (General) | **YES** | **PASSED** |
| **V15.2 Pretrain Backup** | `models/checkpoints/edith_lm_v15/edith_lm_v15_2_100m_best.backup.pt` | 639.86 MB | `11c2818b9d122d9ecf64e27881188d5880020f21026c999ade5116d4f9b46d74` | 4,200 | 68,800,512 | 3.3520 | 28.56 (General) | **YES** | **PASSED** |
| **V15.2 Pretrain Latest** | `models/checkpoints/edith_lm_v15/edith_lm_v15_2_100m_latest.pt` | 213.28 MB | `95cd4daa0bc2ccb7bed7b0de5fce48ddd29b767bd9c1fb941013f05d1cce8d58` | 6,103 | 99,974,144 | 3.5028 | 33.21 (General) | **YES** | **PASSED** |
| **V15.3 SFT Best** | `models/checkpoints/edith_lm_v15_sft/edith_lm_v15_3_sft_best.pt` | 639.86 MB | `e25c57cefe5a158fdf63349a96f283a0d06967f573f9ad6735fca2dec49a454b` | **200** | **406,300** | **2.2737** | **9.72** (SFT) / **30.11** (General) | **YES** | **PASSED** |
| **V15.3 SFT Latest** | `models/checkpoints/edith_lm_v15_sft/edith_lm_v15_3_sft_latest.pt` | 213.28 MB | `814fbdcfc334c2e506f20d2aedadf7a08b55ae7b1e3d5582482d8588b88ce29c` | **1,200** | **2,412,480** | **2.2873** | **9.85** (SFT) / **33.94** (General) | **YES** | **PASSED** |

---

## 3. Architecture Specification & Parameter Breakdown

The architecture configuration was extracted directly from the checkpoint `config` dictionary:

```json
{
  "vocab_size": 16384,
  "max_seq_len": 512,
  "n_layers": 12,
  "n_heads": 6,
  "n_kv_heads": 2,
  "dim": 576,
  "intermediate_dim": 1728,
  "rope_theta": 10000.0,
  "norm_eps": 1e-06,
  "tie_word_embeddings": true
}
```

### Exact Parameter Accounting:
- **Embedding Matrix**: $16,384 \times 576 = 9,437,184$ parameters.
- **12 Transformer Layers**:
  - Attention ($W_q, W_k, W_v, W_o$): $12 \times (576 \times 576 + 576 \times 192 + 576 \times 192 + 576 \times 576) = 10,616,832$ parameters.
  - SwiGLU Feed-Forward ($W_1, W_2, W_3$): $12 \times (576 \times 1728 \times 3) = 35,831,808$ parameters.
  - Normalization (RMSNorm per-layer): $12 \times (576 \times 2) = 13,824$ parameters.
- **Final Layer Norm**: 576 parameters.
- **Output LM Head**: Tied to embedding matrix ($0$ additional unique parameters).
- **Total Parameters**: **55,900,224** ($55.90\text{M}$).

---

## 4. Tokenizer Integrity Audit

Two tokenizer files reside in `models/checkpoints/`:
- `edith_bpe_tokenizer_16k.json`
- `edith_bpe_tokenizer.json`

Both files have **identical SHA-256 hashes**: `984033b7b63e70e50b772d173125edac002d62793b9ca3f4d2d2550a3c7a8a84`.
- **Vocabulary Size**: 16,384 tokens
- **BOS Token ID**: 1 (`<bos>`)
- **EOS Token ID**: 2 (`<eos>`)
- **PAD Token ID**: 0 (`<pad>`)
- **Special Tokens Verified**: `<mood>`, `</mood>`, `<user>`, `</user>`, `<edith>`, `</edith>`.

---

## 5. Artifact Verification Verdict

**Gate A (Artifact Integrity) Status: PASSED**
- Checkpoints load cleanly and independently on both CPU and CUDA devices.
- SHA-256 hashes are immutable and documented.
- Pretrained V15.2 checkpoint is preserved without modification.
- SFT Best checkpoint was unequivocally established as Step 200.

