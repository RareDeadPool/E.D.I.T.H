# E.D.I.T.H. V15.1: Proof-of-Concept Training & Resource Audit Report

## 1. Executive Summary & Training Status
The initial 25M-token Proof-of-Concept (POC) training run for **E.D.I.T.H. V15** was successfully completed on the local **NVIDIA GeForce RTX 4050 6GB Laptop GPU** in an isolated Python 3.12 CUDA environment (`.venv_train/`).

- **Run Status**: **COMPLETED** (1,525 of 1,525 steps).
- **Architecture**: Custom `EdithLM` 55.90M Medium Class (Decoder-Only Causal Transformer, SwiGLU, RoPE $\theta=10,000$, RMSNorm, Tied Word Embeddings, 16,384 Byte-BPE Vocab, Context Length 512).
- **Third-Party Pretrained Weights**: **ZERO**. Qwen, Llama, Gemma, and Phi foundation weights have been 100% removed. Weights were initialized and trained strictly from scratch.
- **Elapsed Duration**: **26.20 minutes** (1,572.16 seconds).
- **CUDA Errors / OOMs**: **0**. Zero CUDA Out-of-Memory faults, zero PCIe paging interrupts, and zero NaN/Inf gradients.

---

## 2. Key Dataset & Token Accounting (Unique vs Processed)

To avoid confusing tokens processed with unique source tokens, all metrics were independently measured on the serialized dataset files:

| Metric | Measured Value | Analysis & Notes |
| :--- | :--- | :--- |
| **Completed Optimizer Steps** | **1,525 steps** | Exactly 1,525 gradient steps completed with AdamW ($\text{lr}=6\times 10^{-4} \rightarrow 6\times 10^{-5}$). |
| **Total Sequence Positions Processed** | **24,985,600 positions** | 1,525 steps $\times$ 16,384 sequence tokens per step (batch size 8, grad accum 4, seq len 512). |
| **Actual Non-Pad Tokens Seen** | **6,044,375 tokens** | Padding positions (`<pad>` id 0) were strictly masked out from loss computation (`ignore_index=-100`). |
| **Unique Training Tokens** | **1,806,642 tokens** | Measured on 14,594 unique samples in `data_pipeline/v15/processed/train.jsonl`. |
| **Unique Validation Tokens** | **198,854 tokens** | Measured on 1,621 unique samples in `data_pipeline/v15/processed/val.jsonl`. |
| **Total Ingested Corpus Tokens** | **2,005,496 tokens** (~2.01M) | Total unique human tokens across both splits. |
| **Training Split Samples** | **14,594 examples** | 14,594 unique SHA-256 hashes (100% deduplicated). |
| **Validation Split Samples** | **1,621 examples** | 1,621 unique SHA-256 hashes (100% deduplicated). |
| **Train/Val Hash Overlap** | **0 samples (0.00%)** | Zero data contamination between splits. |
| **Corpus Repetitions (Valid Tokens)** | **3.35 epochs** | $6,044,375 / 1,806,642 \approx 3.35$ full passes over unique non-pad training text. |
| **Corpus Repetitions (Sequence Frames)**| **13.83 passes** | $24,985,600 / 1,806,642 \approx 13.83$ sequence frame presentations (due to padding). |

---

## 3. Sequence Packing & Document Boundary Verification

A crucial failure mode in causal language model training is allowing sequence packing to concatenate distinct documents without attention boundaries, causing cross-contamination of unrelated contexts.

- **Packing Strategy**: In `PackedJsonlDataset`, each row corresponds to exactly **ONE** isolated document or dialogue.
- **Boundary Handling**: Every sample begins with `<bos>` and terminates with `<eos>`.
- **Padding Behavior**: Sequences shorter than 512 tokens are padded to length 512 with `<pad>` (id 0).
- **Loss Masking**: In both the training loop and validation evaluation, target labels are transformed via:
  ```python
  targets = batch[:, 1:].clone()
  targets[targets == tokenizer.pad_id] = -100
  ```
- **Verification**: Cross-entropy was computed strictly on valid document tokens using PyTorch `ignore_index=-100`. Documents **never crossed sequence boundaries**, ensuring that context from one training document never bled into another.

---

## 4. Hardware Utilization & VRAM Profile (RTX 4050 6GB)

Hardware profiling was conducted throughout the 26.2-minute run:

| Metric | Measured Value | Budget / Ceiling | Headroom |
| :--- | :--- | :--- | :--- |
| **Physical VRAM Allocated** | **4,243.7 MB** | 6,144 MB (GDDR6) | **1,900.3 MB (30.9%)** |
| **VRAM Utilization Ratio** | **69.1%** | 100% | Well below PCIe paging cliff |
| **Average Training Throughput** | **3,844.6 valid tok/s** | — | Sequence rate: 16,384 tok/s |
| **Step Execution Time** | **1.03 s / step** | — | Highly consistent across run |
| **Peak System RAM** | **3.45 GB** | 16.0 GB | Tensor cache in single `.pt` file |
| **Checkpoint Save Duration** | **1.82 s** | — | Saved directly to NVMe SSD |
| **Thermal / Throttling Status** | **Stable (71°C peak)** | 87°C throttle | Zero thermal throttling observed |

---

## 5. Learning Dynamics & Checkpoint Tracking

Validation evaluations and generation journal samples were captured every 300 steps (~5M sequence positions):

| Step | Valid Tokens Seen | Train Loss (Cross-Entropy) | Val Loss | Val Perplexity | Learning Rate | Grad Norm |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0 (Init)**| 0 | 9.8283 | 9.7935 | 17,916.52 | $0.0$ | — |
| **1** | 3,817 | 9.8188 | — | — | $8.00\times 10^{-6}$ | 6.38 |
| **150** | 600,535 | 5.7886 | — | — | $6.00\times 10^{-4}$ | 0.89 |
| **300** | 1,187,977 | 5.2349 | 5.2633 | 193.11 | $5.84\times 10^{-4}$ | 0.74 |
| **600** | 2,379,474 | 4.8550 | 4.8629 | 129.40 | $4.69\times 10^{-4}$ | 0.69 |
| **900** | 3,572,443 | 4.4172 | 4.6259 | 102.09 | $3.25\times 10^{-4}$ | 0.72 |
| **1200** | 4,757,959 | 4.2052 | 4.4937 | 89.46 | $1.76\times 10^{-4}$ | 0.74 |
| **1500** | 5,947,059 | 4.0010 | 4.4353 | 84.38 | $6.04\times 10^{-5}$ | 0.84 |
| **1525** | 6,044,375 | 3.9880 | **4.4345** | **84.31** | $6.00\times 10^{-5}$ | 0.78 |

### Loss Trajectory Visualization
```
Val Loss Curve (Tokens Seen vs Held-Out Loss):
9.79 | * (Baseline Init)
8.50 |
7.00 |
5.26 |       * (1.18M tokens)
4.86 |             * (2.38M tokens)
4.62 |                   * (3.57M tokens)
4.49 |                         * (4.76M tokens)
4.43 |                               * (6.04M tokens - Best Checkpoint)
     +---------------------------------------------------------------> Tokens Seen
       0      1M    2M    3M    4M    5M    6M
```

### Analysis of Learning Trajectory:
1. **Steady Monotonic Decrease**: Validation loss dropped from **9.7935 down to 4.4345**, and perplexity dropped from **17,916 to 84.31**.
2. **Absence of Catastrophic Overfitting**: Training loss reached 3.99 while validation loss reached 4.43. While the gap widened slightly after 2 epochs, validation loss continued to improve through step 1,525, indicating that the model did not memorize the training set to the detriment of validation generalization.
3. **Plateau Warning**: Between step 1,200 and step 1,525, validation loss improvement flattened significantly ($\Delta = -0.0592$), reflecting the limited size (~2.0M tokens) of the underlying corpus.

---

## 6. Checkpoint Integrity & Preservation

All checkpoints were verified on disk and backed up prior to subsequent testing:
- **Best Validation Checkpoint**:
  - Path: `models/checkpoints/edith_lm_v15/edith_lm_v15_poc_25m_best.pt`
  - Size: 670,945,978 bytes (includes model weights, AdamW optimizer states, step metadata, and validation metrics).
  - Best Val Loss: **4.4345** (PPL 84.31) at step 1,525.
- **Verified Offline Backup**:
  - Path: `models/checkpoints/edith_lm_v15/edith_lm_v15_poc_25m_best.backup.pt`
  - Verified identical byte copy.
- **Latest Checkpoint**:
  - Path: `models/checkpoints/edith_lm_v15/edith_lm_v15_poc_25m_latest.pt`
  - Size: 223,645,310 bytes (clean inference model weights and configuration).

