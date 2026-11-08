# E.D.I.T.H. V15.1: Gate 2 Recommendation & Full-Scale Pretraining Plan

## 1. Executive Summary & Verdict
The **E.D.I.T.H. V15 25M-Token Proof-of-Concept** has concluded on the **NVIDIA GeForce RTX 4050 6GB Laptop GPU**. 

### What Has Been Conclusively Proven:
1. **Custom Architecture Viability**: The custom 55.90M parameter Causal Transformer (`EdithLM`) runs with **100% stability**, high throughput (**16,384 sequence tok/s**), and safe memory consumption (**4.2 GB VRAM / 6.0 GB**, leaving 1.9 GB headroom). Zero OOMs or PCIe paging occurred.
2. **Word Salad is Eliminated**: The retrained 16,384-vocab Byte-Level BPE tokenizer solved the sub-word shattering issue that plagued V14. Raw base model outputs achieved **90.0% dictionary word validity** across 50 fixed probes (and 100% on English probes).
3. **Validation Convergence**: Validation loss dropped monotonically from **9.7935 down to 4.4345** (perplexity dropped from **17,916 to 84.31**).
4. **Zero Foundation Model Dependencies**: Qwen foundation weights were completely excised. The model architecture, tokenizer, embeddings, and weights were built and trained 100% from scratch.

### What Has NOT Yet Been Achieved (Why Full 1B Run Must Wait):
1. **Corpus Exhaustion**: The current training corpus contains only **~1.8M unique tokens**. Processing 25M tokens required 3.35 epochs of valid text. Repeating this tiny corpus 500 times to reach a "1 Billion Token" headline count is mathematically degenerate and will cause extreme memorization without general reasoning.
2. **Low Semantic Coherence (~14%)**: The raw model generates valid English words, but lacks world knowledge, causal deduction, and multi-turn conversational tracking. Prompts like *"What is recursion?"* or *"Why is the sky blue?"* produce hallucinatory associations.
3. **Fluency Gate Not Passed**: Semantic coherence ("makes sense") is ~14%, far below the $\ge 70\%$ Gate threshold.

**Definitive Recommendation**: **DO NOT start the 1-Billion Token training run yet.** We must first scale the unique organic human dataset and execute an intermediate **100M-Token Milestone Run** to verify genuine semantic generalization.

---

## 2. Gate 2 Prerequisites & Action Plan

Before launching larger-scale pretraining, the following 4 engineering prerequisites must be executed:

### Prerequisite 1: Expand Organic Human Dataset from 2M to 50M Unique Tokens
We cannot train a healthy 55M model on a 2M token corpus. We must ingest ~50M unique, high-quality, human-authored tokens under Option A:
1. **1,000 Pre-1929 Project Gutenberg Classics**: High-quality narrative prose and human dialogue (~35M tokens, 100% public domain).
2. **Simple English Wikipedia (Clean Dump)**: Clear, structured factual explanations (~10M tokens, CC BY-SA 3.0).
3. **L3Cube Hinglish / Open Hindi-English Corpus**: Multi-turn bilingual conversational exchanges to fix Hinglish failure modes (~3M tokens).
4. **OpenAssistant Human Trees (Human-only nodes)**: Verified multi-turn conversational trees (~2M tokens).

### Prerequisite 2: Calibrate the Fluency Guard
As revealed in [`EDITH_V15_1_FLUENCY_GUARD_AUDIT.md`](file:///c:/Users/Aditya%20Sawant/Desktop/E.D.I.T.H/EDITH_V15_1_FLUENCY_GUARD_AUDIT.md), the guard's 34.8% false rejection rate must be resolved:
1. Exempt question-mark sentences (`?`) from terminal preposition stranding rules.
2. Calibrate the character trigram logprob threshold from **-7.8 to -8.3**.
3. Ingest a 5,000-word software/hardware developer lexicon into the dictionary.
4. Auto-strip closing tags like `</edith>` during postprocessing rather than rejecting the utterance.

### Prerequisite 3: Upgrade Bridge Mode with a Similarity Gate
As proven in [`EDITH_V15_1_BRIDGE_MODE_AUDIT.md`](file:///c:/Users/Aditya%20Sawant/Desktop/E.D.I.T.H/EDITH_V15_1_BRIDGE_MODE_AUDIT.md), Bridge Mode must not return unrelated dialogue when similarity is low:
1. Enforce a minimum cosine similarity threshold ($\ge 0.75$).
2. Fall back to an honest admission of boundary (*"I don't have enough verified knowledge on that topic yet"*) for out-of-domain queries scoring below 0.75.
3. Train `LocalSentenceEncoder` with self-supervised SimCSE contrastive loss over `train.jsonl` to replace the random projection with a learned semantic encoder.

### Prerequisite 4: Execute 100M-Token Intermediate Milestone Run
- **Tokens**: 100,000,000 sequence tokens (6,103 steps).
- **Duration**: **~1.7 hours** on the RTX 4050 Laptop GPU.
- **Goal**: Verify whether held-out perplexity drops below 40.0 and semantic coherence on the 50 probe prompts crosses $\ge 40\%-50\%$.

---

## 3. Road to Full-Scale 1-Billion Token Pretraining

Once the 100M milestone confirms that semantic coherence is rising steadily with unique data:

```
[Current State: POC 25M] (6M valid tokens seen, ~14% coherence, PPL 84.31)
          │
          ▼
[Step 1: Dataset Expansion] (Ingest 50M unique human tokens: Gutenberg + SimpleWiki + Hinglish)
          │
          ▼
[Step 2: 100M Milestone Run] (~1.7 hours on RTX 4050, Target: PPL < 40, Coherence >= 40%)
          │
          ▼
[Fluency Gate Review] (Run 50 Probes + Human Blind Evaluation)
          │
          ▼
[Step 3: Full 1B Pretraining Run] (400M unique tokens, 16.9 hours on RTX 4050, Target: Coherence >= 70%)
          │
          ▼
[Gate 2 Promoted to Production Assistant]
```

---

## 4. Summary Table of Deliverables

The full documentation suite for E.D.I.T.H. V15.1 has been published to the repository:

1. [`EDITH_V15_1_POC_TRAINING_REPORT.md`](file:///c:/Users/Aditya%20Sawant/Desktop/E.D.I.T.H/EDITH_V15_1_POC_TRAINING_REPORT.md) — Comprehensive training, resource, and token audit.
2. [`EDITH_V15_1_GENERATION_QUALITY_REPORT.md`](file:///c:/Users/Aditya%20Sawant/Desktop/E.D.I.T.H/EDITH_V15_1_GENERATION_QUALITY_REPORT.md) — Generation quality tests and decoding benchmark.
3. [`EDITH_V15_1_FLUENCY_GUARD_AUDIT.md`](file:///c:/Users/Aditya%20Sawant/Desktop/E.D.I.T.H/EDITH_V15_1_FLUENCY_GUARD_AUDIT.md) — False rejection and calibration analysis.
4. [`EDITH_V15_1_DATA_QUALITY_REPORT.md`](file:///c:/Users/Aditya%20Sawant/Desktop/E.D.I.T.H/EDITH_V15_1_DATA_QUALITY_REPORT.md) — Source breakdown, licenses, and expansion requirements.
5. [`EDITH_V15_1_BRIDGE_MODE_AUDIT.md`](file:///c:/Users/Aditya%20Sawant/Desktop/E.D.I.T.H/EDITH_V15_1_BRIDGE_MODE_AUDIT.md) — Encoder provenance, retrieval accuracy, and tri-mode comparison.
6. [`EDITH_V15_1_GATE2_RECOMMENDATION.md`](file:///c:/Users/Aditya%20Sawant/Desktop/E.D.I.T.H/EDITH_V15_1_GATE2_RECOMMENDATION.md) — Strategic roadmap and scaling milestones.

