# E.D.I.T.H. V15.1: Dataset Scale, Provenance & Quality Audit Report

## 1. Executive Summary
The V15 training corpus was constructed under **Option A (100% Organic Human Text)**. All synthetic LLM datasets (TinyStories, UltraChat, Alpaca) and ambiguous web crawls (FineWeb-Edu with AI scoring) were strictly barred.

- **Total Ingested Samples**: 16,215 examples across training (14,594) and validation (1,621).
- **Total Unique Source Tokens**: **2,005,496 tokens** (~2.01M tokens).
- **Exact Hash Deduplication**: 100% unique SHA-256 hashes in both splits. Zero train/val overlap.
- **Fundamental Finding**: The 25M-token POC training run processed 24.98M sequence tokens, which corresponds to **3.35 epochs of valid tokens** over a 1.8M unique-token training set. For full-scale pretraining, **repeating a 2M token corpus 500 times to claim "1 Billion Tokens" is mathematically flawed and will cause severe overfitting**. A massive expansion of organic human source material is mandatory.

---

## 2. Ingested Dataset Composition & Licensing Audit

All sources have been verified for commercial compatibility, share-alike requirements, and public domain validity:

| Source Name | Training Samples | Val Samples | Estimated Tokens | Domain | License | Compliance & Attribution Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Databricks Dolly-15k** | 7,309 | 850 | 1,150,000 | Human QA, brainstorming, summarization | **CC BY-SA 3.0** | Commercial OK. Requires attribution to Databricks, Inc. and share-alike on derivatives. |
| **Wikitext-103 (Wikipedia)** | 2,818 | 312 | 620,000 | Encyclopedic knowledge, STEM, history | **CC BY-SA 3.0** | Commercial OK with attribution to Wikimedia Foundation contributors. |
| **Project Gutenberg Classics** | 4,004 | 390 | 400,000 | English literature, narrative prose, dialogue | **Public Domain (Pre-1929 US)** | All Gutenberg license headers and metadata stripped. 100% public domain texts: *Sherlock Holmes*, *Alice in Wonderland*, *Frankenstein*, *A Tale of Two Cities*, *Pride and Prejudice*, *The War of the Worlds*, *Tom Sawyer*, *The Time Machine*. |
| **E.D.I.T.H. Authored Dialogues** | 453 | 52 | 150,000 | Assistant persona, banter, memory handling | **Proprietary Custom** | 100% human-authored by project creator. |
| **Total** | **14,594** | **1,621** | **~2,320,000** | Balanced Compact Organic Corpus | **Option A Verified** |

---

## 3. Linguistic & Domain Distribution Analysis

Analysis of the 16,215 corpus samples reveals the following domain balance:

```
Corpus Domain Breakdown by Token Volume:
+------------------------------------+--------------------------+
| Domain                             | Token Share (%)          |
+------------------------------------+--------------------------+
| Human Instruction & QA (Dolly-15k) | 49.6%                    |
| Encyclopedic Prose (Wikitext-103)  | 26.7%                    |
| Classic Literature & Dialogue      | 17.2%                    |
| Proprietary Assistant Persona      |  6.5%                    |
+------------------------------------+--------------------------+
```

### Critical Distribution Deficits:
1. **Severe Hinglish Scarcity**:
   - The current corpus contains fewer than 10,000 Hinglish tokens (< 0.5% of total).
   - This directly caused the Fluency Gate failure on colloquial Hindi/Hinglish prompts (*"kya chal raha hai"*, *"aaj bilkul man nahi kar raha"*), as the model lacked sufficient token transitions for Romanized Hindi.
2. **Conversational Multi-Turn Imbalance**:
   - 76% of the corpus consists of single-turn instructions or continuous encyclopedic prose.
   - Only 6.5% consists of multi-turn conversational exchanges with state continuity.
   - Consequently, when tested on multi-turn dialogue (*"I had a difficult day" $\rightarrow$ "Mostly because of college"*), the model defaulted to encyclopedic text rather than conversational empathy.

---

## 4. Expansion Plan for Future Milestones (100M & 1B Tokens)

To train a capable custom language model without degenerate memorization, we require independent, non-duplicated human text:

### Milestone 1: 100M-Token Expansion (Target: Gate 2 POC Scaling)
To train 100M tokens at $\le 2$ epochs requires **~50M unique source tokens**:
1. **Full Pre-1929 Project Gutenberg Corpus**: Download 1,000 public domain English novels and narrative essays (~40M tokens).
2. **Clean Simple English Wikipedia**: High-clarity educational prose with verified human authorship (~8M tokens).
3. **L3Cube Hinglish / Open Hindi-English Corpus**: Verified bilingual conversational pairs (~2M tokens).
4. **Estimated Download Size**: ~180 MB uncompressed text.

### Milestone 2: 1B-Token Expansion (Full Pretraining Target)
To train 1 Billion tokens at ~2.5 epochs requires **~400M unique source tokens**:
1. **Pre-2022 English Wikipedia (Complete Clean Dump)**: Stripped of AI-generated content (~250M tokens).
2. **The Open Book Genome & Gutenberg Classics**: ~100M tokens.
3. **OpenAssistant Conversations (Human-Vetted Tree Only)**: ~20M tokens of verified multi-turn dialogue.
4. **Curated Technical Documentation (Linux Kernel docs, Python PEPs, RFCs)**: ~30M tokens of clean technical prose.

