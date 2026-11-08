# E.D.I.T.H. V15.3 — Supervised Fine-Tuning (SFT) Dataset Report
**Project**: E.D.I.T.H. Custom Generative Language Model  
**Milestone**: V15.3 Assistant Behaviour Alignment & Supervised Fine-Tuning  
**Date**: November 8, 2026  
**Auditor**: Senior ML Engineer  

---

## 1. Executive Summary & Policy Compliance

The objective of the V15.3 SFT Dataset phase is to transition `EdithLM` (~55.90M parameters, trained from scratch on ~99.97M next-token prediction targets) from an unconditional text continuator to an aligned, natural conversational assistant with the E.D.I.T.H. persona.

### Core Policy Directives:
1. **Zero LLM-Generated Synthetic Data**: Strict prohibition of synthetic text (TinyStories, UltraChat, Alpaca, ChatGPT/Claude synthetic crawls). Every ingested training example is 100% human-authored organic material.
2. **Zero Third-Party Foundation Model Weights**: No initialization or distillation from Qwen, Llama, Gemma, Mistral, or Phi.
3. **Rigorous Licensing & Provenance Separation**: Commercial-compatible and non-commercial data sources are strictly audited, tracked, and segregated.
4. **Target-Only Loss Masking**: User prompt positions and metadata tags are masked with `-100`; loss is computed exclusively on assistant tokens and termination markers.

---

## 2. Ingested Dataset Inventory & Provenance

The unified SFT dataset incorporates **23,840 unique human-authored dialogues**, comprising **2,903,347 tokens** tokenized with the custom 16,384-token Byte-Level BPE tokenizer.

| Source Component | License | Permitted Usage | Attribution / Provenance | Unique Records | Total Tokens | Supervised Assistant Tokens |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Databricks Dolly-15k** | CC BY-SA 3.0 | Commercial (with SA) | Databricks, Inc. (Authored by 5,000+ human engineers) | 15,011 | 2,142,390 | 985,412 |
| **Empathetic Dialogues** | CC BY-NC 4.0 | Non-Commercial Only | Rashkin et al. / Meta AI (Human crowdsource workers) | 7,976 | 609,840 | 382,104 |
| **Authored Conversational Seeds** | Proprietary / Custom | Full Internal Commercial | Authored conversational repairs, everyday check-ins, Hinglish chat, STT mishearing handlers | 796 | 134,280 | 114,620 |
| **E.D.I.T.H. Personality Curriculum** | Proprietary / Custom | Full Internal Commercial | Handcrafted master persona dialogues across 7 behavioral dimensions | 57 | 16,837 | 14,230 |
| **Total Ingested Corpus** | — | **Segregated Partitions** | **100% Human-Authored Organic** | **23,840** | **2,903,347** | **1,496,366** |

---

## 3. Licensing Partitions & Compliance Audit

To strictly honor intellectual property and open-source licenses, the corpus is partitioned into two distinct profiles:

```
Total SFT Corpus: 23,840 Dialogues (2,903,347 Tokens)
├── Commercial-Compatible Partition: 15,864 Dialogues (66.54%)
│   ├── Databricks Dolly-15k (15,011 records, CC BY-SA 3.0)
│   ├── Authored Conversational Seeds (796 records, Internal/Custom)
│   └── E.D.I.T.H. Personality Curriculum (57 records, Internal/Custom)
└── Non-Commercial Research Partition: 7,976 Dialogues (33.46%)
    └── Empathetic Dialogues (7,976 records, CC BY-NC 4.0)
```

- **Attribution Obligation**: In accordance with Section 3(a)(1) of CC BY-SA 3.0, attribution is maintained for Databricks Dolly-15k.
- **Share-Alike Obligation**: Derivative weights incorporating Dolly-15k are tracked for reciprocal licensing if publicly distributed.
- **Non-Commercial Restriction**: Empathetic Dialogues are restricted strictly to research and internal evaluation. Commercial deployment profiles can cleanly filter out the 7,976 records while retaining 15,864 high-quality instruction records.

---

## 4. Dataset Splits & Data Hygiene

To guarantee unbiased evaluation and prevent data leakage, the 23,840 records were partitioned using a fixed random seed (`seed=42`) into three disjoint sets:

| Split | Records | % of Dataset | Mean Sequence Length | Supervised Assistant Tokens |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | 21,456 | 90.0% | 121.8 tokens | 1,346,366 |
| **Validation** | 1,192 | 5.0% | 121.6 tokens | 74,820 |
| **Test (Held-Out)** | 1,192 | 5.0% | 122.1 tokens | 75,180 |

### Leakage and Deduplication Verification:
- **Exact-Match De-duplication**: 0 duplicates found across splits.
- **Cross-Split Leakage**: 0 overlap between `train`, `val`, and `test` queries.
- **Validation Isolation**: Neither validation nor test records were included in pretraining or SFT gradient updates.

---

## 5. Sequence Formatting & Target Token Masking

### Formatting Schema:
Every conversation follows a structured delimiter format that conditions the model on emotional mood while preserving natural text boundaries:

```
<mood>{mood}</mood><user>{prompt}</user><edith>{response}</edith><eos>
```

Valid moods include:
- `calm_focused` (technical queries, factual answers)
- `witty_playful` (banter, teasing, casual greetings)
- `warm_empathic` (distress, venting, personal challenges)
- `patient_reassuring` (confusion, frustration, mistakes)

### Target Loss Masking (`y = -100`):
In causal language modeling without masking, the loss is computed over all tokens, which teaches the model to predict user prompts and leads to "echoic repetition" (e.g., copying the user's question instead of answering).

In V15.3 SFT:
1. `x[t]` = `token_ids[t]`
2. `y[t]` = `token_ids[t+1]`
3. For all positions `t` where `x[t]` belongs to `<mood>...</mood><user>...</user><edith>`, `y[t]` is set to `-100`.
4. Only positions where the model generates `{response}</edith><eos>` have active cross-entropy loss.

```
Token:   <mood> calm_focused </mood> <user> Explain recursion </user> <edith> Recursion is... </edith> <eos>
Input:   [16380,  ...,        16381,  16382, ...,             16383, 16384,  ...,          16385, 2]
Target:  [ -100,  ...,        -100,   -100,  ...,             -100,  tok_0,  ...,          16385, 2]
                                                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                                                     Only assistant tokens supervised
```

- **Target Token Density**: On average, **62.75 tokens** per sequence are supervised assistant tokens, totaling **1,346,366 active prediction targets** per full epoch across the training split.
- **Tensor Cache Storage**: Pre-tokenized padded tensors are cached at `data_pipeline/v15/processed_sft/train_sft_cache_512.pt` (21,456 $\times$ 512 `int16` pairs, ~44MB) enabling zero data-loading latency during GPU training.

