# Project E.D.I.T.H. V15 — Strict Data Licenses & Provenance Directory

**Status:** Updated for Gate 1B Strict Organic Audit  
**Author:** ML Engineering & Compliance Team, Project E.D.I.T.H.  
**Mandate:** 100% human-authored source material. Zero synthetic LLM-generated text (Option A). Strictly documented provenance, licensing, and attribution obligations.

---

## 1. Executive Summary & Policy Boundary

In accordance with Aditya's V15 directives:
1. **Zero LLM-Generated Text:** All synthetic corpora (TinyStories, UltraChat, Alpaca, ChatDoctor, etc.) are strictly barred.
2. **Web Data Scrutiny:** `FineWeb-Edu` is **excluded from pretraining** because it is derived from post-2022 Common Crawl containing widespread web-scraped AI spam, and its educational scoring relies on synthetic LLM annotations (Llama-3-70B).
3. **Commercial vs. Non-Commercial Bifurcation:** Datasets carrying non-commercial restrictions (`CC-BY-NC`) are segregated from commercial-permissive data (`Public Domain`, `Apache 2.0`, `CC-BY`, `CC-BY-SA`).
4. **Attribution & Share-Alike Integrity:** Ingestion scripts must preserve author metadata and license provenance for all training shards.

---

## 2. Dataset Master Registry: Commercial vs. Non-Commercial Separation

### Category A: Commercial-Compatible Datasets (Permissive & Share-Alike)

| Dataset ID | Name & Origin | License | Commercial Permitted? | Attribution Obligation | Share-Alike Obligation | Estimated Clean Tokens |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `V15-DS-01` | **E.D.I.T.H. Hand-Authored Persona Seed** | Proprietary | **Yes** | Internal | No | ~1.5M |
| `V15-DS-02` | **Project Gutenberg (Pre-1929 US Public Domain)** | Public Domain | **Yes** | Notice required if PG logo/header kept; **None** if raw author text extracted | No | ~180M |
| `V15-DS-03` | **English Wikipedia & Simple English Wikipedia** | CC-BY-SA 4.0 / CC-BY-SA 3.0 | **Yes** | Yes (Wikimedia contributors URL attribution) | **Yes** (Model derived text under CC-BY-SA) | ~250M |
| `V15-DS-04` | **Databricks Dolly-15k** | CC-BY-SA 3.0 | **Yes** | Yes (Databricks attribution notice) | **Yes** | ~4.5M |
| `V15-DS-05` | **OpenAssistant Conversations (oasst1 human)** | Apache 2.0 | **Yes** | Yes (Include Apache 2.0 copyright notice) | No | ~18M |
| `V15-DS-06` | **Tatoeba Project English-Hindi Sentences** | CC-BY 2.0 FR | **Yes** | Yes (Link to Tatoeba.org & contributor list) | No | ~12M |
| `V15-DS-07` | **L3Cube HinglishChat Subset** | CC-BY-SA 4.0 | **Yes** | Yes (L3Cube-Pune citation) | **Yes** | ~2.0M |
| `V15-DS-08` | **DWYL / SCOWL English Lexicon** | Public Domain / MIT | **Yes** | None for word lists | No | 370k entries |

---

### Category B: Non-Commercial Research & Personal Evaluation Datasets

> [!CAUTION]
> If these datasets are included in pretraining or fine-tuning, the resulting model weights inherit non-commercial restrictions and **cannot be commercialized** or deployed in commercial services without explicit relicensing.

| Dataset ID | Name & Origin | License | Commercial Permitted? | Attribution Obligation | Share-Alike Obligation | Estimated Clean Tokens |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `V15-DS-NC01` | **EmpatheticDialogues** (Meta AI) | CC-BY-NC 4.0 | **NO (Non-Commercial Only)** | Yes (Rashkin et al., 2019) | No | ~2.5M |
| `V15-DS-NC02` | **DailyDialog (Cleaned)** | CC-BY-NC-SA 4.0 | **NO (Non-Commercial Only)** | Yes (Li et al., 2017) | **Yes** | ~2.0M |
| `V15-DS-NC03` | **PersonaChat (ConvAI2)** | CC-BY-NC 4.0 | **NO (Non-Commercial Only)** | Yes (Zhang et al., 2018) | No | ~3.5M |

---

## 3. Strict Source-Level Verification & Restrictions

### 3.1 Project Gutenberg Ebook Restrictions (`V15-DS-02`)
- **Legal Reality:** While the underlying literary works (Shakespeare, Austen, Dickens, Conan Doyle, Twain, etc.) published prior to 1929 are in the worldwide **Public Domain**, the name *"Project Gutenberg"* is a registered trademark. The Project Gutenberg header and footer contain trademark restrictions and donation requests under the "Project Gutenberg™ License".
- **Compliance Invariant:** Ingestion scripts must strictly strip all header lines prior to `"*** START OF THE PROJECT GUTENBERG EBOOK ... ***"` and all footer lines following `"*** END OF THE PROJECT GUTENBERG EBOOK ... ***"`. Only the underlying public domain author's text is tokenized.

### 3.2 Tatoeba Sentence-Level Licensing (`V15-DS-06`)
- **Legal Reality:** Sentences on Tatoeba.org are released under Creative Commons Attribution 2.0 France (CC-BY 2.0 FR). While some individual contributors release sentences into the public domain (CC0), the master download archive is governed by CC-BY 2.0 FR.
- **Compliance Invariant:** A manifest file `data/tatoeba_attribution.txt` must accompany the processed dataset documenting Tatoeba source URLs and contributor IDs.

### 3.3 Wikipedia Share-Alike & Attribution (`V15-DS-03`)
- **Legal Reality:** Text is licensed under CC-BY-SA 4.0 and GFDL. Training on Wikipedia is fair use in the US, but model checkpoints distributed openly that quote substantial article excerpts may have attribution requirements.
- **Compliance Invariant:** All Wikipedia dumps used (Simple English Wikipedia and Wikitext-103) retain their raw dump revision IDs and Wikimedia author attribution links.

### 3.4 OpenAssistant Human-Only Tree Filtering (`V15-DS-05`)
- **Provenance Reality:** The OASST1 dataset contains human volunteer conversations. However, some trees in later versions (OASST2) incorporated synthetic bot answers.
- **Compliance Invariant:** Filter strictly with `role == "prompter"` or (`role == "assistant"` AND `synthetic == False`). Discard all synthetic branches.

### 3.5 FineWeb-Edu Exclusion Justification
- **Audit Findings:** Hugging Face's FineWeb-Edu is derived from Common Crawl dumps (2020–2024). Post-2022 Common Crawl is known to have between 10% and 30% machine-generated web text. Furthermore, FineWeb-Edu's educational score threshold is determined entirely by an automated Llama-3-70B-Instruct evaluator.
- **Compliance Determination:** In accordance with the 100% organic human requirement, **FineWeb-Edu is completely excluded**. We replace its factual knowledge allocation with verified human-authored Wikipedia and historical public domain nonfiction.

---

## 4. Pipeline Provenance Tracking

Every processed shard written to `data_pipeline/v15/processed/` must contain a companion manifest `shard_provenance.json` recording:
1. `source_dataset_id`: (e.g. `V15-DS-02`)
2. `source_url_or_hash`: SHA-256 hash of original raw archive.
3. `license_type`: (e.g. `Public Domain`, `CC-BY-SA 3.0`)
4. `commercial_eligibility`: Boolean (`true` / `false`)
5. `clean_token_count`: Exact integer token count.
6. `filter_pass_rate`: Percentage of documents passing PII, deduplication, and quality filters.
