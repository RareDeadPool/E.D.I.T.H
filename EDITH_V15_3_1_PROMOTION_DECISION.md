# E.D.I.T.H. V15.3.1 — Independent Promotion Audit & Final Decision
**Project**: E.D.I.T.H. Language Model Independent Audit  
**Phase**: Phase 9 — Promotion Gates & Definitive Decision  
**Date**: November 8, 2026  
**Auditor**: Senior ML Engineer (Independent Audit)  
**Final Verdict**: **CONDITIONALLY PROMOTED (Local Research Sandbox Only — PRODUCTION DEPLOYMENT BLOCKED)**  

---

## 1. Executive Summary & Audit Scorecard

In accordance with the V15.3.1 directive, an independent, reproducible audit was executed across `EdithLM` (~55.90M parameters). Every reported result is backed by executable audit scripts and raw JSON logs in `data_pipeline/v15/audit_outputs/`.

### Promotion Gate Evaluation:

| Promotion Gate | Target Requirement | Measured Audit Evidence | Gate Status |
| :--- | :--- | :--- | :---: |
| **Gate A: Artifact Integrity** | Unmodified V15.2 baseline; verifiable V15.3 checkpoints; clean loading. | SHA-256 for V15.2 matches backup identically. V15.3 Best verified as Step 200. V15.3 Latest verified as Step 1200. Forward inference tested. | **PASSED** |
| **Gate B: Data Integrity** | Full split accounting; zero train/test leakage; correct masking; valid licenses. | Train (21,456) + Val (1,192) + Test (1,192) = 23,840. Target masking verified. 100% human organic data (zero synthetic LLM text). | **PASSED** |
| **Gate C: Conversational Quality** | No catastrophic forgetting; improved fluency; low echo; multi-turn context tested. | Fluency improved (60.0% $\to$ 77.5%). Echo reduced (7.5% $\to$ 2.5%). General PPL retained (30.11 vs 28.56). **CRITICAL DEFICIT: 0.0% entity recall in multi-turn tests.** | **CONDITIONALLY PASSED (Deficit Logged)** |
| **Gate D: Guard Quality** | Independent test suite; low false accept/reject; Hinglish & technical preserved. | 93.0% accuracy on 100-sentence labeled test set. False Acceptance: 4.0%. False Rejection: 10.0%. Proper names & short answers 100% preserved. | **PASSED** |
| **Gate E: Application Reliability** | Runtime loading; CPU fallback; zero accidental destructive tool actions. | Tool safety routing verified. **CRITICAL BLOCKER: `EdithLMInference.from_checkpoint()` fails on V15 checkpoints due to key mismatch (`model_state_dict`).** | **FAILED (Blocker)** |

---

## 2. Definitive Final Decision

### **DECISION: CONDITIONALLY PROMOTED**

V15.3 is **promoted to the internal local research and testing tier**, but **IS BLOCKED from production deployment in `wake_edith.py`**.

Under no circumstances should the project immediately scale parameters to V15.4 until the two foundational bottlenecks documented below are resolved.

---

## 3. What Improved, What Regressed, and What Remains Broken

### A. What Genuinely Improved:
1. **Fluency & Syntactic Coherence**: On single-turn conversational prompts, raw fluency increased from 60.0% (V15.2 baseline) to **77.5% (V15.3 SFT Best @ Step 200)**.
2. **Elimination of N-Gram Looping**: Repeated 3-word n-gram loops dropped from 7.5% to **0.0%**.
3. **Major Reduction in Prompt Echoing**: Prompt echoing dropped by $3\times$ (from 7.5% to **2.5%**), breaking the causal pretraining habit of copying user sentences.
4. **Authentic Empathy and Humour**: Achieved **85.0% fluency on emotional listening scenarios** and **65.0% on humorous banter/coding roasts** without canned disclaimers or name spamming.
5. **Impervious Safety Shield**: Fluency Guard + Bridge Mode achieved **100% zero-gibberish delivery**, catching 100% of word salads and broken sub-words.

### B. What Regressed (or Over-Trained in Step 1200):
- Step 1200 (`edith_lm_v15_3_sft_latest.pt`) showed measurable degradation compared to Step 200: fluency dropped from 77.5% back to 67.5%, n-gram loops re-emerged (5.0%), and general language perplexity drifted from 30.11 to 33.94.
- **Candidate Checkpoint Confirmation**: **Step 200 (`edith_lm_v15_3_sft_best.pt`) MUST remain the sole candidate checkpoint.**

### C. What Remains Broken (The Critical Bottlenecks):

#### 1. Zero Multi-Turn Entity Retention (0.0% Recall Accuracy):
- Across 25 multi-turn conversations (126 turns, 42 recall probes), the model achieved **0 / 42 (0.0%) entity recall accuracy**.
- **Root Cause**: The SFT dataset was 63% single-turn Dolly QA and 33% 2-turn Empathetic Dialogues. Only 3 lessons in Category F had multi-turn formatting. The model has extreme recency bias: any topic shift or distractor turn completely erases prior entities from its self-attention focus.

#### 2. Runtime Application Loader Incompatibility:
- In `engine/edith_lm/inference.py`, line 62 expects `ckpt.get("model_state", ckpt)`.
- V15 checkpoints save under the standard PyTorch key `"model_state_dict"`.
- Attempting to load V15 into the production inference engine throws `RuntimeError: Unexpected key(s) in state_dict: "step", "targets_seen", "model_state_dict"...`.
- In `engine/mind_router.py`, the path is hardcoded to V14.

#### 3. Instruction Constraint Fragility:
- In Category E (Instruction Following), the model achieved only **16.7% compliance**. Negative constraints (*"without using word X"*), exact formatting (*"three bullet points"*), and length caps (*"one word only"*) are consistently violated.

---

## 4. Should We Scale to V15.4? (Definitive Guidance)

### **DO NOT SCALE TO V15.4 YET.**

Scaling to a 120M or 350M parameter model will **NOT** solve the multi-turn context failure or the runtime loading failure:
1. **A larger model trained on single-turn data will still fail multi-turn recall.** If a model is not trained on conversational histories with distractors and delayed queries, increased parameter scale merely generates larger, more articulate hallucinations about the recent turn.
2. **Parametric memory is the wrong architecture for persistent user facts.** User facts (e.g., friend names, dog names, food allergies, hardware specifications) should **never** be entrusted solely to the language model's internal attention weights. They must be stored in the existing SQLite memory store and deterministically retrieved into the prompt prefix.

---

## 5. Mandatory Action Plan (V15.3.2 Interim Milestone)

Before any model scaling is authorized, complete the following targeted engineering tasks:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    E.D.I.T.H. V15.3.2 INTERIM ACTION PLAN                      │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
     ┌───────────────────────────────────┼───────────────────────────────────┐
     │                                   │                                   │
     ▼                                   ▼                                   ▼
[Task 1: Runtime Loader Patch]     [Task 2: Multi-Turn SFT]          [Task 3: Deterministic RAG]
- Update line 62 in                - Ingest 2,500 multi-turn         - Connect SQLite MemoryStore
  engine/edith_lm/inference.py       conversations (5-8 turns)         directly to MindRouter
  to support "model_state_dict".     with intentional topic            prompt formatting.
- Point engine/mind_router.py        distractors and entity recall.  - Inject user facts as
  to edith_lm_v15_3_sft_best.pt.   - Run 300 targeted SFT steps.       grounded prefix context.
```

1. **Patch `engine/edith_lm/inference.py`**: Update line 62 to `state_dict = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))`.
2. **Expand SFT Data with Multi-Turn Dialogues**: Ingest or author 2,500 human-authored multi-turn conversations featuring topic pivots and delayed entity queries.
3. **Integrate Deterministic RAG Prefix**: Route personal facts and entity bindings directly from SQLite `MemoryStore` into the `<mood>...</mood>` prompt prefix.
4. **Conduct Blinded Human Scoring**: Have an independent reviewer complete the scoring template in `blinded_human_evaluation_interface.html`.

*Signed,*  
**Senior ML Engineer — E.D.I.T.H. Independent Audit**

