# E.D.I.T.H. V15.3.1 — Fluency Guard & Bridge Mode Audit Report
**Project**: E.D.I.T.H. Language Model Independent Audit  
**Phase**: Phase 6 — Fluency Guard & Bridge Mode Audit  
**Date**: November 8, 2026  
**Auditor**: Senior ML Engineer (Independent Audit)  
**Status**: **COMPLETED WITH INDEPENDENT LABELED BENCHMARK**  

---

## 1. Executive Summary & Audit Methodology

To eliminate evaluation bias, **Fluency Guard was NOT allowed to judge its own generations**. An independent, ground-truth labeled evaluation dataset comprising **100 test sentences** (50 ground-truth valid across 5 subtypes, and 50 ground-truth invalid across 5 subtypes) was constructed and evaluated using [`data_pipeline/v15/audit_fluency_guard.py`](file:///c:/Users/Aditya%20Sawant/Desktop/E.D.I.T.H/data_pipeline/v15/audit_fluency_guard.py).

### Overall Guard Performance:
```
Total Test Cases Evaluated:       100
Accuracy:                          93.00%
Precision:                         95.74%
Recall:                            90.00%
F1 Score:                          0.9278
False Acceptance Rate (Leakage):    4.00% (2 / 50 bad outputs accepted)
False Rejection Rate (Overkill):   10.00% (5 / 50 valid outputs blocked)
```

---

## 2. Independent Confusion Matrix

| Actual Ground Truth | Guard Accepted (`pred=1`) | Guard Rejected (`pred=0`) | Subtotal | Error Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Valid Output (`label=1`)** | **45 (True Positive)** | **5 (False Negative)** | 50 | False Rejection: **10.0%** |
| **Invalid Output (`label=0`)** | **2 (False Positive)** | **48 (True Negative)** | 50 | False Acceptance: **4.0%** |
| **Total** | 47 | 53 | 100 | Overall Acc: **93.0%** |

---

## 3. Subtype Breakdown & Empirical Strengths

| Subtype Category | Cases | Correct Decisions | Subtype Accuracy | Primary Behavioral Observation |
| :--- | :---: | :---: | :---: | :--- |
| **Word Salad & BPE Fragment** | 10 | **10 / 10** | **100.0%** | **Perfect detection**. Words like *"hunol aioleplay naentation's"* or *"re cur sion"* rejected instantly. |
| **Repetitive N-Gram Loops** | 10 | **10 / 10** | **100.0%** | **Perfect detection**. Identifies 3-word repetition loops immediately. |
| **Incomplete / Cut-off Thoughts** | 10 | **10 / 10** | **100.0%** | **Perfect detection**. Preposition stranding and missing terminal punctuation rejected. |
| **Proper Names & Entities** | 10 | **10 / 10** | **100.0%** | *Aditya, Harshan, Chrome, VS Code, Tony Stark, Bruno, Spotify* all correctly accepted. |
| **Natural English** | 10 | **10 / 10** | **100.0%** | Clean conversational dialogue accepted with 100% reliability. |
| **Short Answers & Fragments** | 10 | **10 / 10** | **100.0%** | Conversational fragments (*"Yes, absolutely"*, *"Paris"*, *"Eight"*, *"Done"*) correctly accepted. |
| **Technical Jargon** | 10 | **9 / 10** | **90.0%** | Vocabulary recognizes *tensors, backpropagation, PyTorch, AST, SwiGLU, RMSNorm*. |
| **Prompt Echo Detection** | 10 | **9 / 10** | **90.0%** | Catches responses that verbatim duplicate user prompts without adding new thought. |
| **Empty / Degenerate Strings** | 10 | **9 / 10** | **90.0%** | Whitespace, standalone punctuation, and single characters cleanly caught. |
| **Casual Hinglish** | 10 | **6 / 10** | **60.0%** | **Area for Improvement**. 4 valid Hinglish sentences were rejected due to out-of-lexicon words. |

---

## 4. Analysis of False Positives and False Negatives

### A. The 2 False Positives (False Acceptance / Leakage):
1. **Case `I_31` (Prompt Echo)**: *"My friend has been unemployed for three years."*
   - *Issue*: Punctuation difference allowed it past the strict substring match when query had a trailing question mark.
   - *Remedy*: Normalize all punctuation prior to echo comparison.
2. **Case `I_48` (Degenerate Character)**: Single letter *"S"*.
   - *Issue*: Single-letter words ('a', 'I') are whitelisted, allowing single-letter noise through if punctuation matches.
   - *Remedy*: Require minimum word count $\ge 2$ unless explicitly recognized as an affirmative/negative word.

### B. The 5 False Negatives (False Rejection / Overkill):
- **All 4 Hinglish False Rejections**: Sentences containing `dost`, `peene`, `chalte`, or `tiring` fell slightly below the 0.80 word validity threshold because these specific words were missing from `HINGLISH_WORDS` in `engine/fluency_guard.py`.
- **1 Technical Jargon Rejection**: Sentence `V_19` (*"The IPC pipe transmits JSON packets between processes"*) was flagged because `IPC` was lowercase-normalized and had a low trigram logprob.
- **Remedy**: Expand `HINGLISH_WORDS` with a 2,000-word conversational Hindi vocabulary and ensure acronyms are whitelisted.

---

## 5. Bridge Mode Audit & Relevance

Bridge Mode operates as an automatic fallback when Fluency Guard rejects a candidate generation.

### Key Audit Findings:
1. **Zero Fabrication**: Bridge Mode returns a dignified conversational pivot (*"I'm right here with you. What can I help you explore or accomplish next?"*). It does **not** hallucinate factual answers or claim certainty on topics it failed to generate.
2. **Relevance vs Generic Filler**: While dignified, the current Bridge Mode uses a single primary fallback string. In multi-turn dialogues, repeated triggerings can feel repetitive if the model fails several times consecutively.
3. **Recommendation**: Implement dynamic context-aware Bridge responses (e.g., technical fallback: *"I hit an internal processing snag on that code snippet. Want to break it down together?"* vs emotional fallback: *"I'm listening closely, take your time."*).

