# E.D.I.T.H. V15.3.2 — EVALUATION METHODOLOGY REPORT
**Author**: ML Engineering Team  
**Date**: November 8, 2026  
**Status**: UPGRADED & BENCHMARKED  

---

## 1. Weakness of Prior Evaluation Methodology

In V15.3.1, model performance was evaluated using a simplistic binary `PASS_RAW` vs `GUARD_FALLBACK` classification, where passing was heavily predicated on whether `FluencyGuard.inspect()` detected valid English syntax.

### The Fluency Fallacy:
A language model can output grammatically pristine, beautifully punctuated English that completely fails the user's intent:
- *User*: "Can you launch Chrome for me?"
- *Model*: "The gentle afternoon breeze drifted across the valley."
Under the old metric, this counted as a pass because every word was in the dictionary and sentences ended with punctuation. In reality, it was completely unresponsive and irrelevant.

---

## 2. Refined Multi-Dimensional Evaluation Framework

In V15.3.2, evaluation separates independent qualities into dedicated, orthogonal metrics:

1. **Word Validity**: Fraction of tokens resolving to recognized dictionary words (catches subword shattering).
2. **Grammatical Fluency**: Sentence closure, punctuation syntax, no dangling conjunctions.
3. **Prompt Relevance**: Semantic alignment and topic overlap between response and prompt.
4. **Instruction Compliance**: Exact programmatic adherence to formatting, length, and constraint rules.
5. **Context Retention**: Accurate referencing of prior entities or memory facts.
6. **Repetition Penalty**: Detection of repeated n-grams or looped phrases.
7. **Prompt Echoing**: Detection of prompt verbatim repetition.

---

## 3. Standardized Mutually Exclusive Outcome Labels

Every model generation is assigned exactly one outcome label based on strict hierarchical checks:

| Outcome Label | Definition | Evaluation Criterion |
| :--- | :--- | :--- |
| **`PASS_RELEVANT`** | Fluent, on-topic, responsive answer | Passes FluencyGuard, semantic relevance > threshold, no echoes. |
| **`PASS_INSTRUCTION`** | Complies with all explicit constraints | Passes programmatic constraint validator + fluent. |
| **`PASS_CONTEXT`** | Accurate entity recall / multi-turn coherence | Recalls grounded entity or maintains dialogue context. |
| **`FAIL_IRRELEVANT`** | Fluent English but completely off-topic | Fluent text with zero semantic relevance to user prompt. |
| **`FAIL_INSTRUCTION`** | Violates explicit instruction constraint | Fails programmatic constraint (word count, bullets, forbidden words). |
| **`FAIL_CONTEXT`** | Misses or hallucinates context / entity | Fails to recall grounded entity from prior turns. |
| **`FAIL_ECHO`** | Repeats user prompt back to user | Prompt overlap > 85% with no original content. |
| **`FAIL_GIBBERISH`** | Unintelligible subword shattering | Word validity < 70% or non-lexical token soup. |
| **`GUARD_FALLBACK`** | Raw output rejected; safe fallback invoked | Output Contract / FluencyGuard blocked output. |

---

## 4. Double-Blind Human Evaluation Restoration

In V15.3.1, the evaluation key was inadvertently committed into `data_pipeline/v15/audit_outputs/human_eval_key.json`, compromising reviewer blinding.

### Restoration in V15.3.2:
1. `human_eval_key.json` was purged from Git history and untracked.
2. A newly randomized mapping was generated with an independent non-static seed.
3. The decryption key was exported strictly outside Git to:
   `C:\Users\Aditya Sawant\Desktop\EDITH_LOCAL_SECRETS_V15_3_2\human_eval_key_v15_3_2.json`
4. The public HTML review interface (`blinded_human_evaluation_interface.html`) and CSV sheet (`blinded_human_evaluation_sheet.csv`) display only anonymous `Model Alpha` and `Model Beta` labels.
5. Genuine double-blind evaluation is fully restored.
