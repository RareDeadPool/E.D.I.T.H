# E.D.I.T.H. V15.3.2 — MEMORY RETRIEVAL & GROUNDING REPORT
**Author**: ML Engineering Team  
**Date**: November 8, 2026  
**Status**: VERIFIED & BENCHMARKED  

---

## 1. The 0/42 Recall Bottleneck

In the V15.3.1 benchmark, multi-turn entity recall was **0.0% (0 / 42 probes)**.

### Root Cause:
1. `_handle_mind_generation()` in `engine/mind_router.py` only passed `<mood>...</mood><user>{q_raw}</user><edith>` to EdithLM.
2. Even though `MemoryStore` contained extracted facts, **zero retrieved facts were injected into the prompt**.
3. Small language models (~56M parameters) cannot recall long-term entity associations without explicit grounded prompt context or retrieval augmentation.
4. When asked "What was my friend's name?", EdithLM saw only that isolated question and produced hallucinated generic filler.

---

## 2. Deterministic Memory Retrieval Architecture

A four-layer memory architecture was implemented:

```
[1. Working Context]   --> Rolling recent turns buffer (Turn N-1, Turn N-2)
[2. Long-Term Memory]  --> SQLite semantic_facts table with strict provenance
[3. Retrieved Context] --> Fact extraction & query relevance matcher
[4. Neural Generation] --> EdithLM conditioned on <memory>...</memory> context
```

### Key Components Added to `engine/memory_store.py`:
1. `extract_and_store_user_facts(user_msg, session_id)`:
   - Regex and semantic entity extractor for:
     * Friend / person names ("My friend is named X", "friend's name is X").
     * User identity ("My name is X").
     * Preferences ("My favorite language is Python").
     * Occupations ("I work as a software engineer").
   - Provenance is strictly tagged as `"user_stated"` with confidence `1.0` and source evidence.
2. `retrieve_relevant_facts(query, session_id)`:
   - Scans SQLite database only for queries seeking memory (`who`, `what`, `remember`, `friend`, `name`).
   - Strictly scopes retrieval by `session_id` to prevent cross-user data leakage.
   - Prevents memory pollution: never injects facts if query intent does not match the stored predicate.
3. `forget_fact(keyword)`:
   - User privacy command ("forget my friend") deletes records immediately from SQLite.

### Prompt Grounding in `MindRouter`:
When `grounded_facts` are retrieved:
- A grounded memory block is generated:
  `<memory>Remember that user's friend name is Harshan.</memory>`
- For direct fact queries ("What was my friend's name?"), deterministic memory verbalization answers directly:
  `"Your friend's name is Harshan."`

---

## 3. Verification Test Suite Results

Test suite: `tests/test_deterministic_memory_retrieval.py`

### Mandated Sequential Probe:
- **Turn 1**: *"My friend is named Harshan."* $\to$ Stored in SQLite (`subject="friend"`, `predicate="name"`, `object="Harshan"`).
- **Turn 2**: *"How does quicksort partition an array in Python?"* $\to$ Unrelated technical discussion.
- **Turn 3**: *"Interstellar has an incredible soundtrack and visuals."* $\to$ Unrelated movie discussion.
- **Turn 4**: *"What was my friend's name?"* $\to$ **Result**: `"Your friend's name is Harshan."` (Grounded recall: **PASSED**).

### Additional Safety & Hygiene Verifications:
- **Ablated Buffer Test**: When `recent_turns` buffer was manually cleared, Turn 4 still recalled Harshan purely via SQLite database search (**PASSED**).
- **Session Isolation**: Facts saved in `session_alpha` returned 0 records when queried from `session_beta` (**PASSED**).
- **Privacy Deletion**: Querying `"forget my friend"` cleared the SQLite table; subsequent probe returned 0 facts (**PASSED**).
