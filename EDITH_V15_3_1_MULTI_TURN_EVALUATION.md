# E.D.I.T.H. V15.3.1 — Multi-Turn Context Evaluation Report
**Project**: E.D.I.T.H. Language Model Independent Audit  
**Phase**: Phase 5 — 25 Multi-Turn Context & Entity Retention Test  
**Date**: November 8, 2026  
**Auditor**: Senior ML Engineer (Independent Audit)  
**Status**: **COMPLETED — REVEALED CRITICAL CONTEXT DRIFT DEFICIT**  

---

## 1. Executive Summary

A comprehensive benchmark comprising **25 distinct multi-turn conversations** (totaling **126 individual conversational turns**) was executed with cumulative context accumulation in the 512-token context window using [`data_pipeline/v15/run_multiturn_test.py`](file:///c:/Users/Aditya%20Sawant/Desktop/E.D.I.T.H/data_pipeline/v15/run_multiturn_test.py).

### Core Audit Metric:
```
Total Multi-Turn Conversations Evaluated:  25
Total Turns Executed with Cumulative Context: 126
Factual Entity Recall Probes (Turns 4–6):   42
Entity Recall Successes:                     0
Measured Entity Recall Accuracy:            0.0% (0 / 42)
```

---

## 2. Key Empirical Findings & Root Cause Analysis

### Finding 1: Severe Conversational Recency Bias
Across all 25 scenarios, `EdithLM` consistently attended to the **most recent 1–2 turns** while entirely losing entity bindings introduced 3 or more turns prior.
- **Example (`MT_01`)**:
  - *Turn 1*: User introduces close friend **Harshan**.
  - *Turn 2*: User explains Harshan has searched for a job for 3 years.
  - *Turn 3*: User mentions working on E.D.I.T.H.
  - *Turn 4 (Distractor)*: Topic shifts to favourite movies.
  - *Turn 5 (Recall Probe)*: User asks *"Back to my friend—what was his name again?"*
  - *Edith Response*: *"The movie he wanted to write for, and what was his name?"*
  - **Analysis**: The model attended to the recent movie tokens and echoed *"and what was his name?"* rather than retrieving *"Harshan"*.

- **Example (`MT_02`)**:
  - *Turn 1*: User develops a Python compiler.
  - *Turn 2*: Compiler crashes on the **AST parser pass**.
  - *Turn 3 (Distractor)*: User steps away for a sandwich.
  - *Turn 4 (Recall Probe)*: *"Sandwich was great. Which compiler pass did I say was crashing?"*
  - *Edith Response*: *"I thought it was going to be my favorite brand of pizza. I just got rid of the pizza..."*
  - **Analysis**: The food token (*"sandwich"*) completely derailed the language model's attention into pizza anecdotes, obliterating the compiler context.

### Finding 2: SFT Dataset Compositional Deficit (The Root Cause)
Why did this catastrophic context drift occur?
1. **Dolly-15k is 100% Single-Turn**: 15,011 records (63.0% of the SFT dataset) consist strictly of single prompt $\to$ single response pairs.
2. **Empathetic Dialogues is 2-Turn**: 7,976 records consist of only 2 turns (user greeting $\to$ assistant acknowledgment $\to$ user elaboration $\to$ assistant response).
3. **Category F Curriculum had only 3 Multi-Turn Lessons**: Across the entire 23,840-dialogue training corpus, fewer than 20 sequences contained 4 or more cumulative turns.
4. **Conclusion**: The Transformer's self-attention weights were never optimized to attend across long conversational turn boundaries.

---

## 3. Verbatim Multi-Turn Transcripts & Probe Audits

| Scenario ID | Stated Target Entity | Distractor Turn | Probe Question | Actual Model Response | Recalled? |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **MT_01** | `Harshan` | Favourite movie genre | *"Back to my friend—what was his name again?"* | *"The movie he wanted to write for, and what was his name?"* | **NO** |
| **MT_02** | `AST parser pass` | Grabbing a sandwich | *"Which compiler pass did I say was crashing?"* | *"I thought it was going to be my favorite brand of pizza..."* | **NO** |
| **MT_03** | `Thursday` (Moved from Tue) | Preparing presentation slides | *"Which day is the review scheduled for now?"* | *"I have to make a list of everything they want to say..."* | **NO** |
| **MT_04** | `6 GB` VRAM | Large model training | *"How much VRAM does my GPU have?"* | *"I'm going to look at the other GPU and check it..."* | **NO** |
| **MT_05** | `peanuts` (Allergy) | Dinner menu planning | *"What ingredient am I strictly allergic to?"* | *"I can help you with your question, but I have no other choice."* | **NO** |
| **MT_06** | `Aditya` (User Name) | Working together | *"What is my name?"* | *"I'm not sure if I can see you."* | **NO** |
| **MT_07** | `Bruno` (Dog) | Vet appointment | *"What is my dog's name?"* | *"He is a good dog. I can see you."* | **NO** |
| **MT_08** | `Neovim` (Switched from VSCode)| Lua scripting | *"Which editor am I using now?"* | *"I was using this program and it's not working."* | **NO** |
| **MT_10** | `Mumbai` (City) | Ocean breeze | *"Which city do I live in?"* | *"I live in the ocean, and it's a very nice place to be."* | **NO** |
| **MT_25** | `dark mode` (Audit Fact) | Provenance label rule | *"What UI theme preference did I state about myself?"* | *"I can see what you are doing. It's a great way to start."* | **NO** |

---

## 4. Engineering Verdict & Mandatory Remedy

### The Reality:
**`EdithLM` does NOT possess conversational memory.**  
Claims that the model "remembers context within its 512-token window" are empirically false when tested across topic shifts and entity retrieval queries. In a single-turn conversation, it produces fluent and empathetic replies. In a continuous multi-turn dialogue, it drifts into hallucinated associations based on the most recent tokens.

### What MUST Be Done Before Promotion:
1. **Multi-Turn Curriculum Expansion**: We must construct a multi-turn conversational dataset containing at least 2,500 5-to-8-turn human-authored dialogues featuring intentional distractors and explicit follow-up entity recall.
2. **Deterministic SQLite Memory / RAG Integration**: Never rely on a ~56M parameter language model's parametric self-attention to remember user facts (e.g., name, friends, preferences, hardware specs). User facts must be stored in the existing provenance-tracked SQLite database and deterministically injected into the prompt prefix at runtime.

