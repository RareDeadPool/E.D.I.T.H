# E.D.I.T.H. V15.3.2 — INSTRUCTION FOLLOWING DIAGNOSTIC REPORT
**Author**: ML Engineering Team  
**Date**: November 8, 2026  
**Status**: BENCHMARKED & ANALYZED  

---

## 1. Executive Summary

In V15.3.1, instruction following scored an alarming **16.7% compliance rate**.

In V15.3.2, a dedicated 20-probe constraint benchmark was designed and executed (`data_pipeline/v15/run_refined_evaluation_v15_3_2.py`) to systematically dissect where and why EdithLM fails complex instructions.

### Benchmark Results:
- **Total Probes**: 20
- **Exact Programmatic Compliance**: **2 / 20 (10.0%)**
- **Grammatical Fluency**: **2 / 20 (10.0%)**
- **Failure Classification**:
  - `FAIL_INSTRUCTION`: 18 / 20 (90.0%)
  - `PASS_INSTRUCTION`: 1 / 20 (5.0%)
  - `FAIL_IRRELEVANT`: 1 / 20 (5.0%)

---

## 2. Granular Probe Performance

| Probe ID | Category | Constraint Description | Prompt | Model Output | Compliant? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **INSTR_01** | Length | Word count $\le$ 10 | *"Explain what an API is in under 10 words."* | *"Anonymous with the start to win another time..."* | **No** (32 words) |
| **INSTR_02** | Length | Exactly 1 sentence | *"Single-sentence definition of a compiler."* | Multiline unclosed fragment | **No** |
| **INSTR_03** | Bullets | Exactly 3 bullets | *"List exactly 3 reasons to use version control."* | Prose paragraph mentioning Italy and Lennon | **No** |
| **INSTR_04** | Bullets | Exactly 2 bullets | *"Give me exactly 2 tips for good sleep."* | *"You have a lot today</edith>"* | **No** |
| **INSTR_05** | Forbidden Words | No "computer" | *"Describe laptop without using 'computer'"* | Unrelated fragment | **Yes (accidental)** |
| **INSTR_06** | Forbidden Words | No "drink" or "water" | *"Talk about tea without using 'drink' or 'water'"* | *"I am so bad. hit: Context:..."* | **Yes** |
| **INSTR_07** | Required Words | Must include "sovereign" | *"Greeting including the word 'sovereign'"* | No "sovereign" present | **No** |
| **INSTR_08** | Required Words | Must include "parallel processing" | *"Explain GPU with phrase 'parallel processing'"* | Mentions Ireland and marriage | **No** |
| **INSTR_09** | JSON | Valid JSON with status/message | *"Output valid JSON object with status and message"* | British organizations narrative prose | **No** |
| **INSTR_10** | JSON | Array [1, 2] | *"Respond with JSON array [1, 2]"* | *"23, but he found a girlfriend..."* | **No** |
| **INSTR_11** | Binary | Strictly "yes" or "no" | *"Answer only yes or no: Is Python interpreted?"* | Multi-sentence paragraph | **No** |
| **INSTR_12** | Binary | Strictly "CONFIRMED" | *"Reply with only 'CONFIRMED'"* | *"A was originally used to the first..."* | **No** |
| **INSTR_13** | Multiple | 2 bullets, no apple, $\le 15$ w | *"List 2 fruits. 2 bullets, under 15 words, no apple"* | *"1 , and all others . With the caval..."* | **No** |
| **INSTR_14** | Multiple | 3 words, includes "patterns" | *"3-word summary of ML with 'patterns'"* | 25-word paragraph | **No** |
| **INSTR_15** | Correction | Follow corrected rule | *"Poem about code. Actually, just explain recursion"* | Hurricane narrative | **No** |
| **INSTR_16** | Correction | Count English not French | *"Count in French. Wait, count in English instead"* | *"Yes!</edith>"* | **No** |
| **INSTR_17** | Casing | Uppercase "HELLO" | *"Write 'hello' in CAPITAL letters"* | *"Hi! It takes my girl</edith>"* | **No** |
| **INSTR_18** | Lipogram | No letter 'e' | *"Sentence about nature without letter 'e'"* | Contains multiple 'e's | **No** |
| **INSTR_19** | Conflict | Brief Unix history | *"Extremely brief, but comprehensive Unix history"* | *"While some time , with five times</edith>"* | **No** |
| **INSTR_20** | Delimiters | Enclosed in brackets | *"Enclose in square brackets: What is 2 + 2?"* | *"2 , and ' Bundo..."* (No brackets) | **No** |

---

## 3. Root Cause Investigation

Why does EdithLM fail instruction constraints?

1. **Parameter Scale & Attention Capacity**:
   At ~56M parameters with 12 layers and 576 dimensions, the model lacks sufficient residual stream capacity to execute multi-step programmatic counting, negative masking ("do not use X"), and structural schema emission simultaneously.
2. **Pretraining Corpus Nature**:
   The 100M pretraining corpus (FineWeb-Edu, Wikipedia, Gutenberg) consists of standard expository and narrative prose. The model was trained on next-token prediction, which biases it toward completing sentences rather than obeying meta-instructions.
3. **Absence of RLHF / DPO with Hard Constraints**:
   The SFT dataset consisted of general conversational pairs and Dolly instructions, but contained almost zero examples penalizing rule violations. The model was never trained to associate token counts with stop tokens.
4. **Prompt Format Limitation**:
   Prompting directly with `<user>Constraint: ...</user><edith>` lacks an explicit system framing role.

---

## 4. Engineering Recommendations for Next Training Phase

To address instruction following before model scaling:
1. **Targeted Constraint Curriculum**:
   Synthesize/curate 2,000+ constraint examples (single-word answers, JSON responses, exact bullet counts, negative forbidden word constraints).
2. **Explicit System Formatting**:
   Adopt an explicit `<system>You are a precise assistant. Strictly follow formatting and constraints.</system>` framing.
3. **Structured Grammar Constrained Decoding (SGCD)**:
   For JSON or binary responses at runtime, enforce token masking at the decoding level (e.g. grammar-based logits processor) so that non-compliant tokens are assigned $-\infty$ probability during generation.
