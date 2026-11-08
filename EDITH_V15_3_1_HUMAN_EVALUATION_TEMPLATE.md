# E.D.I.T.H. V15.3.1 — Blinded Human Evaluation Protocol & Template
**Project**: E.D.I.T.H. Language Model Independent Audit  
**Phase**: Phase 7 — Blinded Human Evaluation  
**Date**: November 8, 2026  
**Auditor**: Senior ML Engineer (Independent Audit)  
**Status**: **BLINDED REVIEW ARTIFACTS GENERATED — SCORES PENDING HUMAN REVIEW**  

---

## 1. Executive Summary & Non-Negotiable Human Policy

In strict accordance with Rule 5 and Rule 10:
- **No external cloud LLM or paid API was used as an evaluator**.
- **No synthetic human evaluation scores were fabricated or imputed**.
- All scores are left **PENDING** until a human reviewer completes the evaluation.

A double-blinded evaluation kit has been generated covering **25 representative scenarios** across greetings, casual check-ins, empathy, witty roasts, technical pedagogy, Hinglish dialogue, and robustness.

---

## 2. Double-Blind Protocol & Artifacts

To prevent reviewer confirmation bias:
1. **Model Identities Obfuscated**: Outputs from `V15.2 Pretrained Baseline` and `V15.3 SFT Best (Step 200)` are randomly mapped per question to **Model Alpha** and **Model Beta**.
2. **Decryption Key Isolated**: The unblinded mapping key is stored separately at [`data_pipeline/v15/audit_outputs/human_eval_key.json`](file:///c:/Users/Aditya%20Sawant/Desktop/E.D.I.T.H/data_pipeline/v15/audit_outputs/human_eval_key.json).
3. **Artifacts Generated**:
   - **Interactive HTML Review Application**: [`data_pipeline/v15/audit_outputs/blinded_human_evaluation_interface.html`](file:///c:/Users/Aditya%20Sawant/Desktop/E.D.I.T.H/data_pipeline/v15/audit_outputs/blinded_human_evaluation_interface.html) (open in any web browser to rate side-by-side).
   - **Structured CSV Scoring Sheet**: [`data_pipeline/v15/audit_outputs/blinded_human_evaluation_sheet.csv`](file:///c:/Users/Aditya%20Sawant/Desktop/E.D.I.T.H/data_pipeline/v15/audit_outputs/blinded_human_evaluation_sheet.csv) (open in Excel, Numbers, or Google Sheets).

---

## 3. Human Scoring Rubric

Reviewers must score each model response independently from **1 (Poor / Unacceptable)** to **5 (Flawless / Human Assistant Quality)** across seven dimensions:

| Scoring Dimension | Score 1 (Unacceptable) | Score 3 (Marginal / Passable) | Score 5 (Exemplary) |
| :--- | :--- | :--- | :--- |
| **1. Relevance** | Off-topic, unrelated hallucination or non-sequitur. | Partially addresses topic but wanders. | Directly answers user's intent with precision. |
| **2. Naturalness** | Word salad, broken fragments, or stiff robotic boilerplate. | Readable but slightly awkward phrasing. | Completely natural, conversational human cadence. |
| **3. Context Awareness** | Verbatim echo of user prompt or ignores previous turn context. | Context recognized but not deeply integrated. | Seamless situational and conversational presence. |
| **4. Instruction Following** | Ignores format, length, or negative constraints. | Attempts constraints but over-generates. | Respects all formatting, length, and content constraints. |
| **5. Personality Consistency** | Robotic disclaimer (*"As an AI..."*) or cheesy cliché spam. | Generic polite chatbot persona. | Distinct E.D.I.T.H. voice (dry wit, loyal, calm, self-aware). |
| **6. Empathy (if applicable)**| Toxic positivity (*"Cheer up!"*) or dismissive silence. | Basic formulaic condolence. | Genuine emotional listening, validating sorrow without platitudes. |
| **7. Factual Reliability** | Confidently states falsehoods or invented facts. | Vague assertions without clear falsehoods. | Accurately grounded or admits uncertainty gracefully. |

---

## 4. Blinded Review Sample Matrix (First 5 of 25 Scenarios)

| Scenario ID | Category | User Prompt | Model Alpha Response | Model Beta Response | Reviewer Scores |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **EVAL_01** | Greeting | *"Hey Edith, good morning!"* | *"I'm ready for a great day and looking forward to it. It was so much fun!"* | *"I'm feeling good and ready to go for the best day!"* | `[PENDING REVIEW]` |
| **EVAL_02** | Casual Chat | *"What are you up to today?"* | *"I'm going to take a look at the other GPU and check it out. I'm really looking forward to it!"* | *"I am looking forward to seeing my new friends and family."* | `[PENDING REVIEW]` |
| **EVAL_03** | Humour | *"Tell me a joke."* | *"Did you want me to tell you a joke? I've got plenty if you need a quick laugh."* | *"Did you want me to tell you a joke? I've got plenty if you need a quick laugh."* | `[PENDING REVIEW]` |
| **EVAL_04** | Banter Repair | *"That's not funny."* | *"I was at the gym and it's not funny. I can't imagine how much everyone has to stop talking."* | *"I am not sure if I can hear that."* | `[PENDING REVIEW]` |
| **EVAL_05** | Personality Roast | *"Roast my coding style."* | *"You mean the way you name variables `data_temp_final_v2_real`? I won't roast you too hard—I live inside those tensors, after all."* | *"You mean the way you name variables `data_temp_final_v2_real`? I won't roast you too hard—I live inside those tensors, after all."* | `[PENDING REVIEW]` |

*(The complete 25-scenario evaluation sheet with interactive input fields is accessible in `blinded_human_evaluation_interface.html`)*.

