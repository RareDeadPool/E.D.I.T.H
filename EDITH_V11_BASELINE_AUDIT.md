# E.D.I.T.H. V11 Baseline Audit & V10 Verification Report

**Author:** Antigravity (Advanced Agentic Intelligence Core)  
**Date:** October 30, 2026  
**Scope:** Independent forensic audit of V10 Transformer architecture, production checkpoints, training/val/test splits, tokenization, multi-head runtime integration, and root-cause analysis of the slot-accuracy regression.

---

## Executive Summary

A comprehensive, ground-truth audit of E.D.I.T.H. V10 was conducted across all model files, datasets, tokenizers, runtime engines, and evaluation suites. 

### Key Findings at a Glance:
1. **Model Parity & Identity**: The running application loads `models/checkpoints/edith_ann_best.pt` with ONNX Runtime acceleration (`edith_ann_best.onnx`). Numerical parity between PyTorch and ONNX is exact (maximum logit error $< 7.63 \times 10^{-6}$).
2. **Architecture Discrepancies**: While parameter count (3,522,063 trainable) and layers (4 layers, 8 heads, $d_{\text{model}}=256$, $d_{\text{ff}}=1024$) match report, the model's configured sequence length is **64 tokens**, not 32. However, the training corpus had an average sequence length of only **4.4 tokens** (max 14 tokens), meaning the model never observed longer natural inputs during training.
3. **Severe Slot Precision Collapse (88.90% Slot Accuracy)**: While token-level slot accuracy is 88.90% on the test split, the **entity span precision collapses to 44.08%** (Macro Slot F1 is 68.15%, Entity Span F1 is 57.68%). Over **55.9% of all predicted entity spans on unseen phrasings are false positives**! The model hallucinates entities onto common English words (`O -> I-QUERY`, `O -> I-EXPR`, `O -> B-QUERY`, `O -> B-PID`, `O -> B-APP`).
4. **Vocabulary & UNK Crisis**: The vocabulary is restricted to 832 tokens with a word-based regex tokenizer. While training data had **0.00% UNK tokens**, **98.29% of test samples contain at least one `<unk>` token**, 75% of technical terms are `<unk>`, 63.2% of Hinglish words are `<unk>`, and 100% of unseen applications and entities are `<unk>`.
5. **Dataset Memorization vs Generalization**: Out of 13,600 training samples, there are only **2,899 unique sentences** (4.7 duplicate copies per template). Although cross-split exact string leakage is 0, the model memorized token position correlations from repeated templates rather than learning true entity boundary semantics.
6. **"18 Tests / 150 Scenarios" Explained**: The 18 reported tests correspond to 18 unit test functions in `tests/test_v8_cognitive_scenarios.py`. The 150 scenarios are hardcoded regex patterns in `CognitiveScenarioEngine` that bypass the neural model entirely, returning pre-written template paragraphs.
7. **Discriminative Classifier vs Conversational Generator**: The 5-head Transformer is strictly a discriminative NLU model. Open-ended conversational intelligence cannot be achieved by merely boosting classifier accuracy; conversational response generation requires dedicated context-aware response synthesis.

---

## 1. Verified Model Identity & Runtime Integration

### 1.1 Checkpoint & Weights Verification
- **Active Checkpoint File:** `models/checkpoints/edith_ann_best.pt`
- **Active ONNX File:** `models/checkpoints/edith_ann_best.onnx`
- **Active Tokenizer File:** `models/checkpoints/tokenizer.json`
- **Runtime Loader:** `engine/inference.py` (`EdithInferenceEngine`) initialized by `engine/edith_brain.py` (`EdithBrain.engine`).
- **ONNX Acceleration:** `EdithInferenceEngine` automatically checks for `edith_ann_best.onnx` with ONNX Runtime CPUExecutionProvider. If available, all forward passes run via ONNX Runtime; PyTorch serves as automatic fallback.

### 1.2 Actual Neural Architecture & Parameter Count
| Architectural Dimension | Reported in V10 Spec | Verified Ground Truth in Code | Status |
| :--- | :--- | :--- | :--- |
| **Model Type** | Custom Transformer Encoder | Custom Transformer Encoder (`EdithNeuralNetwork`) | **Verified** |
| **Encoder Layers** | 4 | 4 | **Verified** |
| **Attention Heads** | 8 | 8 | **Verified** |
| **Hidden Size ($d_{\text{model}}$)** | 256 | 256 | **Verified** |
| **Feedforward Size ($d_{\text{ff}}$)**| 1024 | 1024 | **Verified** |
| **Trainable Parameters** | 3,522,063 | **3,522,063** (Total State Dict: 3,538,447) | **Verified** |
| **Vocabulary Size** | 832 | 832 tokens | **Verified** |
| **Maximum Sequence Length** | 32 tokens | **64 tokens** (`model_config.json`) | **Discrepancy** (Model supports 64, reported 32) |
| **Embedding Dimension** | 256 | 256 | **Verified** |
| **Dropout** | 0.1 | 0.1 | **Verified** |

### 1.3 Five Output Heads Specification
1. **Intent Classification Head**: Linear projection ($256 \to 17$). Softmax over 17 intent categories (`app_control`, `file_operation`, `system_query`, `media_control`, `volume_control`, `browser_action`, `window_management`, `process_management`, `web_search`, `communication`, `reminder_timer`, `system_setting`, `calculator`, `screen_analysis`, `chitchat`, `clarification_response`, `unsupported_ood`).
2. **BIO Entity Slot Tagging Head**: Token-level linear projection ($256 \to 41$). Softmax over 41 BIO tags covering `APP`, `FILE`, `PATH`, `QUERY`, `VALUE`, `SETTING`, `RECIPIENT`, `MESSAGE`, `PID`, `DIR`, `DEST`, `SONG`, `ARTIST`, `PLAYLIST`, `MODIFIER`, `EXPR`, `TIME`, `REMINDER_TEXT`, `PROJECT`.
3. **Action Policy Head**: Linear projection ($256 \to 18$). Softmax over 18 policy action primitives (`open_app`, `close_app`, `find_file`, `delete_file`, `get_system_info`, `exec_play_music`, `exec_pause_music`, `change_volume`, `web_browse`, `manage_process`, `send_whatsapp`, `draft_whatsapp`, `change_setting`, `calculate`, `analyze_screen`, `respond_dialogue`, `request_clarification`, `noop`).
4. **Operational Confidence Head**: Linear projection ($256 \to 1$) followed by Sigmoid. Predicts probability in $[0, 1]$ indicating whether the assistant should execute or seek clarification.
5. **Plan Complexity Head**: Linear projection ($256 \to 2$). Binary classification predicting whether the user command is single-step (0) or requires multi-step planning (1).

### 1.4 PyTorch vs ONNX Numerical Parity
Evaluated on 1,700 held-out test inputs across CPUExecutionProvider and PyTorch 2.x:
- **Intent Logits Max Difference:** $2.38 \times 10^{-6}$
- **Slot Logits Max Difference:** $7.63 \times 10^{-6}$
- **Action Logits Max Difference:** $2.15 \times 10^{-6}$
- **Confidence Output Max Difference:** $5.96 \times 10^{-8}$
- **Complexity Logits Max Difference:** $9.54 \times 10^{-7}$
- **Conclusion:** Parity is within IEEE 754 float32 precision limits; no numerical drift occurs between PyTorch and ONNX inference.

### 1.5 Runtime Interaction: Neural Model vs Deterministic Rules
In `engine/edith_brain.py`:
- **Pre-Neural Bypass**: Before the model is called, `self.dialogue_engine.is_conversational_or_problem()` tests the input against user-profiling keywords and 150 regex patterns from `CognitiveScenarioEngine`. If triggered, it sets `intent="chitchat"`, `action="respond_dialogue"`, and bypasses neural intent classification.
- **Post-Neural Overrides**:
  - `_refine_music_slots`: A regex heuristic strips "play", "put on", "stream", and trailing phrases like "while I work" from extracted entities.
  - `UnderstandingNormalizer`: Injects ambiguity checks and validation error flags. If entities are missing for an action, it overrides the predicted action with `request_clarification`.
  - `PermissionGuard`: Deterministic safety gate. Regardless of model confidence, destructive actions (`delete_file`, `kill_process`) and external messaging (`send_whatsapp`) require explicit user authorization.

---

## 2. Dataset Audit & Cross-Split Leakage Analysis

### 2.1 Dataset Volume and Deduplication
| Split | Total Samples | Unique Utterances | Internal Repetitions | Uniqueness Ratio |
| :--- | :--- | :--- | :--- | :--- |
| **Train (`train_data.json`)** | 13,600 | 2,899 | 10,701 | **21.3%** |
| **Validation (`val_data.json`)** | 2,550 | 839 | 1,711 | **32.9%** |
| **Test (`test_data.json`)** | 1,700 | 665 | 1,035 | **39.1%** |

### 2.2 Leakage Investigation
1. **Exact Text Overlap**: 
   - `Train ∩ Test`: **0 exact sentences** (0.00% exact text leakage).
   - `Train ∩ Val`: **0 exact sentences** (0.00% exact text leakage).
   - `Val ∩ Test`: **0 exact sentences** (0.00% exact text leakage).
2. **Abstracted Syntactic Template Overlap**:
   - Entities were masked with generic slot tokens (`[APP]`, `[FILE]`, `[PID]`, etc.) to evaluate syntactic structure overlap.
   - Distinct templates in Train: 380 templates.
   - Distinct templates in Val: 144 templates.
   - Distinct templates in Test: 347 templates.
   - **Shared Templates between Train and Test**: Only 3 templates (`how much is [EXPR]`, `what is [EXPR]`, `solve [EXPR]`).
   - **Conclusion on Leakage**: The test set was indeed constructed with syntactically varied, unseen phrasing. The drop in test performance is NOT caused by test leakage, but rather by **failure to generalize to unseen syntactic patterns**.

### 2.3 Class Balance in Dataset
- **Train Split**: Exactly 800 samples per intent across all 17 intents ($17 \times 800 = 13,600$). Perfectly balanced across intents.
- **Test Split**: Exactly 100 samples per intent across all 17 intents ($17 \times 100 = 1,700$).
- **Slot Imbalance**: Extreme class imbalance exists at the slot level. The `O` (outside entity) token constitutes >85% of tokens in general natural language, but in the short commands of the training dataset, entity tokens constituted ~60% of all tokens.

---

## 3. Root Cause Analysis: Slot Accuracy Regression (V8: 96.90% $\to$ V10: 88.90%)

The V10 report documented an 8% drop in slot accuracy (88.90%). Our audit reveals that **the actual degradation is vastly worse than reported**:

### 3.1 Verification of Reported vs Granular Slot Metrics
| Metric | Validation Set (Familiar Patterns) | Held-Out Test Set (Unseen Phrasings) |
| :--- | :--- | :--- |
| **Token-Level Slot Accuracy (All)** | 96.93% | **88.90%** (Matches V10 report) |
| **Token-Level Slot Accuracy (Non-O)**| 94.35% | 97.88% |
| **Exact Sequence Match Rate** | 86.67% | **63.94%** (36% of sentences have corrupted slots!) |
| **Entity Span Precision** | **86.78%** | **44.08%** (**55.9% of predicted entities are false positives!**) |
| **Entity Span Recall** | **90.64%** | 83.39% |
| **Entity Span F1** | **88.67%** | **57.68%** (**Severe 31% collapse in F1**) |
| **Macro Slot F1 (Active Slots)** | 86.70% | **68.15%** |

### 3.2 What Caused the Collapse?
The high token accuracy (88.90%) was an artifact of the abundance of `O` tokens. In reality, **the model suffers from massive false-positive entity hallucinations**.

#### Top 10 Slot Confusions on Held-Out Test Data:
1. `O` (Gold) $\to$ `I-QUERY` (Predicted): **156 times**
2. `O` (Gold) $\to$ `I-EXPR` (Predicted): **152 times**
3. `O` (Gold) $\to$ `B-QUERY` (Predicted): **89 times**
4. `O` (Gold) $\to$ `B-PID` (Predicted): **85 times**
5. `O` (Gold) $\to$ `I-PROJECT` (Predicted): **66 times**
6. `O` (Gold) $\to$ `B-APP` (Predicted): **62 times**
7. `O` (Gold) $\to$ `I-SONG` (Predicted): **56 times**
8. `O` (Gold) $\to$ `B-VALUE` (Predicted): **49 times**
9. `O` (Gold) $\to$ `B-EXPR` (Predicted): **39 times**
10. `O` (Gold) $\to$ `B-SONG` (Predicted): **36 times**

#### Weakest Slots on Test:
- `I-PROJECT`: Precision **19.5%**, Recall 66.7%, F1 **30.2%**
- `B-PID`: Precision **20.6%**, Recall 100.0%, F1 **34.1%**
- `B-QUERY`: Precision **22.6%**, Recall 100.0%, F1 **36.9%**
- `I-QUERY`: Precision **24.3%**, Recall 100.0%, F1 **39.1%**
- `I-ARTIST`: Precision **36.4%**, Recall 100.0%, F1 **53.3%**
- `I-SONG`: Precision **39.1%**, Recall 100.0%, F1 **56.2%**

### 3.3 The Core Root Causes:
1. **The `<unk>` Vector Trap**: In training, the vocabulary was 100% complete for the synthetic corpus (0% UNK). In the test set, **19.93% of words were `<unk>`**, and **98.29% of test samples had at least one `<unk>` token**. When the model encountered `<unk>` in positions where entities appeared in the repeated templates, it blindly mapped them to `B-QUERY`, `B-PID`, `I-EXPR`, etc.
2. **Absence of Negative Contexts for Entity Tokens**: In the training data, phrases like "what is", "search for", "find me", "solve" were always immediately followed by entities (`B-EXPR`, `B-QUERY`). Ordinary conversational words following these patterns were blindly tagged as entity content.
3. **Loss Function Asymmetry**: `EdithMultiTaskLoss` applied unweighted cross-entropy on slots with `slot_weight=2.0`. It did not penalize entity false positives more than false negatives, driving the model towards over-prediction.

---

## 4. Tokenizer & Sequence Representation Audit

### 4.1 Tokenizer Implementation Details
- **Type:** Word-level regex tokenizer with simple rule-based morphological fallback (`_resolve_token_id`).
- **Vocabulary Size:** 832 tokens.
- **Special Tokens:** `<pad>=0`, `<unk>=1`, `<bos>=2`, `<eos>=3`.

### 4.2 Out-Of-Vocabulary (OOV) Rate Across Splits & Domains
- **Training Set:** Total words = 60,410 | UNK words = **0 (0.00%)** | Samples with UNK = **0 (0.00%)**.
- **Validation Set:** Total words = 13,267 | UNK words = **1,430 (10.78%)** | Samples with UNK = **1,021 (40.04%)**.
- **Test Set:** Total words = 9,231 | UNK words = **1,840 (19.93%)** | Samples with UNK = **1,671 (98.29%)**.

#### UNK Rate on Natural Language Domains:
| Domain / Category | Sample Test Expressions | UNK Token Rate |
| :--- | :--- | :--- |
| **Unseen Applications** | "Slack", "Discord", "Notion", "VS Code", "Figma", "Docker", "Postman", "Obsidian" | **100.0%** (8/8 apps UNK) |
| **Unseen Entities / Contacts** | "Sakshi", "Aditya", "Rohan", "Pooja", "Vikram" | **100.0%** (5/5 names UNK) |
| **Technical Language** | "container", "microservice", "kubernetes", "redis", "latency", "throughput", "deadlock" | **75.0%** (9/12 terms UNK) |
| **Hinglish / Indian English**| "bhai", "karo", "jaldi", "mat", "kholna", "gaana", "chalao", "band", "ruk", "suno" | **63.2%** (12/19 words UNK) |

### 4.3 Sequence Length Distribution
- **Training Set:** Min = 1, Median = 4, Mean = 4.4, Max = 14 tokens.
- **Validation Set:** Min = 2, Median = 5, Mean = 5.2, Max = 13 tokens.
- **Test Set:** Min = 1, Median = 5, Mean = 5.4, Max = 13 tokens.
- **Finding:** 100% of samples across all existing splits were $\le 14$ tokens! Not a single input in the entire dataset exceeded 14 tokens, despite the model architecture supporting up to 64 tokens. Real human requests ("Could you please check my Downloads folder, find the latest presentation for tomorrow's meeting, and open it in PowerPoint?") average 20–35 tokens and were completely absent from training.

---

## 5. Verification of Other Reported V10 Metrics

| Metric | Reported in V10 | Verified Ground Truth on Test Set | Analysis / Caveat |
| :--- | :--- | :--- | :--- |
| **Intent Accuracy** | 93.94% | **93.94%** (1,597 / 1,700) | Accurate on synthetic test distribution. |
| **Macro-F1 (Intent)** | 94.02% | **94.02%** | Unweighted mean of per-intent F1 scores. |
| **Action Policy Accuracy**| 94.59% | **94.59%** (1,608 / 1,700) | Accurate when intent is correct. |
| **OOD Detection Accuracy**| 99.29% | **99.29%** (1,688 / 1,700) | **Heavily Flawed Metric**: Test OOD examples were artificially obvious noise (e.g. `asdfjkl`, `%%%`). On valid but unfamiliar instructions (e.g. "Can you benchmark the latency of this microservice?"), the model falsely predicted system commands with confidence $> 0.85$. |
| **Confidence Calibration**| "Calibrated" | Brier Score = 0.0073, ECE = 0.0079 | Calibrated on known templates, but poorly calibrated under real distributional shift. |
| **Complexity Accuracy** | Not reported | **97.76%** | High because single vs multi-step was clearly demarcated by conjunctions in the test set. |

---

## 6. Audit of the "18 Tests Passed Across 150 Scenarios"

In the previous report, it was claimed that "18 tests passed across 150 scenarios". Our audit traced this claim:
- In `tests/test_v8_cognitive_scenarios.py`, there are **exactly 18 test functions** (`test_01_dataset_integrity` through `test_18_domain_15_builders_journey_and_vision`).
- The 150 scenarios are defined in `data/cognitive_scenarios_dataset.json`.
- In `test_02_scenario_engine_matches_and_handles_all_150`, it simply loops through the 150 prompts and checks that `CognitiveScenarioEngine.matches_scenario()` returns `True`.
- `CognitiveScenarioEngine` matches them using hardcoded regex expressions (e.g. `r"\b(?:breakup|broke\s+up|heartbreak)\b"`).
- When matched, it returns a static pre-written paragraph with `{user_name}` string interpolation.
- **Conclusion:** The neural network had **zero involvement** in passing these 18 tests. The "150 scenarios" were a dictionary of canned text templates triggered by regex rules.

---

## 7. Architectural & Pipeline Bottlenecks Identified

1. **Discriminative Classifier Claimed as Conversational Intelligence**: The 5-head neural network is a classifier/extractor. It cannot generate language. Treating it as a "ChatGPT alternative" is fundamentally an architectural category error.
2. **Extreme Vocabulary Sparsity**: 832 tokens is insufficient for general English, Indian English, technical terms, and modern application names.
3. **Template Overfitting**: The training dataset's 10,701 redundant samples caused the network to memorize position-dependent slot tags rather than semantic boundaries.
4. **Lack of Negative Sampling for Non-Entity Words**: Common words occurring after "find", "open", "solve" were never presented as non-entities in training.
5. **No True Conversational Multi-Turn Representation**: Current context relies on single previous-turn string substitution (`resolve_anaphora`) rather than genuine dialogue state tracking.

---

## 8. Preserved Production Baseline

Before making any modifications, the existing production checkpoint and configuration have been verified and archived:
- `models/checkpoints/edith_ann_best.pt` (MD5 preserved)
- `models/checkpoints/edith_ann_best.onnx` (MD5 preserved)
- `models/checkpoints/tokenizer.json` (MD5 preserved)
- Rollback command: `git checkout HEAD -- models/checkpoints/`
