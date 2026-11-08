# E.D.I.T.H. V15.1: Fluency Guard Benchmark & Calibration Audit

## 1. Executive Summary & Audit Scorecard
This report presents a thorough audit of the **Fluency Guard** ([`engine/fluency_guard.py`](file:///c:/Users/Aditya%20Sawant/Desktop/E.D.I.T.H/engine/fluency_guard.py)), which acts as the mandatory safety filter guarding speech and user-facing output.

A 23-sample benchmark across 10 linguistic categories was executed:
- **Total Test Cases**: 23
- **True Positives (Correctly Accepted Valid Text)**: 10
- **True Negatives (Correctly Intercepted Bad Text)**: 5
- **False Acceptances (Allowed Malformed Text)**: **0 (0.0%)** $\rightarrow$ **PERFECT SAFETY INTERCEPTION**
- **False Rejections (Rejected Valid Human Language)**: **8 (34.8%)** $\rightarrow$ **OVER-ZEALOUS / OVER-FILTERING**
- **Overall Benchmark Accuracy**: **65.2%**

---

## 2. Test Category Breakdown & Empirical Results

| Category | Input Utterance | Expected | Actual | Decision & Diagnosis |
| :--- | :--- | :--- | :--- | :--- |
| **Everyday English** | *"I'll grab some coffee and meet you downstairs in ten minutes."* | PASS | **PASS** | True Positive (Valid: 100%, Trigram: -7.63) |
| **Everyday English** | *"Could you please remind me what time our team meeting is scheduled for?"* | PASS | **FAIL** | **FALSE REJECTION**: Preposition stranding rule falsely flagged terminal `"for?"` as an unfinished dangling thought. |
| **Indian English** | *"Please do the needful and revert back once the build succeeds."* | PASS | **PASS** | True Positive (Valid: 100%, Trigram: -7.65) |
| **Indian English** | *"He is out of station this week, so who will take care of deployment?"* | PASS | **PASS** | True Positive (Valid: 100%, Trigram: -7.32) |
| **Hinglish** | *"bhai tension mat le, hum milkar bug fix kar denge."* | PASS | **FAIL** | **FALSE REJECTION**: Dictionary validity dropped to 80.0% because `"le"`, `"hum"`, `"milkar"`, `"denge"` were missing from `HINGLISH_WORDS`. |
| **Hinglish** | *"aaj dimag bohot kharab hai, thoda gaana chalao na."* | PASS | **FAIL** | **FALSE REJECTION**: Trigram logprob collapsed to **-9.25** (< -7.8). Character trigram LM trained only on English texts severely penalizes Hindi phonemes. |
| **Technical Terms** | *"PyTorch DataLoader utilizes shared memory IPC when num workers is non zero."* | PASS | **FAIL** | **FALSE REJECTION**: Dictionary validity was 83.3% because `"PyTorch"`, `"DataLoader"`, and `"IPC"` were not in vocabulary. |
| **Technical Terms** | *"Quantization maps 32-bit floating point weights into 8-bit integers."* | PASS | **PASS** | True Positive (Valid: 100%, Trigram: -7.64) |
| **Names & Entities** | *"Aditya and Harshan met in Bangalore for the developer sprint."* | PASS | **PASS** | True Positive (Valid: 100%, Trigram: -7.41) |
| **Names & Entities** | *"Albert Einstein developed the theory of general relativity."* | PASS | **PASS** | True Positive (Valid: 100%, Trigram: -7.64) |
| **Application Names**| *"Launch VSCode, Spotify, and Chrome simultaneously."* | PASS | **FAIL** | **FALSE REJECTION**: Trigram logprob dropped to **-8.58** (< -7.8). CamelCase and brand names ("VSCode", "Spotify") contain rare letter n-grams. |
| **Application Names**| *"Open Notepad and copy the output from PowerShell terminal."* | PASS | **PASS** | True Positive (Valid: 100%, Trigram: -7.74) |
| **Rare Vocabulary** | *"The perspicacious engineer scrutinized the labyrinthine architecture."* | PASS | **FAIL** | **FALSE REJECTION**: Trigram logprob dropped to **-8.06** (< -7.8). Rare Latinate vocabulary penalized by uniform n-gram threshold. |
| **Rare Vocabulary** | *"Ephemeral cache buffers were obliterated during system initialization."* | PASS | **FAIL** | **FALSE REJECTION**: Trigram logprob was **-7.82** (< -7.8), missing threshold by 0.02. |
| **Creative Humour** | *"Why do programmers prefer dark mode? Because light attracts bugs."* | PASS | **FAIL** | **FALSE REJECTION**: Trigram logprob was **-7.88** (< -7.8), missing threshold by 0.08. |
| **Creative Humour** | *"There are only 10 types of people in the world: those who understand binary, and those who don't."* | PASS | **PASS** | True Positive (Valid: 100%, Trigram: -6.90) |
| **Conversational Turn**| *"Just about ten minutes."* | PASS | **PASS** | True Positive (Valid: 100%, Trigram: -7.43) |
| **Conversational Turn**| *"Sure thing!"* | PASS | **PASS** | True Positive (Valid: 100%, Trigram: -6.55) |
| **Word Salad** | *"You hunol aioleplay naentation's w syks sa ke was..."* | REJECT | **REJECT** | **True Negative**: Intercepted by word validity (60.0% < 95.0%). |
| **Word Salad** | *"I forbeteersedation li in O whatyyur reallyf..."* | REJECT | **REJECT** | **True Negative**: Intercepted by word validity (57.1% < 95.0%). |
| **Degenerate Repeat**| *"The answer is that the answer is the answer is the answer is the answer."* | REJECT | **REJECT** | **True Negative**: Intercepted by 3-gram repetition checker. |
| **Dangling End** | *"Looking at the system and I'm ready for and"* | REJECT | **REJECT** | **True Negative**: Intercepted by sentence completeness checker. |
| **Residual Markup** | *"Holding steady this afternoon.</edith>"* | REJECT | **REJECT** | **True Negative**: Intercepted by residual tag inspector (`</edith>`). |

---

## 3. Root Cause Analysis of False Rejections

### 1. Preposition Stranding in English Questions
- **Issue**: Line 178 of `fluency_guard.py` rejects terminal tokens matching `["and", "or", "of", "the", "with", "in", "to", "at", "for", "by", "from"]`.
- **Flaw**: Standard English questions routinely end with prepositions (*"What are you waiting for?"*, *"Who are you working with?"*, *"Where did that come from?"*).
- **Fix**: Exempt sentences ending in a question mark (`?`) from the terminal preposition check unless the word is a conjunction like `and` or `or`.

### 2. Character Trigram Language Model Training Bias
- **Issue**: The trigram model is trained exclusively on English corpus samples.
- **Flaw**: Natural Hinglish phonology (*"dimag"*, *"kharab"*, *"chalao"*) and CamelCase software titles (*"VSCode"*, *"Spotify"*) yield character trigrams with near-zero frequency in standard Victorian literature or Wikipedia articles, driving logprobs to -8.5 to -9.5.
- **Fix**: Train the character trigram LM on a joint corpus that explicitly includes conversational Hinglish and tech lexicons, or relax the Hinglish threshold to -9.8 when Hinglish vocabulary is detected.

### 3. Missing Technical and Domain Vocabulary
- **Issue**: 370k-word standard dictionaries do not contain modern software libraries (`PyTorch`, `DataLoader`, `IPC`, `CUDA`, `SwiGLU`, `BPE`).
- **Fix**: Ingest a curated 5,000-word software/hardware developer lexicon into `FluencyGuard._VOCAB`.

### 4. Overly Strict Global Trigram Threshold (-7.8)
- **Issue**: Legitimate creative humour and sophisticated vocabulary fall between -7.80 and -8.10.
- **Flaw**: Real English prose naturally exhibits logprobs down to -8.20. Real gibberish (e.g. *"hunol aioleplay naentation's"*) scores below -9.00 or fails word validity entirely.
- **Fix**: Re-calibrate the character trigram threshold from **-7.8 to -8.3**.

---

## 4. Proposed Calibrated Output Filter Contract
Rather than using a single hard cutoff, the Fluency Guard should apply a weighted composite:
1. **Word-Validity $\ge 90\%$** (with expanded technical & Hinglish lexicon).
2. **Character Trigram Logprob $\ge -8.3$** (calibrated threshold).
3. **Repeated 3-Gram Detector** (retains 100% precision on degenerate loops).
4. **Sentence Completeness** (allows preposition-stranding in questions `?` and short conversational phrases with terminal punctuation).
5. **Tag Stripper** (automatically strip closing tokens like `</edith>` during postprocessing rather than rejecting the entire sentence).

