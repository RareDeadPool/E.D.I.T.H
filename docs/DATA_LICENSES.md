# Project E.D.I.T.H. V13 — Data Licenses & Data Cards Directory

This document provides a comprehensive audit of all training datasets, conversational corpora, instruction sets, and synthetic data generators utilized in the training and evaluation of **E.D.I.T.H. V13 ("Organic Mind")**.

In accordance with the E.D.I.T.H. V13 sovereign AI mandate:
- **100% Offline Runtime**: No cloud APIs or external telemetry are called at runtime.
- **License Integrity**: All third-party training data sources are verified for permissive redistribution, academic, and non-commercial/commercial compatibility.
- **Synthetic Integrity**: All synthetic data is generated locally by deterministic Python algorithms without reliance on proprietary external cloud LLMs.

---

## 1. Summary of Datasets & Licenses

| Dataset ID | Name & Origin | License | Primary Modality | Estimated Tokens |
| :--- | :--- | :--- | :--- | :--- |
| `DS-01` | **TinyStories / Cosmopedia Narrative Subset** | Apache 2.0 | Common sense reasoning & simple narratives | ~350M |
| `DS-02` | **SmolTalk / OpenAssistant Dialogue Subset** | Apache 2.0 | Empathetic conversation & conversational flow | ~180M |
| `DS-03` | **Databricks Dolly 15k (Filtered)** | CC-BY-SA 3.0 | Instruction adherence, advice, QA | ~15M |
| `DS-04` | **DailyDialog (Cleaned)** | CC-BY-NC-SA 4.0 | Multi-turn emotional dialogue | ~25M |
| `DS-05` | **Cleaned Humor & Banter Corpus** | Public Domain / MIT | Witty banter, puns, clever one-liners | ~12M |
| `SYN-01`| **STT Acoustic Slip & Homophone Generator** | MIT (Local Engine) | Phonetic slips (*"help me a joke"*, *"kill/chill"*) | ~80M |
| `SYN-02`| **Hinglish Code-Switching Dialogue Generator** | MIT (Local Engine) | Romanized Hindi-English syntax (*"bhai friend sad hai"*) | ~60M |
| `SYN-03`| **Emotional Disclosure & Empathy Generator** | MIT (Local Engine) | 1st/2nd/3rd person distress, job search struggles | ~90M |
| `SYN-04`| **Social Introduction & Entity Memory Generator**| MIT (Local Engine) | Introductions (*"I have a friend named X"*), recall | ~65M |
| `SYN-05`| **Tool Proposal & Structured Thinking Generator** | MIT (Local Engine) | `<think>...</think><act>{json}</act>` tool chains | ~120M |
| `SYN-06`| **Hard-Negative Action Safeguard Generator** | MIT (Local Engine) | Process concepts, negations (*"do not close chrome"*) | ~35M |

---

## 2. Detailed Data Cards

### `DS-01`: TinyStories Narrative Subset
- **Source**: Hugging Face / Microsoft Research (`roneneldan/TinyStories`)
- **License**: Apache 2.0
- **Purpose**: Teaches the small model basic English syntax, narrative consistency, cause-and-effect reasoning, and vocabulary relationships without toxic web scrape noise.
- **Filtering**: Deduplicated using MinHash (Jaccard similarity threshold 0.85). Removed any non-ASCII characters that do not map to common punctuation.

### `DS-02`: SmolTalk / OpenAssistant Dialogue
- **Source**: Hugging Face / OpenAssistant Project (`HuggingFaceTB/smoltalk`)
- **License**: Apache 2.0
- **Purpose**: Provides realistic human conversational turn-taking, empathetic listening, and non-robotic helpfulness.
- **Filtering**: Stripped all corporate disclaimers (e.g., *"As an AI language model..."*, *"I cannot feel emotions..."*). Retained human-like conversational persona turns.

### `DS-03`: Databricks Dolly 15k
- **Source**: Databricks (`databricks/databricks-dolly-15k`)
- **License**: Creative Commons Attribution-ShareAlike 3.0 Unported (CC-BY-SA 3.0)
- **Purpose**: High-quality human-written instructions covering information extraction, brainstorming, classification, and summarization.
- **Filtering**: Selected dialogue, creative writing, and general QA subsets; removed multi-file coding and server infrastructure questions unsuited for an operating system assistant.

### `DS-04`: DailyDialog (Cleaned)
- **Source**: Human-annotated dialogue dataset (`li-yan-yan/DailyDialog`)
- **License**: Open Research / CC-BY-NC-SA 4.0
- **Purpose**: Grounding in casual everyday topics: relationships, school, daily life, mood, and personal feelings.
- **Filtering**: Cleaned redundant whitespace, normalized contraction apostrophes, and tokenized emotion tags (happiness, sadness, surprise, fear).

### `DS-05`: Cleaned Humor & Banter Corpus
- **Source**: Curated public domain joke repositories and Reddit clean humor archives (`taivop/joke-dataset`)
- **License**: Public Domain / MIT
- **Purpose**: Replaces static 20-joke arrays with dynamic humor capabilities. Teaches the model comedic timing, pun structures, and playful witty observations.
- **Filtering**: Strict toxicity, profanity, and offensive content removal. De-duplicated against legacy `data/comedy_dataset.json` to prevent joke repetition.

---

## 3. Synthetic Data Cards (Local Deterministic Generation)

All synthetic data for E.D.I.T.H. V13 is generated **entirely locally** using algorithmic Python scripts in `data_pipeline/generators/`. No external cloud LLMs (OpenAI, Anthropic, Google, etc.) were queried or used to generate any portion of this data.

### `SYN-01`: STT Acoustic Slip & Homophone Generator
- **Algorithm**: `data_pipeline/generators/stt_augmenter.py`
- **Mechanism**: Takes clean imperative and conversational speech and applies a phonetic distortion matrix based on common Whisper and Pocket-VAD acoustic transcription errors:
  - *"tell me a joke"* $\rightarrow$ *"help me a joke"*, *"tell me a choke"*, *"del me a joke"*
  - *"close chrome"* $\rightarrow$ *"clothes chrome"*, *"close roam"*
  - Dropped function words, run-on sentences, repeated syllables.
- **Target**: Teaches the model to understand what the user meant despite acoustic degradation.

### `SYN-02`: Hinglish Code-Switching Dialogue Generator
- **Algorithm**: `data_pipeline/generators/hinglish_generator.py`
- **Mechanism**: Algorithmic mixing of romanized Hindi colloquialisms into conversational English:
  - Pronouns & connectors: *"bhai"*, *"yaar"*, *"kuch"*, *"matlab"*, *"arre"*, *"thoda"*.
  - Emotion expressions: *"friend bohot sad hai"*, *"3 saal se try kar raha hai"*, *"koi solution batao"*.
- **Target**: Natural understanding of urban Indian English / Hinglish speech patterns common in voice interactions.

### `SYN-03`: Emotional Disclosure & Empathy Generator
- **Algorithm**: `data_pipeline/generators/emotion_scenarios.py`
- **Mechanism**: Systematically constructs 1st-person, 2nd-person, and 3rd-person personal hardship narratives:
  - Unemployment and prolonged job searching (*"trying for 3 years"*, *"lost his job"*).
  - Grief, burnout, exam stress, relationship anxiety.
  - Matches with warm, non-robotic, practical supportive responses conditioned with `<mood_empathy>`.
- **Target**: Guarantees E.D.I.T.H. responds with deep human compassion and **never** routes personal sadness to system control tools.

### `SYN-04`: Social Introduction & Entity Memory Generator
- **Algorithm**: `data_pipeline/generators/social_intros.py`
- **Mechanism**: Synthesizes social introductions (*"I have a friend named X"*, *"This is my brother Y"*) paired with warm acknowledgment and structured memory extraction tags:
  - Input: *"I have a friend named Harshan."*
  - Output: `<mem>{"entity": "Harshan", "relation": "friend"}</mem> Nice to meet Harshan through you! How long have you two known each other?`
- **Target**: Automatic fact registration in short-term and semantic memory.

### `SYN-05`: Tool Proposal & Structured Thinking Generator
- **Algorithm**: `data_pipeline/generators/reasoning_acts.py`
- **Mechanism**: Generates multi-turn reasoning traces with `<think>` scratchpads followed by non-destructive structured proposals:
  - Format: `<think>User wants to know running processes. Need to inspect OS metrics.</think><act>{"action": "inspect_processes"}</act>`
  - Destructive action format (requires confirmation): `<think>User wants to kill process 12480. Destructive action.</think><act>{"action": "kill_process", "pid": 12480, "confirm_required": true}</act>`
- **Target**: Safe tool generation that requires confirmation before external OS execution.

### `SYN-06`: Hard-Negative Action Safeguard Generator
- **Algorithm**: `data_pipeline/generators/hard_negatives.py`
- **Mechanism**: Synthesizes edge cases containing ambiguous keywords (*"process"*, *"terminate"*, *"kill"*, *"close"*, *"run"*) that are strictly non-imperative:
  - *"Explain what process termination means."*
  - *"The recruitment process is slow."*
  - *"Do not close Chrome."*
  - *"I don't want to terminate anything."*
- **Target**: Zero tool proposals on conversational or conceptual queries.

---

## 4. Ethical & Safety Constraints
1. **PII Scrubbing**: Regex filters automatically replace any prospective phone numbers, email addresses, and physical addresses in training text with standard generic tokens.
2. **Toxicity & Harm Policy**: Prompts containing self-harm encouragement, hate speech, or malicious exploits are purged. Self-harm distress queries are mapped to empathetic, life-affirming crisis resource guidance.
3. **No Copyrighted Script Reproduction**: The model is trained on structural persona archetypes (witty, confident, supportive) and never memorizes or reproduces copyrighted movie dialog lines.
