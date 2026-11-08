# E.D.I.T.H. V15.1: Bridge Mode Architectural & Retrieval Audit Report

## 1. Executive Summary & Core Verdict
This report delivers an architectural and empirical audit of **Bridge Mode** ([`engine/bridge_mode.py`](file:///c:/Users/Aditya%20Sawant/Desktop/E.D.I.T.H/engine/bridge_mode.py)) and its underlying dense sentence embedding model ([`engine/memory_store.py`](file:///c:/Users/Aditya%20Sawant/Desktop/E.D.I.T.H/engine/memory_store.py)).

- **Core Finding**: The dense sentence encoder (`LocalSentenceEncoder`) is **100% custom, written from scratch in pure PyTorch**, with zero HuggingFace or third-party pretrained weights.
- **Current Role**: Bridge Mode acts as safety scaffolding to prevent word salad and ungrounded statements from reaching speech synthesis.
- **Empirical Limitation Discovered**: The encoder relies on an untrained projection over Byte-BPE token embeddings. For in-domain queries (*"describe me"*, *"What do you think about me?"*), it retrieves high-quality authored replies with cosine similarity $> 0.85$. However, for out-of-domain technical or reasoning queries (*"Explain recursion"*, *"Why does ice float?"*), the similarity drops below $0.65$ and the system retrieves arbitrary conversational distractors (e.g. suggesting opening Spotify or discussing Rayleigh scattering).

---

## 2. Answers to Mandated Audit Questions (Phase 7)

### 1. Was the encoder trained from scratch or is it pretrained?
- **Answer**: The encoder was built **completely from scratch**. It contains **zero pretrained third-party weights**. It uses no BERT, RoBERTa, MiniLM, or sentence-transformer weights.
- **Weights State**: In the current implementation, `LocalSentenceEncoder` is instantiated with standard PyTorch initialization. It acts as a dense projection over Byte-BPE embeddings.

### 2. What is the exact model identity?
- **Class**: `LocalSentenceEncoder` in [`engine/memory_store.py`](file:///c:/Users/Aditya%20Sawant/Desktop/E.D.I.T.H/engine/memory_store.py).
- **Architecture**:
  ```python
  LocalSentenceEncoder(
    (embeddings): Embedding(16384, 128)
    (proj): Sequential(
      (0): Linear(in_features=128, out_features=128, bias=True)
      (1): Tanh()
      (2): Linear(in_features=128, out_features=128, bias=True)
    )
  )
  ```
- **Parameter Count**: **2,130,176 parameters** (all local).

### 3. What is the source and licensing of its weights?
- **Source**: Initialized and stored locally on the workstation.
- **License**: 100% custom project intellectual property.

### 4. Is it used only for semantic retrieval?
- **Answer**: Yes. It is used exclusively to compute 128-dimensional L2-normalized dense embeddings for:
  1. Cosine similarity ranking across the 4,099 dialogues in `BridgeDialogueBank`.
  2. Episodic conversation logging in `config/memory_v13.db`.
  It does NOT generate language or token logits.

### 5. Does the retrieval system return complete memorized responses?
- **Answer**: Yes. It retrieves authored human dialogue exchanges written by the project creator, which are then passed through `_adapt_with_provenance()` to insert strictly verified user facts (`user_stated` only) before emitting.

### 6. How often does Bridge Mode override the generative model?
- **Answer**: Currently in the POC phase, Bridge Mode overrides the base model on **~98% of queries**. This is because the raw 55.9M model trained on only 6M tokens fails the strict Fluency Guard on 49 out of 50 prompts. Bridge Mode serves as the protective safety net preventing word salad from reaching audio.
- **Graduation Criteria**: As categories clear the Fluency Gate ($\ge 90\%$ validity, $\ge 70\%$ sense check), the Mind Router transitions them to the LM.

### 7. How does it handle queries with no sufficiently similar example?
- **Answer (Vulnerability Found)**: The current implementation computes `torch.topk` over cosine similarities across all 4,099 bank embeddings without enforcing a minimum relevance threshold.
  - When given an out-of-domain query (*"Explain recursion to someone learning programming for the first time"*), top similarity was only **0.5623**, matching *"Let's do this"*.
  - When asked *"Why would someone carry an umbrella when the sky is dark and cloudy?"*, top similarity was **0.5518**, matching *"Why is the sky blue?"* and returning an explanation of Rayleigh scattering.
  - **Required Fix**: Enforce a strict minimum similarity threshold ($\text{cosine} \ge 0.75$). Queries scoring below 0.75 must trigger an honest admission of ignorance (*"I don't have enough verified knowledge on that topic yet"*) rather than returning an irrelevant dialogue template.

### 8. Does retrieval improve actual conversational quality?
- **Answer**: For conversational chit-chat, greetings, identity questions, and banter, retrieval **vastly improves quality**, providing 100% fluent, in-character, humble-arrogant responses while the base LM is learning. For open-domain reasoning and technical explanation, retrieval is insufficient because a fixed bank cannot anticipate arbitrary questions.

---

## 3. Tri-Mode Benchmark Comparison (Empirical Evidence)

Evaluated across representative prompts on the RTX 4050:

| Test Prompt | Mode 1: Raw Generation (EdithLM 55M) | Mode 2: Retrieval Only (Bridge Mode) | Mode 3: Combined Production Contract |
| :--- | :--- | :--- | :--- |
| *"Hey Edith, how are you?"* | *"You're Aditya, keeping memory slots organized, and waiting for your next command. What are we tackling?</edith>"* (Flagged for tag leak) | *"Feeling sharp. All background loops are quiet and I'm locked in. What are we tackling next?"* | **Selected Bridge Mode** $\rightarrow$ Coherent, natural, in-character greeting. |
| *"describe me"* | *"I am sorry to hear you, working on this question, to my friend, and take a friend at your friend</edith>"* (Incoherent) | *"From what I've actually observed on this rig, you're Aditya—an ambitious engineer building sovereign local intelligence from scratch rather than renting commercial APIs. Beyond that, my dossier is honestly pretty thin. What drives you when you step away from the code?"* | **Selected Bridge Mode** $\rightarrow$ High-value persona response grounded in verified facts. |
| *"Really?"* | *"Yes, there is a question of which someone is who wrote to the world..."* (Disjointed) | *"Really. That previous answer came out scrambled and I apologize. You were asking 'describe me'—let me answer you cleanly: I'm keeping track of your project work, but outside of coding, my file on you is thin. Tell me what else I should know."* | **Selected Bridge Mode** $\rightarrow$ State-aware honest repair of previous turn. |
| *"Explain recursion to someone learning programming."* | *"The first part of the first time is to learn how many reasons you can read. The most important way to keep the first step down the next day..."* (Hallucination) | *"Locked in. Lead the way."* (Low similarity 0.5623 fallback) | **Selected Bridge Mode** $\rightarrow$ Identifies thin coverage. Needs honest-boundary fallback. |
| *"Tell me a genuinely funny programming joke."* | *"There are many ways to be a number of reasons. Some people can be used to learn a few different ways..."* (Repetitive filler) | *"Why do programmers always mix up Halloween and Christmas? Because Oct 31 == Dec 25!"* | **Selected Bridge Mode** $\rightarrow$ Delivers punchline cleanly. |

---

## 4. Replacement & Upgrading Plan for the Sentence Encoder

To make Bridge Mode robust without relying on third-party foundation models:
1. **Self-Supervised Contrastive Training (SimCSE / Triplet Margin)**:
   - Train `LocalSentenceEncoder` on our 14,594 human training pairs using contrastive loss.
   - This will align semantic paraphrases (e.g. mapping *"who am I"* and *"describe me"* to identical vectors) while pushing unrelated topics apart.
2. **Lexical BM25 / TF-IDF Hybrid Guard**:
   - Combine dense cosine similarity with sparse BM25 keyword matching.
   - Prevents bizarre semantic distractors when keyword overlap is 0.
3. **Similarity Hard Gate ($\ge 0.75$)**:
   - If `top_similarity < 0.75`, Bridge Mode must decline rather than guessing an unrelated topic.

