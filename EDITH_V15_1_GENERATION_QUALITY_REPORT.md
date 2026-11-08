# E.D.I.T.H. V15.1: Generation Quality & Decoding Parameter Report

## 1. Executive Summary
This report documents the generation performance of the scratch-trained **EdithLM 55.90M** checkpoint (`edith_lm_v15_poc_25m_best.pt`, step 1,525, 6.04M valid tokens trained).

All evaluations were executed on **raw, unfiltered generative output directly from the PyTorch model**. No router, no verbalizer, no Bridge Mode, and no heuristic scrapers were used.

---

## 2. Phase 4: Mandated Fixed Prompt Evaluation

The model was tested against the 8 required domain prompts using standard sampling (`temperature=0.7`, `top_k=40`, `seed=42`, `max_new_tokens=60`):

### 1. General Language
- **Prompt**: *"Explain why people sometimes disagree even when they have access to the same information."*
- **Raw Model Output**:
  > *"Yes, there is no way to know which the same way to keep and not understand the world's world. The answer is that the world's first time is to make the best experience. The name of the world is the best place to start in the world. The first place is why the world"*
- **Word Validity**: 100.0% | **Char Trigram Readability**: -6.82
- **Coherence Assessment**: Grammatically constructed from real words, but semantically circular and repetitive. Lacks abstract reasoning about human perspectives or differing interpretations.

### 2. Ordinary Conversation
- **Prompt**: *"Hey Edith, how are you?"*
- **Raw Model Output**:
  > *"You're Aditya, keeping memory slots organized, and waiting for your next command. What are we tackling?</edith>"*
- **Word Validity**: 100.0% | **Char Trigram Readability**: -7.19
- **Coherence Assessment**: **Highly coherent**. Successfully retrieved and articulated its trained persona, user identification (Aditya), and assistant readiness. Emitted closing tag `</edith>`.

### 3. Multi-Turn Conversational Continuity
- **Turn 1 User**: *"I had a difficult day today."*
- **Turn 1 Edith**: *"Yes, there is a number of people in the world. It's a very popular city that is very popular. It has also an average year. The best thing is to learn what is. It"*
- **Turn 2 User**: *"Mostly because of college."*
- **Turn 2 Edith**:
  > *"It is very popular. The most famous sport is great to watch and it's very easy to learn about. It is a great place to buy on what you can make. It's a great place to visit the world. It is the best way"*
- **Word Validity**: 100.0% | **Char Trigram Readability**: -7.08
- **Coherence Assessment**: **FAILED Multi-Turn Continuity**. The model did not maintain context between Turn 1 and Turn 2. When presented with *"Mostly because of college"*, it failed to resolve the emotional distress and defaulted to generic high-frequency Wikipedia prose fragments ("famous sport", "great place to visit").

### 4. Humour
- **Prompt**: *"Tell me a genuinely funny programming joke."*
- **Raw Model Output**:
  > *"There are many ways to be a number of reasons. Some people can be used to learn a few different ways to help you can do. Some people can be a different way to look up in a variety of different things you can do with a few different ways. You can take the most important way"*
- **Word Validity**: 100.0% | **Char Trigram Readability**: -7.19
- **Coherence Assessment**: **FAILED Humour**. Produced generic filler repetitions rather than a joke with setup and punchline.

### 5. Common Sense
- **Prompt**: *"Why would someone carry an umbrella when the sky is dark and cloudy?"*
- **Raw Model Output**:
  > *"It's really good to be the air in the weather. It is a great place to live and will not be very good. A good idea is to be a good way to look up in the morning. It is hard to think that the sun is more important. The most important thing is that"*
- **Word Validity**: 100.0% | **Char Trigram Readability**: -6.89
- **Coherence Assessment**: **FAILED Common Sense**. Associated "dark and cloudy" with general "weather" and "sun", but failed to deduce rain or protection from water.

### 6. Technical Explanation
- **Prompt**: *"Explain recursion to someone who is learning programming for the first time."*
- **Raw Model Output**:
  > *"The first part of the United States is the best sport to be the best player of the world. The first thing is to be an individual. The best way to make the best experience the world is the most successful sport. The best way to play the weather is good and it is a great way"*
- **Word Validity**: 100.0% | **Char Trigram Readability**: -6.80
- **Coherence Assessment**: **FAILED Explanation**. Total hallucination. Emitted sports/Wikipedia fragments completely unrelated to recursion or programming.

### 7. Reasoning & Prioritization
- **Prompt**: *"A student has three assignments due tomorrow, but only enough time to complete two properly. How should they decide what to prioritize?"*
- **Raw Model Output**:
  > *"The following are three days for the day and the day of the day: -The following is a few people of the world - A% of the time is to make the best experience. - the next day is to be a good option. - the day is good for it - the day and"*
- **Word Validity**: 100.0% | **Char Trigram Readability**: -6.60
- **Coherence Assessment**: **FAILED Reasoning**. Emulated markdown bullet points from Dolly-15k instruction formatting, but contents were incoherent gibberish words strung together.

### 8. Indian English
- **Prompt**: *"Explain why somebody might feel frustrated after trying to find a job for several years."*
- **Raw Model Output**:
  > *"Yes, there is no way to spend time for a week. For example, if you want to spend time, you should also be a great way to get a good way to get a good place. This will keep it a big option for you, but also you are looking for a new day"*
- **Word Validity**: 100.0% | **Char Trigram Readability**: -7.20
- **Coherence Assessment**: **FAILED Depth**. Maintained basic English grammar, but showed no comprehension of career frustration or employment search.

---

## 3. Phase 5: Decoding Parameter Benchmark

To identify the optimal inference configuration, we conducted a controlled sweep across decoding methods on the test prompt:
*"Explain why people sometimes disagree even when they have access to the same information."*

| Decoding Configuration | Temperature | Top-K | Top-P | Rep. Penalty | Raw Generated Text | Observed Behavior |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Greedy Decoding** | 0.0 | — | — | 1.0 | *"The answer is that the answer is to the answer. The answer is that the answer is the answer is the answer. The answer is the answer is the answer is the answer..."* | **Catastrophic Mode Collapse**. Unconstrained greedy decoding falls into an infinite deterministic loop of identical tokens. |
| **Low Temperature** | 0.3 | 40 | 1.0 | 1.0 | *"The answer is that the question is the most important way to get the best way to get the best way to keep the best way to get to the time. The answer is to the world's time is to say the best way..."* | **Severe Phrase Looping**. High probability mass concentrated on frequent 3-grams ("the best way"). |
| **Balanced Sampling** | 0.7 | 40 | 1.0 | 1.0 | *"Yes, there is no question that people do not want to understand the world when they are not a way to be able to make an individual. The language is to make the best experience the world's experience..."* | **Fluent English Syntax**. Natural sentence transitions, but low semantic substance. |
| **High Temperature** | 1.0 | 40 | 1.0 | 1.0 | *"Yes, there is no question that there is a question to understand the world when those who do not need to be an understanding in an average year. While there is no number of people like the Earth is an email for humans and its atmosphere..."* | **Bizarre Hallucinations**. Associative leaps become chaotic ("Earth is an email for humans"). |
| **Nucleus Sampling** | 0.7 | — | **0.85** | 1.0 | *"Yes, there is no question that people do not want to write the best way to learn what is the same thing. It also has a few times to become a good way to look up in the world. It is a good place to start in"* | **Cleanest Phrasing**. Top-p truncation effectively clips improbable tail tokens while preventing mode collapse. |
| **Mild Repetition Penalty** | 0.7 | 40 | 1.0 | **1.15** | *"Yes, there is no question that people do not want to understand the world when they are not a way to be able to make an individual. The language of music is important to the world that will take off the world as much as the person in"* | **Improved Lexical Diversity**. Successfully suppresses local phrase repeats ("the best way" disappears). |
| **Moderate Repetition Penalty** | 0.7 | 40 | 1.0 | **1.30** | *"Yes, there is no question that people do not want to understand the world when they are talking about their person. This has also an individual history of time, but they will take off its energy and your experience will cause it as well.</edith>"* | **Clean Termination**. Most coherent overall thought, terminates cleanly with `</edith>` EOS. |

### Variability Across Random Seeds:
Evaluating with different random seeds (`temp=0.7`, `top_k=40`, `rep_pen=1.0`):
- **Seed 42**: Explored "world's experience and individual understanding".
- **Seed 101**: *"In general, some people who see the answer depends on how to buy the word..."*
- **Seed 777**: *"Yes, they have a real-fike is to make a question that is not required to get a good way..."*

**Finding**: High variability confirms that the model has not merely memorized a single fixed path. However, variance also exposes the thinness of semantic grounding: the model freely drifts into unrelated clusters based on initial token selections.

---

## 4. Key Takeaways & Recommendations

1. **Greedy Decoding Must Never Be Used on Undertrained Models**: Mode collapse is severe without sampling or repetition penalties.
2. **Recommended Production Decoding Config for EdithLM**:
   - `temperature = 0.70`
   - `top_p = 0.85` (Nucleus)
   - `top_k = 40`
   - `repetition_penalty = 1.20`
3. **Core Limitation**: The model produces **valid English words and plausible local grammar**, but possesses **near-zero causal reasoning, factual knowledge, or multi-turn conversational memory**. At 6M tokens trained, it is a phonotactically sound language simulator, not an intelligent reasoning agent.

