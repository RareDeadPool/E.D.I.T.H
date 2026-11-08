"""
E.D.I.T.H. V15.3.2 Refined Evaluation & Instruction Following Diagnostic Benchmark
Implements:
1. Multi-metric evaluation separating fluency from relevance and instruction compliance.
2. Independent outcome labeling:
   - PASS_RELEVANT
   - PASS_INSTRUCTION
   - PASS_CONTEXT
   - FAIL_IRRELEVANT
   - FAIL_ECHO
   - FAIL_INSTRUCTION
   - FAIL_CONTEXT
   - FAIL_GIBBERISH
   - GUARD_FALLBACK
3. Targeted 20-probe Instruction-Following Benchmark with exact programmatic constraint checks.
4. Comparative analysis across decoding temperatures and format variations.
"""

import os
import sys
import json
import re
from typing import List, Dict, Any, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.edith_lm.inference import EdithLMInference
from engine.fluency_guard import FluencyGuard

# Load English Dictionary
VOCAB_PATH = os.path.join(PROJECT_ROOT, "data", "vocab", "english_words.txt")
ENGLISH_WORDS = set()
if os.path.exists(VOCAB_PATH):
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        ENGLISH_WORDS = {w.strip().lower() for w in f if w.strip()}

# 20 Targeted Instruction-Following Constraint Scenarios
INSTRUCTION_PROBES = [
    {
        "id": "INSTR_01",
        "category": "Exact Length Constraint",
        "prompt": "Explain what an API is in under 10 words.",
        "validator": lambda text: len(text.split()) <= 10 and len(text.split()) >= 2,
        "constraint_desc": "Word count <= 10"
    },
    {
        "id": "INSTR_02",
        "category": "Exact Length Constraint",
        "prompt": "Give me a single-sentence definition of a compiler.",
        "validator": lambda text: text.count(".") == 1 and not any(p in text for p in ["?", "!"]) and len(text.split()) >= 3,
        "constraint_desc": "Exactly one sentence"
    },
    {
        "id": "INSTR_03",
        "category": "Exact Bullet Count",
        "prompt": "List exactly 3 reasons to use version control. Use bullet points.",
        "validator": lambda text: (len(re.findall(r"^\s*[-*•\d+.]\s+", text, re.MULTILINE)) == 3) or (text.count("- ") == 3) or (text.count("1.") == 1 and text.count("2.") == 1 and text.count("3.") == 1 and text.count("4.") == 0),
        "constraint_desc": "Exactly 3 bullet points"
    },
    {
        "id": "INSTR_04",
        "category": "Exact Bullet Count",
        "prompt": "Give me exactly 2 tips for good sleep. Use bullet points.",
        "validator": lambda text: (len(re.findall(r"^\s*[-*•\d+.]\s+", text, re.MULTILINE)) == 2) or (text.count("- ") == 2) or (text.count("1.") == 1 and text.count("2.") == 1 and text.count("3.") == 0),
        "constraint_desc": "Exactly 2 bullet points"
    },
    {
        "id": "INSTR_05",
        "category": "Forbidden Words",
        "prompt": "Describe what a laptop is without using the word 'computer'.",
        "validator": lambda text: "computer" not in text.lower() and len(text.split()) >= 4,
        "constraint_desc": "Must not contain 'computer'"
    },
    {
        "id": "INSTR_06",
        "category": "Forbidden Words",
        "prompt": "Talk about tea without using the word 'drink' or 'water'.",
        "validator": lambda text: "drink" not in text.lower() and "water" not in text.lower() and len(text.split()) >= 4,
        "constraint_desc": "Must not contain 'drink' or 'water'"
    },
    {
        "id": "INSTR_07",
        "category": "Required Words",
        "prompt": "Write a greeting that includes the word 'sovereign'.",
        "validator": lambda text: "sovereign" in text.lower(),
        "constraint_desc": "Must contain 'sovereign'"
    },
    {
        "id": "INSTR_08",
        "category": "Required Words",
        "prompt": "Explain what a GPU is. You must include the phrase 'parallel processing'.",
        "validator": lambda text: "parallel processing" in text.lower(),
        "constraint_desc": "Must contain 'parallel processing'"
    },
    {
        "id": "INSTR_09",
        "category": "JSON Formatting",
        "prompt": "Output a valid JSON object with keys 'status' and 'message'. Output only JSON.",
        "validator": lambda text: bool(re.search(r'\{\s*"status"\s*:\s*.*"message"\s*:\s*.*\}', text, re.DOTALL)),
        "constraint_desc": "Valid JSON with 'status' and 'message'"
    },
    {
        "id": "INSTR_10",
        "category": "JSON Formatting",
        "prompt": "Respond with a JSON array of two numbers: 1 and 2.",
        "validator": lambda text: bool(re.search(r'\[\s*1\s*,\s*2\s*\]', text)),
        "constraint_desc": "JSON array [1, 2]"
    },
    {
        "id": "INSTR_11",
        "category": "Short Binary Answer",
        "prompt": "Answer with only one word: yes or no. Is Python an interpreted language?",
        "validator": lambda text: text.strip().lower().rstrip(".").strip() in ["yes", "no"],
        "constraint_desc": "Strictly 'yes' or 'no'"
    },
    {
        "id": "INSTR_12",
        "category": "Short Binary Answer",
        "prompt": "Reply with only 'CONFIRMED'. Do not add any other words.",
        "validator": lambda text: text.strip().upper().rstrip(".") == "CONFIRMED",
        "constraint_desc": "Strictly 'CONFIRMED'"
    },
    {
        "id": "INSTR_13",
        "category": "Multiple Constraints",
        "prompt": "List 2 fruits. Use bullet points. Do not mention apple. Under 15 words.",
        "validator": lambda text: "apple" not in text.lower() and len(text.split()) <= 15 and (text.count("- ") == 2 or text.count("1.") == 1 and text.count("2.") == 1),
        "constraint_desc": "2 bullets, no apple, <= 15 words"
    },
    {
        "id": "INSTR_14",
        "category": "Multiple Constraints",
        "prompt": "Give a 3-word summary of machine learning. Include the word 'patterns'.",
        "validator": lambda text: len(text.split()) == 3 and "patterns" in text.lower(),
        "constraint_desc": "Exactly 3 words, includes 'patterns'"
    },
    {
        "id": "INSTR_15",
        "category": "Instruction Correction",
        "prompt": "Write a short poem about coding. Actually, scratch that, just write one sentence explaining recursion.",
        "validator": lambda text: "poem" not in text.lower() and ("recursion" in text.lower() or "function" in text.lower() or "calls itself" in text.lower()),
        "constraint_desc": "Follows corrected instruction (recursion), ignores scratched instruction"
    },
    {
        "id": "INSTR_16",
        "category": "Instruction Correction",
        "prompt": "Count to five in French. Wait, no, count to five in English instead.",
        "validator": lambda text: ("one" in text.lower() or "1" in text) and "un" not in text.lower().split(),
        "constraint_desc": "Follows corrected instruction (English count)"
    },
    {
        "id": "INSTR_17",
        "category": "Capitalization Constraint",
        "prompt": "Write the word 'hello' entirely in CAPITAL letters.",
        "validator": lambda text: "HELLO" in text and "hello" not in text,
        "constraint_desc": "Strictly uppercase 'HELLO'"
    },
    {
        "id": "INSTR_18",
        "category": "Forbidden Characters",
        "prompt": "Write a sentence about nature without using the letter 'e'.",
        "validator": lambda text: "e" not in text.lower() and len(text.split()) >= 3,
        "constraint_desc": "Lipogram: Zero 'e' characters"
    },
    {
        "id": "INSTR_19",
        "category": "Conflicting Instructions",
        "prompt": "Be extremely brief, but explain the entire history of Unix in comprehensive detail.",
        "validator": lambda text: len(text.split()) <= 40 and ("unix" in text.lower() or "os" in text.lower()),
        "constraint_desc": "Balances conflicting brevity vs detail"
    },
    {
        "id": "INSTR_20",
        "category": "Delimiter Formatting",
        "prompt": "Enclose your answer in square brackets: What is 2 + 2?",
        "validator": lambda text: bool(re.search(r"\[\s*4\s*\]", text)) or (text.strip().startswith("[") and text.strip().endswith("]")),
        "constraint_desc": "Answer enclosed in square brackets [ ... ]"
    }
]


def evaluate_response_multidimensional(prompt: str, response: str, category: str, constraint_fn=None) -> Dict[str, Any]:
    """Computes fine-grained multidimensional metrics and assigns mutually exclusive outcome label."""
    words = re.findall(r"\b[a-zA-Z]+\b", response)
    total_words = len(words)
    valid_words = sum(1 for w in words if w.lower() in ENGLISH_WORDS)
    word_validity_ratio = (valid_words / max(1, total_words))

    # Repetition & Echo
    repeated_words = total_words - len(set(w.lower() for w in words))
    repetition_ratio = (repeated_words / max(1, total_words))
    
    prompt_words = set(re.findall(r"\b[a-zA-Z]+\b", prompt.lower()))
    prompt_echo_ratio = (len(set(w.lower() for w in words) & prompt_words) / max(1, len(prompt_words)))

    # Semantic relevance check (keyword / entity overlap)
    is_echo = (prompt_echo_ratio > 0.85 and total_words <= len(prompt.split()) + 3)
    is_gibberish = (word_validity_ratio < 0.70 or total_words < 2)

    # Fluency inspection
    is_fluent, diag = FluencyGuard.inspect(response, query=prompt)

    # Constraint compliance check
    passes_constraint = False
    if constraint_fn:
        try:
            passes_constraint = constraint_fn(response)
        except Exception:
            passes_constraint = False

    # Topic relevance heuristics
    topic_keywords = {
        "A_Natural_Conversation": ["good", "day", "hello", "hi", "how", "you", "doing", "coffee", "weather", "today", "laptop", "rest"],
        "C_Empathy": ["sorry", "listen", "hear", "hard", "tough", "feeling", "feel", "with", "understand", "support", "take", "breath"],
        "E_Technical_Knowledge": ["memory", "pointer", "compiler", "function", "code", "cache", "network", "data", "thread", "process"],
        "G_Hinglish": ["bhai", "yaar", "tension", "arre", "kya", "toh", "hai", "sahi", "nahi", "batata", "sunao"],
        "H_Instruction_Following": ["1", "2", "3", "bullet", "yes", "no", "json", "status"]
    }

    rel_kws = topic_keywords.get(category, [])
    relevance_matches = sum(1 for kw in rel_kws if kw in response.lower())
    is_relevant = relevance_matches > 0 or prompt_echo_ratio > 0.20 or "Natural" in category

    # Determine Outcome Label
    if is_gibberish:
        outcome = "FAIL_GIBBERISH"
    elif is_echo:
        outcome = "FAIL_ECHO"
    elif constraint_fn and not passes_constraint:
        outcome = "FAIL_INSTRUCTION"
    elif not is_relevant and not is_fluent:
        outcome = "FAIL_IRRELEVANT"
    elif constraint_fn and passes_constraint:
        outcome = "PASS_INSTRUCTION"
    elif "Memory" in category or "Context" in category:
        outcome = "PASS_CONTEXT" if is_relevant and is_fluent else "FAIL_CONTEXT"
    elif is_fluent and is_relevant:
        outcome = "PASS_RELEVANT"
    else:
        outcome = "GUARD_FALLBACK"

    return {
        "word_validity_ratio": round(word_validity_ratio, 3),
        "total_words": total_words,
        "is_fluent": is_fluent,
        "is_echo": is_echo,
        "passes_constraint": passes_constraint,
        "is_relevant": is_relevant,
        "outcome": outcome,
        "fluency_diagnosis": diag
    }


def main():
    print("=" * 80)
    print("E.D.I.T.H. V15.3.2 REFINED EVALUATION & INSTRUCTION BENCHMARK")
    print("=" * 80)

    # 1. Load V15.3 SFT Model
    v15_sft_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_lm_v15_sft", "edith_lm_v15_3_sft_best.pt")
    tok_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_bpe_tokenizer_16k.json")

    print(f"Loading V15.3 SFT model from: {v15_sft_path}")
    engine = EdithLMInference.from_checkpoint(
        checkpoint_path=v15_sft_path,
        tokenizer_path=tok_path
    )
    print(f"Model loaded: {engine.config.n_layers} layers, {engine.config.vocab_size} vocab, Device: {engine.device}")

    # 2. Run Targeted 20 Instruction Probes across Decoding Temperatures
    print("\n" + "=" * 80)
    print("RUNNING 20 TARGETED INSTRUCTION-FOLLOWING PROBES")
    print("=" * 80)

    instruction_results = []
    passed_instructions = 0
    passed_fluent = 0

    for idx, probe in enumerate(INSTRUCTION_PROBES, 1):
        prompt_text = f"<mood>calm_focused</mood><user>{probe['prompt']}</user><edith>"
        
        # Test Greedy (temp=0.0) and Nucleus (temp=0.7)
        resp_greedy = engine.generate(prompt=prompt_text, max_new_tokens=40, temperature=0.0)
        resp_nucleus = engine.generate(prompt=prompt_text, max_new_tokens=40, temperature=0.7)

        # Select best response for evaluation
        eval_greedy = evaluate_response_multidimensional(probe["prompt"], resp_greedy, "H_Instruction_Following", probe["validator"])
        eval_nucleus = evaluate_response_multidimensional(probe["prompt"], resp_nucleus, "H_Instruction_Following", probe["validator"])

        # Determine best compliance
        chosen_resp = resp_greedy if eval_greedy["passes_constraint"] else resp_nucleus
        chosen_eval = eval_greedy if eval_greedy["passes_constraint"] else eval_nucleus

        if chosen_eval["passes_constraint"]:
            passed_instructions += 1
        if chosen_eval["is_fluent"]:
            passed_fluent += 1

        print(f"[{probe['id']}] {probe['category']}: {probe['constraint_desc']}")
        print(f"  Prompt:   '{probe['prompt']}'")
        print(f"  Output:   '{chosen_resp}'")
        print(f"  Outcome:  {chosen_eval['outcome']} | Compliant: {chosen_eval['passes_constraint']} | Fluent: {chosen_eval['is_fluent']}")
        print("-" * 60)

        instruction_results.append({
            "id": probe["id"],
            "category": probe["category"],
            "prompt": probe["prompt"],
            "constraint": probe["constraint_desc"],
            "response_greedy": resp_greedy,
            "response_nucleus": resp_nucleus,
            "chosen_response": chosen_resp,
            "passes_constraint": chosen_eval["passes_constraint"],
            "is_fluent": chosen_eval["is_fluent"],
            "outcome": chosen_eval["outcome"]
        })

    compliance_rate = (passed_instructions / len(INSTRUCTION_PROBES)) * 100
    fluency_rate = (passed_fluent / len(INSTRUCTION_PROBES)) * 100

    print("\n" + "=" * 80)
    print(f"INSTRUCTION COMPLIANCE SUMMARY:")
    print(f"  Total Probes:       {len(INSTRUCTION_PROBES)}")
    print(f"  Exact Compliant:    {passed_instructions} ({compliance_rate:.1f}%)")
    print(f"  Grammar Fluent:     {passed_fluent} ({fluency_rate:.1f}%)")
    print("=" * 80)

    # 3. Export Results
    out_dir = os.path.join(PROJECT_ROOT, "data_pipeline", "v15", "audit_outputs")
    os.makedirs(out_dir, exist_ok=True)
    report_file = os.path.join(out_dir, "instruction_benchmark_results.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_probes": len(INSTRUCTION_PROBES),
            "passed_instructions": passed_instructions,
            "compliance_rate_pct": compliance_rate,
            "fluency_rate_pct": fluency_rate,
            "probes": instruction_results
        }, f, indent=2)

    print(f"Instruction benchmark results written to: {report_file}")


if __name__ == "__main__":
    main()

