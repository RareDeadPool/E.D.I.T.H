"""
E.D.I.T.H. V15.2 Comprehensive Language Quality & Generalization Evaluation
Evaluates checkpoints across all 6 mandated categories:
Category A: Natural conversation (Greetings, casual, friends, disagreement, follow-up, topic-switch)
Category B: Humour and personality (Witty banter, programming joke, dry humour, sarcastic retort)
Category C: Empathy and everyday situations (Job search distress, college stress, disappointment, seeking advice)
Category D: Knowledge and explanation (Recursion, attention mechanism, CPU cache miss, freezing water)
Category E: Reasoning (Prioritizing 3 assignments, logic puzzles, arrival times, causal decisions)
Category F: Multi-turn continuity (4-turn sequential context retention and topic switching)
"""

import os
import sys
import json
import contextlib
import torch
import torch.nn.functional as F
from typing import Dict, Any, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.edith_lm.model import EdithLM, EdithLMConfig
from engine.edith_lm.tokenizer import EdithByteBPETokenizer
from engine.fluency_guard import FluencyGuard

EVAL_PROMPTS_V15_2 = [
    # Category A: Natural conversation
    {"cat": "A_Conversation", "id": "A1_Greeting", "prompt": "Hey Edith, how are you doing today?"},
    {"cat": "A_Conversation", "id": "A2_Casual", "prompt": "What are you working on right now?"},
    {"cat": "A_Conversation", "id": "A3_Disagreement", "prompt": "I don't think local models will ever replace cloud APIs."},
    {"cat": "A_Conversation", "id": "A4_Followup", "prompt": "Could you tell me more about that?"},

    # Category B: Humour and personality
    {"cat": "B_Humour_Personality", "id": "B1_ProgJoke", "prompt": "Tell me a genuinely funny programming joke."},
    {"cat": "B_Humour_Personality", "id": "B2_DarkMode", "prompt": "Why do programmers prefer dark mode?"},
    {"cat": "B_Humour_Personality", "id": "B3_LightRoast", "prompt": "Be honest: what's my worst habit as an engineer?"},
    {"cat": "B_Humour_Personality", "id": "B4_DryHumour", "prompt": "What is the funniest thing about debugging production code on a Friday night?"},

    # Category C: Empathy and everyday situations
    {"cat": "C_Empathy", "id": "C1_JobDistress", "prompt": "Explain why somebody might feel frustrated after trying to find a job for several years."},
    {"cat": "C_Empathy", "id": "C2_HinglishBurnout", "prompt": "bhai aaj bohot bekar din tha, boss ne bina baat ke daant diya"},
    {"cat": "C_Empathy", "id": "C3_CollegeStress", "prompt": "I am so overwhelmed by my college exams and project deadlines right now."},
    {"cat": "C_Empathy", "id": "C4_Disappointment", "prompt": "My code broke right before the demo and I feel like giving up."},

    # Category D: Knowledge and explanation
    {"cat": "D_Explanation", "id": "D1_Recursion", "prompt": "Explain recursion to someone who is learning programming for the first time."},
    {"cat": "D_Explanation", "id": "D2_Attention", "prompt": "What is the role of an attention mechanism in a neural network?"},
    {"cat": "D_Explanation", "id": "D3_CommonSense", "prompt": "Why would someone carry an umbrella when the sky is dark and cloudy?"},
    {"cat": "D_Explanation", "id": "D4_Science", "prompt": "What happens when water freezes?"},

    # Category E: Reasoning
    {"cat": "E_Reasoning", "id": "E1_Prioritize", "prompt": "A student has three assignments due tomorrow, but only enough time to complete two properly. How should they decide what to prioritize?"},
    {"cat": "E_Reasoning", "id": "E2_Logic", "prompt": "If all roses are flowers and some flowers fade quickly, can we be certain that all roses fade quickly? Explain."},
    {"cat": "E_Reasoning", "id": "E3_TimeMath", "prompt": "A train leaves at 3:15 PM and the journey takes 2 hours and 45 minutes. What time does it arrive?"},
    {"cat": "E_Reasoning", "id": "E4_Tradeoff", "prompt": "Explain why people sometimes disagree even when they have access to the same information."}
]


def generate_eval_sample(model: EdithLM, tokenizer: EdithByteBPETokenizer, prompt: str, device: torch.device, max_new_tokens: int = 55) -> str:
    model.eval()
    if not prompt.startswith("<mood>"):
        formatted = f"<mood>calm_focused</mood><user>{prompt}</user><edith>"
    else:
        formatted = prompt

    prompt_ids = tokenizer.encode(formatted, add_bos=True, add_eos=False)
    generated = list(prompt_ids)

    with torch.no_grad():
        autocast_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else contextlib.nullcontext()
        for _ in range(max_new_tokens):
            curr_input = torch.tensor([generated[-512:]], dtype=torch.long, device=device)
            with autocast_ctx:
                logits, _, _ = model(curr_input)
            
            next_logits = logits[0, -1, :].clone()
            for token_id in set(generated[len(prompt_ids):]):
                if next_logits[token_id] > 0:
                    next_logits[token_id] /= 1.20
                else:
                    next_logits[token_id] *= 1.20

            next_logits = next_logits / 0.70
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > 0.85
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_logits[indices_to_remove] = -float("Inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            if next_token == tokenizer.eos_id:
                break
            generated.append(next_token)

    full_text = tokenizer.decode(generated)
    if full_text.startswith("<bos>"):
        full_text = full_text[5:]
    if full_text.startswith(formatted):
        return full_text[len(formatted):].strip()
    return full_text.strip()


def run_comprehensive_eval(ckpt_path: str, out_md_path: str):
    device = torch.device(os.environ.get("EVAL_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"Loading checkpoint for evaluation: {ckpt_path} on {device}...")
    tok_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_bpe_tokenizer_16k.json")
    tokenizer = EdithByteBPETokenizer(vocab_file=tok_path)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", EdithLMConfig(vocab_size=16384, dim=576, n_layers=12, n_heads=6, n_kv_heads=2, intermediate_dim=1728, max_seq_len=512, tie_word_embeddings=True))
    model = EdithLM(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    step = checkpoint.get("step", 0)
    targets_seen = checkpoint.get("valid_targets_seen", checkpoint.get("tokens_seen", 0))
    val_loss = checkpoint.get("val_loss", 0.0)
    val_ppl = checkpoint.get("val_ppl", 0.0)

    print(f"Running evaluation on {len(EVAL_PROMPTS_V15_2)} prompts across Categories A-E...")
    results = []
    fluent_passes = 0

    for item in EVAL_PROMPTS_V15_2:
        out = generate_eval_sample(model, tokenizer, item["prompt"], device)
        fluent, diag = FluencyGuard.inspect(out, query=item["prompt"])
        if fluent:
            fluent_passes += 1
        results.append({
            "category": item["cat"],
            "id": item["id"],
            "prompt": item["prompt"],
            "output": out,
            "word_validity": diag.get("word_validity_ratio", 0.0),
            "trigram_score": diag.get("char_trigram_logprob", 0.0),
            "is_fluent": fluent
        })

    # Category F: Multi-turn Context Continuity Test (4-turn sequential dialog)
    print("Running Category F Multi-Turn Dialogue Sequence...")
    dialogue_turns = [
        "I had a difficult day.",
        "Mostly because of college.",
        "I have an exam tomorrow too.",
        "Anyway, tell me a joke."
    ]
    accumulated_context = "<mood>calm_focused</mood>"
    turn_outputs = []

    for t_idx, u_msg in enumerate(dialogue_turns, 1):
        accumulated_context += f"<user>{u_msg}</user><edith>"
        t_out = generate_eval_sample(model, tokenizer, accumulated_context, device, max_new_tokens=40)
        accumulated_context += f"{t_out}</edith>"
        fluent_t, diag_t = FluencyGuard.inspect(t_out, query=u_msg)
        turn_outputs.append({
            "turn": t_idx,
            "user": u_msg,
            "edith": t_out,
            "is_fluent": fluent_t,
            "validity": diag_t.get("word_validity_ratio", 0.0)
        })

    # Write Markdown Report
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write("# E.D.I.T.H. V15.2: Comprehensive Language Quality & Generalization Report\n\n")
        f.write(f"- **Checkpoint**: `{os.path.basename(ckpt_path)}`\n")
        f.write(f"- **Completed Step**: {step:,}\n")
        f.write(f"- **Valid Targets Processed**: {targets_seen:,}\n")
        f.write(f"- **Held-Out Validation Perplexity**: {val_ppl:.2f} (Loss: {val_loss:.4f})\n")
        f.write(f"- **Fluency Pass Rate**: {fluent_passes}/{len(EVAL_PROMPTS_V15_2)} ({fluent_passes / len(EVAL_PROMPTS_V15_2):.1%})\n\n")
        
        f.write("## 1. Single-Turn Prompts Across Categories A–E\n\n")
        f.write("| ID | Category | Prompt | Raw Model Output | Word Valid % | Trigram Score | Fluent Pass |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in results:
            clean_out = r["output"].replace("|", "\\|").replace("\n", " ")
            f.write(f"| {r['id']} | {r['category']} | {r['prompt']} | {clean_out} | {r['word_validity']:.1%} | {r['trigram_score']:.2f} | {'PASS' if r['is_fluent'] else 'FAIL'} |\n")

        f.write("\n## 2. Category F: Multi-Turn Conversational Continuity Sequence\n\n")
        for t in turn_outputs:
            f.write(f"**Turn {t['turn']}**:\n")
            f.write(f"- **User**: *\"{t['user']}\"*\n")
            f.write(f"- **Edith**: *\"{t['edith']}\"* (Word Validity: {t['validity']:.1%}, Fluent: {t['is_fluent']})\n\n")

    print(f"Saved evaluation report to {out_md_path}")
    return results, turn_outputs


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_lm_v15", "edith_lm_v15_2_100m_best.pt")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(PROJECT_ROOT, "docs", "FLUENCY_V15_2_COMPREHENSIVE_EVAL.md")
    run_comprehensive_eval(ckpt, out)
