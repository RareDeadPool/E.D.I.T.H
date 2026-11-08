"""
E.D.I.T.H. V15.1 Generation Quality & Decoding Parameter Benchmark
Runs Phase 4 fixed prompt evaluation and Phase 5 decoding parameter comparison
(greedy, temperature, top-k, top-p, repetition penalty, multi-seed variability).
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.edith_lm.model import EdithLM, EdithLMConfig
from engine.edith_lm.tokenizer import EdithByteBPETokenizer
from engine.fluency_guard import FluencyGuard


def generate_with_config(
    model: EdithLM,
    tokenizer: EdithByteBPETokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 60,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    seed: Optional[int] = None
) -> str:
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    model.eval()
    if not prompt.startswith("<mood>"):
        formatted = f"<mood>calm_focused</mood><user>{prompt}</user><edith>"
    else:
        formatted = prompt

    prompt_ids = tokenizer.encode(formatted, add_bos=True, add_eos=False)
    generated = list(prompt_ids)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            curr_input = torch.tensor([generated[-512:]], dtype=torch.long, device=device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, _, _ = model(curr_input)
            
            next_token_logits = logits[0, -1, :].clone()

            # 1. Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated[len(prompt_ids):]):
                    if next_token_logits[token_id] > 0:
                        next_token_logits[token_id] /= repetition_penalty
                    else:
                        next_token_logits[token_id] *= repetition_penalty

            # 2. Temperature & Greedy check
            if temperature <= 0.01:
                next_token = torch.argmax(next_token_logits).item()
            else:
                next_token_logits = next_token_logits / temperature

                # 3. Top-K filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float("Inf")

                # 4. Top-P (Nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = -float("Inf")

                probs = F.softmax(next_token_logits, dim=-1)
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


def run_phase4_and_phase5():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_lm_v15", "edith_lm_v15_poc_25m_best.pt")
    tok_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_bpe_tokenizer_16k.json")

    tokenizer = EdithByteBPETokenizer(vocab_file=tok_path)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg = checkpoint.get("config", EdithLMConfig(vocab_size=16384, dim=576, n_layers=12, n_heads=6, n_kv_heads=2, intermediate_dim=1728, max_seq_len=512, tie_word_embeddings=True))
    model = EdithLM(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("=" * 80)
    print("PHASE 4: MANDATED GENERATION QUALITY TESTS ON FIXED PROMPTS")
    print("=" * 80)

    phase4_prompts = [
        {"domain": "General language", "prompt": "Explain why people sometimes disagree even when they have access to the same information."},
        {"domain": "Conversation", "prompt": "Hey Edith, how are you?"},
        {"domain": "Humour", "prompt": "Tell me a genuinely funny programming joke."},
        {"domain": "Common sense", "prompt": "Why would someone carry an umbrella when the sky is dark and cloudy?"},
        {"domain": "Explanation", "prompt": "Explain recursion to someone who is learning programming for the first time."},
        {"domain": "Reasoning", "prompt": "A student has three assignments due tomorrow, but only enough time to complete two properly. How should they decide what to prioritize?"},
        {"domain": "Indian English", "prompt": "Explain why somebody might feel frustrated after trying to find a job for several years."}
    ]

    phase4_results = []
    for item in phase4_prompts:
        out = generate_with_config(model, tokenizer, item["prompt"], device, max_new_tokens=60, temperature=0.7, top_k=40, seed=42)
        fluent, diag = FluencyGuard.inspect(out, query=item["prompt"])
        phase4_results.append({
            "domain": item["domain"],
            "prompt": item["prompt"],
            "output": out,
            "word_validity": diag.get("word_validity_ratio", 0.0),
            "trigram_score": diag.get("char_trigram_logprob", 0.0),
            "is_fluent": fluent,
            "reason": diag.get("reason", "")
        })
        print(f"\n[{item['domain'].upper()}] Prompt: \"{item['prompt']}\"")
        print(f"Output: \"{out}\"")
        print(f"Word Validity: {diag.get('word_validity_ratio', 0.0):.1%} | Trigram: {diag.get('char_trigram_logprob', 0.0):.2f} | Fluent: {fluent}")

    # Multi-turn Follow-up Test
    print("\n[MULTI-TURN FOLLOW-UP TEST]")
    turn1_prompt = "I had a difficult day today."
    turn1_out = generate_with_config(model, tokenizer, turn1_prompt, device, max_new_tokens=40, temperature=0.7, top_k=40, seed=42)
    turn2_context = f"<mood>calm_focused</mood><user>{turn1_prompt}</user><edith>{turn1_out}</edith><user>Mostly because of college.</user><edith>"
    turn2_out = generate_with_config(model, tokenizer, turn2_context, device, max_new_tokens=50, temperature=0.7, top_k=40, seed=42)
    fluent_t2, diag_t2 = FluencyGuard.inspect(turn2_out, query="Mostly because of college.")
    print(f"Turn 1 User: \"{turn1_prompt}\"")
    print(f"Turn 1 Edith: \"{turn1_out}\"")
    print(f"Turn 2 User: \"Mostly because of college.\"")
    print(f"Turn 2 Edith: \"{turn2_out}\"")
    print(f"Turn 2 Validity: {diag_t2.get('word_validity_ratio', 0.0):.1%} | Trigram: {diag_t2.get('char_trigram_logprob', 0.0):.2f} | Fluent: {fluent_t2}")

    phase4_results.append({
        "domain": "Follow-up (Multi-turn)",
        "prompt": "T1: 'I had a difficult day today.' -> T2: 'Mostly because of college.'",
        "output": f"T1: {turn1_out} | T2: {turn2_out}",
        "word_validity": diag_t2.get("word_validity_ratio", 0.0),
        "trigram_score": diag_t2.get("char_trigram_logprob", 0.0),
        "is_fluent": fluent_t2,
        "reason": diag_t2.get("reason", "")
    })

    print("\n" + "=" * 80)
    print("PHASE 5: DECODING PARAMETER SWEEP COMPARISON")
    print("=" * 80)

    test_prompt = "Explain why people sometimes disagree even when they have access to the same information."
    decoding_configs = [
        {"name": "Greedy (temp=0.0)", "temp": 0.0, "top_k": 0, "top_p": 1.0, "rep_pen": 1.0, "seed": 42},
        {"name": "Low Temp (temp=0.3, top_k=40)", "temp": 0.3, "top_k": 40, "top_p": 1.0, "rep_pen": 1.0, "seed": 42},
        {"name": "Balanced (temp=0.7, top_k=40)", "temp": 0.7, "top_k": 40, "top_p": 1.0, "rep_pen": 1.0, "seed": 42},
        {"name": "High Temp (temp=1.0, top_k=40)", "temp": 1.0, "top_k": 40, "top_p": 1.0, "rep_pen": 1.0, "seed": 42},
        {"name": "Nucleus (temp=0.7, top_p=0.85)", "temp": 0.7, "top_k": 0, "top_p": 0.85, "rep_pen": 1.0, "seed": 42},
        {"name": "Repetition Penalty 1.15", "temp": 0.7, "top_k": 40, "top_p": 1.0, "rep_pen": 1.15, "seed": 42},
        {"name": "Repetition Penalty 1.30", "temp": 0.7, "top_k": 40, "top_p": 1.0, "rep_pen": 1.30, "seed": 42},
        {"name": "Seed 101 Variation", "temp": 0.7, "top_k": 40, "top_p": 1.0, "rep_pen": 1.0, "seed": 101},
        {"name": "Seed 777 Variation", "temp": 0.7, "top_k": 40, "top_p": 1.0, "rep_pen": 1.0, "seed": 777},
    ]

    phase5_results = []
    for cfg_dec in decoding_configs:
        out = generate_with_config(
            model, tokenizer, test_prompt, device,
            max_new_tokens=50,
            temperature=cfg_dec["temp"],
            top_k=cfg_dec["top_k"],
            top_p=cfg_dec["top_p"],
            repetition_penalty=cfg_dec["rep_pen"],
            seed=cfg_dec["seed"]
        )
        fluent, diag = FluencyGuard.inspect(out, query=test_prompt)
        phase5_results.append({
            "config": cfg_dec["name"],
            "output": out,
            "word_validity": diag.get("word_validity_ratio", 0.0),
            "trigram_score": diag.get("char_trigram_logprob", 0.0),
            "is_fluent": fluent
        })
        print(f"\n[{cfg_dec['name']}]")
        print(f"Output: \"{out}\"")
        print(f"Word Validity: {diag.get('word_validity_ratio', 0.0):.1%} | Trigram: {diag.get('char_trigram_logprob', 0.0):.2f} | Fluent: {fluent}")

    # Save to disk for reports
    out_json = os.path.join(PROJECT_ROOT, "data_pipeline", "v15", "phase4_and_5_eval_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"phase4": phase4_results, "phase5": phase5_results}, f, indent=2)
    print(f"\nSaved evaluation results to {out_json}")


if __name__ == "__main__":
    run_phase4_and_phase5()

