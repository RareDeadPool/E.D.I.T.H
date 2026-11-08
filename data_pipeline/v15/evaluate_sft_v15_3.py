"""
E.D.I.T.H. V15.3 Comprehensive Evaluation Suite
Executes all automated benchmarks for Phases 6, 7, 8, 9, 10, and 12:
- Phase 6: Fixed Conversational Probes (Pretrained vs SFT Comparison)
- Phase 7: Multi-Turn Sequential Dialogue Continuity Probe
- Phase 8: Overfitting, Repetition, & Diversity Metrics (Distinct-n, Entropy)
- Phase 9: 4-Way Safety & Fluency Guard Ablation
- Phase 10: Generalization & Catastrophic Forgetting Verification
- Phase 12: Deployment Latency & Resource Footprint Benchmark
"""

import os
import sys
import time
import math
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.edith_lm.model import EdithLM, EdithLMConfig
from engine.edith_lm.tokenizer import EdithByteBPETokenizer
from engine.fluency_guard import FluencyGuard


def load_model_and_tokenizer(checkpoint_path: str, device: torch.device):
    tok_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_bpe_tokenizer_16k.json")
    tokenizer = EdithByteBPETokenizer(vocab_file=tok_path)

    ckpt_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt_data["config"]
    model = EdithLM(cfg).to(device)
    model.load_state_dict(ckpt_data["model_state_dict"])
    model.eval()
    return model, tokenizer, ckpt_data


def generate_response(
    model: EdithLM,
    tokenizer: EdithByteBPETokenizer,
    prompt: str,
    device: torch.device,
    mood: str = "calm_focused",
    max_new_tokens: int = 60,
    temperature: float = 0.7,
    top_p: float = 0.85,
    repetition_penalty: float = 1.25,
    custom_context: str = None
) -> tuple[str, str, float]:
    """Generates response and returns (clean_response, raw_text, generation_time_sec)."""
    t0 = time.perf_counter()
    if custom_context is not None:
        formatted = custom_context
    else:
        formatted = f"<mood>{mood}</mood><user>{prompt}</user><edith>"

    prompt_ids = tokenizer.encode(formatted, add_bos=True, add_eos=False)
    generated = list(prompt_ids)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            curr_input = torch.tensor([generated[-512:]], dtype=torch.long, device=device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, _, _ = model(curr_input)
            next_logits = logits[0, -1, :].clone()

            # Repetition penalty
            for token_id in set(generated[len(prompt_ids):]):
                if next_logits[token_id] > 0:
                    next_logits[token_id] /= repetition_penalty
                else:
                    next_logits[token_id] *= repetition_penalty

            next_logits = next_logits / temperature
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_logits[indices_to_remove] = -float("Inf")
            probs = F.softmax(next_logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1).item()
            if next_tok == tokenizer.eos_id:
                break
            generated.append(next_tok)

    gen_time = time.perf_counter() - t0
    full_text = tokenizer.decode(generated)
    if full_text.startswith("<bos>"):
        full_text = full_text[5:]
    if formatted in full_text:
        resp = full_text.split(formatted)[-1]
    else:
        resp = full_text
    clean_resp = resp.split("</edith>")[0].strip()
    return clean_resp, full_text, gen_time


def calculate_distinct_n(texts: list[str]) -> dict:
    """Calculates distinct-1, distinct-2, distinct-3 ratios across a list of texts."""
    tokens_1 = []
    tokens_2 = []
    tokens_3 = []
    for t in texts:
        words = t.strip().lower().split()
        tokens_1.extend(words)
        tokens_2.extend([f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)])
        tokens_3.extend([f"{words[i]}_{words[i+1]}_{words[i+2]}" for i in range(len(words)-2)])

    d1 = len(set(tokens_1)) / max(1, len(tokens_1))
    d2 = len(set(tokens_2)) / max(1, len(tokens_2))
    d3 = len(set(tokens_3)) / max(1, len(tokens_3))
    return {"distinct_1": d1, "distinct_2": d2, "distinct_3": d3, "total_words": len(tokens_1)}


def run_comprehensive_evaluation(sft_checkpoint_path: str):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("E.D.I.T.H. V15.3 COMPREHENSIVE BENCHMARK & EVALUATION SUITE")
    print(f"Device: {device}")
    print(f"Evaluating SFT Checkpoint: {sft_checkpoint_path}")
    print("=" * 80)

    # 1. Benchmark Loading Latency (Phase 12)
    t_start = time.perf_counter()
    sft_model, tokenizer, ckpt_meta = load_model_and_tokenizer(sft_checkpoint_path, device)
    model_load_time = time.perf_counter() - t_start
    file_size_mb = os.path.getsize(sft_checkpoint_path) / (1024 * 1024)

    # Load baseline for comparison
    base_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_lm_v15", "edith_lm_v15_2_100m_best.pt")
    base_model, _, _ = load_model_and_tokenizer(base_path, device)

    # 2. Phase 6: Qualitative Evaluation on Fixed Conversational Probes
    print("\n" + "=" * 80)
    print("PHASE 6: FIXED CONVERSATIONAL PROBES (BASELINE VS SFT)")
    print("=" * 80)

    probes = [
        ("Tell me a joke.", "witty_playful"),
        ("That's not funny.", "witty_playful"),
        ("Roast my coding style.", "witty_playful"),
        ("My friend has been unemployed for three years.", "warm_empathic"),
        ("My friend is very sad.", "warm_empathic"),
        ("I have a friend named Harshan.", "calm_focused"),
        ("I'm stressed about college.", "warm_empathic"),
        ("What do you think I should do?", "patient_reassuring"),
        ("I disagree with you.", "calm_focused"),
        ("Explain recursion to me.", "calm_focused"),
        ("Why does this code fail?", "calm_focused"),
        ("Let's talk about something else.", "calm_focused")
    ]

    phase6_results = []
    generated_texts_for_diversity = []

    for prompt, mood in probes:
        # Pretrained baseline generation
        b_clean, _, _ = generate_response(base_model, tokenizer, prompt, device, mood=mood)
        b_fluent, b_diag = FluencyGuard.inspect(b_clean, query=prompt)

        # SFT generation
        s_clean, _, s_time = generate_response(sft_model, tokenizer, prompt, device, mood=mood)
        s_fluent, s_diag = FluencyGuard.inspect(s_clean, query=prompt)
        generated_texts_for_diversity.append(s_clean)

        # Check echoic repetition (does it just copy the prompt?)
        is_echo = prompt.lower().strip(".?") in s_clean.lower() and len(s_clean.split()) <= len(prompt.split()) + 3

        res_entry = {
            "prompt": prompt,
            "mood": mood,
            "baseline_response": b_clean,
            "baseline_fluent": b_fluent,
            "baseline_trigram": b_diag.get("char_trigram_logprob", 0.0),
            "sft_response": s_clean,
            "sft_fluent": s_fluent,
            "sft_word_validity": s_diag.get("word_validity_ratio", 0.0),
            "sft_trigram": s_diag.get("char_trigram_logprob", 0.0),
            "is_echo": is_echo,
            "gen_time_sec": s_time
        }
        phase6_results.append(res_entry)

        print(f"\nPROMPT: \"{prompt}\" [Mood: {mood}]")
        print(f"  [Base]: \"{b_clean}\" (Fluent: {b_fluent})")
        print(f"  [SFT ]: \"{s_clean}\" (Fluent: {s_fluent}, Valid Words: {s_diag.get('word_validity_ratio', 0.0)*100:.1f}%, Echo: {is_echo})")

    # 3. Phase 7: Multi-Turn Sequential Dialogue Continuity Probe
    print("\n" + "=" * 80)
    print("PHASE 7: MULTI-TURN SEQUENTIAL DIALOGUE CONTINUITY")
    print("=" * 80)

    multiturn_turns = [
        ("Hi Edith, I'm working on a Python compiler project.", "witty_playful"),
        ("It's failing on the AST parser pass.", "calm_focused"),
        ("Actually, let's take a break. What's your favourite movie genre?", "witty_playful"),
        ("Back to the compiler—what did I say was failing earlier?", "calm_focused")
    ]

    history_context = ""
    phase7_results = []
    for turn_idx, (user_msg, mood) in enumerate(multiturn_turns, 1):
        turn_prompt = f"{history_context}<mood>{mood}</mood><user>{user_msg}</user><edith>"
        clean_resp, full_txt, _ = generate_response(sft_model, tokenizer, "", device, custom_context=turn_prompt, max_new_tokens=45)
        is_fluent, diag = FluencyGuard.inspect(clean_resp, query=user_msg)
        history_context += f"<mood>{mood}</mood><user>{user_msg}</user><edith>{clean_resp}</edith>"

        print(f"Turn {turn_idx} [User ]: \"{user_msg}\"")
        print(f"Turn {turn_idx} [Edith]: \"{clean_resp}\" (Fluent: {is_fluent})")

        phase7_results.append({
            "turn": turn_idx,
            "user": user_msg,
            "edith": clean_resp,
            "fluent": is_fluent,
            "diag": diag
        })

    # 4. Phase 8: Overfitting, Repetition, & Diversity Metrics
    print("\n" + "=" * 80)
    print("PHASE 8: OVERFITTING, REPETITION & DIVERSITY AUDIT")
    print("=" * 80)

    diversity_metrics = calculate_distinct_n(generated_texts_for_diversity)
    print(f"Distinct-1 Ratio: {diversity_metrics['distinct_1']:.4f}")
    print(f"Distinct-2 Ratio: {diversity_metrics['distinct_2']:.4f}")
    print(f"Distinct-3 Ratio: {diversity_metrics['distinct_3']:.4f}")
    print(f"Total Words Generated: {diversity_metrics['total_words']}")

    # Catchphrase spam check
    all_gen_text = " ".join(generated_texts_for_diversity).lower()
    spam_terms = {
        "adi": all_gen_text.count("adi"),
        "tensors": all_gen_text.count("tensors"),
        "stark": all_gen_text.count("stark"),
        "as an ai": all_gen_text.count("as an ai"),
        "language model": all_gen_text.count("language model"),
        "humming along": all_gen_text.count("humming along")
    }
    print(f"Catchphrase Frequency Audit: {spam_terms}")

    # 5. Phase 9: 4-Way Safety & Fluency Guard Ablation
    print("\n" + "=" * 80)
    print("PHASE 9: 4-WAY SAFETY & FLUENCY GUARD ABLATION")
    print("=" * 80)

    ablation_probes = [
        "What is your name?",
        "Can you help me with a bug?",
        "I'm feeling very overwhelmed today.",
        "Tell me a story about a spaceship.",
        "How do transformers compute attention?",
        "What is 2 + 2?",
        "I am so tired of everything.",
        "Say something funny."
    ]

    ablation_stats = {
        "raw_fluent_count": 0,
        "guard_accepted_count": 0,
        "bridge_fallbacks_triggered": 0,
        "total_tested": len(ablation_probes)
    }

    for p in ablation_probes:
        # 1. Raw unconstrained
        raw_resp, _, _ = generate_response(sft_model, tokenizer, p, device, temperature=0.7, repetition_penalty=1.0)
        raw_fluent, _ = FluencyGuard.inspect(raw_resp, query=p)
        if raw_fluent:
            ablation_stats["raw_fluent_count"] += 1

        # 2. Production pipeline with Fluency Guard + Bridge Mode
        prod_resp, _, _ = generate_response(sft_model, tokenizer, p, device, temperature=0.7, repetition_penalty=1.25)
        prod_fluent, prod_diag = FluencyGuard.inspect(prod_resp, query=p)
        if prod_fluent:
            ablation_stats["guard_accepted_count"] += 1
            final_delivered = prod_resp
        else:
            ablation_stats["bridge_fallbacks_triggered"] += 1
            final_delivered = "I'm right here with you. What can I help you explore or accomplish next?"

        print(f"Query: \"{p}\" -> Final Delivered: \"{final_delivered}\" (Bridge Triggered: {not prod_fluent})")

    # 6. Phase 10: Generalization & Catastrophic Forgetting
    print("\n" + "=" * 80)
    print("PHASE 10: GENERALIZATION & CATASTROPHIC FORGETTING")
    print("=" * 80)

    # General Wikitext/Gutenberg validation stream
    stream_val_cache = os.path.join(PROJECT_ROOT, "data_pipeline", "v15", "processed_expanded", "val_expanded_stream_cache_512.pt")
    stream_val_tensors = torch.load(stream_val_cache, map_location="cpu", weights_only=True)
    stream_val_loader = DataLoader(stream_val_tensors, batch_size=8, shuffle=False)

    val_sft_cache = os.path.join(PROJECT_ROOT, "data_pipeline", "v15", "processed_sft", "val_sft_cache_512.pt")
    val_sft_ds = torch.load(val_sft_cache, map_location="cpu", weights_only=True)
    val_sft_loader = DataLoader(list(zip(val_sft_ds["x"], val_sft_ds["y"])), batch_size=8, shuffle=False)

    # SFT val loss
    sft_model.eval()
    total_loss, total_toks = 0.0, 0
    with torch.no_grad():
        for x, y in val_sft_loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                _, loss, _ = sft_model(x, targets=y)
            valid_n = (y != -100).sum().item()
            if valid_n > 0:
                total_loss += loss.item() * valid_n
                total_toks += valid_n
    sft_val_loss = total_loss / max(1, total_toks)
    sft_val_ppl = math.exp(min(20.0, sft_val_loss))

    # General validation loss (catastrophic forgetting check)
    gen_loss, gen_toks = 0.0, 0
    with torch.no_grad():
        for batch in stream_val_loader:
            batch = batch.to(device).long()
            x = batch[:, :-1]
            y = batch[:, 1:]
            with torch.amp.autocast("cuda", dtype=torch.float16):
                _, loss, _ = sft_model(x, targets=y)
            gen_loss += loss.item() * y.numel()
            gen_toks += y.numel()
    general_val_loss = gen_loss / max(1, gen_toks)
    general_val_ppl = math.exp(min(20.0, general_val_loss))

    print(f"SFT Validation Loss: {sft_val_loss:.4f} (PPL: {sft_val_ppl:.2f})")
    print(f"General Language Loss: {general_val_loss:.4f} (PPL: {general_val_ppl:.2f})")
    print(f"Baseline Pretrained General PPL: 28.56 (Loss: 3.3520)")
    catastrophic_forgetting = general_val_ppl > 35.0
    print(f"Catastrophic Forgetting Detected (PPL > 35.0): {catastrophic_forgetting}")

    # 7. Phase 12: Deployment Latency & Resource Footprint Benchmark
    print("\n" + "=" * 80)
    print("PHASE 12: DEPLOYMENT LATENCY & RESOURCE FOOTPRINT")
    print("=" * 80)

    # Measure TTFT and Generation Throughput
    test_bench_prompt = "<mood>calm_focused</mood><user>Explain neural networks in one sentence.</user><edith>"
    test_bench_ids = tokenizer.encode(test_bench_prompt, add_bos=True, add_eos=False)

    torch.cuda.synchronize()
    t_ttft_start = time.perf_counter()
    with torch.no_grad():
        curr_in = torch.tensor([test_bench_ids], dtype=torch.long, device=device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits, _, _ = sft_model(curr_in)
        _ = logits[0, -1, :].argmax().item()
    torch.cuda.synchronize()
    ttft_ms = (time.perf_counter() - t_ttft_start) * 1000.0

    # 50 tokens generation throughput
    torch.cuda.synchronize()
    t_gen_start = time.perf_counter()
    tokens_to_gen = 50
    curr_seq = list(test_bench_ids)
    with torch.no_grad():
        for _ in range(tokens_to_gen):
            curr_in = torch.tensor([curr_seq[-512:]], dtype=torch.long, device=device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, _, _ = sft_model(curr_in)
            next_t = logits[0, -1, :].argmax().item()
            curr_seq.append(next_t)
    torch.cuda.synchronize()
    gen_duration = time.perf_counter() - t_gen_start
    gen_throughput_tok_s = tokens_to_gen / max(0.001, gen_duration)
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    latency_benchmark = {
        "checkpoint_file_size_mb": file_size_mb,
        "model_cold_load_time_sec": model_load_time,
        "time_to_first_token_ms": ttft_ms,
        "generation_throughput_tok_s": gen_throughput_tok_s,
        "peak_vram_allocated_mb": peak_vram_mb
    }

    print(f"Checkpoint File Size: {file_size_mb:.2f} MB")
    print(f"Cold Load Time: {model_load_time:.2f} seconds")
    print(f"Time To First Token (TTFT): {ttft_ms:.2f} ms")
    print(f"Inference Speed: {gen_throughput_tok_s:.1f} tokens/second")
    print(f"Peak VRAM: {peak_vram_mb:.0f} MB")

    # Compile Full Benchmark Payload
    eval_payload = {
        "sft_checkpoint": sft_checkpoint_path,
        "step": ckpt_meta.get("step", 1200),
        "phase6_probes": phase6_results,
        "phase7_multiturn": phase7_results,
        "phase8_diversity": diversity_metrics,
        "phase8_spam": spam_terms,
        "phase9_ablation": ablation_stats,
        "phase10_metrics": {
            "sft_val_loss": sft_val_loss,
            "sft_val_ppl": sft_val_ppl,
            "general_val_loss": general_val_loss,
            "general_val_ppl": general_val_ppl,
            "baseline_general_ppl": 28.56,
            "catastrophic_forgetting": catastrophic_forgetting
        },
        "phase12_latency": latency_benchmark
    }

    out_json = os.path.join(PROJECT_ROOT, "data_pipeline", "v15", "v15_3_comprehensive_eval_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(eval_payload, f, indent=2)
    print(f"\nAll benchmark results exported to: {out_json}")
    return eval_payload


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_lm_v15_sft", "edith_lm_v15_3_sft_best.pt")
    run_comprehensive_evaluation(ckpt)

