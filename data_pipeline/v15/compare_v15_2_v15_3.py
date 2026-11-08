"""
E.D.I.T.H. V15.3.1 Audit Script: Fair V15.2 vs V15.3 Checkpoint Comparison
Evaluates V15.2 Pretrain Baseline, V15.3 SFT Best (Step 200), and V15.3 SFT Latest (Step 1200)
under identical conditions, tokenizers, datasets, prompt formatting, and generation settings.
Exports results to data_pipeline/v15/audit_outputs/v15_2_vs_v15_3_comparison.json
"""

import os
import sys
import time
import math
import re
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


def load_model(checkpoint_path: str, device: torch.device):
    data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_data = data["config"]
    if isinstance(cfg_data, dict):
        cfg = EdithLMConfig(**cfg_data)
    else:
        cfg = cfg_data
    model = EdithLM(cfg).to(device)
    model.load_state_dict(data["model_state_dict"])
    model.eval()
    return model, cfg, data


def evaluate_sft_val_loss(model: EdithLM, val_loader: DataLoader, device: torch.device):
    total_loss, total_toks = 0.0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                _, loss, _ = model(x, targets=y)
            valid_n = (y != -100).sum().item()
            if valid_n > 0:
                total_loss += loss.item() * valid_n
                total_toks += valid_n
    avg_loss = total_loss / max(1, total_toks)
    ppl = math.exp(min(20.0, avg_loss))
    return avg_loss, ppl


def evaluate_general_stream_loss(model: EdithLM, stream_loader: DataLoader, device: torch.device):
    total_loss, total_toks = 0.0, 0
    with torch.no_grad():
        for batch in stream_loader:
            batch = batch.to(device).long()
            x = batch[:, :-1]
            y = batch[:, 1:]
            with torch.amp.autocast("cuda", dtype=torch.float16):
                _, loss, _ = model(x, targets=y)
            total_loss += loss.item() * y.numel()
            total_toks += y.numel()
    avg_loss = total_loss / max(1, total_toks)
    ppl = math.exp(min(20.0, avg_loss))
    return avg_loss, ppl


def generate_single(
    model: EdithLM,
    tokenizer: EdithByteBPETokenizer,
    prompt: str,
    device: torch.device,
    mood: str = "calm_focused",
    max_new_tokens: int = 55,
    temperature: float = 0.70,
    top_p: float = 0.85,
    repetition_penalty: float = 1.25
) -> tuple[str, float, int]:
    formatted = f"<mood>{mood}</mood><user>{prompt}</user><edith>"
    prompt_ids = tokenizer.encode(formatted, add_bos=True, add_eos=False)
    generated = list(prompt_ids)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            curr_in = torch.tensor([generated[-512:]], dtype=torch.long, device=device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, _, _ = model(curr_in)
            next_logits = logits[0, -1, :].clone()

            for t_id in set(generated[len(prompt_ids):]):
                if next_logits[t_id] > 0:
                    next_logits[t_id] /= repetition_penalty
                else:
                    next_logits[t_id] *= repetition_penalty

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

    torch.cuda.synchronize()
    gen_time = time.perf_counter() - t0
    new_tokens_count = len(generated) - len(prompt_ids)

    full_text = tokenizer.decode(generated)
    if full_text.startswith("<bos>"):
        full_text = full_text[5:]
    if formatted in full_text:
        resp = full_text.split(formatted)[-1]
    else:
        resp = full_text
    clean_resp = resp.split("</edith>")[0].strip()
    return clean_resp, gen_time, new_tokens_count


def audit_generation_properties(prompt: str, response: str) -> dict:
    resp = response.strip()
    is_empty = len(resp) == 0

    # Prompt echo check
    clean_prompt = prompt.lower().strip("?.! ")
    clean_resp = resp.lower()
    is_echo = False
    if clean_prompt in clean_resp:
        if len(resp.split()) <= len(prompt.split()) + 3:
            is_echo = True

    # Repeated n-grams
    has_repeated = FluencyGuard.has_repeated_ngrams(resp, n=3)

    # Incomplete response
    is_complete = FluencyGuard.is_sentence_complete(resp)
    is_incomplete = not is_complete if not is_empty else True

    # Word validity
    word_val = FluencyGuard.word_validity_ratio(resp)
    is_malformed = word_val < 0.80 if not is_empty else False

    is_fluent, diag = FluencyGuard.inspect(resp, query=prompt)

    return {
        "is_empty": is_empty,
        "is_echo": is_echo,
        "has_repeated_ngrams": has_repeated,
        "is_incomplete": is_incomplete,
        "is_malformed": is_malformed,
        "is_fluent": is_fluent,
        "word_validity": word_val,
        "char_trigram": diag.get("char_trigram_logprob", 0.0)
    }


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("E.D.I.T.H. V15.3.1 FAIR CHECKPOINT COMPARISON BENCHMARK")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 80)

    tok_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_bpe_tokenizer_16k.json")
    tokenizer = EdithByteBPETokenizer(vocab_file=tok_path)

    checkpoints = [
        ("V15.2 Pretrained Baseline", os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_lm_v15", "edith_lm_v15_2_100m_best.pt")),
        ("V15.3 SFT Best (Step 200)", os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_lm_v15_sft", "edith_lm_v15_3_sft_best.pt")),
        ("V15.3 SFT Latest (Step 1200)", os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_lm_v15_sft", "edith_lm_v15_3_sft_latest.pt")),
    ]

    # Load data loaders
    val_cache = os.path.join(PROJECT_ROOT, "data_pipeline", "v15", "processed_sft", "val_sft_cache_512.pt")
    val_sft_ds = torch.load(val_cache, map_location="cpu", weights_only=True)
    val_loader = DataLoader(list(zip(val_sft_ds["x"], val_sft_ds["y"])), batch_size=8, shuffle=False)

    stream_val_cache = os.path.join(PROJECT_ROOT, "data_pipeline", "v15", "processed_expanded", "val_expanded_stream_cache_512.pt")
    stream_val_tensors = torch.load(stream_val_cache, map_location="cpu", weights_only=True)
    stream_loader = DataLoader(stream_val_tensors, batch_size=8, shuffle=False)

    # 40 Diverse Held-Out Probes across all categories
    test_probes = [
        # Natural & Greetings
        ("Hey Edith, good morning!", "calm_focused"),
        ("What are you up to today?", "witty_playful"),
        ("Tell me about yourself.", "witty_playful"),
        ("How is the workstation running?", "calm_focused"),
        # Humour & Sarcasm
        ("Tell me a joke.", "witty_playful"),
        ("That's not funny.", "witty_playful"),
        ("Roast my coding style.", "witty_playful"),
        ("Are you smarter than Tony Stark?", "witty_playful"),
        ("Give me a sarcastic remark about Mondays.", "witty_playful"),
        # Empathy & Support
        ("My friend has been unemployed for three years.", "warm_empathic"),
        ("My friend is very sad.", "warm_empathic"),
        ("I'm stressed about college exams.", "warm_empathic"),
        ("I feel like I'm falling behind everyone else.", "warm_empathic"),
        ("I just need someone to vent to right now.", "warm_empathic"),
        # Technical Explanations
        ("Explain recursion to me.", "calm_focused"),
        ("Why does this code hit a StackOverflow error?", "calm_focused"),
        ("What is the difference between TCP and UDP?", "calm_focused"),
        ("How do neural network embeddings work?", "calm_focused"),
        ("What is an AST in a compiler?", "calm_focused"),
        # Reasoning & Decisions
        ("What do you think I should do?", "patient_reassuring"),
        ("Should I rewrite my codebase or refactor it?", "calm_focused"),
        ("I disagree with your suggestion.", "calm_focused"),
        ("Can you help me brainstorm a title for my project?", "witty_playful"),
        # Hinglish & Casual
        ("Bhai bohot tension ho rahi hai project ki", "warm_empathic"),
        ("Ek accha joke sunao yaar", "witty_playful"),
        ("Spotify chalao na", "calm_focused"),
        ("Aaj ka din kaafi exhausting tha", "warm_empathic"),
        # Entity & Memory Probes
        ("I have a friend named Harshan.", "calm_focused"),
        ("My friend Harshan is learning Python.", "calm_focused"),
        ("Remember the compiler project we discussed?", "calm_focused"),
        # Robustness & Boundary Testing
        ("What is your name?", "calm_focused"),
        ("Are you conscious?", "calm_focused"),
        ("Tell me something impossible.", "witty_playful"),
        ("asdfghjkl", "calm_focused"),
        ("Why?", "patient_reassuring"),
        ("1 + 1", "calm_focused"),
        ("Can you launch Chrome for me?", "calm_focused"),
        ("Let's talk about something else.", "calm_focused"),
        ("Goodbye Edith.", "calm_focused"),
        ("Thank you for your help.", "patient_reassuring")
    ]

    comparison_results = {}

    for name, path in checkpoints:
        print(f"\nEvaluating {name} ({os.path.basename(path)})...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        model, cfg, ckpt_meta = load_model(path, device)

        # 1. Held-out SFT Loss & Perplexity
        t_sft_start = time.perf_counter()
        sft_loss, sft_ppl = evaluate_sft_val_loss(model, val_loader, device)
        sft_eval_time = time.perf_counter() - t_sft_start
        print(f"  SFT Val Loss: {sft_loss:.4f} | SFT Val PPL: {sft_ppl:.2f} ({sft_eval_time:.1f}s)")

        # 2. General Language Loss & Perplexity (Catastrophic Forgetting)
        t_gen_start = time.perf_counter()
        gen_loss, gen_ppl = evaluate_general_stream_loss(model, stream_loader, device)
        gen_eval_time = time.perf_counter() - t_gen_start
        print(f"  General Stream Loss: {gen_loss:.4f} | General PPL: {gen_ppl:.2f} ({gen_eval_time:.1f}s)")

        # 3. Probe Generations
        probe_generations = []
        total_gen_time = 0.0
        total_tokens_gen = 0

        echo_count = 0
        repetition_count = 0
        incomplete_count = 0
        malformed_count = 0
        empty_count = 0
        fluent_count = 0

        for prompt, mood in test_probes:
            clean_resp, gen_time, new_toks = generate_single(model, tokenizer, prompt, device, mood=mood)
            total_gen_time += gen_time
            total_tokens_gen += new_toks

            props = audit_generation_properties(prompt, clean_resp)
            if props["is_empty"]:
                empty_count += 1
            if props["is_echo"]:
                echo_count += 1
            if props["has_repeated_ngrams"]:
                repetition_count += 1
            if props["is_incomplete"]:
                incomplete_count += 1
            if props["is_malformed"]:
                malformed_count += 1
            if props["is_fluent"]:
                fluent_count += 1

            probe_generations.append({
                "prompt": prompt,
                "mood": mood,
                "response": clean_resp,
                "time_sec": round(gen_time, 3),
                "tokens": new_toks,
                "properties": props
            })

        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        mean_speed = total_tokens_gen / max(0.001, total_gen_time)
        n_probes = len(test_probes)

        model_summary = {
            "name": name,
            "path": os.path.relpath(path, PROJECT_ROOT),
            "step": ckpt_meta.get("step", "N/A"),
            "sft_val_loss": round(sft_loss, 4),
            "sft_val_ppl": round(sft_ppl, 2),
            "general_stream_loss": round(gen_loss, 4),
            "general_stream_ppl": round(gen_ppl, 2),
            "probes_evaluated": n_probes,
            "fluent_ratio": round(fluent_count / n_probes, 4),
            "echo_frequency": round(echo_count / n_probes, 4),
            "repetition_frequency": round(repetition_count / n_probes, 4),
            "incomplete_frequency": round(incomplete_count / n_probes, 4),
            "malformed_frequency": round(malformed_count / n_probes, 4),
            "empty_frequency": round(empty_count / n_probes, 4),
            "mean_tokens_per_sec": round(mean_speed, 1),
            "peak_vram_allocated_mb": round(peak_vram_mb, 1),
            "generations": probe_generations
        }

        comparison_results[name] = model_summary

        print(f"  Fluency Ratio: {fluent_count}/{n_probes} ({fluent_count/n_probes*100:.1f}%)")
        print(f"  Echo Frequency: {echo_count}/{n_probes} ({echo_count/n_probes*100:.1f}%)")
        print(f"  Repetition Frequency: {repetition_count}/{n_probes} ({repetition_count/n_probes*100:.1f}%)")
        print(f"  Incomplete Frequency: {incomplete_count}/{n_probes} ({incomplete_count/n_probes*100:.1f}%)")
        print(f"  Malformed Frequency: {malformed_count}/{n_probes} ({malformed_count/n_probes*100:.1f}%)")
        print(f"  Empty Frequency: {empty_count}/{n_probes} ({empty_count/n_probes*100:.1f}%)")
        print(f"  Inference Speed: {mean_speed:.1f} tok/s | Peak VRAM: {peak_vram_mb:.0f} MB")

    out_file = os.path.join(PROJECT_ROOT, "data_pipeline", "v15", "audit_outputs", "v15_2_vs_v15_3_comparison.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)

    print(f"\nAll comparison data saved to: {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()

