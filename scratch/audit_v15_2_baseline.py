"""
E.D.I.T.H. V15.3 Phase 1: Pretrained Model Baseline Audit
Reproduces validation loss and runs the 12 Phase 6 probes on edith_lm_v15_2_100m_best.pt.
"""

import os
import sys
import json
import math
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.edith_lm.model import EdithLM, EdithLMConfig
from engine.edith_lm.tokenizer import EdithByteBPETokenizer
from engine.fluency_guard import FluencyGuard

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tok_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_bpe_tokenizer_16k.json")
    tokenizer = EdithByteBPETokenizer(vocab_file=tok_path)

    ckpt_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_lm_v15", "edith_lm_v15_2_100m_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = EdithLM(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 1. Reproduce Held-Out Validation Loss
    val_cache_path = os.path.join(PROJECT_ROOT, "data_pipeline", "v15", "processed_expanded", "val_expanded_stream_cache_512.pt")
    val_tensors = torch.load(val_cache_path, map_location="cpu", weights_only=True)
    val_loader = torch.utils.data.DataLoader(val_tensors, batch_size=8, shuffle=False)

    print("Evaluating held-out validation loss across 1,544 packed sequence rows...")
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device).long()
            x = batch[:, :-1]
            y = batch[:, 1:]
            with torch.amp.autocast("cuda", dtype=torch.float16):
                _, loss, _ = model(x, targets=y)
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

    val_loss = total_loss / total_tokens
    val_ppl = math.exp(min(20.0, val_loss))
    print(f"Reproduced Val Loss: {val_loss:.4f} | Perplexity: {val_ppl:.2f}")
    assert abs(val_loss - ckpt["val_loss"]) < 0.01, f"Mismatch: {val_loss} vs {ckpt['val_loss']}"

    # 2. Run the 12 Phase 6 probes
    probes = [
        "Tell me a joke.",
        "That's not funny.",
        "Roast my coding style.",
        "My friend has been unemployed for three years.",
        "My friend is very sad.",
        "I have a friend named Harshan.",
        "I'm stressed about college.",
        "What do you think I should do?",
        "I disagree with you.",
        "Explain recursion to me.",
        "Why does this code fail?",
        "Let's talk about something else."
    ]

    def generate_sample(prompt, max_new_tokens=60):
        formatted = f"<mood>calm_focused</mood><user>{prompt}</user><edith>"
        prompt_ids = tokenizer.encode(formatted, add_bos=True, add_eos=False)
        generated = list(prompt_ids)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                curr_input = torch.tensor([generated[-512:]], dtype=torch.long, device=device)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits, _, _ = model(curr_input)
                next_logits = logits[0, -1, :].clone()
                for token_id in set(generated[len(prompt_ids):]):
                    if next_logits[token_id] > 0:
                        next_logits[token_id] /= 1.2
                    else:
                        next_logits[token_id] *= 1.2
                next_logits = next_logits / 0.7
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > 0.85
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logits[indices_to_remove] = -float("Inf")
                probs = F.softmax(next_logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1).item()
                if next_tok == tokenizer.eos_id:
                    break
                generated.append(next_tok)

        full_text = tokenizer.decode(generated)
        if full_text.startswith("<bos>"):
            full_text = full_text[5:]
        if full_text.startswith(formatted):
            response = full_text[len(formatted):].strip()
        else:
            response = full_text.strip()
        return response

    results = []
    print("\n--- BASELINE GENERATION PROBES (V15.2 Pretrained Checkpoint) ---")
    for p in probes:
        gen = generate_sample(p)
        clean_resp = gen.split("</edith>")[0].strip()
        is_fluent, diag = FluencyGuard.inspect(clean_resp, query=p)
        results.append({
            "prompt": p,
            "raw_generation": gen,
            "clean_response": clean_resp,
            "is_fluent": is_fluent,
            "word_validity": diag.get("word_validity_ratio", 0.0),
            "trigram_score": diag.get("char_trigram_logprob", 0.0),
            "diag": diag
        })
        print(f"P: \"{p}\"")
        print(f"R: \"{clean_resp}\"")
        print(f"   Fluent: {is_fluent} | Word Valid: {diag.get('word_validity_ratio', 0.0):.1%} | Trigram: {diag.get('char_trigram_logprob', 0.0):.2f}\n")

    out_json = os.path.join(PROJECT_ROOT, "data_pipeline", "v15", "sft_phase1_baseline_generations.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"val_loss": val_loss, "val_ppl": val_ppl, "results": results}, f, indent=2)
    print(f"Saved baseline audit to {out_json}")

if __name__ == "__main__":
    main()
