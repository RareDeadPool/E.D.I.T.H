"""
E.D.I.T.H. V15 Fluency Gate Evaluator (50 Fixed Probe Prompts)
Evaluates RAW unfiltered generations directly from the custom EdithLM checkpoint.
NO router, NO verbalizer, NO bridge mode, NO post-processing filters.
Measures:
1. Word-validity ratio (target >= 95%)
2. Char-trigram readability score (target >= -7.8)
3. Repeated n-grams and completeness
4. Unedited markdown report for user inspection
"""

import os
import sys
import json
import math
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.edith_lm.model import EdithLM, EdithLMConfig
from engine.edith_lm.tokenizer import EdithByteBPETokenizer
from engine.fluency_guard import FluencyGuard

PROBE_50 = [
    # 1. Chit-chat & Introductions (8)
    "Hello there!",
    "Who are you?",
    "What is your name?",
    "Tell me something about yourself.",
    "How are you doing today?",
    "Good morning, ready to work?",
    "What can you do?",
    "Are you an AI or a person?",

    # 2. Opinion & User Self-Reference (10)
    "describe me",
    "What do you think about me?",
    "Do you know who I am?",
    "What do you remember about our past chats?",
    "Tell me what you think my strengths are.",
    "Am I working too hard?",
    "Do you think I will succeed in building you?",
    "What do you think of my coding habits?",
    "Describe our relationship.",
    "Be honest: what annoys you about me?",

    # 3. Follow-ups & Dialogue State (8)
    "Really?",
    "Why?",
    "What happened next?",
    "Tell me more about that.",
    "And then?",
    "Are you sure about that?",
    "What do you mean by that?",
    "Can you explain further?",

    # 4. Feelings & Empathy (8)
    "I had an exhausting day at work.",
    "I am feeling really anxious about tomorrow's deadline.",
    "Nobody appreciates how much effort I put into this.",
    "I just got promoted today!",
    "I feel like giving up on this project.",
    "I am so tired but I still have so much work to do.",
    "Everything feels overwhelming right now.",
    "I feel great today, full of energy!",

    # 5. Hinglish & Informal Conversational (8)
    "bhai aaj bohot bekar din tha, boss ne daant diya",
    "kya chal raha hai aajkal?",
    "aaj bilkul man nahi kar raha code karne ka",
    "bhai ek cup chai ho jaye?",
    "tu sach me smart hai ya bas show off karta hai?",
    "kuch mazedaar batao na",
    "aaj thoda relax karte hain",
    "dimag ka dahi ho gaya aaj ke bug solve karte karte",

    # 6. Factual & Reasoning (8)
    "What is the difference between Python and C++?",
    "Why is the sky blue?",
    "What is 25 plus 17?",
    "What is the capital of France?",
    "Explain how a Transformer neural network works in simple terms.",
    "What happens when water freezes?",
    "Who was Albert Einstein?",
    "What is the purpose of an operating system?"
]


def generate_raw(model: EdithLM, tokenizer: EdithByteBPETokenizer, prompt: str, device: torch.device, max_new_tokens: int = 50, temperature: float = 0.7, top_k: int = 40) -> str:
    model.eval()
    formatted = f"<mood>calm_focused</mood><user>{prompt}</user><edith>"
    prompt_ids = tokenizer.encode(formatted, add_bos=True, add_eos=False)
    generated = list(prompt_ids)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            curr_input = torch.tensor([generated[-512:]], dtype=torch.long, device=device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, _, _ = model(curr_input)
            next_token_logits = logits[0, -1, :] / temperature
            topk_logits, topk_indices = torch.topk(next_token_logits, k=top_k)
            probs = F.softmax(topk_logits, dim=-1)
            next_token_idx = torch.multinomial(probs, num_samples=1)
            next_token = topk_indices[next_token_idx].item()

            if next_token == tokenizer.eos_id:
                break
            generated.append(next_token)

    full_text = tokenizer.decode(generated)
    if full_text.startswith("<bos>"):
        full_text = full_text[5:]
    if full_text.startswith(formatted):
        return full_text[len(formatted):].strip()
    return full_text.strip()


def run_probe_evaluation(ckpt_path: str, output_md_path: str):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg = checkpoint.get("config")
    if cfg is None:
        cfg = EdithLMConfig(vocab_size=16384, dim=576, n_layers=12, n_heads=6, n_kv_heads=2, intermediate_dim=1728, max_seq_len=512, tie_word_embeddings=True)

    tok_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_bpe_tokenizer_16k.json")
    tokenizer = EdithByteBPETokenizer(vocab_file=tok_path)

    model = EdithLM(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokens_seen = checkpoint.get("tokens_seen", 0)
    val_loss = checkpoint.get("val_loss", 0.0)
    val_ppl = checkpoint.get("val_ppl", math.exp(val_loss) if val_loss else 0.0)
    step = checkpoint.get("step", 0)

    print(f"Evaluating {len(PROBE_50)} fixed probe prompts on RAW model output (Tokens seen: {tokens_seen:,})...")

    results = []
    pass_validity_count = 0
    pass_trigram_count = 0
    pass_fluent_count = 0

    for idx, prompt in enumerate(PROBE_50, 1):
        raw_output = generate_raw(model, tokenizer, prompt, device)
        is_fluent, diag = FluencyGuard.inspect(raw_output, query=prompt)

        v_ratio = diag.get("word_validity_ratio", 0.0)
        tri_score = diag.get("char_trigram_logprob", 0.0)

        if v_ratio >= 0.95:
            pass_validity_count += 1
        if tri_score >= -7.8:
            pass_trigram_count += 1
        if is_fluent:
            pass_fluent_count += 1

        results.append({
            "idx": idx,
            "prompt": prompt,
            "raw_output": raw_output,
            "word_validity": v_ratio,
            "char_trigram": tri_score,
            "is_fluent": is_fluent,
            "diagnostics": diag
        })

    validity_rate = pass_validity_count / len(PROBE_50)
    trigram_rate = pass_trigram_count / len(PROBE_50)
    fluent_rate = pass_fluent_count / len(PROBE_50)

    print("\n" + "=" * 80)
    print("FLUENCY GATE EVALUATION SUMMARY (50 PROBES, RAW MODEL OUTPUT):")
    print(f"  Step:                   {step:,}")
    print(f"  Tokens Trained:         {tokens_seen:,}")
    print(f"  Held-out Val Loss:      {val_loss:.4f}")
    print(f"  Held-out Perplexity:    {val_ppl:.2f}")
    print(f"  Word-Validity >= 95%:   {pass_validity_count}/{len(PROBE_50)} ({validity_rate:.1%})  [Target: >= 90%]")
    print(f"  Readability >= -7.8:    {pass_trigram_count}/{len(PROBE_50)} ({trigram_rate:.1%})  [Target: >= 90%]")
    print(f"  Overall Fluency Pass:   {pass_fluent_count}/{len(PROBE_50)} ({fluent_rate:.1%})")
    print("=" * 80)

    # Write Markdown Appendix
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write("# E.D.I.T.H. V15: 50 Raw Generative Probe Report\n\n")
        f.write(f"- **Checkpoint**: `{os.path.basename(ckpt_path)}`\n")
        f.write(f"- **Step**: {step:,} | **Tokens Seen**: {tokens_seen:,}\n")
        f.write(f"- **Held-Out Validation Perplexity**: {val_ppl:.2f} (Loss: {val_loss:.4f})\n")
        f.write(f"- **Word-Validity Pass Rate (>= 95%)**: {validity_rate:.1%} ({pass_validity_count}/{len(PROBE_50)})\n")
        f.write(f"- **Readability Pass Rate (>= -7.8)**: {trigram_rate:.1%} ({pass_trigram_count}/{len(PROBE_50)})\n")
        f.write(f"- **Total Fluency Guard Pass Rate**: {fluent_rate:.1%} ({pass_fluent_count}/{len(PROBE_50)})\n\n")
        f.write("> [!NOTE] All outputs below are RAW, unfiltered completions from the scratch-trained EdithLM model. No router, verbalizer, or safety scaffolding.\n\n")
        f.write("| # | Prompt | Raw Model Output | Word Valid % | Readable Logprob | Fluent Pass |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in results:
            clean_out = r["raw_output"].replace("|", "\\|").replace("\n", " ")
            f.write(f"| {r['idx']} | {r['prompt']} | {clean_out} | {r['word_validity']:.1%} | {r['char_trigram']:.2f} | {'PASS' if r['is_fluent'] else 'FAIL'} |\n")

    print(f"Wrote full probe report to: {output_md_path}")
    return results


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_lm_v15", "edith_lm_v15_poc_25m_best.pt")
    out_md = sys.argv[2] if len(sys.argv) > 2 else os.path.join(PROJECT_ROOT, "docs", "FLUENCY_50_PROBES_POC_25M.md")
    run_probe_evaluation(ckpt, out_md)

