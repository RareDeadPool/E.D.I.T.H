import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.edith_lm.inference import EdithLMInference
from engine.fluency_guard import FluencyGuard

ckpt = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_lm_v14", "edith_lm_v14_chat.pt")
if not os.path.exists(ckpt):
    ckpt = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_lm", "edith_lm_best.pt")

print("Loading checkpoint:", ckpt)
lm = EdithLMInference.from_checkpoint(ckpt)

test_prompts = [
    "describe me",
    "What do you think about me",
    "Really",
    "How are you",
    "Tell me something about myself"
]

print("\n" + "="*80)
print("RAW UNFILTERED MODEL OUTPUTS (NO GUARDS, NO BRIDGE, NO VERBALIZER):")
print("="*80)

for p in test_prompts:
    prompt = f"<user>{p}</user><edith>"
    raw_gen = lm.generate(prompt, max_new_tokens=40, temperature=0.7)
    is_fluent, diag = FluencyGuard.inspect(raw_gen)
    print(f"PROMPT: \"{p}\"")
    print(f"RAW GENERATION: \"{raw_gen}\"")
    print(f"WORD VALIDITY: {diag.get('word_validity_ratio', 0.0):.1%}, CHAR TRIGRAM: {diag.get('char_trigram_logprob', 0.0):.2f}, FLUENT: {is_fluent}")
    print("-" * 60)

