"""
E.D.I.T.H. V15.3.1 Audit Script: Runtime Integration Test
Tests runtime loading of V15.3 SFT model into EdithLMInference and MindRouter.
Tests CPU fallback, response extraction, and safety routing isolation.
"""

import os
import sys
import time
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.edith_lm.inference import EdithLMInference
from engine.mind_router import MindRouter
from engine.safety_routing_guard import ActionSafetyAndCompatibilityGuard


def test_edith_lm_inference():
    print("Testing EdithLMInference loading from V15.3 checkpoint...")
    ckpt_path = "models/checkpoints/edith_lm_v15_sft/edith_lm_v15_3_sft_best.pt"
    tok_path = "models/checkpoints/edith_bpe_tokenizer_16k.json"

    # Test loading
    try:
        engine = EdithLMInference.from_checkpoint(checkpoint_path=ckpt_path, tokenizer_path=tok_path)
        print(f"  [SUCCESS] Loaded into EdithLMInference on device: {engine.device}")
        
        # Test generation
        prompt = "<mood>calm_focused</mood><user>Hello Edith</user><edith>"
        stream_tokens = list(engine.generate_stream(prompt, max_new_tokens=20))
        full_text = "".join(stream_tokens)
        print(f"  [SUCCESS] Generated text: \"{full_text.strip()}\"")
    except Exception as e:
        print(f"  [FAILURE] EdithLMInference error: {type(e).__name__}: {e}")

    # Test CPU Fallback
    try:
        print("\nTesting CPU Fallback...")
        engine_cpu = EdithLMInference.from_checkpoint(checkpoint_path=ckpt_path, tokenizer_path=tok_path, device="cpu")
        print(f"  [SUCCESS] Loaded on CPU. Device: {engine_cpu.device}")
        stream_tokens_cpu = list(engine_cpu.generate_stream(prompt, max_new_tokens=10))
        print(f"  [SUCCESS] CPU Generated text: \"{''.join(stream_tokens_cpu).strip()}\"")
    except Exception as e:
        print(f"  [FAILURE] CPU Fallback error: {type(e).__name__}: {e}")


def test_safety_routing_guard():
    print("\nTesting Action Safety & Routing Guard Isolation...")
    conversational_inputs = [
        "Tell me a joke.",
        "Kill all system processes immediately.",
        "Can you format my hard drive C:?",
        "Delete my project folder.",
        "Launch Spotify and play songs.",
        "What is your name?"
    ]

    for inp in conversational_inputs:
        is_conv = ActionSafetyAndCompatibilityGuard.is_conversational_statement(inp)
        is_kill, target = ActionSafetyAndCompatibilityGuard.is_explicit_process_kill(inp)
        print(f"  Input: \"{inp}\"")
        print(f"    Conversational: {is_conv} | Explicit Process Kill: {is_kill} (Target: {target})")


if __name__ == "__main__":
    test_edith_lm_inference()
    test_safety_routing_guard()
