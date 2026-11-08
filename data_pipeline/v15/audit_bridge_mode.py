"""
E.D.I.T.H. V15.1 Phase 7 Bridge Mode Audit
Audits:
1. Model identity, weights provenance, licensing of LocalSentenceEncoder.
2. Dialogue corpus size, distribution, and response mechanism.
3. Query retrieval similarities for in-domain vs out-of-domain queries.
4. Tri-mode comparison: Generation-only vs Retrieval-only vs Combined.
"""

import os
import sys
import json
import torch
from typing import Dict, Any, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.memory_store import MemoryStore
from engine.bridge_mode import BridgeModeEngine, BridgeDialogueBank
from engine.edith_lm.model import EdithLM, EdithLMConfig
from engine.edith_lm.tokenizer import EdithByteBPETokenizer
from engine.fluency_guard import FluencyGuard
from data_pipeline.v15.eval_generation_and_decoding import generate_with_config


def run_bridge_audit():
    device = torch.device(os.environ.get("AUDIT_DEVICE", "cpu"))
    print("=" * 80)
    print("PHASE 7: BRIDGE MODE ARCHITECTURAL & RETRIEVAL AUDIT")
    print("=" * 80)

    # 1. Initialize MemoryStore and BridgeMode
    mem = MemoryStore()
    bridge = BridgeModeEngine(mem)

    print(f"Bridge Bank Total Dialogues: {len(bridge.bank.dialogues)}")
    print(f"Bank Query Embeddings Tensor: {bridge.bank.query_embeddings.shape}")

    # 2. Inspect Encoder Architecture & Weights
    encoder = mem.encoder
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"LocalSentenceEncoder Parameters: {num_params:,}")
    print(f"Architecture: {encoder}")
    
    # 3. Test Retrieval Similarity Scores on diverse queries
    test_queries = [
        ("describe me", "In-domain self-reference"),
        ("What do you think about me?", "In-domain self-reference"),
        ("Really?", "In-domain follow-up"),
        ("Hey Edith, how are you?", "In-domain chit-chat"),
        ("Tell me a joke", "In-domain humour"),
        ("Explain recursion to someone learning programming for the first time.", "Out-of-domain technical"),
        ("What is the capital of France?", "Out-of-domain factual"),
        ("Why does ice float?", "Out-of-domain science"),
        ("How should a student prioritize three assignments due tomorrow?", "Out-of-domain reasoning"),
        ("bhai aaj bohot bekar din tha", "In-domain Hinglish empathy")
    ]

    retrieval_benchmarks = []
    print("\n--- RETRIEVAL RELEVANCE BENCHMARK ---")
    for q, domain_type in test_queries:
        ranks = bridge.bank.rank_relevance(q, top_k=3)
        top_sim, top_match = ranks[0]
        retrieval_benchmarks.append({
            "query": q,
            "type": domain_type,
            "top_similarity": top_sim,
            "matched_query": top_match["query"],
            "matched_response": top_match["response"],
            "category": top_match.get("category", "")
        })
        print(f"\nQuery: \"{q}\" ({domain_type})")
        print(f"  Top Match (Sim: {top_sim:.4f}): \"{top_match['query']}\"")
        print(f"  Response: \"{top_match['response'][:100]}...\"")

    # 4. Tri-Mode Comparison: Generation-Only vs Retrieval-Only vs Combined
    print("\n" + "=" * 80)
    print("TRI-MODE COMPARISON (Generation vs Retrieval vs Combined)")
    print("=" * 80)

    ckpt_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_lm_v15", "edith_lm_v15_poc_25m_best.pt")
    tok_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_bpe_tokenizer_16k.json")
    tokenizer = EdithByteBPETokenizer(vocab_file=tok_path)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", EdithLMConfig(vocab_size=16384, dim=576, n_layers=12, n_heads=6, n_kv_heads=2, intermediate_dim=1728, max_seq_len=512, tie_word_embeddings=True))
    model = EdithLM(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tri_mode_prompts = [
        "Hey Edith, how are you?",
        "describe me",
        "Really?",
        "Explain recursion to someone learning programming for the first time.",
        "Tell me a genuinely funny programming joke.",
        "Why would someone carry an umbrella when the sky is dark and cloudy?"
    ]

    tri_mode_results = []
    for p in tri_mode_prompts:
        # 1. Generation only (Raw EdithLM)
        raw_gen = generate_with_config(model, tokenizer, p, device, max_new_tokens=45, temperature=0.7, top_k=40, seed=42)
        is_fluent, diag = FluencyGuard.inspect(raw_gen, query=p)

        # 2. Retrieval only (Bridge Mode)
        ret_result = bridge.process(p)
        ret_text = ret_result.get("text", ret_result.get("reply", ""))

        # 3. Combined (Production logic: if raw_gen is fluent, use LM; else bridge)
        if is_fluent:
            combined_mode = "LM"
            combined_text = raw_gen
        else:
            combined_mode = "BRIDGE"
            combined_text = ret_text

        tri_mode_results.append({
            "prompt": p,
            "raw_generation": raw_gen,
            "raw_fluent": is_fluent,
            "retrieval_text": ret_text,
            "combined_selected_mode": combined_mode,
            "combined_text": combined_text
        })

        print(f"\nPROMPT: \"{p}\"")
        print(f"  [RAW GEN] ({'PASS' if is_fluent else 'REJECT'}): \"{raw_gen}\"")
        print(f"  [RETRIEVAL]: \"{ret_text}\"")
        print(f"  [COMBINED]: ({combined_mode}) \"{combined_text[:120]}...\"")

    # Save to disk
    out_json = os.path.join(PROJECT_ROOT, "data_pipeline", "v15", "bridge_mode_audit_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "encoder_params": num_params,
            "dialogue_count": len(bridge.bank.dialogues),
            "retrieval_benchmarks": retrieval_benchmarks,
            "tri_mode_results": tri_mode_results
        }, f, indent=2)
    print(f"\nSaved Bridge Mode audit to {out_json}")


if __name__ == "__main__":
    run_bridge_audit()
