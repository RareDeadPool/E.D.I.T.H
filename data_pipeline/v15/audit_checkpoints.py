"""
E.D.I.T.H. V15.3.1 Audit Script: Checkpoint and Artifact Integrity
Verifies SHA-256, model parameters, architecture configs, training steps, and loadability.
Exports findings to data_pipeline/v15/audit_outputs/checkpoint_audit_ledger.json
"""

import os
import sys
import hashlib
import json
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.edith_lm.model import EdithLM, EdithLMConfig
from engine.edith_lm.tokenizer import EdithByteBPETokenizer


def get_sha256(filepath: str) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            sha256.update(chunk)
    return sha256.hexdigest()


def audit_checkpoint(name: str, path: str, device: str = "cpu") -> dict:
    if not os.path.exists(path):
        return {"name": name, "path": path, "exists": False}

    size_bytes = os.path.getsize(path)
    sha256_hash = get_sha256(path)

    try:
        data = torch.load(path, map_location=device, weights_only=False)
    except Exception as e:
        return {
            "name": name,
            "path": path,
            "exists": True,
            "size_bytes": size_bytes,
            "sha256": sha256_hash,
            "load_error": str(e)
        }

    keys = list(data.keys())
    step = data.get("step", None)
    loss = data.get("loss", data.get("val_loss", data.get("sft_val_loss", None)))
    ppl = data.get("val_ppl", data.get("sft_val_ppl", None))
    general_ppl = data.get("general_val_ppl", None)
    targets_seen = data.get("targets_seen", None)

    cfg_dict = {}
    config_obj = data.get("config", None)
    if config_obj is not None:
        if isinstance(config_obj, dict):
            cfg_dict = config_obj
            cfg = EdithLMConfig(**config_obj)
        else:
            cfg = config_obj
            cfg_dict = {
                "vocab_size": cfg.vocab_size,
                "max_seq_len": cfg.max_seq_len,
                "n_layers": cfg.n_layers,
                "n_heads": cfg.n_heads,
                "n_kv_heads": cfg.n_kv_heads,
                "dim": cfg.dim,
                "intermediate_dim": cfg.intermediate_dim,
                "rope_theta": cfg.rope_theta,
                "norm_eps": cfg.norm_eps,
                "tie_word_embeddings": getattr(cfg, "tie_word_embeddings", True)
            }
    else:
        cfg = None

    # Test loading into EdithLM
    load_success = False
    param_count = None
    tied_param_count = None
    forward_test = False

    if cfg is not None and "model_state_dict" in data:
        try:
            model = EdithLM(cfg).to(device)
            model.load_state_dict(data["model_state_dict"])
            model.eval()
            load_success = True
            param_count = sum(p.numel() for p in model.parameters())
            # Test forward pass with dummy input
            dummy_input = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=device)
            with torch.no_grad():
                logits, _, _ = model(dummy_input)
            if logits.shape == (1, 4, cfg.vocab_size):
                forward_test = True
        except Exception as e:
            load_error = str(e)

    has_optimizer = "optimizer_state_dict" in data

    return {
        "name": name,
        "path": os.path.relpath(path, PROJECT_ROOT),
        "exists": True,
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 2),
        "sha256": sha256_hash,
        "dictionary_keys": keys,
        "step": step,
        "targets_seen": targets_seen,
        "recorded_loss": loss,
        "recorded_ppl": ppl,
        "recorded_general_ppl": general_ppl,
        "has_optimizer_state": has_optimizer,
        "load_success": load_success,
        "forward_test_passed": forward_test,
        "total_parameters": param_count,
        "config": cfg_dict
    }


def main():
    print("=" * 80)
    print("E.D.I.T.H. V15.3.1 ARTIFACT INTEGRITY AUDIT")
    print("=" * 80)

    checkpoints_to_audit = [
        ("v15_2_100m_best", os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_lm_v15", "edith_lm_v15_2_100m_best.pt")),
        ("v15_2_100m_best_backup", os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_lm_v15", "edith_lm_v15_2_100m_best.backup.pt")),
        ("v15_2_100m_latest", os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_lm_v15", "edith_lm_v15_2_100m_latest.pt")),
        ("v15_3_sft_best", os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_lm_v15_sft", "edith_lm_v15_3_sft_best.pt")),
        ("v15_3_sft_latest", os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_lm_v15_sft", "edith_lm_v15_3_sft_latest.pt")),
    ]

    ledger = {}
    for name, path in checkpoints_to_audit:
        print(f"Auditing {name} ({os.path.basename(path)})...")
        res = audit_checkpoint(name, path, device="cpu")
        ledger[name] = res
        print(f"  SHA-256: {res.get('sha256', 'N/A')}")
        print(f"  Step: {res.get('step', 'N/A')} | Size: {res.get('size_mb', 'N/A')} MB | Load: {res.get('load_success')} | Forward: {res.get('forward_test_passed')}")

    # Audit Tokenizers
    tok_paths = [
        ("tokenizer_16k", os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_bpe_tokenizer_16k.json")),
        ("tokenizer_alias", os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_bpe_tokenizer.json")),
    ]
    tok_ledger = {}
    for name, p in tok_paths:
        if os.path.exists(p):
            tok = EdithByteBPETokenizer(vocab_file=p)
            h = get_sha256(p)
            tok_ledger[name] = {
                "path": os.path.relpath(p, PROJECT_ROOT),
                "sha256": h,
                "vocab_size": tok.vocab_size,
                "bos_id": tok.bos_id,
                "eos_id": tok.eos_id,
                "pad_id": tok.pad_id,
                "size_bytes": os.path.getsize(p)
            }
            print(f"Audited Tokenizer {name}: Vocab {tok.vocab_size}, SHA-256: {h}")

    out_ledger = {
        "checkpoints": ledger,
        "tokenizers": tok_ledger
    }

    out_path = os.path.join(PROJECT_ROOT, "data_pipeline", "v15", "audit_outputs", "checkpoint_audit_ledger.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_ledger, f, indent=2)

    print(f"\nLedger saved to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
