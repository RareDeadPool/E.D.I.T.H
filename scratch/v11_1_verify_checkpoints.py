import hashlib
import json
import os
import sys
import torch
import onnx

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from edith_ann.tokenizer import EdithTokenizer
from edith_ann.model import EdithNeuralNetwork

files_to_hash = [
    "models/checkpoints/edith_ann_best.pt",
    "models/checkpoints/edith_ann_best.onnx",
    "models/checkpoints/tokenizer.json",
    "models/checkpoints/edith_ann_v10_baseline.pt",
    "models/checkpoints/edith_ann_v10_baseline.onnx",
    "models/checkpoints/tokenizer_v10_baseline.json",
    "models/checkpoints/edith_ann_v11_candidate.pt",
    "models/checkpoints/edith_ann_v11_candidate.onnx",
    "models/checkpoints/tokenizer_v11_candidate.json",
    "data/test_data_clean.json",
    "data/test_data.json",
    "data/v11_challenge_set.json"
]

print("=== SHA256 HASHES ===")
hashes = {}
for rel_path in files_to_hash:
    abs_path = os.path.join(PROJECT_ROOT, rel_path)
    if os.path.exists(abs_path):
        hasher = hashlib.sha256()
        with open(abs_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        h = hasher.hexdigest()
        hashes[rel_path] = h
        print(f"{rel_path:45s}: {h} ({os.path.getsize(abs_path):,} bytes)")
    else:
        print(f"{rel_path:45s}: NOT FOUND")

# Identity checks
print("\n=== IDENTITY CHECKS ===")
print("edith_ann_best.pt == edith_ann_v10_baseline.pt:", hashes.get("models/checkpoints/edith_ann_best.pt") == hashes.get("models/checkpoints/edith_ann_v10_baseline.pt"))
print("edith_ann_best.onnx == edith_ann_v10_baseline.onnx:", hashes.get("models/checkpoints/edith_ann_best.onnx") == hashes.get("models/checkpoints/edith_ann_v10_baseline.onnx"))
print("tokenizer.json == tokenizer_v10_baseline.json:", hashes.get("models/checkpoints/tokenizer.json") == hashes.get("models/checkpoints/tokenizer_v10_baseline.json"))
print("tokenizer.json == tokenizer_v11_candidate.json:", hashes.get("models/checkpoints/tokenizer.json") == hashes.get("models/checkpoints/tokenizer_v11_candidate.json"))

# Inspect checkpoints
print("\n=== CHECKPOINT INTERNAL METADATA ===")
for ckpt_name in ["edith_ann_v10_baseline.pt", "edith_ann_v11_candidate.pt", "edith_ann_best.pt"]:
    p = os.path.join(PROJECT_ROOT, "models/checkpoints", ckpt_name)
    if not os.path.exists(p):
        continue
    ckpt = torch.load(p, map_location="cpu")
    cfg = ckpt.get("config", {})
    maps = ckpt.get("metadata_maps", {})
    sd = ckpt.get("model_state_dict", {})
    embed_shape = sd.get("embedding.weight", torch.tensor([])).shape
    param_count = sum(p.numel() for p in sd.values())
    print(f"[{ckpt_name}]")
    print(f"  Vocab Size in Ckpt: {ckpt.get('vocab_size')}, Embedding Weight Shape: {embed_shape}")
    print(f"  Total Params in State Dict: {param_count:,}")
    print(f"  Intents: {len(maps.get('intents', {}))}, Slots: {len(maps.get('slots', {}))}, Actions: {len(maps.get('actions', {}))}")
    print(f"  Model Config: d_model={cfg.get('model', {}).get('d_model')}, n_layers={cfg.get('model', {}).get('n_layers')}, max_seq_len={cfg.get('model', {}).get('max_seq_len')}")

# Inspect tokenizers
print("\n=== TOKENIZER INSPECTION ===")
for tok_name in ["tokenizer_v10_baseline.json", "tokenizer_v11_candidate.json", "tokenizer.json"]:
    p = os.path.join(PROJECT_ROOT, "models/checkpoints", tok_name)
    if not os.path.exists(p):
        continue
    tok = EdithTokenizer.load(p)
    print(f"[{tok_name}] Vocab Size: {tok.vocab_size}, Special Tokens: PAD={tok.PAD_ID}, UNK={tok.UNK_ID}, BOS={tok.BOS_ID}, EOS={tok.EOS_ID}")
    test_str = "open visual studio code."
    print(f"  Tokenize '{test_str}': {tok.tokenize_raw(test_str)}")

# Inspect ONNX models
print("\n=== ONNX INSPECTION ===")
for onnx_name in ["edith_ann_v10_baseline.onnx", "edith_ann_v11_candidate.onnx", "edith_ann_best.onnx"]:
    p = os.path.join(PROJECT_ROOT, "models/checkpoints", onnx_name)
    if not os.path.exists(p):
        continue
    m = onnx.load(p)
    inputs = [f"{i.name}: {[d.dim_value for d in i.type.tensor_type.shape.dim]}" for i in m.graph.input]
    outputs = [f"{o.name}: {[d.dim_value for d in o.type.tensor_type.shape.dim]}" for o in m.graph.output]
    print(f"[{onnx_name}] Inputs: {inputs} | Outputs: {outputs}")

# Save hashes to a JSON for reporting
with open(os.path.join(PROJECT_ROOT, "scratch/verified_hashes.json"), "w") as f:
    json.dump(hashes, f, indent=2)
print("\nSaved verified_hashes.json successfully.")
