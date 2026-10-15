"""Audit script to inspect V1 checkpoint, architecture, and metrics."""
import os
import json
import torch

checkpoint_path = "models/checkpoints/edith_ann_best.pt"
tokenizer_path = "models/checkpoints/tokenizer.json"

print("=== CHECKPOINT AUDIT ===")
if os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    print(f"Checkpoint keys: {list(ckpt.keys())}")
    print(f"Epoch: {ckpt.get('epoch')}")
    print(f"Metrics: {ckpt.get('metrics')}")
    print(f"Vocab size in ckpt: {ckpt.get('vocab_size')}")
    print(f"Config: {json.dumps(ckpt.get('config'), indent=2)}")
    
    maps = ckpt.get("metadata_maps", {})
    print(f"Number of intents: {len(maps.get('intents', {}))}")
    print(f"Number of slots: {len(maps.get('slots', {}))}")
    print(f"Number of actions: {len(maps.get('actions', {}))}")
    
    state_dict = ckpt.get("model_state_dict", {})
    total_tensors = len(state_dict)
    total_params = sum(t.numel() for t in state_dict.values())
    print(f"Total state_dict keys: {total_tensors}")
    print(f"Total parameter count: {total_params:,}")
    
    for k, v in list(state_dict.items())[:10]:
        print(f"  {k}: {v.shape}")
else:
    print("Checkpoint not found!")

if os.path.exists(tokenizer_path):
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tok_data = json.load(f)
    vocab = tok_data.get("vocab", {})
    print(f"Tokenizer vocab size: {len(vocab)}")
    sample_items = list(vocab.items())[:15]
    print(f"Sample vocab items: {sample_items}")
