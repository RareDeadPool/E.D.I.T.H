"""E.D.I.T.H. Custom Multi-Task Neural Network Training and Evaluation Pipeline V8.

Features:
- Multi-task Transformer training on intent, BIO slot tagging, action policy, operational confidence, and plan complexity.
- Rich domain vocabulary fitted with V8 conversational, reflective, hard-negative, and screen understanding terms.
- Candidate checkpoint isolation (`edith_ann_v8_candidate.pt`) with ONNX export (`dynamo=False`) and numerical parity testing.
- Rigorous validation against held-out baseline metrics before promotion.
- Full provenance and metric logging to `models/checkpoints/v8_training_metrics.json`.
"""

import os
import sys
import json
import time
import math
import shutil
from typing import Dict, List, Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from edith_ann.tokenizer import EdithTokenizer
from edith_ann.model import EdithNeuralNetwork
from edith_ann.loss import EdithMultiTaskLoss


class EdithOnnxWrapper(nn.Module):
    """Wrapper to flatten dictionary outputs into sequential tuple for TorchScript ONNX export."""
    def __init__(self, model: EdithNeuralNetwork):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.model(input_ids, attention_mask=attention_mask)
        return (
            out["intent_logits"],
            out["slot_logits"],
            out["action_logits"],
            out["confidence"],
            out["complexity_logits"],
            out["contextual_embeddings"],
        )


class EdithDataset(Dataset):
    """Fast pre-encoded PyTorch Dataset for EDITH multi-task training."""

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        tokenizer: EdithTokenizer,
        slot_map: Dict[str, int],
        max_seq_len: int = 64,
        ignore_index: int = -100,
    ):
        self.samples = samples
        self.max_seq_len = max_seq_len
        self.ignore_index = ignore_index

        all_input_ids = []
        all_attention_mask = []
        all_intent = []
        all_slot = []
        all_action = []
        all_confidence = []
        all_complexity = []

        for sample in samples:
            input_ids, attention_mask = tokenizer.encode(
                sample["text"], max_len=max_seq_len, add_special_tokens=True, pad_to_max=True
            )
            raw_slots = sample["slot_tags"]
            slot_targets = [ignore_index] * max_seq_len
            usable_len = min(len(raw_slots), max_seq_len - 2)
            for i in range(usable_len):
                slot_targets[i + 1] = slot_map.get(raw_slots[i], 0)

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_intent.append(sample["intent_id"])
            all_slot.append(slot_targets)
            all_action.append(sample["action_id"])
            all_confidence.append([sample["confidence"]])
            all_complexity.append(sample.get("complexity", 0))

        self.t_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        self.t_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
        self.t_intent = torch.tensor(all_intent, dtype=torch.long)
        self.t_slot = torch.tensor(all_slot, dtype=torch.long)
        self.t_action = torch.tensor(all_action, dtype=torch.long)
        self.t_confidence = torch.tensor(all_confidence, dtype=torch.float32)
        self.t_complexity = torch.tensor(all_complexity, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.t_input_ids[idx],
            "attention_mask": self.t_attention_mask[idx],
            "intent": self.t_intent[idx],
            "slot": self.t_slot[idx],
            "action": self.t_action[idx],
            "confidence": self.t_confidence[idx],
            "complexity": self.t_complexity[idx],
        }


def evaluate(
    model: EdithNeuralNetwork,
    dataloader: DataLoader,
    criterion: EdithMultiTaskLoss,
    device: torch.device,
    num_intents: int = 17,
) -> Dict[str, Any]:
    """Evaluate model computing multi-task metrics, per-class F1, macro-F1, and calibration error."""
    model.eval()
    total_val_loss = 0.0
    correct_intent = 0
    correct_action = 0
    correct_comp = 0
    total_samples = 0

    correct_slots = 0
    total_slots = 0

    ood_correct = 0
    ood_total = 0

    intent_tp = [0] * num_intents
    intent_fp = [0] * num_intents
    intent_fn = [0] * num_intents

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = {k: v.to(device) for k, v in batch.items() if k not in ["input_ids", "attention_mask"]}

            outputs = model(input_ids, attention_mask=attention_mask)
            losses = criterion(outputs, targets)
            total_val_loss += losses["total_loss"].item() * input_ids.size(0)

            # Intent Accuracy & F1 tracking
            intent_preds = outputs["intent_logits"].argmax(dim=-1)
            correct_intent += (intent_preds == targets["intent"]).sum().item()

            for p_id, t_id in zip(intent_preds.tolist(), targets["intent"].tolist()):
                if p_id == t_id:
                    intent_tp[t_id] += 1
                else:
                    intent_fp[p_id] += 1
                    intent_fn[t_id] += 1

            # Action Accuracy
            action_preds = outputs["action_logits"].argmax(dim=-1)
            correct_action += (action_preds == targets["action"]).sum().item()

            # Complexity Accuracy
            comp_preds = outputs["complexity_logits"].argmax(dim=-1)
            correct_comp += (comp_preds == targets["complexity"]).sum().item()

            # OOD Detection Evaluation (targets["confidence"] < 0.5)
            conf_preds = outputs["confidence"].squeeze(-1)
            is_ood_true = targets["confidence"].squeeze(-1) < 0.5
            ood_matches = (conf_preds < 0.5) == is_ood_true
            ood_correct += ood_matches.sum().item()
            ood_total += input_ids.size(0)

            # Slot Accuracy (ignore index -100)
            slot_preds = outputs["slot_logits"].argmax(dim=-1)
            valid_mask = targets["slot"] != criterion.ignore_slot_index
            correct_slots += ((slot_preds == targets["slot"]) & valid_mask).sum().item()
            total_slots += valid_mask.sum().item()

            total_samples += input_ids.size(0)

    val_loss = total_val_loss / max(1, total_samples)
    intent_acc = (correct_intent / max(1, total_samples)) * 100.0
    action_acc = (correct_action / max(1, total_samples)) * 100.0
    comp_acc = (correct_comp / max(1, total_samples)) * 100.0
    slot_acc = (correct_slots / max(1, total_slots)) * 100.0 if total_slots > 0 else 0.0
    ood_acc = (ood_correct / max(1, ood_total)) * 100.0

    # Macro-F1 across intents
    f1_scores = []
    per_intent_f1 = {}
    for i in range(num_intents):
        tp = intent_tp[i]
        fp = intent_fp[i]
        fn = intent_fn[i]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_scores.append(f1)
        per_intent_f1[i] = {"precision": prec * 100.0, "recall": rec * 100.0, "f1": f1 * 100.0, "support": tp + fn}
    macro_f1 = (sum(f1_scores) / len(f1_scores)) * 100.0

    return {
        "val_loss": val_loss,
        "intent_acc": intent_acc,
        "macro_f1": macro_f1,
        "action_acc": action_acc,
        "slot_acc": slot_acc,
        "complexity_acc": comp_acc,
        "ood_acc": ood_acc,
        "per_intent": per_intent_f1,
    }


def export_and_verify_onnx(
    model: EdithNeuralNetwork,
    onnx_path: str,
    max_seq_len: int = 64,
) -> bool:
    """Export PyTorch model to ONNX using legacy TorchScript and verify exact output parity."""
    print(f"\nExporting candidate model to ONNX: {onnx_path} ...", flush=True)
    model.eval()
    wrapper = EdithOnnxWrapper(model)
    wrapper.eval()

    dummy_input_ids = torch.ones((1, max_seq_len), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, max_seq_len), dtype=torch.long)

    try:
        torch.onnx.export(
            wrapper,
            (dummy_input_ids, dummy_attention_mask),
            onnx_path,
            input_names=["input_ids", "attention_mask"],
            output_names=[
                "intent_logits",
                "slot_logits",
                "action_logits",
                "confidence",
                "complexity_logits",
                "contextual_embeddings",
            ],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "seq_len"},
                "attention_mask": {0: "batch_size", 1: "seq_len"},
                "intent_logits": {0: "batch_size"},
                "slot_logits": {0: "batch_size", 1: "seq_len"},
                "action_logits": {0: "batch_size"},
                "confidence": {0: "batch_size"},
                "complexity_logits": {0: "batch_size"},
                "contextual_embeddings": {0: "batch_size", 1: "seq_len"},
            },
            opset_version=14,
            dynamo=False,
        )
        print("  -> ONNX export succeeded.", flush=True)
    except Exception as e:
        print(f"  [ERROR] ONNX export failed: {e}", flush=True)
        return False

    # Verify ONNX Runtime Parity
    try:
        import onnxruntime as ort
        import numpy as np

        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        with torch.no_grad():
            py_out = model(dummy_input_ids, attention_mask=dummy_attention_mask)

        ort_in = {
            "input_ids": dummy_input_ids.numpy(),
            "attention_mask": dummy_attention_mask.numpy(),
        }
        ort_outs = session.run(None, ort_in)

        # Check maximum absolute error across intent logits
        diff_intent = np.max(np.abs(py_out["intent_logits"].numpy() - ort_outs[0]))
        diff_slot = np.max(np.abs(py_out["slot_logits"].numpy() - ort_outs[1]))
        diff_action = np.max(np.abs(py_out["action_logits"].numpy() - ort_outs[2]))
        diff_conf = np.max(np.abs(py_out["confidence"].numpy() - ort_outs[3]))

        print(f"  -> Numerical Parity Check: max diff intent={diff_intent:.6f}, slot={diff_slot:.6f}, action={diff_action:.6f}, conf={diff_conf:.6f}", flush=True)
        if diff_intent > 1e-3 or diff_slot > 1e-3:
            print("  [WARNING] ONNX parity discrepancy exceeds 1e-3 threshold!", flush=True)
            return False

        print("  -> ONNX numerical parity verified within tolerance (max error < 1e-3).", flush=True)
        return True
    except Exception as e:
        print(f"  [WARNING] Could not verify ONNX runtime parity: {e}", flush=True)
        return False


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==================================================", flush=True)
    print(" [E.D.I.T.H. NEURAL NETWORK TRAINING ENGINE V8] ", flush=True)
    print(f" Execution Device: {device}", flush=True)
    print("==================================================", flush=True)

    # 1. Load Configurations
    with open("config/model_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    checkpoint_dir = config["training"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 2. Load Datasets & Maps
    with open("data/train_data.json", "r", encoding="utf-8") as f:
        train_samples = json.load(f)
    with open("data/val_data.json", "r", encoding="utf-8") as f:
        val_samples = json.load(f)
    with open("data/metadata_maps.json", "r", encoding="utf-8") as f:
        maps = json.load(f)

    intent_map = maps["intents"]
    slot_map = maps["slots"]
    action_map = maps["actions"]

    # 3. Fit Tokenizer with rich V8 seed words
    all_texts = [s["text"] for s in train_samples]
    tokenizer = EdithTokenizer()
    domain_seed_words = [
        "browser", "chrome", "firefox", "edge", "notepad", "code", "vscode", "terminal",
        "calculator", "calc", "spotify", "discord", "slack", "explorer", "editor",
        "file", "folder", "desktop", "downloads", "documents", "resume", "notes", "report",
        "presentation", "budget", "project", "todo", "summary", "contract", "invoice",
        "find", "locate", "search", "google", "web", "online", "internet", "look", "browse",
        "delete", "remove", "trash", "erase", "rid", "open", "create", "make", "copy", "move",
        "volume", "sound", "audio", "brightness", "screen", "display", "wifi", "bluetooth",
        "louder", "quieter", "brighter", "dimmer", "down", "up", "mute", "unmute",
        "battery", "storage", "disk", "memory", "ram", "cpu", "temperature",
        "kill", "terminate", "stop", "close", "end", "pid", "process", "task",
        "screenshot", "capture", "snip", "recycle", "bin", "empty", "minimize",
        "play", "pause", "resume", "unpause", "next", "previous", "prev", "skip", "track", "song",
        "send", "email", "mail", "draft", "message", "write", "type", "letter",
        "first", "second", "third", "fourth", "fifth", "latest", "newer", "older", "one", "recent",
        "most", "it", "that", "this", "them", "everything", "rahul", "john", "alex",
        # V8 domain seed words (Emotional, Hard Negatives, Systems, Screen, Multi-turn)
        "coding", "environment", "dev", "workspace", "ide", "edith", "tests", "test", "suite",
        "unit", "pass", "fail", "whether", "properly", "starts", "started", "working", "work",
        "meant", "other", "version", "result", "results", "relaxing", "relax", "chill", "lofi",
        "undo", "cancel", "there", "instead", "archive", "check", "inspect", "bring", "could",
        "would", "can", "let", "'s", "n't", "is", "are", "tell", "show", "give", "back",
        "ping", "connectivity", "reachability", "latency", "host", "diagnose", "hello", "hi",
        "greetings", "morning", "afternoon", "evening", "thanks", "thank", "who", "purpose",
        "dismiss", "shut", "quit", "load", "utilization", "capacity", "percent", "percentage",
        "eval", "evaluate", "solve", "math", "sqrt", "square", "root", "clean", "hide", "snapshot",
        "stressed", "exam", "anxiety", "college", "decision", "decisions", "thoughts", "thinking", "listen",
        "struggling", "recursion", "audit", "temporary", "interface", "negation", "hate", "slow", "old",
        "duplicate", "java", "git", "status", "error", "explain", "why", "choose", "misunderstood",
        "wrong", "approach", "bored", "rough", "interesting", "disagree", "overwhelmed", "recommendation",
        "assignment", "compiler", "diagnostic", "storage", "cleanup", "visible", "multitask",
        "organize", "organization", "revert", "rollback",
        # V10 Expanded Cognitive, Profiling & Systems Seed Words
        "dating", "ghost", "ghosted", "closure", "scarcity", "mindset", "pedestal", "boundary", "boundaries",
        "consensus", "distributed", "cache", "stampede", "pool", "exhaustion", "latency", "throughput", "monolith",
        "microservices", "consciousness", "sentience", "qualia", "subjective", "solitude", "resilience", "stoic",
        "stoicism", "negotiation", "salary", "equity", "dopamine", "hyperfocus", "burnout", "procrastination",
        "masterpiece", "builder", "journey", "profile", "profiling", "judge", "roast", "describe", "personality"
    ]
    tokenizer.build_vocab(all_texts, min_freq=1, seed_words=domain_seed_words)
    candidate_tokenizer_path = os.path.join(checkpoint_dir, "tokenizer_v8_candidate.json")
    tokenizer.save(candidate_tokenizer_path)
    print(f"Built custom domain vocabulary: {tokenizer.vocab_size} tokens.", flush=True)

    # 4. Pre-encode Datasets
    max_seq_len = config["model"]["max_seq_len"]
    t0 = time.time()
    train_dataset = EdithDataset(train_samples, tokenizer, slot_map, max_seq_len=max_seq_len)
    val_dataset = EdithDataset(val_samples, tokenizer, slot_map, max_seq_len=max_seq_len)
    print(f"Pre-encoded {len(train_dataset):,} train and {len(val_dataset):,} val tensors in {time.time() - t0:.2f}s.", flush=True)

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 5. Instantiate Architecture
    model = EdithNeuralNetwork(
        vocab_size=tokenizer.vocab_size,
        num_intents=len(intent_map),
        num_slots=len(slot_map),
        num_actions=len(action_map),
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        n_layers=config["model"]["n_layers"],
        d_ff=config["model"]["d_ff"],
        max_seq_len=max_seq_len,
        dropout=config["model"]["dropout"],
    ).to(device)

    total_params = model.count_parameters()
    print(f"Initialized EDITH Neural Network with {total_params:,} trainable parameters.", flush=True)

    # 6. Loss and Optimizer
    weights = config["loss_weights"]
    criterion = EdithMultiTaskLoss(
        intent_weight=weights["intent"],
        slot_weight=weights["slot"],
        action_weight=weights["action"],
        confidence_weight=weights["confidence"],
        complexity_weight=weights.get("complexity", 0.5),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )

    epochs = config["training"]["epochs"]
    best_val_loss = float("inf")
    candidate_checkpoint_path = os.path.join(checkpoint_dir, "edith_ann_v8_candidate.pt")
    candidate_onnx_path = os.path.join(checkpoint_dir, "edith_ann_v8_candidate.onnx")

    print("\nStarting Training Loop...\n", flush=True)
    training_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = {k: v.to(device) for k, v in batch.items() if k not in ["input_ids", "attention_mask"]}

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            losses = criterion(outputs, targets)

            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += losses["total_loss"].item()

        epoch_loss = running_loss / len(train_loader)
        metrics = evaluate(model, val_loader, criterion, device, num_intents=len(intent_map))
        elapsed = time.time() - start_time

        epoch_stat = {
            "epoch": epoch,
            "train_loss": epoch_loss,
            "val_loss": metrics["val_loss"],
            "intent_acc": metrics["intent_acc"],
            "macro_f1": metrics["macro_f1"],
            "slot_acc": metrics["slot_acc"],
            "action_acc": metrics["action_acc"],
            "ood_acc": metrics["ood_acc"],
            "elapsed_sec": elapsed,
        }
        training_history.append(epoch_stat)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"| Train Loss: {epoch_loss:.4f} "
            f"| Val Loss: {metrics['val_loss']:.4f} "
            f"| Intent Acc: {metrics['intent_acc']:.1f}% "
            f"| Macro-F1: {metrics['macro_f1']:.1f}% "
            f"| Slot Acc: {metrics['slot_acc']:.1f}% "
            f"| Action Acc: {metrics['action_acc']:.1f}% "
            f"| OOD Acc: {metrics['ood_acc']:.1f}% "
            f"| Time: {elapsed:.2f}s",
            flush=True
        )

        if metrics["val_loss"] < best_val_loss:
            best_val_loss = metrics["val_loss"]
            save_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "metrics": metrics,
                "vocab_size": tokenizer.vocab_size,
                "metadata_maps": maps,
            }
            torch.save(save_payload, candidate_checkpoint_path)
            print(f"  -> Candidate model saved to {candidate_checkpoint_path} (Val Loss: {best_val_loss:.4f})", flush=True)

    print("\n==================================================", flush=True)
    print(" Training Finished Successfully!", flush=True)
    print(f" Candidate Checkpoint: {candidate_checkpoint_path}", flush=True)
    print("==================================================", flush=True)

    # 7. Evaluate Candidate Checkpoint on Held-Out Test Set
    test_metrics = {}
    if os.path.exists("data/test_data.json"):
        print("\nEvaluating Candidate Checkpoint on Held-Out Test Split (Unseen Phrasings)...", flush=True)
        best_ckpt = torch.load(candidate_checkpoint_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        model.eval()

        with open("data/test_data.json", "r", encoding="utf-8") as f:
            test_samples = json.load(f)
        test_dataset = EdithDataset(test_samples, tokenizer, slot_map, max_seq_len=max_seq_len)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        test_metrics = evaluate(model, test_loader, criterion, device, num_intents=len(intent_map))
        print("Candidate Held-Out Test Results:", flush=True)
        print(f"  Loss: {test_metrics['val_loss']:.4f}", flush=True)
        print(f"  Intent Accuracy: {test_metrics['intent_acc']:.2f}%", flush=True)
        print(f"  Macro-F1: {test_metrics['macro_f1']:.2f}%", flush=True)
        print(f"  Action Accuracy: {test_metrics['action_acc']:.2f}%", flush=True)
        print(f"  Slot Accuracy: {test_metrics['slot_acc']:.2f}%", flush=True)
        print(f"  OOD Detection: {test_metrics['ood_acc']:.2f}%", flush=True)

    # 8. Export to ONNX and Verify Parity
    onnx_ok = export_and_verify_onnx(model, candidate_onnx_path, max_seq_len=max_seq_len)

    # 9. Gatekeeper: Compare Candidate vs Baseline for Promotion
    # Baseline Test Macro-F1 on new V8 dataset: 80.28%, Val Loss: 2.58
    baseline_test_macro_f1 = 80.28
    candidate_test_macro_f1 = test_metrics.get("macro_f1", 0.0)

    print("\n=================== PROMOTION AUDIT ===================", flush=True)
    print(f"Baseline Test Macro-F1: {baseline_test_macro_f1:.2f}%", flush=True)
    print(f"Candidate Test Macro-F1: {candidate_test_macro_f1:.2f}%", flush=True)
    print(f"ONNX Parity Verified:    {onnx_ok}", flush=True)

    should_promote = (candidate_test_macro_f1 >= baseline_test_macro_f1) and onnx_ok
    if should_promote:
        print("  -> PROMOTION APPROVED! Candidate outperforms baseline and satisfies all parity gates.", flush=True)
        prod_pt = os.path.join(checkpoint_dir, "edith_ann_best.pt")
        prod_onnx = os.path.join(checkpoint_dir, "edith_ann_best.onnx")
        prod_tok = os.path.join(checkpoint_dir, "tokenizer.json")

        shutil.copyfile(candidate_checkpoint_path, prod_pt)
        shutil.copyfile(candidate_onnx_path, prod_onnx)
        shutil.copyfile(candidate_tokenizer_path, prod_tok)
        print(f"  -> Promoted to production: {prod_pt}, {prod_onnx}, {prod_tok}", flush=True)
    else:
        print("  -> PROMOTION WITHHELD. Candidate did not strictly outperform baseline or parity failed.", flush=True)

    # 10. Save Full Metrics Report
    report_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_config": config,
        "vocab_size": tokenizer.vocab_size,
        "total_parameters": total_params,
        "training_history": training_history,
        "best_val_loss": best_val_loss,
        "test_metrics": test_metrics,
        "baseline_test_macro_f1": baseline_test_macro_f1,
        "candidate_test_macro_f1": candidate_test_macro_f1,
        "promoted_to_production": should_promote,
    }
    metrics_path = os.path.join(checkpoint_dir, "v8_training_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"Training metrics and provenance logged to: {metrics_path}\n", flush=True)


if __name__ == "__main__":
    train()
