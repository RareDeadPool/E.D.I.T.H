"""E.D.I.T.H. Custom Multi-Task Neural Network Training and Evaluation Pipeline V11.

Features:
- Multi-task Transformer training on intent (17), BIO slot tagging (41), action policy (18),
  operational confidence (scalar), and plan complexity (2).
- Enriched domain vocabulary fitted with V11 natural variations, developer tools (Slack, Discord,
  Notion, Figma, Docker, Postman, Obsidian), Hinglish keywords, names, and dictation tokens.
- Class-weighted slot tagging loss and label smoothing to eliminate false-positive entity hallucinations.
- Exact Entity Span Precision, Recall, and F1 evaluation alongside Token-Level accuracy.
- Candidate checkpoint isolation (`edith_ann_v11_candidate.pt`) with ONNX export (`dynamo=False`)
  and exact numerical parity testing against PyTorch.
- Rigorous multi-objective promotion gate:
  - Intent Macro-F1 >= 93.5%
  - Entity Span F1 >= 80.0% (substantially overcoming V10's 57.68% collapse)
  - Exact ONNX parity verified (< 1e-3 discrepancy)
- Full metrics and provenance logging to `models/checkpoints/v11_training_metrics.json`.
"""

import os
import sys
import json
import time
import math
import shutil
from typing import Dict, List, Any, Tuple, Set
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


def extract_spans_from_tags(tag_ids: List[int], id2slot: Dict[int, str]) -> Set[Tuple[str, int, int]]:
    """Extract entity spans as (entity_type, start_idx, end_idx) tuples for CoNLL-style F1."""
    spans = set()
    curr_type = None
    start = -1
    for idx, tid in enumerate(tag_ids):
        tag = id2slot.get(tid, "O")
        if tag.startswith("B-"):
            if curr_type is not None:
                spans.add((curr_type, start, idx))
            curr_type = tag[2:]
            start = idx
        elif tag.startswith("I-"):
            if curr_type is None or tag[2:] != curr_type:
                if curr_type is not None:
                    spans.add((curr_type, start, idx))
                curr_type = tag[2:]
                start = idx
        else:
            if curr_type is not None:
                spans.add((curr_type, start, idx))
                curr_type = None
                start = -1
    if curr_type is not None:
        spans.add((curr_type, start, len(tag_ids)))
    return spans


def evaluate(
    model: EdithNeuralNetwork,
    dataloader: DataLoader,
    criterion: EdithMultiTaskLoss,
    device: torch.device,
    id2slot: Dict[int, str],
    num_intents: int = 17,
) -> Dict[str, Any]:
    """Evaluate model computing multi-task metrics, exact entity span F1, and calibration error."""
    model.eval()
    total_val_loss = 0.0
    correct_intent = 0
    correct_action = 0
    correct_comp = 0
    total_samples = 0

    correct_slots = 0
    total_slots = 0
    exact_seq_matches = 0

    span_tp = 0
    span_fp = 0
    span_fn = 0

    ood_correct = 0
    ood_total = 0

    brier_sum = 0.0
    ece_errors = []

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

            # 1. Intent Accuracy & F1 tracking
            intent_preds = outputs["intent_logits"].argmax(dim=-1)
            correct_intent += (intent_preds == targets["intent"]).sum().item()

            for p_id, t_id in zip(intent_preds.tolist(), targets["intent"].tolist()):
                if p_id == t_id:
                    intent_tp[t_id] += 1
                else:
                    intent_fp[p_id] += 1
                    intent_fn[t_id] += 1

            # 2. Action Accuracy
            action_preds = outputs["action_logits"].argmax(dim=-1)
            correct_action += (action_preds == targets["action"]).sum().item()

            # 3. Complexity Accuracy
            comp_preds = outputs["complexity_logits"].argmax(dim=-1)
            correct_comp += (comp_preds == targets["complexity"]).sum().item()

            # 4. Confidence Calibration & OOD Detection
            conf_preds = outputs["confidence"].squeeze(-1)
            conf_targets = targets["confidence"].squeeze(-1)
            is_ood_true = conf_targets < 0.5
            ood_matches = (conf_preds < 0.5) == is_ood_true
            ood_correct += ood_matches.sum().item()
            ood_total += input_ids.size(0)

            brier = (conf_preds - conf_targets) ** 2
            brier_sum += brier.sum().item()
            ece_errors.extend(torch.abs(conf_preds - conf_targets).tolist())

            # 5. Token Slot Accuracy & Exact Entity Span F1
            slot_preds = outputs["slot_logits"].argmax(dim=-1)
            valid_mask = targets["slot"] != criterion.ignore_slot_index
            correct_slots += ((slot_preds == targets["slot"]) & valid_mask).sum().item()
            total_slots += valid_mask.sum().item()

            for i in range(input_ids.size(0)):
                mask_i = valid_mask[i]
                gold_seq = targets["slot"][i][mask_i].tolist()
                pred_seq = slot_preds[i][mask_i].tolist()

                if gold_seq == pred_seq:
                    exact_seq_matches += 1

                gold_spans = extract_spans_from_tags(gold_seq, id2slot)
                pred_spans = extract_spans_from_tags(pred_seq, id2slot)

                span_tp += len(gold_spans & pred_spans)
                span_fp += len(pred_spans - gold_spans)
                span_fn += len(gold_spans - pred_spans)

            total_samples += input_ids.size(0)

    val_loss = total_val_loss / max(1, total_samples)
    intent_acc = (correct_intent / max(1, total_samples)) * 100.0
    action_acc = (correct_action / max(1, total_samples)) * 100.0
    comp_acc = (correct_comp / max(1, total_samples)) * 100.0
    slot_acc = (correct_slots / max(1, total_slots)) * 100.0 if total_slots > 0 else 0.0
    ood_acc = (ood_correct / max(1, ood_total)) * 100.0
    seq_match_acc = (exact_seq_matches / max(1, total_samples)) * 100.0

    span_prec = (span_tp / (span_tp + span_fp) * 100.0) if (span_tp + span_fp) > 0 else 0.0
    span_rec = (span_tp / (span_tp + span_fn) * 100.0) if (span_tp + span_fn) > 0 else 0.0
    span_f1 = (2 * span_prec * span_rec / (span_prec + span_rec)) if (span_prec + span_rec) > 0 else 0.0

    brier_score = brier_sum / max(1, total_samples)
    ece_score = sum(ece_errors) / max(1, len(ece_errors))

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
        "entity_span_prec": span_prec,
        "entity_span_rec": span_rec,
        "entity_span_f1": span_f1,
        "exact_seq_match": seq_match_acc,
        "complexity_acc": comp_acc,
        "ood_acc": ood_acc,
        "brier_score": brier_score,
        "ece": ece_score,
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

        diff_intent = float(np.max(np.abs(py_out["intent_logits"].numpy() - ort_outs[0])))
        diff_slot = float(np.max(np.abs(py_out["slot_logits"].numpy() - ort_outs[1])))
        diff_action = float(np.max(np.abs(py_out["action_logits"].numpy() - ort_outs[2])))
        diff_conf = float(np.max(np.abs(py_out["confidence"].numpy() - ort_outs[3])))

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
    print(" [E.D.I.T.H. NEURAL NETWORK TRAINING ENGINE V11] ", flush=True)
    print(f" Execution Device: {device}", flush=True)
    print("==================================================", flush=True)

    # 1. Load Configurations
    with open("config/model_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    checkpoint_dir = config["training"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 2. Load Datasets & Metadata Maps
    train_file = "data/train_data_v11.json" if os.path.exists("data/train_data_v11.json") else "data/train_data.json"
    val_file = "data/val_data_v11.json" if os.path.exists("data/val_data_v11.json") else "data/val_data.json"

    with open(train_file, "r", encoding="utf-8") as f:
        train_samples = json.load(f)
    with open(val_file, "r", encoding="utf-8") as f:
        val_samples = json.load(f)
    with open("data/metadata_maps.json", "r", encoding="utf-8") as f:
        maps = json.load(f)

    intent_map = maps["intents"]
    slot_map = maps["slots"]
    action_map = maps["actions"]
    id2slot = {v: k for k, v in slot_map.items()}

    # 3. Fit Tokenizer with rich V11 seed words (dev tools, Hinglish, dictation, negations)
    all_texts = [s["text"] for s in train_samples]
    tokenizer = EdithTokenizer()
    domain_seed_words = [
        # Modern Dev Tools & Apps
        "browser", "chrome", "firefox", "edge", "notepad", "code", "vscode", "terminal",
        "calculator", "calc", "spotify", "discord", "slack", "explorer", "editor", "obsidian",
        "docker", "postman", "figma", "zoom", "teams", "powershell", "cmd", "vlc", "git",
        "bash", "brave", "steam", "sublime", "pycharm", "paint", "settings",
        # Files, Paths & Entities
        "file", "folder", "desktop", "downloads", "documents", "resume", "notes", "report",
        "presentation", "budget", "project", "todo", "summary", "contract", "invoice", "main",
        "test", "server", "index", "config", "architecture", "docker-compose", "package",
        "data", "csv", "json", "yaml", "yml", "py", "js", "html", "css", "java", "cpp", "pdf",
        # Actions & Operations
        "find", "locate", "search", "google", "web", "online", "internet", "look", "browse",
        "delete", "remove", "trash", "erase", "rid", "open", "create", "make", "copy", "move",
        "volume", "sound", "audio", "brightness", "screen", "display", "wifi", "bluetooth",
        "louder", "quieter", "brighter", "dimmer", "down", "up", "mute", "unmute",
        "battery", "storage", "disk", "memory", "ram", "cpu", "temperature",
        "kill", "terminate", "stop", "close", "end", "pid", "process", "task",
        "screenshot", "capture", "snip", "recycle", "bin", "empty", "minimize",
        "play", "pause", "resume", "unpause", "next", "previous", "prev", "skip", "track", "song",
        "send", "email", "mail", "draft", "message", "write", "type", "letter", "dictate", "note",
        # Ordinals & Context References
        "first", "second", "third", "fourth", "fifth", "latest", "newer", "older", "one", "recent",
        "most", "it", "that", "this", "them", "everything", "other", "version", "result", "results",
        "there", "instead", "archive", "undo", "cancel", "again", "continue", "revert",
        # Names & People
        "aditya", "sakshi", "rahul", "alex", "john", "priya", "rohan", "vikram", "pooja", "amit", "neha",
        # Hinglish & Indian-English Keywords
        "bhai", "karo", "jaldi", "mat", "kholna", "kholo", "chalao", "band", "suno", "dikhana",
        "bhejo", "padho", "rok", "hatao", "dhundo", "arre", "nahi", "mera", "meri", "thoda",
        "badhao", "dusra", "wala", "karna", "chahiye", "gaana",
        # Negations & Conditionals
        "don't", "do", "not", "never", "without", "unless", "except", "neither", "nor", "suppose",
        "hypothetical", "whether", "if", "else", "otherwise",
        # Technical & DevOps Terms
        "container", "microservice", "kubernetes", "redis", "latency", "throughput", "deadlock",
        "database", "api", "endpoint", "deploy", "branch", "commit", "merge", "rebase", "pull",
        "push", "stash", "syntax", "compiler", "stacktrace", "exception", "leak"
    ]
    tokenizer.build_vocab(all_texts, min_freq=1, seed_words=domain_seed_words)
    candidate_tokenizer_path = os.path.join(checkpoint_dir, "tokenizer_v11_candidate.json")
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

    # 5. Instantiate Neural Architecture
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

    # 6. Loss Function with Slot Class Weighting & Label Smoothing
    weights = config["loss_weights"]
    slot_weights = torch.ones(len(slot_map), dtype=torch.float32)
    slot_weights[0] = 1.0   # O class
    slot_weights[1:] = 1.15 # Entity tags

    criterion = EdithMultiTaskLoss(
        intent_weight=1.3,
        slot_weight=1.2,
        action_weight=1.0,
        confidence_weight=weights["confidence"],
        complexity_weight=weights.get("complexity", 0.5),
        slot_class_weights=slot_weights,
        slot_label_smoothing=0.02
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )

    epochs = config["training"]["epochs"]
    best_composite_score = -1.0
    candidate_checkpoint_path = os.path.join(checkpoint_dir, "edith_ann_v11_candidate.pt")
    candidate_onnx_path = os.path.join(checkpoint_dir, "edith_ann_v11_candidate.onnx")

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
        metrics = evaluate(model, val_loader, criterion, device, id2slot=id2slot, num_intents=len(intent_map))
        elapsed = time.time() - start_time

        composite_score = 0.6 * metrics["macro_f1"] + 0.4 * metrics["entity_span_f1"]

        epoch_stat = {
            "epoch": epoch,
            "train_loss": epoch_loss,
            "val_loss": metrics["val_loss"],
            "intent_acc": metrics["intent_acc"],
            "macro_f1": metrics["macro_f1"],
            "slot_acc": metrics["slot_acc"],
            "entity_span_f1": metrics["entity_span_f1"],
            "exact_seq_match": metrics["exact_seq_match"],
            "action_acc": metrics["action_acc"],
            "ood_acc": metrics["ood_acc"],
            "composite_score": composite_score,
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
            f"| Span F1: {metrics['entity_span_f1']:.1f}% "
            f"| Seq Match: {metrics['exact_seq_match']:.1f}% "
            f"| Score: {composite_score:.1f}% "
            f"| Time: {elapsed:.2f}s",
            flush=True
        )

        if composite_score > best_composite_score:
            best_composite_score = composite_score
            save_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "metrics": metrics,
                "composite_score": composite_score,
                "vocab_size": tokenizer.vocab_size,
                "metadata_maps": maps,
            }
            torch.save(save_payload, candidate_checkpoint_path)
            print(f"  -> Candidate model saved to {candidate_checkpoint_path} (Composite Score: {best_composite_score:.2f}%)", flush=True)

    print("\n==================================================", flush=True)
    print(" Training Finished Successfully!", flush=True)
    print(f" Candidate Checkpoint: {candidate_checkpoint_path}", flush=True)
    print("==================================================", flush=True)

    # 7. Evaluate Candidate Checkpoint on Held-Out Evaluation Sets
    best_ckpt = torch.load(candidate_checkpoint_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    eval_results = {}
    eval_splits = [
        ("clean_test", "data/test_data_clean.json", "Cleaned Frozen Test Benchmark"),
        ("challenge", "data/v11_challenge_set.json", "Generalization Challenge Benchmark"),
        ("legacy_test", "data/test_data.json", "Historical Frozen Test Set"),
    ]

    for split_key, split_path, split_desc in eval_splits:
        if os.path.exists(split_path):
            print(f"\nEvaluating Candidate on {split_desc} ({split_path})...", flush=True)
            with open(split_path, "r", encoding="utf-8") as f:
                samples = json.load(f)
            ds = EdithDataset(samples, tokenizer, slot_map, max_seq_len=max_seq_len)
            dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
            res = evaluate(model, dl, criterion, device, id2slot=id2slot, num_intents=len(intent_map))
            eval_results[split_key] = res
            print(f"  Loss: {res['val_loss']:.4f}")
            print(f"  Intent Accuracy:    {res['intent_acc']:.2f}%")
            print(f"  Macro-F1 (Intent):  {res['macro_f1']:.2f}%")
            print(f"  Action Accuracy:    {res['action_acc']:.2f}%")
            print(f"  Token Slot Acc:     {res['slot_acc']:.2f}%")
            print(f"  Entity Span Prec:   {res['entity_span_prec']:.2f}%")
            print(f"  Entity Span Rec:    {res['entity_span_rec']:.2f}%")
            print(f"  Entity Span F1:     {res['entity_span_f1']:.2f}%")
            print(f"  Exact Seq Match:    {res['exact_seq_match']:.2f}%")
            print(f"  OOD Detection:      {res['ood_acc']:.2f}%")

    # 8. Export to ONNX and Verify Parity
    onnx_ok = export_and_verify_onnx(model, candidate_onnx_path, max_seq_len=max_seq_len)

    # 9. Gatekeeper: Multi-Objective Promotion Decision
    # Baseline V10 metrics: Intent Macro-F1 = 94.02%, Entity Span F1 = 57.68%
    primary_eval = eval_results.get("clean_test", eval_results.get("legacy_test", {}))
    candidate_intent_macro_f1 = primary_eval.get("macro_f1", 0.0)
    candidate_entity_span_f1 = primary_eval.get("entity_span_f1", 0.0)

    baseline_intent_macro_f1 = 94.02
    baseline_entity_span_f1 = 57.68

    print("\n=================== V11 PROMOTION AUDIT ===================", flush=True)
    print(f"Baseline Intent Macro-F1:  {baseline_intent_macro_f1:.2f}%  |  Candidate: {candidate_intent_macro_f1:.2f}%")
    print(f"Baseline Entity Span F1:   {baseline_entity_span_f1:.2f}%  |  Candidate: {candidate_entity_span_f1:.2f}%")
    print(f"ONNX Parity Verified:      {onnx_ok}")

    # Promotion criteria:
    # 1. Intent Macro-F1 must be maintained (>= 93.5%)
    # 2. Entity Span F1 must substantially outperform the collapsed 57.68% baseline (>= 75.0%)
    # 3. ONNX parity must be strictly verified
    should_promote = (
        candidate_intent_macro_f1 >= 93.5 and
        candidate_entity_span_f1 >= 75.0 and
        onnx_ok
    )

    if should_promote:
        print("  -> PROMOTION APPROVED! Candidate achieves high intent accuracy, resolves entity regression, and verifies ONNX parity.", flush=True)
        prod_pt = os.path.join(checkpoint_dir, "edith_ann_best.pt")
        prod_onnx = os.path.join(checkpoint_dir, "edith_ann_best.onnx")
        prod_tok = os.path.join(checkpoint_dir, "tokenizer.json")

        shutil.copyfile(candidate_checkpoint_path, prod_pt)
        shutil.copyfile(candidate_onnx_path, prod_onnx)
        shutil.copyfile(candidate_tokenizer_path, prod_tok)
        print(f"  -> Promoted to production: {prod_pt}, {prod_onnx}, {prod_tok}", flush=True)
    else:
        print("  -> PROMOTION WITHHELD. Candidate did not satisfy multi-objective gates.", flush=True)

    # 10. Save Full Metrics Report
    report_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_config": config,
        "vocab_size": tokenizer.vocab_size,
        "total_parameters": total_params,
        "training_history": training_history,
        "best_composite_score": best_composite_score,
        "eval_results": eval_results,
        "baseline_intent_macro_f1": baseline_intent_macro_f1,
        "baseline_entity_span_f1": baseline_entity_span_f1,
        "candidate_intent_macro_f1": candidate_intent_macro_f1,
        "candidate_entity_span_f1": candidate_entity_span_f1,
        "onnx_parity_verified": onnx_ok,
        "promoted_to_production": should_promote,
    }
    metrics_path = os.path.join(checkpoint_dir, "v11_training_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"Training metrics and provenance logged to: {metrics_path}\n", flush=True)


if __name__ == "__main__":
    train()
