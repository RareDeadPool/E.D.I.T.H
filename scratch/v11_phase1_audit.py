"""V11 Baseline Audit Script for E.D.I.T.H. Intelligence Core.
Performs thorough, independent, programmatic verification of:
1. Model identity, weights, shapes, runtime loading, heads, overrides.
2. Dataset uniqueness, duplicates, leakage, label contradictions, class balance.
3. Slot-accuracy regression investigation (token-level, entity-span, per-slot, confusion).
4. Metric verification (Intent, Action, Macro-F1, OOD, Confidence Calibration ECE & Brier).
"""

import os
import sys
import json
import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Set
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from edith_ann.tokenizer import EdithTokenizer
from edith_ann.model import EdithNeuralNetwork
from train import EdithDataset, EdithMultiTaskLoss, EdithOnnxWrapper


def audit_model_identity():
    print("=" * 60)
    print(" 1. AUDIT MODEL IDENTITY & RUNTIME CONFIGURATION")
    print("=" * 60)

    pt_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_ann_best.pt")
    onnx_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_ann_best.onnx")
    tok_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "tokenizer.json")
    v8_metrics_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "v8_training_metrics.json")

    print(f"Loading checkpoint: {pt_path}")
    ckpt = torch.load(pt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    maps = ckpt.get("metadata_maps", {})
    vocab_size = ckpt.get("vocab_size", -1)
    state_dict = ckpt.get("model_state_dict", {})

    print(f"Checkpoint keys: {list(ckpt.keys())}")
    print(f"Epoch trained: {ckpt.get('epoch')}")
    print(f"Config: {json.dumps(cfg, indent=2)}")
    print(f"Vocab size in checkpoint: {vocab_size}")
    print(f"Intents count: {len(maps.get('intents', {}))}")
    print(f"Slots count: {len(maps.get('slots', {}))}")
    print(f"Actions count: {len(maps.get('actions', {}))}")

    # Count parameters
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"Total parameter tensors in state_dict: {len(state_dict)}")
    print(f"Total parameters: {total_params:,}")

    # Check Tokenizer
    tok = EdithTokenizer.load(tok_path)
    print(f"Tokenizer vocab size from {tok_path}: {tok.vocab_size}")
    vocab_match = (tok.vocab_size == vocab_size)
    print(f"Tokenizer matches checkpoint vocab size: {vocab_match}")

    # Check Sequence Length
    cfg_max_len = cfg.get("model", {}).get("max_seq_len", -1)
    print(f"Config max_seq_len: {cfg_max_len}")

    # Check ONNX
    onnx_exists = os.path.exists(onnx_path)
    print(f"ONNX file exists: {onnx_exists} ({onnx_path})")
    onnx_info = {}
    if onnx_exists:
        try:
            import onnx
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            inputs = [(inp.name, [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]) for inp in onnx_model.graph.input]
            outputs = [(out.name, [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]) for out in onnx_model.graph.output]
            print(f"ONNX Graph Inputs: {inputs}")
            print(f"ONNX Graph Outputs: {outputs}")
            onnx_info = {"inputs": inputs, "outputs": outputs}
        except Exception as e:
            print(f"Error checking ONNX: {e}")

    # Verify PyTorch and ONNX numerical output
    dummy_input = torch.randint(0, min(100, vocab_size), (1, cfg_max_len), dtype=torch.long)
    dummy_mask = torch.ones((1, cfg_max_len), dtype=torch.long)

    model = EdithNeuralNetwork(
        vocab_size=vocab_size,
        num_intents=len(maps["intents"]),
        num_slots=len(maps["slots"]),
        num_actions=len(maps["actions"]),
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        n_layers=cfg["model"]["n_layers"],
        d_ff=cfg["model"]["d_ff"],
        max_seq_len=cfg_max_len,
        dropout=0.0
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    with torch.no_grad():
        py_out = model(dummy_input, attention_mask=dummy_mask)

    if onnx_exists:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"input_ids": dummy_input.numpy(), "attention_mask": dummy_mask.numpy()})
        diff_intent = float(np.max(np.abs(py_out["intent_logits"].numpy() - ort_out[0])))
        diff_slot = float(np.max(np.abs(py_out["slot_logits"].numpy() - ort_out[1])))
        diff_action = float(np.max(np.abs(py_out["action_logits"].numpy() - ort_out[2])))
        diff_conf = float(np.max(np.abs(py_out["confidence"].numpy() - ort_out[3])))
        diff_comp = float(np.max(np.abs(py_out["complexity_logits"].numpy() - ort_out[4])))
        print(f"PyTorch vs ONNX max diffs: intent={diff_intent:.6e}, slot={diff_slot:.6e}, action={diff_action:.6e}, conf={diff_conf:.6e}, comp={diff_comp:.6e}")

    # Check v8 training metrics
    v8_metrics = {}
    if os.path.exists(v8_metrics_path):
        with open(v8_metrics_path, "r", encoding="utf-8") as f:
            v8_metrics = json.load(f)
        print(f"Logged Test Macro-F1: {v8_metrics.get('test_metrics', {}).get('macro_f1')}%")
        print(f"Logged Test Slot Acc: {v8_metrics.get('test_metrics', {}).get('slot_acc')}%")
        print(f"Logged Promoted: {v8_metrics.get('promoted_to_production')}")

    return {
        "vocab_size": vocab_size,
        "total_params": total_params,
        "max_seq_len": cfg_max_len,
        "onnx_info": onnx_info,
        "v8_metrics": v8_metrics,
        "metadata_maps": maps
    }


def audit_dataset_integrity():
    print("\n" + "=" * 60)
    print(" 2. AUDIT DATASET INTEGRITY & SPLIT LEAKAGE")
    print("=" * 60)

    train_path = os.path.join(PROJECT_ROOT, "data", "train_data.json")
    val_path = os.path.join(PROJECT_ROOT, "data", "val_data.json")
    test_path = os.path.join(PROJECT_ROOT, "data", "test_data.json")

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    print(f"Total raw counts: Train={len(train_data):,}, Val={len(val_data):,}, Test={len(test_data):,}")

    # 1. Exact sentence duplicates within and across splits
    train_texts = [d["text"].strip().lower() for d in train_data]
    val_texts = [d["text"].strip().lower() for d in val_data]
    test_texts = [d["text"].strip().lower() for d in test_data]

    train_uniq = set(train_texts)
    val_uniq = set(val_texts)
    test_uniq = set(test_texts)

    print(f"Unique texts: Train={len(train_uniq):,} ({len(train_data) - len(train_uniq)} internal dupes)")
    print(f"Unique texts: Val={len(val_uniq):,} ({len(val_data) - len(val_uniq)} internal dupes)")
    print(f"Unique texts: Test={len(test_uniq):,} ({len(test_data) - len(test_uniq)} internal dupes)")

    overlap_tr_va = train_uniq & val_uniq
    overlap_tr_te = train_uniq & test_uniq
    overlap_va_te = val_uniq & test_uniq

    print(f"Exact text overlaps:")
    print(f"  Train & Val overlap: {len(overlap_tr_va)} examples")
    print(f"  Train & Test overlap: {len(overlap_tr_te)} examples")
    print(f"  Val & Test overlap: {len(overlap_va_te)} examples")
    if overlap_tr_te:
        print(f"  Sample Train-Test overlap: {list(overlap_tr_te)[:5]}")

    # 2. Semantic / Template Leakage Analysis
    # Let's inspect template patterns: how were these examples generated?
    # Strip slots / entities and see if the underlying syntactic template is identical across train & test!
    def abstract_entities(d):
        text = d["text"].strip().lower()
        # replace words that have non-O tags with [ENTITY]
        words = text.split()
        tags = d.get("slot_tags", [])
        abstracted = []
        for i, w in enumerate(words):
            tag = tags[i] if i < len(tags) else "O"
            if tag != "O":
                entity_name = tag.split("-")[-1]
                if not abstracted or abstracted[-1] != f"[{entity_name}]":
                    abstracted.append(f"[{entity_name}]")
            else:
                abstracted.append(w)
        return " ".join(abstracted)

    train_templates = Counter(abstract_entities(d) for d in train_data)
    val_templates = Counter(abstract_entities(d) for d in val_data)
    test_templates = Counter(abstract_entities(d) for d in test_data)

    print(f"\nAbstracted syntactic structures:")
    print(f"  Train templates count: {len(train_templates)}")
    print(f"  Val templates count: {len(val_templates)}")
    print(f"  Test templates count: {len(test_templates)}")

    template_overlap_tr_va = set(train_templates.keys()) & set(val_templates.keys())
    template_overlap_tr_te = set(train_templates.keys()) & set(test_templates.keys())
    template_overlap_va_te = set(val_templates.keys()) & set(test_templates.keys())

    print(f"  Train-Val template overlap: {len(template_overlap_tr_va)}")
    print(f"  Train-Test template overlap: {len(template_overlap_tr_te)}")
    print(f"  Val-Test template overlap: {len(template_overlap_va_te)}")
    if template_overlap_tr_te:
        print(f"  Examples of shared templates in Train & Test: {list(template_overlap_tr_te)[:5]}")

    # 3. Label Contradictions: Check if same text has different intent or action or slots
    text_to_labels = defaultdict(list)
    for split_name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        for item in data:
            t = item["text"].strip().lower()
            text_to_labels[t].append((split_name, item["intent_id"], item["action_id"], tuple(item.get("slot_tags", []))))

    contradictory_intents = {}
    contradictory_actions = {}
    contradictory_slots = {}
    for t, instances in text_to_labels.items():
        intents = set(inst[1] for inst in instances)
        actions = set(inst[2] for inst in instances)
        slots = set(inst[3] for inst in instances)
        if len(intents) > 1:
            contradictory_intents[t] = instances
        if len(actions) > 1:
            contradictory_actions[t] = instances
        if len(slots) > 1:
            contradictory_slots[t] = instances

    print(f"\nLabel Consistency Audit:")
    print(f"  Texts with contradictory intent labels: {len(contradictory_intents)}")
    if contradictory_intents:
        for k in list(contradictory_intents.keys())[:3]:
            print(f"    '{k}': {contradictory_intents[k]}")
    print(f"  Texts with contradictory action labels: {len(contradictory_actions)}")
    if contradictory_actions:
        for k in list(contradictory_actions.keys())[:3]:
            print(f"    '{k}': {contradictory_actions[k]}")
    print(f"  Texts with contradictory slot labels: {len(contradictory_slots)}")
    if contradictory_slots:
        for k in list(contradictory_slots.keys())[:3]:
            print(f"    '{k}': {contradictory_slots[k]}")

    # 4. Class Balance
    train_intent_dist = Counter(d["intent_id"] for d in train_data)
    test_intent_dist = Counter(d["intent_id"] for d in test_data)
    print(f"\nClass balance:")
    print(f"  Train intent distribution: {dict(sorted(train_intent_dist.items()))}")
    print(f"  Test intent distribution: {dict(sorted(test_intent_dist.items()))}")

    return {
        "train_count": len(train_data),
        "val_count": len(val_data),
        "test_count": len(test_data),
        "train_unique": len(train_uniq),
        "exact_overlaps": {
            "tr_va": len(overlap_tr_va),
            "tr_te": len(overlap_tr_te),
            "va_te": len(overlap_va_te),
        },
        "template_overlaps": {
            "tr_va": len(template_overlap_tr_va),
            "tr_te": len(template_overlap_tr_te),
            "va_te": len(template_overlap_va_te),
        },
        "contradictions": {
            "intent": len(contradictory_intents),
            "action": len(contradictory_actions),
            "slots": len(contradictory_slots)
        }
    }


def audit_slot_regression_and_metrics():
    print("\n" + "=" * 60)
    print(" 3. AUDIT SLOT REGRESSION & MULTI-HEAD EVALUATION")
    print("=" * 60)

    pt_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_ann_best.pt")
    tok_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "tokenizer.json")
    val_path = os.path.join(PROJECT_ROOT, "data", "val_data.json")
    test_path = os.path.join(PROJECT_ROOT, "data", "test_data.json")

    ckpt = torch.load(pt_path, map_location="cpu")
    cfg = ckpt["config"]
    maps = ckpt["metadata_maps"]
    tokenizer = EdithTokenizer.load(tok_path)

    intent_map = maps["intents"]
    slot_map = maps["slots"]
    action_map = maps["actions"]
    inv_slot_map = {v: k for k, v in slot_map.items()}
    inv_intent_map = {v: k for k, v in intent_map.items()}
    inv_action_map = {v: k for k, v in action_map.items()}

    model = EdithNeuralNetwork(
        vocab_size=ckpt["vocab_size"],
        num_intents=len(intent_map),
        num_slots=len(slot_map),
        num_actions=len(action_map),
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        n_layers=cfg["model"]["n_layers"],
        d_ff=cfg["model"]["d_ff"],
        max_seq_len=cfg["model"]["max_seq_len"],
        dropout=0.0
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    def evaluate_comprehensive(split_name: str, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        max_seq_len = cfg["model"]["max_seq_len"]

        # Trackers
        total_samples = len(samples)
        correct_intent = 0
        correct_action = 0
        correct_comp = 0

        # Slot tracking
        total_tokens_all = 0
        correct_tokens_all = 0
        total_non_o_tokens = 0
        correct_non_o_tokens = 0

        exact_sequence_matches = 0

        # Per-slot TP, FP, FN
        num_slots = len(slot_map)
        slot_tp = [0] * num_slots
        slot_fp = [0] * num_slots
        slot_fn = [0] * num_slots

        # Slot confusion matrix (top confusion pairs)
        slot_confusion = Counter()

        # Entity Span Exact Match tracking
        # An entity span is (start_word_idx, end_word_idx, entity_type)
        total_gold_spans = 0
        total_pred_spans = 0
        matched_spans = 0

        # Calibration & OOD tracking
        conf_preds = []
        conf_golds = []
        is_ood_preds = []
        is_ood_golds = []

        # Weakest slot error examples
        slot_errors = defaultdict(list)

        for s_idx, sample in enumerate(samples):
            text = sample["text"]
            gold_slots = sample.get("slot_tags", [])
            gold_intent = sample["intent_id"]
            gold_action = sample["action_id"]
            gold_conf = sample["confidence"]
            gold_comp = sample.get("complexity", 0)

            input_ids, attention_mask = tokenizer.encode(
                text, max_len=max_seq_len, add_special_tokens=True, pad_to_max=True
            )

            t_ids = torch.tensor([input_ids], dtype=torch.long)
            t_mask = torch.tensor([attention_mask], dtype=torch.long)

            with torch.no_grad():
                out = model(t_ids, attention_mask=t_mask)

            pred_intent = int(torch.argmax(out["intent_logits"], dim=-1)[0].item())
            pred_action = int(torch.argmax(out["action_logits"], dim=-1)[0].item())
            pred_conf = float(out["confidence"][0][0].item())
            pred_comp = int(torch.argmax(out["complexity_logits"], dim=-1)[0].item())
            pred_slots_all = torch.argmax(out["slot_logits"], dim=-1)[0].tolist()

            if pred_intent == gold_intent:
                correct_intent += 1
            if pred_action == gold_action:
                correct_action += 1
            if pred_comp == gold_comp:
                correct_comp += 1

            # Confidence & OOD
            conf_preds.append(pred_conf)
            conf_golds.append(gold_conf)
            is_ood_gold = (gold_conf < 0.5)
            is_ood_pred = (pred_conf < 0.5)
            is_ood_golds.append(is_ood_gold)
            is_ood_preds.append(is_ood_pred)

            # Slot alignment: words map to positions 1..min(len(gold_slots), max_seq_len-2)
            usable_len = min(len(gold_slots), max_seq_len - 2)
            seq_match = True

            gold_spans = []
            curr_g_type = None
            curr_g_start = -1

            pred_spans = []
            curr_p_type = None
            curr_p_start = -1

            words = text.split()

            for i in range(usable_len):
                g_tag_str = gold_slots[i]
                g_id = slot_map.get(g_tag_str, 0)
                p_id = pred_slots_all[i + 1]
                p_tag_str = inv_slot_map.get(p_id, "O")

                total_tokens_all += 1
                if p_id == g_id:
                    correct_tokens_all += 1
                    slot_tp[g_id] += 1
                else:
                    seq_match = False
                    slot_fp[p_id] += 1
                    slot_fn[g_id] += 1
                    slot_confusion[(g_tag_str, p_tag_str)] += 1
                    if len(slot_errors[g_tag_str]) < 5:
                        slot_errors[g_tag_str].append({
                            "text": text,
                            "word": words[i] if i < len(words) else "?",
                            "pos": i,
                            "gold": g_tag_str,
                            "pred": p_tag_str
                        })

                if g_tag_str != "O":
                    total_non_o_tokens += 1
                    if p_id == g_id:
                        correct_non_o_tokens += 1

                # Span parsing for gold
                if g_tag_str.startswith("B-"):
                    if curr_g_type:
                        gold_spans.append((curr_g_start, i, curr_g_type))
                    curr_g_type = g_tag_str[2:]
                    curr_g_start = i
                elif g_tag_str.startswith("I-") and curr_g_type == g_tag_str[2:]:
                    pass
                else:
                    if curr_g_type:
                        gold_spans.append((curr_g_start, i, curr_g_type))
                        curr_g_type = None

                # Span parsing for pred
                if p_tag_str.startswith("B-"):
                    if curr_p_type:
                        pred_spans.append((curr_p_start, i, curr_p_type))
                    curr_p_type = p_tag_str[2:]
                    curr_p_start = i
                elif p_tag_str.startswith("I-") and curr_p_type == p_tag_str[2:]:
                    pass
                else:
                    if curr_p_type:
                        pred_spans.append((curr_p_start, i, curr_p_type))
                        curr_p_type = None

            if curr_g_type:
                gold_spans.append((curr_g_start, usable_len, curr_g_type))
            if curr_p_type:
                pred_spans.append((curr_p_start, usable_len, curr_p_type))

            if seq_match and usable_len == len(gold_slots):
                exact_sequence_matches += 1

            total_gold_spans += len(gold_spans)
            total_pred_spans += len(pred_spans)
            matched_spans += len(set(gold_spans) & set(pred_spans))

        # Metrics computation
        intent_acc = (correct_intent / total_samples) * 100.0
        action_acc = (correct_action / total_samples) * 100.0
        comp_acc = (correct_comp / total_samples) * 100.0
        ood_acc = (sum(1 for p, g in zip(is_ood_preds, is_ood_golds) if p == g) / total_samples) * 100.0

        # Token-level slot accuracy
        token_slot_acc = (correct_tokens_all / max(1, total_tokens_all)) * 100.0
        non_o_slot_acc = (correct_non_o_tokens / max(1, total_non_o_tokens)) * 100.0
        exact_seq_acc = (exact_sequence_matches / total_samples) * 100.0

        # Entity Span metrics
        span_prec = (matched_spans / max(1, total_pred_spans)) * 100.0
        span_rec = (matched_spans / max(1, total_gold_spans)) * 100.0
        span_f1 = (2 * span_prec * span_rec / max(1e-9, span_prec + span_rec))

        # Per-slot F1
        per_slot_metrics = {}
        slot_f1_list = []
        for s_id in range(num_slots):
            tp = slot_tp[s_id]
            fp = slot_fp[s_id]
            fn = slot_fn[s_id]
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
            tag_name = inv_slot_map[s_id]
            per_slot_metrics[tag_name] = {
                "support": tp + fn,
                "precision": round(p * 100, 2),
                "recall": round(r * 100, 2),
                "f1": round(f1 * 100, 2)
            }
            if (tp + fn) > 0:
                slot_f1_list.append(f1 * 100)

        macro_slot_f1 = sum(slot_f1_list) / max(1, len(slot_f1_list))

        # Calibration metrics: Brier score & ECE
        brier_score = float(np.mean([(p - g) ** 2 for p, g in zip(conf_preds, conf_golds)]))

        # ECE (Expected Calibration Error) with 10 bins
        bins = np.linspace(0.0, 1.0, 11)
        ece = 0.0
        for i in range(10):
            bin_lower = bins[i]
            bin_upper = bins[i + 1]
            in_bin = [(p, g) for p, g in zip(conf_preds, conf_golds) if bin_lower <= p < bin_upper]
            if in_bin:
                prop = len(in_bin) / len(conf_preds)
                acc_in_bin = np.mean([g for p, g in in_bin])
                conf_in_bin = np.mean([p for p, g in in_bin])
                ece += prop * abs(acc_in_bin - conf_in_bin)

        print(f"\n--- EVALUATION ON {split_name.upper()} SPLIT ({total_samples} samples) ---")
        print(f"  Intent Accuracy:           {intent_acc:.2f}%")
        print(f"  Action Accuracy:           {action_acc:.2f}%")
        print(f"  Complexity Accuracy:       {comp_acc:.2f}%")
        print(f"  OOD Detection Accuracy:    {ood_acc:.2f}%")
        print(f"  Token Slot Accuracy (All): {token_slot_acc:.2f}%")
        print(f"  Token Slot Accuracy (Non-O): {non_o_slot_acc:.2f}%")
        print(f"  Exact Sequence Match:      {exact_seq_acc:.2f}%")
        print(f"  Entity Span Precision:     {span_prec:.2f}%")
        print(f"  Entity Span Recall:        {span_rec:.2f}%")
        print(f"  Entity Span F1:            {span_f1:.2f}%")
        print(f"  Macro Slot F1 (active):    {macro_slot_f1:.2f}%")
        print(f"  Confidence Brier Score:    {brier_score:.4f}")
        print(f"  Expected Calibration Error: {ece:.4f}")

        # Weakest 10 slots by F1 (with support > 5)
        active_slots = [(k, v) for k, v in per_slot_metrics.items() if v["support"] >= 5]
        active_slots.sort(key=lambda x: x[1]["f1"])
        print(f"\n  Weakest Active Slots on {split_name}:")
        for tag, m in active_slots[:10]:
            print(f"    {tag:<15} (Support: {m['support']:3d}) | Prec: {m['precision']:5.1f}% | Rec: {m['recall']:5.1f}% | F1: {m['f1']:5.1f}%")

        print(f"\n  Top 10 Slot Confusions on {split_name} (Gold -> Pred):")
        for (g_t, p_t), count in slot_confusion.most_common(10):
            print(f"    {g_t:<15} -> {p_t:<15} : {count:4d} times")

        return {
            "intent_acc": intent_acc,
            "action_acc": action_acc,
            "comp_acc": comp_acc,
            "ood_acc": ood_acc,
            "token_slot_acc": token_slot_acc,
            "non_o_slot_acc": non_o_slot_acc,
            "exact_seq_acc": exact_seq_acc,
            "span_f1": span_f1,
            "macro_slot_f1": macro_slot_f1,
            "brier_score": brier_score,
            "ece": ece,
            "per_slot_metrics": per_slot_metrics,
            "slot_confusion": dict(slot_confusion.most_common(20)),
            "slot_errors": {k: slot_errors[k] for k in list(slot_errors.keys())[:5]}
        }

    val_res = evaluate_comprehensive("val", val_path)
    test_res = evaluate_comprehensive("test", test_path)

    return {"val": val_res, "test": test_res}


def audit_tokenization_and_vocab():
    print("\n" + "=" * 60)
    print(" 4. AUDIT TOKENIZATION, VOCABULARY & LENGTH COVERAGE")
    print("=" * 60)

    tok_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "tokenizer.json")
    train_path = os.path.join(PROJECT_ROOT, "data", "train_data.json")
    val_path = os.path.join(PROJECT_ROOT, "data", "val_data.json")
    test_path = os.path.join(PROJECT_ROOT, "data", "test_data.json")

    tokenizer = EdithTokenizer.load(tok_path)
    print(f"Loaded tokenizer with {tokenizer.vocab_size} vocabulary tokens.")
    print(f"Special tokens: PAD={tokenizer.PAD_ID}, UNK={tokenizer.UNK_ID}, BOS={tokenizer.BOS_ID}, EOS={tokenizer.EOS_ID}")

    # Inspect tokenization strategy
    sample_text = "Please send a message to Sakshi on WhatsApp and play some lofi music!"
    raw_tokens = tokenizer.tokenize_raw(sample_text)
    encoded_ids, mask = tokenizer.encode(sample_text, max_len=64)
    decoded_words = [tokenizer.inv_vocab.get(i, f"[{i}]") for i in encoded_ids if i != tokenizer.PAD_ID]
    print(f"Tokenize raw sample: '{sample_text}'")
    print(f"Raw tokens: {raw_tokens}")
    print(f"Decoded back from vocab: {decoded_words}")

    # Measure UNK token rates across datasets
    all_splits = [("train", train_path), ("val", val_path), ("test", test_path)]
    unk_stats = {}

    for name, path in all_splits:
        with open(path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        total_words = 0
        unk_words = 0
        samples_with_unk = 0
        exceeds_32 = 0
        exceeds_64 = 0
        lengths = []
        unk_by_intent = defaultdict(lambda: [0, 0])

        for s in samples:
            text = s["text"]
            intent_id = s["intent_id"]
            tokens = tokenizer.tokenize_raw(text)
            lengths.append(len(tokens))
            if len(tokens) > 32:
                exceeds_32 += 1
            if len(tokens) > 64:
                exceeds_64 += 1

            has_unk = False
            for t in tokens:
                total_words += 1
                unk_by_intent[intent_id][1] += 1
                if t not in tokenizer.vocab:
                    unk_words += 1
                    unk_by_intent[intent_id][0] += 1
                    has_unk = True

            if has_unk:
                samples_with_unk += 1

        unk_rate = (unk_words / max(1, total_words)) * 100.0
        sample_unk_rate = (samples_with_unk / max(1, len(samples))) * 100.0

        print(f"\nSplit: {name.upper()} ({len(samples)} samples)")
        print(f"  Total words: {total_words:,} | UNK words: {unk_words:,} ({unk_rate:.2f}%)")
        print(f"  Samples containing at least one UNK: {samples_with_unk} ({sample_unk_rate:.2f}%)")
        print(f"  Token lengths: min={min(lengths)}, median={int(np.median(lengths))}, max={max(lengths)}, mean={np.mean(lengths):.1f}")
        print(f"  Inputs > 32 tokens: {exceeds_32} ({exceeds_32 / len(samples) * 100:.2f}%)")
        print(f"  Inputs > 64 tokens: {exceeds_64} ({exceeds_64 / len(samples) * 100:.2f}%)")

        unk_stats[name] = {
            "total_words": total_words,
            "unk_words": unk_words,
            "unk_rate": unk_rate,
            "sample_unk_rate": sample_unk_rate,
            "exceeds_32": exceeds_32,
            "exceeds_64": exceeds_64,
            "max_len": max(lengths)
        }

    # Evaluate challenging vocabulary categories
    test_categories = {
        "technical": ["docker containerization", "kubernetes cluster", "postgres deadlock", "asyncio event loop", "kafka partition lag"],
        "hinglish": ["spotify pe gaana bajao", "sakshi ko message bhejo", "chrome kholo aur time kya hua", "downloads ko organize kar do"],
        "unseen_apps": ["notion", "telegram", "postman", "wireshark", "blender", "steam", "obsidian", "figma"],
        "unseen_entities": ["ananya", "rohit", "quarterly_budget_2026.xlsx", "thesis_final_approved.pdf", "main_router_gateway"]
    }

    print("\nVocabulary coverage on challenging natural categories:")
    for cat, phrases in test_categories.items():
        cat_words = 0
        cat_unk = 0
        for p in phrases:
            for w in tokenizer.tokenize_raw(p):
                cat_words += 1
                if w not in tokenizer.vocab:
                    cat_unk += 1
        print(f"  Category '{cat}': {cat_unk}/{cat_words} words are UNK ({cat_unk/cat_words*100:.1f}%)")

    return unk_stats


def main():
    print("STARTING E.D.I.T.H. V11 PHASE 1 BASELINE AUDIT...")
    m_info = audit_model_identity()
    d_info = audit_dataset_integrity()
    s_info = audit_slot_regression_and_metrics()
    t_info = audit_tokenization_and_vocab()

    audit_summary = {
        "model_identity": m_info,
        "dataset_integrity": d_info,
        "slot_and_metrics": s_info,
        "tokenization_and_vocab": t_info
    }

    out_file = os.path.join(PROJECT_ROOT, "scratch", "v11_phase1_audit_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        # Convert non-serializable objects
        json.dump(audit_summary, f, indent=2, default=str)
    print(f"\nAudit complete! Full JSON saved to: {out_file}")


if __name__ == "__main__":
    main()
