import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Set, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from edith_ann.tokenizer import EdithTokenizer
from edith_ann.model import EdithNeuralNetwork


def extract_spans(tag_ids: List[int], id2slot: Dict[int, str]) -> Set[Tuple[str, int, int]]:
    """Extract entity spans as (entity_type, start_idx, end_idx) tuples."""
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


def compute_ece(confidences: List[float], correct_flags: List[bool], num_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    total_samples = len(confidences)
    if total_samples == 0:
        return 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        mask = [(conf >= bin_lower) and (conf < bin_upper or (i == num_bins - 1 and conf <= bin_upper))
                for conf in confidences]
        bin_count = sum(mask)
        if bin_count > 0:
            bin_acc = sum([c for c, m in zip(correct_flags, mask) if m]) / bin_count
            bin_conf = sum([conf for conf, m in zip(confidences, mask) if m]) / bin_count
            ece += (bin_count / total_samples) * abs(bin_acc - bin_conf)
    return float(ece)


def load_model_and_tokenizer(ckpt_rel: str, tok_rel: str, device: torch.device):
    ckpt_path = os.path.join(PROJECT_ROOT, ckpt_rel)
    tok_path = os.path.join(PROJECT_ROOT, tok_rel)

    tokenizer = EdithTokenizer.load(tok_path)
    checkpoint = torch.load(ckpt_path, map_location=device)

    config = checkpoint["config"]
    maps = checkpoint["metadata_maps"]

    intents = {v: k for k, v in maps["intents"].items()}
    slots = {v: k for k, v in maps["slots"].items()}
    actions = {v: k for k, v in maps["actions"].items()}

    ckpt_vocab = checkpoint.get("vocab_size", tokenizer.vocab_size)
    model = EdithNeuralNetwork(
        vocab_size=ckpt_vocab,
        num_intents=len(intents),
        num_slots=len(slots),
        num_actions=len(actions),
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        n_layers=config["model"]["n_layers"],
        d_ff=config["model"]["d_ff"],
        max_seq_len=config["model"]["max_seq_len"],
        dropout=0.0
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    return model, tokenizer, maps, config


def evaluate_dataset(
    model: EdithNeuralNetwork,
    tokenizer: EdithTokenizer,
    maps: Dict[str, Any],
    dataset_path: str,
    device: torch.device,
    max_seq_len: int = 64
) -> Dict[str, Any]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    slot_map = maps["slots"]
    id2slot = {v: k for k, v in slot_map.items()}
    intent_map = maps["intents"]
    id2intent = {v: k for k, v in intent_map.items()}
    action_map = maps["actions"]
    id2action = {v: k for k, v in action_map.items()}

    num_intents = len(id2intent)
    num_actions = len(id2action)

    intent_tp = [0] * num_intents
    intent_fp = [0] * num_intents
    intent_fn = [0] * num_intents

    action_tp = [0] * num_actions
    action_fp = [0] * num_actions
    action_fn = [0] * num_actions

    intent_cm = np.zeros((num_intents, num_intents), dtype=int)
    action_cm = np.zeros((num_actions, num_actions), dtype=int)

    correct_intent = 0
    correct_action = 0
    correct_comp = 0
    exact_seq_matches = 0

    total_token_slots = 0
    correct_token_slots = 0

    span_tp = 0
    span_fp = 0
    span_fn = 0

    ood_correct = 0
    ood_total = 0

    conf_list = []
    seq_correct_flags = []
    correct_confs = []
    incorrect_confs = []

    raw_records = []

    with torch.no_grad():
        for s in samples:
            text = s["text"]
            true_intent_id = s.get("intent_id", intent_map.get(s.get("intent"), 0))
            true_action_id = s.get("action_id", action_map.get(s.get("action"), 0))
            true_conf = float(s.get("confidence", 1.0))
            true_comp = int(s.get("complexity", 0))

            input_ids, attention_mask = tokenizer.encode(
                text, max_len=max_seq_len, add_special_tokens=True, pad_to_max=True
            )

            t_in = torch.tensor([input_ids], dtype=torch.long, device=device)
            t_mask = torch.tensor([attention_mask], dtype=torch.long, device=device)

            out = model(t_in, attention_mask=t_mask)

            pred_intent_id = int(out["intent_logits"].argmax(dim=-1).item())
            pred_action_id = int(out["action_logits"].argmax(dim=-1).item())
            pred_comp_id = int(out["complexity_logits"].argmax(dim=-1).item())
            pred_conf = float(out["confidence"].item())
            pred_slot_ids = out["slot_logits"].argmax(dim=-1)[0].tolist()

            # Intent tracking
            is_intent_correct = (pred_intent_id == true_intent_id)
            if is_intent_correct:
                correct_intent += 1
                intent_tp[true_intent_id] += 1
            else:
                intent_fp[pred_intent_id] += 1
                intent_fn[true_intent_id] += 1
            intent_cm[true_intent_id, pred_intent_id] += 1

            # Action tracking
            is_action_correct = (pred_action_id == true_action_id)
            if is_action_correct:
                correct_action += 1
                action_tp[true_action_id] += 1
            else:
                action_fp[pred_action_id] += 1
                action_fn[true_action_id] += 1
            action_cm[true_action_id, pred_action_id] += 1

            # Complexity tracking
            if pred_comp_id == true_comp:
                correct_comp += 1

            # Slot alignment: target slot tags
            raw_slots = s.get("slot_tags", [])
            usable_len = min(len(raw_slots), max_seq_len - 2)
            gold_slot_seq = [slot_map.get(tag, 0) for tag in raw_slots[:usable_len]]
            pred_slot_seq = pred_slot_ids[1: 1 + usable_len]

            is_slot_exact = (gold_slot_seq == pred_slot_seq)
            for g, p in zip(gold_slot_seq, pred_slot_seq):
                if g == p:
                    correct_token_slots += 1
                total_token_slots += 1

            # Spans
            gold_spans = extract_spans(gold_slot_seq, id2slot)
            pred_spans = extract_spans(pred_slot_seq, id2slot)

            tp_s = len(gold_spans & pred_spans)
            fp_s = len(pred_spans - gold_spans)
            fn_s = len(gold_spans - pred_spans)

            span_tp += tp_s
            span_fp += fp_s
            span_fn += fn_s

            # Exact sequence match: intent & action & slots
            is_exact_seq = is_intent_correct and is_action_correct and is_slot_exact
            if is_exact_seq:
                exact_seq_matches += 1

            # OOD
            is_true_ood = (true_conf < 0.5) or (s.get("intent") == "unsupported_ood")
            is_pred_ood = (pred_conf < 0.5) or (pred_intent_id == intent_map.get("unsupported_ood", -1))
            if is_pred_ood == is_true_ood:
                ood_correct += 1
            ood_total += 1

            # Calibration
            conf_list.append(pred_conf)
            seq_correct_flags.append(is_intent_correct)
            if is_intent_correct:
                correct_confs.append(pred_conf)
            else:
                incorrect_confs.append(pred_conf)

            # Record
            raw_records.append({
                "id": s.get("id"),
                "text": text,
                "true_intent": id2intent.get(true_intent_id),
                "pred_intent": id2intent.get(pred_intent_id),
                "true_action": id2action.get(true_action_id),
                "pred_action": id2action.get(pred_action_id),
                "true_conf": true_conf,
                "pred_conf": round(pred_conf, 4),
                "gold_spans": list(gold_spans),
                "pred_spans": list(pred_spans),
                "is_intent_correct": is_intent_correct,
                "is_action_correct": is_action_correct,
                "is_exact_seq": is_exact_seq
            })

    total_samples = len(samples)

    # Compute macro-F1 for Intent
    intent_f1_list = []
    for i in range(num_intents):
        tp = intent_tp[i]
        fp = intent_fp[i]
        fn = intent_fn[i]
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-8, prec + rec)
        # only average across classes that appear in targets
        if (tp + fn) > 0:
            intent_f1_list.append(f1)
    intent_macro_f1 = float(np.mean(intent_f1_list)) if intent_f1_list else 0.0

    # Compute macro-F1 for Action
    action_f1_list = []
    for i in range(num_actions):
        tp = action_tp[i]
        fp = action_fp[i]
        fn = action_fn[i]
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-8, prec + rec)
        if (tp + fn) > 0:
            action_f1_list.append(f1)
    action_macro_f1 = float(np.mean(action_f1_list)) if action_f1_list else 0.0

    # Spans
    span_precision = span_tp / max(1, span_tp + span_fp)
    span_recall = span_tp / max(1, span_tp + span_fn)
    span_f1 = 2 * span_precision * span_recall / max(1e-8, span_precision + span_recall)

    # Calibration
    ece = compute_ece(conf_list, seq_correct_flags, num_bins=10)

    results = {
        "dataset": os.path.basename(dataset_path),
        "total_samples": total_samples,
        "intent_accuracy": round(correct_intent / max(1, total_samples) * 100.0, 2),
        "intent_macro_f1": round(intent_macro_f1 * 100.0, 2),
        "action_accuracy": round(correct_action / max(1, total_samples) * 100.0, 2),
        "action_macro_f1": round(action_macro_f1 * 100.0, 2),
        "token_slot_accuracy": round(correct_token_slots / max(1, total_token_slots) * 100.0, 2),
        "span_precision": round(span_precision * 100.0, 2),
        "span_recall": round(span_recall * 100.0, 2),
        "span_f1": round(span_f1 * 100.0, 2),
        "exact_sequence_match": round(exact_seq_matches / max(1, total_samples) * 100.0, 2),
        "complexity_accuracy": round(correct_comp / max(1, total_samples) * 100.0, 2),
        "ood_detection_accuracy": round(ood_correct / max(1, ood_total) * 100.0, 2),
        "ece": round(ece * 100.0, 2),
        "mean_conf_correct": round(float(np.mean(correct_confs)) if correct_confs else 0.0, 4),
        "mean_conf_incorrect": round(float(np.mean(incorrect_confs)) if incorrect_confs else 0.0, 4),
    }

    return results, raw_records, intent_cm, action_cm


def main():
    device = torch.device("cpu")
    print(f"Evaluating models on device: {device}")

    models_info = [
        {
            "name": "V10_Baseline",
            "ckpt": "models/checkpoints/edith_ann_v10_baseline.pt",
            "tok": "models/checkpoints/tokenizer_v10_baseline.json"
        },
        {
            "name": "V11_Candidate",
            "ckpt": "models/checkpoints/edith_ann_v11_candidate.pt",
            "tok": "models/checkpoints/tokenizer_v11_candidate.json"
        }
    ]

    datasets = [
        "data/test_data_clean.json",
        "data/test_data.json",
        "data/v11_challenge_set.json"
    ]

    all_metrics = {}
    all_confusion_matrices = {}

    for minfo in models_info:
        mname = minfo["name"]
        print(f"\n==========================================")
        print(f" LOADING {mname} ({minfo['ckpt']})")
        print(f"==========================================")
        model, tokenizer, maps, config = load_model_and_tokenizer(minfo["ckpt"], minfo["tok"], device)

        all_metrics[mname] = {}
        all_confusion_matrices[mname] = {}

        for dpath in datasets:
            dname = os.path.basename(dpath)
            print(f"\n--- Evaluating {mname} on {dname} ---")
            res, raw_recs, icm, acm = evaluate_dataset(model, tokenizer, maps, dpath, device)
            all_metrics[mname][dname] = res
            all_confusion_matrices[mname][dname] = {
                "intent_cm": icm.tolist(),
                "action_cm": acm.tolist(),
                "intents": maps["intents"],
                "actions": maps["actions"]
            }

            for k, v in res.items():
                print(f"  {k:25s}: {v}")

            # Save raw predictions for challenge set and clean test set
            save_name = f"scratch/v11_1_raw_predictions_{mname}_{dname}"
            with open(save_name, "w", encoding="utf-8") as f:
                json.dump(raw_recs, f, indent=2)
            print(f"  Saved raw predictions to {save_name}")

    # Save summary metrics
    with open("scratch/v11_1_recomputed_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print("\nSaved scratch/v11_1_recomputed_metrics.json")

    # Save confusion matrices
    with open("scratch/v11_1_confusion_matrices.json", "w", encoding="utf-8") as f:
        json.dump(all_confusion_matrices, f, indent=2)
    print("Saved scratch/v11_1_confusion_matrices.json")


if __name__ == "__main__":
    main()
