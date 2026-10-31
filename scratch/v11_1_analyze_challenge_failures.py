import json
import os
import re
from collections import Counter, defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

raw_preds_path = os.path.join(PROJECT_ROOT, "scratch/v11_1_raw_predictions_V11_Candidate_v11_challenge_set.json")
challenge_path = os.path.join(PROJECT_ROOT, "data/v11_challenge_set.json")
tok_path = os.path.join(PROJECT_ROOT, "models/checkpoints/tokenizer_v11_candidate.json")

with open(raw_preds_path, "r", encoding="utf-8") as f:
    preds = json.load(f)

with open(challenge_path, "r", encoding="utf-8") as f:
    challenges = json.load(f)

with open(tok_path, "r", encoding="utf-8") as f:
    tok_data = json.load(f)
vocab = tok_data.get("vocab", {})

print(f"Total Challenge Samples: {len(preds)}")
print(f"Total Intent Failures: {sum(1 for p in preds if not p['is_intent_correct'])}")
print(f"Total Action Failures: {sum(1 for p in preds if not p['is_action_correct'])}")
print(f"Total Exact Sequence Failures: {sum(1 for p in preds if not p['is_exact_seq'])}")

# Group failures into root causes
failing_examples = []
category_counts = Counter()
failure_groups = defaultdict(list)

for p, c in zip(preds, challenges):
    text = p["text"]
    text_lower = text.lower()
    true_intent = p["true_intent"]
    pred_intent = p["pred_intent"]
    true_action = p["true_action"]
    pred_action = p["pred_action"]
    is_intent_ok = p["is_intent_correct"]
    is_action_ok = p["is_action_correct"]

    if is_intent_ok and is_action_ok:
        continue

    # Diagnostic checks
    tokens = [t.lower() for t in re.findall(r"\w+|[^\w\s]", text)]
    unk_tokens = [t for t in tokens if t not in vocab]
    unk_ratio = len(unk_tokens) / max(1, len(tokens))

    # Identify primary root cause
    cause = "other"
    notes = ""

    # 1. Negation or Conditional failure
    if any(neg in text_lower for neg in ["don't", "do not", "never", "under no circumstances", "stop", "prevent", "should you", "unless", "if "]):
        if true_action in ["noop", "request_clarification"] and pred_action != true_action:
            cause = "Negation or conditional failure"
            notes = f"Model executed {pred_action} despite negation/conditional guard (expected {true_action})"
        elif "if " in text_lower or "unless" in text_lower or "suppose" in text_lower:
            cause = "Negation or conditional failure"
            notes = f"Conditional/hypothetical clause misclassified as direct operational command"

    # 2. Hinglish parsing failure
    elif any(h in text_lower for h in ["bhai", "yaar", "kholo", "chalao", "band karo", "rok do", "khatam", "sunao", "dheeme", "tez", "kar do", "karo", "pe"]):
        cause = "Hinglish parsing failure"
        notes = f"Mixed Hindi-English sentence misclassified"

    # 3. Missing vocabulary (>25% UNK tokens)
    elif unk_ratio >= 0.25:
        cause = "Missing vocabulary"
        notes = f"High UNK rate ({unk_ratio*100:.1f}%): unknown words {unk_tokens}"

    # 4. Incorrect intent taxonomy or annotation ambiguity
    elif true_intent == "unsupported_ood" and pred_intent != "unsupported_ood":
        cause = "Unsupported request"
        notes = f"Out-of-domain query classified as domain '{pred_intent}'"
    elif true_intent in ["chitchat", "system_query"] and pred_intent in ["chitchat", "system_query"]:
        cause = "Incorrect intent taxonomy"
        notes = f"Subtle boundary between conversational query and system query ({true_intent} vs {pred_intent})"

    # 5. Incorrect action-policy mapping
    elif is_intent_ok and not is_action_ok:
        cause = "Incorrect action-policy mapping"
        notes = f"Intent was correct ({true_intent}), but action mapped to {pred_action} instead of {true_action}"

    # 6. Entity extraction failure (for sequence mismatch)
    elif p["gold_spans"] != p["pred_spans"]:
        cause = "Entity extraction failure"
        notes = f"Entity mismatch: gold={p['gold_spans']} vs pred={p['pred_spans']}"

    # 7. Runtime preprocessing mismatch
    elif any(c in text for c in ["'", '"', ":", ";", "/", "\\", "@"]):
        cause = "Runtime preprocessing mismatch"
        notes = f"Punctuation or quoting pattern shifted token boundaries"

    else:
        cause = "Context failure"
        notes = f"Implicit reference or discourse marker requiring conversational history"

    category_counts[cause] += 1
    failure_groups[cause].append({
        "id": p["id"],
        "text": text,
        "true_intent": true_intent,
        "pred_intent": pred_intent,
        "true_action": true_action,
        "pred_action": pred_action,
        "conf": p["pred_conf"],
        "unk_tokens": unk_tokens,
        "notes": notes
    })

print("\n=== FAILURE ROOT CAUSE BREAKDOWN ===")
for cat, count in category_counts.most_common():
    print(f"  {cat:35s}: {count:3d} ({count/len(preds)*100:.1f}%)")

# Save detailed failure analysis to JSON
out_path = os.path.join(PROJECT_ROOT, "scratch/v11_1_challenge_failure_analysis.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({
        "summary": dict(category_counts),
        "failures_by_category": failure_groups
    }, f, indent=2)

print(f"\nSaved detailed analysis to {out_path}")
