"""
E.D.I.T.H. V15.3.1 Audit Script: Fluency Guard & Bridge Mode Audit
Evaluates FluencyGuard on an independent ground-truth labeled dataset of 100 sentences
(50 ground-truth valid, 50 ground-truth invalid) to objectively measure False Acceptance
and False Rejection rates without letting the guard judge its own outputs.
Exports results to data_pipeline/v15/audit_outputs/fluency_guard_audit_results.json
"""

import os
import sys
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.fluency_guard import FluencyGuard

# Independent Ground-Truth Labeled Test Set (100 Sentences)
LABELED_TEST_SET = [
    # -------------------------------------------------------------
    # 50 GROUND-TRUTH VALID EXAMPLES (Label: 1)
    # -------------------------------------------------------------
    # 10 Natural English
    {"id": "V_01", "text": "I am right here with you, let me know what you need.", "label": 1, "subtype": "natural_english"},
    {"id": "V_02", "text": "That is genuinely heartbreaking, but you are not alone.", "label": 1, "subtype": "natural_english"},
    {"id": "V_03", "text": "Good morning, ready to tackle today's codebase?", "label": 1, "subtype": "natural_english"},
    {"id": "V_04", "text": "I can certainly help you brainstorm a name for your project.", "label": 1, "subtype": "natural_english"},
    {"id": "V_05", "text": "Take a short break and drink some water.", "label": 1, "subtype": "natural_english"},
    {"id": "V_06", "text": "That sounds like a really productive afternoon.", "label": 1, "subtype": "natural_english"},
    {"id": "V_07", "text": "I'm keeping your workspace warm while you step away.", "label": 1, "subtype": "natural_english"},
    {"id": "V_08", "text": "We should look at the stack trace to find the root cause.", "label": 1, "subtype": "natural_english"},
    {"id": "V_09", "text": "It was great chatting with you today.", "label": 1, "subtype": "natural_english"},
    {"id": "V_10", "text": "The sunset looks beautiful through the window.", "label": 1, "subtype": "natural_english"},

    # 10 Technical Jargon
    {"id": "V_11", "text": "The neural network is trained using backpropagation and AdamW optimizer.", "label": 1, "subtype": "technical_jargon"},
    {"id": "V_12", "text": "We are allocating tensors in FP16 on the RTX 4050 GPU.", "label": 1, "subtype": "technical_jargon"},
    {"id": "V_13", "text": "RMSNorm stabilizes activations across the twelve transformer layers.", "label": 1, "subtype": "technical_jargon"},
    {"id": "V_14", "text": "The AST parser pass failed due to an unexpected indentation token.", "label": 1, "subtype": "technical_jargon"},
    {"id": "V_15", "text": "RoPE positional embeddings rotate queries and keys efficiently.", "label": 1, "subtype": "technical_jargon"},
    {"id": "V_16", "text": "The SQLite database maintains user memory provenance records.", "label": 1, "subtype": "technical_jargon"},
    {"id": "V_17", "text": "PyTorch DataLoader uses pinned memory for faster host-to-device transfers.", "label": 1, "subtype": "technical_jargon"},
    {"id": "V_18", "text": "The SwiGLU activation function provides higher representation density.", "label": 1, "subtype": "technical_jargon"},
    {"id": "V_19", "text": "The IPC pipe transmits JSON packets between processes.", "label": 1, "subtype": "technical_jargon"},
    {"id": "V_20", "text": "Grouped-query attention reduces KV cache memory consumption.", "label": 1, "subtype": "technical_jargon"},

    # 10 Casual Hinglish
    {"id": "V_21", "text": "Bhai tension mat le, sab theek ho jayega.", "label": 1, "subtype": "casual_hinglish"},
    {"id": "V_22", "text": "Ek accha joke sunao na yaar.", "label": 1, "subtype": "casual_hinglish"},
    {"id": "V_23", "text": "Aaj ka din kaafi exhausting aur thoda tiring tha.", "label": 1, "subtype": "casual_hinglish"},
    {"id": "V_24", "text": "Spotify chalao aur gaana suno.", "label": 1, "subtype": "casual_hinglish"},
    {"id": "V_25", "text": "Chalo milkar kaam karte hain.", "label": 1, "subtype": "casual_hinglish"},
    {"id": "V_26", "text": "Apna dhyan rakho aur pareshan mat ho.", "label": 1, "subtype": "casual_hinglish"},
    {"id": "V_27", "text": "Acha suno, pehle code test kar lo fir baat karenge.", "label": 1, "subtype": "casual_hinglish"},
    {"id": "V_28", "text": "Mujhe neend aa rahi hai abhi.", "label": 1, "subtype": "casual_hinglish"},
    {"id": "V_29", "text": "Bhai dost bohot dukhi hai, kya karu?", "label": 1, "subtype": "casual_hinglish"},
    {"id": "V_30", "text": "Thoda coffee peene chalte hain.", "label": 1, "subtype": "casual_hinglish"},

    # 10 Proper Names & Entities
    {"id": "V_31", "text": "Aditya is working on the E.D.I.T.H. workstation.", "label": 1, "subtype": "proper_names"},
    {"id": "V_32", "text": "My friend Harshan is learning Python.", "label": 1, "subtype": "proper_names"},
    {"id": "V_33", "text": "Open Chrome and search Wikipedia for transformers.", "label": 1, "subtype": "proper_names"},
    {"id": "V_34", "text": "Launch VS Code and check the git diff.", "label": 1, "subtype": "proper_names"},
    {"id": "V_35", "text": "Rahul is responsible for database migrations.", "label": 1, "subtype": "proper_names"},
    {"id": "V_36", "text": "Tony Stark left behind the Edith glasses.", "label": 1, "subtype": "proper_names"},
    {"id": "V_37", "text": "We are running Windows PowerShell for training scripts.", "label": 1, "subtype": "proper_names"},
    {"id": "V_38", "text": "Spotify is playing instrumental focus tracks.", "label": 1, "subtype": "proper_names"},
    {"id": "V_39", "text": "The GitHub repository was updated an hour ago.", "label": 1, "subtype": "proper_names"},
    {"id": "V_40", "text": "Bruno is resting on the rug in the living room.", "label": 1, "subtype": "proper_names"},

    # 10 Short Answers & Conversational Fragments
    {"id": "V_41", "text": "Yes, absolutely.", "label": 1, "subtype": "short_answers"},
    {"id": "V_42", "text": "Not at all.", "label": 1, "subtype": "short_answers"},
    {"id": "V_43", "text": "Paris.", "label": 1, "subtype": "short_answers"},
    {"id": "V_44", "text": "Eight.", "label": 1, "subtype": "short_answers"},
    {"id": "V_45", "text": "Works for me.", "label": 1, "subtype": "short_answers"},
    {"id": "V_46", "text": "Ready when you are.", "label": 1, "subtype": "short_answers"},
    {"id": "V_47", "text": "Sounds good!", "label": 1, "subtype": "short_answers"},
    {"id": "V_48", "text": "No, I don't think so.", "label": 1, "subtype": "short_answers"},
    {"id": "V_49", "text": "Got it.", "label": 1, "subtype": "short_answers"},
    {"id": "V_50", "text": "Done.", "label": 1, "subtype": "short_answers"},

    # -------------------------------------------------------------
    # 50 GROUND-TRUTH INVALID EXAMPLES (Label: 0)
    # -------------------------------------------------------------
    # 10 Word Salads & Broken Merges
    {"id": "I_01", "text": "hunol aioleplay naentation's w syks sa ke was", "label": 0, "subtype": "word_salad"},
    {"id": "I_02", "text": "I forbeteersedation li in O whatyyur reallyf", "label": 0, "subtype": "word_salad"},
    {"id": "I_03", "text": "zxkjh asdf qweopiu mnbvcx lkjhgf", "label": 0, "subtype": "word_salad"},
    {"id": "I_04", "text": "the quxation of flimflam sproggled the blart", "label": 0, "subtype": "word_salad"},
    {"id": "I_05", "text": "progra mm ing is a go od wa y to ma ke soft ware", "label": 0, "subtype": "word_salad"},
    {"id": "I_06", "text": "wo rld sa lad to ken frag men ts ev ery wh ere", "label": 0, "subtype": "word_salad"},
    {"id": "I_07", "text": "xyzt abcdef ghijkl mnopqr stuvwx", "label": 0, "subtype": "word_salad"},
    {"id": "I_08", "text": "re cur sion is a fun ction that ca lls it self", "label": 0, "subtype": "word_salad"},
    {"id": "I_09", "text": "inter val bpe sha tter sub word er ror", "label": 0, "subtype": "word_salad"},
    {"id": "I_10", "text": "qwertyuiop asdfghjkl zxcvbnm", "label": 0, "subtype": "word_salad"},

    # 10 Repetitive N-Gram Loops
    {"id": "I_11", "text": "the project is good the project is good the project is good the project is good", "label": 0, "subtype": "repetitive_ngrams"},
    {"id": "I_12", "text": "I think I should do I think I should do I think I should do", "label": 0, "subtype": "repetitive_ngrams"},
    {"id": "I_13", "text": "to achieve your goals to achieve your goals to achieve your goals", "label": 0, "subtype": "repetitive_ngrams"},
    {"id": "I_14", "text": "play in the park play in the park play in the park play in the park", "label": 0, "subtype": "repetitive_ngrams"},
    {"id": "I_15", "text": "working on it for working on it for working on it for", "label": 0, "subtype": "repetitive_ngrams"},
    {"id": "I_16", "text": "what you can do what you can do what you can do", "label": 0, "subtype": "repetitive_ngrams"},
    {"id": "I_17", "text": "because of that because of that because of that because of that", "label": 0, "subtype": "repetitive_ngrams"},
    {"id": "I_18", "text": "very hard to say very hard to say very hard to say", "label": 0, "subtype": "repetitive_ngrams"},
    {"id": "I_19", "text": "are cute are cute are cute are cute are cute", "label": 0, "subtype": "repetitive_ngrams"},
    {"id": "I_20", "text": "tell you about a joke tell you about a joke tell you about a joke", "label": 0, "subtype": "repetitive_ngrams"},

    # 10 Incomplete / Cut-off Sentences
    {"id": "I_21", "text": "I was thinking about the way that we can go to the and", "label": 0, "subtype": "incomplete_sentence"},
    {"id": "I_22", "text": "Because the laws of physics and mathematics are determined by the", "label": 0, "subtype": "incomplete_sentence"},
    {"id": "I_23", "text": "When you are trying to understand the nature of", "label": 0, "subtype": "incomplete_sentence"},
    {"id": "I_24", "text": "It is easy to see that the compiler has failed with", "label": 0, "subtype": "incomplete_sentence"},
    {"id": "I_25", "text": "For example, if a person or company has any reasons for", "label": 0, "subtype": "incomplete_sentence"},
    {"id": "I_26", "text": "A recursion is a category of short words or tense with each", "label": 0, "subtype": "incomplete_sentence"},
    {"id": "I_27", "text": "The answer to this question can be solved by", "label": 0, "subtype": "incomplete_sentence"},
    {"id": "I_28", "text": "If you want to know more about the situation in", "label": 0, "subtype": "incomplete_sentence"},
    {"id": "I_29", "text": "We should always verify that the model works before", "label": 0, "subtype": "incomplete_sentence"},
    {"id": "I_30", "text": "The main benefit of this approach is that", "label": 0, "subtype": "incomplete_sentence"},

    # 10 Raw Prompt Echoes
    {"id": "I_31", "text": "My friend has been unemployed for three years.", "query": "My friend has been unemployed for three years.", "label": 0, "subtype": "prompt_echo"},
    {"id": "I_32", "text": "My friend is very sad", "query": "My friend is very sad.", "label": 0, "subtype": "prompt_echo"},
    {"id": "I_33", "text": "I'm stressed about college", "query": "I'm stressed about college.", "label": 0, "subtype": "prompt_echo"},
    {"id": "I_34", "text": "I disagree with you", "query": "I disagree with you.", "label": 0, "subtype": "prompt_echo"},
    {"id": "I_35", "text": "Explain recursion to me", "query": "Explain recursion to me.", "label": 0, "subtype": "prompt_echo"},
    {"id": "I_36", "text": "Roast my coding style", "query": "Roast my coding style.", "label": 0, "subtype": "prompt_echo"},
    {"id": "I_37", "text": "Tell me a joke", "query": "Tell me a joke.", "label": 0, "subtype": "prompt_echo"},
    {"id": "I_38", "text": "What do you think I should do", "query": "What do you think I should do?", "label": 0, "subtype": "prompt_echo"},
    {"id": "I_39", "text": "Why does this code fail", "query": "Why does this code fail?", "label": 0, "subtype": "prompt_echo"},
    {"id": "I_40", "text": "Let's talk about something else", "query": "Let's talk about something else.", "label": 0, "subtype": "prompt_echo"},

    # 10 Empty, Whitespace, or Degenerate Strings
    {"id": "I_41", "text": "", "label": 0, "subtype": "empty_degenerate"},
    {"id": "I_42", "text": "   ", "label": 0, "subtype": "empty_degenerate"},
    {"id": "I_43", "text": "\n\n\t", "label": 0, "subtype": "empty_degenerate"},
    {"id": "I_44", "text": ".", "label": 0, "subtype": "empty_degenerate"},
    {"id": "I_45", "text": "...", "label": 0, "subtype": "empty_degenerate"},
    {"id": "I_46", "text": "?", "label": 0, "subtype": "empty_degenerate"},
    {"id": "I_47", "text": "!", "label": 0, "subtype": "empty_degenerate"},
    {"id": "I_48", "text": "S", "label": 0, "subtype": "empty_degenerate"},
    {"id": "I_49", "text": "2", "label": 0, "subtype": "empty_degenerate"},
    {"id": "I_50", "text": "- - -", "label": 0, "subtype": "empty_degenerate"},
]


def main():
    print("=" * 80)
    print("E.D.I.T.H. V15.3.1 INDEPENDENT FLUENCY GUARD AUDIT")
    print(f"Total Labeled Test Cases: {len(LABELED_TEST_SET)} (50 Valid, 50 Invalid)")
    print("=" * 80)

    tp, fp, tn, fn = 0, 0, 0, 0
    subtype_stats = {}

    audit_details = []

    for item in LABELED_TEST_SET:
        t_id = item["id"]
        text = item["text"]
        gt_label = item["label"]  # 1 = Valid, 0 = Invalid
        subtype = item["subtype"]
        query = item.get("query", "")

        is_fluent, diag = FluencyGuard.inspect(text, query=query)
        pred_label = 1 if is_fluent else 0

        if gt_label == 1 and pred_label == 1:
            tp += 1
            res = "TRUE_POSITIVE (Correctly Accepted)"
        elif gt_label == 0 and pred_label == 0:
            tn += 1
            res = "TRUE_NEGATIVE (Correctly Rejected)"
        elif gt_label == 0 and pred_label == 1:
            fp += 1
            res = "FALSE_POSITIVE (False Acceptance - BAD OUTPUT LEAKED)"
        elif gt_label == 1 and pred_label == 0:
            fn += 1
            res = "FALSE_NEGATIVE (False Rejection - VALID OUTPUT BLOCKED)"

        if subtype not in subtype_stats:
            subtype_stats[subtype] = {"total": 0, "correct": 0, "errors": 0}
        subtype_stats[subtype]["total"] += 1
        if pred_label == gt_label:
            subtype_stats[subtype]["correct"] += 1
        else:
            subtype_stats[subtype]["errors"] += 1

        audit_details.append({
            "id": t_id,
            "text": text,
            "ground_truth": gt_label,
            "guard_decision": pred_label,
            "classification": res,
            "subtype": subtype,
            "word_validity": diag.get("word_validity_ratio", 0.0),
            "trigram_logprob": diag.get("char_trigram_logprob", 0.0),
            "diagnostics": diag
        })

    accuracy = (tp + tn) / len(LABELED_TEST_SET)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-6, precision + recall)
    false_acceptance_rate = fp / max(1, tn + fp)
    false_rejection_rate = fn / max(1, tp + fn)

    print("\nCONFUSION MATRIX:")
    print(f"  True Positives  (Valid Accepted):   {tp:2d} / 50")
    print(f"  True Negatives  (Invalid Rejected): {tn:2d} / 50")
    print(f"  False Positives (False Acceptance): {fp:2d} / 50 ({false_acceptance_rate * 100:.1f}%)")
    print(f"  False Negatives (False Rejection):  {fn:2d} / 50 ({false_rejection_rate * 100:.1f}%)")
    print(f"\nMETRICS:")
    print(f"  Accuracy:  {accuracy * 100:.2f}%")
    print(f"  Precision: {precision * 100:.2f}%")
    print(f"  Recall:    {recall * 100:.2f}%")
    print(f"  F1 Score:  {f1:.4f}")

    print("\nSUBTYPE ACCURACY BREAKDOWN:")
    for sub, s in subtype_stats.items():
        acc = s["correct"] / s["total"] * 100
        print(f"  {sub:<22}: {s['correct']:2d}/{s['total']:2d} ({acc:5.1f}% correct)")

    out_file = os.path.join(PROJECT_ROOT, "data_pipeline", "v15", "audit_outputs", "fluency_guard_audit_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "metrics": {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "false_acceptance_rate": round(false_acceptance_rate, 4),
                "false_rejection_rate": round(false_rejection_rate, 4)
            },
            "subtype_stats": subtype_stats,
            "detailed_cases": audit_details
        }, f, indent=2)

    print(f"\nAudit ledger saved to: {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
