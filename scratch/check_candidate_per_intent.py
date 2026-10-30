import json

with open("models/checkpoints/v11_training_metrics.json", "r", encoding="utf-8") as f:
    report = json.load(f)

with open("data/metadata_maps.json", "r", encoding="utf-8") as f:
    maps = json.load(f)

id2intent = {v: k for k, v in maps["intents"].items()}

clean_res = report["eval_results"]["clean_test"]
print("Per-intent F1 on clean_test for Candidate (Epoch 2):")
for intent_id_str, stat in clean_res["per_intent"].items():
    iid = int(intent_id_str)
    name = id2intent.get(iid, f"intent_{iid}")
    print(f"  {name:22s} (ID {iid:2d}) | Prec: {stat['precision']:5.1f}% | Rec: {stat['recall']:5.1f}% | F1: {stat['f1']:5.1f}% | Support: {stat['support']}")
