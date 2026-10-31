import json

d = json.load(open('scratch/v11_1_challenge_failure_analysis.json', encoding='utf-8'))
for cat, items in d['failures_by_category'].items():
    print(f"\n=== {cat} ({len(items)} samples) ===")
    for x in items[:4]:
        print(f"  [{x['id']}] \"{x['text']}\"")
        print(f"      True: {x['true_intent']} / {x['true_action']} (conf {x.get('conf')})")
        print(f"      Pred: {x['pred_intent']} / {x['pred_action']}")
        print(f"      Notes: {x['notes']}")
        if x.get('unk_tokens'):
            print(f"      UNKs: {x['unk_tokens']}")
