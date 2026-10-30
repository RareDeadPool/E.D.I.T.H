import json

with open("data/test_data.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

corrupted_count = 0
for idx, s in enumerate(test_data):
    tokens = s["tokens"]
    slots = s["slot_tags"]
    text = s["text"]
    for t, tag in zip(tokens, slots):
        if t.endswith(".") and tag == "O":
            clean_t = t.rstrip(".")
            if clean_t in [
                "spotify", "chrome", "notepad", "firefox", "edge", "discord", "slack", "calc",
                "resume.pdf", "notes.txt", "budget.xlsx", "report.pdf", "data.csv", "test.py",
                "starboy", "blinding", "believer", "unstoppable", "flowers", "stay"
            ]:
                corrupted_count += 1
                if corrupted_count <= 10:
                    print(f"Sample {idx}: \"{text}\" -> token '{t}' is labeled '{tag}'")

print(f"\nTotal corrupted test entity tokens labeled as 'O' due to trailing period: {corrupted_count}")
