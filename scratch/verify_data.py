import json

with open('data/train_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

mod = [d for d in data if any('MODIFIER' in t for t in d.get('slot_tags', []))]
print("Training samples with MODIFIER slot:", len(mod), "/", len(data))
for m in mod[:5]:
    print("  Text:", m["text"])
    print("  Slots:", m["slot_tags"])
    print()

# Also check music_playback distribution
music = [d for d in data if d.get('intent') == 'music_playback']
print("Total music_playback:", len(music))
