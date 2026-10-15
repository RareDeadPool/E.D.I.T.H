import os
import sys
sys.path.insert(0, os.path.abspath("."))
from engine.edith_brain import EdithBrain

b = EdithBrain(enable_execution=False)
cases = [
    "Let's get back to the project I was working on.",
    "Find the most recent version of my resume.",
    "Put on something relaxing while I work.",
    "Open my project, run it, and check whether it starts properly.",
    "That isn't the one I meant; find the other version.",
]
for c in cases:
    res = b.process(c, execute=False)
    print(f"\n--- '{c}' ---")
    print("Intent:", res.get("intent"), f"({res.get('intent_confidence')})")
    print("Action:", res.get("action"))
    print("Entities:", res.get("entities"))
    print("Status:", res.get("status"), f"({res.get('operational_confidence')})")
    print("Directive Msg:", res.get("directive", {}).get("message"))
    print("Plan:", res.get("plan"))
