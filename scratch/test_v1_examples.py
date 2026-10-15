import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
from engine.edith_brain import EdithBrain

brain = EdithBrain(enable_execution=False)

prompts = [
    "Could you bring up my coding environment?",
    "Let's get back to the project I was working on.",
    "Find the most recent version of my resume.",
    "Put on something relaxing while I work.",
    "Open my project, run it, and check whether it starts properly.",
    "That isn't the one I meant; find the other version.",
    "Find my latest EDITH project, open it in VS Code, run the tests, and tell me whether they pass.",
    "open notepad and write hello world",
    "search for cats and dogs"
]

print("=== BASELINE EVALUATION ON TARGET PROMPTS ===")
for p in prompts:
    res = brain.process(p, execute=False)
    print(f"\nPrompt: '{p}'")
    print(f"  Intent: {res.get('intent')} ({res.get('intent_confidence')})")
    print(f"  Action: {res.get('action')} ({res.get('action_confidence')})")
    print(f"  Entities: {res.get('entities')}")
    print(f"  Operational Conf: {res.get('operational_confidence')}")
    print(f"  Directive Message: {res.get('directive', {}).get('message')}")
    print(f"  Directive Handler: {res.get('directive', {}).get('handler')}")
    print(f"  Directive Params: {res.get('directive', {}).get('parameters')}")
    if res.get('plan'):
        print(f"  Plan: {res.get('plan')}")
