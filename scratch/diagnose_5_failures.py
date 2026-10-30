import sys, os
sys.path.insert(0, os.path.abspath("."))

from engine.edith_brain import EdithBrain
from engine.inference import EdithInferenceEngine

brain = EdithBrain(enable_execution=False)
engine = brain.engine

queries = [
    ("Hinglish", "bhai chrome kholo jaldi"),
    ("Indian-English", "can you please do one thing, open notepad"),
    ("Speech Error", "opn notepd please"),
    ("Ambiguity", "open"),
    ("File Discovery", "find all python files in my project folder"),
]

for label, q in queries:
    p = engine.predict(q)
    b = brain.process(q, execute=False)
    print(f"=== {label}: \"{q}\" ===")
    print(f"  Engine Predict: Intent={p.get('intent')} (conf: {p.get('intent_confidence'):.2f}), Action={p.get('action')}, Conf={p.get('operational_confidence'):.2f}, Entities={p.get('entities')}")
    print(f"  Brain Process:  Intent={b.get('intent')}, Action={b.get('action')}, Status={b.get('status')}, Directive={b.get('directive')}")
