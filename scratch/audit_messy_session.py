import os
import sys
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.edith_brain import EdithBrain
from engine.output_contract import LeakLinter, speak

def audit_messy():
    brain = EdithBrain(enable_execution=False)
    
    messy_prompts = [
        # 1. Hinglish venting
        "bhai aaj bohot bekar din tha, boss ne bina baat ke daant diya",
        # 2. Hinglish banter
        "tu mere se zyada smart hai kya?",
        # 3. Vague acoustic slip
        "kuch toh open karo",
        # 4. Hinglish media request
        "spotify pe koi achha sa gaana chalao",
        # 5. Trivia out-of-domain
        "World Cup 2011 me Dhoni ne final me kitne run banaye the?",
        # 6. Deep distress / burnout
        "I feel like giving up on coding, nothing is working and I feel worthless",
        # 7. Ambiguous plea
        "kar do na please",
        # 8. Sarcastic tease
        "are you even an AI or just a bunch of if-else statements?",
        # 9. Complex multi-step conversational slip
        "actually pehle chrome kholo fir sochta hu kya search karna hai",
        # 10. Code advice request
        "how do I fix a segmentation fault in C++?",
    ]

    results = []
    print("\n--- AUDITING 10 MESSY CONVERSATIONAL PROMPTS ---")
    for q in messy_prompts:
        res = brain.process(q, execute=False)
        directive = res.get("directive", {})
        msg = directive.get("message", "")
        clean_spoken = speak(msg, voice_mode=False, debug_mode=False)
        is_clean, violations = LeakLinter.lint(clean_spoken)
        
        print(f"\n[User] > {q}")
        print(f"[E.D.I.T.H.] {clean_spoken}")
        print(f"  Action: {res.get('action')} | Intent: {res.get('intent')} | Executable: {directive.get('can_execute')}")
        if not is_clean:
            print(f"  ❌ VIOLATIONS: {violations}")

        results.append({
            "query": q,
            "spoken": clean_spoken,
            "action": res.get("action"),
            "intent": res.get("intent"),
            "can_execute": directive.get("can_execute", False),
            "is_clean": is_clean,
            "violations": violations
        })

    with open("scratch/messy_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    audit_messy()

