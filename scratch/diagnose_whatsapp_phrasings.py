import sys
import os
sys.path.insert(0, os.path.abspath("."))

from engine.edith_brain import EdithBrain

brain = EdithBrain(enable_execution=False)

test_phrases = [
    "Send Rahul a WhatsApp message saying I'll be 15 minutes late.",
    "Open WhatsApp and message Mom that I've reached college.",
    "Send a message to my project group saying the meeting starts at 4 PM.",
    "Draft a WhatsApp message to my professor asking about the assignment deadline.",
    "Send the message we just drafted.",
    "No, send it to Rahul instead.",
    "WhatsApp Rahul that I'm on my way.",
    "Tell Rahul on WhatsApp I'll be there in ten minutes.",
    "Can you ping Mom that I've reached?",
    "Message my professor about tomorrow's submission.",
    "Send a WhatsApp to the same person as before.",
    "Use the other Rahul.",
    "Make the message sound more polite.",
    "Send the revised version.",
    "Don't send it yet; just keep it as a draft.",
    "Actually, cancel that.",
    "Tell Rahul 'open the browser when you reach home'",
    "whenever u try to text while testing message on 8591211342"
]

print("=== DIAGNOSING NATURAL LANGUAGE UNDERSTANDING & EXTRACTION ===")
for p in test_phrases:
    res = brain.process(p)
    print(f"\nUser: \"{p}\"")
    print(f"  Status: {res.get('status')}")
    print(f"  Action/Intent: {res.get('action') or res.get('predicted_intent') or res.get('plan_id')}")
    print(f"  Directive/Handler: {res.get('directive')}")
    print(f"  Response: {res.get('response')[:120] if res.get('response') else 'None'}")
