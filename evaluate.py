"""E.D.I.T.H. Neural Network Comprehensive Task Benchmark Suite."""

import sys
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from engine.edith_brain import EdithBrain


def run_benchmark():
    exec_mode = "--exec" in sys.argv
    print("==================================================================")
    print("        E.D.I.T.H. COMPREHENSIVE NEURAL TASK BENCHMARK            ")
    print(f"       Mode: {'LIVE OS EXECUTION' if exec_mode else 'DRY RUN / PLAN VALIDATION (Pass --exec for live)'}")
    print("==================================================================")

    brain = EdithBrain(enable_execution=exec_mode)

    test_commands = [
        # Multi-Step Operations
        "open notepad and type hello world",
        "open chrome and search for artificial intelligence",
        "take a screenshot and open downloads",
        "create file test.txt on desktop and open it",
        "open spotify, play starboy and increase volume",
        "open calculator then open notepad",
        "what is 50 * 20 and what time is it",
        "close chrome and lock workstation",
        "empty recycle bin and minimize all windows",
        "open chrome go on gmail and draft an apology email",
        "open mail and draft a mail to people",
        "open notepad and draft a resignation letter",
        "go to youtube and search for python tutorial",
        "open edge go to github",
        
        # Single-Step & Precision Tests
        "draft a mail to John saying I will be late for the meeting",
        "play blinding lights on youtube",
        "play slowed and reverb believer",
        "what is 20 percent of 1500",
        "what is 50 + 25",
        "calculate (45 * 12) + 100 / 2",
        "what is the square root of 144",
        "what time is it",
        "what is the current cpu usage",
        "search for cats and dogs",
        "create a file named notes.txt on desktop",
        "ping google.com",
        "hello edith who are you",
        "fly a spaceship to mars",
    ]

    print(f"\nEvaluating {len(test_commands)} diverse task instructions through EDITH Brain:\n")

    for idx, cmd in enumerate(test_commands, 1):
        result = brain.process(cmd, execute=exec_mode)
        directive = result["directive"]
        execution = result.get("execution", {})
        steps = result.get("steps", [])

        print(f"[{idx:02d}] Input: \"{cmd}\"")
        if len(steps) > 1:
            print(f"      |-- Multi-Step : {len(steps)} steps detected -> {steps}")
        print(f"      |-- Intent     : {result['intent']} ({result['intent_confidence']:.1%})")
        print(f"      |-- Action     : {result['action']} ({result['action_confidence']:.1%})")
        if result.get("entities"):
            print(f"      |-- Entities   : {result['entities']}")
        print(f"      |-- Status     : {result['status']} (Operational: {result['operational_confidence']:.1%})")
        print(f"      |-- Brain Msg  : {directive['message']}")
        if execution:
            print(f"      \\-- Executed   : [{execution.get('status', 'ok')}] {execution.get('message')}")
        else:
            print(f"      \\-- Executed   : [dry run / planned]")
        print("-" * 70)


if __name__ == "__main__":
    run_benchmark()
