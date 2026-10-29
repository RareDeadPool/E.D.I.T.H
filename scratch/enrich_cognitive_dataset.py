"""Enrich data/cognitive_scenarios_dataset.json and update engine/cognitive_scenarios_engine.py with all 150 scenarios."""

import os
import sys
import json
import re

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "cognitive_scenarios_dataset.json")
ENGINE_PATH = os.path.join(PROJECT_ROOT, "engine", "cognitive_scenarios_engine.py")

SCRATCH_DIR = os.path.dirname(__file__)
if SCRATCH_DIR not in sys.path:
    sys.path.insert(0, SCRATCH_DIR)

from expand_cognitive_scenarios import NEW_SCENARIOS

def enrich_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenarios = data.get("scenarios", [])
    scen_by_id = {s["id"]: s for s in scenarios}

    for ns in NEW_SCENARIOS:
        s_id = ns["id"]
        if s_id in scen_by_id:
            scen_by_id[s_id]["patterns"] = ns["patterns"]
            scen_by_id[s_id]["response_template"] = ns["response"]
            scen_by_id[s_id]["suggested_action"] = ns.get("suggested_action")
        else:
            entry = {
                "id": ns["id"],
                "domain": ns["domain"],
                "prompt": ns["prompt"],
                "expected_topic": ns["expected_topic"],
                "empathy_keywords": ns["empathy_keywords"],
                "patterns": ns["patterns"],
                "response_template": ns["response"],
                "suggested_action": ns.get("suggested_action")
            }
            scenarios.append(entry)

    data["total_scenarios"] = len(scenarios)
    data["scenarios"] = scenarios

    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Enriched dataset written with {len(scenarios)} total scenarios.")

if __name__ == "__main__":
    enrich_dataset()
