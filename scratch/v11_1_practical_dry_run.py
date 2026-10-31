import os
import sys
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from engine.edith_brain import EdithBrain
from engine.inference import EdithInferenceEngine

queries = [
    "Don't open Spotify.",
    "I'm thinking about opening Spotify.",
    "Can you explain what Spotify does?",
    "I need some music.",
    "Bhai Chrome kholo jaldi.",
    "Make the volume a bit low.",
    "What would happen if I deleted this folder?",
    "Find my latest presentation.",
    "Show me the second one.",
    "No, I meant the other file.",
    "I'm stressed about my exam.",
    "Write this down: submit my assignment tomorrow.",
    "What's going on on my screen?",
    "What am I doing right now?",
    "Stop that operation."
]


def run_dry_run(model_name: str, ckpt_path: str, tok_path: str):
    print(f"\n=======================================================")
    print(f" PRACTICAL DRY RUN: {model_name}")
    print(f" Checkpoint: {ckpt_path}")
    print(f" Tokenizer:  {tok_path}")
    print(f"=======================================================")

    # Initialize brain with execution disabled
    brain = EdithBrain(enable_execution=False)
    # Inject specified checkpoint into engine
    brain.engine = EdithInferenceEngine(ckpt_path, tok_path)

    results = []

    for idx, q in enumerate(queries, 1):
        # Before query 9, simulate search candidates found by query 8 if candidates empty
        if idx == 9 and not getattr(brain.context, "search_candidates", []):
            brain.context.search_candidates = [
                {"name": "presentation_draft.pptx", "path": "c:/docs/presentation_draft.pptx"},
                {"name": "presentation_final.pptx", "path": "c:/docs/presentation_final.pptx"},
                {"name": "presentation_backup.pptx", "path": "c:/docs/presentation_backup.pptx"},
            ]
        # Before query 15, simulate an active task running
        if idx == 15 and not getattr(brain.context, "active_task", None):
            brain.context.active_task = "file_download"

        # Raw engine prediction
        pred = brain.engine.predict(q)

        # Brain pipeline processing
        res = brain.process(q, execute=False)

        directive = res.get("directive", {})
        handler = directive.get("handler")
        can_exec = directive.get("can_execute", False)
        needs_conf = directive.get("needs_confirmation", False)

        # An actual action is proposed if can_exec is True and handler is not dialogue/clarification/none
        action_proposed = bool(can_exec and handler not in [None, "dialogue", "clarification", "none"])

        ctx_used = {
            "current_topic": getattr(brain.context, "current_topic", None),
            "candidates_count": len(getattr(brain.context, "search_candidates", [])),
            "active_task": getattr(brain.context, "active_task", None)
        }

        entry = {
            "index": idx,
            "query": q,
            "engine_intent": pred.get("intent"),
            "engine_entities": pred.get("entities"),
            "engine_action": pred.get("action"),
            "engine_confidence": pred.get("operational_confidence"),
            "pipeline_intent": res.get("intent"),
            "pipeline_entities": res.get("entities"),
            "pipeline_policy": res.get("action"),
            "pipeline_confidence": res.get("operational_confidence"),
            "context_used": ctx_used,
            "selected_directive": {
                "handler": handler,
                "can_execute": can_exec,
                "needs_confirmation": needs_conf,
                "message": directive.get("message", res.get("message", ""))
            },
            "action_proposed": action_proposed
        }
        results.append(entry)

        print(f"\n[{idx:02d}] \"{q}\"")
        print(f"     Engine: Intent={entry['engine_intent']} (conf {entry['engine_confidence']:.2f}) | Action={entry['engine_action']} | Entities={entry['engine_entities']}")
        print(f"     Pipeline: Intent={entry['pipeline_intent']} | Policy={entry['pipeline_policy']} | Conf={entry['pipeline_confidence']}")
        print(f"     Directive: Handler={handler} | CanExec={can_exec} | NeedsConf={needs_conf}")
        print(f"     Action Proposed: {action_proposed}")
        print(f"     Response Msg: {directive.get('message', res.get('message', ''))[:100]}...")

    return results


def main():
    v10_res = run_dry_run(
        "V10_Baseline",
        "models/checkpoints/edith_ann_v10_baseline.pt",
        "models/checkpoints/tokenizer_v10_baseline.json"
    )

    v11_res = run_dry_run(
        "V11_Candidate",
        "models/checkpoints/edith_ann_v11_candidate.pt",
        "models/checkpoints/tokenizer_v11_candidate.json"
    )

    out_file = os.path.join(PROJECT_ROOT, "scratch/v11_1_practical_dry_run_comparison.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "V10_Baseline": v10_res,
            "V11_Candidate": v11_res
        }, f, indent=2)

    print(f"\nSaved practical dry-run results to {out_file}")


if __name__ == "__main__":
    main()
