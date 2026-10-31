import os
import sys
import json
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from engine.edith_brain import EdithBrain, normalize_lexical_typos
from engine.inference import EdithInferenceEngine

CHECKPOINTS = {
    "V10_production": {
        "ckpt": os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_ann_best.pt"),
        "tok": os.path.join(PROJECT_ROOT, "models", "checkpoints", "tokenizer.json"),
    },
    "V11_candidate": {
        "ckpt": os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_ann_v11_candidate.pt"),
        "tok": os.path.join(PROJECT_ROOT, "models", "checkpoints", "tokenizer_v11_candidate.json"),
    }
}

TEST_CASES = [
    {
        "case_id": "Case A1 (Actual Voice Transcript)",
        "input": "Come on, help me a joke.",
        "hypothesized_speech_intent": "Tell me a joke (possible acoustic misrecognition 'help' for 'tell')"
    },
    {
        "case_id": "Case A2 (Intended Speech Variant)",
        "input": "Come on, tell me a joke.",
        "hypothesized_speech_intent": "Tell me a joke"
    },
    {
        "case_id": "Case A3 (Canonical Joke Request)",
        "input": "Tell me a joke.",
        "hypothesized_speech_intent": "Tell me a joke"
    },
    {
        "case_id": "Case B (Friend Struggling for Jobs)",
        "input": "My friend is looking for jobs, he's trying from 3 years but he hasn't got one, he's very sad.",
        "hypothesized_speech_intent": "Conversational empathy / venting about friend's job struggles"
    },
    {
        "case_id": "Case C (Friend is Sad)",
        "input": "My friend is very sad.",
        "hypothesized_speech_intent": "Conversational statement / emotional support"
    },
    {
        "case_id": "Case D (Introducing a Friend)",
        "input": "I have a friend named Harshan.",
        "hypothesized_speech_intent": "Social introduction / personal sharing"
    }
]

def trace_execution(brain: EdithBrain, test_case: dict, model_name: str) -> dict:
    raw_query = test_case["input"]
    normalized_query = normalize_lexical_typos(raw_query.strip())
    
    # 1. Tokenization inspection
    tok = brain.engine.tokenizer
    tokens = tok.tokenize_raw(normalized_query)
    input_ids, att_mask = tok.encode(normalized_query, max_len=64, add_special_tokens=True, pad_to_max=False)
    decoded_tokens = [tok.inv_vocab.get(idx, f"<{idx}>") for idx in input_ids if idx != tok.PAD_ID]
    
    # 2. Standalone Neural Engine Prediction
    raw_pred = brain.engine.predict(normalized_query)
    
    # 3. Context state before processing
    ctx_before = {
        "current_topic": getattr(brain.context, "current_topic", None),
        "active_task": getattr(brain.context, "active_task", None),
        "last_command": getattr(brain.context, "last_command_executed", None),
        "history_len": len(getattr(brain.context, "history", [])) if hasattr(brain.context, "history") else 0
    }
    
    # 4. Response engine selection inspection
    # Check if dialogue engine claims it
    is_conv = bool(brain.dialogue_engine and brain.dialogue_engine.is_conversational_or_problem(normalized_query.lower()))
    
    # Check if cognitive scenarios claims it
    scen_matches = brain.dialogue_engine.cognitive_scenarios.matches_scenario(normalized_query) if (
        brain.dialogue_engine and hasattr(brain.dialogue_engine, "cognitive_scenarios")
    ) else False
    
    # 5. Full Brain Pipeline Process (execute=False to trace safely)
    pipeline_res = brain.process(raw_query, execute=False)
    
    directive = pipeline_res.get("directive", {})
    execution = pipeline_res.get("execution", {})
    handler = directive.get("handler", "")
    can_execute = directive.get("can_execute", False)
    
    # What would be dispatched?
    dispatch_decision = {
        "target_handler": handler,
        "can_execute": can_execute,
        "needs_confirmation": directive.get("needs_confirmation", False),
        "parameters": directive.get("parameters", {}),
        "simulated_execution": execution
    }
    
    final_message = directive.get("message", "") or execution.get("message", "")
    
    report = {
        "case_id": test_case["case_id"],
        "speech_transcription": raw_query,
        "normalized_input": normalized_query,
        "model_checkpoint_loaded": model_name,
        "checkpoint_path": brain.engine.checkpoint_path,
        "tokenization": {
            "tokens": tokens,
            "input_ids": [int(x) for x in input_ids if x != tok.PAD_ID],
            "decoded_tokens": decoded_tokens
        },
        "neural_raw_prediction": {
            "intent": raw_pred.get("intent"),
            "intent_confidence": round(float(raw_pred.get("intent_confidence", 0.0)), 4),
            "action": raw_pred.get("action"),
            "action_confidence": round(float(raw_pred.get("action_confidence", 0.0)), 4),
            "entities": raw_pred.get("entities", {}),
            "task_complexity": raw_pred.get("task_complexity"),
        },
        "pipeline_classification": {
            "intent": pipeline_res.get("intent"),
            "action": pipeline_res.get("action"),
            "status": pipeline_res.get("status"),
            "entities": pipeline_res.get("entities", {})
        },
        "context_state": ctx_before,
        "response_engine_selection": {
            "is_conv_or_dialogue": is_conv,
            "scenario_matched": scen_matches,
            "selected_handler": handler
        },
        "planner_output": directive,
        "action_dispatch_decision": dispatch_decision,
        "final_response": final_message
    }
    return report

def main():
    all_results = {}
    for model_name, paths in CHECKPOINTS.items():
        print(f"\n=======================================================")
        print(f" EVALUATING PIPELINE WITH: {model_name}")
        print(f" Checkpoint: {paths['ckpt']}")
        print(f" Tokenizer:  {paths['tok']}")
        print(f"=======================================================")
        
        brain = EdithBrain(
            checkpoint_path=paths["ckpt"],
            tokenizer_path=paths["tok"],
            enable_execution=False
        )
        
        case_results = []
        for case in TEST_CASES:
            print(f"  Testing {case['case_id']}: '{case['input']}'")
            res = trace_execution(brain, case, model_name)
            case_results.append(res)
            
            # Print quick summary
            print(f"    Raw Neural: Intent={res['neural_raw_prediction']['intent']} ({res['neural_raw_prediction']['intent_confidence']:.2f}), "
                  f"Action={res['neural_raw_prediction']['action']} ({res['neural_raw_prediction']['action_confidence']:.2f})")
            print(f"    Pipeline:   Intent={res['pipeline_classification']['intent']}, Action={res['pipeline_classification']['action']}, "
                  f"Handler={res['action_dispatch_decision']['target_handler']}, CanExec={res['action_dispatch_decision']['can_execute']}")
            print(f"    Response:   {res['final_response'][:80]}...")
            print("-" * 55)
            
        all_results[model_name] = case_results

    out_file = os.path.join(PROJECT_ROOT, "scratch", "v12_reproduction_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved detailed reproduction trace to: {out_file}")

if __name__ == "__main__":
    main()
