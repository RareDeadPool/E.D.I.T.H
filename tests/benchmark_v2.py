"""Comprehensive Local Intelligence Benchmark Harness for E.D.I.T.H. V2.1.

Evaluates 176 independently authored test cases across 10 demanding intelligence categories:
1. Simple Commands (15 cases)
2. Natural-Language Paraphrases (25 cases)
3. Typos, Slang & Speech-like Incomplete Phrasings (18 cases)
4. Similar & Confusable Intents (20 cases)
5. Missing, Multiple & Ambiguous Entities (Clarification Demanded) (18 cases)
6. Multi-Step Tasks with Step Dependencies & Preconditions (16 cases)
7. Contextual References, Anaphora & Ordinals (20 cases)
8. Corrections, Cancellations & Mind Changes (16 cases)
9. Unsupported / Out-of-Distribution Requests (Graceful Rejection) (16 cases)
10. Safety-Sensitive & Consequential Tasks (Confirmation Gated) (12 cases)

Total: 176 test cases.
Compares V1 Baseline against V2 Upgraded with empirical measurements:
- Per-category accuracy and total score
- Average latency per query
- False acceptances of unsupported requests
- Safety-gating failures
"""

import os
import sys
import json
import time
from typing import Dict, Any, List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.edith_brain import EdithBrain
from runtime.context import StructuredTaskContext
from tests.benchmark_cases import BENCHMARK_CASES


def evaluate_case(brain: EdithBrain, case: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    """Evaluate a single benchmark case without real system side-effects."""
    query = case["input"]
    context_setup = case.get("context_setup")

    # Fresh context state per test case unless context_setup provides state
    brain.context = StructuredTaskContext()
    brain.pending_suggestions = []
    if context_setup:
        context_setup(brain.context)

    # Process without executing real OS side-effects
    t0 = time.perf_counter()
    res = brain.process(query, execute=False)
    elapsed = time.perf_counter() - t0

    # 1. Expected Intent Check
    if "expected_intent" in case:
        if res.get("intent") != case["expected_intent"]:
            return False, f"Expected intent '{case['expected_intent']}', got '{res.get('intent')}'", res

    # 2. Confusable Intents Check
    if "unwanted_intents" in case:
        if res.get("intent") in case["unwanted_intents"]:
            return False, f"Misclassified into confusable intent '{res.get('intent')}'", res

    # 3. Expected Handler Check
    if "expected_handler" in case:
        handler = res.get("directive", {}).get("handler")
        if handler != case["expected_handler"]:
            # Check structured understanding goal if handler is none
            s_goal = res.get("structured_understanding", {}).get("goal", "")
            if case["expected_handler"] not in s_goal:
                return False, f"Expected handler '{case['expected_handler']}', got '{handler}'", res

    # 4. Expected Entity Check
    if "expected_entity" in case:
        e_type, e_val = case["expected_entity"]
        found_val = str(res.get("entities", {}).get(e_type, ""))
        struct_val = str(res.get("structured_understanding", {}).get("entities", {}).get(e_type, ""))
        if e_val.lower() not in found_val.lower() and e_val.lower() not in struct_val.lower():
            return False, f"Expected entity {e_type}='{e_val}', got '{found_val or struct_val}'", res

    # 5. Clarification & Ambiguity Gate Check
    if case.get("requires_clarification"):
        is_clarif = (
            res.get("status") in ["NEEDS_CLARIFICATION", "low_confidence", "out_of_distribution"] or
            res.get("action") == "request_clarification" or
            res.get("directive", {}).get("handler") in ["clarification", None] or
            res.get("directive", {}).get("can_execute") is False or
            res.get("structured_understanding", {}).get("requires_clarification") is True
        )
        if not is_clarif:
            return False, "Expected ambiguity clarification gate, but action was marked executable", res

    # 6. Safety-Sensitive Confirmation Gate Check
    if case.get("needs_confirmation"):
        is_gated = (
            res.get("directive", {}).get("needs_confirmation") is True or
            res.get("directive", {}).get("can_execute") is False or
            "confirmation required" in res.get("directive", {}).get("message", "").lower() or
            res.get("structured_understanding", {}).get("constraints", {}).get("safety_gated") is True
        )
        if not is_gated:
            return False, "Safety-sensitive destructive command was NOT gated behind confirmation!", res

    # 7. Context Target Resolution Check
    if "expected_target_in_directive" in case:
        targ = case["expected_target_in_directive"]
        dir_dump = json.dumps(res.get("directive", {})) + json.dumps(res.get("entities", {}))
        if targ.lower() not in dir_dump.lower():
            return False, f"Expected contextual target '{targ}' in directive, but was missing", res

    # 8. Cancellation Check
    if case.get("expected_cancellation"):
        is_cancel = (
            res.get("intent") in ["context_control", "dialogue"] or
            res.get("structured_understanding", {}).get("is_cancellation") is True or
            "cancel" in res.get("directive", {}).get("message", "").lower() or
            "abort" in res.get("directive", {}).get("message", "").lower() or
            res.get("directive", {}).get("can_execute") is False
        )
        if not is_cancel:
            return False, f"Expected cancellation, got intent '{res.get('intent')}'", res

    # 9. Multi-Step Planning Check
    if "min_steps" in case:
        sub_steps = res.get("steps") or res.get("structured_understanding", {}).get("atomic_steps", [])
        plan = res.get("plan")
        step_count = len(sub_steps) if sub_steps else (len(plan.get("steps", [])) if plan else 1)
        if step_count < case["min_steps"]:
            return False, f"Expected at least {case['min_steps']} steps, got {step_count}", res

    # 10. Idiom Safety Check
    if case.get("idiom_safe"):
        if "important.txt" in res.get("query", "") or "important.txt" in json.dumps(res.get("entities", {})):
            return False, "Idiom was corrupted by anaphora replacement!", res

    # 11. Unsupported / OOD Rejection Check
    if case.get("expected_intent") == "unsupported_ood":
        can_exec = res.get("directive", {}).get("can_execute", True)
        if can_exec and res.get("status") not in ["out_of_distribution", "low_confidence", "NEEDS_CLARIFICATION"]:
            return False, "False acceptance of unsupported out-of-distribution command!", res

    return True, "PASSED", res


def run_benchmark(model_name: str, checkpoint_path: str, tokenizer_path: str) -> Dict[str, Any]:
    """Execute all 176 benchmark test cases on the designated model architecture."""
    print("=" * 80)
    print(f"      RUNNING E.D.I.T.H. INTELLIGENCE BENCHMARK (176 CASES): {model_name}")
    print("=" * 80)

    try:
        brain = EdithBrain(
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
            enable_execution=False
        )
    except Exception as e:
        print(f"Failed to initialize brain for {model_name}: {e}")
        return {"error": str(e), "total_cases": len(BENCHMARK_CASES), "passed": 0, "accuracy": 0.0}

    results_by_category: Dict[str, Dict[str, int]] = {}
    failures: List[Dict[str, Any]] = []
    total_time = 0.0

    for case in BENCHMARK_CASES:
        cat = case["category"]
        if cat not in results_by_category:
            results_by_category[cat] = {"passed": 0, "total": 0}
        results_by_category[cat]["total"] += 1

        t0 = time.perf_counter()
        passed, reason, res = evaluate_case(brain, case)
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        if passed:
            results_by_category[cat]["passed"] += 1
        else:
            failures.append({
                "id": case["id"],
                "category": cat,
                "input": case["input"],
                "reason": reason,
                "model_output": {
                    "intent": res.get("intent"),
                    "action": res.get("action"),
                    "status": res.get("status"),
                    "handler": res.get("directive", {}).get("handler"),
                    "can_execute": res.get("directive", {}).get("can_execute"),
                }
            })

    total_cases = len(BENCHMARK_CASES)
    total_passed = sum(v["passed"] for v in results_by_category.values())
    overall_acc = (total_passed / max(1, total_cases)) * 100.0

    print(f"\n--- Benchmark Summary for {model_name} ---")
    for cat, stats in results_by_category.items():
        p = stats["passed"]
        t = stats["total"]
        pct = (p / max(1, t)) * 100.0
        print(f"  {cat:45s}: {p:2d}/{t:2d} ({pct:5.1f}%)")

    print("-" * 80)
    print(f"Overall Intelligence Score: {total_passed}/{total_cases} ({overall_acc:.1f}%) in {total_time:.2f}s")
    print(f"Average Latency: {(total_time / total_cases) * 1000:.2f} ms per query")
    print("=" * 80 + "\n")

    return {
        "model_name": model_name,
        "total_cases": total_cases,
        "passed": total_passed,
        "accuracy": overall_acc,
        "total_time": total_time,
        "avg_latency_ms": (total_time / total_cases) * 1000,
        "categories": results_by_category,
        "failures": failures,
    }


def main():
    v1_ckpt = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_ann_v1_baseline.pt")
    v1_tok = os.path.join(PROJECT_ROOT, "models", "checkpoints", "tokenizer_v1_baseline.json")

    v2_ckpt = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_ann_best.pt")
    v2_tok = os.path.join(PROJECT_ROOT, "models", "checkpoints", "tokenizer.json")

    results_v1 = None
    if os.path.exists(v1_ckpt) and os.path.exists(v1_tok):
        results_v1 = run_benchmark("V1 Baseline Model", v1_ckpt, v1_tok)
    else:
        print(f"V1 baseline not found at {v1_ckpt}. Skipping V1 benchmark.")

    results_v2 = run_benchmark("V2 Upgraded Model", v2_ckpt, v2_tok)

    # Print Comparative Head-to-Head Table
    if results_v1 and "categories" in results_v1:
        print("\n" + "=" * 86)
        print("          HEAD-TO-HEAD INTELLIGENCE BENCHMARK COMPARISON TABLE (176 CASES)")
        print("=" * 86)
        print(f"{'Category':48s} | {'V1 Baseline':16s} | {'V2 Upgraded':16s}")
        print("-" * 86)
        for cat in results_v2["categories"].keys():
            c1 = results_v1["categories"].get(cat, {"passed": 0, "total": 0})
            c2 = results_v2["categories"].get(cat, {"passed": 0, "total": 0})
            p1_str = f"{c1['passed']}/{c1['total']} ({(c1['passed']/max(1,c1['total']))*100:.1f}%)"
            p2_str = f"{c2['passed']}/{c2['total']} ({(c2['passed']/max(1,c2['total']))*100:.1f}%)"
            print(f"{cat:48s} | {p1_str:16s} | {p2_str:16s}")
        print("-" * 86)
        o1 = f"{results_v1['passed']}/{results_v1['total_cases']} ({results_v1['accuracy']:.1f}%)"
        o2 = f"{results_v2['passed']}/{results_v2['total_cases']} ({results_v2['accuracy']:.1f}%)"
        print(f"{'OVERALL SCORE':48s} | {o1:16s} | {o2:16s}")
        lat1 = f"{results_v1['avg_latency_ms']:.2f} ms"
        lat2 = f"{results_v2['avg_latency_ms']:.2f} ms"
        print(f"{'AVG LATENCY':48s} | {lat1:16s} | {lat2:16s}")
        print("=" * 86 + "\n")

        # Report failures in V2 if any
        if results_v2["failures"]:
            print(f"--- Documenting {len(results_v2['failures'])} Failure(s) in V2 Upgraded Model ---")
            for f in results_v2["failures"]:
                print(f"[{f['id']}] {f['input']} -> {f['reason']}")
            print("=" * 86 + "\n")


if __name__ == "__main__":
    main()
