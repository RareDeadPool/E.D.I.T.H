"""Comprehensive Evaluation Benchmark Suite for E.D.I.T.H. V5.1 Cognitive Architecture.

Benchmarks the learned neural appraisal and temporal affective models against baselines:
1. Mean/Constant Baseline: Predicts empirical mean from training set.
2. Deterministic Heuristic Baseline: Rule-based formulas derived from task cues.
3. Rule-Based Appraisal Engine: ContextualAppraisalEngine without neural weights.
4. Learned Model without Context: CognitiveNeuralSystem with with_context=False.
5. Full Learned Model with Context: CognitiveNeuralSystem with with_context=True.

Evaluated on two distinct held-out evaluation sets:
- Set A: Zero-leakage held-out synthetic scenario families (Families 19-24).
- Set B: Author-curated novel held-out scenarios (data/novel_held_out_scenarios.json).

Produces granular dimension-by-dimension MAE/MSE, calibration analysis, and
exports machine-readable benchmark report models/checkpoints/benchmark_baseline_comparison.json.
"""

from __future__ import annotations
import os
import sys
import json
import math
from typing import Dict, Any, List, Tuple
import torch
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.cognitive_state import ContinuousAffectState, CognitiveState
from engine.appraisal import ContextualAppraisalEngine, AppraisalFrame
from engine.learned_appraisal_model import (
    CognitiveNeuralSystem,
    APPRAISAL_DIMENSIONS,
    AFFECT_DIMENSIONS
)


class MeanConstantBaseline:
    """Predicts empirical constant mean vector computed from training data."""

    def __init__(self, train_scenarios: List[Dict[str, Any]]):
        appr_sums = {d: 0.0 for d in APPRAISAL_DIMENSIONS}
        aff_sums = {d: 0.0 for d in AFFECT_DIMENSIONS}
        agency_map = {"self": 0.0, "user": 0.33, "system": 0.66, "environment": 1.0}

        n = len(train_scenarios)
        for s in train_scenarios:
            for d in APPRAISAL_DIMENSIONS:
                val = s["target_appraisal"].get(d, 0.0)
                if isinstance(val, str):
                    val = agency_map.get(val.lower(), 0.5)
                appr_sums[d] += float(val)

            t_aff = s["target_affect"]
            for d in AFFECT_DIMENSIONS:
                aff_sums[d] += float(t_aff.get(d, 0.0))

        self.mean_appraisal = {d: appr_sums[d] / n for d in APPRAISAL_DIMENSIONS}
        self.mean_affect = {d: aff_sums[d] / n for d in AFFECT_DIMENSIONS}

    def predict_appraisal(self, scenario: Dict[str, Any]) -> Tuple[Dict[str, float], float]:
        return self.mean_appraisal.copy(), 0.5

    def predict_affect(self, scenario: Dict[str, Any]) -> Dict[str, float]:
        return self.mean_affect.copy()


class SimpleDeterministicBaseline:
    """Deterministic heuristic baseline mapping cues directly to appraisal and affect."""

    def predict_appraisal(self, scenario: Dict[str, Any]) -> Tuple[Dict[str, float], float]:
        text = scenario["event"].lower()
        urgency = scenario.get("goal_urgency", 0.5)
        progress = scenario.get("goal_progress", 0.0)
        pred_err = scenario.get("prediction_error", 0.0)
        cog_load = scenario.get("cognitive_load", 0.2)

        # Basic sentiment/congruence heuristic
        is_success = any(w in text for w in ["success", "passed", "perfect", "cleaned", "awesome", "promotion", "matched"])
        is_failure = any(w in text for w in ["fail", "error", "exception", "abort", "timeout", "kill", "conflict", "blocked"])

        if is_success:
            congruence = 0.8
            violation = 0.1
        elif is_failure:
            congruence = -0.7 if urgency > 0.5 else -0.3
            violation = max(0.4, pred_err)
        else:
            congruence = 0.0
            violation = pred_err

        appr = {
            "novelty": min(1.0, 0.2 + 0.5 * pred_err),
            "goal_relevance": urgency,
            "goal_congruence": congruence,
            "expectation_violation": violation,
            "agency": 0.5,
            "controllability": max(0.2, min(0.9, 1.0 - cog_load * 0.5)),
            "coping_potential": 0.8 if is_success else (0.5 if urgency > 0.8 else 0.75),
            "uncertainty": min(1.0, 0.2 + 0.6 * pred_err),
            "effort_required": cog_load,
            "task_progress": progress,
            "social_significance": 0.8 if "user" in text else 0.3,
            "norm_consistency": 0.4 if "security" in text or "restricted" in text else 0.95
        }
        return appr, 0.75

    def predict_affect(self, scenario: Dict[str, Any]) -> Dict[str, float]:
        appr, _ = self.predict_appraisal(scenario)
        v = appr["goal_congruence"] * 0.7 + appr["task_progress"] * 0.3
        a = max(0.1, min(1.0, 0.2 + 0.5 * appr["expectation_violation"] + 0.3 * appr["effort_required"]))
        d = appr["controllability"] * 0.5 + appr["coping_potential"] * 0.5
        u = appr["uncertainty"]
        f = max(0.0, min(1.0, -appr["goal_congruence"] * 0.8)) if appr["goal_congruence"] < 0 else 0.0
        eng = min(1.0, 0.4 + 0.4 * appr["goal_relevance"])
        trust = 0.85 if appr["social_significance"] > 0.7 else 0.7

        return {
            "valence": max(-1.0, min(1.0, v)),
            "arousal": max(0.0, min(1.0, a)),
            "dominance": max(0.0, min(1.0, d)),
            "uncertainty": max(0.0, min(1.0, u)),
            "frustration": max(0.0, min(1.0, f)),
            "engagement": max(0.0, min(1.0, eng)),
            "social_trust": max(0.0, min(1.0, trust))
        }


class RuleBasedAppraisalEngineBaseline:
    """Wraps the actual production ContextualAppraisalEngine without neural weights."""

    def __init__(self):
        self.engine = ContextualAppraisalEngine(neural_appraisal_model=None)

    def predict_appraisal(self, scenario: Dict[str, Any]) -> Tuple[Dict[str, float], float]:
        st = CognitiveState()
        st.goals.active_goal = "run_task"
        st.goals.priority = scenario.get("goal_urgency", 0.5)
        st.goals.progress = scenario.get("goal_progress", 0.0)
        st.condition.cognitive_load = scenario.get("cognitive_load", 0.2)
        st.condition.prediction_error = scenario.get("prediction_error", 0.0)

        evt_repr = {
            "type": "event",
            "text": scenario["event"],
            "intent": "execute",
            "action": "action",
            "confidence": 0.8,
            "is_social": "user says" in scenario["event"].lower(),
            "recovery_available": True
        }

        frame = self.engine.appraise_event(evt_repr, st)
        appr_dict = frame.to_dict()
        agency_map = {"self": 0.0, "user": 0.33, "system": 0.66, "environment": 1.0}
        out = {}
        for d in APPRAISAL_DIMENSIONS:
            val = appr_dict.get(d, 0.0)
            if isinstance(val, str):
                val = agency_map.get(val.lower(), 0.5)
            out[d] = float(val)
        return out, getattr(frame, "confidence", 0.85)

    def predict_affect(self, scenario: Dict[str, Any]) -> Dict[str, float]:
        appr, _ = self.predict_appraisal(scenario)
        v = appr["goal_congruence"] * 0.7 + appr["task_progress"] * 0.3
        a = max(0.1, min(1.0, 0.2 + 0.4 * appr["expectation_violation"] + 0.3 * appr["novelty"] + 0.2 * appr["effort_required"]))
        d = appr["controllability"] * 0.5 + appr["coping_potential"] * 0.5
        u = appr["uncertainty"]
        f = max(0.0, -appr["goal_congruence"] * (1.2 - 0.5 * appr["coping_potential"])) if appr["goal_congruence"] < 0 else 0.0
        eng = min(1.0, 0.3 + 0.4 * appr["goal_relevance"] + 0.3 * appr["controllability"])
        trust = 0.85 if appr["social_significance"] > 0.6 else 0.7

        return {
            "valence": max(-1.0, min(1.0, v)),
            "arousal": max(0.0, min(1.0, a)),
            "dominance": max(0.0, min(1.0, d)),
            "uncertainty": max(0.0, min(1.0, u)),
            "frustration": max(0.0, min(1.0, f)),
            "engagement": max(0.0, min(1.0, eng)),
            "social_trust": max(0.0, min(1.0, trust))
        }


def extract_targets(scenario: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Extract float target vectors for appraisal and affect."""
    agency_map = {"self": 0.0, "user": 0.33, "system": 0.66, "environment": 1.0}
    appr_target = {}
    for d in APPRAISAL_DIMENSIONS:
        val = scenario["target_appraisal"].get(d, 0.0)
        if isinstance(val, str):
            val = agency_map.get(val.lower(), 0.5)
        appr_target[d] = float(val)

    aff_target = {d: float(scenario["target_affect"].get(d, 0.0)) for d in AFFECT_DIMENSIONS}
    return appr_target, aff_target


def evaluate_system_on_dataset(
    predict_fn_appraisal,
    predict_fn_affect,
    dataset: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Compute overall and dimension-specific MAE, MSE, and calibration metrics."""
    appr_errors = {d: [] for d in APPRAISAL_DIMENSIONS}
    aff_errors = {d: [] for d in AFFECT_DIMENSIONS}
    all_appr_abs = []
    all_aff_abs = []
    all_appr_sq = []
    all_aff_sq = []

    confidences = []
    mean_errors = []

    for s in dataset:
        target_appr, target_aff = extract_targets(s)
        pred_appr, conf = predict_fn_appraisal(s)
        pred_aff = predict_fn_affect(s)

        case_appr_errs = []
        for d in APPRAISAL_DIMENSIONS:
            err = abs(pred_appr.get(d, 0.0) - target_appr[d])
            appr_errors[d].append(err)
            all_appr_abs.append(err)
            all_appr_sq.append(err ** 2)
            case_appr_errs.append(err)

        for d in AFFECT_DIMENSIONS:
            err = abs(pred_aff.get(d, 0.0) - target_aff[d])
            aff_errors[d].append(err)
            all_aff_abs.append(err)
            all_aff_sq.append(err ** 2)

        confidences.append(conf)
        mean_errors.append(float(np.mean(case_appr_errs)))

    # Compute calibration correlation: higher confidence should correlate with lower error
    if len(confidences) > 1 and np.std(confidences) > 1e-6 and np.std(mean_errors) > 1e-6:
        calib_corr = float(np.corrcoef(confidences, mean_errors)[0, 1])
    else:
        calib_corr = 0.0

    return {
        "overall_appraisal_mae": round(float(np.mean(all_appr_abs)), 4),
        "overall_appraisal_mse": round(float(np.mean(all_appr_sq)), 4),
        "overall_affect_mae": round(float(np.mean(all_aff_abs)), 4),
        "overall_affect_mse": round(float(np.mean(all_aff_sq)), 4),
        "dimension_appraisal_mae": {d: round(float(np.mean(appr_errors[d])), 4) for d in APPRAISAL_DIMENSIONS},
        "dimension_affect_mae": {d: round(float(np.mean(aff_errors[d])), 4) for d in AFFECT_DIMENSIONS},
        "confidence_error_correlation": round(calib_corr, 4),
        "sample_count": len(dataset)
    }


def run_comprehensive_benchmark() -> Dict[str, Any]:
    print("=" * 70)
    print("📊 Executing E.D.I.T.H. V5.1 Cognitive Benchmark Across 5 Baselines & 2 Test Sets")
    print("=" * 70)

    # 1. Load Data
    data_path = os.path.join(PROJECT_ROOT, "data", "cognitive_scenarios.json")
    novel_path = os.path.join(PROJECT_ROOT, "data", "novel_held_out_scenarios.json")

    with open(data_path, "r", encoding="utf-8") as f:
        all_scenarios = json.load(f)

    with open(novel_path, "r", encoding="utf-8") as f:
        novel_scenarios = json.load(f)

    held_out_family_ids = {
        "family_19_silent_output_corruption",
        "family_20_hardware_audio_disconnect",
        "family_21_long_idle_recovery",
        "family_22_background_task_cancellation",
        "family_23_exploratory_eureka_discovery",
        "family_24_powershell_security_restriction"
    }
    train_scenarios = [s for s in all_scenarios if s.get("family_id") not in held_out_family_ids]
    val_scenarios_zero_leakage = [s for s in all_scenarios if s.get("family_id") in held_out_family_ids]

    print(f"Loaded {len(train_scenarios)} train scenarios, {len(val_scenarios_zero_leakage)} zero-leakage synthetic test scenarios, {len(novel_scenarios)} novel authored test scenarios.")

    # 2. Initialize Models
    neural_model = CognitiveNeuralSystem(device="cpu")
    ckpt_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "cognitive_appraisal.pt")
    if os.path.exists(ckpt_path):
        neural_model.load_checkpoint(ckpt_path)
        print("Loaded neural checkpoint successfully.")
    else:
        print("Warning: Checkpoint not found; running with initialized weights.")

    mean_baseline = MeanConstantBaseline(train_scenarios)
    simple_baseline = SimpleDeterministicBaseline()
    rule_engine_baseline = RuleBasedAppraisalEngineBaseline()

    # Define prediction adapters
    def predict_neural_no_context_appr(s):
        prior = ContinuousAffectState.from_dict(s.get("prior_affect", {}))
        return neural_model.predict_appraisal(
            event_text=s["event"],
            goal_urgency=s.get("goal_urgency", 0.5),
            goal_progress=s.get("goal_progress", 0.0),
            prior_affect=prior,
            cognitive_load=s.get("cognitive_load", 0.2),
            prediction_error=s.get("prediction_error", 0.0),
            with_context=False
        )

    def predict_neural_no_context_aff(s):
        appr, _ = predict_neural_no_context_appr(s)
        prior = ContinuousAffectState.from_dict(s.get("prior_affect", {}))
        aff, _, _, _ = neural_model.predict_transition(
            appraisal_dict=appr,
            prior_affect=prior,
            cognitive_load=s.get("cognitive_load", 0.2),
            prediction_error=s.get("prediction_error", 0.0),
            delta_t_seconds=1.0,
            with_context=False
        )
        return aff.to_dict()

    def predict_neural_with_context_appr(s):
        prior = ContinuousAffectState.from_dict(s.get("prior_affect", {}))
        return neural_model.predict_appraisal(
            event_text=s["event"],
            goal_urgency=s.get("goal_urgency", 0.5),
            goal_progress=s.get("goal_progress", 0.0),
            prior_affect=prior,
            cognitive_load=s.get("cognitive_load", 0.2),
            prediction_error=s.get("prediction_error", 0.0),
            with_context=True
        )

    def predict_neural_with_context_aff(s):
        appr, _ = predict_neural_with_context_appr(s)
        prior = ContinuousAffectState.from_dict(s.get("prior_affect", {}))
        aff, _, _, _ = neural_model.predict_transition(
            appraisal_dict=appr,
            prior_affect=prior,
            cognitive_load=s.get("cognitive_load", 0.2),
            prediction_error=s.get("prediction_error", 0.0),
            delta_t_seconds=1.0,
            with_context=True
        )
        return aff.to_dict()

    baselines = {
        "mean_constant_baseline": (mean_baseline.predict_appraisal, mean_baseline.predict_affect),
        "deterministic_heuristic_baseline": (simple_baseline.predict_appraisal, simple_baseline.predict_affect),
        "rule_based_appraisal_engine": (rule_engine_baseline.predict_appraisal, rule_engine_baseline.predict_affect),
        "learned_model_no_context": (predict_neural_no_context_appr, predict_neural_no_context_aff),
        "full_learned_model_with_context": (predict_neural_with_context_appr, predict_neural_with_context_aff)
    }

    benchmark_results = {
        "metadata": {
            "version": "v5.1",
            "eval_sets": {
                "set_a_zero_leakage_families": len(val_scenarios_zero_leakage),
                "set_b_novel_held_out": len(novel_scenarios)
            },
            "checkpoint": "models/checkpoints/cognitive_appraisal.pt"
        },
        "results_set_a_zero_leakage_families": {},
        "results_set_b_novel_held_out": {}
    }

    print("\n--- Evaluating on Set A: Zero-Leakage Held-Out Families (180 samples) ---")
    for name, (fn_appr, fn_aff) in baselines.items():
        res = evaluate_system_on_dataset(fn_appr, fn_aff, val_scenarios_zero_leakage)
        benchmark_results["results_set_a_zero_leakage_families"][name] = res
        print(f"[{name}] Appr MAE: {res['overall_appraisal_mae']:.4f} | Aff MAE: {res['overall_affect_mae']:.4f} | Corr: {res['confidence_error_correlation']:+.3f}")

    print("\n--- Evaluating on Set B: Author-Curated Novel Held-Out Scenarios (20 samples) ---")
    for name, (fn_appr, fn_aff) in baselines.items():
        res = evaluate_system_on_dataset(fn_appr, fn_aff, novel_scenarios)
        benchmark_results["results_set_b_novel_held_out"][name] = res
        print(f"[{name}] Appr MAE: {res['overall_appraisal_mae']:.4f} | Aff MAE: {res['overall_affect_mae']:.4f} | Corr: {res['confidence_error_correlation']:+.3f}")

    # Output file
    out_file = os.path.join(PROJECT_ROOT, "models", "checkpoints", "benchmark_baseline_comparison.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)

    print(f"\n✅ Benchmark complete! Exported machine-readable results to: {out_file}")
    return benchmark_results


if __name__ == "__main__":
    run_comprehensive_benchmark()
