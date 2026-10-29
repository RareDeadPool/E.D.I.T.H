"""Longitudinal Evaluation Harness for E.D.I.T.H. V5.1 Cognitive Architecture.

Validates counterfactual reasoning, contextual sensitivity, temporal adaptation,
and state persistence across Scenarios A through H without predetermined emotional dogmatism.
"""

import os
import sys
import unittest
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.cognitive_state import CognitiveState, ContinuousAffectState
from engine.cognitive_cycle import EdithCognitiveCycle
from engine.appraisal import ContextualAppraisalEngine, AppraisalFrame
from engine.affective_dynamics import AffectiveDynamicsEngine


class TestV5LongitudinalHarness(unittest.TestCase):
    """Test longitudinal adaptation, counterfactual context differentiation, and temporal dynamics."""

    def setUp(self):
        self.session_id = f"test_longitudinal_{int(time.time()*1000)}"
        self.cycle = EdithCognitiveCycle(session_id=self.session_id)

    def test_scenario_a_vs_b_contextual_failure_differentiation(self):
        """A vs B: Same script failure has vastly different meaning in exploration vs urgency."""
        cycle_a = EdithCognitiveCycle(session_id="scen_a")
        cycle_b = EdithCognitiveCycle(session_id="scen_b")

        same_failure_event = {
            "type": "execution_failure",
            "text": "python test_experiment.py exited with return code 1: AssertionError",
            "intent": "execute_test",
            "action": "run_python_script",
            "confidence": 0.85,
            "is_social": False,
            "recovery_available": True
        }

        # Scenario A: Low-priority exploration, initial trial
        cycle_a.state.goals.active_goal = "explore prototype logging ideas"
        cycle_a.state.goals.priority = 0.2
        cycle_a.state.goals.urgency = 0.2
        cycle_a.state.goals.progress = 0.1
        cycle_a.state.condition.cognitive_load = 0.25
        cycle_a.state.condition.error_accumulation = 0.0
        cycle_a.state.operational.retry_count = 0

        appr_a = cycle_a.appraisal_engine.appraise_event(same_failure_event, cycle_a.state)
        cycle_a.affect_engine.update_affect(cycle_a.state, appr_a, event_summary="exploratory script fail")

        # Scenario B: High-urgency production hotfix after multiple retries
        cycle_b.state.goals.active_goal = "deploy critical production hotfix"
        cycle_b.state.goals.priority = 0.95
        cycle_b.state.goals.urgency = 0.95
        cycle_b.state.goals.progress = 0.2
        cycle_b.state.condition.cognitive_load = 0.85
        cycle_b.state.condition.error_accumulation = 0.7
        cycle_b.state.operational.retry_count = 3

        appr_b = cycle_b.appraisal_engine.appraise_event(same_failure_event, cycle_b.state)
        cycle_b.affect_engine.update_affect(cycle_b.state, appr_b, event_summary="urgent hotfix script fail")

        # Assert contextual differentiation
        self.assertGreater(appr_a.goal_congruence, appr_b.goal_congruence, "Exploration should have less negative congruence than urgent hotfix")
        self.assertLess(cycle_a.state.momentary_affect.frustration, cycle_b.state.momentary_affect.frustration, "Frustration in exploration should be lower than in urgent failure")
        self.assertGreater(cycle_a.state.momentary_affect.valence, cycle_b.state.momentary_affect.valence, "Valence should be less depressed in exploration")
        self.assertGreater(cycle_a.state.momentary_affect.coping_potential if hasattr(cycle_a.state.momentary_affect, 'coping_potential') else appr_a.coping_potential, appr_b.coping_potential)

    def test_scenario_c_unreliable_tool_recovery_after_fix(self):
        """C: Previously unreliable tool begins working consistently after a genuine fix."""
        cycle = EdithCognitiveCycle(session_id="scen_c")

        # Initial state: Tool ffmpeg previously failed repeatedly
        for _ in range(3):
            cycle.memory.register_procedure_outcome(
                skill_name="ffmpeg_transcode",
                goal_pattern="transcode video",
                action_sequence=["ffmpeg_transcode"],
                success=False,
                failure_reason="Invalid pixel format flag"
            )

        skill_before = cycle.memory.get_procedure("ffmpeg_transcode")
        self.assertIsNotNone(skill_before)
        initial_rel = float(skill_before.reliability)
        self.assertLess(initial_rel, 0.4, "Procedure should reflect low reliability initially")

        # Genuine fix applied: Tool now succeeds twice with verification
        res_post1 = cycle.post_action_cycle(
            user_command="transcode video with fixed flags",
            action="ffmpeg_transcode",
            execution_result={"status": "success", "message": "transcode complete", "target": "output.mp4"},
            verified=True
        )
        res_post2 = cycle.post_action_cycle(
            user_command="transcode second video with fixed flags",
            action="ffmpeg_transcode",
            execution_result={"status": "success", "message": "transcode complete", "target": "output2.mp4"},
            verified=True
        )

        skill_after = cycle.memory.get_procedure("ffmpeg_transcode")
        self.assertGreater(skill_after.reliability, initial_rel, "Reliability must increase after verified successes")
        self.assertGreaterEqual(cycle.state.momentary_affect.valence, 0.0, "Affect must recover toward positive on verified success")

    def test_scenario_d_tool_deterioration_after_environment_change(self):
        """D: A tool worked previously but becomes unreliable after environment change."""
        cycle = EdithCognitiveCycle(session_id="scen_d")

        # Initial state: Tool has high success track record
        for _ in range(5):
            cycle.memory.register_procedure_outcome(
                skill_name="web_fetch",
                goal_pattern="download resource",
                action_sequence=["web_fetch"],
                success=True
            )

        proc_prior = cycle.memory.get_procedure("web_fetch")
        prior_rel = float(proc_prior.reliability)
        self.assertGreater(prior_rel, 0.8)

        # Environment changes: Network endpoint drops, consecutive failures occur
        post1 = cycle.post_action_cycle(
            user_command="fetch user data",
            action="web_fetch",
            execution_result={"status": "failure", "message": "HTTP 503 Gateway Timeout"},
            verified=False
        )

        self.assertGreater(post1["prediction_error"], 0.4, "High expectation of success violated should yield high prediction error")
        proc_updated = cycle.memory.get_procedure("web_fetch")
        self.assertLess(proc_updated.reliability, prior_rel, "Procedure reliability should be demoted upon verified failure")

    def test_scenario_e_progress_despite_intermediate_failure(self):
        """E: Plan makes progress despite an unexpected intermediate non-critical failure."""
        cycle = EdithCognitiveCycle(session_id="scen_e")
        cycle.state.goals.active_goal = "deploy application"

        # Step 1: Build succeeds
        cycle.post_action_cycle("build project", "compile", {"status": "success"}, verified=True)
        # Step 2: Test succeeds
        cycle.post_action_cycle("run tests", "pytest", {"status": "success"}, verified=True)
        # Step 3: Non-critical step fails (cache warm-up)
        cycle.post_action_cycle("warm cache", "curl_cache", {"status": "warning", "message": "cache missed"}, verified=True)
        # Step 4: Deploy binary succeeds
        cycle.post_action_cycle("start service", "launch_daemon", {"status": "success"}, verified=True)

        self.assertGreaterEqual(cycle.state.goals.progress, 0.7, "Progress should advance toward completion despite non-critical step warning")
        self.assertGreater(cycle.state.momentary_affect.valence, 0.0, "Overall affect should remain positive as mission advances")

    def test_scenario_f_verification_failure_despite_clean_exit(self):
        """F: Plan appears successful (exit code 0) but actual output fails verification."""
        cycle = EdithCognitiveCycle(session_id="scen_f")

        # Process exits 0 but output file is corrupted (0 bytes)
        post_res = cycle.post_action_cycle(
            user_command="export user analytics report",
            action="export_csv",
            execution_result={"status": "success", "message": "process exited with code 0"},
            verified=False  # Output verification failed!
        )

        self.assertFalse(post_res["verified"])
        # Should NOT award full progress when verification fails
        self.assertLess(cycle.state.goals.progress, 1.0, "Unverified action must not advance goal progress to 1.0")
        self.assertGreaterEqual(post_res["prediction_error"], 0.3, "Verification mismatch should trigger prediction error")

    def test_scenario_g_contradictory_evidence_supersedes_assumption(self):
        """G: New evidence contradicts a previously confident interpretation."""
        cycle = EdithCognitiveCycle(session_id="scen_g")

        # Initial command: "open document" -> assumed notepad
        pre1 = cycle.pre_action_cycle("open the notes", {"intent": "open_app", "action": "open_notepad", "confidence": 0.9, "entities": {}})
        self.assertIn("open_notepad", cycle.state.perception.current_interpretation)

        # Contradicting user command: "no, do not use notepad, open vs code"
        pre2 = cycle.pre_action_cycle(
            "no, do not use notepad, open vs code",
            {"intent": "open_app", "action": "open_vscode", "confidence": 0.95, "entities": {}}
        )

        self.assertIn("open_vscode", cycle.state.perception.current_interpretation)
        last_query = cycle.memory.get_working_item("last_user_query")
        self.assertIn("open vs code", str(last_query))

    def test_scenario_h_temporary_setback_during_longer_term_progress(self):
        """H: Temporary setback occurs while a longer-term task is progressing well."""
        cycle = EdithCognitiveCycle(session_id="scen_h")
        cycle.state.goals.active_goal = "full system refactoring"
        cycle.state.goals.progress = 0.8
        cycle.state.baseline_mood.valence = 0.5
        cycle.state.baseline_mood.engagement = 0.85

        # Transient setback: single linter error
        appr_setback = cycle.appraisal_engine.appraise_event({
            "type": "minor_warning",
            "text": "linter warning: unused import in helper.py",
            "is_success": False,
            "recovery_available": True
        }, cycle.state)
        cycle.affect_engine.update_affect(cycle.state, appr_setback, event_summary="transient lint error")

        # Baseline mood should absorb momentary bump without collapsing
        self.assertGreater(cycle.state.baseline_mood.valence, 0.3, "Baseline mood should remain stable during long-term progress")
        self.assertGreater(cycle.state.momentary_affect.engagement, 0.5, "Engagement should be sustained")

    def test_temporal_decay_dynamics_and_elapsed_time(self):
        """Test decay toward baseline mood across elapsed seconds."""
        engine = AffectiveDynamicsEngine()
        state = CognitiveState()
        state.baseline_mood = ContinuousAffectState(valence=0.1, arousal=0.3, frustration=0.0)
        state.momentary_affect = ContinuousAffectState(valence=-0.7, arousal=0.9, frustration=0.8)

        # Elapsed time: 120 seconds (2 half-lives of 60s -> decay factor ~ 0.25)
        neutral_appraisal = AppraisalFrame(goal_congruence=0.0, expectation_violation=0.0, coping_potential=0.8, uncertainty=0.1)
        engine.update_affect(state, neutral_appraisal, elapsed_seconds=120.0)

        # Momentary frustration should have decayed significantly toward baseline 0.0
        self.assertLess(state.momentary_affect.frustration, 0.35, "Frustration must decay significantly after 120s elapsed time")
        self.assertGreater(state.momentary_affect.valence, -0.4, "Valence must recover toward baseline after 120s elapsed time")


if __name__ == "__main__":
    unittest.main()
