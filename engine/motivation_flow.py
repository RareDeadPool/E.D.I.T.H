"""Intrinsic Motivation, Goal Competition Arbitration & Flow State Model for E.D.I.T.H. V5.

Implements:
1. Intrinsic Motivational Dynamics: Completion drive, curiosity/epistemic drive, competence drive,
   and resource conservation.
2. Goal Competition Arbitration: Resolves tradeoffs among urgency, thoroughness, safety, and clarification.
3. Functional Flow State Model: Operational proxy for sustained task engagement, context stability,
   and seamless interruption recovery.
"""

from __future__ import annotations
import math
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from .cognitive_state import CognitiveState, GoalMotivationState


@dataclass
class FlowStateEvaluation:
    """Diagnostic evaluation of the functional flow condition."""
    in_flow: bool
    flow_intensity: float         # 0.0 to 1.0
    challenge_skill_ratio: float  # ~1.0 is optimal balance
    goal_clarity: float          # 0.0 to 1.0
    feedback_immediacy: float    # 0.0 to 1.0
    attention_stability: float   # 0.0 to 1.0
    controllability: float       # 0.0 to 1.0
    interruption_penalty: float  # 0.0 to 1.0
    behavioral_directive: str    # 'sustain_focus', 'seek_clarification', 'recover_from_interruption', 'relax_mode'


class MotivationAndFlowEngine:
    """Arbitrates competing goals and models functional flow dynamics."""

    def __init__(self):
        self.last_switch_time = time.time()
        self.interruption_count = 0

    def evaluate_flow_state(
        self,
        state: CognitiveState,
        recent_interruption: bool = False
    ) -> FlowStateEvaluation:
        """
        Evaluate functional flow state without arbitrary single-rule shortcuts.
        Flow emerges when:
        - Active goal is clear and prioritized.
        - Perceived capability matches task complexity (balanced challenge-skill).
        - Feedback is reliable and verified.
        - Controllability is high and frustration is low.
        - Interruptions are absent or resolved.
        """
        goals = state.goals
        cond = state.condition
        affect = state.momentary_affect
        oper = state.operational

        # 1. Goal Clarity
        has_goal = bool(goals.active_goal)
        goal_clarity = 0.85 if has_goal and goals.priority > 0.4 else (0.4 if has_goal else 0.1)

        # 2. Challenge-Skill Balance
        # Skill estimate derived from dominance (control) and plan confidence
        estimated_skill = max(0.2, (affect.dominance + cond.plan_confidence) / 2.0)
        challenge = max(0.1, cond.task_complexity + 0.2 * oper.retry_count)
        # Optimal ratio is near 1.0; severe mismatch (<0.5 = bored, >1.8 = anxious/overwhelmed)
        cs_ratio = round(challenge / estimated_skill, 2)
        cs_balance_score = math.exp(-0.8 * abs(cs_ratio - 1.0))

        # 3. Feedback Immediacy & Verifiability
        feedback_immediacy = max(0.1, 1.0 - 0.5 * cond.prediction_error - 0.2 * oper.consecutive_failures)

        # 4. Attention Stability & Controllability
        attention_stability = max(0.1, 1.0 - 0.4 * cond.cognitive_load - 0.3 * affect.uncertainty)
        controllability = affect.dominance * (1.0 - affect.frustration * 0.5)

        # 5. Interruption penalty
        if recent_interruption:
            self.interruption_count += 1
            interruption_penalty = 0.5
        else:
            interruption_penalty = 0.0

        # Combine into continuous flow intensity
        if has_goal and cs_balance_score > 0.4 and controllability > 0.4 and affect.frustration < 0.4:
            raw_intensity = (
                0.30 * goal_clarity +
                0.25 * cs_balance_score +
                0.20 * feedback_immediacy +
                0.15 * attention_stability +
                0.10 * controllability -
                interruption_penalty
            )
            flow_intensity = max(0.0, min(1.0, raw_intensity))
        else:
            flow_intensity = max(0.0, 0.2 - interruption_penalty)

        in_flow = (flow_intensity >= 0.55) and (not recent_interruption)

        # Determine functional behavioral directive
        if recent_interruption:
            directive = "recover_from_interruption"
        elif in_flow:
            directive = "sustain_focus"
        elif goals.clarification_need > 0.5 or cond.prediction_error > 0.6:
            directive = "seek_clarification"
        else:
            directive = "relax_mode"

        state.in_flow_state = in_flow
        state.flow_intensity = round(flow_intensity, 3)

        return FlowStateEvaluation(
            in_flow=in_flow,
            flow_intensity=round(flow_intensity, 3),
            challenge_skill_ratio=cs_ratio,
            goal_clarity=round(goal_clarity, 3),
            feedback_immediacy=round(feedback_immediacy, 3),
            attention_stability=round(attention_stability, 3),
            controllability=round(controllability, 3),
            interruption_penalty=round(interruption_penalty, 3),
            behavioral_directive=directive
        )

    def arbitrate_goals(
        self,
        candidate_actions: List[Dict[str, Any]],
        state: CognitiveState
    ) -> Dict[str, Any]:
        """
        Arbitrate among competing candidate actions based on:
        - Goal relevance & completion value
        - Safety & authorization constraints (consequential actions require confirmation)
        - Computational cost & cognitive load
        - Task switching penalty (protecting flow continuity)
        - Curiosity vs exploitation balance
        """
        if not candidate_actions:
            return {"action": "noop", "directive": "idle", "reason": "no_candidates"}

        if len(candidate_actions) == 1:
            return candidate_actions[0]

        scored_candidates: List[Tuple[float, Dict[str, Any], str]] = []

        flow_active = state.in_flow_state
        explore_bias = state.goals.explore_vs_exploit
        cog_load = state.condition.cognitive_load

        for cand in candidate_actions:
            act_name = cand.get("action", "")
            cost = cand.get("cost", 0.3)
            goal_alignment = cand.get("goal_alignment", 0.5)
            safety_risk = cand.get("safety_risk", 0.0)

            # Invariant: Consequential safety risks MUST NEVER be prioritized automatically
            if safety_risk > 0.7:
                score = -10.0
                reason = "rejected_high_safety_risk"
                scored_candidates.append((score, cand, reason))
                continue

            # Base utility: Goal alignment minus cost
            utility = goal_alignment * 2.0 - cost * 0.8

            # Flow inertia: Heavily favor staying on current track rather than jumping to unrelated action
            if flow_active and cand.get("matches_current_goal", False):
                utility += 0.8

            # Under heavy cognitive load, penalize complex actions
            if cog_load > 0.7 and cost > 0.5:
                utility -= 0.6

            # Exploration vs exploitation adjustment
            if cand.get("is_novel", False):
                utility += explore_bias * 0.5
            else:
                utility += (1.0 - explore_bias) * 0.3

            scored_candidates.append((utility, cand, "arbitrated_best_utility"))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        winner = scored_candidates[0]
        chosen = dict(winner[1])
        chosen["arbitration_reason"] = winner[2]
        chosen["arbitrated_score"] = round(winner[0], 3)

        return chosen
