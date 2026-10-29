"""Unified Cognitive State Model for E.D.I.T.H. V5.

Defines a strongly typed, persistent, versioned representation of E.D.I.T.H.'s
internal cognitive, affective, motivational, and operational condition.
"""

from __future__ import annotations
import os
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict


@dataclass
class PerceptionBeliefState:
    """Perception and belief tracking."""
    current_interpretation: str = ""
    entities: Dict[str, Any] = field(default_factory=dict)
    belief_confidence: float = 1.0  # 0.0 to 1.0
    competing_interpretations: List[Dict[str, Any]] = field(default_factory=list)
    observed_events: List[str] = field(default_factory=list)
    inferred_events: List[str] = field(default_factory=list)
    prior_expectations: Dict[str, Any] = field(default_factory=dict)
    detected_contradictions: List[str] = field(default_factory=list)


@dataclass
class GoalMotivationState:
    """Goals, intentions, and motivational drives."""
    active_goal: Optional[str] = None
    desired_outcome: Optional[str] = None
    intermediate_objectives: List[str] = field(default_factory=list)
    priority: float = 0.5            # 0.0 (low) to 1.0 (critical)
    urgency: float = 0.5             # 0.0 (relaxed) to 1.0 (immediate)
    progress: float = 0.0            # 0.0 (not started) to 1.0 (completed)
    perceived_effort: float = 0.3    # 0.0 to 1.0
    curiosity_drive: float = 0.5     # Information-seeking motivation (0.0 to 1.0)
    completion_drive: float = 0.7    # Drive to finish active goal (0.0 to 1.0)
    explore_vs_exploit: float = 0.4  # 0.0 (exploit routine) to 1.0 (explore alternative)
    clarification_need: float = 0.0  # Need for user clarification (0.0 to 1.0)
    long_term_commitments: List[str] = field(default_factory=list)


@dataclass
class CognitiveConditionState:
    """Cognitive condition, capacity, attention, and metacognition."""
    attention_focus: Optional[str] = None
    working_memory_occupancy: float = 0.0  # 0.0 to 1.0 capacity ratio
    task_complexity: float = 0.2           # 0.0 (trivial) to 1.0 (deep multi-step)
    unresolved_questions: List[str] = field(default_factory=list)
    cognitive_load: float = 0.2            # Estimated mental workload (0.0 to 1.0)
    plan_confidence: float = 0.9           # Confidence in current active strategy (0.0 to 1.0)
    prediction_error: float = 0.0          # Magnitude of outcome violation (0.0 to 1.0)
    error_accumulation: float = 0.0        # Cumulative recent failures (0.0 to 1.0)
    task_switching_cost: float = 0.1       # Inertia penalty for switching tasks
    inhibition_active: bool = False        # Inhibit risky/unverified action
    deep_reasoning_needed: bool = False    # Whether deeper deliberation is triggered


@dataclass
class ContinuousAffectState:
    """Multi-dimensional continuous affect representation."""
    valence: float = 0.2      # [-1.0 (very negative) to +1.0 (very positive)]
    arousal: float = 0.3      # [0.0 (calm/dormant) to 1.0 (highly activated/alert)]
    dominance: float = 0.7    # [0.0 (helpless/no control) to 1.0 (in control/confident)]
    uncertainty: float = 0.1  # [0.0 (completely certain) to 1.0 (highly uncertain)]
    frustration: float = 0.0  # [0.0 (none) to 1.0 (high)]
    engagement: float = 0.6   # [0.0 (disengaged) to 1.0 (immersed)]
    social_trust: float = 0.8 # [0.0 (guarded) to 1.0 (collaborative/warm)]

    def clamp(self):
        """Clamp all dimensions to their bounded valid ranges."""
        self.valence = max(-1.0, min(1.0, float(self.valence)))
        self.arousal = max(0.0, min(1.0, float(self.arousal)))
        self.dominance = max(0.0, min(1.0, float(self.dominance)))
        self.uncertainty = max(0.0, min(1.0, float(self.uncertainty)))
        self.frustration = max(0.0, min(1.0, float(self.frustration)))
        self.engagement = max(0.0, min(1.0, float(self.engagement)))
        self.social_trust = max(0.0, min(1.0, float(self.social_trust)))

    def to_vector(self) -> List[float]:
        return [
            self.valence,
            self.arousal,
            self.dominance,
            self.uncertainty,
            self.frustration,
            self.engagement,
            self.social_trust
        ]

    @classmethod
    def from_vector(cls, v: List[float]) -> ContinuousAffectState:
        obj = cls(
            valence=v[0] if len(v) > 0 else 0.0,
            arousal=v[1] if len(v) > 1 else 0.3,
            dominance=v[2] if len(v) > 2 else 0.7,
            uncertainty=v[3] if len(v) > 3 else 0.1,
            frustration=v[4] if len(v) > 4 else 0.0,
            engagement=v[5] if len(v) > 5 else 0.5,
            social_trust=v[6] if len(v) > 6 else 0.8,
        )
        obj.clamp()
        return obj

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ContinuousAffectState:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class UserSocialEstimate:
    """Estimated emotional, cognitive, and communicative state of the user."""
    detected_user_mood: str = "neutral"
    user_valence: float = 0.0
    user_arousal: float = 0.3
    confidence_in_user_estimate: float = 0.7
    perceived_user_goal: Optional[str] = None
    communicative_need: str = "execution"  # execution, advice, emotional_support, chat, clarification
    preferred_directness: float = 0.7     # 0.0 (gentle/exploratory) to 1.0 (concise/direct)
    prior_user_attempts: List[str] = field(default_factory=list)


@dataclass
class OperationalResourceState:
    """Observable software system condition and computational budget."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    task_queue_depth: int = 0
    recent_latency_ms: float = 0.0
    retry_count: int = 0
    consecutive_failures: int = 0
    available_compute_budget: float = 1.0  # 0.0 (throttled) to 1.0 (free)


@dataclass
class CognitiveState:
    """The master versioned cognitive state of E.D.I.T.H."""
    version: int = 1
    timestamp: float = field(default_factory=time.time)
    session_id: str = "default_session"

    # Core cognitive sub-states
    perception: PerceptionBeliefState = field(default_factory=PerceptionBeliefState)
    goals: GoalMotivationState = field(default_factory=GoalMotivationState)
    condition: CognitiveConditionState = field(default_factory=CognitiveConditionState)

    # Affective dual-scale dynamics
    momentary_affect: ContinuousAffectState = field(default_factory=ContinuousAffectState)
    baseline_mood: ContinuousAffectState = field(
        default_factory=lambda: ContinuousAffectState(
            valence=0.25, arousal=0.25, dominance=0.75, uncertainty=0.1, frustration=0.0, engagement=0.6, social_trust=0.85
        )
    )

    # Social and operational estimates
    user_estimate: UserSocialEstimate = field(default_factory=UserSocialEstimate)
    operational: OperationalResourceState = field(default_factory=OperationalResourceState)

    # Flow proxy state
    in_flow_state: bool = False
    flow_intensity: float = 0.0  # 0.0 to 1.0

    # Diagnostic provenance
    last_appraisal_trace: Dict[str, Any] = field(default_factory=dict)
    state_update_reason: str = "initialized"

    def advance_version(self, reason: str = ""):
        self.version += 1
        self.timestamp = time.time()
        self.state_update_reason = reason
        self.momentary_affect.clamp()
        self.baseline_mood.clamp()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state safely to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CognitiveState:
        """Instantiate state from dictionary with safe defaults."""
        cs = cls(
            version=data.get("version", 1),
            timestamp=data.get("timestamp", time.time()),
            session_id=data.get("session_id", "default_session"),
            in_flow_state=data.get("in_flow_state", False),
            flow_intensity=data.get("flow_intensity", 0.0),
            last_appraisal_trace=data.get("last_appraisal_trace", {}),
            state_update_reason=data.get("state_update_reason", "loaded"),
        )
        if "perception" in data:
            cs.perception = PerceptionBeliefState(**data["perception"])
        if "goals" in data:
            cs.goals = GoalMotivationState(**data["goals"])
        if "condition" in data:
            cs.condition = CognitiveConditionState(**data["condition"])
        if "momentary_affect" in data:
            cs.momentary_affect = ContinuousAffectState(**data["momentary_affect"])
        if "baseline_mood" in data:
            cs.baseline_mood = ContinuousAffectState(**data["baseline_mood"])
        if "user_estimate" in data:
            cs.user_estimate = UserSocialEstimate(**data["user_estimate"])
        if "operational" in data:
            cs.operational = OperationalResourceState(**data["operational"])

        cs.momentary_affect.clamp()
        cs.baseline_mood.clamp()
        return cs

    def save_snapshot(self, path: str):
        """Save durable snapshot of cognitive state to JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load_snapshot(cls, path: str) -> Optional[CognitiveState]:
        """Load durable snapshot from JSON file."""
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return cls.from_dict(data)
            except Exception:
                return None
        return None
