"""Multi-System Memory Architecture for E.D.I.T.H. V5.

Implements clear functional memory separation inspired by human cognitive science:
1. Working Memory (bounded capacity, active items, current goal).
2. Episodic Memory (contextual episodes: goal, action, outcome, prediction error).
3. Semantic Memory (verified facts, beliefs, entity knowledge with provenance).
4. Procedural Memory (reusable workflows, success rates, recovery recipes).
5. Prospective Memory (commitments, intentions, conditional triggers).
6. Autobiographical Task History (verifiable record of agent's own actions).
7. Consolidation & Selective Forgetting (salience-based promotion & decay).
"""

from __future__ import annotations
import os
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict


@dataclass
class WorkingMemoryItem:
    """An active item in working memory."""
    key: str
    value: Any
    salience: float = 1.0       # 0.0 to 1.0 importance
    inserted_at: float = field(default_factory=time.time)
    item_type: str = "fact"     # 'goal', 'fact', 'question', 'step', 'evidence'


@dataclass
class EpisodicTrace:
    """A rich episodic memory of an interaction or task episode."""
    episode_id: str
    timestamp: float
    goal: Optional[str]
    user_input: str
    action_taken: str
    outcome_status: str        # 'success', 'failure', 'interrupted', 'needs_clarification'
    verified: bool
    prediction_error: float
    appraisal_congruence: float
    salience: float = 0.5
    lesson_learned: Optional[str] = None
    context_tags: List[str] = field(default_factory=list)


@dataclass
class SemanticFact:
    """A verified factual belief with provenance and confidence."""
    subject: str
    predicate: str
    target: Any
    confidence: float = 1.0
    provenance: str = "user_stated"  # 'user_stated', 'system_observed', 'inferred'
    verified: bool = True
    updated_at: float = field(default_factory=time.time)


@dataclass
class ProceduralSkill:
    """A learned workflow or procedure with reliability statistics."""
    skill_name: str
    goal_pattern: str
    action_sequence: List[str]
    success_count: int = 1
    failure_count: int = 0
    common_failure_reasons: List[str] = field(default_factory=list)
    recovery_strategy: Optional[str] = None
    last_used: float = field(default_factory=time.time)

    @property
    def reliability(self) -> float:
        total = self.success_count + self.failure_count
        return round(self.success_count / total, 3) if total > 0 else 0.5


@dataclass
class ProspectiveCommitment:
    """A future commitment or conditional trigger."""
    commitment_id: str
    condition_description: str
    intended_action: str
    created_at: float = field(default_factory=time.time)
    due_timestamp: Optional[float] = None
    fulfilled: bool = False


class MultiSystemMemory:
    """Unified coordinator managing all memory tiers, retrieval, and consolidation."""

    WORKING_MEMORY_CAPACITY = 7  # Miller's Law bound
    MAX_EPISODIC_ENTRIES = 200
    MAX_AUTOBIOGRAPHICAL_LOG = 300

    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = storage_dir or os.path.join("config", "cognitive_memory")
        os.makedirs(self.storage_dir, exist_ok=True)

        # 1. Working Memory (in-memory only, strictly bounded)
        self.working_memory: List[WorkingMemoryItem] = []

        # 2. Episodic Memory
        self.episodic_memory: List[EpisodicTrace] = []

        # 3. Semantic Memory
        self.semantic_memory: Dict[str, SemanticFact] = {}

        # 4. Procedural Memory
        self.procedural_memory: Dict[str, ProceduralSkill] = {}

        # 5. Prospective Memory
        self.prospective_memory: List[ProspectiveCommitment] = []

        # 6. Autobiographical Task History
        self.autobiographical_history: List[Dict[str, Any]] = []

        # Load persisted knowledge
        self.load_consolidated_memory()

    # -------------------------------------------------------------------------
    # Tier 1: Working Memory Management
    # -------------------------------------------------------------------------
    def set_working_item(self, key: str, value: Any, salience: float = 1.0, item_type: str = "fact"):
        """Insert or update an active working memory item with bounded capacity eviction."""
        # Update existing
        for item in self.working_memory:
            if item.key == key:
                item.value = value
                item.salience = salience
                item.inserted_at = time.time()
                return

        # Check capacity
        if len(self.working_memory) >= self.WORKING_MEMORY_CAPACITY:
            # Evict item with lowest salience / oldest recency
            self.working_memory.sort(key=lambda x: (x.salience, x.inserted_at))
            self.working_memory.pop(0)

        self.working_memory.append(WorkingMemoryItem(
            key=key, value=value, salience=salience, item_type=item_type
        ))

    def get_working_item(self, key: str) -> Optional[Any]:
        for item in self.working_memory:
            if item.key == key:
                return item.value
        return None

    def clear_working_memory(self):
        self.working_memory.clear()

    # -------------------------------------------------------------------------
    # Tier 2: Episodic Memory Management
    # -------------------------------------------------------------------------
    def record_episode(
        self,
        goal: Optional[str],
        user_input: str,
        action_taken: str,
        outcome_status: str,
        verified: bool = True,
        prediction_error: float = 0.0,
        appraisal_congruence: float = 0.0,
        lesson_learned: Optional[str] = None,
        context_tags: Optional[List[str]] = None
    ) -> EpisodicTrace:
        """Record an episodic trace of an action and its outcome."""
        ep_id = f"ep_{int(time.time() * 1000)}"
        # Salience is higher for surprising outcomes or prediction errors
        salience = min(1.0, 0.4 + 0.6 * prediction_error + (0.3 if not verified else 0.0))

        trace = EpisodicTrace(
            episode_id=ep_id,
            timestamp=time.time(),
            goal=goal,
            user_input=user_input,
            action_taken=action_taken,
            outcome_status=outcome_status,
            verified=verified,
            prediction_error=prediction_error,
            appraisal_congruence=appraisal_congruence,
            salience=salience,
            lesson_learned=lesson_learned,
            context_tags=context_tags or []
        )
        self.episodic_memory.append(trace)
        if len(self.episodic_memory) > self.MAX_EPISODIC_ENTRIES:
            # Keep highest salience episodes and drop lowest
            self.episodic_memory.sort(key=lambda x: (x.salience, x.timestamp))
            self.episodic_memory.pop(0)

        return trace

    def retrieve_relevant_episodes(self, query: str, limit: int = 3) -> List[EpisodicTrace]:
        """Retrieve most relevant past episodes matching query words and high salience."""
        q_tokens = set(query.lower().split())
        scored: List[Tuple[float, EpisodicTrace]] = []
        for ep in self.episodic_memory:
            ep_tokens = set(ep.user_input.lower().split() + (ep.goal.lower().split() if ep.goal else []))
            overlap = len(q_tokens.intersection(ep_tokens))
            if overlap > 0:
                score = overlap * 1.5 + ep.salience * 2.0
                scored.append((score, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:limit]]

    # -------------------------------------------------------------------------
    # Tier 3: Semantic Memory Management
    # -------------------------------------------------------------------------
    def store_fact(
        self,
        subject: str,
        predicate: str,
        target: Any,
        confidence: float = 1.0,
        provenance: str = "user_stated"
    ):
        """Store or update a verified semantic belief."""
        # Sanitize sensitive data from persistent storage
        if any(term in str(target).lower() for term in ["password", "secret", "token"]):
            return

        key = f"{subject.strip().lower()}::{predicate.strip().lower()}"
        self.semantic_memory[key] = SemanticFact(
            subject=subject,
            predicate=predicate,
            target=target,
            confidence=confidence,
            provenance=provenance,
            verified=True,
            updated_at=time.time()
        )

    def query_fact(self, subject: str, predicate: str) -> Optional[Any]:
        key = f"{subject.strip().lower()}::{predicate.strip().lower()}"
        fact = self.semantic_memory.get(key)
        return fact.target if fact else None

    # -------------------------------------------------------------------------
    # Tier 4: Procedural Memory (Skills & Recipes)
    # -------------------------------------------------------------------------
    def register_procedure_outcome(
        self,
        skill_name: str,
        goal_pattern: str,
        action_sequence: List[str],
        success: bool,
        failure_reason: Optional[str] = None,
        recovery_strategy: Optional[str] = None
    ):
        """Update procedural skill execution reliability based on real feedback."""
        if skill_name in self.procedural_memory:
            skill = self.procedural_memory[skill_name]
            if success:
                skill.success_count += 1
            else:
                skill.failure_count += 1
                if failure_reason and failure_reason not in skill.common_failure_reasons:
                    skill.common_failure_reasons.append(failure_reason)
            if recovery_strategy:
                skill.recovery_strategy = recovery_strategy
            skill.last_used = time.time()
        else:
            self.procedural_memory[skill_name] = ProceduralSkill(
                skill_name=skill_name,
                goal_pattern=goal_pattern,
                action_sequence=action_sequence,
                success_count=1 if success else 0,
                failure_count=0 if success else 1,
                common_failure_reasons=[failure_reason] if failure_reason else [],
                recovery_strategy=recovery_strategy,
                last_used=time.time()
            )

    def get_procedure(self, skill_name: str) -> Optional[ProceduralSkill]:
        """Retrieve procedural skill record regardless of reliability threshold."""
        return self.procedural_memory.get(skill_name)

    def get_reliable_procedure(self, skill_name: str) -> Optional[ProceduralSkill]:
        skill = self.procedural_memory.get(skill_name)
        if skill and skill.reliability >= 0.5:
            return skill
        return None

    # -------------------------------------------------------------------------
    # Tier 5: Prospective Commitments
    # -------------------------------------------------------------------------
    def add_commitment(self, condition: str, action: str, due_timestamp: Optional[float] = None) -> str:
        cid = f"comm_{int(time.time() * 1000)}"
        self.prospective_memory.append(ProspectiveCommitment(
            commitment_id=cid,
            condition_description=condition,
            intended_action=action,
            due_timestamp=due_timestamp,
            fulfilled=False
        ))
        return cid

    def check_pending_commitments(self) -> List[ProspectiveCommitment]:
        return [c for c in self.prospective_memory if not c.fulfilled]

    def fulfill_commitment(self, commitment_id: str):
        for c in self.prospective_memory:
            if c.commitment_id == commitment_id:
                c.fulfilled = True

    # -------------------------------------------------------------------------
    # Tier 6: Autobiographical Task History
    # -------------------------------------------------------------------------
    def log_autobiographical_action(self, action: str, target: str, result: str, verified: bool):
        entry = {
            "timestamp": time.time(),
            "action": action,
            "target": target,
            "result": result,
            "verified": verified
        }
        self.autobiographical_history.append(entry)
        if len(self.autobiographical_history) > self.MAX_AUTOBIOGRAPHICAL_LOG:
            self.autobiographical_history.pop(0)

    def forget_fact(self, subject: str, predicate: str) -> bool:
        """Explicitly forget and delete a stored semantic fact."""
        key = f"{subject.strip().lower()}::{predicate.strip().lower()}"
        if key in self.semantic_memory:
            del self.semantic_memory[key]
            self.consolidate_and_save()
            return True
        return False

    # -------------------------------------------------------------------------
    # Tier 7: Consolidation, Forgetting, and Persistence
    # -------------------------------------------------------------------------
    def consolidate_and_save(self):
        """Persist semantic facts, procedural skills, commitments, and episodic memory safely."""
        now = time.time()
        sem_path = os.path.join(self.storage_dir, "semantic_facts.json")
        proc_path = os.path.join(self.storage_dir, "procedural_skills.json")
        comm_path = os.path.join(self.storage_dir, "commitments.json")
        ep_path = os.path.join(self.storage_dir, "episodic_memory.json")

        try:
            with open(sem_path, "w", encoding="utf-8") as f:
                json.dump({k: asdict(v) for k, v in self.semantic_memory.items()}, f, indent=2)
            with open(proc_path, "w", encoding="utf-8") as f:
                json.dump({k: asdict(v) for k, v in self.procedural_memory.items()}, f, indent=2)

            # Filter fulfilled commitments and commitments expired by retention policy (>24h overdue)
            valid_commitments = [
                asdict(c) for c in self.prospective_memory
                if not c.fulfilled and (c.due_timestamp is None or c.due_timestamp > (now - 86400.0))
            ]
            with open(comm_path, "w", encoding="utf-8") as f:
                json.dump(valid_commitments, f, indent=2)

            # Persist top salience episodic traces (up to 100)
            sorted_episodes = sorted(self.episodic_memory, key=lambda e: (e.salience, e.timestamp))
            with open(ep_path, "w", encoding="utf-8") as f:
                json.dump([asdict(e) for e in sorted_episodes[-100:]], f, indent=2)
        except Exception:
            pass

    def load_consolidated_memory(self):
        sem_path = os.path.join(self.storage_dir, "semantic_facts.json")
        proc_path = os.path.join(self.storage_dir, "procedural_skills.json")
        comm_path = os.path.join(self.storage_dir, "commitments.json")
        ep_path = os.path.join(self.storage_dir, "episodic_memory.json")

        if os.path.exists(sem_path):
            try:
                with open(sem_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for k, v in data.items():
                        self.semantic_memory[k] = SemanticFact(**v)
            except Exception:
                pass

        if os.path.exists(proc_path):
            try:
                with open(proc_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for k, v in data.items():
                        self.procedural_memory[k] = ProceduralSkill(**v)
            except Exception:
                pass

        if os.path.exists(comm_path):
            try:
                with open(comm_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for v in data:
                        self.prospective_memory.append(ProspectiveCommitment(**v))
            except Exception:
                pass

        if os.path.exists(ep_path):
            try:
                with open(ep_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.episodic_memory = [EpisodicTrace(**v) for v in data]
            except Exception:
                pass
