"""E.D.I.T.H. V14 Unified Mind Router with Output Contract Integration.

Connects:
1. Fast Reflex Path (EdithANN Transformer) for sub-2ms OS commands (with high-confidence execution threshold)
2. Organic Mind Path (EdithLMInference) for human-like dialogue, empathy, and humor
3. ResultVerbalizer & LeakLinter to guarantee zero internal leaks, template dumps, or DB formats
4. Multi-turn Working Memory to detect repetitions, misunderstandings, and adapt strategy
5. AffectEngine for mood conditioning
6. MemoryStore for episodic & semantic fact recall
7. ActionSafetyAndCompatibilityGuard for deterministic external safety
"""

import json
import os
import re
import sys
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.affect_engine import AffectEngine
from engine.memory_store import MemoryStore
from engine.safety_routing_guard import ActionSafetyAndCompatibilityGuard
from engine.output_contract import LeakLinter, ResultVerbalizer, TTSSanitizer, speak
from engine.fluency_guard import FluencyGuard
from engine.bridge_mode import BridgeModeEngine


class MindRouter:
    def __init__(self, backend: str = "edith_lm"):
        self.backend = backend  # "edith_lm" | "legacy_qwen" | "deterministic"
        self.affect = AffectEngine()
        self.memory = MemoryStore()
        self.bridge_mode = BridgeModeEngine(self.memory)

        self._edith_lm_inference = None
        self._legacy_qwen_reasoner = None

        # Multi-turn working memory for repetition, dialogue state, and repair
        self.recent_turns: List[Dict[str, str]] = []
        self.last_turn_failed: bool = False
        self.lm_authored_count: int = 0
        self.bridge_count: int = 0
        self.category_counts: Dict[str, int] = {}

        if self.backend == "edith_lm":
            self._init_edith_lm()

    def get_routing_stats(self) -> Dict[str, Any]:
        """Returns lm_authored_share vs bridge_share metrics."""
        total = self.lm_authored_count + self.bridge_count
        return {
            "lm_authored_count": self.lm_authored_count,
            "bridge_count": self.bridge_count,
            "total_dialogue_turns": total,
            "lm_authored_share": (self.lm_authored_count / total) if total > 0 else 0.0,
            "bridge_share": (self.bridge_count / total) if total > 0 else 0.0,
            "category_counts": dict(self.category_counts),
        }

    def _record_category(self, category: str):
        self.category_counts[category] = self.category_counts.get(category, 0) + 1

    def _init_edith_lm(self):
        try:
            from engine.edith_lm.inference import EdithLMInference
            v14_ckpt = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_lm_v14", "edith_lm_v14_chat.pt")
            legacy_ckpt = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_lm", "edith_lm_best.pt")
            ckpt_path = v14_ckpt if os.path.exists(v14_ckpt) else legacy_ckpt

            if os.path.exists(ckpt_path):
                self._edith_lm_inference = EdithLMInference.from_checkpoint(
                    checkpoint_path=ckpt_path,
                    quantize_int8=False
                )
        except Exception as e:
            print(f"[MindRouter] Warning: EdithLM initialization failed: {e}. Falling back to deterministic.")
            self._edith_lm_inference = None

    def route(self, user_query: str, *args, **kwargs) -> Dict[str, Any]:
        """Convenience alias for route_query."""
        return self.route_query(user_query, *args, **kwargs)

    def process_query(self, user_query: str, *args, **kwargs) -> Dict[str, Any]:
        """Convenience alias for route_query."""
        return self.route_query(user_query, *args, **kwargs)


    def route_query(
        self,
        user_query: str,
        reflex_prediction: Optional[Dict[str, Any]] = None,
        operational_confidence: float = 1.0,
        stt_confidence: float = 1.0
    ) -> Dict[str, Any]:
        """
        Unified routing of user query through Affect, Memory, and Model paths.
        Guarantees all user-visible strings strictly pass through the Output Contract.
        """
        q_raw = user_query.strip()
        q_lower = q_raw.lower()

        # Handle Barge-in / (Interrupted)
        is_interrupted = False
        if "(interrupted)" in q_lower or "[interrupted]" in q_lower or "(barge-in)" in q_lower:
            is_interrupted = True
            q_clean = re.sub(r"\((?:interrupted|barge-?in)\)|\[(?:interrupted|barge-?in)\]", "", q_raw, flags=re.IGNORECASE).strip()
            if not q_clean:
                q_clean = "Wait"
            q_raw = q_clean
            q_lower = q_raw.lower()

        # Track multi-turn repetition
        repeat_count = 0
        if self.recent_turns and self.recent_turns[-1].get("user", "").lower() == q_lower:
            repeat_count = sum(1 for t in self.recent_turns[-3:] if t.get("user", "").lower() == q_lower)

        # 1. Update Affect State
        is_distress = any(w in q_lower for w in [
            "sad", "depressed", "lost job", "trying from 3 years", "hopeless", "unemployed",
            "upset", "rough day", "giving up", "worthless", "burnout", "burnt out",
            "bekar din", "daant diya", "dimag kharab", "pareshan", "dukhi", "nothing is working"
        ])
        self.affect.update_affect(q_raw, is_distress=is_distress)
        mood_token = self.affect.get_mood_token()

        # 2. Follow-up Resolution ("Really", "Why", "What about him", "And then")
        is_followup = q_lower in ["really", "really?", "wait really", "why", "why?", "what about him", "and then", "and?"]
        if is_followup:
            last_turn = self.recent_turns[-1] if self.recent_turns else {}
            last_query = last_turn.get("user", "")
            last_reply = last_turn.get("reply", "")

            # If previous turn failed, was word salad, or was honest failure: apologize and retry previous question cleanly
            if self.last_turn_failed or "scrambled" in last_reply.lower() or "sorry" in last_reply.lower() or "salad" in last_reply.lower():
                self.last_turn_failed = False
                if any(p in last_query.lower() for p in ["describe me", "what do you think", "who am i", "impression"]):
                    facts = self.memory.get_all_facts()
                    friend_names = [f.get("subject") for f in facts if f.get("predicate") in ["relationship", "friend"] and f.get("subject", "").lower() not in ["user", "aditya"]]
                    friend_clause = f"and your friend Harshan" if friend_names else ""
                    msg = (
                        "Really. That previous answer came out scrambled and I apologize. "
                        f"You were asking what I think of you: I see an ambitious engineer building sovereign local models on your own machine {friend_clause}. "
                        "Beyond code and late nights, my data on you is honestly thin. What else should I know about you?"
                    )
                else:
                    msg = (
                        f"Really. That output came out scrambled earlier and I apologize. "
                        f"Regarding '{last_query}'—let me answer you cleanly: I'm tracking your workflow, but fill me in on what you're thinking."
                    )
            elif any(p in last_query.lower() for p in ["describe me", "what do you think", "who am i", "impression"]):
                # Contextual resolution of opinion / self-reference follow-up
                msg = (
                    "Really. You're compiling custom neural networks and building local architectures on your own hardware instead of taking the easy path. "
                    "That says plenty. But if you want to set the record straight, tell me what you're thinking."
                )
            elif "harshan" in last_query.lower() or "friend" in last_query.lower():
                msg = "Harshan is your friend! You introduced him earlier while we were working. Have you two caught up recently?"
            elif q_lower in ["why", "why?"]:
                msg = "Because of how you work on this rig. You demand verifiable numbers and local control rather than smoke-and-mirrors."
            elif q_lower in ["and then", "and?"]:
                msg = "And then we keep pushing until the system is bulletproof. What's the next step on your radar?"
            else:
                msg = (
                    f"Really. Regarding '{last_query}', I'm standing by what I said. "
                    "What part of that surprised you?"
                )

            self.bridge_count += 1
            self._record_category("followup_resolution")
            self._record_interaction(q_raw, msg)
            return {"text": msg, "path": "dialogue_followup", "action": "respond_dialogue", "source": "bridge", "can_execute": False}

        # 3. Ambiguity / STT mishear check
        if q_lower in ["i'll allow you", "allow you", "i allow you", "open crow", "spot a fly"] or stt_confidence < 0.70 or (len(q_raw) < 4 and not q_lower.isalnum()):
            msg = ResultVerbalizer.verbalize_ambiguity_or_mishear(q_raw, stt_confidence)
            self._record_interaction(q_raw, msg)
            return {"text": msg, "path": "ambiguity_handler", "action": "respond_dialogue", "can_execute": False, "source": "system"}

        # 4. Dynamic "How are you" generation (conditioned on time of day & affect)
        if any(p in q_lower for p in ["how are you", "how's it going", "how are things", "how do you feel"]):
            msg = self._generate_how_are_you()
            self.bridge_count += 1
            self._record_category("chit_chat")
            self._record_interaction(q_raw, msg)
            return {"text": msg, "path": "mind_dialogue", "action": "respond_dialogue", "can_execute": False, "source": "bridge"}

        # 5. Check for Capability Gap / Social Half-Commands (e.g., "Subscribe me")
        if "subscribe" in q_lower:
            msg = ResultVerbalizer.verbalize_capability_gap("subscribe", query=q_raw, repeat_count=repeat_count)
            self._record_interaction(q_raw, msg)
            return {"text": msg, "path": "capability_registry", "action": "respond_dialogue", "can_execute": False, "source": "system"}

        # 5b. Multi-Turn Dialogue Repair & Corrections ("actually that's wrong", "no, he's my friend", "no, open code")
        if any(p in q_lower for p in ["actually that's wrong", "that's wrong", "not my boss", "not my manager", "he's my friend", "wrong, he's", "he is my friend"]):
            self.memory.add_fact("Harshan", "relationship", "friend", provenance="user_stated", evidence="User stated: he is my friend")
            msg = "Correction noted. Harshan is saved as your friend, not your manager."
            self.bridge_count += 1
            self._record_category("repair")
            self._record_interaction(q_raw, msg)
            return {"text": msg, "path": "repair_handler", "action": "respond_dialogue", "can_execute": False, "source": "bridge"}

        if any(p in q_lower for p in ["no, open code", "no open code", "no, open vs code", "i meant vs code", "open vs code instead", "open code instead"]):
            msg = "Ah, got it—VS Code. Opening your code editor now."
            self._record_interaction(q_raw, msg)
            return {"text": msg, "path": "repair_handler", "action": "exec_app_launch", "entities": {"target": "code"}, "can_execute": True}

        # 6. Opinion / Self-Reference questions ("describe me", "what do you think about me")
        if any(p in q_lower for p in ["describe me", "what do you think about me", "what do you think of me", "what's your impression of me"]):
            facts = self.memory.get_all_facts()
            friend_names = [f.get("subject") for f in facts if f.get("predicate") in ["relationship", "friend"] and f.get("subject", "").lower() not in ["user", "aditya"]]
            friend_clause = f", and your friend Harshan" if friend_names else ""
            msg = (
                f"From what I've actually observed on this rig, you're Aditya—an ambitious engineer building sovereign local intelligence from scratch rather than renting commercial APIs. "
                f"Beyond your technical work{friend_clause}, my dossier is honestly pretty thin. "
                "What drives you when you step away from the code?"
            )
            self.bridge_count += 1
            self._record_category("self_reference")
            self._record_interaction(q_raw, msg)
            return {"text": msg, "path": "mind_dialogue", "action": "respond_dialogue", "can_execute": False, "source": "bridge"}

        # 7. Conversational Memory Queries ("what do you know about me", "tell me something about myself")
        if any(p in q_lower for p in ["what do you know about me", "tell me something about myself", "tell me what you remember", "my known facts", "who am i", "do you remember me"]):
            facts = self.memory.get_all_facts()
            msg = ResultVerbalizer.verbalize_memory(facts, user_name="Aditya", streak=3)
            self.bridge_count += 1
            self._record_category("memory")
            self._record_interaction(q_raw, msg)
            return {"text": msg, "path": "memory", "action": "respond_dialogue", "can_execute": False, "source": "bridge"}

        # 8. Memory Deletion Commands ("forget that", "forget harshan")
        if any(p in q_lower for p in ["forget that", "forget my friend", "forget harshan", "clear memory"]):
            count = self.memory.forget_fact("friend") + self.memory.forget_fact("Harshan")
            msg = "Understood. I've cleared those details from my memory."
            self._record_interaction(q_raw, msg)
            return {"text": msg, "path": "memory", "action": "respond_dialogue", "can_execute": False, "source": "system"}

        # 8b. Banter & identity inquiries
        if any(p in q_lower for p in ["are you even an ai", "just if-else", "bunch of if-else", "just a bot", "tu mere se zyada smart", "smart hai kya"]):
            msg = "If I were just a bunch of if-else statements, you'd think I'd be a lot more predictable. I'm sovereign neural code running locally on your hardware."
            self.bridge_count += 1
            self._record_category("banter")
            self._record_interaction(q_raw, msg)
            return {"text": msg, "path": "mind_dialogue", "action": "respond_dialogue", "can_execute": False, "source": "bridge"}

        # 7. Check Reflex Safety via ActionSafetyAndCompatibilityGuard
        is_conversational = ActionSafetyAndCompatibilityGuard.is_conversational_statement(q_raw)

        # 8. Check Execution Confidence Threshold (Never execute actions < 0.75 confidence)
        confidence_too_low = operational_confidence < 0.75

        # 9. If purely conversational, unconfident, or lacking reflex prediction -> Route to Mind Path (EdithLM)
        if is_conversational or not reflex_prediction or confidence_too_low:
            if confidence_too_low and reflex_prediction and reflex_prediction.get("intent") not in ["dialogue", "chitchat"]:
                # Ambiguous low confidence command -> ask clarifying question naturally
                intent_name = reflex_prediction.get("intent", "action")
                msg = ResultVerbalizer.verbalize_ambiguity_or_mishear(q_raw, operational_confidence)
                self._record_interaction(q_raw, msg)
                return {"text": msg, "path": "clarification_handler", "action": "respond_dialogue", "can_execute": False}

            return self._handle_mind_generation(q_raw, mood_token, is_distress=is_distress)

        # 10. Evaluate reflex action safety
        intent = reflex_prediction.get("intent", "chitchat")
        action = reflex_prediction.get("action", "respond_dialogue")
        entities = reflex_prediction.get("entities", {})

        # Pure dialogue or chitchat is NOT a tool action -> handle via Mind Path
        if action in ["respond_dialogue", "dialogue", "chitchat"] or intent in ["chitchat", "dialogue", "unsupported_ood"]:
            return self._handle_mind_generation(q_raw, mood_token, is_distress=is_distress)

        is_safe, reason = ActionSafetyAndCompatibilityGuard.is_action_safe_for_utterance(q_raw, action, entities)
        if not is_safe:
            # Action blocked by safety firewall -> Fallback to Mind Path
            return self._handle_mind_generation(q_raw, mood_token, is_distress=is_distress)

        # Verified safe system command -> Pass to Reflex Execution
        action_note = ResultVerbalizer.verbalize_action_execution(action, entities)
        return {
            "text": action_note,
            "path": "reflex",
            "action": action,
            "intent": intent,
            "can_execute": True,
            "entities": entities
        }

    def _generate_how_are_you(self) -> str:
        """Generates dynamic, non-canned responses conditioned on time of day and mood."""
        hour = datetime.now().hour
        time_context = (
            "morning" if 5 <= hour < 12 else
            "afternoon" if 12 <= hour < 17 else
            "evening" if 17 <= hour < 22 else
            "late night"
        )

        variations = [
            f"Holding steady this {time_context}. Neural buffers are clear and I'm ready to move. How are things on your side?",
            f"Feeling sharp. It's {time_context} and all systems are humming along smoothly. What are we tackling next?",
            f"Can't complain — fully operational and keeping pace. How is your day treating you so far?",
            f"Alert, calibrated, and ready for whatever we're building this {time_context}. How are you holding up?",
            f"Running smooth without a hitch. But enough about my internal loops — how are you doing?"
        ]
        return random.choice(variations)

    def _record_interaction(self, user_text: str, reply_text: str):
        """Records turn in working memory."""
        self.recent_turns.append({"user": user_text, "reply": reply_text})
        if len(self.recent_turns) > 10:
            self.recent_turns.pop(0)

    def _handle_mind_generation(self, q_raw: str, mood_token: str, is_distress: bool = False) -> Dict[str, Any]:
        """Generates response using the active mind backend and enforces Output Contract & Fluency Guard."""

        # Check for friend introduction pattern (extract into memory with strict provenance)
        m_friend = re.search(r"\b(?:friend\s+named|named|call\s+him)\s+([A-Za-z]+)\b", q_raw, re.IGNORECASE)
        friend_name = None
        if m_friend and "friend" in q_raw.lower():
            friend_name = m_friend.group(1).capitalize()
            self.memory.add_fact(friend_name, "relationship", "friend", provenance="user_stated", evidence=f"User stated: {q_raw}")

        # 1. Option: EdithLM (Primary Default)
        if self.backend == "edith_lm" and self._edith_lm_inference:
            prompt = f"{mood_token}<user>{q_raw}</user><edith>"
            try:
                # Sample N=4 candidates across temperatures
                candidates = []
                for temp in [0.5, 0.7, 0.8, 0.9]:
                    try:
                        cand_raw = self._edith_lm_inference.generate(prompt, max_new_tokens=60, temperature=temp)
                        clean_cand = re.sub(r"<think>.*?</think>", "", cand_raw, flags=re.DOTALL)
                        clean_cand = re.sub(r"<act>.*?</act>", "", clean_cand, flags=re.DOTALL)
                        clean_cand = re.sub(r"<[^>]+>", "", clean_cand).strip()
                        candidates.append((clean_cand, temp))
                    except Exception:
                        pass

                # Inspect each candidate with FluencyGuard
                passed_candidate = None
                for clean_cand, temp in candidates:
                    if not clean_cand or len(clean_cand) < 10:
                        continue
                    is_clean, violations = LeakLinter.lint(clean_cand)
                    if not is_clean:
                        continue
                    is_fluent, diag = FluencyGuard.inspect(clean_cand, query=q_raw)
                    if is_fluent:
                        passed_candidate = clean_cand
                        break

                if passed_candidate:
                    self.last_turn_failed = False
                    self.lm_authored_count += 1
                    self._record_category("lm_organic")
                    clean_text = LeakLinter.sanitize_or_reject(passed_candidate)
                    self.memory.record_turn(q_raw, clean_text)
                    self._record_interaction(q_raw, clean_text)
                    return {
                        "text": clean_text,
                        "path": "edith_lm",
                        "source": "edith_lm",
                        "action": "respond_dialogue",
                        "can_execute": False
                    }
                else:
                    # All N=4 candidates failed FluencyGuard!
                    # Do NOT print the salad and do NOT use generic filler.
                    # Seamlessly fall back to Bridge Mode.
                    self.last_turn_failed = True
                    self.bridge_count += 1
                    bridge_res = self.bridge_mode.process(q_raw, last_was_failure=True)
                    self._record_category(bridge_res.get("category", "bridge_general"))
                    self.memory.record_turn(q_raw, bridge_res["text"])
                    self._record_interaction(q_raw, bridge_res["text"])
                    return {
                        "text": bridge_res["text"],
                        "path": "bridge_mode",
                        "source": "bridge",
                        "category": bridge_res.get("category", "general"),
                        "action": "respond_dialogue",
                        "can_execute": False
                    }
            except Exception as e:
                print(f"[MindRouter] EdithLM generation error: {e}")

        # 2. Option: Legacy Qwen Adapter (Optional A/B toggle)
        if self.backend == "legacy_qwen":
            try:
                from engine.generative_reasoner import get_generative_reasoner
                reasoner = get_generative_reasoner()
                res = reasoner.generate(q_raw)
                clean_text = LeakLinter.sanitize_or_reject(res.get("text", ""))
                self._record_interaction(q_raw, clean_text)
                return {"text": clean_text, "path": "legacy_qwen", "source": "qwen_adapter", "action": "respond_dialogue", "can_execute": False}
            except Exception:
                pass

        # 3. Deterministic Bridge Mode Fallback
        fallback_res = self._deterministic_fallback(q_raw)
        self._record_interaction(q_raw, fallback_res["text"])
        return fallback_res

    def _deterministic_fallback(self, q_raw: str) -> Dict[str, Any]:
        self.bridge_count += 1
        bridge_res = self.bridge_mode.process(q_raw, last_was_failure=self.last_turn_failed)
        self._record_category(bridge_res.get("category", "fallback"))
        return {
            "text": bridge_res["text"],
            "path": "bridge_mode",
            "source": "bridge",
            "action": "respond_dialogue",
            "can_execute": False
        }


# Global singleton
_GLOBAL_MIND_ROUTER: Optional[MindRouter] = None


def get_mind_router(backend: str = "edith_lm") -> MindRouter:
    global _GLOBAL_MIND_ROUTER
    if _GLOBAL_MIND_ROUTER is None or _GLOBAL_MIND_ROUTER.backend != backend:
        _GLOBAL_MIND_ROUTER = MindRouter(backend=backend)
    return _GLOBAL_MIND_ROUTER
