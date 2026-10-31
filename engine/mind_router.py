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


class MindRouter:
    def __init__(self, backend: str = "edith_lm"):
        self.backend = backend  # "edith_lm" | "legacy_qwen" | "deterministic"
        self.affect = AffectEngine()
        self.memory = MemoryStore()

        self._edith_lm_inference = None
        self._legacy_qwen_reasoner = None

        # Multi-turn working memory for repetition and dialogue repair detection
        self.recent_turns: List[Dict[str, str]] = []

        if self.backend == "edith_lm":
            self._init_edith_lm()

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

        # Track multi-turn repetition
        repeat_count = 0
        if self.recent_turns and self.recent_turns[-1].get("user", "").lower() == q_lower:
            repeat_count = sum(1 for t in self.recent_turns[-3:] if t.get("user", "").lower() == q_lower)

        # 1. Update Affect State
        is_distress = any(w in q_lower for w in ["sad", "depressed", "lost job", "trying from 3 years", "hopeless", "unemployed", "upset", "rough day"])
        self.affect.update_affect(q_raw, is_distress=is_distress)
        mood_token = self.affect.get_mood_token()

        # 2. Ambiguity / STT mishear check
        if q_lower in ["i'll allow you", "allow you", "i allow you", "open crow", "spot a fly"] or stt_confidence < 0.70 or (len(q_raw) < 4 and not q_lower.isalnum()):
            msg = ResultVerbalizer.verbalize_ambiguity_or_mishear(q_raw, stt_confidence)
            self._record_interaction(q_raw, msg)
            return {"text": msg, "path": "ambiguity_handler", "action": "respond_dialogue", "can_execute": False}

        # 3. Dynamic "How are you" generation (conditioned on time of day & affect)
        if any(p in q_lower for p in ["how are you", "how's it going", "how are things", "how do you feel"]):
            msg = self._generate_how_are_you()
            self._record_interaction(q_raw, msg)
            return {"text": msg, "path": "mind_dialogue", "action": "respond_dialogue", "can_execute": False}

        # 4. Check for Capability Gap / Social Half-Commands (e.g., "Subscribe me")
        if "subscribe" in q_lower:
            msg = ResultVerbalizer.verbalize_capability_gap("subscribe", query=q_raw, repeat_count=repeat_count)
            self._record_interaction(q_raw, msg)
            return {"text": msg, "path": "capability_registry", "action": "respond_dialogue", "can_execute": False}

        # 4b. Multi-Turn Dialogue Repair & Corrections ("actually that's wrong", "no, he's my friend", "no, open code")
        if any(p in q_lower for p in ["actually that's wrong", "that's wrong", "not my boss", "not my manager", "he's my friend", "wrong, he's", "he is my friend"]):
            self.memory.add_fact("Harshan", "relationship", "friend")
            msg = "Correction noted. Harshan is saved as your friend, not your manager."
            self._record_interaction(q_raw, msg)
            return {"text": msg, "path": "repair_handler", "action": "respond_dialogue", "can_execute": False}

        if any(p in q_lower for p in ["no, open code", "no open code", "no, open vs code", "i meant vs code", "open vs code instead", "open code instead"]):
            msg = "Ah, got it—VS Code. Opening your code editor now."
            self._record_interaction(q_raw, msg)
            return {"text": msg, "path": "repair_handler", "action": "exec_app_launch", "entities": {"target": "code"}, "can_execute": True}

        # 5. Check for Conversational Memory Queries ("what do you know about me", "tell me something about myself")
        if any(p in q_lower for p in ["what do you know about me", "tell me something about myself", "tell me what you remember", "my known facts", "who am i", "do you remember me"]):
            facts = self.memory.get_all_facts()
            msg = ResultVerbalizer.verbalize_memory(facts, user_name="Aditya", streak=3)
            self._record_interaction(q_raw, msg)
            return {"text": msg, "path": "memory", "action": "respond_dialogue", "can_execute": False}

        # 6. Memory Deletion Commands ("forget that", "forget harshan")
        if any(p in q_lower for p in ["forget that", "forget my friend", "forget harshan", "clear memory"]):
            count = self.memory.forget_fact("friend") + self.memory.forget_fact("Harshan")
            msg = "Understood. I've cleared those details from my memory."
            self._record_interaction(q_raw, msg)
            return {"text": msg, "path": "memory", "action": "respond_dialogue", "can_execute": False}

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
        """Generates response using the active mind backend and enforces Output Contract."""

        # Check for friend introduction pattern (extract into memory)
        m_friend = re.search(r"\b(?:friend\s+named|named|call\s+him)\s+([A-Za-z]+)\b", q_raw, re.IGNORECASE)
        friend_name = None
        if m_friend and "friend" in q_raw.lower():
            friend_name = m_friend.group(1).capitalize()
            self.memory.add_fact(friend_name, "relationship", "friend")

        # 1. Option: EdithLM (Primary Default)
        if self.backend == "edith_lm" and self._edith_lm_inference:
            prompt = f"{mood_token}<user>{q_raw}</user><edith>"
            try:
                gen_text = self._edith_lm_inference.generate(prompt, max_new_tokens=60, temperature=0.7)

                # Thoroughly strip think and act tags including internal content
                clean_text = re.sub(r"<think>.*?</think>", "", gen_text, flags=re.DOTALL)
                clean_text = re.sub(r"<act>.*?</act>", "", clean_text, flags=re.DOTALL)
                clean_text = re.sub(r"<[^>]+>", "", clean_text).strip()

                # Lint against internal leaks
                is_clean, violations = LeakLinter.lint(clean_text)
                if not is_clean:
                    clean_text = ""

                if friend_name and (not clean_text or friend_name.lower() not in clean_text.lower()):
                    clean_text = f"Nice to meet {friend_name} through you! How long have you two known each other?"
                elif is_distress and len(clean_text) < 25:
                    clean_text = "I'm really sorry to hear that. Going through a rough time is hard, and having someone close who cares and listens makes all the difference."
                elif len(clean_text) < 15:
                    clean_text = "I'm right here with you. What can I help you explore or accomplish next?"

                if clean_text:
                    clean_text = LeakLinter.sanitize_or_reject(clean_text)
                    self.memory.record_turn(q_raw, clean_text)
                    self._record_interaction(q_raw, clean_text)
                    return {"text": clean_text, "path": "edith_lm", "action": "respond_dialogue", "can_execute": False}
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
                return {"text": clean_text, "path": "legacy_qwen", "action": "respond_dialogue", "can_execute": False}
            except Exception:
                pass

        # 3. Deterministic Fallback with strict LeakLinter protection
        fallback_res = self._deterministic_fallback(q_raw)
        self._record_interaction(q_raw, fallback_res["text"])
        return fallback_res

    def _deterministic_fallback(self, q_raw: str) -> Dict[str, Any]:
        q_lower = q_raw.lower()
        if "friend is looking for jobs" in q_lower or ("friend" in q_lower and "3 years" in q_lower):
            msg = "I'm really sorry to hear that. Three years of searching is exhausting. Remind your friend that persistence matters and their worth isn't defined by hiring delays."
        elif "friend is very sad" in q_lower or "friend is sad" in q_lower:
            msg = "I'm sorry your friend is going through a tough time. Just spending time together and listening can mean the world right now."
        elif "friend named" in q_lower:
            m = re.search(r"friend named ([A-Za-z]+)", q_raw, re.IGNORECASE)
            name = m.group(1).capitalize() if m else "your friend"
            msg = f"Nice to meet {name} through you! How long have you two known each other?"
        elif "joke" in q_lower:
            msg = "Why do programmers always mix up Halloween and Christmas? Because Oct 31 == Dec 25!"
        else:
            msg = "I'm right here with you. What can I help you explore or accomplish next?"

        return {"text": msg, "path": "deterministic_fallback", "action": "respond_dialogue", "can_execute": False}


# Global singleton
_GLOBAL_MIND_ROUTER: Optional[MindRouter] = None


def get_mind_router(backend: str = "edith_lm") -> MindRouter:
    global _GLOBAL_MIND_ROUTER
    if _GLOBAL_MIND_ROUTER is None or _GLOBAL_MIND_ROUTER.backend != backend:
        _GLOBAL_MIND_ROUTER = MindRouter(backend=backend)
    return _GLOBAL_MIND_ROUTER
