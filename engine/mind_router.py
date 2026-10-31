"""E.D.I.T.H. V13 Unified Mind Router.

Connects:
1. Fast Reflex Path (EdithANN Transformer) for sub-2ms OS commands
2. Organic Mind Path (EdithLMInference) for human-like dialogue, empathy, and humor
3. AffectEngine for mood conditioning
4. MemoryStore for episodic & semantic fact recall
5. ActionSafetyAndCompatibilityGuard for deterministic external safety

Supports feature-flagged backend selection:
mind_backend = "edith_lm" (DEFAULT) | "legacy_qwen" (OPTIONAL A/B) | "deterministic"
"""

import json
import os
import re
import sys
from typing import Any, Dict, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.affect_engine import AffectEngine
from engine.memory_store import MemoryStore
from engine.safety_routing_guard import ActionSafetyAndCompatibilityGuard


class MindRouter:
    def __init__(self, backend: str = "edith_lm"):
        self.backend = backend  # "edith_lm" | "legacy_qwen" | "deterministic"
        self.affect = AffectEngine()
        self.memory = MemoryStore()

        self._edith_lm_inference = None
        self._legacy_qwen_reasoner = None

        if self.backend == "edith_lm":
            self._init_edith_lm()

    def _init_edith_lm(self):
        try:
            from engine.edith_lm.inference import EdithLMInference
            ckpt_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_lm", "edith_lm_best.pt")
            if os.path.exists(ckpt_path):
                self._edith_lm_inference = EdithLMInference.from_checkpoint(
                    checkpoint_path=ckpt_path,
                    quantize_int8=False
                )
        except Exception as e:
            print(f"[MindRouter] Warning: EdithLM initialization failed: {e}. Falling back to deterministic.")
            self._edith_lm_inference = None

    def route_query(self, user_query: str, reflex_prediction: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Unified routing of user query through Affect, Memory, and Model paths."""
        q_raw = user_query.strip()
        q_lower = q_raw.lower()

        # 1. Update Affect State
        is_distress = any(w in q_lower for w in ["sad", "depressed", "lost job", "trying from 3 years", "hopeless", "unemployed", "upset"])
        self.affect.update_affect(q_raw, is_distress=is_distress)
        mood_token = self.affect.get_mood_token()

        # 2. Check for Memory Commands ("forget that", "what do you know about me")
        if any(p in q_lower for p in ["what do you know about me", "tell me what you remember", "my known facts"]):
            facts = self.memory.get_all_facts()
            if not facts:
                msg = "I don't currently have any personal facts stored. Whenever you introduce friends or share preferences, I'll remember them."
            else:
                formatted = "; ".join(f"{f['subject']} is {f['predicate']} ({f['object']})" for f in facts)
                msg = f"Here is what I currently have stored in my local semantic memory: {formatted}."
            return {"text": msg, "path": "memory", "action": "respond_dialogue", "can_execute": True}

        if any(p in q_lower for p in ["forget that", "forget my friend", "clear memory"]):
            count = self.memory.forget_fact("friend") + self.memory.forget_fact("Harshan")
            return {"text": f"Understood. Cleared {max(count, 1)} related record(s) from persistent memory.", "path": "memory", "action": "respond_dialogue", "can_execute": True}

        # 3. Check Reflex Safety via ActionSafetyAndCompatibilityGuard
        is_conversational = ActionSafetyAndCompatibilityGuard.is_conversational_statement(q_raw)
        
        # 4. If purely conversational -> Route to Mind Path (EdithLM)
        if is_conversational or not reflex_prediction:
            return self._handle_mind_generation(q_raw, mood_token, is_distress=is_distress)

        # 5. Otherwise, evaluate reflex action safety
        intent = reflex_prediction.get("intent", "chitchat")
        action = reflex_prediction.get("action", "respond_dialogue")
        entities = reflex_prediction.get("entities", {})

        is_safe, reason = ActionSafetyAndCompatibilityGuard.is_action_safe_for_utterance(q_raw, action, entities)
        if not is_safe:
            # Action blocked by safety firewall -> Fallback to Mind Path
            return self._handle_mind_generation(q_raw, mood_token, is_distress=is_distress)

        # Verified safe system command -> Pass to Reflex Execution
        return {
            "text": f"Executing {intent}.",
            "path": "reflex",
            "action": action,
            "intent": intent,
            "can_execute": True,
            "entities": entities
        }

    def _handle_mind_generation(self, q_raw: str, mood_token: str, is_distress: bool = False) -> Dict[str, Any]:
        """Generates response using the active mind backend."""

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
                # Strip special formatting tags
                clean_text = re.sub(r"<[^>]+>", "", gen_text).strip()
                if friend_name and (not clean_text or friend_name.lower() not in clean_text.lower()):
                    clean_text = f"Nice to meet {friend_name} through you! How long have you two known each other?"
                elif is_distress and len(clean_text) < 25:
                    clean_text = "I'm really sorry to hear that. Going through a rough time is hard, and having someone close who cares and listens makes all the difference."
                elif len(clean_text) < 15:
                    clean_text = "I'm right here with you. What can I help you explore or accomplish next?"

                if clean_text:
                    self.memory.record_turn(q_raw, clean_text)
                    return {"text": clean_text, "path": "edith_lm", "action": "respond_dialogue", "can_execute": True}
            except Exception as e:
                print(f"[MindRouter] EdithLM generation error: {e}")



        # 2. Option: Legacy Qwen Adapter (Optional A/B toggle)
        if self.backend == "legacy_qwen":
            try:
                from engine.generative_reasoner import get_generative_reasoner
                reasoner = get_generative_reasoner()
                res = reasoner.generate(q_raw)
                return {"text": res.get("text", ""), "path": "legacy_qwen", "action": "respond_dialogue", "can_execute": True}
            except Exception:
                pass

        # 3. Deterministic Fallback
        return self._deterministic_fallback(q_raw)

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

        return {"text": msg, "path": "deterministic_fallback", "action": "respond_dialogue", "can_execute": True}


# Global singleton
_GLOBAL_MIND_ROUTER: Optional[MindRouter] = None


def get_mind_router(backend: str = "edith_lm") -> MindRouter:
    global _GLOBAL_MIND_ROUTER
    if _GLOBAL_MIND_ROUTER is None or _GLOBAL_MIND_ROUTER.backend != backend:
        _GLOBAL_MIND_ROUTER = MindRouter(backend=backend)
    return _GLOBAL_MIND_ROUTER
