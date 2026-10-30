"""E.D.I.T.H. V11 Local Diagnostic & Realistic Engineering Inspection Engine.

Provides deep diagnostic observability for pairs programming, engineering audits,
and natural language evaluation without cluttering ordinary user interactions.

Diagnostic Card Fields:
- Raw input
- Normalized input
- Tokenized input
- Unknown-token count & percentage
- Truncation status
- Predicted intent & confidence
- Predicted entities
- Action policy & confidence
- Operational confidence
- Complexity prediction
- Context summary (active task, topic, candidates)
- Selected plan & tool handler
- Verification result & deterministic safety gate
- Outcome marking & privacy-safe feedback recording
"""

import os
import sys
import json
import time
from typing import Dict, Any, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.edith_brain import EdithBrain
from engine.inference import EdithInferenceEngine
from edith_ann.tokenizer import EdithTokenizer

DIAGNOSTIC_FEEDBACK_PATH = os.path.join(PROJECT_ROOT, "logs", "diagnostic_feedback.json")


class EdithDiagnosticInspector:
    """Diagnostic inspector providing full neural and contextual transparency."""

    VALID_OUTCOMES = [
        "correct",
        "incorrect_intent",
        "incorrect_entity",
        "context_failure",
        "wrong_action",
        "missing_clarification",
        "incorrect_response",
        "tool_execution_failure"
    ]

    def __init__(self, brain: Optional[EdithBrain] = None):
        self.brain = brain or EdithBrain(enable_execution=False)
        self.engine: EdithInferenceEngine = self.brain.engine
        self.tokenizer: EdithTokenizer = self.engine.tokenizer
        os.makedirs(os.path.dirname(DIAGNOSTIC_FEEDBACK_PATH), exist_ok=True)

    def inspect(self, raw_command: str) -> Dict[str, Any]:
        """Generate a complete diagnostic trace card for a given input."""
        t_start = time.perf_counter()

        # 1. Tokenization and Vocabulary Analysis
        tokens = self.tokenizer.tokenize_raw(raw_command)
        unk_tokens = []
        for t in tokens:
            tid = self.tokenizer._resolve_token_id(t)
            if tid == self.tokenizer.UNK_ID:
                unk_tokens.append(t)

        unk_pct = (len(unk_tokens) / max(1, len(tokens))) * 100.0
        max_len = 64
        is_truncated = len(tokens) > (max_len - 2)

        # 2. Neural Predictions
        pred = self.engine.predict(raw_command)

        # 3. Brain & Pipeline Processing (Simulation without destructive execution)
        brain_res = self.brain.process(raw_command, execute=False)

        # 4. Context Inspection
        context_summary = {}
        if self.brain.context:
            ctx = self.brain.context
            context_summary = {
                "active_task": getattr(ctx, "active_task", None),
                "current_topic": getattr(ctx, "current_topic", None),
                "last_app": getattr(ctx, "last_app", None),
                "last_file": getattr(ctx, "last_file", None),
                "candidates_count": len(getattr(ctx, "search_candidates", [])),
                "has_draft": ctx.has_draft() if hasattr(ctx, "has_draft") else False,
                "pending_clarification": bool(getattr(ctx, "pending_clarification", None))
            }

        directive = brain_res.get("directive", {})
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        diagnostic_card = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "raw_input": raw_command,
            "normalized_input": raw_command.strip(),
            "tokenized_input": tokens,
            "token_count": len(tokens),
            "unknown_tokens": unk_tokens,
            "unknown_token_percentage": round(unk_pct, 2),
            "truncation_status": is_truncated,
            "predicted_intent": brain_res.get("intent", pred.get("intent")),
            "intent_confidence": brain_res.get("intent_confidence", pred.get("intent_confidence")),
            "predicted_entities": brain_res.get("entities", pred.get("entities")),
            "action_policy": brain_res.get("action", pred.get("action")),
            "action_confidence": brain_res.get("action_confidence", pred.get("action_confidence")),
            "operational_confidence": brain_res.get("operational_confidence", pred.get("operational_confidence")),
            "complexity": "multi_step" if pred.get("is_multi_step") else "single_step",
            "context_summary": context_summary,
            "selected_plan": brain_res.get("status", "OPERATIONAL"),
            "selected_tool": directive.get("handler", "none"),
            "can_execute": directive.get("can_execute", False),
            "safety_gate": "PASSED" if directive.get("can_execute") else "GATED_OR_DIALOGUE",
            "system_response": directive.get("message", brain_res.get("message", "")),
            "latency_ms": round(latency_ms, 2)
        }
        return diagnostic_card

    def format_card(self, card: Dict[str, Any]) -> str:
        """Format diagnostic card into readable ASCII text."""
        lines = [
            "---------------- DIAGNOSTIC TRACE CARD ----------------",
            f" Raw Input:           {card['raw_input']}",
            f" Tokens ({card['token_count']}):         {card['tokenized_input']}",
            f" UNK Rate:            {card['unknown_token_percentage']:.1f}% ({card['unknown_tokens']})",
            f" Truncated:           {card['truncation_status']}",
            f" Predicted Intent:    {card['predicted_intent']} (Conf: {card['intent_confidence']:.2f})",
            f" Entities:            {card['predicted_entities']}",
            f" Action Policy:       {card['action_policy']} (Conf: {card['action_confidence']:.2f})",
            f" Operational Conf:    {card['operational_confidence']:.2f}",
            f" Complexity:          {card['complexity']}",
            f" Selected Tool:       {card['selected_tool']}",
            f" Safety Gate:         {card['safety_gate']}",
            f" Context State:       {card['context_summary']}",
            f" Response:            {card['system_response']}",
            f" Latency:             {card['latency_ms']} ms",
            "-------------------------------------------------------"
        ]
        return "\n".join(lines)

    def mark_outcome(self, card: Dict[str, Any], outcome: str, notes: str = "") -> bool:
        """Record privacy-safe diagnostic feedback for failure-driven improvement."""
        outcome_clean = outcome.strip().lower()
        if outcome_clean not in self.VALID_OUTCOMES:
            print(f"[ERROR] Invalid outcome: {outcome_clean}. Must be one of {self.VALID_OUTCOMES}")
            return False

        feedback_entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "outcome": outcome_clean,
            "notes": notes,
            "query": card["raw_input"],
            "intent": card["predicted_intent"],
            "entities": card["predicted_entities"],
            "action": card["action_policy"],
            "confidence": card["operational_confidence"],
            "tool": card["selected_tool"],
            "unknown_token_percentage": card["unknown_token_percentage"],
            "latency_ms": card["latency_ms"]
        }

        existing = []
        if os.path.exists(DIAGNOSTIC_FEEDBACK_PATH):
            try:
                with open(DIAGNOSTIC_FEEDBACK_PATH, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = []

        existing.append(feedback_entry)
        with open(DIAGNOSTIC_FEEDBACK_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

        print(f"[DIAGNOSTIC] Recorded outcome '{outcome_clean}' to {DIAGNOSTIC_FEEDBACK_PATH}")
        return True


def cli_interactive():
    """Interactive command-line diagnostic mode."""
    inspector = EdithDiagnosticInspector()
    print("=========================================================")
    print(" E.D.I.T.H. V11 Interactive Engineering Diagnostic Mode")
    print(" Enter any query to view full internal predictions.")
    print(" Type 'quit' or 'exit' to terminate.")
    print("=========================================================")

    while True:
        try:
            cmd = input("\n[EDITH Diagnostic] > ").strip()
            if not cmd:
                continue
            if cmd.lower() in ["quit", "exit"]:
                break

            card = inspector.inspect(cmd)
            print("\n---------------- DIAGNOSTIC TRACE CARD ----------------")
            print(f" Raw Input:           {card['raw_input']}")
            print(f" Tokens ({card['token_count']}):         {card['tokenized_input']}")
            print(f" UNK Rate:            {card['unknown_token_percentage']:.1f}% ({card['unknown_tokens']})")
            print(f" Truncated:           {card['truncation_status']}")
            print(f" Predicted Intent:    {card['predicted_intent']} (Conf: {card['intent_confidence']:.2f})")
            print(f" Entities:            {card['predicted_entities']}")
            print(f" Action Policy:       {card['action_policy']} (Conf: {card['action_confidence']:.2f})")
            print(f" Operational Conf:    {card['operational_confidence']:.2f}")
            print(f" Complexity:          {card['complexity']}")
            print(f" Selected Tool:       {card['selected_tool']}")
            print(f" Safety Gate:         {card['safety_gate']}")
            print(f" Context State:       {card['context_summary']}")
            print(f" Response:            {card['system_response']}")
            print(f" Latency:             {card['latency_ms']} ms")
            print("-------------------------------------------------------")

            mark = input("Mark outcome? (c=correct, i=incorrect_intent, e=incorrect_entity, a=wrong_action, skip=Enter): ").strip().lower()
            mapping = {
                "c": "correct",
                "i": "incorrect_intent",
                "e": "incorrect_entity",
                "a": "wrong_action",
                "f": "context_failure",
                "m": "missing_clarification"
            }
            if mark in mapping:
                notes = input("Optional engineering notes: ").strip()
                inspector.mark_outcome(card, mapping[mark], notes)

        except (KeyboardInterrupt, EOFError):
            break

    print("\nExiting diagnostic mode.")


if __name__ == "__main__":
    cli_interactive()
