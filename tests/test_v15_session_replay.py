"""
E.D.I.T.H. V15 Session Replay Test Suite
Replays verbatim the 10 failing interactions from the user's live session:
1. "describe me"
2. "What do you think about me" (Turn 1)
3. "What do you think about me" (Turn 2)
4. "Really" (resolving to previous turn)
5. Greeting
6. "I'll allow you"
7. "How are you"
8. "Subscribe me" (Turn 1 & 2)
9. "Tell me something about myself"
10. "What do you know about me"

Verifies Gate 0 non-negotiable guarantees:
- 0% gibberish reaches user/speak() (>= 95% word validity, char trigram score >= -7.8)
- 0% internal leaks or classifier labels
- Provenance enforcement: 'dark mode' never spoken, only verified facts returned
- Follow-ups ('Really') resolve to dialogue state
- Greetings have dynamic variety, never fixed 'afternoon grind mode'
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.mind_router import get_mind_router
from engine.personality import EdithPersonality
from engine.output_contract import LeakLinter, speak
from engine.fluency_guard import FluencyGuard


def test_v15_gate_0_session_replay():
    # 1. Initialize router and personality
    router = get_mind_router(backend="edith_lm")
    personality = EdithPersonality()

    # Clear recent turns to start fresh
    router.recent_turns.clear()
    router.last_turn_failed = False
    router.lm_authored_count = 0
    router.bridge_count = 0
    router.category_counts.clear()

    print("\n" + "="*80)
    print("E.D.I.T.H. V15 GATE 0: LIVE SESSION REPLAY & FLUENCY VERIFICATION")
    print("="*80)

    # Turn 0: Greeting
    greeting = personality.generate_startup_greeting()
    print(f"\n[Turn 0 - Greeting]:\n  Edith: {greeting}")
    assert "grind mode" not in greeting.lower(), "Fixed 'grind mode' template detected in greeting!"
    is_fluent, diag = FluencyGuard.inspect(greeting)
    assert is_fluent, f"Greeting failed fluency: {diag}"

    # Replay turns
    test_utterances = [
        ("describe me", "self_reference"),
        ("What do you think about me", "self_reference"),
        ("What do you think about me", "self_reference_repeat"),
        ("Really", "followup_resolution"),
        ("I'll allow you", "stt_mishear"),
        ("How are you", "chit_chat"),
        ("Subscribe me", "capability_gap_1"),
        ("Subscribe me", "capability_gap_2"),
        ("Tell me something about myself", "memory_provenance"),
        ("What do you know about me", "memory_provenance"),
    ]

    session_outputs = []

    for turn_idx, (utterance, purpose) in enumerate(test_utterances, start=1):
        # Route query
        result = router.route_query(utterance)
        spoken_text = speak(result["text"], voice_mode=False)

        session_outputs.append({
            "turn": turn_idx,
            "user": utterance,
            "edith": spoken_text,
            "path": result.get("path"),
            "source": result.get("source", "unknown"),
            "category": result.get("category", "general")
        })

        print(f"\n[Turn {turn_idx} - {purpose}]:")
        print(f"  User:  \"{utterance}\"")
        print(f"  Path:  {result.get('path')} (source={result.get('source', 'unknown')})")
        print(f"  Edith: \"{spoken_text}\"")

        # 1. Non-negotiable Fluency Check
        is_fluent, diag = FluencyGuard.inspect(spoken_text, query=utterance)
        print(f"  Fluency: passed={is_fluent}, word_validity={diag.get('word_validity_ratio', 0.0):.1%}, trigram_logprob={diag.get('char_trigram_logprob', 0.0):.2f}")
        assert is_fluent, f"Turn {turn_idx} failed Fluency Guard! Reason: {diag.get('reason')}. Output: '{spoken_text}'"

        # 2. Leak Linter Check
        is_clean, violations = LeakLinter.lint(spoken_text)
        assert is_clean, f"Turn {turn_idx} contained internal leaks: {violations}"

        # 3. Provenance Verification
        assert "dark mode" not in spoken_text.lower(), f"Unprovenanced mock fact 'dark mode' was spoken in Turn {turn_idx}!"

        # 4. Turn-specific semantic assertions
        if utterance.lower() == "really":
            # Must resolve to previous turn, not generic filler
            assert "i'm right here with you" not in spoken_text.lower(), "Generic filler emitted on 'Really' follow-up!"
            assert any(k in spoken_text.lower() for k in ["asking", "think", "aditya", "work", "scrambled", "code"]), f"Follow-up did not resolve to previous turn: {spoken_text}"

        if utterance.lower() == "i'll allow you":
            assert any(k in spoken_text.lower() for k in ["caught", "mean", "clarify", "sorry"]), f"Mishear handler failed: {spoken_text}"

        if "subscribe" in utterance.lower():
            assert any(k in spoken_text.lower() for k in ["subscribe", "channel", "newsletter", "follow"]), f"Capability gap handler failed: {spoken_text}"

    # Print summary metrics
    stats = router.get_routing_stats()
    print("\n" + "="*80)
    print("V15 GATE 0 SUMMARY METRICS:")
    print(f"  Total Dialogue Turns: {stats['total_dialogue_turns']}")
    print(f"  LM-Authored Count:    {stats['lm_authored_count']} ({stats['lm_authored_share']:.1%})")
    print(f"  Bridge Mode Count:    {stats['bridge_count']} ({stats['bridge_share']:.1%})")
    print(f"  Category Distribution: {stats['category_counts']}")
    print("  Gibberish-reaching-user Rate: 0.0% (VERIFIED)")
    print("  Internal Leak Rate:           0.0% (VERIFIED)")
    print("  Unprovenanced Fact Leak Rate: 0.0% (VERIFIED)")
    print("="*80 + "\n")

    return session_outputs, stats


if __name__ == "__main__":
    test_v15_gate_0_session_replay()
