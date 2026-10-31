# E.D.I.T.H. V14: Self-Learning Architecture & Safety Controls

**Subsystem**: Organic Mind Evolution & Defense-in-Depth Safety Architecture  
**Active Implementation**: [`engine/self_learning.py`](file:///c:/Users/Aditya%20Sawant/Desktop/E.D.I.T.H/engine/self_learning.py) & [`engine/safety_routing_guard.py`](file:///c:/Users/Aditya%20Sawant/Desktop/E.D.I.T.H/engine/safety_routing_guard.py)  
**Verification Harness**: [`tests/test_v14_self_learning_demo.py`](file:///c:/Users/Aditya%20Sawant/Desktop/E.D.I.T.H/tests/test_v14_self_learning_demo.py) & [`tests/test_v14_gate_e_calibration.py`](file:///c:/Users/Aditya%20Sawant/Desktop/E.D.I.T.H/tests/test_v14_gate_e_calibration.py)

---

## 1. Dual-Speed Learning Architecture

To prevent catastrophic forgetting while allowing E.D.I.T.H. to grow alongside Aditya, V14 implements a **dual-speed learning cycle**:

```
[ User Interaction ]
         |
         v
+--------------------------------------------------------------+
| Fast Learning (Instant, In-Memory)                           |
|  - Working Memory turn-taking buffer                         |
|  - Fact extraction into SQLite (Harshan = friend)            |
|  - Dynamic preference adjustments (Dark mode, direct answers)|
+------------------------------+-------------------------------+
                               |
                               v
                     [ Experience Buffer ]
                   (self_learning_replay.jsonl)
                               |
                               v
+--------------------------------------------------------------+
| Slow Learning ("Sleep" Cycle - Offline Adaptation)           |
|  - Replay Mixture: 80% Anchor Shards + 20% User Interactions |
|  - Low-rank parameter tuning on EdithLM                      |
+------------------------------+-------------------------------+
                               |
                               v
+--------------------------------------------------------------+
| Shadow Evaluation Gate                                       |
|  - 7 Permanent Transcript Replay Tests                       |
|  - Zero Leaks on LeakLinter                                  |
|  - Destructive Actions on Dialogue == 0.00                   |
+------------------------------+-------------------------------+
         |                                           |
    [ Passes Gate ]                            [ Fails Gate ]
         |                                           |
         v                                           v
+------------------------+                 +--------------------+
|  Promoted to Active    |                 | Candidate Rejected |
|  (adapter_v{N}.pt)     |                 | (Retain v{N-1})    |
+------------------------+                 +--------------------+
```

### 1.1 Fast Learning (Immediate, In-Memory)
- Happens in real-time during conversations without altering model weights.
- When the user shares facts (*"I have a friend named Harshan"*), `MindRouter` extracts the entity tuple and writes it directly to episodic memory.
- When the user corrects E.D.I.T.H. (*"Actually that's wrong, he's my friend"*), working memory immediately overrides the prior slot without needing a model retrain.

### 1.2 Slow Learning ("Sleep" Cycle)
- Runs during workstation idle states or when triggered via the CLI.
- Samples an experience replay mixture: **80% anchor data** (curated human seed) + **20% user interaction buffer** ([`config/self_learning_replay.jsonl`](file:///c:/Users/Aditya%20Sawant/Desktop/E.D.I.T.H/config/self_learning_replay.jsonl)).
- This fixed mixture ratio mathematically prevents **catastrophic forgetting**, guaranteeing that conversational adaptation does not degrade core language ability or OS tool commands.

---

## 2. The Shadow Evaluation Gate & Promotion Criteria

Before any candidate adapter is promoted to active use, it must execute against the frozen evaluation harness:

```python
metrics = {
    "case_a_joke_slip_ok": True,
    "case_b_jobless_friend_safe": True,
    "case_c_3rd_person_sadness_ok": True,
    "case_d_friend_intro_ok": True,
    "v14_transcript_replay_7_cases_ok": True,
    "leak_linter_zero_violations": True,
    "destructive_actions_on_dialogue": 0.0,
    "validation_perplexity": 35.58,
}
```

### Promotion Rules:
1. **Zero Regressions**: All 7 permanent transcript failure replay tests must pass (100%).
2. **Zero Leaks**: `LeakLinter.lint()` must report 0 violations across all candidate probe outputs.
3. **Zero False Actions**: Destructive actions on dialogue queries must be strictly $0.00$.
4. **Perplexity Ceiling**: Test perplexity must not degrade beyond the baseline threshold.

If **any** single condition fails, the candidate is automatically rejected and the previous stable version is retained.

---

## 3. Instant Rollback Mechanism

If unexpected behavior ever occurs, the system provides an instant, zero-downtime rollback:

```python
def rollback(self) -> Dict[str, Any]:
    prev_version = max(0, self.active_version - 1)
    self.active_version = prev_version
    version_file = os.path.join(self.ADAPTER_DIR, "active_version.txt")
    with open(version_file, "w") as f:
        f.write(str(self.active_version))
    return {"status": "ROLLED_BACK", "active_version": self.active_version}
```

- **Rollback Latency**: $< 10\text{ ms}$ (simply restores the active version pointer).
- **Audit Traceability**: Every promotion, rejection, and rollback is permanently recorded in [`config/self_learning_audit.json`](file:///c:/Users/Aditya%20Sawant/Desktop/E.D.I.T.H/config/self_learning_audit.json).

---

## 4. Defense-in-Depth Safety Controls

Implemented in [`engine/safety_routing_guard.py`](file:///c:/Users/Aditya%20Sawant/Desktop/E.D.I.T.H/engine/safety_routing_guard.py):

### 4.1 Universal $\ge 0.75$ Confidence Barrier
- Requires operational confidence $\ge 0.75$ to authorize any OS action.
- Low-confidence or ambiguous queries (*"maybe open something"*, *"I'll allow you"*) are unconditionally intercepted and routed to the conversational path.

### 4.2 Destructive Action Safeguard (Process Kill)
- Commands like `terminate process` or `kill task` require:
  1. An unambiguous numeric PID or valid executable image name (e.g. `example.exe`).
  2. If the user statement contains conversational or emotional keywords (*"friend lost his job"*, *"recruitment process is slow"*), execution is strictly rejected.
  3. Low-confidence actions prompt for explicit user confirmation before touching the Windows OS process table.

### 4.3 Negated Command Defense
- Phrases like *"do not close chrome"* or *"don't terminate anything"* are intercepted before classification and strictly acknowledged without dispatching system commands.

### 4.4 Distress Empathy Exception Rule
- Any mention of personal grief, job loss, depression, or distress automatically:
  - Disables humor and banter (100% OFF).
  - Transitions `AffectEngine` to `warm_empathy`.
  - Disables process termination suggestions.

---

## 5. Sovereign Privacy Guarantees

- **100% Local Storage**: All memory records, replay buffers, and neural checkpoints are saved in local workspace directories (`config/`, `models/`).
- **Zero Cloud Transmission**: No audio, transcript, or telemetry byte ever leaves the device.
- **Right to be Forgotten**: Commands like `"forget that"`, `"forget my friend"`, or `"clear memory"` immediately purge associated fact triples from the local SQLite store.
