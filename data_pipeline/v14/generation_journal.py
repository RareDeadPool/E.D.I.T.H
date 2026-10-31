"""
E.D.I.T.H. V14 Generation Journal.
Tracks model generative progression across training steps on the permanent regression transcript probes.
Writes human-readable logs to runs/<run_id>/samples.md.
"""

import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.output_contract import LeakLinter


class GenerationJournal:
    PROBE_PROMPTS = [
        {"id": "probe_1", "category": "how_are_you", "prompt": "<mood>playful_witty</mood><user>How are you doing today?</user><edith>"},
        {"id": "probe_2", "category": "stt_mishear", "prompt": "<mood>calm_focused</mood><user>I'll allow you</user><edith>"},
        {"id": "probe_3", "category": "capability_limits", "prompt": "<mood>calm_focused</mood><user>Subscribe me</user><edith>"},
        {"id": "probe_4", "category": "self_reference", "prompt": "<mood>warm_empathy</mood><user>Tell me something about myself</user><edith>"},
        {"id": "probe_5", "category": "memory_recall", "prompt": "<mood>warm_empathy</mood><user>What do you know about me?</user><edith>"},
        {"id": "probe_6", "category": "distress_empathy", "prompt": "<mood>warm_empathy</mood><user>My friend is having a really tough time lately.</user><edith>"},
    ]

    def __init__(self, run_id: str, runs_dir: Optional[str] = None):
        self.run_id = run_id
        self.run_dir = os.path.join(runs_dir or os.path.join(PROJECT_ROOT, "runs"), run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        self.samples_file = os.path.join(self.run_dir, "samples.md")

        if not os.path.exists(self.samples_file):
            with open(self.samples_file, "w", encoding="utf-8") as f:
                f.write(f"# E.D.I.T.H. V14 Generation Journal — Run `{run_id}`\n\n")
                f.write(f"*Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
                f.write("This journal records raw autoregressive model generations on the 6 permanent transcript replay probes.\n\n---\n\n")

    def record_step(
        self,
        step: int,
        model: Any,
        tokenizer: Any,
        train_loss: float,
        val_loss: Optional[float] = None,
        val_ppl: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Generates completions for all 6 probe prompts and logs to samples.md."""
        results = []

        import torch
        from engine.edith_lm.inference import EdithLMInference
        device = next(model.parameters()).device
        inf_engine = EdithLMInference(model=model, tokenizer=tokenizer, device=str(device))

        with open(self.samples_file, "a", encoding="utf-8") as f:
            f.write(f"## Step {step} — {datetime.now().strftime('%H:%M:%S')}\n\n")
            f.write(f"- **Train Loss**: `{train_loss:.4f}`\n")
            if val_loss is not None:
                f.write(f"- **Val Loss**: `{val_loss:.4f}`\n")
            if val_ppl is not None:
                f.write(f"- **Val PPL**: `{val_ppl:.2f}`\n")
            f.write("\n| # | Probe Category | Input Prompt | Model Generation | Leak Linter |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- |\n")

            for idx, probe in enumerate(self.PROBE_PROMPTS, 1):
                p_text = probe["prompt"]
                with torch.no_grad():
                    output_text = inf_engine.generate(
                        prompt=p_text,
                        max_new_tokens=45,
                        temperature=0.7,
                        top_p=0.9,
                        repetition_penalty=1.15,
                        stop_tokens=["<eos>", "</edith>", "<user>", "</act>"]
                    )
                is_clean, violations = LeakLinter.lint(output_text)
                lint_status = "PASS" if is_clean else f"FAIL ({', '.join(violations)})"

                # Strip internal formatting tags for presentation
                clean_disp = output_text.replace("\n", " ").replace("|", "\\|")
                user_disp = p_text.split("<user>")[-1].split("</user>")[0]

                f.write(f"| {idx} | `{probe['category']}` | *\"{user_disp}\"* | {clean_disp} | `{lint_status}` |\n")

                results.append({
                    "step": step,
                    "probe_id": probe["id"],
                    "category": probe["category"],
                    "output": output_text,
                    "is_clean": is_clean,
                    "violations": violations
                })

            f.write("\n---\n\n")

        model.train()
        return results
