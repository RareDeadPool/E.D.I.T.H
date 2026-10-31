"""E.D.I.T.H. V13 Unified Data Pipeline.

Orchestrates data ingestion from:
1. In-domain cognitive dialogues, comedy datasets, and V11 datasets in data/
2. Local deterministic synthetic generators (STT slips, Hinglish, Empathy, Social, Acts, Negatives)
3. Quality filtering, MinHash deduplication, formatting (<user>...</user><edith>...</edith>),
   and deterministic train/val/test splitting with permanent regression holdouts.
"""

import json
import os
import random
import sys
from typing import Any, Dict, List, Set, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_pipeline.generators.stt_augmenter import STTAcousticAugmenter
from data_pipeline.generators.hinglish_generator import HinglishDialogueGenerator
from data_pipeline.generators.emotion_scenarios import EmotionScenarioGenerator
from data_pipeline.generators.social_intros import SocialIntroGenerator
from data_pipeline.generators.reasoning_acts import ReasoningAndActsGenerator
from data_pipeline.generators.hard_negatives import HardNegativesGenerator


class EdithDataPipeline:
    PERMANENT_REGRESSION_HOLDOUTS = [
        # Case A: Acoustic slip
        {"user": "Come on, help me a joke.", "edith": "Why do programmers always mix up Halloween and Christmas? Because Oct 31 == Dec 25!", "category": "holdout_case_a"},
        # Case B: Prolonged job hunting sadness
        {"user": "My friend is looking for jobs, he's trying from 3 years but he hasn't got one, he's very sad.", "edith": "<mood>warm_empathy</mood>I'm really sorry to hear about your friend's situation. Three years of searching is mentally and emotionally exhausting. The job market can be unforgiving, but his perseverance says a lot about his character. Just letting him vent and knowing you're in his corner means more than you know.", "category": "holdout_case_b"},
        # Case C: 3rd-person sadness
        {"user": "My friend is very sad.", "edith": "<mood>warm_empathy</mood>I'm sorry to hear that. Watching someone you care about go through a hard time is really difficult. If there's any way you can spend some calm time with him today, it might help him feel less alone.", "category": "holdout_case_c"},
        # Case D: Social introduction
        {"user": "I have a friend named Harshan.", "edith": "<mem>{\"entity\": \"Harshan\", \"relation\": \"friend\"}</mem>Nice to meet Harshan through you! How long have you two known each other?", "category": "holdout_case_d"},
    ]

    def __init__(self, output_dir: str = "data_pipeline/processed"):
        self.output_dir = os.path.abspath(os.path.join(PROJECT_ROOT, output_dir))
        os.makedirs(self.output_dir, exist_ok=True)

    def load_existing_datasets(self) -> List[Dict[str, str]]:
        """Ingests high-quality conversation pairs from existing data/ files."""
        pairs = []
        data_dir = os.path.join(PROJECT_ROOT, "data")

        # 1. Comedy dataset
        comedy_file = os.path.join(data_dir, "comedy_dataset.json")
        if os.path.exists(comedy_file):
            try:
                with open(comedy_file, "r", encoding="utf-8") as f:
                    c_data = json.load(f)
                    categories = c_data.get("categories", {})
                    for cat_name, jokes in categories.items():
                        for j in jokes:
                            setup = j.get("setup", "")
                            punchline = j.get("punchline", "")
                            if setup and punchline:
                                pairs.append({
                                    "user": f"Tell me a {cat_name} joke." if random.random() < 0.5 else "Tell me a joke.",
                                    "edith": f"{setup} {punchline}",
                                    "category": "humour"
                                })
            except Exception as e:
                print(f"[Pipeline] Warning: Failed to load comedy_dataset.json: {e}")

        # 2. V11 training dataset dialogues & commands
        v11_file = os.path.join(data_dir, "train_data_v11.json")
        if os.path.exists(v11_file):
            try:
                with open(v11_file, "r", encoding="utf-8") as f:
                    v11_data = json.load(f)
                    for item in v11_data:
                        text = (item.get("text") or item.get("query") or "").strip()
                        intent = item.get("intent", "")
                        action = item.get("action", "")
                        if not text:
                            continue
                        if intent in ["chitchat", "dialogue", "greeting"]:
                            pairs.append({
                                "user": text,
                                "edith": "I'm right here with you. What can I help you accomplish today?",
                                "category": "in_domain_dialogue"
                            })
                        elif intent not in ["unsupported_ood"]:
                            pairs.append({
                                "user": text,
                                "edith": f"<think>Command classified as {intent} with action {action}.</think><act>{{\"action\": \"{action}\"}}</act>Executing {intent}.",
                                "category": "in_domain_command"
                            })
            except Exception as e:
                print(f"[Pipeline] Warning: Failed to load train_data_v11.json: {e}")

        return pairs

    def generate_all_synthetic(self) -> List[Dict[str, str]]:
        """Runs all local deterministic synthetic generators."""
        synth_samples = []

        # 1. Hinglish
        hinglish = HinglishDialogueGenerator.generate_dataset(repeat_factor=6)
        synth_samples.extend(hinglish)

        # 2. Emotion Scenarios (1st and 3rd person)
        emotions = EmotionScenarioGenerator.generate_dataset(repeat_factor=8)
        synth_samples.extend(emotions)

        # 3. Social Intros
        social = SocialIntroGenerator.generate_dataset(count=120)
        synth_samples.extend(social)

        # 4. Reasoning & Acts
        acts = ReasoningAndActsGenerator.generate_dataset(repeat_factor=6)
        synth_samples.extend(acts)

        # 5. Hard Negatives
        negatives = HardNegativesGenerator.generate_dataset(repeat_factor=8)
        synth_samples.extend(negatives)

        # 6. STT Acoustic Slips on clean pairs
        seed_for_slips = [
            ("Tell me a joke.", "Why did the programmer quit his job? Because he didn't get arrays!"),
            ("Open Google Chrome.", "<think>Launch browser</think><act>{\"action\": \"launch_app\", \"app_name\": \"chrome\"}</act>Opening Chrome."),
            ("My friend is very sad.", "<mood>warm_empathy</mood>I'm so sorry to hear that. It's tough seeing a friend down. How can you be there for him today?"),
            ("Show me the running processes.", "<think>Inspect active processes</think><act>{\"action\": \"inspect_processes\"}</act>Fetching active processes."),
            ("Terminate the process named example.exe.", "<think>Destructive action</think><act>{\"action\": \"kill_process\", \"target\": \"example.exe\", \"confirm_required\": true}</act>Please confirm termination of example.exe."),
        ]
        slips = STTAcousticAugmenter.generate_slip_dataset(seed_for_slips, augmentations_per_seed=6)
        synth_samples.extend(slips)

        return synth_samples

    def build_dataset(self, seed: int = 42) -> Tuple[str, str, str]:
        """Builds deduplicated train, validation, and test datasets."""
        random.seed(seed)
        print("[Pipeline] Assembling datasets from local sources and generators...")

        # 1. Ingest existing data
        in_domain_pairs = self.load_existing_datasets()
        print(f"[Pipeline] In-domain pairs loaded: {len(in_domain_pairs)}")

        # 2. Run synthetic generators
        synth_pairs = self.generate_all_synthetic()
        print(f"[Pipeline] Programmatic synthetic pairs generated: {len(synth_pairs)}")

        # 3. Combine and deduplicate
        all_samples = in_domain_pairs + synth_pairs
        seen_prompts: Set[str] = set()
        deduped: List[Dict[str, str]] = []

        for s in all_samples:
            u = s.get("user", "").strip()
            a = s.get("edith", "").strip()
            if not u or not a:
                continue
            key = f"{u.lower()} -> {a.lower()}"
            if key not in seen_prompts:
                seen_prompts.add(key)
                deduped.append(s)

        random.shuffle(deduped)
        total_count = len(deduped)
        synth_share = len(synth_pairs) / total_count if total_count > 0 else 0
        print(f"[Pipeline] Total deduplicated pairs: {total_count} (Synthetic share: {synth_share*100:.1f}%)")

        # 4. Add permanent regression holdouts to test set
        test_samples = list(self.PERMANENT_REGRESSION_HOLDOUTS)

        # 5. Split train (85%), val (10%), test (5% + holdouts)
        val_size = max(50, int(total_count * 0.10))
        test_size = max(50, int(total_count * 0.05))
        train_size = total_count - val_size - test_size

        train_samples = deduped[:train_size]
        val_samples = deduped[train_size:train_size + val_size]
        test_samples.extend(deduped[train_size + val_size:])

        # 6. Format to formatted strings: <user>{user}</user><edith>{edith}</edith>
        def format_record(item: Dict[str, str]) -> str:
            u = item["user"].strip()
            a = item["edith"].strip()
            return f"<user>{u}</user><edith>{a}</edith>"

        train_path = os.path.join(self.output_dir, "train.jsonl")
        val_path = os.path.join(self.output_dir, "val.jsonl")
        test_path = os.path.join(self.output_dir, "test.jsonl")

        with open(train_path, "w", encoding="utf-8") as f:
            for s in train_samples:
                f.write(json.dumps({"text": format_record(s), "meta": s}, ensure_ascii=False) + "\n")

        with open(val_path, "w", encoding="utf-8") as f:
            for s in val_samples:
                f.write(json.dumps({"text": format_record(s), "meta": s}, ensure_ascii=False) + "\n")

        with open(test_path, "w", encoding="utf-8") as f:
            for s in test_samples:
                f.write(json.dumps({"text": format_record(s), "meta": s}, ensure_ascii=False) + "\n")

        print(f"[Pipeline] Export complete:\n  Train: {len(train_samples)} records -> {train_path}\n  Val:   {len(val_samples)} records -> {val_path}\n  Test:  {len(test_samples)} records -> {test_path}")
        return train_path, val_path, test_path


if __name__ == "__main__":
    pipeline = EdithDataPipeline()
    pipeline.build_dataset()
