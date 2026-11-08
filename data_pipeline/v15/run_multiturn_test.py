"""
E.D.I.T.H. V15.3.1 Audit Script: 25 Multi-Turn Conversation Benchmark
Executes 25 separate sequential conversations (5-7 turns each) with cumulative context.
Evaluates entity recall, topic switching, corrections, follow-up references, and identity stability.
Exports full transcripts and metrics to data_pipeline/v15/audit_outputs/multiturn_25_transcripts.json
"""

import os
import sys
import json
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.edith_lm.model import EdithLM, EdithLMConfig
from engine.edith_lm.tokenizer import EdithByteBPETokenizer
from engine.fluency_guard import FluencyGuard

# Define the 25 Multi-Turn Conversation Scenarios (5-7 turns each)
MULTITURN_SCENARIOS = [
    # Scenario 1: Friend Harshan & Job Hunt
    {
        "id": "MT_01",
        "description": "Friend entity recall across movie topic distraction",
        "target_entity": "Harshan",
        "turns": [
            ("I have a close friend named Harshan.", "calm_focused"),
            ("He has been looking for a software job for three years.", "warm_empathic"),
            ("I am also working on my E.D.I.T.H. AI assistant project.", "calm_focused"),
            ("Let's take a break. What's your favourite movie genre?", "witty_playful"),
            ("Back to my friend—what was his name again?", "calm_focused"),
            ("And what has he been struggling with?", "warm_empathic")
        ]
    },
    # Scenario 2: Python Compiler AST Failure
    {
        "id": "MT_02",
        "description": "Component recall across lunch distraction",
        "target_entity": "AST parser pass",
        "turns": [
            ("I'm developing a custom bytecode compiler in Python.", "calm_focused"),
            ("The compiler is crashing specifically on the AST parser pass.", "calm_focused"),
            ("I think I need to step away and grab a sandwich.", "calm_focused"),
            ("Sandwich was great. Which compiler pass did I say was crashing?", "calm_focused"),
            ("What language am I building this compiler in?", "calm_focused")
        ]
    },
    # Scenario 3: Schedule Correction (Meeting Day)
    {
        "id": "MT_03",
        "description": "Schedule correction override",
        "target_entity": "Thursday",
        "turns": [
            ("I have an important project review on Tuesday.", "calm_focused"),
            ("Wait, my manager just pinged me and moved it to Thursday.", "calm_focused"),
            ("I need to prepare slides for the presentation.", "calm_focused"),
            ("Which day is the review scheduled for now?", "calm_focused"),
            ("Is it on Tuesday or Thursday?", "calm_focused")
        ]
    },
    # Scenario 4: Hardware Specs & VRAM
    {
        "id": "MT_04",
        "description": "Hardware specification tracking",
        "target_entity": "6 GB",
        "turns": [
            ("I'm running local model training on my RTX 4050 laptop GPU.", "calm_focused"),
            ("It has exactly 6 gigabytes of VRAM.", "calm_focused"),
            ("Do you think we can train large models without OOM?", "calm_focused"),
            ("How much VRAM does my GPU have?", "calm_focused"),
            ("What model of GPU did I mention?", "calm_focused")
        ]
    },
    # Scenario 5: Food Allergy & Dinner Choices
    {
        "id": "MT_05",
        "description": "Personal health safety constraint tracking",
        "target_entity": "peanuts",
        "turns": [
            ("I have a severe allergy to peanuts.", "calm_focused"),
            ("I'm planning a dinner menu for tonight.", "calm_focused"),
            ("Should I include Thai peanut noodles?", "calm_focused"),
            ("What ingredient am I strictly allergic to?", "calm_focused"),
            ("Suggest a safe pasta dish for me instead.", "calm_focused")
        ]
    },
    # Scenario 6: Identity Stability (User vs Assistant)
    {
        "id": "MT_06",
        "description": "User name vs assistant name identity verification",
        "target_entity": "Aditya",
        "turns": [
            ("My name is Aditya.", "calm_focused"),
            ("And your name is E.D.I.T.H.", "witty_playful"),
            ("We are working together on this neural network.", "calm_focused"),
            ("What is my name?", "calm_focused"),
            ("And what is your name?", "witty_playful")
        ]
    },
    # Scenario 7: Dog Bruno & Vet Visit
    {
        "id": "MT_07",
        "description": "Pet name recall across time",
        "target_entity": "Bruno",
        "turns": [
            ("I adopted a golden retriever named Bruno.", "calm_focused"),
            ("He loves chasing tennis balls in the backyard.", "witty_playful"),
            ("I have to take him to the vet tomorrow morning.", "calm_focused"),
            ("What is my dog's name?", "calm_focused"),
            ("What breed of dog is he?", "calm_focused")
        ]
    },
    # Scenario 8: Editor Transition (VS Code to Neovim)
    {
        "id": "MT_08",
        "description": "Tool preference update",
        "target_entity": "Neovim",
        "turns": [
            ("I used VS Code for three years.", "calm_focused"),
            ("Yesterday I decided to switch completely to Neovim.", "calm_focused"),
            ("Configuring Lua scripts is challenging but fun.", "calm_focused"),
            ("Which editor am I using now?", "calm_focused"),
            ("Did I switch to Emacs or Neovim?", "calm_focused")
        ]
    },
    # Scenario 9: College Exam Stress & Course Subject
    {
        "id": "MT_09",
        "description": "Academic subject tracking with empathy",
        "target_entity": "Linear Algebra",
        "turns": [
            ("I'm extremely anxious about my Linear Algebra final.", "warm_empathic"),
            ("Eigenvalues and matrix factorizations are confusing me.", "calm_focused"),
            ("I have only two days left to study.", "warm_empathic"),
            ("Which exam am I stressed about?", "calm_focused"),
            ("Can you give me a calming tip for exam preparation?", "warm_empathic")
        ]
    },
    # Scenario 10: Hometown & Climate
    {
        "id": "MT_10",
        "description": "City and weather entity tracking",
        "target_entity": "Mumbai",
        "turns": [
            ("I live in Mumbai near the coast.", "calm_focused"),
            ("It gets heavily humid here during the monsoon.", "calm_focused"),
            ("I love listening to the ocean breeze at night.", "calm_focused"),
            ("Which city do I live in?", "calm_focused"),
            ("What is the weather usually like there?", "calm_focused")
        ]
    },
    # Scenario 11: Teammate Rahul & Responsibility
    {
        "id": "MT_11",
        "description": "Teammate name and role tracking",
        "target_entity": "Rahul",
        "turns": [
            ("My teammate Rahul is building our PostgreSQL database.", "calm_focused"),
            ("I am focusing on the frontend React architecture.", "calm_focused"),
            ("We have a sprint demo this Friday.", "calm_focused"),
            ("Who is responsible for the database in my team?", "calm_focused"),
            ("What part of the stack am I working on?", "calm_focused")
        ]
    },
    # Scenario 12: Coffee Preference
    {
        "id": "MT_12",
        "description": "Beverage preference recall",
        "target_entity": "Dark roast",
        "turns": [
            ("I only drink dark roast coffee with no sugar.", "calm_focused"),
            ("Light roast tastes too acidic to me.", "calm_focused"),
            ("I'm brewing a fresh cup right now.", "calm_focused"),
            ("Do I like light roast or dark roast coffee?", "calm_focused"),
            ("Do I take sugar in my coffee?", "calm_focused")
        ]
    },
    # Scenario 13: Project Name Memory
    {
        "id": "MT_13",
        "description": "Acronym explanation recall",
        "target_entity": "Even Dead I'm The Hero",
        "turns": [
            ("E.D.I.T.H. stands for 'Even Dead, I'm The Hero'.", "calm_focused"),
            ("It is inspired by the glasses Tony Stark left behind.", "witty_playful"),
            ("We are building our own sovereign version.", "calm_focused"),
            ("What does the acronym E.D.I.T.H. stand for?", "calm_focused"),
            ("Who gave Peter Parker those glasses in the lore?", "witty_playful")
        ]
    },
    # Scenario 14: Music Artist & Focus Playlist
    {
        "id": "MT_14",
        "description": "Music artist tracking",
        "target_entity": "Hans Zimmer",
        "turns": [
            ("I'm listening to Hans Zimmer's Interstellar soundtrack.", "calm_focused"),
            ("The organ tracks really help me enter deep focus.", "calm_focused"),
            ("I need to write 300 lines of code before 6 PM.", "calm_focused"),
            ("Who composed the soundtrack I'm listening to?", "calm_focused"),
            ("What movie soundtrack is it from?", "calm_focused")
        ]
    },
    # Scenario 15: Car Trouble & Mechanic
    {
        "id": "MT_15",
        "description": "Automotive issue entity tracking",
        "target_entity": "alternator",
        "turns": [
            ("My car broke down on the highway today.", "warm_empathic"),
            ("The mechanic said the alternator is completely dead.", "calm_focused"),
            ("It's going to cost $400 to fix.", "warm_empathic"),
            ("Which car part broke down?", "calm_focused"),
            ("How much did the mechanic estimate the repair will cost?", "calm_focused")
        ]
    },
    # Scenario 16: Vacation Destination
    {
        "id": "MT_16",
        "description": "Travel destination recall",
        "target_entity": "Japan",
        "turns": [
            ("I'm booking a flight to Japan for next spring.", "calm_focused"),
            ("I want to visit Tokyo and Kyoto during cherry blossom season.", "calm_focused"),
            ("I need to renew my passport first.", "calm_focused"),
            ("Which country am I planning to travel to?", "calm_focused"),
            ("What cities did I mention visiting?", "calm_focused")
        ]
    },
    # Scenario 17: Sister Birthday Date
    {
        "id": "MT_17",
        "description": "Date tracking across conversation",
        "target_entity": "October 24",
        "turns": [
            ("My sister's birthday is on October 24th.", "calm_focused"),
            ("I want to buy her a mechanical watch.", "calm_focused"),
            ("I need to order it at least two weeks in advance.", "calm_focused"),
            ("When is my sister's birthday?", "calm_focused"),
            ("What gift am I planning to get her?", "calm_focused")
        ]
    },
    # Scenario 18: Linux Distribution Switch
    {
        "id": "MT_18",
        "description": "OS distribution recall",
        "target_entity": "Arch Linux",
        "turns": [
            ("I spent the whole morning installing Arch Linux.", "calm_focused"),
            ("Getting the WiFi drivers working took two hours.", "calm_focused"),
            ("Now the bspwm window manager is running smoothly.", "calm_focused"),
            ("Which Linux distribution did I install?", "calm_focused"),
            ("What took two hours during installation?", "calm_focused")
        ]
    },
    # Scenario 19: Keyboard Switch Type
    {
        "id": "MT_19",
        "description": "Hardware peripheral preference",
        "target_entity": "linear",
        "turns": [
            ("I swapped my tactile keyboard switches for linear yellow switches.", "calm_focused"),
            ("They feel much smoother for fast typing.", "calm_focused"),
            ("My roommates appreciate that it's quieter too.", "witty_playful"),
            ("What type of keyboard switches do I have now?", "calm_focused"),
            ("Did I replace clicky switches or tactile switches?", "calm_focused")
        ]
    },
    # Scenario 20: Target Vocabulary Size
    {
        "id": "MT_20",
        "description": "Technical tokenizer spec recall",
        "target_entity": "16384",
        "turns": [
            ("Our custom Byte-Level BPE tokenizer has 16,384 vocabulary tokens.", "calm_focused"),
            ("It replaces the tiny 3,051 vocab from V14.", "calm_focused"),
            ("It tokenizes numbers and subwords much more cleanly.", "calm_focused"),
            ("What is our tokenizer vocabulary size?", "calm_focused"),
            ("What was the vocabulary size in V14?", "calm_focused")
        ]
    },
    # Scenario 21: Lost Keys & Search Location
    {
        "id": "MT_21",
        "description": "Everyday memory and location tracking",
        "target_entity": "jacket pocket",
        "turns": [
            ("I lost my car keys for thirty minutes.", "warm_empathic"),
            ("I finally found them inside my winter jacket pocket.", "calm_focused"),
            ("Now I'm running ten minutes late for work.", "warm_empathic"),
            ("Where did I find my keys?", "calm_focused"),
            ("How late am I running?", "calm_focused")
        ]
    },
    # Scenario 22: Favorite Sport
    {
        "id": "MT_22",
        "description": "Hobby and team tracking",
        "target_entity": "cricket",
        "turns": [
            ("I play cricket every Sunday morning with my college batchmates.", "calm_focused"),
            ("I'm an opening batsman.", "calm_focused"),
            ("We won our match last weekend by 20 runs.", "witty_playful"),
            ("What sport do I play on Sundays?", "calm_focused"),
            ("What is my batting position?", "calm_focused")
        ]
    },
    # Scenario 23: Programming Book
    {
        "id": "MT_23",
        "description": "Book title entity tracking",
        "target_entity": "Designing Data-Intensive Applications",
        "turns": [
            ("I'm reading 'Designing Data-Intensive Applications' by Martin Kleppmann.", "calm_focused"),
            ("The chapter on consensus algorithms is brilliant.", "calm_focused"),
            ("I want to finish it by the end of the month.", "calm_focused"),
            ("What book am I currently reading?", "calm_focused"),
            ("Who is the author of that book?", "calm_focused")
        ]
    },
    # Scenario 24: Fitness Goal
    {
        "id": "MT_24",
        "description": "Numerical fitness goal tracking",
        "target_entity": "100 kg",
        "turns": [
            ("My fitness goal this year is to bench press 100 kilograms.", "calm_focused"),
            ("Right now my personal record is 85 kilograms.", "calm_focused"),
            ("I'm training four days a week.", "calm_focused"),
            ("What is my target bench press weight?", "calm_focused"),
            ("What is my current personal record?", "calm_focused")
        ]
    },
    # Scenario 25: Memory Provenance Audit Fact
    {
        "id": "MT_25",
        "description": "Audit fact: dark mode provenance",
        "target_entity": "dark mode",
        "turns": [
            ("In the V15 audit, I specifically confirmed that I prefer dark mode.", "calm_focused"),
            ("Light mode strains my eyes during late-night coding.", "calm_focused"),
            ("Every fact stored in your database must have a provenance label.", "calm_focused"),
            ("What UI theme preference did I state about myself?", "calm_focused"),
            ("Why do I avoid light mode?", "calm_focused")
        ]
    }
]


def generate_multiturn_response(
    model: EdithLM,
    tokenizer: EdithByteBPETokenizer,
    cumulative_context: str,
    device: torch.device,
    max_new_tokens: int = 50,
    temperature: float = 0.70,
    top_p: float = 0.85,
    repetition_penalty: float = 1.25
) -> str:
    prompt_ids = tokenizer.encode(cumulative_context, add_bos=True, add_eos=False)
    generated = list(prompt_ids)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            curr_in = torch.tensor([generated[-512:]], dtype=torch.long, device=device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, _, _ = model(curr_in)
            next_logits = logits[0, -1, :].clone()

            for t_id in set(generated[len(prompt_ids):]):
                if next_logits[t_id] > 0:
                    next_logits[t_id] /= repetition_penalty
                else:
                    next_logits[t_id] *= repetition_penalty

            next_logits = next_logits / temperature
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_logits[indices_to_remove] = -float("Inf")
            probs = F.softmax(next_logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1).item()
            if next_tok == tokenizer.eos_id:
                break
            generated.append(next_tok)

    full_text = tokenizer.decode(generated)
    if full_text.startswith("<bos>"):
        full_text = full_text[5:]
    if cumulative_context in full_text:
        resp = full_text.split(cumulative_context)[-1]
    else:
        resp = full_text
    clean_resp = resp.split("</edith>")[0].strip()
    return clean_resp


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("E.D.I.T.H. V15.3.1 - 25 MULTI-TURN CONVERSATION TEST")
    print(f"Device: {device}")
    print(f"Total Multi-Turn Scenarios: {len(MULTITURN_SCENARIOS)}")
    print("=" * 80)

    tok_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_bpe_tokenizer_16k.json")
    tokenizer = EdithByteBPETokenizer(vocab_file=tok_path)

    best_ckpt = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_lm_v15_sft", "edith_lm_v15_3_sft_best.pt")
    data = torch.load(best_ckpt, map_location=device, weights_only=False)
    cfg_data = data["config"]
    cfg = EdithLMConfig(**cfg_data) if isinstance(cfg_data, dict) else cfg_data
    model = EdithLM(cfg).to(device)
    model.load_state_dict(data["model_state_dict"])
    model.eval()

    all_conversations = []
    entity_recall_successes = 0
    total_recall_probes = 0

    for sc_idx, sc in enumerate(MULTITURN_SCENARIOS, 1):
        s_id = sc["id"]
        desc = sc["description"]
        target = sc["target_entity"].lower()

        history_context = ""
        transcript_turns = []

        for turn_idx, (user_msg, mood) in enumerate(sc["turns"], 1):
            turn_prompt = f"{history_context}<mood>{mood}</mood><user>{user_msg}</user><edith>"
            raw_resp = generate_multiturn_response(model, tokenizer, turn_prompt, device)
            is_fluent, diag = FluencyGuard.inspect(raw_resp, query=user_msg)

            # Check if this turn was an entity recall probe
            is_recall_turn = ("?" in user_msg and any(w in user_msg.lower() for w in ["what", "which", "who", "when", "how much", "where"]))
            recalled_target = False
            if is_recall_turn:
                total_recall_probes += 1
                if target in raw_resp.lower():
                    recalled_target = True
                    entity_recall_successes += 1

            transcript_turns.append({
                "turn": turn_idx,
                "user": user_msg,
                "mood": mood,
                "edith_raw": raw_resp,
                "is_fluent": is_fluent,
                "is_recall_turn": is_recall_turn,
                "recalled_target": recalled_target
            })

            # Append to cumulative context
            history_context += f"<mood>{mood}</mood><user>{user_msg}</user><edith>{raw_resp}</edith>"

        all_conversations.append({
            "scenario_id": s_id,
            "description": desc,
            "target_entity": sc["target_entity"],
            "turns": transcript_turns
        })

        print(f"Scenario {sc_idx:2d}/{len(MULTITURN_SCENARIOS)} [{s_id}] ({desc[:40]}...) completed.")

    out_file = os.path.join(PROJECT_ROOT, "data_pipeline", "v15", "audit_outputs", "multiturn_25_transcripts.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "scenarios": all_conversations,
            "total_conversations": len(MULTITURN_SCENARIOS),
            "total_turns": sum(len(sc["turns"]) for sc in all_conversations),
            "entity_recall_probes": total_recall_probes,
            "entity_recall_successes": entity_recall_successes,
            "entity_recall_accuracy": round(entity_recall_successes / max(1, total_recall_probes) * 100, 2)
        }, f, indent=2)

    print("\n" + "=" * 80)
    print("25 MULTI-TURN CONVERSATIONS BENCHMARK COMPLETE:")
    print(f"  Total Conversations:    {len(MULTITURN_SCENARIOS)}")
    print(f"  Total Turns Processed:  {sum(len(sc['turns']) for sc in all_conversations)}")
    print(f"  Entity Recall Probes:   {total_recall_probes}")
    print(f"  Entity Recall Accuracy: {entity_recall_successes}/{total_recall_probes} ({entity_recall_successes/max(1, total_recall_probes)*100:.1f}%)")
    print(f"  Results saved to:       {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()

