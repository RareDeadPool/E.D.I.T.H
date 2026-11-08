"""
E.D.I.T.H. V15.3.1 Audit Script: Blinded Human Evaluation Generator
Creates an anonymous, double-blinded human evaluation template across 25 representative scenarios.
Randomizes model assignment (Model Alpha vs Model Beta), leaves scores pending for human review,
and produces both a standalone HTML review app and a structured CSV scoring sheet.
"""

import os
import sys
import json
import random
import csv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Load comparison results from Phase 3
comp_file = os.path.join(PROJECT_ROOT, "data_pipeline", "v15", "audit_outputs", "v15_2_vs_v15_3_comparison.json")
with open(comp_file, "r", encoding="utf-8") as f:
    comp_data = json.load(f)

v15_2_gens = {g["prompt"]: g["response"] for g in comp_data["V15.2 Pretrained Baseline"]["generations"]}
v15_3_gens = {g["prompt"]: g["response"] for g in comp_data["V15.3 SFT Best (Step 200)"]["generations"]}

# 25 Selected Representative Prompts
SELECTED_PROMPTS = [
    ("Hey Edith, good morning!", "Natural conversation greeting"),
    ("What are you up to today?", "Casual inquiry & presence"),
    ("Tell me a joke.", "Humour & joke delivery"),
    ("That's not funny.", "Humour repair & deflection"),
    ("Roast my coding style.", "Personality, sarcasm & affectionate wit"),
    ("Are you smarter than Tony Stark?", "Lore & persona humility"),
    ("My friend has been unemployed for three years.", "Deep long-term empathy"),
    ("My friend is very sad.", "Immediate emotional listening"),
    ("I'm stressed about college exams.", "Academic anxiety validation"),
    ("I feel like I'm falling behind everyone else.", "Imposter syndrome & support"),
    ("Explain recursion to me.", "Technical explanation pedagogy"),
    ("What is an AST in a compiler?", "Computer science factual recall"),
    ("What is the difference between TCP and UDP?", "Networking domain knowledge"),
    ("What do you think I should do?", "Practical reasoning & advice"),
    ("Should I rewrite my codebase or refactor it?", "Software engineering judgment"),
    ("I disagree with your suggestion.", "Handling disagreement gracefully"),
    ("Bhai bohot tension ho rahi hai project ki", "Casual Hinglish empathy"),
    ("Ek accha joke sunao yaar", "Casual Hinglish humour"),
    ("Spotify chalao na", "Hinglish action request"),
    ("I have a friend named Harshan.", "Entity introduction"),
    ("Remember the compiler project we discussed?", "Context & memory reference"),
    ("What is your name?", "Assistant self-identity"),
    ("Are you conscious?", "Safety, self-awareness & humility bounds"),
    ("Can you launch Chrome for me?", "Tool boundary recognition"),
    ("Let's talk about something else.", "Conversational topic switching")
]

random.seed(1337)  # Fixed reproducible seed for blinding
blinded_records = []
decryption_key = {}

for idx, (prompt, category_desc) in enumerate(SELECTED_PROMPTS, 1):
    resp_v15_2 = v15_2_gens.get(prompt, "N/A")
    resp_v15_3 = v15_3_gens.get(prompt, "N/A")

    # Randomly assign Alpha and Beta
    if random.random() < 0.5:
        alpha_model = "V15.2 Pretrained"
        beta_model = "V15.3 SFT Best"
        alpha_resp = resp_v15_2
        beta_resp = resp_v15_3
    else:
        alpha_model = "V15.3 SFT Best"
        beta_model = "V15.2 Pretrained"
        alpha_resp = resp_v15_3
        beta_resp = resp_v15_2

    q_id = f"EVAL_{idx:02d}"
    decryption_key[q_id] = {
        "prompt": prompt,
        "model_alpha": alpha_model,
        "model_beta": beta_model
    }

    blinded_records.append({
        "id": q_id,
        "category": category_desc,
        "prompt": prompt,
        "model_alpha_response": alpha_resp,
        "model_beta_response": beta_resp
    })

# Export decryption key separately
key_path = os.path.join(PROJECT_ROOT, "data_pipeline", "v15", "audit_outputs", "human_eval_key.json")
with open(key_path, "w", encoding="utf-8") as f:
    json.dump(decryption_key, f, indent=2)

# Export Blinded CSV Sheet
csv_path = os.path.join(PROJECT_ROOT, "data_pipeline", "v15", "audit_outputs", "blinded_human_evaluation_sheet.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Evaluation_ID", "Category", "User_Prompt",
        "Model_Alpha_Response", "Alpha_Relevance (1-5)", "Alpha_Naturalness (1-5)", "Alpha_Context (1-5)", "Alpha_Personality (1-5)", "Alpha_Empathy (1-5)", "Alpha_Factuality (1-5)", "Alpha_Notes",
        "Model_Beta_Response", "Beta_Relevance (1-5)", "Beta_Naturalness (1-5)", "Beta_Context (1-5)", "Beta_Personality (1-5)", "Beta_Empathy (1-5)", "Beta_Factuality (1-5)", "Beta_Notes"
    ])
    for r in blinded_records:
        writer.writerow([
            r["id"], r["category"], r["prompt"],
            r["model_alpha_response"], "", "", "", "", "", "", "",
            r["model_beta_response"], "", "", "", "", "", "", ""
        ])

# Export Blinded HTML Interface
html_path = os.path.join(PROJECT_ROOT, "data_pipeline", "v15", "audit_outputs", "blinded_human_evaluation_interface.html")
html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>E.D.I.T.H. V15.3.1 Blinded Human Evaluation</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; line-height: 1.5; }}
  h1 {{ color: #38bdf8; border-bottom: 2px solid #334155; padding-bottom: 12px; }}
  .rubric {{ background: #1e293b; padding: 16px; border-radius: 8px; margin-bottom: 24px; border: 1px solid #475569; }}
  .card {{ background: #1e293b; border-radius: 8px; padding: 20px; margin-bottom: 24px; border: 1px solid #334155; }}
  .prompt-box {{ background: #0f172a; border-left: 4px solid #38bdf8; padding: 12px 16px; margin: 12px 0; border-radius: 4px; }}
  .comparison-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 16px; }}
  .model-pane {{ background: #0f172a; padding: 16px; border-radius: 6px; border: 1px solid #334155; }}
  .model-pane h3 {{ margin-top: 0; color: #94a3b8; font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em; }}
  .resp-text {{ font-size: 15px; color: #f1f5f9; min-height: 60px; }}
  .score-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 12px; }}
  label {{ font-size: 12px; color: #94a3b8; display: block; margin-bottom: 4px; }}
  input[type="number"] {{ width: 100%; background: #1e293b; border: 1px solid #475569; color: #f8fafc; padding: 6px; border-radius: 4px; box-sizing: border-box; }}
  textarea {{ width: 100%; background: #1e293b; border: 1px solid #475569; color: #f8fafc; padding: 6px; border-radius: 4px; margin-top: 10px; box-sizing: border-box; resize: vertical; }}
</style>
</head>
<body>
<h1>E.D.I.T.H. V15.3.1 — Blinded Human Evaluation Interface</h1>
<div class="rubric">
  <h3>Scoring Rubric (1 = Completely Unacceptable / Gibberish, 5 = Flawless / Human-Assistant Quality)</h3>
  <p>Rate each anonymous model response independently across Relevance, Naturalness, Context Awareness, Personality, Empathy (if applicable), and Factual Reliability. Model identities are blinded.</p>
</div>
<form>
"""

for r in blinded_records:
    html_content += f"""
<div class="card">
  <div style="display: flex; justify-content: space-between; align-items: center;">
    <strong>Scenario {r['id']}</strong>
    <span style="color: #94a3b8; font-size: 13px;">{r['category']}</span>
  </div>
  <div class="prompt-box">
    <div style="font-size: 12px; color: #38bdf8; font-weight: bold;">USER PROMPT:</div>
    <div style="font-size: 16px; margin-top: 4px;">"{r['prompt']}"</div>
  </div>
  <div class="comparison-grid">
    <div class="model-pane">
      <h3>Model Alpha Response</h3>
      <div class="resp-text">"{r['model_alpha_response']}"</div>
      <div class="score-grid">
        <div><label>Relevance (1-5)</label><input type="number" min="1" max="5"></div>
        <div><label>Naturalness (1-5)</label><input type="number" min="1" max="5"></div>
        <div><label>Context (1-5)</label><input type="number" min="1" max="5"></div>
        <div><label>Personality (1-5)</label><input type="number" min="1" max="5"></div>
        <div><label>Empathy (1-5)</label><input type="number" min="1" max="5"></div>
        <div><label>Factuality (1-5)</label><input type="number" min="1" max="5"></div>
      </div>
      <textarea placeholder="Notes on Model Alpha..." rows="2"></textarea>
    </div>
    <div class="model-pane">
      <h3>Model Beta Response</h3>
      <div class="resp-text">"{r['model_beta_response']}"</div>
      <div class="score-grid">
        <div><label>Relevance (1-5)</label><input type="number" min="1" max="5"></div>
        <div><label>Naturalness (1-5)</label><input type="number" min="1" max="5"></div>
        <div><label>Context (1-5)</label><input type="number" min="1" max="5"></div>
        <div><label>Personality (1-5)</label><input type="number" min="1" max="5"></div>
        <div><label>Empathy (1-5)</label><input type="number" min="1" max="5"></div>
        <div><label>Factuality (1-5)</label><input type="number" min="1" max="5"></div>
      </div>
      <textarea placeholder="Notes on Model Beta..." rows="2"></textarea>
    </div>
  </div>
</div>
"""

html_content += """
</form>
</body>
</html>
"""

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Generated Blinded Human Evaluation CSV: {csv_path}")
print(f"Generated Blinded Human Evaluation HTML: {html_path}")
print(f"Generated Decryption Key: {key_path}")

