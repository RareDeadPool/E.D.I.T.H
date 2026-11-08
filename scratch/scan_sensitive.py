import subprocess
import re
import os
import json

git_files = subprocess.check_output(['git', 'ls-files']).decode('utf-8').splitlines()

# Patterns for sensitive information
phone_pat = re.compile(r'\+?\d{10,13}')
email_pat = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
secret_pat = re.compile(r'(?i)(api[_-]?key|secret|token|password|bearer)[\'\"\s:=]+[a-zA-Z0-9_\-]{8,}')

results = {}

for path in git_files:
    if path.endswith(('.png', '.bmp', '.wav', '.safetensors', '.bin', '.idx', '.pt', '.pyc')):
        continue
    if not os.path.exists(path):
        continue
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        findings = []
        if 'data/vocab' not in path and 'docs/DATA_LICENSES' not in path and not path.startswith('data_pipeline/v15/gutenberg'):
            phones = phone_pat.findall(content)
            emails = email_pat.findall(content)
            secrets = secret_pat.findall(content)
            
            # Filter obvious false positives in code/git
            valid_phones = [p for p in phones if not p.startswith(('1234567890', '0000000000'))]
            valid_emails = [e for e in emails if not e.endswith(('example.com', 'test.com', 'domain.com')) and 'sawantaditya0708@gmail.com' not in e]
            
            if valid_phones:
                findings.append(f"Phones ({len(valid_phones)})")
            if valid_emails:
                findings.append(f"Emails ({len(valid_emails)})")
            if secrets:
                findings.append(f"Secrets ({len(secrets)})")

        # Specific directory audits
        if path.startswith('config/cognitive_memory/'):
            findings.append("Cognitive Memory / Personal profile")
        elif path.startswith('config/experience_data/'):
            findings.append("Experience Data / Runtime state")
        elif path in [
            'config/cognitive_state_snapshot.json',
            'config/affect_state_v13.json',
            'config/personality_state.json',
            'config/dpo_preference_pairs.jsonl',
            'config/self_learning_audit.json',
            'config/self_learning_replay.jsonl',
            'data/contacts.json',
            'data/whatsapp_config.json',
            'data_pipeline/v15/audit_outputs/human_eval_key.json'
        ]:
            findings.append("Runtime State / Personal config / Evaluation Key")

        if findings:
            results[path] = findings

    except Exception as e:
        results[path] = [f"Error: {e}"]

print(f"Total sensitive files identified: {len(results)}")
for p, f in sorted(results.items()):
    print(f"  {p} -> {', '.join(f)}")

with open('scratch/sensitive_files_audit.json', 'w', encoding='utf-8') as out:
    json.dump(results, out, indent=2)

