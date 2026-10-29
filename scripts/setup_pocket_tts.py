"""Setup and Download Script for E.D.I.T.H. Pocket TTS Integration.

Downloads:
1. Pocket TTS Multilingual Model (Thanushttk/pocket-tts-multilingual, rev: 80c7f2835138c62750d76a93d4333c5c5bb6a754)
   - config.yaml
   - model.safetensors (~418 MB)
   - tokenizer.model (~1.5 MB)
   Target location: models/pocket_tts/

2. Authorized Indian Female English Reference Voice
   - Source: CSTR VCTK Corpus speaker p248 (Female, 23, Indian English) via kyutai/tts-voices (enhanced studio edition)
   - License: CC BY 4.0
   Target location: assets/voices/edith_indian_english.wav
"""

import os
import sys
import shutil
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import huggingface_hub

HF_REPO_MODEL = "Thanushttk/pocket-tts-multilingual"
HF_MODEL_REVISION = "80c7f2835138c62750d76a93d4333c5c5bb6a754"
HF_VOICES_REPO = "kyutai/tts-voices"
HF_VOICE_FILE = "vctk/p248_023_enhanced.wav"

TARGET_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "pocket_tts")
TARGET_VOICE_DIR = os.path.join(PROJECT_ROOT, "assets", "voices")
TARGET_VOICE_FILE = os.path.join(TARGET_VOICE_DIR, "edith_indian_english.wav")


def setup_pocket_tts():
    os.makedirs(TARGET_MODEL_DIR, exist_ok=True)
    os.makedirs(TARGET_VOICE_DIR, exist_ok=True)

    print("==================================================================")
    print("        E.D.I.T.H. POCKET TTS SETUP & MODEL DOWNLOAD              ")
    print("==================================================================")

    # 1. Download Model Files
    files = ["config.yaml", "model.safetensors", "tokenizer.model"]
    for fname in files:
        target_path = os.path.join(TARGET_MODEL_DIR, fname)
        if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
            print(f"[OK] {fname} already exists ({os.path.getsize(target_path) / 1024 / 1024:.2f} MB)")
            continue

        print(f"[DOWNLOADING] {fname} from {HF_REPO_MODEL}@{HF_MODEL_REVISION[:8]}...")
        downloaded = huggingface_hub.hf_hub_download(
            repo_id=HF_REPO_MODEL,
            filename=fname,
            revision=HF_MODEL_REVISION,
        )
        shutil.copy2(downloaded, target_path)
        print(f"[SAVED] {target_path} ({os.path.getsize(target_path) / 1024 / 1024:.2f} MB)")

    # 2. Adjust local config.yaml to ensure offline path resolution
    cfg_path = os.path.join(TARGET_MODEL_DIR, "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["weights_path"] = os.path.join(TARGET_MODEL_DIR, "model.safetensors").replace("\\", "/")
    cfg["weights_path_without_voice_cloning"] = cfg["weights_path"]
    if "flow_lm" in cfg and "lookup_table" in cfg["flow_lm"]:
        cfg["flow_lm"]["lookup_table"]["tokenizer_path"] = os.path.join(TARGET_MODEL_DIR, "tokenizer.model").replace("\\", "/")

    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"[CONFIG] Updated local {cfg_path} with explicit offline paths.")

    # 3. Download Default Indian Female Voice Reference
    if not os.path.exists(TARGET_VOICE_FILE) or os.path.getsize(TARGET_VOICE_FILE) == 0:
        print(f"[DOWNLOADING] Voice reference {HF_VOICE_FILE} from {HF_VOICES_REPO}...")
        downloaded_voice = huggingface_hub.hf_hub_download(
            repo_id=HF_VOICES_REPO,
            filename=HF_VOICE_FILE,
        )
        shutil.copy2(downloaded_voice, TARGET_VOICE_FILE)
        print(f"[SAVED] {TARGET_VOICE_FILE} ({os.path.getsize(TARGET_VOICE_FILE) / 1024:.2f} KB)")
    else:
        print(f"[OK] Voice reference already present at {TARGET_VOICE_FILE}")

    print("\n[COMPLETE] Pocket TTS model & Indian Female voice reference ready!")
    print(f"Model Directory: {TARGET_MODEL_DIR}")
    print(f"Voice Reference: {TARGET_VOICE_FILE}")


if __name__ == "__main__":
    setup_pocket_tts()
