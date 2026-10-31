import os
import sys
from huggingface_hub import hf_hub_download

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEST_DIR = os.path.join(PROJECT_ROOT, "models", "llm")
os.makedirs(DEST_DIR, exist_ok=True)

REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

print(f"Downloading {FILENAME} from {REPO_ID} to {DEST_DIR}...")
local_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=FILENAME,
    local_dir=DEST_DIR,
    local_dir_use_symlinks=False
)
print(f"Successfully downloaded to: {local_path}")
print(f"File size: {os.path.getsize(local_path) / (1024 * 1024):.2f} MB")
