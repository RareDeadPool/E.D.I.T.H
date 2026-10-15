import os
import subprocess
import time
import sys

sys.path.insert(0, os.getcwd())
from runtime.executor import EdithTaskExecutor

print("1. Killing Spotify completely...")
os.system("taskkill /f /im spotify.exe /t >nul 2>&1")
time.sleep(1.5)

print("2. Launching via start spotify:search:believer ...")
t0 = time.time()
subprocess.Popen("cmd.exe /c start spotify:search:believer", shell=True)

exe = EdithTaskExecutor()
hwnd = None
for i in range(30):
    time.sleep(0.3)
    hwnd = exe._find_and_focus_spotify()
    if hwnd:
        print(f"HWND {hwnd} found at {time.time() - t0:.2f}s")
        break

if not hwnd:
    print("Could not find HWND with spotify:search:believer")
    exit(1)

print("Waiting 5.5s for Spotify to fully load search page...")
time.sleep(5.5)

img = exe._capture_window_image(hwnd)
if img:
    from PIL import Image
    img_flipped = img.transpose(Image.FLIP_TOP_BOTTOM)
    img_flipped.save("scratch/cold_search_result.png")
    print("Saved scratch/cold_search_result.png")
