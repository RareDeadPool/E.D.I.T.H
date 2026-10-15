import ctypes
import os
import subprocess
import time
import sys

sys.path.insert(0, os.getcwd())
from runtime.executor import EdithTaskExecutor

print("1. Killing Spotify...")
os.system("taskkill /f /im spotify.exe /t >nul 2>&1")
time.sleep(1.0)

exe = EdithTaskExecutor()

print("2. Starting timer and launching Spotify...")
t0 = time.time()
subprocess.Popen(
    "cmd.exe /c start shell:AppsFolder\\SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify",
    shell=True
)

hwnd = None
while time.time() - t0 < 10.0:
    time.sleep(0.2)
    hwnd = exe._find_and_focus_spotify()
    if hwnd:
        print(f"HWND {hwnd} found at {time.time() - t0:.2f}s")
        break

if not hwnd:
    print("Failed to find HWND")
    exit(1)

# Now let's see how the window content evolves
for delay in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
    elapsed = time.time() - t0
    img = exe._capture_window_image(hwnd)
    if img:
        # Check some pixel variance across the image to see if it's rendered or black/blank
        colors = [img.getpixel((int(img.width * 0.4), int(img.height * y)))[:3] for y in [0.1, 0.3, 0.5, 0.7]]
        print(f"Elapsed: {elapsed:.2f}s, size: {img.size}, sample colors: {colors}")
    time.sleep(0.5)
