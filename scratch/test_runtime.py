import sys
import os
import time

sys.path.insert(0, os.getcwd())
from runtime.executor import EdithTaskExecutor

exe = EdithTaskExecutor()
hwnd = exe._find_and_focus_spotify()
print("HWND:", hwnd)
if hwnd:
    img = exe._capture_window_image(hwnd)
    print("Captured image:", img.size if img else None)
    if img:
        img.save("scratch/current_spotify.png")
        print("Saved to scratch/current_spotify.png")
