import ctypes
import os
import subprocess
import time
import sys
from ctypes import wintypes
from PIL import Image

sys.path.insert(0, os.getcwd())
from runtime.executor import EdithTaskExecutor

print("1. Killing Spotify...")
os.system("taskkill /f /im spotify.exe /t >nul 2>&1")
time.sleep(1.0)

exe = EdithTaskExecutor()

print("2. Launching Spotify cold...")
t0 = time.time()
subprocess.Popen(
    "cmd.exe /c start shell:AppsFolder\\SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify",
    shell=True
)

hwnd = None
for _ in range(20):
    time.sleep(0.3)
    hwnd = exe._find_and_focus_spotify()
    if hwnd:
        print(f"HWND {hwnd} found at {time.time() - t0:.2f}s")
        break

if not hwnd:
    print("Could not find HWND")
    exit(1)

# Wait until the UI actually renders (monitor window image until not solid color)
print("Waiting for Spotify UI to mount...")
ready = False
for _ in range(30):
    time.sleep(0.3)
    img = exe._capture_window_image(hwnd)
    if img:
        # Check if top bar has rendered (top bar y~30 has non-(18,18,18) pixels like white icons or black bg)
        w, h = img.size
        top_pixels = [img.getpixel((int(w * x), int(h * 0.03)))[:3] for x in [0.2, 0.35, 0.5, 0.7]]
        # In unrendered state, all pixels are (18,18,18) or (32,32,32)
        distinct = set(top_pixels)
        if len(distinct) > 1 or (0, 0, 0) in distinct:
            print(f"UI rendered at {time.time() - t0:.2f}s: sample top pixels = {top_pixels}")
            ready = True
            break

if not ready:
    print("UI render timeout, proceeding anyway...")

time.sleep(0.5)

# Bring Spotify to front
user32 = ctypes.windll.user32
user32.AllowSetForegroundWindow(0xFFFFFFFF)
user32.ShowWindow(hwnd, 9)
user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002)
user32.SetWindowPos(hwnd, -2, 0, 0, 0, 0, 0x0001 | 0x0002)
user32.SetForegroundWindow(hwnd)
time.sleep(0.3)

rect = wintypes.RECT()
user32.GetWindowRect(hwnd, ctypes.byref(rect))
win_w = rect.right - rect.left
win_h = rect.bottom - rect.top

# Click directly on the search bar in the top navigation bar!
# In Spotify desktop, top bar search pill is centered at x ~ 35% of window width, y ~ 32px
search_click_x = rect.left + int(win_w * 0.35)
search_click_y = rect.top + 32
print(f"Clicking search bar at ({search_click_x}, {search_click_y})...")
user32.SetCursorPos(search_click_x, search_click_y)
time.sleep(0.1)
user32.mouse_event(0x0002, 0, 0, 0, 0)
time.sleep(0.05)
user32.mouse_event(0x0004, 0, 0, 0, 0)
time.sleep(0.2)

# Select all and clear any existing text
exe._spotify_press_ctrl_combo(0x41) # Ctrl + A
time.sleep(0.05)
exe._spotify_press_key(0x08)        # Backspace
time.sleep(0.1)

# Type target query and press Enter
print("Typing Believer and pressing Enter...")
exe._spotify_type_string("Believer")
time.sleep(0.2)
exe._spotify_press_key(0x0D) # VK_RETURN
time.sleep(1.5)

# Take screenshot of search results
img_search = exe._capture_window_image(hwnd)
if img_search:
    img_flipped = img_search.transpose(Image.FLIP_TOP_BOTTOM)
    img_flipped.save("scratch/test_search_results.png")
    print("Saved scratch/test_search_results.png")

# Now hover over Top Result card and look for green button
hover_x = rect.left + int(win_w * 0.22)
hover_y = rect.top + int(win_h * 0.35)
user32.SetCursorPos(hover_x, hover_y)
user32.mouse_event(0x0001, 1, 1, 0, 0)
time.sleep(0.4)

img_hover = exe._capture_window_image(hwnd)
btn_pos = exe._find_spotify_green_play_button(img_hover) if img_hover else None
print("Green button found at:", btn_pos)

if btn_pos and img_hover:
    scale_x = win_w / img_hover.size[0] if img_hover.size[0] > 0 else 1.0
    scale_y = win_h / img_hover.size[1] if img_hover.size[1] > 0 else 1.0
    click_x = rect.left + int(btn_pos[0] * scale_x)
    click_y = rect.top + int(btn_pos[1] * scale_y)
    print(f"Clicking green play button at ({click_x}, {click_y})")
    user32.SetCursorPos(click_x, click_y)
    time.sleep(0.1)
    user32.mouse_event(0x0002, 0, 0, 0, 0)
    time.sleep(0.05)
    user32.mouse_event(0x0004, 0, 0, 0, 0)
else:
    print(f"Double clicking Top Result card at ({hover_x}, {hover_y})")
    user32.SetCursorPos(hover_x, hover_y)
    time.sleep(0.05)
    user32.mouse_event(0x0002, 0, 0, 0, 0)
    user32.mouse_event(0x0004, 0, 0, 0, 0)
    time.sleep(0.05)
    user32.mouse_event(0x0002, 0, 0, 0, 0)
    user32.mouse_event(0x0004, 0, 0, 0, 0)

time.sleep(1.5)

# Check media transport
print("Checking playing song via check_media.ps1...")
os.system("powershell -ExecutionPolicy Bypass -File scratch/check_media.ps1")
