import ctypes
import os
import subprocess
import time
import sys
import re
import urllib.parse
from ctypes import wintypes
from PIL import Image

sys.path.insert(0, os.getcwd())
from runtime.executor import EdithTaskExecutor

def capture_topdown(hwnd):
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    w = rect.right - rect.left
    h = rect.bottom - rect.top
    if w <= 0 or h <= 0:
        return None
    hwndDC = user32.GetWindowDC(hwnd)
    memDC = gdi32.CreateCompatibleDC(hwndDC)
    bitmap = gdi32.CreateCompatibleBitmap(hwndDC, w, h)
    gdi32.SelectObject(memDC, bitmap)
    user32.PrintWindow(hwnd, memDC, 2)

    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", wintypes.DWORD),
            ("biWidth", wintypes.LONG),
            ("biHeight", wintypes.LONG),
            ("biPlanes", wintypes.WORD),
            ("biBitCount", wintypes.WORD),
            ("biCompression", wintypes.DWORD),
            ("biSizeImage", wintypes.DWORD),
            ("biXPelsPerMeter", wintypes.LONG),
            ("biYPelsPerMeter", wintypes.LONG),
            ("biClrUsed", wintypes.DWORD),
            ("biClrImportant", wintypes.DWORD)
        ]
    bih = BITMAPINFOHEADER()
    bih.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bih.biWidth = w
    bih.biHeight = -h
    bih.biPlanes = 1
    bih.biBitCount = 32
    bih.biCompression = 0
    bih.biSizeImage = w * h * 4
    buf = (ctypes.c_char * (w * h * 4))()
    gdi32.GetDIBits(memDC, bitmap, 0, h, ctypes.byref(buf), ctypes.byref(bih), 0)
    gdi32.DeleteObject(bitmap)
    gdi32.DeleteDC(memDC)
    user32.ReleaseDC(hwnd, hwndDC)
    return Image.frombuffer("RGBA", (w, h), bytes(buf), "raw", "BGRA", 0, 1)

def find_green_button(img):
    w, h = img.size
    clusters = []
    for y in range(int(h * 0.15), int(h * 0.55)):
        for x in range(int(w * 0.15), int(w * 0.75)):
            r, g, b = img.getpixel((x, y))[:3]
            if g > 165 and r < 80 and b < 130 and (g - r > 100) and (g - b > 60):
                placed = False
                for c in clusters:
                    if abs(c["cx"] - x) < 30 and abs(c["cy"] - y) < 30:
                        c["pts"].append((x, y))
                        c["cx"] = sum(p[0] for p in c["pts"]) // len(c["pts"])
                        c["cy"] = sum(p[1] for p in c["pts"]) // len(c["pts"])
                        placed = True
                        break
                if not placed:
                    clusters.append({"pts": [(x, y)], "cx": x, "cy": y})
    valid = [c for c in clusters if len(c["pts"]) > 30]
    if valid:
        valid.sort(key=lambda c: len(c["pts"]), reverse=True)
        return (valid[0]["cx"], valid[0]["cy"])
    return None

def test_cold_start_play(target_title):
    print(f"\n--- Testing cold start play for: '{target_title}' ---")
    print("1. Killing Spotify...")
    os.system("taskkill /f /im spotify.exe /t >nul 2>&1")
    time.sleep(1.0)

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass

    user32 = ctypes.windll.user32
    hwinsta = user32.OpenWindowStationW("WinSta0", False, 0x037F)
    if hwinsta:
        user32.SetProcessWindowStation(hwinsta)
    hdesk = user32.OpenDesktopW("Default", 0, False, 0x01FF)
    if hdesk:
        user32.SetThreadDesktop(hdesk)

    exe = EdithTaskExecutor()

    # Launch Spotify
    print("2. Launching Spotify cold...")
    t0 = time.time()
    subprocess.Popen("cmd.exe /c start shell:AppsFolder\\SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify", shell=True)

    hwnd = None
    for _ in range(25):
        time.sleep(0.3)
        hwnd = exe._find_and_focus_spotify()
        if hwnd:
            print(f"HWND {hwnd} detected at {time.time() - t0:.2f}s")
            break

    if not hwnd:
        print("Failed to find HWND")
        return False

    # Wait for UI to mount
    print("3. Waiting for Spotify DOM initialization...")
    ready = False
    for _ in range(30):
        time.sleep(0.3)
        img = capture_topdown(hwnd)
        if img:
            w, h = img.size
            top_pixels = [img.getpixel((int(w * x), int(h * 0.03)))[:3] for x in [0.2, 0.35, 0.5, 0.7]]
            distinct = set(top_pixels)
            if len(distinct) > 1 or (0, 0, 0) in distinct:
                print(f"DOM mounted at {time.time() - t0:.2f}s")
                ready = True
                break

    time.sleep(1.0) # Let React attach event listeners

    # Bring to foreground
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

    # Click search pill and press Ctrl + L
    search_click_x = rect.left + int(win_w * 0.35)
    search_click_y = rect.top + 32
    print(f"4. Focusing search bar at ({search_click_x}, {search_click_y})...")
    user32.SetCursorPos(search_click_x, search_click_y)
    time.sleep(0.05)
    user32.mouse_event(0x0002, 0, 0, 0, 0)
    time.sleep(0.05)
    user32.mouse_event(0x0004, 0, 0, 0, 0)
    time.sleep(0.1)

    exe._spotify_press_ctrl_combo(0x4C) # Ctrl + L
    time.sleep(0.1)

    # Clear previous text
    exe._spotify_press_ctrl_combo(0x41) # Ctrl + A
    time.sleep(0.05)
    exe._spotify_press_key(0x08)        # Backspace
    time.sleep(0.1)

    # Type query and submit
    print(f"5. Typing query '{target_title}' and submitting...")
    exe._spotify_type_string(target_title)
    time.sleep(0.2)
    exe._spotify_press_key(0x0D) # VK_RETURN

    # Hover over Top Result area
    hover_x = rect.left + int(win_w * 0.22)
    hover_y = rect.top + int(win_h * 0.35)
    user32.SetCursorPos(hover_x, hover_y)
    user32.mouse_event(0x0001, 1, 1, 0, 0)

    # Poll for green play button (results take ~1-2s over network)
    print("6. Polling for Top Result green play button...")
    btn_pos = None
    img_res = None
    for i in range(12):
        time.sleep(0.3)
        img_res = capture_topdown(hwnd)
        if img_res:
            btn_pos = find_green_button(img_res)
            if btn_pos:
                print(f"Green button found at iteration {i}: {btn_pos}")
                break

    clicked = False
    if btn_pos and img_res:
        scale_x = win_w / img_res.size[0] if img_res.size[0] > 0 else 1.0
        scale_y = win_h / img_res.size[1] if img_res.size[1] > 0 else 1.0
        click_x = rect.left + int(btn_pos[0] * scale_x)
        click_y = rect.top + int(btn_pos[1] * scale_y)
        print(f"7. Clicking green play button at ({click_x}, {click_y})...")
        user32.SetCursorPos(click_x, click_y)
        time.sleep(0.05)
        user32.mouse_event(0x0002, 0, 0, 0, 0)
        time.sleep(0.05)
        user32.mouse_event(0x0004, 0, 0, 0, 0)
        clicked = True
    else:
        print(f"7. Fallback: double-clicking Top Result card at ({hover_x}, {hover_y})...")
        user32.SetCursorPos(hover_x, hover_y)
        time.sleep(0.05)
        user32.mouse_event(0x0002, 0, 0, 0, 0)
        user32.mouse_event(0x0004, 0, 0, 0, 0)
        time.sleep(0.05)
        user32.mouse_event(0x0002, 0, 0, 0, 0)
        user32.mouse_event(0x0004, 0, 0, 0, 0)
        clicked = True

    time.sleep(1.5)
    print("8. Checking current media session...")
    os.system("powershell -ExecutionPolicy Bypass -File scratch/check_media.ps1")
    return True

test_cold_start_play("Starboy")
