import ctypes
import os
import subprocess
import time
from ctypes import wintypes
from PIL import Image

def get_spotify_hwnd():
    user32 = ctypes.windll.user32
    spot_pids = set()
    import psutil
    for p in psutil.process_iter(['pid', 'name']):
        if 'spotify' in (p.info['name'] or '').lower():
            spot_pids.add(p.info['pid'])

    spotify_hwnd = None
    def enum_proc(hwnd, lparam):
        nonlocal spotify_hwnd
        if not user32.IsWindowVisible(hwnd):
            return True
        rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        if (rect.right - rect.left) < 200 or (rect.bottom - rect.top) < 200:
            return True
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        cbuf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, cbuf, 256)
        tbuf = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(hwnd, tbuf, 256)
        if cbuf.value == "Chrome_WidgetWin_1":
            if (spot_pids and pid.value in spot_pids) or "spotify" in tbuf.value.lower():
                spotify_hwnd = hwnd
                return False
        return True

    WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows(WNDENUMPROC(enum_proc), 0)
    return spotify_hwnd

def capture_window(hwnd):
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
    bih.biHeight = h
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

os.makedirs("scratch/cold_shots", exist_ok=True)
print("Killing spotify...")
os.system("taskkill /f /im spotify.exe /t >nul 2>&1")
time.sleep(1.0)

print("Launching Spotify cold...")
subprocess.Popen("cmd.exe /c start shell:AppsFolder\\SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify", shell=True)

start_t = time.time()
hwnd = None
for i in range(20):
    time.sleep(0.3)
    hwnd = get_spotify_hwnd()
    if hwnd:
        print(f"HWND {hwnd} found at {time.time() - start_t:.2f}s")
        break

if not hwnd:
    print("Could not find Spotify HWND")
    exit(1)

# Now monitor the window state and capture snapshots
for t_mark in [1.0, 2.0, 3.0, 4.0, 5.0]:
    elapsed = time.time() - start_t
    if elapsed < t_mark:
        time.sleep(t_mark - elapsed)
    img = capture_window(hwnd)
    if img:
        img.save(f"scratch/cold_shots/shot_{t_mark:.1f}s.png")
        # Check non-black/non-transparent pixels in center
        w, h = img.size
        # Sample pixels
        colors = [img.getpixel((int(w * 0.5), int(h * y_pct)))[:3] for y_pct in [0.2, 0.4, 0.6, 0.8]]
        print(f"At {t_mark}s ({img.size}): center sample colors = {colors}")
