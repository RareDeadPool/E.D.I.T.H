import sys
import os
import ctypes
from ctypes import wintypes
from PIL import Image

sys.path.insert(0, os.getcwd())
from runtime.executor import EdithTaskExecutor

exe = EdithTaskExecutor()
hwnd = exe._find_and_focus_spotify()
print("Spotify HWND:", hwnd)

u = ctypes.windll.user32
g = ctypes.windll.gdi32

rect = wintypes.RECT()
u.GetWindowRect(hwnd, ctypes.byref(rect))
w = rect.right - rect.left
h = rect.bottom - rect.top

hwndDC = u.GetWindowDC(hwnd)
memDC = g.CreateCompatibleDC(hwndDC)
bitmap = g.CreateCompatibleBitmap(hwndDC, w, h)
g.SelectObject(memDC, bitmap)
u.PrintWindow(hwnd, memDC, 2)

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
bih.biHeight = -h # Top-down DIB!
bih.biPlanes = 1
bih.biBitCount = 32
bih.biCompression = 0
bih.biSizeImage = w * h * 4

buf = (ctypes.c_char * (w * h * 4))()
g.GetDIBits(memDC, bitmap, 0, h, ctypes.byref(buf), ctypes.byref(bih), 0)
g.DeleteObject(bitmap)
g.DeleteDC(memDC)
u.ReleaseDC(hwnd, hwndDC)

img = Image.frombuffer("RGBA", (w, h), bytes(buf), "raw", "BGRA", 0, 1)
img.save("scratch/top_down_test.png")
print("Saved scratch/top_down_test.png")

# Now find green button on top-down image
clusters = []
for y in range(int(h * 0.1), int(h * 0.5)):
    for x in range(int(w * 0.15), int(w * 0.75)):
        r, g, b = img.getpixel((x, y))[:3]
        if g > 165 and r < 80 and b < 130 and (g - r > 100) and (g - b > 60):
            placed = False
            for c in clusters:
                if abs(c['cx'] - x) < 30 and abs(c['cy'] - y) < 30:
                    c['pts'].append((x, y))
                    c['cx'] = sum(p[0] for p in c['pts']) // len(c['pts'])
                    c['cy'] = sum(p[1] for p in c['pts']) // len(c['pts'])
                    placed = True
                    break
            if not placed:
                clusters.append({'pts': [(x, y)], 'cx': x, 'cy': y})

for i, c in enumerate(clusters):
    if len(c['pts']) > 30:
        print(f"Cluster {i}: Center=({c['cx']}, {c['cy']}), Count={len(c['pts'])}")
