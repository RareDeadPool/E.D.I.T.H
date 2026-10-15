import ctypes
from ctypes import wintypes
from PIL import Image

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32

import sys, os
sys.path.insert(0, os.getcwd())
from runtime.executor import EdithTaskExecutor
exe = EdithTaskExecutor()
hwnd = exe._find_and_focus_spotify()

rect = wintypes.RECT()
user32.GetWindowRect(hwnd, ctypes.byref(rect))
w = rect.right - rect.left
h = rect.bottom - rect.top

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

# Case 1: bih.biHeight = -h
bih_neg = BITMAPINFOHEADER()
bih_neg.biSize = ctypes.sizeof(BITMAPINFOHEADER)
bih_neg.biWidth = w
bih_neg.biHeight = -h
bih_neg.biPlanes = 1
bih_neg.biBitCount = 32
bih_neg.biCompression = 0
bih_neg.biSizeImage = w * h * 4
buf_neg = (ctypes.c_char * (w * h * 4))()
res_neg = gdi32.GetDIBits(memDC, bitmap, 0, h, ctypes.byref(buf_neg), ctypes.byref(bih_neg), 0)
print("GetDIBits with -h returned lines:", res_neg)
if res_neg > 0:
    img_neg = Image.frombuffer("RGBA", (w, h), bytes(buf_neg), "raw", "BGRA", 0, 1)
    img_neg.save("scratch/test_dib_neg.png")
    # Check top pixel (search bar area y=30)
    print("test_dib_neg.png pixel at y=30:", img_neg.getpixel((int(w * 0.35), 30)))
    print("test_dib_neg.png pixel at y=h-30:", img_neg.getpixel((int(w * 0.35), h - 30)))
