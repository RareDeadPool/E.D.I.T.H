import ctypes
from ctypes import wintypes
import time
import os

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32

def capture_window(hwnd, save_path):
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.3)
    
    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    w = rect.right - rect.left
    h = rect.bottom - rect.top
    if w <= 0 or h <= 0:
        print("Invalid window dimensions:", w, h)
        return False
        
    hwndDC = user32.GetWindowDC(hwnd)
    memDC = gdi32.CreateCompatibleDC(hwndDC)
    bitmap = gdi32.CreateCompatibleBitmap(hwndDC, w, h)
    gdi32.SelectObject(memDC, bitmap)
    
    # PrintWindow flag 2 = PW_RENDERFULLCONTENT
    user32.PrintWindow(hwnd, memDC, 2)
    
    # Save bitmap to file
    import struct
    # We can write a simple BMP file
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
    
    # BMP header: 'BM' + size + 0 + offset (54)
    file_size = 54 + len(buf)
    bmp_header = struct.pack("<2sIHHI", b"BM", file_size, 0, 0, 54)
    info_header = struct.pack("<IIIHHIIIIII", 40, w, h, 1, 32, 0, len(buf), 0, 0, 0, 0)
    
    with open(save_path, "wb") as f:
        f.write(bmp_header)
        f.write(info_header)
        f.write(buf)
        
    gdi32.DeleteObject(bitmap)
    gdi32.DeleteDC(memDC)
    user32.ReleaseDC(hwnd, hwndDC)
    print(f"Captured window {hwnd} ({w}x{h}) to {save_path}")
    return True

from runtime.executor import EdithTaskExecutor
ex = EdithTaskExecutor()
hwnd = ex._find_and_focus_spotify()
print("Spotify HWND:", hwnd)
if hwnd:
    capture_window(hwnd, "spotify_screen.bmp")
