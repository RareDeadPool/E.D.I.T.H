import ctypes
from ctypes import wintypes
import psutil

user32 = ctypes.windll.user32
titles = []

def cb(hwnd, _):
    if user32.IsWindowVisible(hwnd):
        length = user32.GetWindowTextLengthW(hwnd)
        buff = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buff, length + 1)
        t = buff.value
        
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        try:
            pname = psutil.Process(pid.value).name().lower()
        except Exception:
            pname = ""
        
        if t and ("chrome" in pname or "edge" in pname or "whatsapp" in t.lower() or "whatsapp" in pname):
            titles.append((hwnd, pname, t))
    return True

WNDENUM = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
user32.EnumWindows(WNDENUM(cb), 0)

print(f"Browser / WhatsApp windows found: {len(titles)}")
for h, p, t in titles:
    print(f"[{p}] HWND {h}: '{t}'")
