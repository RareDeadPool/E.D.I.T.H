import sys
import ctypes
from ctypes import wintypes
import psutil

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

user32 = ctypes.windll.user32
hdesk = user32.OpenInputDesktop(0, False, 0x01FF)
user32.SetThreadDesktop(hdesk)

WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
user32.EnumDesktopWindows.argtypes = [wintypes.HANDLE, WNDENUMPROC, wintypes.LPARAM]
user32.EnumDesktopWindows.restype = wintypes.BOOL

all_windows = []
def cb(hwnd, lparam):
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    cbuf = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, cbuf, 256)
    tbuf = ctypes.create_unicode_buffer(256)
    user32.GetWindowTextW(hwnd, tbuf, 256)
    proc_name = ""
    try:
        proc_name = psutil.Process(pid.value).name()
    except Exception:
        pass
    all_windows.append((hwnd, pid.value, proc_name, cbuf.value, tbuf.value))
    return True

user32.EnumDesktopWindows(hdesk, WNDENUMPROC(cb), 0)

for h, p, name, cls, title in all_windows:
    if "spot" in name.lower() or "spot" in title.lower() or cls == "Chrome_WidgetWin_1":
        print(f"HWND: {h} | PID: {p} ({name}) | Class: '{cls}' | Title: '{title}'")
