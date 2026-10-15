import ctypes
from ctypes import wintypes

user32 = ctypes.windll.user32
hdesk = user32.OpenInputDesktop(0, False, 0x01FF)
user32.SetThreadDesktop(hdesk)

WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
user32.EnumDesktopWindows.argtypes = [wintypes.HANDLE, WNDENUMPROC, wintypes.LPARAM]
user32.EnumDesktopWindows.restype = wintypes.BOOL

found = []
def enum_proc(hwnd, lparam):
    tbuf = ctypes.create_unicode_buffer(512)
    user32.GetWindowTextW(hwnd, tbuf, 512)
    cbuf = ctypes.create_unicode_buffer(512)
    user32.GetClassNameW(hwnd, cbuf, 512)
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    vis = user32.IsWindowVisible(hwnd)
    if tbuf.value:
        found.append((hwnd, pid.value, vis, cbuf.value, tbuf.value))
    return True

user32.EnumDesktopWindows(hdesk, WNDENUMPROC(enum_proc), 0)
print(f"Total windows with titles: {len(found)}")
for h, p, v, c, t in found[:20]:
    print(f"HWND {h} | PID {p} | Vis: {v} | Class: '{c}' | Title: '{t}'")
