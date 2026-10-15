import ctypes
from ctypes import wintypes
import psutil

user32 = ctypes.windll.user32
hdesk = user32.OpenInputDesktop(0, False, 0x01FF)
if hdesk:
    user32.SetThreadDesktop(hdesk)

spotify_pids = {p.info['pid'] for p in psutil.process_iter(['pid', 'name']) if 'spot' in (p.info['name'] or '').lower()}
print("Spotify PIDs:", spotify_pids)

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
    
    if pid.value in spotify_pids or "spotify" in tbuf.value.lower() or "spotify" in cbuf.value.lower():
        found.append((hwnd, pid.value, vis, cbuf.value, tbuf.value))
    return True

user32.EnumDesktopWindows(hdesk, WNDENUMPROC(enum_proc), 0)
print(f"Total found: {len(found)}")
for h, p, v, c, t in found:
    print(f"HWND {h} | PID {p} | Vis: {v} | Class: '{c}' | Title: '{t}'")
