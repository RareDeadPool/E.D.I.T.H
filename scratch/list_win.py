import ctypes
import ctypes.wintypes
import psutil

u = ctypes.windll.user32

hwinsta = u.OpenWindowStationW("WinSta0", False, 0x037F)
if hwinsta:
    u.SetProcessWindowStation(hwinsta)
hdesk = u.OpenDesktopW("Default", 0, False, 0x01FF)
if hdesk:
    u.SetThreadDesktop(hdesk)

spot_pids = {p.pid: p.name() for p in psutil.process_iter() if 'spot' in p.name().lower()}
print("Spot PIDs:", spot_pids)

def proc(h, l):
    p = ctypes.wintypes.DWORD()
    u.GetWindowThreadProcessId(h, ctypes.byref(p))
    c = ctypes.create_unicode_buffer(256)
    t = ctypes.create_unicode_buffer(256)
    r = ctypes.wintypes.RECT()
    u.GetClassNameW(h, c, 256)
    u.GetWindowTextW(h, t, 256)
    u.GetWindowRect(h, ctypes.byref(r))
    cls = c.value
    title = t.value
    
    # If pid is in spot_pids or 'spotify' in title.lower() or 'chrome_widgetwin' in cls.lower():
    if p.value in spot_pids or 'spotify' in title.lower() or (r.right - r.left > 300 and 'chrome_widgetwin' in cls.lower()):
        pname = spot_pids.get(p.value, "other")
        print(f"HWND: {h}, PID: {p.value} ({pname}), Class: {cls!r}, Title: {title!r}, Vis: {bool(u.IsWindowVisible(h))}, Rect: ({r.left},{r.top},{r.right},{r.bottom})")
    return True

W = ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
if hdesk:
    print("Enumerating via EnumDesktopWindows...")
    u.EnumDesktopWindows(hdesk, W(proc), 0)
print("Enumerating via EnumWindows...")
u.EnumWindows(W(proc), 0)
