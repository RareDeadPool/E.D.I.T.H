import ctypes
import ctypes.wintypes
import time
import os

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass

u = ctypes.windll.user32
hwinsta = u.OpenWindowStationW("WinSta0", False, 0x037F)
if hwinsta:
    u.SetProcessWindowStation(hwinsta)
hdesk = u.OpenDesktopW("Default", 0, False, 0x01FF)
if hdesk:
    u.SetThreadDesktop(hdesk)

pt = ctypes.wintypes.POINT()
u.GetCursorPos(ctypes.byref(pt))
print(f"Current cursor before move: ({pt.x}, {pt.y})")

# Let's set to (578, 224)
u.SetCursorPos(578, 224)
time.sleep(0.05)
u.GetCursorPos(ctypes.byref(pt))
print(f"Cursor after SetCursorPos(578, 224): ({pt.x}, {pt.y})")

# Now let's click and check media!
u.mouse_event(0x0002, 0, 0, 0, 0)
time.sleep(0.05)
u.mouse_event(0x0004, 0, 0, 0, 0)
time.sleep(1.0)

os.system("powershell -ExecutionPolicy Bypass -File scratch/check_media.ps1")
