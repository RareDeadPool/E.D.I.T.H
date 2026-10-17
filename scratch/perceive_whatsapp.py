import sys
import os
sys.path.insert(0, os.path.abspath("."))
import time
import ctypes
sys.stdout.reconfigure(encoding='utf-8')

user32 = ctypes.windll.user32
hdesk_default = user32.OpenDesktopW("default", 0, False, 0x01FF)
if hdesk_default:
    user32.SetThreadDesktop(hdesk_default)

from runtime import actions
from runtime.screen_engine import ScreenEngine

hwnd = actions.find_window_by_process(["whatsapp", "whatsapp.root"])
print("WhatsApp HWND:", hwnd)
t0 = time.time()
engine = ScreenEngine()
elements = engine.perceive(hwnd=hwnd)
t1 = time.time()
print(f"Perceived {len(elements)} elements in {(t1-t0)*1000:.1f}ms")

for el in elements:
    if any(k in el.name.lower() for k in ["search", "type a message", "send", "8591211342", "you", "attach", "chat"]):
        print(f"Role={el.role:10s} Name={el.name[:40]:40s} BBox={el.bbox} Center={el.center}")
