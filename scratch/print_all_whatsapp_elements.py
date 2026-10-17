import sys
import os
sys.path.insert(0, os.path.abspath("."))
import ctypes

# Set utf-8 encoding for stdout
sys.stdout.reconfigure(encoding='utf-8')

user32 = ctypes.windll.user32
hdesk_default = user32.OpenDesktopW("default", 0, False, 0x01FF)
user32.SetThreadDesktop(hdesk_default)

from runtime import actions
hwnd = actions.find_window_by_process(["whatsapp", "whatsapp.root"])

from runtime.screen_engine import ScreenEngine
engine = ScreenEngine()
elements = engine.perceive(hwnd=hwnd)

print(f"--- ALL {len(elements)} ELEMENTS IN WHATSAPP WINDOW ---")
for i, el in enumerate(elements):
    clean_name = el.name.encode('ascii', errors='replace').decode('ascii')
    print(f"[{i:02d}] {el.role:12} | Name: {clean_name:35} | BBox: {el.bbox} | AutoId: '{el.automationId}' | Class: '{el.className}'")
