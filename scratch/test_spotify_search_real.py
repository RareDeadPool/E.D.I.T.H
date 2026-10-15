import sys, os, time, ctypes
from ctypes import wintypes
sys.path.insert(0, os.path.abspath("."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from runtime.actions import user32, find_window_by_process
from scratch.test_force_fg import force_foreground_window

hwnd = find_window_by_process(['spotify'])
print(f'Spotify HWND: {hwnd}')

# 1. Force foreground
assert force_foreground_window(hwnd), "Failed to foreground Spotify"
time.sleep(0.3)

# 2. In Spotify, Ctrl+K or Ctrl+L opens search
# Press Ctrl + L (0x4C)
user32.keybd_event(0x11, 0, 0, 0) # Ctrl down
user32.keybd_event(0x4C, 0, 0, 0) # L down
time.sleep(0.05)
user32.keybd_event(0x4C, 0, 2, 0) # L up
user32.keybd_event(0x11, 0, 2, 0) # Ctrl up
time.sleep(0.3)

# Press Ctrl + A to select all existing text
user32.keybd_event(0x11, 0, 0, 0) # Ctrl down
user32.keybd_event(0x41, 0, 0, 0) # A down
time.sleep(0.05)
user32.keybd_event(0x41, 0, 2, 0) # A up
user32.keybd_event(0x11, 0, 2, 0) # Ctrl up
time.sleep(0.05)

# 3. Type 'Believer'
for ch in "Believer":
    user32.keybd_event(0, ord(ch), 4, 0)
    time.sleep(0.02)
    user32.keybd_event(0, ord(ch), 6, 0)
    time.sleep(0.02)

time.sleep(0.3)
# 4. Press Enter
user32.keybd_event(0x0D, 0, 0, 0)
time.sleep(0.05)
user32.keybd_event(0x0D, 0, 2, 0)
time.sleep(1.5)

print('Search for Believer submitted! Now running screen engine perception...')
from runtime.gui_agent import GUIAgent
agent = GUIAgent()
elements = agent.see(hwnd=hwnd, force_ocr=True)

print(f'[+] Detected {len(elements)} elements on Spotify after search:')
for i, e in enumerate(elements[:30]):
    print(f'  {i:2d} | {e.source} | {e.role:10s} | {e.name:35s} | center={e.center}')
