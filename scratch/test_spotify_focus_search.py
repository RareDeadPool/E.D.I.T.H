import sys, os
sys.path.insert(0, os.path.abspath("."))
import ctypes
from ctypes import wintypes
import time
from runtime.actions import user32, find_window_by_process

hwnd = find_window_by_process(['spotify'])
print(f'Spotify HWND: {hwnd}')

# 1. Force foreground using AttachThreadInput
fg_hwnd = user32.GetForegroundWindow()
fg_pid = wintypes.DWORD()
fg_tid = user32.GetWindowThreadProcessId(fg_hwnd, ctypes.byref(fg_pid))
cur_tid = ctypes.windll.kernel32.GetCurrentThreadId()

user32.AttachThreadInput(cur_tid, fg_tid, True)
user32.SetForegroundWindow(hwnd)
user32.ShowWindow(hwnd, 9)
user32.AttachThreadInput(cur_tid, fg_tid, False)
time.sleep(0.4)

now_fg = user32.GetForegroundWindow()
print(f'Is Spotify now foreground? {now_fg == hwnd} (Current FG: {now_fg})')

# 2. Click directly on search pill at (410, 41)
user32.SetCursorPos(410, 41)
time.sleep(0.05)
user32.mouse_event(0x0002, 0, 0, 0, 0)
time.sleep(0.05)
user32.mouse_event(0x0004, 0, 0, 0, 0)
time.sleep(0.2)

# 3. Press Ctrl+A, type Believer, press Enter
user32.keybd_event(0x11, 0, 0, 0) # Ctrl
user32.keybd_event(0x41, 0, 0, 0) # A
time.sleep(0.05)
user32.keybd_event(0x41, 0, 2, 0)
user32.keybd_event(0x11, 0, 2, 0)
time.sleep(0.05)

# Type 'Believer'
for ch in "Believer":
    user32.keybd_event(0, ord(ch), 4, 0)
    time.sleep(0.02)
    user32.keybd_event(0, ord(ch), 6, 0)
    time.sleep(0.02)

time.sleep(0.2)
# Press Enter (VK_RETURN = 0x0D)
user32.keybd_event(0x0D, 0, 0, 0)
time.sleep(0.05)
user32.keybd_event(0x0D, 0, 2, 0)

print('Search action sent! Waiting 2 seconds...')
time.sleep(2.0)
