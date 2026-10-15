import time
from capture_spotify import capture_window
import ctypes
from ctypes import wintypes
from runtime.executor import EdithTaskExecutor

user32 = ctypes.windll.user32
hdesk = user32.OpenInputDesktop(0, False, 0x01FF)
user32.SetThreadDesktop(hdesk)

ex = EdithTaskExecutor()
hwnd = ex._find_and_focus_spotify()
print("Spotify HWND:", hwnd)

if hwnd:
    user32.ShowWindow(hwnd, 9)
    user32.SwitchToThisWindow(hwnd, True)
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.5)

    # Press Ctrl + K
    ex._spotify_press_ctrl_combo(0x4B)
    time.sleep(0.5)
    capture_window(hwnd, "spotify_after_ctrl_k.bmp")

    # Type 'Taare'
    ex._spotify_type_string("Taare")
    time.sleep(1.0)
    capture_window(hwnd, "spotify_after_type.bmp")

    # Press Down Arrow
    ex._spotify_press_key(0x28)
    time.sleep(0.3)
    capture_window(hwnd, "spotify_after_down.bmp")

    # Press Enter
    ex._spotify_press_key(0x0D)
    time.sleep(0.5)
    capture_window(hwnd, "spotify_after_enter.bmp")
    print("Done! Screenshots saved.")
