import sys, os, time, ctypes
from ctypes import wintypes
sys.path.insert(0, os.path.abspath("."))
from runtime.actions import user32, find_window_by_process

def force_foreground_window(hwnd: int) -> bool:
    if not hwnd or not user32.IsWindow(hwnd):
        return False

    fg_hwnd = user32.GetForegroundWindow()
    if fg_hwnd == hwnd:
        return True

    # Win32 Alt-key trick: unlocks SetForegroundWindow for any background process
    user32.keybd_event(0x12, 0, 0, 0)  # VK_MENU (Alt) down
    user32.keybd_event(0x12, 0, 2, 0)  # VK_MENU (Alt) up

    fg_pid = wintypes.DWORD()
    fg_tid = user32.GetWindowThreadProcessId(fg_hwnd, ctypes.byref(fg_pid))
    cur_tid = ctypes.windll.kernel32.GetCurrentThreadId()

    user32.AttachThreadInput(cur_tid, fg_tid, True)
    user32.ShowWindow(hwnd, 9)  # SW_RESTORE
    user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002)
    user32.SetWindowPos(hwnd, -2, 0, 0, 0, 0, 0x0001 | 0x0002)
    user32.SetForegroundWindow(hwnd)
    user32.BringWindowToTop(hwnd)
    user32.AttachThreadInput(cur_tid, fg_tid, False)
    time.sleep(0.15)
    return user32.GetForegroundWindow() == hwnd

hwnd = find_window_by_process(['spotify'])
print(f'Spotify HWND: {hwnd}')
print(f'Initial FG Window: {user32.GetForegroundWindow()}')

success = force_foreground_window(hwnd)
print(f'Foreground success: {success}')
print(f'Current FG Window: {user32.GetForegroundWindow()}')
