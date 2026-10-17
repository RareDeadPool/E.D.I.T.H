import ctypes
from ctypes import wintypes
import psutil

user32 = ctypes.windll.user32
windows = []

def enum_cb(hwnd, extra):
    if user32.IsWindowVisible(hwnd):
        length = user32.GetWindowTextLengthW(hwnd)
        buff = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buff, length + 1)
        t = buff.value
        
        class_buff = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, class_buff, 256)
        cname = class_buff.value

        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        try:
            pname = psutil.Process(pid.value).name()
        except Exception:
            pname = ""

        rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        w = rect.right - rect.left
        h = rect.bottom - rect.top

        if w > 100 and h > 100:
            windows.append({
                "hwnd": hwnd,
                "pid": pid.value,
                "process": pname,
                "class": cname,
                "title": t,
                "size": (w, h),
                "pos": (rect.left, rect.top)
            })
    return True

WNDENUM = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
user32.EnumWindows(WNDENUM(enum_cb), 0)

for win in windows:
    if win['title'] or 'whatsapp' in win['process'].lower() or 'chrome' in win['process'].lower():
        print(f"HWND {win['hwnd']} | Process: {win['process']:20} | Class: {win['class']:25} | Title: '{win['title']}' | Size: {win['size']}")
