import ctypes
from ctypes import wintypes
import psutil

user32 = ctypes.windll.user32

def inspect_all():
    found = []
    def enum_cb(hwnd, _):
        length = user32.GetWindowTextLengthW(hwnd)
        buff = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buff, length + 1)
        title = buff.value
        
        cls_buff = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, cls_buff, 256)
        cls_name = cls_buff.value

        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        try:
            pname = psutil.Process(pid.value).name()
        except Exception:
            pname = ""

        vis = user32.IsWindowVisible(hwnd)
        rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        w = rect.right - rect.left
        h = rect.bottom - rect.top

        if "whatsapp" in title.lower() or "whatsapp" in cls_name.lower() or "whatsapp" in pname.lower():
            found.append({
                "hwnd": hwnd,
                "title": title,
                "class": cls_name,
                "pid": pid.value,
                "pname": pname,
                "visible": vis,
                "size": (w, h),
                "rect": (rect.left, rect.top, rect.right, rect.bottom)
            })
        return True

    WNDENUM = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows(WNDENUM(enum_cb), 0)
    return found

res = inspect_all()
print(f"WhatsApp windows in entire OS: {len(res)}")
for r in res:
    print(r)
