"""Generic OS-level input action primitives for E.D.I.T.H.

Provides resolution-independent, application-agnostic mouse/keyboard
actions that any module can call without reimplementing Win32 boilerplate.
"""

import time
import ctypes
from ctypes import wintypes
from typing import Optional, Tuple, Any, List

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

# DPI Awareness
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
except Exception:
    try:
        user32.SetProcessDPIAware()
    except Exception:
        pass


# ──────────────────────── Win32 SendInput Structures ────────────────────────

_extra = ctypes.c_ulong(0)


class _KeyBdInput(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _HardwareInput(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _InputUnion(ctypes.Union):
    _fields_ = [("ki", _KeyBdInput), ("mi", _MouseInput), ("hi", _HardwareInput)]


class _Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("ii", _InputUnion)]


INPUT_KEYBOARD = 1
INPUT_MOUSE = 0

KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_UNICODE = 0x0004

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800

# ──────────────────────── Low-Level Helpers ─────────────────────────────────


def _send_key(vk: int, flags: int = 0):
    """Send a single keyboard event via SendInput."""
    scan = user32.MapVirtualKeyW(vk, 0)
    ii = _InputUnion()
    ii.ki = _KeyBdInput(vk, scan, flags, 0, ctypes.pointer(_extra))
    inp = _Input(ctypes.c_ulong(INPUT_KEYBOARD), ii)
    user32.SendInput(1, ctypes.pointer(inp), ctypes.sizeof(inp))


def _send_mouse(flags: int, dx: int = 0, dy: int = 0, data: int = 0):
    """Send a single mouse event via SendInput."""
    ii = _InputUnion()
    ii.mi = _MouseInput(dx, dy, data, flags, 0, ctypes.pointer(_extra))
    inp = _Input(ctypes.c_ulong(INPUT_MOUSE), ii)
    user32.SendInput(1, ctypes.pointer(inp), ctypes.sizeof(inp))


# ──────────────────────── Public API ────────────────────────────────────────


def click(x: int, y: int, button: str = "left", clicks: int = 1, delay: float = 0.04):
    """Click at absolute screen coordinates (x, y).

    Args:
        x: Absolute X pixel coordinate on screen.
        y: Absolute Y pixel coordinate on screen.
        button: 'left', 'right', or 'middle'.
        clicks: Number of clicks (1 = single, 2 = double).
        delay: Seconds between mouse-down and mouse-up.
    """
    user32.SetCursorPos(x, y)
    time.sleep(0.02)

    down_flag, up_flag = {
        "left": (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
        "right": (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP),
        "middle": (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP),
    }.get(button, (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP))

    for _ in range(clicks):
        _send_mouse(down_flag)
        time.sleep(delay)
        _send_mouse(up_flag)
        time.sleep(delay)


def double_click(x: int, y: int, button: str = "left"):
    """Double-click at absolute screen coordinates."""
    click(x, y, button=button, clicks=2, delay=0.03)


def move_to(x: int, y: int):
    """Move the mouse cursor to absolute screen coordinates."""
    user32.SetCursorPos(x, y)


def scroll(x: int, y: int, amount: int = -3):
    """Scroll the mouse wheel at absolute screen coordinates.

    Args:
        amount: Positive = scroll up, Negative = scroll down.
                Each unit is 120 (one Windows wheel notch).
    """
    user32.SetCursorPos(x, y)
    time.sleep(0.02)
    _send_mouse(MOUSEEVENTF_WHEEL, data=amount * 120)


VK_NAMES = {
    "ctrl": 0x11, "control": 0x11, "shift": 0x10, "alt": 0x12,
    "enter": 0x0D, "return": 0x0D, "esc": 0x1B, "escape": 0x1B,
    "tab": 0x09, "backspace": 0x08, "space": 0x20,
    "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
    "win": 0x5B, "windows": 0x5B, "cmd": 0x5B,
    "delete": 0x2E, "del": 0x2E, "home": 0x24, "end": 0x23,
}


def _normalize_vk(key: Any) -> int:
    """Convert virtual key code or string key name ('enter', 'ctrl', 'v') to integer VK."""
    if isinstance(key, int):
        return key
    if isinstance(key, str):
        k_low = key.lower().strip()
        if k_low in VK_NAMES:
            return VK_NAMES[k_low]
        if len(k_low) == 1:
            return ord(k_low.upper())
    return int(key)


def press_key(key: Any, delay: float = 0.04):
    """Press and release a single virtual key (int VK or string name like 'enter')."""
    vk = _normalize_vk(key)
    _send_key(vk, 0)
    time.sleep(delay)
    _send_key(vk, KEYEVENTF_KEYUP)
    time.sleep(delay)


def hotkey(*keys: Any, delay: float = 0.04):
    """Press a key combination (e.g. hotkey('ctrl', 'v') or hotkey(0x11, 0x41)).

    Keys are pressed in order and released in reverse order.
    """
    vks = [_normalize_vk(k) for k in keys]
    for vk in vks:
        _send_key(vk, 0)
        time.sleep(delay)
    for vk in reversed(vks):
        _send_key(vk, KEYEVENTF_KEYUP)
        time.sleep(delay)


def type_text(text: str, delay: float = 0.01):
    """Type a string character-by-character using Unicode SendInput events.

    This works with any keyboard layout and handles Unicode characters.
    """
    for ch in text:
        code = ord(ch)
        # Key down (unicode)
        ii_down = _InputUnion()
        ii_down.ki = _KeyBdInput(0, code, KEYEVENTF_UNICODE, 0, ctypes.pointer(_extra))
        inp_down = _Input(ctypes.c_ulong(INPUT_KEYBOARD), ii_down)
        user32.SendInput(1, ctypes.pointer(inp_down), ctypes.sizeof(inp_down))
        time.sleep(delay)
        # Key up (unicode)
        ii_up = _InputUnion()
        ii_up.ki = _KeyBdInput(
            0, code, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, 0, ctypes.pointer(_extra)
        )
        inp_up = _Input(ctypes.c_ulong(INPUT_KEYBOARD), ii_up)
        user32.SendInput(1, ctypes.pointer(inp_up), ctypes.sizeof(inp_up))
        time.sleep(delay)


def press_media_key(vk: int):
    """Press a media virtual key (VK_MEDIA_PLAY_PAUSE = 0xB3, etc.)."""
    scan = user32.MapVirtualKeyW(vk, 0)
    user32.keybd_event(vk, scan, KEYEVENTF_EXTENDEDKEY, 0)
    time.sleep(0.04)
    user32.keybd_event(vk, scan, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0)


# ──────────────────────── Background / Cursorless Actions ───────────────────


def screen_to_client(hwnd: int, screen_x: int, screen_y: int) -> Tuple[int, int]:
    """Convert screen coordinates to window-relative client coordinates."""
    pt = wintypes.POINT(screen_x, screen_y)
    user32.ScreenToClient(hwnd, ctypes.byref(pt))
    return (pt.x, pt.y)


def post_click_background(
    hwnd: int,
    screen_x: int,
    screen_y: int,
    button: str = "left",
    clicks: int = 1,
) -> bool:
    """Send click message directly to a window's message queue without moving the physical mouse.

    Args:
        hwnd: Target window handle.
        screen_x: Absolute screen X coordinate.
        screen_y: Absolute screen Y coordinate.
        button: 'left' or 'right'.
        clicks: 1 for single click, 2 for double click.

    Returns:
        True if the message was posted successfully.
    """
    if not hwnd or not user32.IsWindow(hwnd):
        return False

    client_x, client_y = screen_to_client(hwnd, screen_x, screen_y)
    lparam = (client_y << 16) | (client_x & 0xFFFF)

    down_msg, up_msg, wparam = (
        (0x0201, 0x0202, 0x0001) if button == "left" else (0x0204, 0x0205, 0x0002)
    )

    for _ in range(clicks):
        user32.PostMessageW(hwnd, down_msg, wparam, lparam)
        time.sleep(0.02)
        user32.PostMessageW(hwnd, up_msg, 0, lparam)
        time.sleep(0.02)

    return True


def post_text_background(hwnd: int, text: str, delay: float = 0.01) -> bool:
    """Send characters directly to a window's message queue without stealing keyboard focus."""
    if not hwnd or not user32.IsWindow(hwnd):
        return False
    for ch in text:
        user32.PostMessageW(hwnd, 0x0102, ord(ch), 0)  # WM_CHAR
        time.sleep(delay)
    return True


class cursor_guard:
    """Context manager: save cursor position before an action and restore it immediately afterwards.

    Ensures that even if hardware mouse emulation is used as a fallback,
    the cursor is instantly restored to the user's hand within milliseconds.
    """

    def __enter__(self):
        try:
            hdesk = user32.OpenInputDesktop(0, False, 0x01FF)
            if hdesk:
                user32.SetThreadDesktop(hdesk)
        except Exception:
            pass
        pt = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        self.orig_x = pt.x
        self.orig_y = pt.y
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        user32.SetCursorPos(self.orig_x, self.orig_y)



def click_with_restore(x: int, y: int, button: str = "left", clicks: int = 1, hwnd: Optional[int] = None):
    """Click at (x, y) with hardware emulation, but instantly restore the user's cursor position."""
    if hwnd and clicks > 1:
        focus_window(hwnd)
        time.sleep(0.08)
    with cursor_guard():
        click(x, y, button=button, clicks=clicks)


# ──────────────────────── Window Management ─────────────────────────────────


def get_window_rect(hwnd: int) -> Tuple[int, int, int, int]:
    """Return (left, top, right, bottom) of a window handle."""
    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    return (rect.left, rect.top, rect.right, rect.bottom)


def get_window_size(hwnd: int) -> Tuple[int, int]:
    """Return (width, height) of a window handle."""
    l, t, r, b = get_window_rect(hwnd)
    return (r - l, b - t)


def focus_window(hwnd: int) -> bool:
    """Bring a window to the foreground reliably using the Win32 Alt-key bypass and SwitchToThisWindow."""
    ensure_default_desktop()
    if not hwnd or not user32.IsWindow(hwnd):
        return False

    fg_hwnd = user32.GetForegroundWindow()
    if fg_hwnd == hwnd:
        return True

    # Win32 Alt-key trick: unlocks SetForegroundWindow for background processes
    user32.keybd_event(0x12, 0, 0, 0)  # VK_MENU (Alt) down
    user32.keybd_event(0x12, 0, 2, 0)  # VK_MENU (Alt) up

    fg_pid = wintypes.DWORD()
    fg_tid = user32.GetWindowThreadProcessId(fg_hwnd, ctypes.byref(fg_pid))
    cur_tid = ctypes.windll.kernel32.GetCurrentThreadId()

    try:
        user32.AttachThreadInput(cur_tid, fg_tid, True)
    except Exception:
        pass

    user32.AllowSetForegroundWindow(0xFFFFFFFF)
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
    else:
        user32.ShowWindow(hwnd, 5)  # SW_SHOW
    user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002)  # HWND_TOPMOST
    user32.SetWindowPos(hwnd, -2, 0, 0, 0, 0, 0x0001 | 0x0002)  # HWND_NOTOPMOST
    user32.SetForegroundWindow(hwnd)
    user32.BringWindowToTop(hwnd)

    try:
        user32.AttachThreadInput(cur_tid, fg_tid, False)
    except Exception:
        pass

    # Verified focus retry loop using SwitchToThisWindow
    for _ in range(4):
        if user32.GetForegroundWindow() == hwnd:
            return True
        try:
            user32.SwitchToThisWindow(hwnd, True)
        except Exception:
            pass
        user32.SetForegroundWindow(hwnd)
        time.sleep(0.06)

    return user32.GetForegroundWindow() == hwnd


def ensure_default_desktop():
    """Ensure the calling thread is attached to the interactive 'default' desktop on WinSta0."""
    try:
        hdesk = user32.OpenDesktopW("default", 0, False, 0x01FF)
        if hdesk:
            user32.SetThreadDesktop(hdesk)
    except Exception:
        pass


def find_window_by_process(process_names: list, title_contains: Optional[str] = None) -> Optional[int]:
    """Find a top-level visible window belonging to any of the given process names.

    Filters out zero-sized and hidden utility helper windows, scoring candidates
    by title match, non-empty title, and largest visible client area.
    """
    ensure_default_desktop()
    try:
        import psutil
        target_pids = set()
        for p in psutil.process_iter(["pid", "name"]):
            pname = (p.info["name"] or "").lower()
            if any(n in pname for n in process_names):
                target_pids.add(p.info["pid"])
    except Exception:
        target_pids = set()

    candidates: List[Tuple[int, int]] = []

    def _enum_callback(hwnd, lparam):
        if not user32.IsWindowVisible(hwnd):
            return True

        rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        w = rect.right - rect.left
        h = rect.bottom - rect.top
        if w < 100 or h < 100:
            return True  # Ignore tiny or invisible helper frames

        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))

        buf = ctypes.create_unicode_buffer(512)
        user32.GetWindowTextW(hwnd, buf, 512)
        title = buf.value.lower()

        cls_buf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, cls_buf, 256)
        cls_name = cls_buf.value.lower()

        pid_match = bool(target_pids and pid.value in target_pids)
        title_match = bool(title_contains and title_contains.lower() in title)
        name_match = any(name.lower() in title or name.lower() in cls_name for name in process_names)

        if pid_match or title_match or name_match:
            # Score candidate: prefer title match, non-empty title, and larger area
            score = w * h
            if title_match or (title and name_match):
                score += 10_000_000
            elif title:
                score += 1_000_000
            if "chrome_widgetwin" in cls_name and not title:
                score -= 500_000  # Deprioritize untitled chromium utility frames
            candidates.append((score, hwnd))


        return True

    WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    callback = WNDENUMPROC(_enum_callback)

    hdesk_default = user32.OpenDesktopW("default", 0, False, 0x01FF)
    if hdesk_default:
        user32.EnumDesktopWindows.argtypes = [wintypes.HANDLE, WNDENUMPROC, wintypes.LPARAM]
        user32.EnumDesktopWindows.restype = wintypes.BOOL
        user32.EnumDesktopWindows(hdesk_default, callback, 0)
        user32.CloseDesktop(hdesk_default)
    else:
        user32.EnumWindows.argtypes = [WNDENUMPROC, wintypes.LPARAM]
        user32.EnumWindows.restype = wintypes.BOOL
        user32.EnumWindows(callback, 0)

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    return None


def capture_window(hwnd: int):
    """Capture a window as a PIL Image (top-down orientation).

    Tries screen compositor capture first for GPU-accelerated applications,
    falling back to PrintWindow GDI capture.

    Returns:
        PIL.Image.Image in RGBA mode, or None on failure.
    """
    if not hwnd or not user32.IsWindow(hwnd):
        return None

    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        time.sleep(0.15)

    gdi32 = ctypes.windll.gdi32
    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    w = rect.right - rect.left
    h = rect.bottom - rect.top
    if w <= 0 or h <= 0:
        return None

    # For foregrounded windows on screen, ImageGrab captures GPU/DirectX compositor surfaces
    try:
        from PIL import ImageGrab
        if user32.GetForegroundWindow() == hwnd and rect.left >= 0 and rect.top >= 0:
            img = ImageGrab.grab(bbox=(rect.left, rect.top, rect.right, rect.bottom))
            if img:
                return img.convert("RGBA") if img.mode != "RGBA" else img
    except Exception:
        pass

    hwndDC = user32.GetWindowDC(hwnd)
    memDC = gdi32.CreateCompatibleDC(hwndDC)
    bitmap = gdi32.CreateCompatibleBitmap(hwndDC, w, h)
    hOld = gdi32.SelectObject(memDC, bitmap)
    user32.PrintWindow(hwnd, memDC, 2)  # PW_RENDERFULLCONTENT

    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
            ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
            ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
            ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
            ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
            ("biClrImportant", wintypes.DWORD),
        ]

    bih = BITMAPINFOHEADER()
    bih.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bih.biWidth = w
    bih.biHeight = -h  # Negative = top-down DIB
    bih.biPlanes = 1
    bih.biBitCount = 32
    bih.biCompression = 0
    bih.biSizeImage = w * h * 4
    buf = (ctypes.c_char * (w * h * 4))()
    gdi32.GetDIBits(memDC, bitmap, 0, h, ctypes.byref(buf), ctypes.byref(bih), 0)

    gdi32.SelectObject(memDC, hOld)
    gdi32.DeleteObject(bitmap)
    gdi32.DeleteDC(memDC)
    user32.ReleaseDC(hwnd, hwndDC)

    try:
        from PIL import Image
        return Image.frombuffer("RGBA", (w, h), bytes(buf), "raw", "BGRA", 0, 1)
    except Exception:
        return None


def capture_screen(region: Optional[Tuple[int, int, int, int]] = None):
    """Capture the full screen or a region as a PIL Image.

    Args:
        region: Optional (left, top, right, bottom) in screen coordinates.

    Returns:
        PIL.Image.Image in RGB mode, or None on failure.
    """
    try:
        from PIL import ImageGrab
        return ImageGrab.grab(bbox=region).convert("RGB")
    except Exception:
        return None


def set_clipboard_text(text: str) -> bool:
    """Safely place arbitrary Unicode multiline text on Windows clipboard via native Win32 API.
    
    Avoids PowerShell command-line length limits, cmd.exe quotation truncation, and process spawning.
    """
    CF_UNICODETEXT = 13
    GMEM_MOVEABLE = 0x0002

    kernel32.GlobalAlloc.argtypes = [ctypes.c_uint, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = ctypes.c_void_p
    kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalFree.argtypes = [ctypes.c_void_p]
    user32.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]
    user32.SetClipboardData.restype = ctypes.c_void_p

    opened = False
    for _ in range(10):
        if user32.OpenClipboard(0):
            opened = True
            break
        time.sleep(0.02)
    if not opened:
        return False
    try:
        user32.EmptyClipboard()
        data = text.encode("utf-16le") + b"\x00\x00"
        h = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data))
        if not h:
            return False
        p = kernel32.GlobalLock(h)
        if not p:
            kernel32.GlobalFree(h)
            return False
        ctypes.memmove(p, data, len(data))
        kernel32.GlobalUnlock(h)
        res = user32.SetClipboardData(CF_UNICODETEXT, h)
        return bool(res)
    finally:
        user32.CloseClipboard()


def get_clipboard_text() -> str:
    """Safely read Unicode text from Windows clipboard via native Win32 API."""
    CF_UNICODETEXT = 13
    user32.GetClipboardData.argtypes = [ctypes.c_uint]
    user32.GetClipboardData.restype = ctypes.c_void_p
    kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]

    if not user32.OpenClipboard(0):
        return ""
    try:
        h = user32.GetClipboardData(CF_UNICODETEXT)
        if not h:
            return ""
        p = kernel32.GlobalLock(h)
        if not p:
            return ""
        try:
            return ctypes.wstring_at(p)
        finally:
            kernel32.GlobalUnlock(h)
    finally:
        user32.CloseClipboard()


def set_clipboard_file(file_path: str) -> bool:
    """Place a file on the Windows clipboard as a CF_HDROP object for pasting into apps."""
    abs_path = os.path.abspath(file_path)
    if not os.path.exists(abs_path):
        return False
    CF_HDROP = 15
    GMEM_MOVEABLE = 0x0002

    class DROPFILES(ctypes.Structure):
        _fields_ = [
            ("pFiles", wintypes.DWORD),
            ("pt", wintypes.POINT),
            ("fNC", wintypes.BOOL),
            ("fWide", wintypes.BOOL),
        ]

    df = DROPFILES()
    df.pFiles = ctypes.sizeof(DROPFILES)
    df.pt = wintypes.POINT(0, 0)
    df.fNC = False
    df.fWide = True

    path_bytes = abs_path.encode("utf-16le") + b"\x00\x00\x00\x00"
    total_bytes = ctypes.sizeof(DROPFILES) + len(path_bytes)

    if not user32.OpenClipboard(0):
        return False
    try:
        user32.EmptyClipboard()
        h = kernel32.GlobalAlloc(GMEM_MOVEABLE, total_bytes)
        if not h:
            return False
        p = kernel32.GlobalLock(h)
        if not p:
            kernel32.GlobalFree(h)
            return False
        ctypes.memmove(p, ctypes.byref(df), ctypes.sizeof(DROPFILES))
        ctypes.memmove(p + ctypes.sizeof(DROPFILES), path_bytes, len(path_bytes))
        kernel32.GlobalUnlock(h)
        res = user32.SetClipboardData(CF_HDROP, h)
        return bool(res)
    finally:
        user32.CloseClipboard()


