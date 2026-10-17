import time
import win32gui
import win32process
import psutil

try:
    import uiautomation as uia
except ImportError:
    uia = None

def find_whatsapp_windows():
    results = []
    def enum_cb(hwnd, extra):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            classname = win32gui.GetClassName(hwnd)
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            try:
                pname = psutil.Process(pid).name()
            except Exception:
                pname = "unknown"
            if "whatsapp" in title.lower() or "whatsapp" in pname.lower():
                rect = win32gui.GetWindowRect(hwnd)
                results.append({
                    "hwnd": hwnd,
                    "title": title,
                    "class": classname,
                    "pid": pid,
                    "process": pname,
                    "rect": rect
                })
        return True
    win32gui.EnumWindows(enum_cb, None)
    return results

print("=== Scanning for WhatsApp Windows ===")
windows = find_whatsapp_windows()
for w in windows:
    print(f"HWND: {w['hwnd']} | PID: {w['pid']} ({w['process']}) | Class: {w['class']} | Rect: {w['rect']} | Title: '{w['title']}'")

if uia:
    print("\n=== UI Automation Inspection ===")
    root = uia.GetRootControl()
    wa_ctrl = uia.WindowControl(searchDepth=2, SubName="WhatsApp")
    if wa_ctrl.Exists(maxSearchSeconds=2):
        print(f"Found UIA WhatsApp Window: Name='{wa_ctrl.Name}', ClassName='{wa_ctrl.ClassName}', AutomationId='{wa_ctrl.AutomationId}'")
        edit_controls = wa_ctrl.GetChildren()
        for idx, child in enumerate(edit_controls):
            print(f"Child {idx}: Name='{child.Name}', ControlType='{child.ControlTypeName}', AutomationId='{child.AutomationId}'")
    else:
        print("UIA WindowControl with SubName='WhatsApp' not found at depth 2.")
else:
    print("uiautomation library not imported.")
