import sys
import os
sys.path.insert(0, os.path.abspath("."))
import time
import ctypes
import comtypes.client

sys.stdout.reconfigure(encoding='utf-8')

user32 = ctypes.windll.user32
hdesk_default = user32.OpenDesktopW("default", 0, False, 0x01FF)
if hdesk_default:
    user32.SetThreadDesktop(hdesk_default)

from runtime import actions

hwnd = actions.find_window_by_process(["whatsapp", "whatsapp.root"])
print("WhatsApp HWND:", hwnd)
actions.focus_window(hwnd)
time.sleep(0.5)

ctypes.windll.ole32.CoInitialize(None)
mod = comtypes.client.GetModule("UIAutomationCore.dll")
automation = comtypes.client.CreateObject("{ff48dba4-60ef-4201-aa87-54103eef594e}", interface=mod.IUIAutomation)
root = automation.ElementFromHandle(hwnd)

cond_edit = automation.CreatePropertyCondition(30003, 50004)
edits = root.FindAll(mod.TreeScope_Descendants, cond_edit)
for i in range(edits.Length if edits else 0):
    el = edits.GetElement(i)
    r = el.CurrentBoundingRectangle
    print(f"Edit {i}: Name='{el.CurrentName}', BBox=({r.left}, {r.top}, {r.right}, {r.bottom})")
    if r.top > 700 and r.left > 500:
        print("Focusing composer via SetFocus()...")
        el.SetFocus()
        time.sleep(0.2)
        cx = (r.left + r.right) // 2
        cy = (r.top + r.bottom) // 2
        print(f"Clicking at ({cx}, {cy})...")
        actions.click(cx, cy)
        time.sleep(0.2)

        # Check focused element
        focused = automation.GetFocusedElement()
        print(f"Currently focused element: Name='{focused.CurrentName}', Class='{focused.CurrentClassName}'")

        # Paste test text
        actions.set_clipboard_text("Test typing live 8591211342")
        time.sleep(0.1)
        actions.hotkey("ctrl", "v")
        time.sleep(0.3)
        print("Pasted text. Checking Send button...")

        cond_btn = automation.CreatePropertyCondition(30003, 50000)
        buttons = root.FindAll(mod.TreeScope_Descendants, cond_btn)
        for j in range(buttons.Length if buttons else 0):
            b = buttons.GetElement(j)
            if b.CurrentName.lower() == "send":
                br = b.CurrentBoundingRectangle
                print(f"Send button found at ({br.left}, {br.top}, {br.right}, {br.bottom}). Clicking...")
                actions.click((br.left + br.right) // 2, (br.top + br.bottom) // 2)
                time.sleep(0.5)
                break
