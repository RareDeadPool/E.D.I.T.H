import sys
import os
sys.path.insert(0, os.path.abspath("."))
import ctypes

sys.stdout.reconfigure(encoding='utf-8')

user32 = ctypes.windll.user32
hdesk_default = user32.OpenDesktopW("default", 0, False, 0x01FF)
user32.SetThreadDesktop(hdesk_default)

from runtime import actions
hwnd = actions.find_window_by_process(["whatsapp", "whatsapp.root"])

import comtypes
import comtypes.client

mod = comtypes.client.GetModule("UIAutomationCore.dll")
automation = comtypes.client.CreateObject("{ff48dba4-60ef-4201-aa87-54103eef594e}", interface=mod.IUIAutomation)
root = automation.ElementFromHandle(hwnd)

print("Searching for Edit controls, Text controls, or Button controls in WhatsApp...")

# Create Condition for ControlType Edit or Button
cond_edit = automation.CreatePropertyCondition(30003, 50004) # UIA_EditControlTypeId
cond_btn = automation.CreatePropertyCondition(30003, 50000)  # UIA_ButtonControlTypeId
cond_or = automation.CreateOrCondition(cond_edit, cond_btn)

edits = root.FindAll(mod.TreeScope_Descendants, cond_edit)
print(f"Total Edit controls found: {edits.Length if edits else 0}")
if edits:
    for i in range(edits.Length):
        el = edits.GetElement(i)
        name = el.CurrentName
        aid = el.CurrentAutomationId
        rect = el.CurrentBoundingRectangle
        bbox = (rect.left, rect.top, rect.right, rect.bottom)
        print(f"Edit [{i}]: Name='{name}', AutoId='{aid}', BBox={bbox}")

buttons = root.FindAll(mod.TreeScope_Descendants, cond_btn)
print(f"\nTotal Button controls found: {buttons.Length if buttons else 0}")
if buttons:
    for i in range(buttons.Length):
        el = buttons.GetElement(i)
        name = el.CurrentName
        aid = el.CurrentAutomationId
        rect = el.CurrentBoundingRectangle
        bbox = (rect.left, rect.top, rect.right, rect.bottom)
        print(f"Button [{i}]: Name='{name}', AutoId='{aid}', BBox={bbox}")
