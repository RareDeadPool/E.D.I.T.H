import sys
import os
sys.path.insert(0, os.path.abspath("."))

from runtime.screen_engine import UIAProvider

provider = UIAProvider()
provider._ensure_init()

if provider._automation:
    root = provider._automation.GetRootElement()
    print("Root element:", root)
    child = provider._walker.GetFirstChildElement(root)
    count = 0
    while child and count < 50:
        name = child.CurrentName
        cls = child.CurrentClassName
        aid = child.CurrentAutomationId
        pid = child.CurrentProcessId
        if name or aid or "whatsapp" in name.lower() or "whatsapp" in cls.lower():
            print(f"[{count}] Name: '{name}' | Class: '{cls}' | AutoId: '{aid}' | PID: {pid}")
        child = provider._walker.GetNextSiblingElement(child)
        count += 1
else:
    print("Could not initialize UIA automation.")
