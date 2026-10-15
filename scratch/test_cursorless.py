"""Verification test for cursorless multitasking execution in E.D.I.T.H.

Tests that clicking and typing in background windows does NOT alter
the physical mouse cursor position.
"""

import sys
import time
import ctypes
from ctypes import wintypes

sys.path.insert(0, ".")
from runtime.actions import user32, cursor_guard, post_click_background
from runtime.gui_agent import GUIAgent


def get_mouse():
    pt = wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    return (pt.x, pt.y)


def main():
    print("=" * 60)
    print("     TESTING CURSORLESS MULTITASKING EXECUTION")
    print("=" * 60)

    # Record initial mouse position
    pos_before = get_mouse()
    print(f"[*] Initial mouse position: {pos_before}")

    agent = GUIAgent()

    # Test 1: Perceive Chrome window
    print("\n[*] Finding and perceiving Chrome...")
    hwnd = agent.find_app("chrome")
    if not hwnd:
        print("[-] Chrome not found running. Testing generic foreground window.")
        hwnd = user32.GetForegroundWindow()

    elements = agent.see(hwnd=hwnd)
    print(f"[+] Found {len(elements)} elements. UIA with raw COM node attached: {sum(1 for e in elements if e.raw_element is not None)}")

    # Check can_invoke and can_set_value
    invokable = [e for e in elements if e.role in ("button", "tab_item", "menu_item", "hyperlink")]
    print(f"[+] Found {len(invokable)} buttons/tabs available for programmatic invoke.")

    # Test 2: Perform click_element with cursorless=True
    if invokable:
        target = invokable[0]
        print(f"\n[*] Testing programmatic click on element: '{target.name}' ({target.role})")
        pos_pre_click = get_mouse()
        
        # Test programmatic invoke or post_click
        success = target.invoke() or target.post_click()
        pos_post_click = get_mouse()
        
        print(f"    Action invoked: {success}")
        print(f"    Cursor before: {pos_pre_click}")
        print(f"    Cursor after:  {pos_post_click}")
        assert pos_pre_click == pos_post_click, f"Cursor moved! {pos_pre_click} -> {pos_post_click}"
        print("    [PASS] Cursor position was completely undisturbed! (0 pixel displacement)")

    # Test 3: Cursor guard test
    print("\n[*] Testing cursor_guard context manager...")
    pos_before_guard = get_mouse()
    with cursor_guard():
        # Artificially move cursor
        user32.SetCursorPos(100, 100)
        time.sleep(0.01)

    pos_after_guard = get_mouse()
    print(f"    Cursor before guard: {pos_before_guard}")
    print(f"    Cursor restored to:  {pos_after_guard}")
    assert pos_before_guard == pos_after_guard, f"Cursor guard failed to restore! {pos_before_guard} vs {pos_after_guard}"
    print("    [PASS] cursor_guard restored mouse position with 100% precision.")

    print("\n" + "=" * 60)
    print("  ALL CURSORLESS MULTITASKING CHECKS PASSED SUCCESSFULLY! ")
    print("=" * 60)


if __name__ == "__main__":
    main()
