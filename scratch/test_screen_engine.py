"""Quick smoke-test for the Screen Engine.

Runs UIA + OCR on the current foreground window and prints what E.D.I.T.H. sees.
Usage:
    python scratch\test_screen_engine.py
    python scratch\test_screen_engine.py --app spotify
    python scratch\test_screen_engine.py --app chrome
"""

import os, sys, argparse, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runtime.screen_engine import ScreenEngine
from runtime.actions import find_window_by_process, focus_window, capture_window, get_window_rect


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", type=str, default=None, help="App to target (e.g. spotify, chrome, notepad)")
    parser.add_argument("--find", type=str, default=None, help="Element to search for (e.g. 'Search', 'Play')")
    parser.add_argument("--save", type=str, default=None, help="Save annotated screenshot to this path")
    args = parser.parse_args()

    engine = ScreenEngine()

    hwnd = None
    if args.app:
        print(f"[*] Looking for '{args.app}'...")
        hwnd = find_window_by_process([args.app.lower()], title_contains=args.app.lower())
        if hwnd:
            print(f"[+] Found window HWND={hwnd}")
            focus_window(hwnd)
            time.sleep(0.3)
        else:
            print(f"[-] Could not find a window for '{args.app}'. Using foreground window.")

    print("\n[*] Perceiving screen elements...")
    t0 = time.time()
    elements = engine.perceive(hwnd=hwnd)
    elapsed = time.time() - t0
    print(f"[+] Found {len(elements)} elements in {elapsed:.3f}s\n")

    # Print the dump
    print(engine.dump(elements))

    # Search for a specific element
    if args.find:
        print(f"\n[*] Searching for '{args.find}'...")
        target = engine.find(args.find, elements)
        if target:
            print(f"[+] Found: {target}")
            print(f"    Center: {target.center}")
            print(f"    BBox:   {target.bbox}")
        else:
            print(f"[-] No match found for '{args.find}'")

    # Save annotated screenshot
    if args.save and hwnd:
        img = capture_window(hwnd)
        if img:
            rect = get_window_rect(hwnd)
            annotated = engine.annotate_image(img, elements, offset=(rect[0], rect[1]))
            annotated.save(args.save)
            print(f"\n[+] Saved annotated screenshot to {args.save}")

    # Stats
    uia_count = len([e for e in elements if e.source == "uia"])
    ocr_count = len([e for e in elements if e.source == "ocr"])
    print(f"\n[*] Summary: {uia_count} UIA elements, {ocr_count} OCR elements")


if __name__ == "__main__":
    main()
