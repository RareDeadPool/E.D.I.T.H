import sys, os, time, difflib
sys.path.insert(0, os.path.abspath("."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from runtime.gui_agent import GUIAgent
from runtime import actions
from scratch.test_force_fg import force_foreground_window

def test_play_song(song_query: str):
    hwnd = actions.find_window_by_process(['spotify'])
    print(f"Spotify HWND: {hwnd}")
    if not hwnd:
        print("Spotify is not running!")
        return False

    # 1. Bring Spotify to foreground
    force_foreground_window(hwnd)
    time.sleep(0.2)

    # 2. Press Ctrl+L to focus search box
    actions.user32.keybd_event(0x11, 0, 0, 0) # Ctrl down
    actions.user32.keybd_event(0x4C, 0, 0, 0) # L down
    time.sleep(0.04)
    actions.user32.keybd_event(0x4C, 0, 2, 0) # L up
    actions.user32.keybd_event(0x11, 0, 2, 0) # Ctrl up
    time.sleep(0.1)

    # 3. Ctrl+A to select all and clear
    actions.user32.keybd_event(0x11, 0, 0, 0)
    actions.user32.keybd_event(0x41, 0, 0, 0)
    time.sleep(0.04)
    actions.user32.keybd_event(0x41, 0, 2, 0)
    actions.user32.keybd_event(0x11, 0, 2, 0)
    time.sleep(0.04)
    actions.user32.keybd_event(0x08, 0, 0, 0) # Backspace
    time.sleep(0.04)
    actions.user32.keybd_event(0x08, 0, 2, 0)
    time.sleep(0.1)

    # 4. Type query
    for ch in song_query:
        actions.user32.keybd_event(0, ord(ch), 4, 0)
        time.sleep(0.015)
        actions.user32.keybd_event(0, ord(ch), 6, 0)
        time.sleep(0.015)

    time.sleep(0.15)
    # 5. Press Enter
    actions.user32.keybd_event(0x0D, 0, 0, 0)
    time.sleep(0.04)
    actions.user32.keybd_event(0x0D, 0, 2, 0)
    print(f"Submitted search: '{song_query}', waiting for results...")
    time.sleep(1.2)

    # 6. Perceive results
    agent = GUIAgent()
    elements = agent.see(hwnd=hwnd, force_ocr=True)
    print(f"Perceived {len(elements)} elements.")

    # 7. Locate song item in results
    # We want elements in the main results area (y between 120 and 880)
    # matching the query
    q_words = [w.lower() for w in song_query.split() if len(w) > 1]
    best_elem = None
    best_score = 0.0

    for elem in elements:
        cx, cy = elem.center
        if cy < 120 or cy > 880:
            continue
        name_lower = elem.name.lower()
        # Direct word match
        score = 0.0
        if any(w in name_lower for w in q_words):
            score = 0.8
            if song_query.lower() in name_lower or name_lower in song_query.lower():
                score = 1.0
        else:
            sim = difflib.SequenceMatcher(None, song_query.lower(), name_lower).ratio()
            if sim > 0.6:
                score = sim

        if score > best_score:
            best_score = score
            best_elem = elem

    if best_elem and best_score >= 0.6:
        print(f"Found song row: '{best_elem.name}' at {best_elem.center} (score={best_score:.2f})")
        print("Double-clicking to play (with cursor restoration)...")
        actions.click_with_restore(best_elem.center[0], best_elem.center[1], clicks=2)
        time.sleep(0.5)
        print("Playback triggered successfully!")
        return True
    else:
        print(f"Could not locate '{song_query}' in perceived results.")
        return False

if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "Starboy"
    test_play_song(query)
