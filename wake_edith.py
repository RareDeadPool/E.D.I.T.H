"""E.D.I.T.H. Passive Wake-Word Detection Launcher.

Listens in the background for your wake-up word:
- 'Hey Edith'
- 'Edith'
- 'Wake up Edith'
- 'Yo Edith'

When you say the wake-up word, E.D.I.T.H. immediately:
1. Plays a futuristic wake chime.
2. Wakes up and speaks: "I'm here, Aditya! What do you need?"
3. Listens for your voice command.
4. Executes the OS command or chats with you.
5. Speaks the response back in ChatGPT-style Neural Voice!
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runtime.stt_engine import EdithWakeWordDetector


def main():
    detector = EdithWakeWordDetector(enable_tts=True)
    detector.listen_for_wake_word()


if __name__ == "__main__":
    main()
