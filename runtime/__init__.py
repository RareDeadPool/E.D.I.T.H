"""E.D.I.T.H. Runtime Package.

Modules:
    executor       — Legacy task executor (direct OS calls)
    actions        — Generic OS-level input primitives (click, type, hotkey, etc.)
    screen_engine  — Unified screen perception engine (UIA + OCR + Vision)
    gui_agent      — Application-agnostic GUI interaction agent
"""

from .executor import EdithTaskExecutor
from .screen_engine import ScreenEngine, ScreenElement
from .gui_agent import GUIAgent

__all__ = [
    "EdithTaskExecutor",
    "ScreenEngine",
    "ScreenElement",
    "GUIAgent",
]
