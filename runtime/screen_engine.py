"""Screen Perception Engine for E.D.I.T.H.

Provides a unified API to *understand any screen* by combining three
perception layers in priority order:

  1. **Windows UI Automation (UIA)** — Queries the OS accessibility tree
     to get exact element names, roles, and bounding boxes.  Instant,
     zero-GPU, and resolution-independent.

  2. **Local OCR (RapidOCR)** — Runs a lightweight ONNX-based OCR pass
     on a window screenshot to detect every visible text token and its
     pixel coordinates.  Catches text inside custom-rendered canvases
     (Electron apps, games) where UIA returns nothing.

  3. **Visual Fallback (planned)** — Florence-2 / OmniParser icon and
     widget detection for non-textual interactive elements.

Usage
-----
>>> from runtime.screen_engine import ScreenEngine
>>> engine = ScreenEngine()
>>> elements = engine.perceive(hwnd=spotify_hwnd)
>>> target = engine.find("Search", elements)
>>> engine.click(target)
"""

from __future__ import annotations

import os
import re
import time
import math
import difflib
import ctypes
from ctypes import wintypes
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────── Data Structures ────────────────────────────────


@dataclass
class ScreenElement:
    """A single interactive element detected on screen."""

    name: str                         # Human-readable label ("Search", "Play", "Liked Songs")
    role: str                         # Semantic role ("button", "edit", "text", "link", "image", "pane", "list_item")
    bbox: Tuple[int, int, int, int]   # Absolute screen coords (left, top, right, bottom)
    center: Tuple[int, int] = (0, 0)  # Center point for clicking
    confidence: float = 1.0           # 0..1 detection confidence
    source: str = "uia"              # "uia", "ocr", or "vision"
    automationId: str = ""            # UIA AutomationId (if available)
    className: str = ""               # UIA ClassName (if available)
    value: str = ""                   # Current value (e.g. text in an edit box)
    is_enabled: bool = True
    is_offscreen: bool = False
    children: List["ScreenElement"] = field(default_factory=list)
    raw_element: Any = field(default=None, repr=False)  # Native COM IUIAutomationElement
    hwnd: Optional[int] = None                          # Window handle

    def __post_init__(self):
        if self.center == (0, 0) and self.bbox != (0, 0, 0, 0):
            l, t, r, b = self.bbox
            self.center = ((l + r) // 2, (t + b) // 2)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    def invoke(self) -> bool:
        """Click this element programmatically via COM without moving the mouse."""
        if not self.raw_element:
            return False
        try:
            import comtypes.client
            mod = comtypes.client.GetModule("UIAutomationCore.dll")

            # 1. Try InvokePattern (buttons, links, menu items)
            try:
                pattern = self.raw_element.GetCurrentPattern(10000)  # UIA_InvokePatternId
                if pattern:
                    inv = pattern.QueryInterface(mod.IUIAutomationInvokePattern)
                    inv.Invoke()
                    return True
            except Exception:
                pass

            # 2. Try TogglePattern (checkboxes, toggles)
            try:
                pattern = self.raw_element.GetCurrentPattern(10015)  # UIA_TogglePatternId
                if pattern:
                    tog = pattern.QueryInterface(mod.IUIAutomationTogglePattern)
                    tog.Toggle()
                    return True
            except Exception:
                pass

            # 3. Try SelectionItemPattern (list items, tab items)
            try:
                pattern = self.raw_element.GetCurrentPattern(10010)  # UIA_SelectionItemPatternId
                if pattern:
                    sel = pattern.QueryInterface(mod.IUIAutomationSelectionItemPattern)
                    sel.Select()
                    return True
            except Exception:
                pass

            # 4. Try ExpandCollapsePattern
            try:
                pattern = self.raw_element.GetCurrentPattern(10005)  # UIA_ExpandCollapsePatternId
                if pattern:
                    exp = pattern.QueryInterface(mod.IUIAutomationExpandCollapsePattern)
                    exp.Expand()
                    return True
            except Exception:
                pass
        except Exception:
            pass
        return False

    def set_value(self, text: str) -> bool:
        """Inject text programmatically via COM without stealing keyboard focus."""
        if not self.raw_element:
            return False
        try:
            import comtypes.client
            mod = comtypes.client.GetModule("UIAutomationCore.dll")
            try:
                pattern = self.raw_element.GetCurrentPattern(10002)  # UIA_ValuePatternId
                if pattern:
                    val = pattern.QueryInterface(mod.IUIAutomationValuePattern)
                    val.SetValue(text)
                    return True
            except Exception:
                pass
        except Exception:
            pass
        return False

    def post_click(self, double: bool = False) -> bool:
        """Send background click message via PostMessageW without moving the cursor."""
        if not self.hwnd:
            return False
        from runtime import actions
        return actions.post_click_background(self.hwnd, self.center[0], self.center[1], clicks=2 if double else 1)

    def post_text(self, text: str) -> bool:
        """Send background text via PostMessageW without stealing keyboard focus."""
        if not self.hwnd:
            return False
        from runtime import actions
        return actions.post_text_background(self.hwnd, text)

    def __repr__(self):
        return (
            f"<ScreenElement name={self.name!r} role={self.role!r} "
            f"bbox={self.bbox} source={self.source!r} conf={self.confidence:.2f}>"
        )


# ──────────────────────── UIA Control Type Constants ────────────────────────

# From UIAutomationClient.h — maps ControlTypeId to human-readable role names
_UIA_CONTROL_TYPE_NAMES = {
    50000: "button",
    50001: "calendar",
    50002: "checkbox",
    50003: "combobox",
    50004: "edit",
    50005: "hyperlink",
    50006: "image",
    50007: "list_item",
    50008: "list",
    50009: "menu",
    50010: "menu_bar",
    50011: "menu_item",
    50012: "progressbar",
    50013: "radio_button",
    50014: "scrollbar",
    50015: "slider",
    50016: "spinner",
    50017: "statusbar",
    50018: "tab",
    50019: "tab_item",
    50020: "text",
    50021: "toolbar",
    50022: "tooltip",
    50023: "tree",
    50024: "tree_item",
    50025: "custom",
    50026: "group",
    50027: "thumb",
    50028: "datagrid",
    50029: "data_item",
    50030: "document",
    50031: "splitbutton",
    50032: "window",
    50033: "pane",
    50034: "header",
    50035: "header_item",
    50036: "table",
    50037: "title_bar",
    50038: "separator",
}

# ─────────────────────── Provider: Windows UI Automation ────────────────────


class UIAProvider:
    """Extracts UI elements from a window's accessibility tree via COM/UIA.

    Uses the Windows UIAutomation COM API through comtypes with proper
    interface definitions for IUIAutomation, IUIAutomationElement, and
    IUIAutomationTreeWalker.
    """

    def __init__(self):
        self._automation = None
        self._walker = None
        self._initialized = False

    def _ensure_init(self):
        """Lazy-initialize the UIAutomation COM interface with proper type info."""
        if self._initialized:
            return
        try:
            import comtypes
            import comtypes.client

            # Generate the UIAutomationClient type library wrapper
            # This gives us properly typed IUIAutomation, IUIAutomationElement, etc.
            try:
                mod = comtypes.client.GetModule("UIAutomationCore.dll")
            except Exception:
                # If the typelib isn't registered, try generating from the GUID
                try:
                    mod = comtypes.client.GetModule(
                        ["{944DE083-8FB8-45CF-BCB7-C477ACB2F897}", 1, 0]
                    )
                except Exception:
                    mod = None

            if mod:
                # Create CUIAutomation with the proper interface from the typelib
                self._automation = comtypes.client.CreateObject(
                    "{ff48dba4-60ef-4201-aa87-54103eef594e}",
                    interface=mod.IUIAutomation,
                )
            else:
                # Final fallback: create without explicit interface and use dynamic dispatch
                self._automation = comtypes.client.CreateObject(
                    "{ff48dba4-60ef-4201-aa87-54103eef594e}",
                    clsctx=comtypes.CLSCTX_INPROC_SERVER,
                )

            # Acquire a RawViewWalker for tree traversal
            try:
                self._walker = self._automation.RawViewWalker
            except Exception:
                self._walker = None

            self._initialized = True
        except Exception as e:
            print(f"[ScreenEngine/UIA] Failed to initialize UI Automation: {e}")
            self._automation = None
            self._initialized = True  # Don't retry on failure

    def get_elements(
        self,
        hwnd: Optional[int] = None,
        max_depth: int = 8,
        min_area: int = 100,
    ) -> List[ScreenElement]:
        """Walk the UIA tree rooted at `hwnd` and return all interactive elements.

        Args:
            hwnd: Window handle.  If None, uses the foreground window.
            max_depth: Maximum tree recursion depth to prevent infinite loops.
            min_area: Minimum element area in pixels to include (filters noise).

        Returns:
            List of ScreenElement objects with absolute screen coordinates.
        """
        self._ensure_init()
        if not self._automation:
            return []

        try:
            if hwnd:
                root = self._automation.ElementFromHandle(hwnd)
            else:
                # Try to get the focused element first
                try:
                    root = self._automation.GetFocusedElement()
                except Exception:
                    root = None
                if not root:
                    fg = ctypes.windll.user32.GetForegroundWindow()
                    if fg:
                        root = self._automation.ElementFromHandle(fg)

            if not root:
                return []

            target_hwnd = hwnd
            if not target_hwnd:
                try:
                    target_hwnd = ctypes.windll.user32.GetForegroundWindow()
                except Exception:
                    target_hwnd = None

            elements: List[ScreenElement] = []
            self._walk(root, elements, depth=0, max_depth=max_depth, min_area=min_area, target_hwnd=target_hwnd)
            return elements
        except Exception as e:
            print(f"[ScreenEngine/UIA] Error walking tree: {e}")
            return []

    def _walk(self, node, elements: List[ScreenElement], depth: int, max_depth: int, min_area: int, target_hwnd: Optional[int] = None):
        """Recursively walk the UIA tree."""
        if depth > max_depth:
            return
        try:
            # Extract element properties safely
            name = ""
            try:
                name = node.CurrentName or ""
            except Exception:
                pass

            control_type_id = 0
            try:
                control_type_id = node.CurrentControlType
            except Exception:
                pass

            auto_id = ""
            try:
                auto_id = node.CurrentAutomationId or ""
            except Exception:
                pass

            class_name = ""
            try:
                class_name = node.CurrentClassName or ""
            except Exception:
                pass

            is_enabled = True
            try:
                is_enabled = bool(node.CurrentIsEnabled)
            except Exception:
                pass

            is_offscreen = False
            try:
                is_offscreen = bool(node.CurrentIsOffscreen)
            except Exception:
                pass

            # Get bounding rectangle — this can be a RECT struct or a tagRECT
            bbox = (0, 0, 0, 0)
            try:
                rect = node.CurrentBoundingRectangle
                if hasattr(rect, "left"):
                    bbox = (int(rect.left), int(rect.top), int(rect.right), int(rect.bottom))
                elif isinstance(rect, (tuple, list)) and len(rect) == 4:
                    bbox = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
                else:
                    bbox = (int(rect[0]), int(rect[1]), int(rect[0]) + int(rect[2]), int(rect[1]) + int(rect[3]))
            except Exception:
                pass

            role = _UIA_CONTROL_TYPE_NAMES.get(control_type_id, "unknown")
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            area = w * h

            # Only include elements with a valid size and either a name or an automationId
            if area >= min_area and (name.strip() or auto_id.strip()):
                elem = ScreenElement(
                    name=name.strip(),
                    role=role,
                    bbox=bbox,
                    confidence=1.0,
                    source="uia",
                    automationId=auto_id,
                    className=class_name,
                    is_enabled=is_enabled,
                    is_offscreen=is_offscreen,
                    raw_element=node,
                    hwnd=target_hwnd,
                )
                elements.append(elem)

            # Recurse into children using TreeWalker
            if self._walker:
                try:
                    child = self._walker.GetFirstChildElement(node)
                    while child:
                        self._walk(child, elements, depth + 1, max_depth, min_area, target_hwnd=target_hwnd)
                        try:
                            child = self._walker.GetNextSiblingElement(child)
                        except Exception:
                            break
                except Exception:
                    pass

        except Exception:
            pass


# ──────────────────────── Provider: OCR (RapidOCR) ──────────────────────────


class OCRProvider:
    """Extracts visible text from a screenshot using local ONNX-based OCR."""

    def __init__(self):
        self._engine = None
        self._initialized = False

    def _ensure_init(self):
        if self._initialized:
            return
        try:
            from rapidocr_onnxruntime import RapidOCR
            self._engine = RapidOCR()
            self._initialized = True
        except Exception as e:
            print(f"[ScreenEngine/OCR] Failed to initialize RapidOCR: {e}")
            self._engine = None
            self._initialized = True

    def get_elements(
        self,
        image,
        offset: Tuple[int, int] = (0, 0),
        min_confidence: float = 0.5,
        hwnd: Optional[int] = None,
    ) -> List[ScreenElement]:
        """Run OCR on a PIL Image and return detected text elements.

        Args:
            image: PIL Image (RGB or RGBA).
            offset: (left, top) offset to add to coordinates to convert
                    from image-local to absolute screen coordinates.
            min_confidence: Minimum OCR confidence to include a result.
            hwnd: Optional window handle.

        Returns:
            List of ScreenElement objects with text and bounding boxes.
        """
        self._ensure_init()
        if not self._engine or image is None:
            return []

        try:
            import numpy as np
            # Convert PIL Image to numpy array for RapidOCR
            img_rgb = image.convert("RGB") if image.mode != "RGB" else image
            img_np = np.array(img_rgb)

            result, elapse = self._engine(img_np)
            if not result:
                return []

            elements: List[ScreenElement] = []
            ox, oy = offset

            for detection in result:
                # detection format: [box_points, text, confidence]
                box_points = detection[0]  # 4 corner points [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                text = detection[1]
                conf = detection[2]

                if conf < min_confidence or not text.strip():
                    continue

                # Convert polygon to bounding box
                xs = [p[0] for p in box_points]
                ys = [p[1] for p in box_points]
                left = int(min(xs)) + ox
                top = int(min(ys)) + oy
                right = int(max(xs)) + ox
                bottom = int(max(ys)) + oy

                elem = ScreenElement(
                    name=text.strip(),
                    role="text",
                    bbox=(left, top, right, bottom),
                    confidence=float(conf),
                    source="ocr",
                    hwnd=hwnd,
                )
                elements.append(elem)

            return elements
        except Exception as e:
            print(f"[ScreenEngine/OCR] Error during OCR: {e}")
            return []


# ────────────────────────── Unified Screen Engine ───────────────────────────


class ScreenEngine:
    """Unified screen perception engine that combines UIA + OCR + Vision.

    This is the single entry point for E.D.I.T.H.'s "eyes" — it
    understands what is on any window or screen and can locate any
    interactive element dynamically.
    """

    def __init__(self, enable_uia: bool = True, enable_ocr: bool = True):
        self.uia = UIAProvider() if enable_uia else None
        self.ocr = OCRProvider() if enable_ocr else None
        self._element_cache: Dict[int, Tuple[float, List[ScreenElement]]] = {}
        self._cache_ttl = 2.0  # seconds

    def perceive(
        self,
        hwnd: Optional[int] = None,
        use_ocr: bool = True,
        max_depth: int = 8,
    ) -> List[ScreenElement]:
        """Perceive all elements in a window or the entire screen.

        Strategy:
          1. Run UIA tree walk (instant, high accuracy)
          2. If UIA returns < 5 interactive elements, run OCR as supplement
          3. Merge and deduplicate

        Args:
            hwnd: Window handle.  None = foreground window.
            use_ocr: Whether to enable OCR supplement.
            max_depth: UIA tree walk depth.

        Returns:
            Merged, deduplicated list of ScreenElement objects.
        """
        # Check cache
        cache_key = hwnd or 0
        if cache_key in self._element_cache:
            ts, cached = self._element_cache[cache_key]
            if time.time() - ts < self._cache_ttl:
                return cached

        elements: List[ScreenElement] = []

        # Layer 1: UIA
        uia_elements = []
        if self.uia:
            uia_elements = self.uia.get_elements(hwnd=hwnd, max_depth=max_depth)
            elements.extend(uia_elements)

        # Layer 2: OCR (supplement if UIA is sparse)
        interactive_uia = [e for e in uia_elements if e.role in ("button", "edit", "hyperlink", "menu_item", "tab_item", "list_item", "checkbox", "combobox", "slider")]
        if use_ocr and self.ocr and len(interactive_uia) < 5:
            from runtime.actions import capture_window as _cap_win
            if hwnd:
                img = _cap_win(hwnd)
                if img:
                    from runtime.actions import get_window_rect
                    rect = get_window_rect(hwnd)
                    ocr_elements = self.ocr.get_elements(img, offset=(rect[0], rect[1]), hwnd=hwnd)
                    # Deduplicate: don't add OCR text that UIA already found
                    uia_names_lower = {e.name.lower() for e in uia_elements if e.name}
                    for oe in ocr_elements:
                        if oe.name.lower() not in uia_names_lower:
                            elements.append(oe)

        # Cache
        self._element_cache[cache_key] = (time.time(), elements)
        return elements

    def perceive_ocr_only(
        self,
        hwnd: Optional[int] = None,
        image=None,
    ) -> List[ScreenElement]:
        """Force-run OCR on a window or provided image.

        This is useful when UIA is known to be empty (e.g. game UIs,
        custom-rendered Electron views).

        Args:
            hwnd: Window handle to capture and OCR.
            image: Pre-captured PIL Image (skip capture if provided).

        Returns:
            List of ScreenElement objects from OCR.
        """
        if not self.ocr:
            return []

        offset = (0, 0)
        if image is None and hwnd:
            from runtime.actions import capture_window as _cap_win, get_window_rect
            image = _cap_win(hwnd)
            if image:
                rect = get_window_rect(hwnd)
                offset = (rect[0], rect[1])

        if image is None:
            return []

        return self.ocr.get_elements(image, offset=offset, hwnd=hwnd)

    # ────────────────────── Element Search / Grounding ──────────────────────

    def find(
        self,
        query: str,
        elements: Optional[List[ScreenElement]] = None,
        hwnd: Optional[int] = None,
        role: Optional[str] = None,
        source: Optional[str] = None,
        top_k: int = 1,
    ) -> Optional[ScreenElement]:
        """Find the best-matching element by name (fuzzy search).

        Args:
            query: The text to search for (e.g. "Search", "Play", "Settings").
            elements: Pre-computed element list.  If None, runs perceive().
            hwnd: Window handle (used only if elements is None).
            role: Optional role filter (e.g. "button", "edit").
            source: Optional source filter ("uia", "ocr").
            top_k: Return the k-th best match (1 = best).

        Returns:
            Best matching ScreenElement, or None if nothing matches.
        """
        if elements is None:
            elements = self.perceive(hwnd=hwnd)

        if not elements:
            return None

        # Filter
        candidates = elements
        role_enforced = False
        if role:
            role_filtered = [e for e in candidates if e.role == role]
            if role_filtered:
                candidates = role_filtered
                role_enforced = True
            else:
                # If no elements match requested role, only allow interactive-compatible candidates or OCR
                candidates = [e for e in candidates if e.source == "ocr" or e.role in ("button", "edit", "text", "hyperlink", "menu_item", "tab_item")]
        if source:
            src_filtered = [e for e in candidates if e.source == source]
            if src_filtered:
                candidates = src_filtered

        # Score each candidate
        query_lower = query.lower().strip()
        scored: List[Tuple[float, ScreenElement]] = []

        for elem in candidates:
            name_lower = elem.name.lower().strip()
            if not name_lower:
                continue

            # Exact match
            if name_lower == query_lower:
                score = 1.0
            # Substring match
            elif query_lower in name_lower or name_lower in query_lower:
                score = 0.9
            # Word-level match
            elif any(w == query_lower for w in name_lower.split()):
                score = 0.85
            # Fuzzy match
            else:
                score = difflib.SequenceMatcher(None, query_lower, name_lower).ratio()

            # Boost interactive roles
            if elem.role in ("button", "edit", "hyperlink", "menu_item", "tab_item"):
                score += 0.05

            # Boost UIA over OCR (more reliable coordinates)
            if elem.source == "uia":
                score += 0.02

            # Penalize offscreen elements
            if elem.is_offscreen:
                score -= 0.3

            # If a specific role was requested but not met, penalize non-matching elements
            if role and not role_enforced and elem.role != role:
                score -= 0.15

            scored.append((score, elem))

        if not scored:
            return None

        scored.sort(key=lambda x: x[0], reverse=True)
        idx = min(top_k - 1, len(scored) - 1)
        # Require higher confidence threshold (0.55) to prevent unrelated fuzzy collisions
        min_threshold = 0.65 if (role and not role_enforced) else 0.55
        return scored[idx][1] if scored[idx][0] >= min_threshold else None

    def find_all(
        self,
        query: str,
        elements: Optional[List[ScreenElement]] = None,
        hwnd: Optional[int] = None,
        role: Optional[str] = None,
        threshold: float = 0.5,
    ) -> List[ScreenElement]:
        """Find all elements matching a query above a confidence threshold.

        Returns:
            List of matching ScreenElements, sorted by match score descending.
        """
        if elements is None:
            elements = self.perceive(hwnd=hwnd)

        query_lower = query.lower().strip()
        results: List[Tuple[float, ScreenElement]] = []

        for elem in elements:
            if role and elem.role != role:
                continue
            name_lower = elem.name.lower().strip()
            if not name_lower:
                continue

            if name_lower == query_lower:
                score = 1.0
            elif query_lower in name_lower or name_lower in query_lower:
                score = 0.9
            else:
                score = difflib.SequenceMatcher(None, query_lower, name_lower).ratio()

            if score >= threshold:
                results.append((score, elem))

        results.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in results]

    def find_by_role(
        self,
        role: str,
        elements: Optional[List[ScreenElement]] = None,
        hwnd: Optional[int] = None,
    ) -> List[ScreenElement]:
        """Return all elements with a specific role (e.g. 'button', 'edit')."""
        if elements is None:
            elements = self.perceive(hwnd=hwnd)
        return [e for e in elements if e.role == role]

    # ────────────────────── Convenience Actions ─────────────────────────────

    def click(self, element: ScreenElement, button: str = "left", clicks: int = 1):
        """Click the center of a ScreenElement."""
        from runtime.actions import click as _click
        _click(element.center[0], element.center[1], button=button, clicks=clicks)

    def double_click(self, element: ScreenElement):
        """Double-click the center of a ScreenElement."""
        from runtime.actions import double_click as _dclick
        _dclick(element.center[0], element.center[1])

    def type_into(self, element: ScreenElement, text: str, clear_first: bool = True):
        """Click an element (typically an edit box), optionally clear it, then type text.

        Args:
            element: Target element (should be an edit/input).
            text: Text to type.
            clear_first: If True, sends Ctrl+A → Backspace before typing.
        """
        from runtime import actions
        self.click(element)
        time.sleep(0.1)
        if clear_first:
            actions.hotkey(0x11, 0x41)  # Ctrl+A
            time.sleep(0.05)
            actions.press_key(0x08)     # Backspace
            time.sleep(0.1)
        actions.type_text(text)

    def scroll_to(self, element: ScreenElement, amount: int = -3):
        """Scroll at the element's center position."""
        from runtime.actions import scroll as _scroll
        _scroll(element.center[0], element.center[1], amount=amount)

    # ────────────────────── Debug / Inspection ──────────────────────────────

    def dump(
        self,
        elements: Optional[List[ScreenElement]] = None,
        hwnd: Optional[int] = None,
        max_display: int = 80,
    ) -> str:
        """Return a formatted text dump of all perceived elements.

        Useful for debugging and understanding what E.D.I.T.H. sees.
        """
        if elements is None:
            elements = self.perceive(hwnd=hwnd)

        lines = [f"{'#':>3} | {'Source':>4} | {'Role':<12} | {'Name':<40} | {'BBox'}"]
        lines.append("-" * 100)
        for i, elem in enumerate(elements):
            name_disp = elem.name[:max_display] if len(elem.name) > max_display else elem.name
            lines.append(
                f"{i:>3} | {elem.source:>4} | {elem.role:<12} | {name_disp:<40} | {elem.bbox}"
            )
        return "\n".join(lines)

    def annotate_image(self, image, elements: List[ScreenElement], offset: Tuple[int, int] = (0, 0)):
        """Draw numbered bounding boxes on a PIL Image (Set-of-Marks style).

        Args:
            image: PIL Image to annotate (will be modified in-place).
            elements: Elements to draw.
            offset: (left, top) to subtract from absolute coords to get image-local coords.

        Returns:
            Annotated PIL Image.
        """
        try:
            from PIL import ImageDraw, ImageFont
        except ImportError:
            return image

        draw = ImageDraw.Draw(image)
        ox, oy = offset

        colors = {
            "uia": (0, 180, 255),   # Cyan-ish for UIA
            "ocr": (255, 180, 0),   # Orange for OCR
            "vision": (180, 0, 255), # Purple for Vision
        }

        for i, elem in enumerate(elements):
            color = colors.get(elem.source, (255, 255, 255))
            l, t, r, b = elem.bbox
            # Convert to image-local coordinates
            il, it, ir, ib = l - ox, t - oy, r - ox, b - oy

            draw.rectangle([il, it, ir, ib], outline=color, width=2)

            label = f"[{i}] {elem.name[:30]}"
            try:
                draw.text((il + 2, it - 14), label, fill=color)
            except Exception:
                draw.text((il + 2, it + 2), label, fill=color)

        return image

    def clear_cache(self):
        """Clear the element cache."""
        self._element_cache.clear()
