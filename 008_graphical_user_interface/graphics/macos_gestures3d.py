"""Native macOS trackpad gestures for GLFW, without PyObjC.

Why this exists
---------------

GLFW 3.4 has exactly one input channel for the trackpad: the scroll callback.
So:

* a mouse **wheel** is reported as integer scroll deltas, but so is a trackpad
  **two-finger scroll**, only scaled by 0.1 — the callback cannot tell them apart;
* AppKit's ``magnifyWithEvent:`` (the **pinch** gesture) is *not forwarded at all*
  (checked against ``src/cocoa_window.m`` in GLFW 3.4 and master): `GLFWContentView`
  implements ``scrollWheel:`` but no magnification handler, so a pinch produces
  no GLFW event.

This module fills both holes by talking to the Objective-C runtime through
``ctypes`` (no PyObjC dependency):

1. it adds ``magnifyWithEvent:`` to GLFW's content-view class, which calls
   ``on_pinch(magnification)`` for every pinch step;
2. it wraps the existing ``scrollWheel:`` (calling the original implementation
   first-class, so GLFW's scroll callback still fires) and records
   ``hasPreciseScrollingDeltas`` for the duration of that call. Read it from the
   scroll callback with :func:`scroll_is_precise` to tell a trackpad pan from a
   wheel notch.

Everything degrades gracefully: off macOS (or if the runtime looks different from
what we expect) :func:`install` returns ``False`` and `main.py` falls back to
wheel-zoom plus middle-drag pan.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import sys
import traceback
from typing import Any

# Set for the duration of one precise (trackpad) scroll event. The GLFW scroll
# callback runs *inside* GLFW's original `scrollWheel:`, i.e. between the two
# assignments in `_scroll_imp`, so the flag is valid exactly when we need it.
_scroll_is_precise = False

# Handler installed by `install`; kept at module level so the ctypes callbacks
# below stay alive for the lifetime of the process.
_on_pinch = None


def scroll_is_precise() -> bool:
    """True if GLFW's scroll callback is running for a precise (trackpad) scroll.

    Precise deltas come from a trackpad's two-finger scroll, which we use for
    panning; non-precise deltas are a mouse wheel (or equivalent), used for zoom.
    Always ``False`` off macOS, where the app uses the fallback mapping.
    """
    return _scroll_is_precise


# --- ctypes plumbing ------------------------------------------------------


class _Runtime:
    """The handful of Objective-C runtime calls needed to hook a view class."""

    def __init__(self) -> None:
        path = ctypes.util.find_library("objc") or "/usr/lib/libobjc.A.dylib"
        self.lib = ctypes.CDLL(path)
        self.register_name = _bind(self.lib, "sel_registerName", ctypes.c_void_p, [ctypes.c_char_p])
        self.object_get_class = _bind(
            self.lib, "object_getClass", ctypes.c_void_p, [ctypes.c_void_p]
        )
        self.class_get_name = _bind(self.lib, "class_getName", ctypes.c_char_p, [ctypes.c_void_p])
        self.class_add_method = _bind(
            self.lib,
            "class_addMethod",
            ctypes.c_bool,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p],
        )
        self.class_get_instance_method = _bind(
            self.lib,
            "class_getInstanceMethod",
            ctypes.c_void_p,
            [ctypes.c_void_p, ctypes.c_void_p],
        )
        self.method_get_implementation = _bind(
            self.lib,
            "method_getImplementation",
            ctypes.c_void_p,
            [ctypes.c_void_p],
        )
        self.class_replace_method = _bind(
            self.lib,
            "class_replaceMethod",
            ctypes.c_void_p,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p],
        )

        # objc_msgSend is deliberately untyped in C; each use needs its own cast
        # so ctypes gets the calling convention of the return value right.
        self.send_ptr = _msg_send(self.lib, ctypes.c_void_p)
        self.send_double = _msg_send(self.lib, ctypes.c_double)
        self.send_bool = _msg_send(self.lib, ctypes.c_bool)

    def sel(self, name: str) -> int:
        return int(self.register_name(name.encode()))


def _bind(lib: Any, name: str, restype: Any, argtypes: list[Any]) -> Any:
    """Fetch one exported C function with an explicit signature."""
    fn = getattr(lib, name)
    fn.restype = restype
    fn.argtypes = argtypes
    return fn


def _msg_send(lib: Any, restype: Any) -> Any:
    """A typed ``objc_msgSend``: ``(receiver, selector) -> restype``."""
    prototype = ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p)
    return prototype(("objc_msgSend", lib))


# An Objective-C IMP for `void method(id self, SEL _cmd, id event)`.
_IMP = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)

_original_scroll: Any = None
_hooked_classes: set[int] = set()


def _magnify_imp(self_view, _cmd, event) -> None:
    """- (void)magnifyWithEvent:(NSEvent *)event — pinch on the content view."""
    runtime = _runtime
    handler = _on_pinch
    try:
        if runtime is None or handler is None:
            return
        magnification = runtime.send_double(event, runtime.sel("magnification"))
        if magnification:
            handler(float(magnification))
    except Exception:  # never let a callback kill Cocoa's event loop
        traceback.print_exc()


def _scroll_imp(self_view, _cmd, event) -> None:
    """- (void)scrollWheel:(NSEvent *)event — record precision, then defer to GLFW."""
    global _scroll_is_precise
    runtime = _runtime
    try:
        if runtime is None:
            _scroll_is_precise = False
        else:
            _scroll_is_precise = bool(
                runtime.send_bool(event, runtime.sel("hasPreciseScrollingDeltas"))
            )
    except Exception:
        traceback.print_exc()
        _scroll_is_precise = False
    try:
        if _original_scroll is not None:
            _original_scroll(self_view, _cmd, event)
    finally:
        _scroll_is_precise = False


# Keep the IMP objects alive: AppKit holds raw function pointers to them.
_magnify_imp_ref = _IMP(_magnify_imp)
_scroll_imp_ref = _IMP(_scroll_imp)

_runtime: Any = None


def _hook(window) -> bool:
    """Add ``magnifyWithEvent:`` and wrap ``scrollWheel:`` on GLFW's content view."""
    global _runtime, _original_scroll

    import glfw

    get_cocoa_window = getattr(glfw, "get_cocoa_window", None)
    if get_cocoa_window is None:
        return False

    ns_window = get_cocoa_window(window)
    if not ns_window:
        return False

    runtime = _Runtime()
    view = runtime.send_ptr(ns_window, runtime.sel("contentView"))
    if not view:
        return False
    view_class = runtime.object_get_class(view)
    if view_class in _hooked_classes:
        _runtime = runtime
        return True

    # `class_replaceMethod` adds the method on this class if it is absent and
    # replaces the IMP if it is present, without touching the superclass (NSView).
    # Read the current IMP first: if the selector were only inherited (or missing),
    # replacing it would install a wrapper that has nothing to delegate to and
    # would silently swallow every scroll event, so bail out instead.
    selector = runtime.sel("scrollWheel:")
    method = runtime.class_get_instance_method(view_class, selector)
    original = runtime.method_get_implementation(method) if method else 0
    if not original:
        return False

    previous = runtime.class_replace_method(view_class, selector, _scroll_imp_ref, b"v@:@")
    _original_scroll = _IMP(previous or original)

    # Not implemented by GLFW, so this is a pure addition: NSView has a no-op
    # default and AppKit reaches it through normal responder dispatch.
    runtime.class_add_method(
        view_class, runtime.sel("magnifyWithEvent:"), _magnify_imp_ref, b"v@:@"
    )

    _hooked_classes.add(view_class)
    _runtime = runtime
    name = runtime.class_get_name(view_class)
    print(f"[gestures] native pinch + precise-scroll hook installed on {name.decode()}")
    return True


def install(window, on_pinch) -> bool:
    """Attach the macOS gesture hooks to `window`; `on_pinch(magnification)` is called per pinch.

    Returns True when native gestures are active. Safe to call on any platform:
    it returns False immediately off macOS and leaves GLFW untouched.
    """
    global _on_pinch
    _on_pinch = on_pinch
    if sys.platform != "darwin":
        return False
    try:
        return _hook(window)
    except Exception:
        traceback.print_exc()
        return False
