"""macOS NSWindow helpers (safe, no CFUNCTYPE)."""
import sys


def boost_window_level(widget) -> None:
    """
    Set the NSWindow backing `widget` to NSStatusBarWindowLevel (25),
    join all Spaces, and disable hidesOnDeactivate so the window never
    disappears when the user switches to another app.
    """
    if sys.platform != "darwin":
        return
    try:
        import ctypes
        lib = ctypes.CDLL("/usr/lib/libobjc.A.dylib")

        lib.sel_registerName.restype  = ctypes.c_void_p
        lib.sel_registerName.argtypes = [ctypes.c_char_p]

        # NSView → NSWindow
        lib.objc_msgSend.restype  = ctypes.c_void_p
        lib.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        ns_win = lib.objc_msgSend(
            ctypes.c_void_p(int(widget.winId())),
            lib.sel_registerName(b"window"),
        )
        if not ns_win:
            return
        win = ctypes.c_void_p(ns_win)

        # integer-arg calls (NSInteger / NSUInteger = c_long)
        lib.objc_msgSend.restype  = None
        lib.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]
        lib.objc_msgSend(win, lib.sel_registerName(b"setLevel:"), ctypes.c_long(25))
        # CanJoinAllSpaces(1) | Stationary(16) | FullScreenAuxiliary(256)
        lib.objc_msgSend(win, lib.sel_registerName(b"setCollectionBehavior:"),
                         ctypes.c_long(1 | 16 | 256))

        # bool-arg call — disable NSPanel's hidesOnDeactivate
        lib.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool]
        lib.objc_msgSend(win, lib.sel_registerName(b"setHidesOnDeactivate:"),
                         ctypes.c_bool(False))
    except Exception:
        pass
