"""Capture the Steam Dominion window (XWayland) as a numpy BGR image.

The game can run in a borderless window or in a window with a Wine-drawn title bar and
border (after leaving fullscreen). grab() returns only the game area either way, and
content_offset() tells click.py where that area starts inside the X window.
"""
import ctypes
import ctypes.util
import sys

import numpy as np
from Xlib import display, X

TITLE_BAR = (250, 150, 50)       # BGR of the Wine title bar
FRAME = (4, 30, 4, 4)            # left, top, right, bottom border of the decorated window


def find_window(d, name="Dominion"):
    best = None

    def walk(w):
        nonlocal best
        try:
            if w.get_wm_name() == name:
                g = w.get_geometry()
                if g.width > 100 and (best is None or g.width * g.height > best[1]):
                    best = (w, g.width * g.height)       # the main window, not a dialog
        except Exception:
            pass
        for c in w.query_tree().children:
            walk(c)
    walk(d.screen().root)
    return best[0] if best else None


def content_offset(d, win):
    """(left, top, right, bottom) border around the game area inside the X window.

    The Wine frame's title bar is blue while the window has focus and grey when it hasn't, so the frame is told by
    its light top edge (the game's own top edge is dark wood), checked on every call."""
    g = win.get_geometry()
    px = np.frombuffer(win.get_image(0, 2, g.width, 1, X.ZPixmap, 0xffffffff).data, np.uint8).reshape(-1, 4)[:, :3]
    decorated = px.mean() > 180 or (abs(int(px[g.width // 2][0]) - TITLE_BAR[0]) < 12
                                     and abs(int(px[g.width // 2][1]) - TITLE_BAR[1]) < 12)
    return FRAME if decorated else (0, 0, 0, 0)


def grab(d=None, win=None):
    d = d or display.Display()
    win = win or find_window(d)
    g = win.get_geometry()
    l, t, r, b = content_offset(d, win)
    h, w = g.height - t - b, g.width - l - r
    img = _fast_image(win.id, l, t, w, h)
    if img is not None:
        return img
    raw = win.get_image(l, t, w, h, X.ZPixmap, 0xffffffff)
    img = np.frombuffer(raw.data, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
    return img.copy()


# libX11's XGetImage through ctypes: 5-10x faster than python-xlib, which unpacks the image in Python.
class _XImage(ctypes.Structure):
    _fields_ = [("width", ctypes.c_int), ("height", ctypes.c_int), ("xoffset", ctypes.c_int),
                ("format", ctypes.c_int), ("data", ctypes.c_void_p), ("byte_order", ctypes.c_int),
                ("bitmap_unit", ctypes.c_int), ("bitmap_bit_order", ctypes.c_int), ("bitmap_pad", ctypes.c_int),
                ("depth", ctypes.c_int), ("bytes_per_line", ctypes.c_int), ("bits_per_pixel", ctypes.c_int)]


_x11 = None


def _fast_image(wid, x, y, w, h):
    """The window area as BGR, or None when libX11 is not usable (python-xlib does it then)."""
    global _x11
    if _x11 is None:
        try:
            lib = ctypes.CDLL(ctypes.util.find_library("X11"))
            lib.XOpenDisplay.restype = ctypes.c_void_p
            lib.XOpenDisplay.argtypes = [ctypes.c_char_p]
            lib.XGetImage.restype = ctypes.POINTER(_XImage)
            lib.XGetImage.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int, ctypes.c_uint,
                                      ctypes.c_uint, ctypes.c_ulong, ctypes.c_int]
            lib.XDestroyImage.argtypes = [ctypes.POINTER(_XImage)]
            dpy = lib.XOpenDisplay(None)
            _x11 = (lib, dpy) if dpy else False
        except (OSError, AttributeError, TypeError):
            _x11 = False
    if not _x11:
        return None
    lib, dpy = _x11
    im = lib.XGetImage(dpy, wid, x, y, w, h, 0xffffffff, 2)      # 2 = ZPixmap
    if not im:
        return None
    try:
        c = im.contents
        if c.bits_per_pixel != 32:
            return None
        buf = ctypes.string_at(c.data, c.bytes_per_line * c.height)
        return np.frombuffer(buf, np.uint8).reshape(c.height, c.bytes_per_line // 4, 4)[:, :c.width, :3].copy()
    finally:
        lib.XDestroyImage(im)


if __name__ == "__main__":
    import cv2
    cv2.imwrite(sys.argv[1] if len(sys.argv) > 1 else "shot.png", grab())
