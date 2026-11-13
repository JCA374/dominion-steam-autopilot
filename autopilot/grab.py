"""Capture the Steam Dominion window (XWayland) as a numpy BGR image.

The game can run in a borderless window or in a window with a Wine-drawn title bar and
border (after leaving fullscreen). grab() returns only the game area either way, and
content_offset() tells click.py where that area starts inside the X window.
"""
import sys
import numpy as np
from Xlib import display, X

TITLE_BAR = (250, 150, 50)       # BGR of the Wine title bar
FRAME = (4, 30, 4, 4)            # left, top, right, bottom border of the decorated window
_offsets = {}


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
    """(left, top, right, bottom) border around the game area inside the X window."""
    g = win.get_geometry()
    key = (win.id, g.width, g.height)
    if key not in _offsets:
        px = win.get_image(g.width // 2, 12, 1, 1, X.ZPixmap, 0xffffffff).data
        b, gr, r = px[0], px[1], px[2]
        decorated = abs(b - TITLE_BAR[0]) < 12 and abs(gr - TITLE_BAR[1]) < 12 and abs(r - TITLE_BAR[2]) < 12
        _offsets[key] = FRAME if decorated else (0, 0, 0, 0)
    return _offsets[key]


def grab(d=None, win=None):
    d = d or display.Display()
    win = win or find_window(d)
    g = win.get_geometry()
    l, t, r, b = content_offset(d, win)
    raw = win.get_image(l, t, g.width - l - r, g.height - t - b, X.ZPixmap, 0xffffffff)
    h, w = g.height - t - b, g.width - l - r
    img = np.frombuffer(raw.data, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
    return img.copy()


if __name__ == "__main__":
    import cv2
    cv2.imwrite(sys.argv[1] if len(sys.argv) > 1 else "shot.png", grab())
