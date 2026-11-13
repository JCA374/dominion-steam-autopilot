"""Send mouse clicks to the Dominion window via XTEST (works for XWayland windows).

The game only registers a click after the pointer has hovered on the target for a
moment, so glide there, rest, then press and hold briefly.
"""
import time
from Xlib import X
from Xlib.ext import xtest


def click(d, win, x, y, hover=0.5, hold=0.15):
    """Click at window-relative (x, y)."""
    from grab import content_offset
    left, top, _, _ = content_offset(d, win)      # (x, y) are game-area coordinates
    pos = win.translate_coords(d.screen().root, 0, 0)
    ax, ay = -pos.x + left + x, -pos.y + top + y
    for dx in (6, 3, 0):
        xtest.fake_input(d, X.MotionNotify, x=ax + dx, y=ay + dx)
        d.sync()
        time.sleep(0.04)
    time.sleep(hover)
    xtest.fake_input(d, X.ButtonPress, 1)
    d.sync()
    time.sleep(hold)
    xtest.fake_input(d, X.ButtonRelease, 1)
    d.sync()
