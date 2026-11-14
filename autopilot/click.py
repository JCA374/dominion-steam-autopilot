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


def drag(d, win, x0, y0, x1, y1, hover=0.5, steps=20, step_time=0.02):
    """Drag with the left button from (x0, y0) to (x1, y1), game-area coordinates."""
    from grab import content_offset
    left, top, _, _ = content_offset(d, win)
    pos = win.translate_coords(d.screen().root, 0, 0)
    ox, oy = -pos.x + left, -pos.y + top
    xtest.fake_input(d, X.MotionNotify, x=ox + x0, y=oy + y0)
    d.sync()
    time.sleep(hover)
    xtest.fake_input(d, X.ButtonPress, 1)
    d.sync()
    time.sleep(0.2)
    for i in range(1, steps + 1):
        xtest.fake_input(d, X.MotionNotify, x=ox + x0 + (x1 - x0) * i // steps, y=oy + y0 + (y1 - y0) * i // steps)
        d.sync()
        time.sleep(step_time)
    time.sleep(0.3)
    xtest.fake_input(d, X.ButtonRelease, 1)
    d.sync()


def key(d, win, name, shift=False):
    """Press one key (an X keysym name such as "a", "space", "Return") in the window."""
    from Xlib import XK
    code = d.keysym_to_keycode(XK.string_to_keysym(name))
    shift_code = d.keysym_to_keycode(XK.string_to_keysym("Shift_L"))
    win.set_input_focus(X.RevertToParent, X.CurrentTime)
    if shift:
        xtest.fake_input(d, X.KeyPress, shift_code)
    xtest.fake_input(d, X.KeyPress, code)
    d.sync()
    time.sleep(0.05)
    xtest.fake_input(d, X.KeyRelease, code)
    if shift:
        xtest.fake_input(d, X.KeyRelease, shift_code)
    d.sync()
    time.sleep(0.05)


def type_text(d, win, text):
    """Type plain text (letters, digits, spaces) into the focused field."""
    for ch in text:
        if ch == " ":
            key(d, win, "space")
        else:
            key(d, win, ch.lower(), shift=ch.isupper())
