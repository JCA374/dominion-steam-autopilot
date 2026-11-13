"""Capture the Steam Dominion window (XWayland) as a numpy BGR image."""
import sys
import numpy as np
from Xlib import display, X

def find_window(d, name="Dominion"):
    def walk(w):
        try:
            if w.get_wm_name() == name and w.get_geometry().width > 100:
                return w
        except Exception:
            pass
        for c in w.query_tree().children:
            r = walk(c)
            if r:
                return r
    return walk(d.screen().root)

def grab(d=None, win=None):
    d = d or display.Display()
    win = win or find_window(d)
    g = win.get_geometry()
    raw = win.get_image(0, 0, g.width, g.height, X.ZPixmap, 0xffffffff)
    img = np.frombuffer(raw.data, dtype=np.uint8).reshape(g.height, g.width, 4)[:, :, :3]
    return img.copy()

if __name__ == "__main__":
    import cv2
    cv2.imwrite(sys.argv[1] if len(sys.argv) > 1 else "shot.png", grab())
