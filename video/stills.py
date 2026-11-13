"""Still images for the site (docs/img/), all taken from the recordings.

    .venv/bin/python video/stills.py
"""
import os
import sys

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import BAD, GOLD, Recording, font, game_view  # noqa: E402

ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "docs", "img")
os.makedirs(OUT, exist_ok=True)


def save(img, name, w=None):
    if w:
        img = img.resize((w, int(img.height * w / img.width)), Image.LANCZOS)
    img.save(os.path.join(OUT, name), quality=82)
    print(name, img.size)


def main():
    r1 = Recording(os.path.join(ROOT, "recordings", "session1"))

    # what it reads: the moment of the first action decision, with the last reads drawn
    e = r1.find("decide_action")
    save(game_view(r1, e["t"] + 0.05, ocr_window=25, scale_h=1048), "ocr.jpg", 700)

    # the upside-down Smithy: turn 7 of session1, the leftmost card banner
    e = r1.find("decide_action", lambda e: e["turn"] == 7)
    img = r1.frame(e["t"] - 0.5).copy()
    d = ImageDraw.Draw(img)
    d.rectangle((134, 870, 300, 940), outline=BAD, width=5)
    d.rounded_rectangle((150, 760, 520, 820), radius=10, fill=(20, 20, 26))
    d.text((164, 772), 'read as "AHLIWS"', font=font(30, True), fill=GOLD)
    save(img.crop((0, 700, 927, 1048)), "flip.jpg", 800)

    # the empty-pile stall: the end of session1 (turn 52)
    img = r1.frame(r1.times[-1] - 1).copy()
    d = ImageDraw.Draw(img)
    d.rectangle((54, 102, 124, 230), outline=BAD, width=5)
    save(img, "empty.jpg", 600)

    # score screens
    for name in os.listdir(os.path.join(ROOT, "autopilot", "results")):
        if name.endswith(".png"):
            img = Image.open(os.path.join(ROOT, "autopilot", "results", name)).convert("RGB")
            save(img.crop((60, 250, 870, 810)), name.replace(".png", ".jpg"), 640)


if __name__ == "__main__":
    main()
