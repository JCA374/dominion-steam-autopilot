"""Read Steam Dominion's screen: prompt line, hand, coins/actions/buys, buttons, supply piles.

All coordinates are for the reference window size (REF_W x REF_H) and are scaled
to the live window size. The game scales its board with the window width and pins
the top half (supply, prompt line) to the top and the bottom half (hand, counters,
buttons) to the bottom, so any window shape works, landscape too.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

REF_W, REF_H = 927, 1048

# Regions (x0, y0, x1, y1) in reference coordinates.
R_PROMPT = (150, 292, 780, 326)
R_PROMPT_TIGHT = (0, 300, 927, 330)
R_OPPONENT = (230, 0, 700, 40)     # the opponent's bar: name, or "Added: <cards>" after a gain
R_HAND = (100, 820, 780, 1000)
R_HUD = (0, 735, 200, 782)
R_BUTTONS = (760, 700, 927, 900)
R_KINGDOM = (190, 40, 870, 200)
R_TOP = (0, 0, 927, 1048)

# Base piles: centre of the pile (where to click) and the box holding its count.
BASE_PILES = {
    "Province": (90, 75), "Gold": (157, 75),
    "Duchy": (90, 138), "Silver": (157, 138),
    "Estate": (90, 200), "Copper": (157, 200),
    "Curse": (90, 262),
}
PROVINCE_COUNT = (100, 84, 128, 104)

_ocr = None


def ocr_engine():
    global _ocr
    if _ocr is None:
        _ocr = RapidOCR()
    return _ocr


@dataclass
class Text:
    text: str
    x: int   # centre, window coordinates
    y: int
    conf: float
    count: int = 1   # hand cards: a large hand shows duplicates as one card with a count badge


@dataclass
class Screen:
    prompt: str = ""
    hand: list[Text] = field(default_factory=list)
    buttons: list[Text] = field(default_factory=list)
    actions: int | None = None
    coins: int | None = None
    buys: int | None = None


class Vision:
    def __init__(self, width: int, height: int):
        self.sx = self.sy = width / REF_W      # one scale for distances
        self.height = height
        self.on_read = None   # optional callback(region_px, texts, ms), used by the recorder

    def scale(self, x, y):
        """Reference -> window coordinates: the top half hangs from the top, the bottom half stands on the bottom."""
        return int(x * self.sx), int(y * self.sy if y < REF_H / 2 else self.height - (REF_H - y) * self.sy)

    def unscale(self, x, y):
        """Window -> reference coordinates."""
        ry = y / self.sy
        if ry >= REF_H / 2:
            ry = REF_H - (self.height - y) / self.sy
        return x / self.sx, ry

    def read(self, img: np.ndarray, region, cls=True) -> list[Text]:
        """OCR one region. cls=False turns off the reader's 180-degree orientation check."""
        x0, y0 = self.scale(*region[:2])
        x1, y1 = self.scale(*region[2:])
        t = time.time()
        res, _ = ocr_engine()(img[y0:y1, x0:x1], use_cls=cls)
        ms = int((time.time() - t) * 1000)
        out = []
        for box, txt, conf in res or []:
            cx = int(sum(p[0] for p in box) / 4) + x0
            cy = int(sum(p[1] for p in box) / 4) + y0
            out.append(Text(txt.strip(), cx, cy, float(conf)))
        if self.on_read:
            self.on_read((x0, y0, x1, y1), out, ms)
        return out

    # -- individual readers ------------------------------------------------

    def opponent_bar(self, img) -> str:
        return " ".join(t.text for t in sorted(self.read(img, R_OPPONENT, cls=False), key=lambda t: t.x))

    def prompt(self, img) -> str:
        text = " ".join(t.text for t in self.read(img, R_PROMPT))
        if not text:       # big played cards right under the line can hide it from the wider crop
            text = " ".join(t.text for t in self.read(img, R_PROMPT_TIGHT, cls=False))
        return text

    def hand(self, img, card_names: set[str]) -> list[Text]:
        """Card-name banners in the hand, matched against known card names."""
        upper = {n.upper(): n for n in card_names}
        cards = []
        # Hand cards are never upside down, but the orientation check sometimes thinks the tilted
        # leftmost card is (SMITHY read as "AHLIWS"), so it is off for the hand.
        texts = self.read(img, R_HAND, cls=False)
        unmatched = []
        for t in texts:
            name = match_name(t.text, upper)
            if name:
                cards.append(Text(name, t.x, t.y, t.conf))
            elif len(re.sub(r"[^A-Za-z]", "", t.text)) >= 4:
                unmatched.append(t)
        near = int(60 * self.sx)
        for t in unmatched:          # a second read can find a banner already read nearby: skip those
            if any(abs(t.x - c.x) < near for c in cards):
                continue
            name = self._reread_flipped(img, t, upper)
            if name:
                cards.append(Text(name, t.x, t.y, t.conf))
        # Stack badges: a small number just left of a card's name (the coin value sits below it).
        for c in cards:
            for t in texts:
                if t.text.isdigit() and -75 * self.sx <= t.x - c.x <= -35 * self.sx \
                        and abs(t.y - c.y) <= 25 * self.sy and 1 < int(t.text) < 10:
                    c.count = int(t.text)
        return sorted(cards, key=lambda t: t.x)

    def _reread_flipped(self, img, t, upper):
        """Read one unmatched word's patch again: as it is, then turned 180 degrees,
        both with the orientation check off."""
        h, w = img.shape[:2]
        dx, dy = int(90 * self.sx), int(30 * self.sy)
        patch = img[max(0, t.y - dy):min(h, t.y + dy), max(0, t.x - dx):min(w, t.x + dx)]
        for p in (patch, cv2.rotate(patch, cv2.ROTATE_180)):
            res, _ = ocr_engine()(p, use_cls=False)
            for _, txt, _ in res or []:
                name = match_name(txt, upper)
                if name:
                    return name
        return None

    def hud(self, img):
        """(actions, coins, buys) from the counters left of the hand."""
        nums = [t for t in self.read(img, R_HUD) if re.fullmatch(r"\d+", t.text)]
        nums.sort(key=lambda t: t.x)
        vals = [int(t.text) for t in nums]
        if len(vals) >= 3:
            return vals[0], vals[1], vals[2]
        return None, None, None

    def buttons(self, img) -> list[Text]:
        """Button labels; two-line labels ("Play" / "Treasures") are joined."""
        parts = sorted(self.read(img, R_BUTTONS), key=lambda t: t.y)
        merged: list[Text] = []
        for t in parts:
            if merged and abs(merged[-1].x - t.x) < 40 and t.y - merged[-1].y < 30:
                m = merged[-1]
                merged[-1] = Text(f"{m.text} {t.text}", m.x, (m.y + t.y) // 2, min(m.conf, t.conf))
            else:
                merged.append(t)
        return merged

    def kingdom_piles(self, img, kingdom: list[str]) -> dict[str, tuple[int, int]]:
        """Click positions for the kingdom piles, found by their name banners."""
        upper = {n.upper(): n for n in kingdom}
        piles = {}
        for t in self.read(img, R_KINGDOM, cls=False):
            name = match_name(t.text, upper)
            if name:
                piles[name] = (t.x, t.y + int(40 * self.sy))  # click the art, below the banner
        return piles

    def base_pile(self, name: str) -> tuple[int, int]:
        return self.scale(*BASE_PILES[name])

    def provinces_left(self, img) -> int | None:
        nums = [t for t in self.read(img, PROVINCE_COUNT) if t.text.isdigit()]
        return int(nums[0].text) if nums else None

    def screen(self, img, card_names: set[str]) -> Screen:
        a, c, b = self.hud(img)
        return Screen(prompt=self.prompt(img), hand=self.hand(img, card_names),
                      buttons=self.buttons(img), actions=a, coins=c, buys=b)


def match_name(text: str, upper_to_name: dict[str, str]) -> str | None:
    """Fuzzy-match OCR text to a card name (OCR sometimes drops a letter)."""
    t = re.sub(r"[^A-Z ]", "", text.upper()).strip()
    if not t:
        return None
    if t in upper_to_name:
        return upper_to_name[t]
    best, best_score = None, 0.0
    for up, name in upper_to_name.items():
        score = _similarity(t, up)
        if score > best_score:
            best, best_score = name, score
    return best if best_score >= 0.8 else None


def parse_added(text: str, card_names) -> list[str] | None:
    """The opponent's bar after a gain: 'Added: 2 Cellars' / 'Added: Harbinger and Copper'
    -> ['Cellar', 'Cellar', 'Harbinger', 'Copper']; None when the bar shows no gain."""
    m = re.search(r"added\s*:?\s*(.*)", text, re.I)
    if not m:
        return None
    names = {n.upper().replace(" ", ""): n for n in card_names}
    out = []
    for item in re.split(r",|\band\b|&", m.group(1)):
        item = item.strip()
        k = re.match(r"(\d+)\s*(.*)", item)
        count, word = (int(k.group(1)), k.group(2)) if k else (1, item)
        key = re.sub(r"[^A-Za-z]", "", word).upper()
        name = match_name(key, names) or (match_name(key[:-1], names) if key.endswith("S") else None)
        if name and count < 10:
            out += [name] * count
    return out or None


def _similarity(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio()
