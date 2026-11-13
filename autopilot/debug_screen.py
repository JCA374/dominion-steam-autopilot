"""Print what the vision layer reads from a screenshot file (or the live window)."""
import sys, time
import cv2
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from vision import Vision

NAMES = {"Copper", "Silver", "Gold", "Estate", "Duchy", "Province", "Curse",
         "Chapel", "Harbinger", "Vassal", "Workshop", "Gardens", "Moneylender",
         "Remodel", "Smithy", "Market", "Artisan"}

if len(sys.argv) > 1:
    img = cv2.imread(sys.argv[1])
else:
    from grab import grab
    img = grab()
v = Vision(img.shape[1], img.shape[0])
t = time.time()
print(v.screen(img, NAMES))
print("provinces", v.provinces_left(img))
print("piles", v.kingdom_piles(img, sorted(NAMES)))
print("secs", round(time.time() - t, 1))
