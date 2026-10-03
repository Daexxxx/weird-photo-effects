import os
import numpy as np
from PIL import Image
from numpy.random import rand
import sys

SRC = sys.argv[1]
LO, HI = 20, 245        # brightness range that gets sorted
BREAK = 0.005            # 0.005 = long streaks, 0.05 = choppy
BAND = 20               # streak thickness in px
CELL = 8                # halftone dot spacing in px (bigger = chunkier dots)
DARK, LIGHT = (26, 26, 46), (255, 209, 102)  # use (0,0,0), (255,255,255) for plain B/W
#DARK = tuple(int(c) for c in np.random.randint(0, 256, 3))
#LIGHT = tuple(int(c) for c in np.random.randint(0, 256, 3))
print(DARK, LIGHT)
NAME = os.path.splitext(os.path.basename(SRC))[0]   # P1011864

os.makedirs("looks", exist_ok=True)
W3 = np.array([0.299, 0.587, 0.114])

im = np.array(Image.open(SRC).convert("RGB"))
H, W = im.shape[:2]
lum = im @ W3
out = im.copy()
sorted_mask = np.zeros((H, W), dtype=bool)

# banded pixel sort
for y0 in range(0, H, BAND):
    row = lum[y0:y0+BAND].mean(axis=0)
    mask = ((row > LO) & (row < HI)).astype(np.int8)
    mask = mask & (np.random.rand(row.size) > BREAK).astype(np.int8)
    idx = np.flatnonzero(np.diff(np.concatenate(([0], mask, [0]))))
    for s, e in zip(idx[::2], idx[1::2]):
        order = np.argsort(row[s:e])
        out[y0:y0+BAND, s:e] = im[y0:y0+BAND, s:e][:, order]

#Image.fromarray(out).save(f"looks/{NAME}-sort.png")
CHANGE = 40  # how different a pixel must be from the original to count as sorted
sorted_mask = np.abs(out.astype(int) - im.astype(int)).sum(axis=2) > CHANGE

# halftone threshold map: dots grow outward from the center of each cell
yy, xx = np.mgrid[0:CELL, 0:CELL]
d = np.hypot(xx - (CELL - 1) / 2, yy - (CELL - 1) / 2)
rank = d.ravel().argsort().argsort().reshape(CELL, CELL)
thr = 1 - (rank + 0.5) / CELL**2
tile = np.tile(thr, (H // CELL + 1, W // CELL + 1))[:H, :W]

gray = (out @ W3) / 255
dith = np.where((gray > tile)[..., None], LIGHT, DARK).astype(np.uint8)

# dithered only where sorted, original color elsewhere
result = np.where(sorted_mask[..., None], dith, out)
Image.fromarray(result).save(f"looks/{NAME}.png")

# optional: whole image dithered
#Image.fromarray(dith).save(f"looks/{NAME}-full.png")
