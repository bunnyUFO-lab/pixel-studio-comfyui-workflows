"""Re-pixelating a motion clip whose background drifts (the video model was given flat green and turned it teal, blue,
purple): every frame's background must still come out transparent, not as a solid block of sprite colours.

    python tests/test_rekey_background.py      (no ComfyUI or GPU needed)
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "custom_nodes", "comfyui_workflows_utils"))
import pixel as px  # noqa: E402

RESULTS = []


def check(cond, msg):
    RESULTS.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + msg)


F, CW, CH = 4, 24, 32                                   # 24x32 art px canvas, x4 for the video model
PAL = np.array([[230, 225, 210], [40, 30, 60], [120, 60, 40], [90, 90, 100]], np.uint8)   # cream, navy, brown, grey
KEY = np.array([0, 255, 0], np.uint8)
spec = {"factor": F, "canvas": (CW, CH), "offset": (6, 6), "size": (12, 20), "palette": PAL.tolist(),
        "counts": [10, 10, 10, 10], "bg": KEY.tolist(), "detected_input_scale": 1}


def frame(bg):
    """A model frame: a flat background in colour bg, the figure (cream body, navy legs, brown shield) in the middle."""
    img = np.zeros((CH * F, CW * F, 3), np.uint8)
    img[...] = bg
    img[6 * F:16 * F, 8 * F:16 * F] = PAL[0]
    img[16 * F:26 * F, 9 * F:15 * F] = PAL[1]
    img[10 * F:18 * F, 14 * F:19 * F] = PAL[2]
    noise = np.random.default_rng(len(str(bg))).integers(-3, 4, img.shape)        # video compression noise
    return (img.astype(np.int16) + noise).clip(0, 255).astype(np.uint8)


DRIFT = [(0, 255, 0), (40, 220, 120), (90, 130, 170), (95, 90, 170), (60, 40, 120), (200, 196, 180)]
out = px.repixelate([frame(np.array(c, np.uint8)) for c in DRIFT], spec, stability=0.7)
for c, o in zip(DRIFT, out):
    a = o[..., 3]
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    check((border == 0).mean() > 0.99, f"background {c}: the border comes out transparent ({100 * (border == 0).mean():.0f}%)")
    body = o[10, 11]
    check(a[10, 11] == 255 and tuple(body[:3]) == tuple(PAL[0]), f"background {c}: the figure is kept, in its own colours")

# a frame with no flat border (the figure fills the frame edge to edge) is not treated as background
full = np.tile(PAL[0], (CH * F, CW * F, 1))
check((px._rekey_background(full, KEY, 18.0) == full).all(), "a frame without a flat background is left alone")
# a frame whose border is already the key colour is left exactly as it was
clean = frame(KEY)
check((px._rekey_background(clean, KEY, 18.0) == clean).all(), "a frame still on the key colour is untouched")

print(f"{sum(RESULTS)}/{len(RESULTS)} passed")
sys.exit(0 if all(RESULTS) else 1)
