"""Re-pixelating a clip whose figure gets paler towards the end (Wan lifts the dark colours and lowers the contrast in
its last frames): with hold_colors the last frame must come out in the first frame's colours, so a loop does not flash.

    python tests/test_hold_colors.py      (no ComfyUI or GPU needed)
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
# a dark sprite with close shades, as in the game: near-black, two dark purples, a mid grey and one light accent
PAL = np.array([[14, 10, 20], [40, 30, 60], [66, 50, 92], [110, 108, 122], [214, 208, 226]], np.uint8)
KEY = np.array([0, 255, 0], np.uint8)
spec = {"factor": F, "canvas": (CW, CH), "offset": (6, 6), "size": (12, 20), "palette": PAL.tolist(),
        "counts": [10, 10, 10, 10, 10], "bg": KEY.tolist(), "detected_input_scale": 1}


def frame(dx=0, wash=0.0, seed=0):
    """A model frame: the figure on the flat key colour, moved dx art pixels, its tones washed out by `wash`
    (0 = as drawn; 0.25 = dark colours lifted and contrast down, like Wan's last frames)."""
    img = np.zeros((CH * F, CW * F, 3), np.float64)
    img[...] = KEY
    fig = np.zeros(img.shape[:2], bool)
    for (y0, y1, x0, x1), c in (((6, 14, 8, 16), 1), ((14, 26, 9, 15), 0), ((8, 12, 10, 14), 4), ((14, 20, 14, 18), 2),
                                ((20, 26, 6, 9), 3)):
        img[y0 * F:y1 * F, (x0 + dx) * F:(x1 + dx) * F] = PAL[c]
        fig[y0 * F:y1 * F, (x0 + dx) * F:(x1 + dx) * F] = True
    img[fig] = img[fig] * (1 - wash) + 150 * wash
    noise = np.random.default_rng(seed).integers(-3, 4, img.shape)                # video compression noise
    return (img + noise).clip(0, 255).astype(np.uint8)


def lum(rgba):
    m = rgba[..., 3] > 0
    return float((rgba[m][:, :3].astype(float) @ [0.299, 0.587, 0.114]).mean())


KW = dict(stability=0.7, deflicker=True, cleanup=False)
# a standing clip that gets paler in its last third
clip = [frame(wash=w, seed=i) for i, w in enumerate([0, 0, 0, 0, 0, 0, 0.1, 0.25, 0.25])]
plain = px.repixelate(clip, spec, **KW)
held = px.repixelate(clip, spec, hold_colors=True, **KW)
check(lum(plain[-1]) - lum(plain[0]) > 3, f"without the option the last frame is paler ({lum(plain[-1]) - lum(plain[0]):.1f} grey levels)")
check((held[-1] == held[0]).all(), "with hold_colors the last frame equals the first: the loop closes with no flash")
check(all((h[..., 3] == p[..., 3]).all() for h, p in zip(held, plain)), "the silhouette is the same with and without the option")
check(set(map(tuple, held[-1][held[-1][..., 3] > 0][:, :3])) <= set(map(tuple, PAL)), "only the sprite's own colours come out")

# off by default, and a clip that does not drift comes out the same either way
steady = [frame(seed=i) for i in range(6)]
check(all((a == b).all() for a, b in zip(px.repixelate(steady, spec, **KW), px.repixelate(steady, spec, hold_colors=True, **KW))),
      "a clip with steady colours is not changed")
check(all((a == b).all() for a, b in zip(px.repixelate(clip, spec, **KW), plain)), "the default is off")

# the figure moves across the frame while it gets paler: colours still hold, and it still moves
walk = [frame(dx=d, wash=w, seed=i) for i, (d, w) in enumerate([(0, 0), (1, 0), (2, 0.1), (3, 0.25), (4, 0.25)])]
wp, wh = px.repixelate(walk, spec, **KW), px.repixelate(walk, spec, hold_colors=True, **KW)
check(abs(lum(wh[-1]) - lum(wh[0])) < 1.0 < lum(wp[-1]) - lum(wp[0]), "a moving figure keeps its brightness too")
check((np.roll(wh[0], 4, axis=1) == wh[-1]).all(), "the moved figure is the first frame's figure, 4 pixels over")

# edge cases: one frame, a frame with no figure, a clip whose first frame has no figure
one = px._rekey_background(clip[0], KEY, 18.0)
check(px.hold_first_colours([one], KEY)[0] is one, "a single frame is returned as it is")
empty = np.tile(KEY, (CH * F, CW * F, 1)).astype(np.uint8)
out = px.hold_first_colours([clip[0], empty, clip[-1]], KEY)
check((out[1] == empty).all(), "a frame with no figure in it is left alone")
check(all(a is b for a, b in zip(px.hold_first_colours([empty, clip[0]], KEY), [empty, clip[0]])), "nothing to hold to when the first frame is empty")
check((out[2][..., :3][(out[2][..., :3] == KEY).all(-1)] == KEY).all() and (out[0] == clip[0]).all(), "the background and the first frame are untouched")

# the node passes the option through
import torch  # noqa: E402

node_in = torch.from_numpy(np.stack(clip).astype(np.float32) / 255.0)
frames_t, info, _ = px.RepixelateFrames().run(node_in, spec, False, 18.0, "dominant", 0, "sprite", 0.7, True, False, "none", hold_colors=True)
check("colours held to the first frame" in info and torch.equal(frames_t[-1], frames_t[0]), "RepixelateFrames takes hold_colors and reports it")
old_call = px.RepixelateFrames().run(node_in, spec, False, 18.0, "dominant", 0, "sprite", 0.7, True, False, "none")
check("held" not in old_call[1], "a workflow saved before the option existed runs as before")

print(f"{sum(RESULTS)}/{len(RESULTS)} passed")
sys.exit(0 if all(RESULTS) else 1)
