"""Holding a clip's figure at one height, ground line and centre on the raw video frames, before re-pixelate.

    python tests/test_hold_figure.py      (no ComfyUI or GPU needed)
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "custom_nodes", "comfyui_workflows_utils"))
import pixel as px  # noqa: E402

RESULTS = []


def check(cond, msg):
    RESULTS.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + msg)


W, H = 160, 200
KEY = np.array([0, 255, 0], np.uint8)
FIG = np.array([150, 90, 60], np.uint8)


def frame(h=100, w=40, cx=80, bottom=180, noise_seed=None):
    """A green-screen frame: a figure `h` tall and `w` wide, its feet at `bottom`, centred on `cx`."""
    img = np.zeros((H, W, 3), np.uint8)
    img[...] = KEY
    x0 = cx - w // 2
    img[bottom - h + 1:bottom + 1, x0:x0 + w] = FIG
    img[bottom - h + 1:bottom - h + 11, x0 + 10:x0 + 30] = (240, 220, 200)          # a head, so it is not one flat block
    if noise_seed is not None:                                                    # video compression noise
        n = np.random.default_rng(noise_seed).integers(-4, 5, img.shape)
        img = (img.astype(int) + n).clip(0, 255).astype(np.uint8)
    return img


def box(fr):
    return px._figure_box(fr, KEY, 60)


def height(fr):
    b = box(fr)
    return b[3] - b[1] + 1


# a walk that grows toward the viewer and drifts sideways: height 80 -> 130, feet 170 -> 190, centre 60 -> 100
hs = [80, 90, 100, 110, 120, 130]
clip = [frame(h=h, bottom=170 + 4 * i, cx=60 + 8 * i, noise_seed=i) for i, h in enumerate(hs)]
before = px.figure_swing(clip, KEY)
held = px.hold_figure(clip, KEY, strength=1.0)
after = px.figure_swing(held, KEY)
check(before > 50, f"the test clip grows by {before:.0f}%")
check(after < 8, f"with strength 1 the height swing falls to {after:.0f}%")
bottoms = [box(f)[3] for f in held]
cxs = [(box(f)[0] + box(f)[2]) / 2 for f in held]
check(max(bottoms) - min(bottoms) <= 2, f"the ground line is held (spread {max(bottoms) - min(bottoms)} px)")
check(max(cxs) - min(cxs) <= 3, f"the horizontal centre is held (spread {max(cxs) - min(cxs):.1f} px)")

# the size correction is partial by default: a crouch stays shorter than a lunge, but the swing still drops a lot
part = px.hold_figure(clip, KEY, strength=0.7)
check(px.figure_swing(part, KEY) < before * 0.5 and px.figure_swing(part, KEY) > after, "strength 0.7 corrects part of the swing and keeps some")
none = px.hold_figure(clip, KEY, strength=0.0)
check(max(box(f)[3] for f in none) - min(box(f)[3] for f in none) <= 2 and abs(px.figure_swing(none, KEY) - before) < 3,
      "strength 0 holds position only and leaves the size alone")

# nothing is invented: only the figure's colours and the background come out (the key colour fills what a shift uncovers)
PAL = np.array([KEY, FIG, (240, 220, 200)], np.int32)
solid = [frame(h=h, bottom=170 + 4 * i, cx=60 + 8 * i) for i, h in enumerate(hs)]
got = px.hold_figure(solid, KEY, strength=1.0)[0]
near = (np.abs(got.reshape(-1, 1, 3).astype(np.int32) - PAL[None]).sum(-1).min(1) <= 60).mean()
check(near > 0.99, f"only the figure's own colours and the background appear ({near:.1%} of pixels; scaling blends the edges)")
check(all(f.shape == (H, W, 3) for f in held), "frame size is unchanged")

# the background colour can be read from the frame border when no spec is given
magenta = [np.where((f == KEY).all(-1, keepdims=True), np.array([255, 0, 255], np.uint8), f) for f in solid]
auto = px.hold_figure(magenta, None, strength=1.0)
check(px.figure_swing(auto, np.array([255, 0, 255])) < 8, "the border colour is found on its own")
check(np.abs(auto[-1][0, 0].astype(int) - (255, 0, 255)).max() <= 8, "what a shift uncovers gets that border colour")

# edge cases: one frame, a frame with no figure, a clip with no figure
empty = np.tile(KEY, (H, W, 1)).astype(np.uint8)
one = [clip[0]]
check(px.hold_figure(one, KEY)[0] is one[0], "a single frame is returned as it is")
mixed = px.hold_figure([clip[0], empty, clip[-1]], KEY, strength=1.0)
check((mixed[1] == empty).all(), "a frame with no figure is left alone")
check(all(f is e for f, e in zip(px.hold_figure([empty, empty], KEY), [empty, empty])), "a clip with no figure is returned as it is")
steady = [frame(noise_seed=i) for i in range(5)]
check(all(np.array_equal(a, b) for a, b in zip(px.hold_figure(steady, KEY, strength=1.0), steady)), "a clip that is already steady is not changed")

# the node: off by default, on changes the frames, takes the spec's background
node_in = torch.from_numpy(np.stack(clip).astype(np.float32) / 255.0)
spec = {"bg": KEY.tolist()}
check(px.HoldFigure.INPUT_TYPES()["required"]["enabled"][1]["default"] is False, "the node is off by default")
t_off, info_off = px.HoldFigure().run(node_in, False, 0.7, spec)
check(t_off is node_in and "off" in info_off, "off: frames pass through untouched")
t_on, info_on = px.HoldFigure().run(node_in, True, 1.0, spec)
check(t_on.shape == node_in.shape and not torch.equal(t_on, node_in), "on: the frames are changed, same shape")
check("%" in info_on and "->" in info_on, f"on: it reports the swing ({info_on})")
t_auto, _ = px.HoldFigure().run(node_in, True, 1.0)
check(t_auto.shape == node_in.shape, "the pixel_spec input is optional")

print(f"{sum(RESULTS)}/{len(RESULTS)} passed")
sys.exit(0 if all(RESULTS) else 1)
