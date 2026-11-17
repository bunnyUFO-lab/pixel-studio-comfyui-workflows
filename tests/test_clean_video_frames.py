"""CPU tests for the MiniMax H3 sprite chain: PrepareSpriteForVideo in whole-canvas mode, CleanVideoFrames and the
neutral background pick.

    python tests/test_clean_video_frames.py      (no ComfyUI or GPU needed)
"""
import os
import sys
import types

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "custom_nodes", "comfyui_workflows_utils"))
sys.modules.setdefault("folder_paths", types.ModuleType("folder_paths"))
import pixel as px  # noqa: E402

RESULTS = []


def check(cond, msg):
    RESULTS.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + msg)


# ---------------------------------------------------------------- a small dark sprite on a 32 x 32 canvas
PAL = np.array([[18, 12, 24], [60, 40, 90], [120, 80, 60], [200, 160, 120], [240, 230, 210]], np.uint8)
W = H = 32


def make_sprite():
    s = np.zeros((H, W, 4), np.uint8)
    s[8:26, 10:20] = (*PAL[1], 255)                         # body
    s[10:14, 11:19] = (*PAL[3], 255)                        # head
    s[11, 13] = (*PAL[4], 255)                              # an eye
    s[18:22, 12:18] = (*PAL[2], 255)                        # belt
    a = s[..., 3] > 0                                       # a one-colour outline on the silhouette edge
    pad = np.pad(a, 1)
    edge = a & (~pad[:-2, 1:-1] | ~pad[2:, 1:-1] | ~pad[1:-1, :-2] | ~pad[1:-1, 2:])
    s[edge] = (*PAL[0], 255)
    return s


sprite = make_sprite()
t = torch.from_numpy(sprite.astype(np.float32)[None] / 255.0)
img, spec, w, h, info = px.PrepareSpriteForVideo().run(t, 128, 0.0, "auto", 32, "whole canvas", 4)
check((w, h) == (128, 128) and spec["canvas"] == (32, 32) and spec["factor"] == 4,
      f"whole canvas x4: 32 -> 128, nothing cropped, no padding ({info})")
check(spec["offset"] == (10, 8) and spec["size"] == (10, 18), "the sprite stays where it is on its canvas")
check(tuple(spec["bg"]) in {(255, 255, 255), (224, 224, 224), (200, 200, 200), (160, 160, 160), (128, 128, 128)},
      f"the background is a neutral grey or white ({tuple(spec['bg'])}), never a magenta or green key")
check(spec["sprite"].shape == (32, 32, 4) and (spec["sprite"] == np.where(sprite[..., 3:] > 0, sprite, 0)).all(),
      "the spec carries the sprite alone on its canvas, for the clean-up")
big = (img[0].numpy() * 255).round().astype(np.uint8)
check((big[::4, ::4][sprite[..., 3] > 0] == sprite[..., :3][sprite[..., 3] > 0]).all()
      and (big[::4, ::4][sprite[..., 3] == 0] == spec["bg"]).all(), "the enlargement is exact (nearest neighbour) on the key colour")
_, spec_f, w2, h2, _ = px.PrepareSpriteForVideo().run(t, 200, 0.0, "auto", 32, "whole canvas", 0)
check(spec_f["factor"] == 6 and (w2, h2) == (192, 192), "factor 0 is chosen from target_size")
_, spec_c, w3, h3, _ = px.PrepareSpriteForVideo().run(t, 640, 0.35, "auto", 16)
check(spec_c["canvas"] != (32, 32) and "sprite" in spec_c, "the default crop + padding mode is unchanged (and still reports the sprite)")
odd = np.zeros((30, 30, 4), np.uint8)
odd[5:20, 5:15] = (*PAL[1], 255)
odd[5:20:2, 6:15:3] = (*PAL[3], 255)       # texture, so it is not mistaken for an upscaled sprite
odd[7, 8] = (*PAL[4], 255)
_, spec_o, w4, h4, _ = px.PrepareSpriteForVideo().run(torch.from_numpy(odd.astype(np.float32)[None] / 255.0), 120, 0.0, "auto", 32, "whole canvas", 4)
check(w4 % 32 == 0 and h4 % 32 == 0 and spec_o["canvas"][0] >= 30, "a canvas that does not divide by 32 grows on the right and bottom")
dark = np.array([[10, 10, 10], [30, 30, 30]], np.uint8)
light = np.array([[250, 250, 250], [230, 230, 230]], np.uint8)
check(tuple(px.pick_neutral_background(dark)) == (255, 255, 255) and tuple(px.pick_neutral_background(light)) == (128, 128, 128),
      "dark sprites get white, light sprites get mid grey")

# ---------------------------------------------------------------- video frames of that sprite
K = 4
bg = np.array(spec["bg"], np.int32)
rng = np.random.default_rng(1)


def video_frame(dx=0, noise=3, effect=None, speck=None, seam=True):
    """The enlarged sprite (moved dx art pixels) on the key colour, with compression noise and blurred seams."""
    a = np.roll(sprite, dx, axis=1)
    rgb = np.empty((H, W, 3), np.int32)
    rgb[...] = bg
    m = a[..., 3] > 0
    rgb[m] = a[..., :3][m]
    fr = np.repeat(np.repeat(rgb, K, 0), K, 1).astype(np.float64)
    if seam:                                                # a soft 1 px blend at every colour change
        for ax in (0, 1):
            d = np.abs(np.diff(fr, axis=ax)).sum(-1) > 0
            idx = np.argwhere(d)
            for y, x in idx:
                yy, xx = (y + 1, x) if ax == 0 else (y, x + 1)
                mid = (fr[y, x] + fr[yy, xx]) / 2
                fr[y, x] = mid
    if effect is not None:                                  # a bright 2 x 2 art-pixel spark in a colour the sprite does not have
        y, x = effect
        fr[y * K:(y + 2) * K, x * K:(x + 2) * K] = (255, 40, 40)
    if speck is not None:                                   # one art pixel of the wrong sprite colour inside the body
        y, x = speck
        fr[y * K:(y + 1) * K, x * K:(x + 1) * K] = PAL[3]
    fr += rng.integers(-noise, noise + 1, fr.shape)
    return fr.clip(0, 255).astype(np.uint8)


clip = [video_frame(0, effect=None), video_frame(1, effect=(4, 24)), video_frame(2, speck=(15, 15)), video_frame(2)]
frames_t = torch.from_numpy(np.stack(clip).astype(np.float32) / 255.0)
out_t, cinfo, pal_text = px.CleanVideoFrames().run(frames_t, spec, 60.0, 4, 40.0, True, True, True)
out = (out_t.numpy() * 255).round().astype(np.uint8)
check(out.shape == (4, 32, 32, 4), f"frames come back on the sprite's own grid, RGBA ({cinfo})")
check((out[0] == spec["sprite"]).all(), "frame 1 is the input sprite itself")
body_pal = {tuple(c) for c in PAL}
for j in (1, 2, 3):
    op = out[j][..., 3] > 0
    cols = {tuple(c) for c in out[j][op][:, :3]}
    extra = cols - body_pal
    check(len(extra) <= 4 and (j != 1 or len(extra) >= 1), f"frame {j + 1}: only sprite colours plus at most 4 effect colours ({len(extra)} new)")
check(out[1][4:6, 24:26, 3].min() == 255 and tuple(out[1][4, 24, :3]) not in body_pal, "the spark is kept as an effect colour, not snapped into the body")
check(((out[1][..., 3] > 0) == (np.roll(sprite, 1, axis=1)[..., 3] > 0)).sum() >= 32 * 32 - 4 * 2 - 12, "the silhouette follows the figure (key blocks transparent)")
check(tuple(out[2][15, 15, :3]) != tuple(PAL[3]) and tuple(out[2][15, 15, :3]) in body_pal, "a wrong-coloured pixel inside the body takes its neighbours' colour")
check((out[3][..., 3][:2] == 0).all() and (out[3][..., 3][:, :2] == 0).all(), "the background is transparent")
noeff, _, _ = px.CleanVideoFrames().run(frames_t, spec, 60.0, 0, 40.0, True, True, True)
check({tuple(c) for c in (noeff.numpy()[1] * 255).round().astype(np.uint8)[..., :3].reshape(-1, 3)[(noeff.numpy()[1][..., 3] > 0).reshape(-1)]} <= body_pal,
      "effect_colours 0: everything snaps to the sprite's palette")
edge = px._edge4(out[3][..., 3] > 0)
check((out[3][edge][:, :3] == PAL[0]).all(), "the sprite's one-colour outline is redrawn on the silhouette")
noline, _, _ = px.CleanVideoFrames().run(frames_t, spec, 60.0, 4, 40.0, True, False, True)
check(noline.shape == frames_t.shape[:1] + (32, 32, 4), "the outline can be switched off")
nofirst, _, _ = px.CleanVideoFrames().run(frames_t, spec, 60.0, 4, 40.0, True, True, False)
check(not np.array_equal(nofirst.numpy()[0], (spec["sprite"].astype(np.float32) / 255.0)), "first_is_sprite off: frame 1 is cleaned from the video like the others")

# the clip is steady: the same clean-up on two noisy copies of one frame gives the same pixels
two = [video_frame(0, noise=2), video_frame(0, noise=2)]
a = px.clean_video_frames(two, spec)
b = px.clean_video_frames(two, spec, first_is_sprite=False)
check((a[1] == b[1]).all() and (a[0] == spec["sprite"]).all(), "per-frame clean-up: only frame 1 depends on the first_is_sprite option")
check(px.CleanVideoFrames.INPUT_TYPES()["required"]["effect_colours"][1]["default"] == 4, "defaults of the recipe: 4 effect colours, tolerance 60")

# ---------------------------------------------------------------- the frame pick of the recipe
recipe = [round(i * 38 / 12) for i in range(13)]
check(px.PickEvenFrames.indices(39, 13, False) == recipe == [0, 3, 6, 10, 13, 16, 19, 22, 25, 28, 32, 35, 38], "13 of 39 frames: round(i * 38 / 12), the first one the input sprite")

print(f"{sum(RESULTS)}/{len(RESULTS)} passed")
sys.exit(0 if all(RESULTS) else 1)
