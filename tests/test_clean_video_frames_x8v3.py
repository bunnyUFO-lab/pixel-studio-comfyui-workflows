"""The x8-v3 clean-up (MiniMax H3, round 6: 9.83 against 9.0 for x8-v1 on the same video frames): CleanVideoFrames with the
x8-v1 settings plus follow_hold and feet_band 8 must reproduce the research clean-up (h3match/_tools/clean.py --preset
x8-v3) byte for byte on two clips: dwarf smash (a yellow impact that must stay) and armour slash (a red splat at the feet
that must go).

    python tests/test_clean_video_frames_x8v3.py      (no ComfyUI or GPU needed)

Fixtures (tests/fixtures/h3_x8v3_dwarf_smash, h3_x8v3_armour_slash): v00..v12 are the 13 picked raw 768 x 768 video frames
(lossless WebP: the hold needs the per-channel median of every 8 x 8 block, so they are not reduced), sprite.png the input
sprite, expected.json the SHA-256 (first 16 hex) of each expected frame's RGBA bytes after the hold alone and after both
passes. The hash covers all four bytes, so the colour stored under transparent pixels counts: the hold compares it.
"""
import hashlib
import json
import os
import sys
import time
import types

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "custom_nodes", "comfyui_workflows_utils"))
sys.modules.setdefault("folder_paths", types.ModuleType("folder_paths"))
import pixel as px  # noqa: E402

RESULTS = []


def check(cond, msg):
    RESULTS.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + msg)


def digest(frames):
    return [hashlib.sha256(np.ascontiguousarray(f).tobytes()).hexdigest()[:16] for f in frames]


def vis(a):
    a = np.array(a)
    a[a[..., 3] == 0] = 0
    return a


X8 = dict(pick="centre", min_hole=4, edge_trim=1, effect_min=3)
STD = (60.0, 4, 40.0, False, True, True)
timings = []

for tag in ("dwarf_smash", "armour_slash"):
    fx = os.path.join(ROOT, "tests", "fixtures", f"h3_x8v3_{tag}")
    exp = json.load(open(os.path.join(fx, "expected.json")))
    sprite = np.array(Image.open(os.path.join(fx, "sprite.png")).convert("RGBA"))
    t = torch.from_numpy(sprite.astype(np.float32)[None] / 255.0)
    _, spec, _, _, _ = px.PrepareSpriteForVideo().run(t, 768, 0.0, "auto", 96, "whole canvas", 8)
    spec["bg"] = tuple(exp["key_colour"])
    frames = [np.array(Image.open(os.path.join(fx, f"v{j:02d}.webp")).convert("RGB")) for j in range(13)]
    want_hold = [exp["after_hold"][f"c{j:02d}.png"] for j in range(13)]
    want_out = [exp["x8v3_out"][f"c{j:02d}.png"] for j in range(13)]

    base = px.clean_video_frames(frames, spec, *STD, **X8)
    held = px.clean_video_frames(frames, spec, *STD, **X8, follow_hold_on=True)
    check(digest(held) == want_hold, f"{tag}: the hold alone matches the research frames byte for byte (13 of 13)")
    t0 = time.perf_counter()
    full = px.clean_video_frames(frames, spec, *STD, **X8, follow_hold_on=True, feet_band=8)
    timings.append(time.perf_counter() - t0)
    check(digest(full) == want_out, f"{tag}: hold + feet band match the research x8-v3 frames byte for byte (13 of 13)")

    # alpha is never changed by the hold; the sprite frame is never changed by either pass
    check(all(np.array_equal(a[..., 3], b[..., 3]) for a, b in zip(base, held)), f"{tag}: the hold leaves every alpha value alone")
    check(np.array_equal(full[0], sprite), f"{tag}: frame 1 stays the input sprite")
    check(any(not np.array_equal(vis(a), vis(b)) for a, b in zip(base, held)), f"{tag}: the hold changes some colours")

    # both off = x8-v1, exactly, also through the node with its input names
    off = px.clean_video_frames(frames, spec, *STD, **X8, follow_hold_on=False, feet_band=0)
    check(all(np.array_equal(a, b) for a, b in zip(base, off)), f"{tag}: with both inputs off the output is x8-v1, unchanged")
    nd, _, _ = px.CleanVideoFrames().run(torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0), spec, *STD, **X8,
                                         follow_hold=True, feet_band=8)
    nd = (nd.numpy() * 255).round().astype(np.uint8)
    check(all(np.array_equal(a, b) for a, b in zip(nd, full)), f"{tag}: the node with follow_hold and feet_band 8 gives the same 13 frames")

    if tag == "armour_slash":
        gone = int(sum(((a[..., 3] > 0) & (b[..., 3] == 0)).sum() for a, b in zip(held, full)))
        check(gone == 131, f"armour slash: the feet band removes the 131 pixel splat (removed {gone})")
    else:
        check(all(np.array_equal(a, b) for a, b in zip(held, full)), "dwarf smash: the feet band removes nothing (the impact stays)")

# the feet band on its own, on a small synthetic clip: a red group touching the bottom rows goes, one high up stays,
# and one low group sharing its colours with the high one (a trail) stays
sp = np.zeros((20, 12, 4), np.uint8)
sp[4:16, 4:8] = (30, 30, 90, 255)
fr = [sp.copy() for _ in range(3)]
fr[1][5, 1] = (250, 0, 0, 255)                # high group (colour seen above the band)
fr[1][17, 10] = (250, 0, 0, 255)              # low group, same colour: a trail, stays
fr[2][17, 1] = (0, 250, 0, 255)               # low group, colour nowhere else: a splat, goes
fr[2][16, 2] = (0, 250, 0, 255)
res = px.feet_band_rule(fr, 8)
check(res[1][5, 1, 3] == 255 and res[1][17, 10, 3] == 255 and res[2][17, 1, 3] == 0 and res[2][16, 2, 3] == 0,
      "feet band: a splat at the feet goes, effect colours also seen higher up stay")
check(np.array_equal(px.feet_band_rule(fr, 8)[0], sp), "feet band: frame 1 is untouched")

print(f"x8-v3 post-passes with x8-v1 included, 13 frames of 96 px: {', '.join(f'{x:.2f} s' for x in timings)}")
check(max(timings) < 5.0, "the whole clean-up with both passes takes under 5 s on the CPU")
print(f"\n{sum(RESULTS)}/{len(RESULTS)} passed")
sys.exit(0 if all(RESULTS) else 1)
