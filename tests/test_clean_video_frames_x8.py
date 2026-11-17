"""The x8-v1 clean-up (MiniMax H3 at whole-canvas x8): CleanVideoFrames with pick centre, despeckle off, min_hole 4,
edge_trim 1, effect_min 3 must reproduce the research clean-up (h3match/_tools/clean.py --preset x8-v1) pixel for pixel on
the dwarf-smash clip: 13 frames, 96 x 96 RGBA.

    python tests/test_clean_video_frames_x8.py      (no ComfyUI or GPU needed)

Fixture (tests/fixtures/h3_x8_dwarf_smash): v00..v12 are the 13 picked 768 x 768 video frames, sprite.png the input sprite,
c00..c12 the expected output. The research script leaves a colour in fully transparent pixels and the node writes 0
there, so those are zeroed before comparing: every visible pixel and the alpha channel are compared. To keep the fixture small the video frames are reduced without changing the result: a block
that is background by majority is flat key colour, and in a body block every pixel except the centre one and the one
farthest from the key is replaced by a copy of the one of the same background/body class (the clean-up reads only those
two and the block's background share). The reduction was checked against the full raw frames.
"""
import os
import sys
import types

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "custom_nodes", "comfyui_workflows_utils"))
sys.modules.setdefault("folder_paths", types.ModuleType("folder_paths"))
import pixel as px  # noqa: E402

FX = os.path.join(ROOT, "tests", "fixtures", "h3_x8_dwarf_smash")
RESULTS = []


def check(cond, msg):
    RESULTS.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + msg)


def vis(a):
    """RGBA with the colour of fully transparent pixels zeroed."""
    a = np.array(a)
    a[a[..., 3] == 0] = 0
    return a


def load(name, mode):
    return np.array(Image.open(os.path.join(FX, name)).convert(mode))


sprite = load("sprite.png", "RGBA")
t = __import__("torch").from_numpy(sprite.astype(np.float32)[None] / 255.0)
_, spec, w, h, _ = px.PrepareSpriteForVideo().run(t, 768, 0.0, "auto", 96, "whole canvas", 8)
spec["bg"] = (160, 160, 160)                       # the clip was made on this key colour
check(spec["factor"] == 8 and spec["canvas"] == (96, 96), "the sprite is enlarged x8 on its whole 96 x 96 canvas")
frames = [load(f"v{j:02d}.png", "RGB") for j in range(13)]
expected = [load(f"c{j:02d}.png", "RGBA") for j in range(13)]

X8 = dict(pick="centre", min_hole=4, edge_trim=1, effect_min=3)
out = px.clean_video_frames(frames, spec, 60.0, 4, 40.0, False, True, True, **X8)
same = [bool(np.array_equal(vis(o), vis(e))) for o, e in zip(out, expected)]
check(len(out) == 13 and all(same), f"x8-v1: {sum(same)} of 13 frames identical to the research clean-up, pixel for pixel")

# through the node, with its input names
import torch  # noqa: E402
node_out, _, _ = px.CleanVideoFrames().run(torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0), spec, 60.0, 4, 40.0,
                                           False, True, True, **X8)
nd = (node_out.numpy() * 255).round().astype(np.uint8)
check(all(np.array_equal(vis(nd[j]), vis(expected[j])) for j in range(13)), "the node gives the same 13 frames")

# each option does something: dropping any one of them moves at least one pixel
for name, off in (("pick", "vote"), ("min_hole", 1), ("effect_min", 1)):
    kw = dict(X8, **{name: off})
    o2 = px.clean_video_frames(frames, spec, 60.0, 4, 40.0, False, True, True, **kw)
    check(any(not np.array_equal(vis(a), vis(b)) for a, b in zip(o2, expected)), f"{name} matters: {off} changes the clip")
d_old = px.clean_video_frames(frames, spec)
check(any(not np.array_equal(vis(a), vis(b)) for a, b in zip(d_old, expected)), "the old defaults give a different clip")

# edge_trim on a small synthetic clip (this clip has no blend pixel on its edge, so it cannot show the option): a column of
# art pixels coloured between the key and the body, on the silhouette edge, is dropped by edge_trim 1 and kept without it
K = 8
syn_spec = {"factor": K, "canvas": (8, 8), "bg": (160, 160, 160), "palette": [(20, 20, 60), (200, 40, 40)], "sprite": None}
vid = np.zeros((8 * K, 8 * K, 3), np.uint8) + 160
vid[2 * K:6 * K, 2 * K:5 * K] = (20, 20, 60)                # the body, 3 x 4 art pixels
vid[2 * K:6 * K, 5 * K:6 * K] = (120, 120, 130)             # one column of edge blend right of it
SYN = dict(first_is_sprite=False, outline=False, despeckle=False)
a = px.clean_video_frames([vid, vid], syn_spec, **SYN, pick="centre")[1]
b = px.clean_video_frames([vid, vid], syn_spec, **SYN, pick="centre", edge_trim=1)[1]
check(a[3, 5, 3] == 255 and b[3, 5, 3] == 0 and b[3, 4, 3] == 255, "edge_trim 1 removes the blend column on the edge and keeps the body")
# min_hole: a 2 x 2 key-coloured region inside the body is cut at min_hole 1 and stays body above its size
vid2 = np.zeros((8 * K, 8 * K, 3), np.uint8) + 160
vid2[1 * K:7 * K, 1 * K:7 * K] = (20, 20, 60)
vid2[3 * K:5 * K, 3 * K:5 * K] = (160, 160, 160)
h1 = px.clean_video_frames([vid2, vid2], syn_spec, **SYN, min_hole=1)[1]
h5 = px.clean_video_frames([vid2, vid2], syn_spec, **SYN, min_hole=5)[1]
check(h1[3, 3, 3] == 0 and h5[3, 3, 3] == 255, "min_hole: a small enclosed key-coloured region is a hole at 1 and body above its size")

print(f"\n{sum(RESULTS)}/{len(RESULTS)} passed")
sys.exit(0 if all(RESULTS) else 1)
