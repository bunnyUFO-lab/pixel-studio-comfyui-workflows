"""MakeTileable axis="x": a strip that joins sideways and must NOT join top to bottom.

Run: python tests/test_tileable_x.py   (no ComfyUI, no GPU; needs torch for the pixel module)

A parallax layer repeats left to right while its sky stays at the top and its ground at the bottom. The square
tile this node used to make would fold one into the other, so the vertical wrap not happening is as much the
point as the horizontal one happening.
"""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "custom_nodes" / "comfyui_workflows_utils"))
sys.modules.setdefault("folder_paths", types.ModuleType("folder_paths"))

import numpy as np  # noqa: E402
import pixel  # noqa: E402

RESULTS = []


def check(cond, msg):
    RESULTS.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + msg)


def layer(h=180, w=900, seed=0):
    """A strip shaped like a real layer: a sky gradient on top, a dark jagged ground along the bottom. The point is
    that the top and bottom are utterly unlike each other, so a vertical wrap would be obvious."""
    rng = np.random.default_rng(seed)
    a = np.zeros((h, w, 4), np.uint8)
    a[..., 3] = 255
    for y in range(h):
        a[y, :, :3] = (60 + y // 3, 50 + y // 4, 120 - y // 6)        # sky, light at the top
    sky = h // 2
    for x in range(w):
        top = sky + int(20 * np.sin(x / 37.0) + 8 * np.sin(x / 7.0) + rng.integers(0, 3))
        a[top:, x, :3] = (20, 60, 30)                                  # ground, dark green
    return a


def seam_cost(col_a, col_b):
    """Mean Lab distance between two columns (or rows) of pixels."""
    return float(np.sqrt(((pixel.rgb_to_lab(col_a[..., :3]) - pixel.rgb_to_lab(col_b[..., :3])) ** 2).sum(-1)).mean())


src = layer()
W, H = 640, 180
strip = pixel.make_tileable(src, W, height=H, axis="x")
check(strip.shape == (H, W, 4), "axis=x gives the width and height asked for, not a square")

# sideways: the strip's right edge must meet its own left edge as well as two neighbouring columns inside it do
inner = np.mean([seam_cost(strip[:, i], strip[:, i + 1]) for i in range(0, W - 1, 37)])
wrap_x = seam_cost(strip[:, -1], strip[:, 0])
check(wrap_x <= inner * 3 + 2,
      "the left and right edges join: wrap cost %.1f vs %.1f inside the strip" % (wrap_x, inner))

# vertically: it must NOT join. Sky over ground is the whole point of a layer.
wrap_y = seam_cost(strip[-1, :], strip[0, :])
check(wrap_y > inner * 5,
      "the top and bottom do NOT join, as a layer must not: cost %.1f" % wrap_y)

# and the pixels are the texture's own, not blended
pal_src = {tuple(c) for c in src.reshape(-1, 4)}
pal_out = {tuple(c) for c in strip.reshape(-1, 4)}
check(pal_out <= pal_src, "every pixel came from the source texture (no blending)")

# the old behaviour is untouched
tile = pixel.make_tileable(layer(300, 300, 1), 64)
check(tile.shape == (64, 64, 4), "axis=both still gives a square tile")
t_inner = np.mean([seam_cost(tile[:, i], tile[:, i + 1]) for i in range(0, 63, 7)])
check(seam_cost(tile[:, -1], tile[:, 0]) <= t_inner * 3 + 2 and seam_cost(tile[-1, :], tile[0, :]) <= t_inner * 3 + 2,
      "axis=both still wraps BOTH ways")

# sizes a layer actually needs, which the old 512 cap refused
big = pixel.make_tileable(layer(360, 1800, 2), 1280, height=360, axis="x")
check(big.shape == (360, 1280, 4), "1280x360 works: the 640x360 screen's layer, which the 512 cap used to refuse")

# a texture too small to cut from says so rather than failing obscurely
try:
    pixel.make_tileable(layer(180, 300), 640, height=180, axis="x")
    check(False, "too-small texture raises")
except ValueError as e:
    check("too small" in str(e) and "640" in str(e), "a texture too small to cut from says so, with the sizes")

print()
print("%d/%d passed" % (sum(RESULTS), len(RESULTS)))
sys.exit(0 if all(RESULTS) else 1)
