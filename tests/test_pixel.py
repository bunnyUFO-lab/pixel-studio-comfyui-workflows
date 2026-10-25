"""CPU tests for custom_nodes/comfyui_workflows_utils/pixel.py (no ComfyUI or GPU needed).

Run from the repo root:  python tests/test_pixel.py
"""
import io
import os
import sys
import tempfile
import types

import numpy as np
import torch
from PIL import Image, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "custom_nodes", "comfyui_workflows_utils"))

OUT_DIR = tempfile.mkdtemp()
fp = types.ModuleType("folder_paths")              # stub for SaveSpriteSheet
fp.get_output_directory = lambda: OUT_DIR


def _save_path(prefix, out_dir, w=0, h=0):
    sub, name = os.path.split(prefix)
    full = os.path.join(out_dir, sub)
    os.makedirs(full, exist_ok=True)
    return full, name, 1, sub, prefix


fp.get_save_image_path = _save_path
fp.get_temp_directory = lambda: os.path.join(OUT_DIR, "temp")
sys.modules["folder_paths"] = fp

import pixel  # noqa: E402

RESULTS = []


def check(cond, msg):
    RESULTS.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + msg)


def make_sprite(w=24, h=32, seed=1):
    """A small character-like sprite: outline, 2-tone body, eyes, transparent background."""
    rng = np.random.default_rng(seed)
    pal = np.array([[20, 16, 28], [86, 98, 112], [140, 152, 168], [196, 160, 120], [240, 240, 232], [120, 60, 40]], np.uint8)
    img = np.zeros((h, w, 4), np.uint8)
    yy, xx = np.mgrid[:h, :w]
    body = ((xx - w / 2) ** 2 / (w * 0.36) ** 2 + (yy - h * 0.58) ** 2 / (h * 0.40) ** 2) < 1
    head = ((xx - w / 2) ** 2 + (yy - h * 0.22) ** 2) < (w * 0.26) ** 2
    shape = body | head
    from scipy import ndimage
    inner = ndimage.binary_erosion(shape)
    img[shape, :3] = pal[0]
    img[inner, :3] = pal[2]
    img[inner & (xx > w / 2), :3] = pal[1]                       # shading
    img[inner & (yy > h * 0.75), :3] = pal[5]                    # boots
    img[int(h * 0.2), int(w * 0.42)] = [*pal[4], 255]            # eyes (single pixels)
    img[int(h * 0.2), int(w * 0.58)] = [*pal[4], 255]
    img[shape, 3] = 255
    return img


def degrade(rgb, seed=0):
    """What a video model does to a clean upscaled sprite: blur, noise, JPEG."""
    im = Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(1.2))
    arr = np.asarray(im).astype(np.int16) + np.random.default_rng(seed).normal(0, 4, rgb.shape).astype(np.int16)
    buf = io.BytesIO()
    Image.fromarray(arr.clip(0, 255).astype(np.uint8)).save(buf, "JPEG", quality=80)
    return np.asarray(Image.open(buf).convert("RGB"))


def tensor(frames):
    return torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)


sprite = make_sprite()

# 1. grid detection + exact native recovery
up7 = pixel.nearest_upscale(sprite, 7)
s, ox, oy, score = pixel.detect_scale(up7)
check(s == 7, f"detect_scale on x7 upscale -> x{s} (score {score:.2f})")
check(pixel.detect_scale(sprite)[0] == 1, "detect_scale on native sprite -> x1")
native, det = pixel.recover_native(up7)
x0, y0, x1, y1 = pixel.subject_bbox(native[..., 3] > 0)
sx0, sy0, sx1, sy1 = pixel.subject_bbox(sprite[..., 3] > 0)
check(det == 7 and np.array_equal(native[y0:y1, x0:x1], sprite[sy0:sy1, sx0:sx1]), "recover_native(x7) == original sprite exactly")

# upscaled on an opaque white background (as many sprites are shared)
flat = up7.copy(); flat[flat[..., 3] == 0] = [255, 255, 255, 0]; flat[..., 3] = 255
native2, _ = pixel.recover_native(flat)
x0, y0, x1, y1 = pixel.subject_bbox(native2[..., 3] > 0)
check(np.array_equal(native2[y0:y1, x0:x1], sprite[sy0:sy1, sx0:sx1]), "recover_native(x7 on white bg) == original exactly")

# 2. scale-preserving round trip through a (simulated) video model
prep = pixel.PrepareSpriteForVideo()
img, spec, W, H, info = prep.run(tensor([up7]), 640, 0.35, "auto")
print("     ", info)
check(W % 16 == 0 and H % 16 == 0 and W == spec["canvas"][0] * spec["factor"], f"working size {W}x{H} is /16 and = canvas x factor")
(vmask,) = pixel.SpriteVideoMask().run(img, spec)
f_, (ox_, oy_), (w_, h_) = spec["factor"], spec["offset"], spec["size"]
want = np.zeros((H, W), bool)
want[oy_ * f_:(oy_ + h_) * f_, ox_ * f_:(ox_ + w_) * f_] = np.kron(sprite[sy0:sy1, sx0:sx1, 3] > 0, np.ones((f_, f_), bool))
check(vmask.shape == (1, H, W) and np.array_equal(vmask[0].numpy() > 0.5, want), "SpriteVideoMask = the upscaled sprite's alpha")
(views,) = pixel.MatchSpriteCanvas().run(tensor([up7]), spec)              # the sprite itself as an "extra view"
check(views.shape == img.shape and np.array_equal((views[0].numpy() * 255).round(), (img[0].numpy() * 255).round()),
      "MatchSpriteCanvas puts a view on the prepared canvas exactly where the sprite is")
big = (img[0].numpy() * 255).round().astype(np.uint8)
frames = [degrade(big, seed=i) for i in range(3)]
out = pixel.repixelate(frames, spec)
cw, ch = spec["canvas"]; ox, oy = spec["offset"]; w, h = spec["size"]
ref = np.zeros((ch, cw, 4), np.uint8); ref[oy:oy + h, ox:ox + w] = sprite[sy0:sy1, sx0:sx1]
exact = np.mean([np.mean(np.all(o == ref, -1)) for o in out])
check(exact > 0.995, f"degraded frames re-pixelated on the same grid: {100 * exact:.2f}% pixels exact")
check(all(o.shape[:2] == (ch, cw) for o in out), "re-pixelated frames keep the sprite canvas size (pixel size unchanged)")
used = np.unique(np.concatenate([o[o[..., 3] > 0][:, :3] for o in out]), axis=0)
check(set(map(tuple, used)) <= set(map(tuple, np.array(spec["palette"]))), "re-pixelated frames use only the sprite's palette")

# 3. height match (flow 3): the character got 30% bigger in the generated frame
bigger = np.asarray(Image.fromarray(big).resize((round(W * 1.3), round(H * 1.3)), Image.NEAREST))
bg = np.array(spec["bg"], np.uint8)
canvas = np.zeros_like(big); canvas[...] = bg
yo, xo = H - bigger.shape[0] + int(0.3 * H * 0.0), (W - bigger.shape[1]) // 2
crop = bigger[max(0, -yo):, max(0, -xo):][:H, :W]
canvas[max(0, yo):max(0, yo) + crop.shape[0], max(0, xo):max(0, xo) + crop.shape[1]] = crop
hm = pixel.repixelate([canvas], spec, height_match=True)[0]
hh = pixel.subject_bbox(hm[..., 3] > 0)
check(abs((hh[3] - hh[1]) - h) <= 1, f"height match: subject {hh[3] - hh[1]} px tall (sprite {h})")

# 4. Pixelize: auto grid on upscaled art, and pixels= on a smooth (non-pixel) image
node = pixel.Pixelize()
pa, prev, pal_txt, inf = node.run(tensor([up7]), 64, "auto (detect grid)", "dominant", 0.0, 16, 8.0, "auto", True, "auto remove", 12.0, False, True, "none", 1, 4)
pa = (pa[0].numpy() * 255).round().astype(np.uint8)
b = pixel.subject_bbox(pa[..., 3] > 0)
check(np.array_equal(pa[b[1]:b[3], b[0]:b[2]], sprite[sy0:sy1, sx0:sx1]), "Pixelize auto(detect grid) on x7 sprite == original exactly")

smooth = Image.fromarray(up7).resize((up7.shape[1] * 3 // 2, up7.shape[0] * 3 // 2), Image.BICUBIC)   # non-integer, blurry
smooth = np.asarray(smooth)
pa2, prev2, pal2, inf2 = node.run(tensor([smooth]), 40, "subject height", "dominant", 0.0, 8, 8.0, "auto", True, "auto remove", 12.0, True, True, "none", 1, 4)
pa2 = (pa2[0].numpy() * 255).round().astype(np.uint8)
b2 = pixel.subject_bbox(pa2[..., 3] > 0)
check(abs((b2[3] - b2[1]) - 40) <= 1, f"Pixelize pixels=40 -> subject {b2[3] - b2[1]} art px tall")
report, ok = pixel.check_pixel_art([pa2], max_colors=8, expected_height=40)
check(ok, "Pixelize output passes Pixel Art Check (hard alpha, x1, <=8 colours)")
check(prev2.shape[1] == pa2.shape[0] * 4, "preview is a x4 nearest-neighbour zoom")

pa3, *_ = node.run(tensor([smooth]), 40, "subject height", "dominant", 0.0, 16, 8.0, "PICO-8", True, "auto remove", 12.0, True, True, "none", 1, 4)
pa3 = (pa3[0].numpy() * 255).round().astype(np.uint8)
pico = set(map(tuple, pixel.parse_palette("PICO-8")))
check(set(map(tuple, np.unique(pa3[pa3[..., 3] > 0][:, :3], axis=0))) <= pico, "palette preset PICO-8 is respected")

noisy = smooth.copy()
noisy[..., :3] = np.clip(noisy[..., :3].astype(int) + np.random.default_rng(3).integers(-40, 41, noisy[..., :3].shape), 0, 255).astype(np.uint8)
iso = []
for sm in (0.0, 0.6):
    pa4, *_ = node.run(tensor([noisy]), 40, "subject height", "median", sm, 8, 8.0, "auto", True, "auto remove", 12.0, False, True, "none", 1, 4)
    pa4 = (pa4[0].numpy() * 255).round().astype(np.uint8)
    q = pa4[..., :3].astype(int) @ [65536, 256, 1]
    q[pa4[..., 3] == 0] = -1
    p = np.pad(q, 1, constant_values=-2)
    iso.append(int(sum((p[1 + dy:1 + dy + q.shape[0], 1 + dx:1 + dx + q.shape[1]] != q) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx).__eq__(8).sum()))
check(iso[1] < iso[0], f"smooth=0.6 reduces isolated pixels on a noisy image ({iso[0]} -> {iso[1]})")

# 5. Pixel Art Check rejects fake pixel art
bad_report, bad_ok = pixel.check_pixel_art([np.dstack([smooth[..., :3], np.full(smooth.shape[:2], 255, np.uint8)])], 32)
check(not bad_ok, "Pixel Art Check FAILs a blurry non-pixel image")

# 6. sprite sheet + Aseprite JSON
ss = pixel.SaveSpriteSheet()
walk = [np.roll(out[0], k, axis=1) for k in range(3)]          # 3 distinct frames + a repeat of the first
res = ss.run(tensor(walk + [walk[0]]), "pixel/test", 1, 0, True, 0, 0, True, 10, "walk", 2)
sheet = res["result"][0][0].numpy()
import json
files = sorted(os.listdir(os.path.join(OUT_DIR, "pixel")))
meta = json.load(open(os.path.join(OUT_DIR, "pixel", [f for f in files if f.endswith(".json")][0])))
check(len(meta["frames"]) == 3 and meta["meta"]["frameTags"][0]["name"] == "walk", f"sheet: loop-trimmed to {len(meta['frames'])} frames, Aseprite tags ok")
check(any(f.endswith(".webp") for f in files) and any(f.endswith(".png") for f in files), f"sheet files written: {files}")

# palette control: merge to own colours, fewest colours, steadier frames
pal16 = np.array([[200, 30, 30], [190, 35, 32], [205, 28, 40], [30, 30, 200], [35, 28, 190], [240, 240, 240],
                  [10, 10, 10], [15, 12, 14]], np.uint8)
red = pixel.reduce_palette(pal16, [5, 1, 1, 4, 1, 3, 6, 1], 4)
check(len(red) == 4 and all(any((c == q).all() for q in pal16) for c in red),
      f"reduce_palette keeps 4 of the sprite's own colours ({pixel.palette_to_text(red)})")
px = np.repeat(pal16, 50, 0)
ap = pixel.auto_palette(px, 32, 20.0, min_colors=2)
check(len(ap) == 4, f"auto_palette: near-duplicate shades merge -> {len(ap)} colours")
check(len(pixel.auto_palette(px, 32, 20.0)) == 8, "auto_palette: art with fewer distinct colours than the floor keeps them all (8)")
ramp = np.repeat(np.stack([np.linspace(40, 60, 40), np.linspace(120, 135, 40), np.linspace(40, 50, 40)], 1).astype(np.uint8), 30, 0)
check(len(pixel.auto_palette(ramp, 32, 20.0, min_colors=2)) < 10 and len(pixel.auto_palette(ramp, 32, 20.0)) == 10,
      f"auto_palette: never fewer than {pixel.AUTO_MIN_COLORS} colours when the art has them (a soft green ramp would give "
      f"{len(pixel.auto_palette(ramp, 32, 20.0, min_colors=2))})")
rng2 = np.random.default_rng(5)
spec_f = {"factor": 8, "canvas": (8, 8), "offset": (0, 0), "size": (8, 8), "palette": [[200, 30, 30], [185, 40, 35]],
          "counts": [10, 10], "bg": [0, 255, 0]}
clip = [np.clip(np.full((64, 64, 3), (193, 35, 33)) + rng2.normal(0, 6, (64, 64, 3)), 0, 255).astype(np.uint8) for _ in range(12)]
raw_f = pixel.repixelate(clip, spec_f, stability=0, deflicker=False, cleanup=False)
steady = pixel.repixelate(clip, spec_f)
check(pixel.flicker_rate(steady) < pixel.flicker_rate(raw_f),
      f"stability + deflicker cut colour popping ({pixel.flicker_rate(raw_f):.1f}% -> {pixel.flicker_rate(steady):.1f}%)")

# edges: background pocket cleared, halo rim defringed, outline drawn, thin parts kept
ring_img = np.zeros((120, 120, 4), np.uint8); ring_img[..., :3] = 255; ring_img[..., 3] = 255     # white background
ring_img[20:100, 20:100, :3] = (200, 30, 30)                    # red body
ring_img[45:75, 45:75, :3] = 255                                # white pocket (background seen through the body)
ring_img[18:20, 20:100, :3] = (240, 200, 200)                   # pale halo along the top edge
ring_img[20:100, 104:107, :3] = (60, 60, 200)                   # thin blue staff, 3 px
fl = pixel.prepare_alpha(ring_img, "auto remove", 12.0)
cl = pixel.clear_enclosed_background(fl, fl[..., :3], 100)
check(cl[60, 60, 3] == 0 and cl[30, 30, 3] == 255, "enclosed background pocket -> transparent, body kept")
bl = pixel.bleed_edges(cl, 3)
check(tuple(bl[19, 50, :3]) == (200, 30, 30), "defringe: halo rim takes the solid interior colour")
pal_e = np.array([[200, 30, 30], [90, 10, 15], [10, 10, 10], [60, 60, 200]], np.uint8)
spr = np.zeros((12, 14, 4), np.uint8); spr[2:10, 2:9] = (200, 30, 30, 255); spr[2:10, 11] = (60, 60, 200, 255)
o1 = pixel.apply_outline(spr, pal_e, "darker shade")
check(tuple(o1[5, 2, :3]) == (90, 10, 15) and tuple(o1[5, 5, :3]) == (200, 30, 30) and tuple(o1[5, 11, :3]) == (60, 60, 200),
      "outline: darker shade on the edge, fill and 1 px staff unchanged")
o2 = pixel.apply_outline(spr, pal_e, "darkest")
check(tuple(o2[2, 5, :3]) == (10, 10, 10) and o2.shape == spr.shape, "outline: darkest, drawn inside (size unchanged)")

sh = np.zeros((100, 100, 4), np.uint8); sh[20:85, 40:60] = (200, 30, 30, 255); sh[80:85, 40:60] = (60, 60, 70, 255)
sh[85:92, 20:80] = (180, 180, 190, 255)                         # light grey ground shadow ellipse under dark boots
ns = pixel.remove_ground_shadow(sh)
check(ns[88, 30, 3] == 0 and ns[82, 50, 3] == 255 and ns[50, 50, 3] == 255, "ground shadow removed, dark boots kept")

# body height ignores a thin prop (staff above the head)
m = np.zeros((400, 300), bool)
m[100:390, 100:200] = True            # body
m[20:390, 210:216] = True             # staff, 6 px wide, taller than the body
check(abs(pixel.body_bbox(m)[1] - 100) <= 4 and pixel.subject_bbox(m)[1] == 20,
      f"body_bbox ignores thin staff (top {pixel.body_bbox(m)[1]}, with staff {pixel.subject_bbox(m)[1]})")

check(res["ui"].get("animated") == (True,) and res["ui"]["images"][0]["filename"].endswith(".webp") and res["result"][3] == 10,
      "sheet node shows the animated WebP and outputs its fps")
pv = pixel.SpriteAnimationPreview().run(tensor(walk), 10, 3, "checker", "ping-pong")
pfile = os.path.join(OUT_DIR, "temp", pv["ui"]["images"][0]["filename"])
anim = Image.open(pfile)
check(pv["ui"]["animated"] == (True,) and anim.n_frames == 4 and anim.size == (walk[0].shape[1] * 3, walk[0].shape[0] * 3),
      f"animation preview: ping-pong 3 -> {anim.n_frames} frames at x3, checker background")

# sprite edit plumbing: restore the original canvas, palette grows with new colours, seamless tiles
src_canvas = np.zeros((40, 30, 4), np.uint8); src_canvas[10:30, 8:20] = (200, 30, 30, 255); src_canvas[12:16, 10:14] = (20, 20, 20, 255)
img_t, spec_e, w_e, h_e, _ = pixel.PrepareSpriteForVideo().run(tensor([src_canvas]), 512, 0.3, "auto")
big = (img_t[0].numpy() * 255).round().astype(np.uint8)
big[spec_e["offset"][1] * spec_e["factor"]:(spec_e["offset"][1] + 3) * spec_e["factor"], :, :] = (30, 200, 60)   # edit adds green
rf = pixel.repixelate([big], spec_e, palette_text="sprite+new", colors=4, stability=0, deflicker=False, cleanup=False)[0]
cols = set(map(tuple, rf[rf[..., 3] > 0][:, :3]))
check(any(abs(c[1] - 200) < 30 and c[0] < 80 for c in cols), "repixelate sprite+new: the edit's new colour joins the palette")
restored = pixel.restore_canvas(pixel.repixelate([(img_t[0].numpy() * 255).round().astype(np.uint8)], spec_e, stability=0, deflicker=False, cleanup=False)[0], spec_e)
check(restored.shape == src_canvas.shape and np.array_equal(restored[..., 3] > 0, src_canvas[..., 3] > 0),
      "restore_canvas: unedited round trip lands on the exact original canvas")
rng_t = np.random.default_rng(1)
pal_t = np.array([[40, 120, 40], [60, 160, 60], [90, 190, 80]], np.uint8)
tex = np.dstack([pal_t[rng_t.integers(0, 3, (24, 24)).repeat(4, 0).repeat(4, 1)], np.full((96, 96), 255, np.uint8)])
tile_t = pixel.make_tileable(tex, 32)
check(tile_t.shape == (32, 32, 4) and set(map(tuple, tile_t[..., :3].reshape(-1, 3))) <= set(map(tuple, pal_t)),
      "make_tileable: 32x32 tile from the texture's own colours")

# 7. directional views: 5 views -> 8 directions, left sides mirrored
sd = pixel.SaveDirections()
views = [np.roll(out[0], k, axis=1) for k in range(5)]
res = sd.run(tensor(views), "pixel/dirs", "front, front-right, right, back-right, back", True, 0, 2)
dirs = (res["result"][1].numpy() * 255).round().astype(np.uint8)
files = sorted(os.listdir(os.path.join(OUT_DIR, "pixel")))
check(dirs.shape[0] == 8 and np.array_equal(dirs[6], dirs[2][:, ::-1]),
      f"directions: 8 views, left = mirrored right ({[f for f in files if f.startswith('dirs') and 'left' in f]})")

# 8 drawn views (mirror off): every direction is the drawn one, saved in clockwise order
views8 = [np.roll(out[0], k, axis=0) for k in range(8)]
order8 = "front, front-right, right, back-right, back, front-left, left, back-left"
res8 = sd.run(tensor(views8), "pixel/dirs8", order8, False, 0, 2)
d8 = (res8["result"][1].numpy() * 255).round().astype(np.uint8)
clockwise = ["front", "front-right", "right", "back-right", "back", "back-left", "left", "front-left"]
names8 = [n.strip() for n in order8.split(",")]
check(d8.shape[0] == 8 and all(np.array_equal(d8[clockwise.index(n)], views8[i]) for i, n in enumerate(names8)),
      "directions: 8 drawn views with mirror off are kept as drawn, reordered clockwise")

# 9. Grid Cutout: on-grid pixel art + the full-size cut-out -> the sprite, no resampling, no new colours
gpal = np.array([[250, 250, 250], [30, 20, 40], [200, 60, 60]], np.uint8)
grid = np.zeros((16, 16, 4), np.uint8)
grid[..., :3], grid[..., 3] = gpal[0], 255                      # white background, opaque
grid[4:13, 5:11, :3] = gpal[1]                                  # the subject: a 6x9 block with a red core
grid[6:10, 7:9, :3] = gpal[2]
full = pixel.nearest_upscale(grid, 8)                           # the 128 px image it came from
cut = full.copy()
cut[..., 3] = 0
cut[4 * 8:13 * 8 - 2, 5 * 8:11 * 8 - 2, 3] = 255                # BiRefNet's mask is a little off the grid
sprite = pixel.grid_cutout(grid, cut, padding=1)
opaque = sprite[sprite[..., 3] > 0][:, :3]
check(sprite.shape[:2] == (11, 8) and set(sprite[..., 3].ravel()) <= {0, 255}
      and set(map(tuple, opaque)) <= {tuple(gpal[1]), tuple(gpal[2])} and len(opaque) == 6 * 9,
      f"grid_cutout: the 6x9 subject cells kept exactly, no background, padded 1 ({sprite.shape[1]}x{sprite.shape[0]})")
halo = grid.copy()
halo[4, 5:11, :3] = gpal[0]                                     # the refiner gave the top edge row the background colour
wide = cut.copy()
wide[4 * 8:13 * 8, 5 * 8:11 * 8, 3] = 255                       # ... and the cut-out still covers it
h1, h0 = pixel.grid_cutout(halo, wide, padding=0), pixel.grid_cutout(halo, wide, padding=0, key_background=False)
check(h1.shape[0] == 8 and h0.shape[0] == 9 and (h0[0, :, :3] == gpal[0]).all(),
      "grid_cutout: background-coloured edge cells are keyed out (without it they stay as a white halo row)")

print(f"\n{sum(RESULTS)}/{len(RESULTS)} passed")
sys.exit(0 if all(RESULTS) else 1)
