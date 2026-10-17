"""True pixel-art nodes (numpy / scipy / Pillow only).

Guarantees the pixel-art workflows rely on:
- Native resolution: 1 art pixel = 1 image pixel, on one uniform grid (no mixed pixel sizes, no anti-aliasing).
- Limited palette (N colours, a preset, a hex list, or the input's exact palette), shared across frames/views.
- Hard alpha (0 or 255) with no background halo.
- Scale preservation for pixel inputs: a sprite is upscaled by an integer factor for an AI video model, and the
  generated frames are brought back down on exactly the same grid, so the art-pixel size can't change.

Ideas borrowed (not code) from MIT/Apache projects: spritefusion-pixel-snapper, unfake, PixelOE, K-Centroid.
"""
import json
import math
import os

import numpy as np
import torch
from PIL import Image

# ----------------------------------------------------------------------------- palettes
PALETTES = {
    "PICO-8": "000000 1d2b53 7e2553 008751 ab5236 5f574f c2c3c7 fff1e8 ff004d ffa300 ffec27 00e436 29adff 83769c ff77a8 ffccaa",
    "GameBoy": "0f380f 306230 8bac0f 9bbc0f",
    "DB32": ("000000 222034 45283c 663931 8f563b df7126 d9a066 eec39a fbf236 99e550 6abe30 37946e 4b692f 524b24 323c39 "
             "3f3f74 306082 5b6ee1 639bff 5fcde4 cbdbfc ffffff 9badb7 847e87 696a6a 595652 76428a ac3232 d95763 d77bba "
             "8f974a 8a6f30"),
    "Endesga-32": ("be4a2f d77643 ead4aa e4a672 b86f50 733e39 3e2731 a22633 e43b44 f77622 feae34 fee761 63c74d 3e8948 "
                   "265c42 193c3e 124e89 0099db 2ce8f5 ffffff c0cbdc 8b9bb4 5a6988 3a4466 262b44 181425 ff0044 68386c "
                   "b55088 f6757a e8b796 c28569"),
}


def parse_palette(text):
    """Preset name or hex list -> (N, 3) uint8 array, or None for automatic."""
    text = (text or "").strip()
    if not text or text.lower() == "auto":
        return None
    for name, hexes in PALETTES.items():
        if text.lower() == name.lower():
            text = hexes
    hexes = [h for h in text.replace(",", " ").replace("#", " ").split() if len(h) == 6]
    if not hexes:
        return None
    return np.array([[int(h[i:i + 2], 16) for i in (0, 2, 4)] for h in hexes], np.uint8)


def palette_to_text(pal):
    return " ".join("#%02x%02x%02x" % tuple(int(v) for v in c) for c in pal)


# ----------------------------------------------------------------------------- colour maths
def rgb_to_lab(rgb):
    """sRGB uint8/float (..., 3) -> CIELAB (..., 3) float32."""
    c = np.asarray(rgb, np.float32) / (255.0 if np.asarray(rgb).dtype == np.uint8 or np.max(rgb) > 1.5 else 1.0)
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.2126, 0.0193], [0.3576, 0.7152, 0.1192], [0.1805, 0.0722, 0.9505]], np.float32)
    xyz /= np.array([0.95047, 1.0, 1.08883], np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def nearest_index(lab, pal_lab):
    """Index of the nearest palette colour for every pixel (chunked to bound memory)."""
    flat = lab.reshape(-1, 3)
    out = np.empty(len(flat), np.int32)
    step = max(1, 4_000_000 // max(1, len(pal_lab)))
    for i in range(0, len(flat), step):
        d = ((flat[i:i + step, None, :] - pal_lab[None]) ** 2).sum(-1)
        out[i:i + step] = d.argmin(1)
    return out.reshape(lab.shape[:-1])


def kmeans_palette(rgb_pixels, k, iters=12, seed=0):
    """k-means in Lab on (N, 3) uint8 pixels -> (k', 3) uint8 palette (k' <= k), most common colour first."""
    px = np.asarray(rgb_pixels, np.uint8).reshape(-1, 3)
    uniq = np.unique(px, axis=0)
    if len(uniq) <= k:
        return uniq
    rng = np.random.default_rng(seed)
    sample = px[rng.choice(len(px), size=min(len(px), 30000), replace=False)]
    lab = rgb_to_lab(sample)
    centers = [lab[rng.integers(len(lab))]]                 # k-means++ init
    for _ in range(1, k):
        d = np.min(((lab[:, None] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        centers.append(lab[rng.choice(len(lab), p=d / d.sum())] if d.sum() > 0 else lab[rng.integers(len(lab))])
    centers = np.array(centers, np.float32)
    for _ in range(iters):
        idx = nearest_index(lab, centers)
        for j in range(len(centers)):
            m = idx == j
            if m.any():
                centers[j] = lab[m].mean(0)
    idx = nearest_index(lab, centers)
    pal = []
    for j in np.argsort(-np.bincount(idx, minlength=len(centers))):
        m = idx == j
        if m.any():                                           # use a real pixel colour (medoid-ish): median RGB
            pal.append(np.median(sample[m], axis=0))
    return np.unique(np.array(pal, np.uint8), axis=0) if pal else uniq[:k]


def palette_pixels(rgba, flat_only=True, max_step=10.0):
    """Opaque pixels to build a palette from. flat_only: skip anti-aliased / blurred transition pixels (a big colour
    step to a neighbour), so in-between shades don't become palette colours and small solid areas (a blade) decide."""
    opaque = rgba[..., 3] >= 128
    if not opaque.any():
        return rgba[..., :3].reshape(-1, 3)
    if flat_only:
        lab = rgb_to_lab(rgba[..., :3])
        step = np.zeros(opaque.shape, np.float32)
        for axis in (0, 1):
            d = np.sqrt((np.diff(lab, axis=axis) ** 2).sum(-1))
            pad = [(0, 0), (0, 0)]
            pad[axis] = (1, 0)
            step = np.maximum(step, np.pad(d, pad))
            pad[axis] = (0, 1)
            step = np.maximum(step, np.pad(d, pad))
        flat = opaque & (step < max_step)
        if flat.sum() >= 0.2 * opaque.sum():
            return rgba[..., :3][flat]
    return rgba[..., :3][opaque]


def palette_error(rgb_pixels, palette, percentile=97):
    """Colour error (Lab distance) of pixels mapped to their nearest palette colour, at a high percentile so small
    distinct areas (a blade, skin, eyes; a few % of the pixels) count, not just the big areas that dominate a mean.
    Meant for flat pixels (palette_pixels), so anti-aliased edges don't inflate it."""
    lab = rgb_to_lab(np.asarray(rgb_pixels, np.uint8).reshape(-1, 3))
    pal = rgb_to_lab(palette)
    return float(np.percentile(np.sqrt(((lab - pal[nearest_index(lab, pal)]) ** 2).sum(-1)), percentile))


def auto_palette(rgb_pixels, max_colors=32, max_error=20.0):
    """Fewest colours (2..max_colors) whose palette keeps the colour error (97th percentile of flat pixels, Lab) under max_error."""
    px = np.asarray(rgb_pixels, np.uint8).reshape(-1, 3)
    if len(px) > 30000:
        px = px[np.random.default_rng(0).choice(len(px), 30000, replace=False)]
    pal = None
    for k in range(2, max_colors + 1):
        pal = kmeans_palette(px, k)
        if palette_error(px, pal) <= max_error or len(pal) < k:
            break
    return rescue_accents(px, pal, max_colors)


def rescue_accents(px, pal, max_colors=32, min_delta=25.0, min_share=0.0015):
    """Small but distinct colours (glowing eyes, a lantern, a gem) are too few pixels to move an error percentile,
    so they get merged into a big neighbour. Add a palette colour for every such cluster: pixels far (Lab > min_delta)
    from every palette colour, if there are at least min_share of them (and >= 12 pixels)."""
    pal = np.asarray(pal, np.uint8)
    for _ in range(4):
        if len(pal) >= max_colors:
            break
        lab = rgb_to_lab(px)
        d = np.sqrt(((lab - rgb_to_lab(pal)[nearest_index(lab, rgb_to_lab(pal))]) ** 2).sum(-1))
        far = px[d > min_delta]
        if len(far) < max(12, min_share * len(px)):
            break
        extra = kmeans_palette(far, min(3, max_colors - len(pal)))
        lab_far = rgb_to_lab(far)
        counts = np.bincount(nearest_index(lab_far, rgb_to_lab(extra)), minlength=len(extra))
        keep = []
        for c in extra[counts >= max(12, min_share * len(px))]:           # no near-duplicates of what we have
            ref = np.concatenate([pal] + ([np.array(keep, np.uint8)] if keep else []), 0)
            if np.sqrt(((rgb_to_lab(c[None])[0] - rgb_to_lab(ref)) ** 2).sum(-1)).min() > 12:
                keep.append(c)
        if not keep:
            break
        pal = np.concatenate([pal, np.array(keep, np.uint8)], 0)
    return pal


def reduce_palette(palette, weights, n):
    """Merge a palette down to n of its OWN colours (weighted k-means in Lab, each cluster keeps its most used
    member), so the sprite's look and colours are kept, just fewer of them."""
    palette = np.asarray(palette, np.uint8)
    if n <= 0 or len(palette) <= n:
        return palette
    w = np.maximum(np.asarray(weights, np.float64), 1e-6)
    lab = rgb_to_lab(palette)
    order = np.argsort(-w)
    centers = [lab[order[0]]]                                   # farthest-point init, from the most used colour
    for _ in range(1, n):
        d = np.min(((lab[:, None] - np.array(centers)[None]) ** 2).sum(-1), axis=1) * w
        centers.append(lab[int(d.argmax())])
    centers = np.array(centers, np.float32)
    for _ in range(20):
        idx = nearest_index(lab, centers)
        for j in range(n):
            m = idx == j
            if m.any():
                centers[j] = (lab[m] * w[m, None]).sum(0) / w[m].sum()
    idx = nearest_index(lab, centers)
    keep = [int(np.flatnonzero(idx == j)[w[idx == j].argmax()]) for j in range(n) if (idx == j).any()]
    return palette[sorted(set(keep), key=lambda i: -w[i])]


def flicker_rate(frames, max_delta_e=30.0):
    """Share of opaque cells that change to a SIMILAR colour between consecutive frames (colour popping), in %."""
    if len(frames) < 2:
        return 0.0
    changed = total = 0
    for a, b in zip(frames[:-1], frames[1:]):
        if a.shape != b.shape:
            continue
        both = (a[..., 3] > 0) & (b[..., 3] > 0)
        diff = both & (a[..., :3] != b[..., :3]).any(-1)
        if diff.any():
            d = np.sqrt(((rgb_to_lab(a[..., :3][diff]) - rgb_to_lab(b[..., :3][diff])) ** 2).sum(-1))
            changed += int((d < max_delta_e).sum())
        total += int(both.sum())
    return 100.0 * changed / max(1, total)


# ----------------------------------------------------------------------------- tensors <-> numpy
def to_rgba_np(t):
    """ComfyUI IMAGE (B,H,W,3|4) float -> list of (H,W,4) uint8."""
    arr = (t.detach().cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    if arr.shape[-1] == 3:
        arr = np.concatenate([arr, np.full(arr.shape[:-1] + (1,), 255, np.uint8)], -1)
    return list(arr)


def to_image_tensor(frames):
    """list of (H,W,4) uint8 (same size) -> IMAGE (B,H,W,4) float."""
    return torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)


def pad_to_same_size(frames, anchor="bottom-center"):
    H = max(f.shape[0] for f in frames)
    W = max(f.shape[1] for f in frames)
    out = []
    for f in frames:
        c = np.zeros((H, W, 4), np.uint8)
        y = H - f.shape[0] if anchor.startswith("bottom") else (H - f.shape[0]) // 2
        x = (W - f.shape[1]) // 2
        c[y:y + f.shape[0], x:x + f.shape[1]] = f
        out.append(c)
    return out


def nearest_upscale(img, f):
    return np.repeat(np.repeat(img, f, 0), f, 1) if f > 1 else img


# ----------------------------------------------------------------------------- background
def flood_background(rgb, tolerance=12.0):
    """Mask of the plain background: pixels connected to the border whose colour is close to the border colour."""
    from scipy import ndimage
    lab = rgb_to_lab(rgb)
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    ref = np.median(border, axis=0)
    close = np.sqrt(((lab - ref) ** 2).sum(-1)) < tolerance
    lbl, _ = ndimage.label(close)
    edge = np.unique(np.concatenate([lbl[0], lbl[-1], lbl[:, 0], lbl[:, -1]]))
    return np.isin(lbl, edge[edge > 0])


# ----------------------------------------------------------------------------- grid detection
def detect_scale(img_rgba, max_scale=64):
    """Integer upscale factor of nearest-neighbour pixel art (1 = native). Returns (scale, offset_x, offset_y, score)."""
    rgb = img_rgba[..., :3].astype(np.int16)
    a = img_rgba[..., 3].astype(np.int16)
    best = (1, 0, 0, 0.0)

    def axis_changes(axis):
        d = (np.abs(np.diff(rgb, axis=axis)).sum(-1) > 24) | (np.abs(np.diff(a, axis=axis)) > 64)
        return np.nonzero(d.any(axis=1 - axis))[0] + 1        # positions where any row/col changes colour

    cx, cy = axis_changes(1), axis_changes(0)
    if len(cx) < 2 or len(cy) < 2:
        return best
    for s in range(max_scale, 1, -1):
        if s * 4 > min(img_rgba.shape[:2]):
            continue
        ox = np.bincount(cx % s, minlength=s).argmax()
        oy = np.bincount(cy % s, minlength=s).argmax()
        score = 0.5 * (np.mean(cx % s == ox) + np.mean(cy % s == oy))
        if score > 0.97:
            return (s, int(ox), int(oy), float(score))
    return best


# ----------------------------------------------------------------------------- core: grid downsample
def grid_downsample(rgba, cell_w, cell_h, x0, y0, nx, ny, palette, method="dominant", alpha_cut=0.5, bg_index=None):
    """Sample an (ny, nx) grid of cells (size cell_w x cell_h, origin x0,y0; may be fractional) to palette indices.

    Returns (indices (ny,nx) int, alpha (ny,nx) bool). Pixels are first mapped to the palette (plus an optional
    background colour), then each cell takes the most common palette colour of its pixels ("dominant"), which keeps
    outlines crisp, or the palette colour nearest the cell's median/mean/centre colour.
    """
    H, W = rgba.shape[:2]
    if (method == "dominant" and float(cell_w).is_integer() and cell_w == cell_h and float(x0).is_integer()
            and float(y0).is_integer() and x0 >= 0 and y0 >= 0 and x0 + nx * cell_w <= W and y0 + ny * cell_h <= H):
        return _grid_downsample_fast(rgba, int(cell_w), int(x0), int(y0), nx, ny, palette, alpha_cut, bg_index)
    xs = np.clip(np.round(x0 + np.arange(nx + 1) * cell_w).astype(int), 0, W)
    ys = np.clip(np.round(y0 + np.arange(ny + 1) * cell_h).astype(int), 0, H)
    pal_lab = rgb_to_lab(palette)
    idx_img = nearest_index(rgb_to_lab(rgba[..., :3]), pal_lab) if method == "dominant" else None
    out = np.zeros((ny, nx), np.int32)
    alpha = np.zeros((ny, nx), bool)
    K = len(palette)
    for j in range(ny):
        for i in range(nx):
            ya, yb, xa, xb = ys[j], max(ys[j + 1], ys[j] + 1), xs[i], max(xs[i + 1], xs[i] + 1)
            # use the cell interior (drop a 15% border) so edge blur from the model doesn't leak in
            iy = max(0, int((yb - ya) * 0.15)); ix = max(0, int((xb - xa) * 0.15))
            sl = (slice(ya + iy, max(yb - iy, ya + iy + 1)), slice(xa + ix, max(xb - ix, xa + ix + 1)))
            a = rgba[sl][..., 3]
            if a.size == 0:
                continue
            opaque = a >= 255 * alpha_cut
            if opaque.mean() < 0.5:
                continue
            if method == "dominant":
                counts = np.bincount(idx_img[sl][opaque], minlength=K)
                k = int(counts.argmax())
            else:
                px = rgba[sl][..., :3][opaque].astype(np.float32)
                c = (np.median(px, 0) if method == "median" else px.mean(0) if method == "mean"
                     else rgba[(ya + yb) // 2, (xa + xb) // 2, :3].astype(np.float32))
                k = int(nearest_index(rgb_to_lab(c[None]), pal_lab)[0])
            if bg_index is not None and k == bg_index:
                continue
            out[j, i] = k
            alpha[j, i] = True
    return out, alpha


def _grid_downsample_fast(rgba, f, x0, y0, nx, ny, palette, alpha_cut, bg_index, return_counts=False):
    """Vectorised 'dominant' sampling for an integer grid (video frames): mode of palette indices per cell interior."""
    region = rgba[y0:y0 + ny * f, x0:x0 + nx * f]
    b = int(f * 0.15) if f >= 4 else 0
    idx = nearest_index(rgb_to_lab(region[..., :3]), rgb_to_lab(palette)).reshape(ny, f, nx, f)
    a = (region[..., 3] >= 255 * alpha_cut).reshape(ny, f, nx, f)
    idx = idx[:, b:f - b, :, b:f - b].transpose(0, 2, 1, 3).reshape(ny * nx, -1)
    a = a[:, b:f - b, :, b:f - b].transpose(0, 2, 1, 3).reshape(ny * nx, -1)
    K = len(palette)
    counts = np.zeros((ny * nx, K), np.int32)
    cells = np.repeat(np.arange(ny * nx), idx.shape[1])
    keep = a.ravel()
    np.add.at(counts, (cells[keep], idx.ravel()[keep]), 1)
    out = counts.argmax(1).reshape(ny, nx)
    alpha = (a.mean(1) >= 0.5).reshape(ny, nx)
    if bg_index is not None:
        alpha &= out != bg_index
    if return_counts:
        return np.where(alpha, out, 0).astype(np.int32), alpha, counts.reshape(ny, nx, K)
    return np.where(alpha, out, 0).astype(np.int32), alpha


def compose(indices, alpha, palette):
    img = np.zeros(indices.shape + (4,), np.uint8)
    img[..., :3] = np.where(alpha[..., None], palette[indices], 0)     # transparent pixels are (0,0,0,0)
    img[..., 3] = np.where(alpha, 255, 0)
    return img


def cleanup_strays(img, max_delta_e=22.0):
    """Replace isolated single pixels (no same-colour neighbour) by their neighbours' majority colour when that colour
    is perceptually close; distinct details such as eyes (large colour difference) are kept."""
    out = img.copy()
    H, W = img.shape[:2]
    if H < 3 or W < 3:
        return out
    key = (img[..., 0].astype(np.int64) << 24) | (img[..., 1].astype(np.int64) << 16) | (img[..., 2].astype(np.int64) << 8) | img[..., 3]
    offs = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
    pad = np.pad(key, 1, constant_values=-1)
    nbs = np.stack([pad[1 + dy:1 + dy + H, 1 + dx:1 + dx + W] for dy, dx in offs], -1)
    lone = (img[..., 3] > 0) & ~(nbs == key[..., None]).any(-1)
    lone[0, :] = lone[-1, :] = lone[:, 0] = lone[:, -1] = False
    lab = rgb_to_lab(img[..., :3])
    for y, x in zip(*np.nonzero(lone)):
        vals, cnt = np.unique(nbs[y, x], return_counts=True)
        maj = vals[cnt.argmax()]
        if cnt.max() < 5 or maj < 0 or (maj & 255) == 0:
            continue
        k = int(np.flatnonzero(nbs[y, x] == maj)[0])
        yy, xx = y + offs[k][0], x + offs[k][1]
        if np.sqrt(((lab[y, x] - lab[yy, xx]) ** 2).sum()) < max_delta_e:
            out[y, x] = img[yy, xx]
    return out


def clear_enclosed_background(rgba, original_rgb, min_area, tolerance=8.0):
    """Background trapped inside the silhouette (between arm and body, between the legs): opaque pockets that are
    the background colour (estimated from the transparent pixels / image border) and at least min_area pixels big
    become transparent. Smaller ones (eye whites, highlights) are kept."""
    from scipy import ndimage
    a = rgba[..., 3] >= 128
    if a.all() or not a.any():
        return rgba
    clear = ~a
    ref = np.median(original_rgb[clear], axis=0) if clear.sum() > 100 else np.median(
        np.concatenate([original_rgb[0], original_rgb[-1], original_rgb[:, 0], original_rgb[:, -1]]), axis=0)
    close = np.sqrt(((rgb_to_lab(original_rgb) - rgb_to_lab(ref.astype(np.uint8)[None, None])) ** 2).sum(-1)) < tolerance
    lbl, n = ndimage.label(a & close)
    if n == 0:
        return rgba
    areas = np.bincount(lbl.ravel())
    big = np.flatnonzero(areas >= min_area)
    big = big[big > 0]
    if len(big) == 0:
        return rgba
    out = rgba.copy()
    out[np.isin(lbl, big), 3] = 0
    return out


def remove_ground_shadow(rgba, band=0.2, min_l=55.0, max_chroma=18.0):
    """Drop shadows / ground ellipses the model drew under the subject: light, low-saturation regions in the bottom
    `band` of the subject that are wider than tall become transparent. Dark grey parts (armoured boots) are kept."""
    from scipy import ndimage
    a = rgba[..., 3] >= 128
    if not a.any():
        return rgba
    x0, y0, x1, y1 = subject_bbox(a)
    lab = rgb_to_lab(rgba[..., :3])
    chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    cand = a & (lab[..., 0] > min_l) & (lab[..., 0] < 97) & (chroma < max_chroma)
    cand[:max(0, int(y1 - band * (y1 - y0)))] = False
    lbl, n = ndimage.label(cand)
    if n == 0:
        return rgba
    out = rgba.copy()
    for i, sl in enumerate(ndimage.find_objects(lbl), 1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if w >= 2 * h and w >= 0.15 * (x1 - x0) and sl[0].stop >= y1 - 2:     # a flat patch at the very bottom
            out[lbl == i, 3] = 0
    return out


def bleed_edges(rgba, inset=2):
    """Defringe: every pixel that isn't solid interior (the anti-aliased / background-mixed rim, and everything
    transparent) takes the colour of the nearest solid interior pixel. Alpha is unchanged. Stops light halo colours
    from reaching the palette and from mixing into border cells."""
    from scipy import ndimage
    solid = rgba[..., 3] >= 250
    if inset > 0:
        solid = ndimage.binary_erosion(solid, iterations=inset)
    if not solid.any() or solid.all():
        return rgba
    _, (iy, ix) = ndimage.distance_transform_edt(~solid, return_indices=True)
    out = rgba.copy()
    out[..., :3] = rgba[iy, ix, :3]
    return out


def smooth_silhouette(img):
    """Art-pixel silhouette cleanup: drop isolated/spur pixels (at most one opaque neighbour of 8) and fill one-pixel
    notches (a transparent pixel with 3+ opaque 4-neighbours takes their most common colour)."""
    out = img.copy()
    a = out[..., 3] > 0
    H, W = a.shape
    pa = np.pad(a, 1)
    n8 = sum(pa[1 + dy:1 + dy + H, 1 + dx:1 + dx + W] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
    spur = a & (n8 <= 1)
    out[spur] = 0
    a = out[..., 3] > 0
    pa = np.pad(a, 1)
    n4 = sum(pa[1 + dy:1 + dy + H, 1 + dx:1 + dx + W] for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)))
    po = np.pad(out, ((1, 1), (1, 1), (0, 0)))
    for y, x in zip(*np.nonzero(~a & (n4 >= 3))):
        nb = [po[y + 1 + dy, x + 1 + dx] for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)) if po[y + 1 + dy, x + 1 + dx, 3] > 0]
        keys = [tuple(c) for c in nb]
        out[y, x] = max(set(keys), key=keys.count)
    return out


def darker_shade(rgb, palette, min_drop=12.0):
    """The palette colour that best works as a darker shade of rgb (same hue family, clearly darker); the darkest
    palette colour if nothing fits."""
    pal_lab = rgb_to_lab(palette)
    lab = rgb_to_lab(np.asarray(rgb, np.uint8)[None])[0]
    target = np.array([lab[0] * 0.5, lab[1] * 0.85, lab[2] * 0.85])
    cand = np.flatnonzero(pal_lab[:, 0] < lab[0] - min_drop)
    if len(cand) == 0:
        return palette[int(pal_lab[:, 0].argmin())]
    return palette[cand[int(np.sqrt(((pal_lab[cand] - target) ** 2).sum(-1)).argmin())]]


def existing_outline_share(img, palette):
    """Share of silhouette edge pixels that already are the darkest palette colour (an existing dark outline)."""
    a = img[..., 3] > 0
    H, W = a.shape
    pa = np.pad(a, 1)
    edge = a & ~(pa[:-2, 1:-1] & pa[2:, 1:-1] & pa[1:-1, :-2] & pa[1:-1, 2:])
    if not edge.any():
        return 0.0
    darkest = palette[int(rgb_to_lab(palette)[:, 0].argmin())]
    return float((img[edge][:, :3] == darkest).all(-1).mean())


def apply_outline(img, palette, mode="auto"):
    """Make the silhouette edge a clean outline, inside the sprite (size unchanged). Only where the shape is at least
    3 px thick across the edge, so thin parts (blades, staffs, hair strands) keep their colour. Edge pixels that are
    already clearly darker than the fill behind them are kept."""
    if mode == "none":
        return img
    if mode == "auto":
        mode = "darkest" if existing_outline_share(img, palette) >= 0.4 else "darker shade"
    if mode == "outside darkest":
        return add_outline(np.pad(img, ((1, 1), (1, 1), (0, 0))), palette)
    out = img.copy()
    a = img[..., 3] > 0
    H, W = a.shape
    lab = rgb_to_lab(img[..., :3])
    darkest = palette[int(rgb_to_lab(palette)[:, 0].argmin())]
    shade_cache = {}
    pa = np.pad(a, 2)
    for y, x in zip(*np.nonzero(a)):
        fill = None
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if not pa[y + 2 + dy, x + 2 + dx] and pa[y + 2 - dy, x + 2 - dx] and pa[y + 2 - 2 * dy, x + 2 - 2 * dx]:
                fill = (y - 2 * dy, x - 2 * dx)                 # 2 px inside, away from this transparent side
                break
        if fill is None:
            continue
        fy, fx = fill
        if lab[y, x, 0] < lab[fy, fx, 0] - 12:                  # already an outline
            continue
        if mode == "darkest":
            out[y, x, :3] = darkest
            continue
        key = tuple(img[fy, fx, :3])
        if key not in shade_cache:
            shade_cache[key] = darker_shade(img[fy, fx, :3], palette)
        out[y, x, :3] = shade_cache[key]
    return out


def add_outline(img, palette):
    from scipy import ndimage
    a = img[..., 3] > 0
    ring = ndimage.binary_dilation(a, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])) & ~a
    dark = palette[np.argmin(rgb_to_lab(palette)[:, 0])]
    out = img.copy()
    out[ring, :3] = dark
    out[ring, 3] = 255
    return out



# ----------------------------------------------------------------------------- PixelOE (Apache-2.0) backend
def pixeloe_available():
    try:
        from pixeloe.legacy.pixelize import pixelize  # noqa: F401
        return True
    except Exception:
        return False


def pixeloe_grid(rgba, cell, gx, gy, nx, ny, thickness=2):
    """Run PixelOE's contrast-aware pixelization (outline expansion + contrast-based cell choice) on the grid area.

    Returns an (ny, nx, 3) uint8 RGB grid aligned with our grid; alpha is decided separately from the cell coverage.
    """
    import cv2
    from pixeloe.legacy.pixelize import pixelize
    H, W = rgba.shape[:2]
    x0, y0 = int(round(gx)), int(round(gy))
    w, h = int(round(nx * cell)), int(round(ny * cell))
    # PixelOE needs an opaque image. Use the colours as they are (Pixelize bleeds the interior colours outwards
    # first), NOT composited on white, which would mix white into every border cell.
    canvas = np.full((h, w, 3), 255, np.uint8)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = rgba[sy0:sy1, sx0:sx1, :3]
    # PixelOE takes ONE target size and works its own stride out from it, so a square-ish number on a frame that
    # is far from square rounds one stride to zero and torch raises. Ask by the longest side instead: the result
    # is resized to (ny, nx) just below in any case.
    target = max(4, max(nx, ny))
    out = pixelize(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), mode="contrast", target_size=target,
                   patch_size=max(1, int(round(cell))), thickness=thickness, no_upscale=True)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    if out.shape[:2] != (ny, nx):
        out = np.asarray(Image.fromarray(out).resize((nx, ny), Image.NEAREST))
    return out


def coverage_alpha(rgba, cell, gx, gy, nx, ny, cut=0.5):
    """Hard alpha per cell: opaque if at least `cut` of the cell is opaque."""
    H, W = rgba.shape[:2]
    xs = np.clip(np.round(gx + np.arange(nx + 1) * cell).astype(int), 0, W)
    ys = np.clip(np.round(gy + np.arange(ny + 1) * cell).astype(int), 0, H)
    a = (rgba[..., 3] >= 128).astype(np.float32)
    ii = np.pad(a.cumsum(0).cumsum(1), ((1, 0), (1, 0)))         # integral image
    area = np.maximum(1, np.outer(np.diff(ys), np.diff(xs)))
    s = ii[ys[1:], :][:, xs[1:]] - ii[ys[:-1], :][:, xs[1:]] - ii[ys[1:], :][:, xs[:-1]] + ii[ys[:-1], :][:, xs[:-1]]
    return (s / area) >= cut

# ----------------------------------------------------------------------------- Clean Cutout
class CleanCutout:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "drop_shadow": ("BOOLEAN", {"default": True, "tooltip": "Remove a ground / drop shadow the model drew under the subject: a light, low-saturation, flat patch at the very bottom. Dark grey parts (armoured boots) are kept."}),
            "enclosed_background": ("BOOLEAN", {"default": False, "tooltip": "Make background seen through the subject (between arm and body, between legs) transparent: opaque pockets of the background colour bigger than min_pocket. Off by default: white eye whites / highlights on a white background look the same, so check the result."}),
            "min_pocket": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.1, "step": 0.001, "tooltip": "Smallest enclosed pocket removed, as a fraction of the subject's area (0.004 = 0.4%). Smaller ones (eye whites, highlights) are kept."}),
            "bg_tolerance": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 40.0, "step": 0.5, "tooltip": "How close (Lab) a pocket must be to the background colour."}),
            "shadow_band": ("FLOAT", {"default": 0.2, "min": 0.05, "max": 0.5, "step": 0.05, "tooltip": "Bottom part of the subject searched for a shadow (fraction of its height)."}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "removed", "info")
    FUNCTION = "run"
    CATEGORY = "comfyui_workflows/pixel"
    DESCRIPTION = "After background removal: also remove drop shadows and background pockets inside the subject."

    def run(self, images, drop_shadow, enclosed_background, min_pocket, bg_tolerance, shadow_band):
        outs, removed, infos = [], [], []
        for f in to_rgba_np(images):
            before = f[..., 3] >= 128
            g = remove_ground_shadow(f, shadow_band) if drop_shadow else f
            shadow_px = int((before & (g[..., 3] < 128)).sum())
            if enclosed_background:
                g = clear_enclosed_background(g, f[..., :3], max(1, int(min_pocket * before.sum())), bg_tolerance)
            gone = before & (g[..., 3] < 128)
            outs.append(g)
            removed.append(gone.astype(np.float32))
            infos.append(f"shadow {shadow_px} px, pockets {int(gone.sum()) - shadow_px} px removed")
        return (to_image_tensor(outs), torch.from_numpy(np.stack(removed)), "\n".join(infos))


# ----------------------------------------------------------------------------- Pixelize
def subject_bbox(alpha_mask):
    ys, xs = np.nonzero(alpha_mask)
    if len(ys) == 0:
        return 0, 0, alpha_mask.shape[1], alpha_mask.shape[0]
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def body_bbox(alpha_mask, frac=0.03):
    """Bounding box of the character's body without thin props (staffs, weapon tips, hair strands): the largest
    blobs left after a morphological opening with a disk of `frac` x subject height. Falls back to subject_bbox."""
    from scipy import ndimage
    x0, y0, x1, y1 = subject_bbox(alpha_mask)
    k = max(1, (y1 - y0) // 256)                                  # work at <= ~256 px tall for speed
    small = alpha_mask[y0:y1:k, x0:x1:k]
    r = max(1, int(round(frac * small.shape[0])))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    opened = ndimage.binary_opening(small, structure=(xx * xx + yy * yy) <= r * r)
    lbl, n = ndimage.label(opened)
    if n == 0:
        return x0, y0, x1, y1
    areas = np.bincount(lbl.ravel())[1:]
    big = np.isin(lbl, np.nonzero(areas >= 0.25 * areas.max())[0] + 1)    # body may split (legs, cape gap)
    bx0, by0, bx1, by1 = subject_bbox(big)
    return x0 + bx0 * k, y0 + by0 * k, min(x1, x0 + bx1 * k), min(y1, y0 + by1 * k)


def prepare_alpha(rgba, background, tolerance):
    rgba = rgba.copy()
    has_alpha = (rgba[..., 3] < 250).any()
    if background == "auto remove" and not has_alpha:
        rgba[flood_background(rgba[..., :3], tolerance), 3] = 0
    elif background == "keep":
        rgba[..., 3] = 255
    return rgba


def kuwahara(rgba, radius):
    """Edge-preserving smoothing: each pixel takes the mean of whichever of its 4 overlapping quadrant windows has
    the lowest colour variance. Flattens texture/noise into the solid areas pixel art needs, keeps shape edges."""
    from scipy import ndimage
    r = int(radius)
    if r < 2:
        return rgba
    f = rgba[..., :3].astype(np.float32)
    k, h = r + 1, r // 2
    mean = np.stack([ndimage.uniform_filter(f[..., c], k, mode="nearest") for c in range(3)], -1)
    var = sum(ndimage.uniform_filter(f[..., c] ** 2, k, mode="nearest") for c in range(3)) - (mean ** 2).sum(-1)
    H, W = var.shape
    pv = np.pad(var, h, mode="edge")
    pm = np.pad(mean, ((h, h), (h, h), (0, 0)), mode="edge")
    best = np.full((H, W), np.inf, np.float32)
    res = f.copy()
    for dy in (-h, h):
        for dx in (-h, h):
            v = pv[h + dy:h + dy + H, h + dx:h + dx + W]
            sel = v < best
            best[sel] = v[sel]
            res[sel] = pm[h + dy:h + dy + H, h + dx:h + dx + W][sel]
    out = rgba.copy()
    out[..., :3] = res.clip(0, 255).astype(np.uint8)
    return out


def pixelize_frames(frames, pixels, measure, method, colors, palette_text, shared_palette, background, tolerance,
                    cleanup, outline, padding, smooth=0.0, color_error=20.0, edge_cleanup=True):
    frames = [prepare_alpha(f, background, tolerance) for f in frames]
    edge_cleanup = edge_cleanup and measure != "auto (detect grid)"   # recovering existing pixel art: keep it exact
    if edge_cleanup:                    # defringe: no background-mixed rim colours in the palette or border cells
        frames = [bleed_edges(f, max(1, round(min(f.shape[:2]) / 512))) for f in frames]
    fixed = parse_palette(palette_text)
    infos, results, palettes = [], [], []

    def pick_palette(px):
        if fixed is not None:
            return fixed if colors <= 0 or len(fixed) <= colors else reduce_palette(fixed, np.bincount(
                nearest_index(rgb_to_lab(px[:30000]), rgb_to_lab(fixed)), minlength=len(fixed)), colors)
        return auto_palette(px, 32, color_error) if colors <= 0 else kmeans_palette(px, colors)

    shared = None
    if shared_palette and len(frames) > 1:
        allpx = np.concatenate([palette_pixels(f) for f in frames])
        shared = pick_palette(allpx)
    # one cell size for all frames/views: from the first frame's subject (equal subject height across views)
    cell = None
    for f in frames:
        a = f[..., 3] >= 128
        x0, y0, x1, y1 = subject_bbox(a)
        if measure == "auto (detect grid)":
            s, ox, oy, score = detect_scale(f)
            cs = float(s); gx, gy = float(ox % s), float(oy % s)
            nx, ny = int(math.ceil((f.shape[1] - gx) / cs)), int(math.ceil((f.shape[0] - gy) / cs))
        else:
            if measure == "body height (per frame)":
                bx0, by0, bx1, by1 = body_bbox(a)
                cs = max(1.0, (by1 - by0) / max(1, pixels))
                cx = (bx0 + bx1) / 2                              # centre on the body, not on props
                nx = int(math.ceil(2 * max(cx - x0, x1 - cx) / cs)) + 2 * padding
                ny = int(math.ceil((y1 - y0) / cs)) + 2 * padding
                gx = cx - nx * cs / 2
                gy = y1 + padding * cs - ny * cs
            elif cell is None or measure != "subject height":
                ref = {"subject height": y1 - y0, "longest side": max(y1 - y0, x1 - x0), "image height": f.shape[0],
                       "whole frame": f.shape[0], "cell size": None}[measure]
                cell = float(pixels) if measure == "cell size" else max(1.0, ref / max(1, pixels))
            if measure == "whole frame":
                # the frame IS the art: no bounding box, no trimming, no padding. A parallax layer has to stay
                # exactly as tall as the screen with its empty sky intact, or it will not line up with the layers
                # behind it; and exactly as wide, or it will not repeat.
                cs = cell
                nx, ny = int(round(f.shape[1] / cs)), int(round(f.shape[0] / cs))
                gx, gy = 0.0, 0.0
            elif measure != "body height (per frame)":
                cs = cell
                # anchor the grid on the subject's bottom-centre so the feet line and centre are stable
                nx = int(math.ceil((x1 - x0) / cs)) + 2 * padding
                ny = int(math.ceil((y1 - y0) / cs)) + 2 * padding
                gx = (x0 + x1) / 2 - nx * cs / 2
                gy = y1 + padding * cs - ny * cs
        if smooth > 0 and measure != "auto (detect grid)":
            f = kuwahara(f, round(smooth * cs))
        pal = shared if shared is not None else pick_palette(palette_pixels(f))
        use_oe = method == "pixeloe" and measure != "auto (detect grid)" and pixeloe_available()
        if use_oe:
            try:
                grid_rgb = pixeloe_grid(f, cs, gx, gy, nx, ny)
            except Exception as e:                            # noqa: BLE001
                # PixelOE is an optimisation, not the method: losing it costs the outline-expansion pass, and the
                # median grid below is what runs when it is not installed at all. Failing the whole job over it
                # would be the wrong trade - and its errors arrive as unreadable torch shape complaints.
                infos.append(f"PixelOE declined this frame ({type(e).__name__}: {e}); used the median grid instead")
                use_oe = False
            else:
                alpha = coverage_alpha(f, cs, gx, gy, nx, ny)
                idx = nearest_index(rgb_to_lab(grid_rgb), rgb_to_lab(pal))
        if not use_oe:
            m = "median" if method == "pixeloe" else method
            idx, alpha = grid_downsample(f, cs, cs, gx, gy, nx, ny, pal, m)
        out = compose(idx, alpha, pal)
        if edge_cleanup:
            out = smooth_silhouette(out)
        if cleanup:
            out = cleanup_strays(out)
        if outline is True:                                   # old boolean value: the 1 px outside outline
            outline = "outside darkest"
        if outline:
            out = apply_outline(out, pal, outline)
        results.append(out)
        palettes.append(pal)
        infos.append(f"cell {cs:.2f}px -> {out.shape[1]}x{out.shape[0]} art px, {len(np.unique(out[out[..., 3] > 0][:, :3], axis=0))} colours")
    return results, palettes, infos


class Pixelize:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "pixels": ("INT", {"default": 64, "min": 4, "max": 4096, "tooltip": "Size in art pixels, measured as set in 'measure' (default: subject height). Over 1024 is for wide strips - a parallax layer measured on its longest side - not for sprites."}),
            "measure": (["subject height", "longest side", "image height", "whole frame", "cell size", "body height (per frame)", "auto (detect grid)"],
                        {"tooltip": "What 'pixels' means. subject height uses the first image's grid for all. whole frame = the frame itself becomes 'pixels' tall, nothing cropped to the subject and no padding: for a parallax layer or anything that must keep its exact size and its empty areas. body height (per frame) = each image's body (thin props like staffs ignored) becomes 'pixels' tall: for views drawn at different scales. auto = the input is already (upscaled) pixel art: detect its grid and recover it exactly."}),
            "method": (["pixeloe", "dominant", "median", "center", "mean"], {"tooltip": "How each art pixel is chosen. pixeloe = PixelOE's contrast-aware method (outline expansion; cleanest for illustrations). dominant = most common palette colour in the cell. median/center/mean = that colour of the cell, snapped to the palette."}),
            "smooth": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.1, "tooltip": "Edge-preserving smoothing before pixelizing (Kuwahara radius in art pixels). Flattens texture/noise into solid areas; 0 = off. Ignored for 'auto (detect grid)'."}),
            "colors": ("INT", {"default": 16, "min": 0, "max": 256, "tooltip": "Number of colours. 0 = automatic: the fewest colours that keep the colour error under color_error. With a preset/hex palette: use only its N most fitting colours (0 = all)."}),
            "color_error": ("FLOAT", {"default": 20.0, "min": 2.0, "max": 60.0, "step": 0.5, "tooltip": "colors = 0 only: allowed colour difference (Lab, 90th percentile, so small areas like blades and eyes count) from the original. Higher = fewer colours (12 = faithful, 20 = simplified, 30+ = very few)."}),
            "palette": ("STRING", {"default": "auto", "tooltip": "auto, a preset (PICO-8, DB32, Endesga-32, GameBoy) or hex colours (#1a1c2c #5d275d ...)."}),
            "shared_palette": ("BOOLEAN", {"default": True, "tooltip": "Batches (frames/views): one palette for all."}),
            "background": (["auto remove", "keep", "alpha only"], {"tooltip": "auto remove: plain border-connected background becomes transparent (inputs with alpha keep theirs)."}),
            "bg_tolerance": ("FLOAT", {"default": 12.0, "min": 1.0, "max": 60.0, "tooltip": "Colour distance for background removal."}),
            "cleanup": ("BOOLEAN", {"default": True, "tooltip": "Remove isolated stray pixels (distinct ones like eyes are kept)."}),
            "edge_cleanup": ("BOOLEAN", {"default": True, "tooltip": "Defringe before pixelizing (edge pixels take the nearest solid colour, so no light halo colours) and smooth the silhouette after (no 1-pixel spurs or notches)."}),
            "outline": (["auto", "darker shade", "darkest", "outside darkest", "none"], {"tooltip": "Silhouette outline, drawn on the sprite's edge pixels (size unchanged) where the shape is 3+ px thick. auto = complete an existing dark outline if the art has one, else darker shade. darker shade = a darker palette colour of the fill behind it (selective outline). darkest = the darkest palette colour (black). outside darkest = an extra 1 px ring around the sprite."}),
            "padding": ("INT", {"default": 1, "min": 0, "max": 64, "tooltip": "Transparent art pixels around the subject."}),
            "preview_scale": ("INT", {"default": 8, "min": 1, "max": 32, "tooltip": "Nearest-neighbour zoom for the preview output."}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("pixel_art", "preview", "palette", "info")
    FUNCTION = "run"
    CATEGORY = "comfyui_workflows/pixel"
    DESCRIPTION = "Turn images into true pixel art: exact grid at native resolution, limited palette, hard alpha."

    def run(self, images, pixels, measure, method, smooth, colors, color_error, palette, shared_palette, background,
            bg_tolerance, cleanup, edge_cleanup, outline, padding, preview_scale):
        frames, pals, infos = pixelize_frames(to_rgba_np(images), pixels, measure, method, colors, palette,
                                              shared_palette, background, bg_tolerance, cleanup, outline, padding, smooth,
                                              color_error, edge_cleanup)
        frames = pad_to_same_size(frames)
        preview = [nearest_upscale(f, preview_scale) for f in frames]
        return (to_image_tensor(frames), to_image_tensor(preview), palette_to_text(pals[0]), "\n".join(infos))


def grid_cutout(native, cutout, coverage=0.5, padding=1, cleanup=False):
    """Pixel art already on its grid (one image pixel per art pixel, opaque) + the full-size cut-out it was made from
    -> the sprite with hard alpha, trimmed to the subject. A cell is opaque when `coverage` of it is subject."""
    ny, nx = native.shape[:2]
    cell_x, cell_y = cutout.shape[1] / nx, cutout.shape[0] / ny
    if abs(cell_x - cell_y) > 1e-6:
        raise ValueError(f"cut-out {cutout.shape[1]}x{cutout.shape[0]} is not a whole grid of the {nx}x{ny} pixel art")
    keep = coverage_alpha(cutout, cell_x, 0.0, 0.0, nx, ny, coverage)
    out = native.copy()
    out[..., 3] = np.where(keep, 255, 0)
    out[~keep] = 0
    if cleanup:
        out = cleanup_strays(out)
    x0, y0, x1, y1 = subject_bbox(out[..., 3] > 0)
    out = out[y0:y1, x0:x1]
    return np.pad(out, ((padding, padding), (padding, padding), (0, 0))) if padding else out


class GridCutout:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "pixel_art": ("IMAGE", {"tooltip": "Pixel art at its native grid, background included (e.g. Krea 2 Pixel Art Refiner with scale_to_original off)."}),
            "cutout": ("IMAGE", {"tooltip": "The full-size image it was made from, with the background removed (alpha)."}),
            "coverage": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 0.95, "step": 0.05, "tooltip": "Share of a cell that must be subject for the art pixel to stay."}),
            "padding": ("INT", {"default": 1, "min": 0, "max": 64, "tooltip": "Transparent art pixels around the subject."}),
            "cleanup": ("BOOLEAN", {"default": False, "tooltip": "Remove isolated stray pixels (distinct ones like eyes are kept)."}),
            "preview_scale": ("INT", {"default": 8, "min": 1, "max": 32, "tooltip": "Nearest-neighbour zoom for the preview output."}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("pixel_art", "preview", "info")
    FUNCTION = "run"
    CATEGORY = "comfyui_workflows/pixel"
    DESCRIPTION = "Cut a sprite out of pixel art that is already on its grid, using the full-size cut-out's alpha. No resampling and no new colours."

    def run(self, pixel_art, cutout, coverage, padding, cleanup, preview_scale):
        arts, cuts = to_rgba_np(pixel_art), to_rgba_np(cutout)
        frames = [grid_cutout(a, cuts[min(i, len(cuts) - 1)], coverage, padding, cleanup) for i, a in enumerate(arts)]
        frames = pad_to_same_size(frames)
        colours = len(np.unique(frames[0][frames[0][..., 3] > 0][:, :3], axis=0))
        info = f"{frames[0].shape[1]}x{frames[0].shape[0]} art px from a {arts[0].shape[1]}x{arts[0].shape[0]} grid, {colours} colours"
        return (to_image_tensor(frames), to_image_tensor([nearest_upscale(f, preview_scale) for f in frames]), info)


# ----------------------------------------------------------------------------- sprites through video models
def pick_background(palette):
    """A flat background colour far from every palette colour (so it can be keyed out exactly)."""
    cands = np.array([[255, 0, 255], [0, 255, 0], [0, 255, 255], [255, 255, 0], [0, 0, 255], [255, 128, 0],
                      [128, 128, 128], [255, 255, 255], [0, 0, 0]], np.uint8)
    pal_lab, c_lab = rgb_to_lab(palette), rgb_to_lab(cands)
    d = np.sqrt(((c_lab[:, None] - pal_lab[None]) ** 2).sum(-1)).min(1)
    return cands[int(d.argmax())]


def recover_native(rgba, tolerance=12.0):
    """Upscaled pixel art (nearest) -> native image, exactly. Native art is returned unchanged."""
    s, ox, oy, _ = detect_scale(rgba)
    if (rgba[..., 3] >= 250).all():
        rgba = rgba.copy()
        rgba[flood_background(rgba[..., :3], tolerance), 3] = 0
    if s == 1:
        out = rgba.copy()
    else:
        out = rgba[oy + s // 2::s, ox + s // 2::s].copy()          # centre sample of each cell
    out[..., 3] = np.where(out[..., 3] >= 128, 255, 0)
    out[out[..., 3] == 0] = 0
    return out, s


class PrepareSpriteForVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "sprite": ("IMAGE",),
            "target_size": ("INT", {"default": 640, "min": 128, "max": 1536, "step": 16, "tooltip": "Approximate longest side sent to the video model."}),
            "padding": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Free space around the sprite for motion, as a fraction of its size (each side)."}),
            "bg_color": ("STRING", {"default": "auto", "tooltip": "auto = a flat colour far from the sprite palette, or a hex colour."}),
        }}

    RETURN_TYPES = ("IMAGE", "PIXEL_SPEC", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "pixel_spec", "width", "height", "info")
    FUNCTION = "run"
    CATEGORY = "comfyui_workflows/pixel"
    DESCRIPTION = "Upscale a pixel sprite by an integer factor onto a flat background for a video model, and remember the grid."

    def run(self, sprite, target_size, padding, bg_color):
        native, detected = recover_native(to_rgba_np(sprite)[0])
        src_h, src_w = native.shape[:2]
        x0, y0, x1, y1 = subject_bbox(native[..., 3] > 0)
        native = native[y0:y1, x0:x1]
        h, w = native.shape[:2]
        palette, counts = np.unique(native[native[..., 3] > 0][:, :3], axis=0, return_counts=True)
        pad_x, pad_y = int(math.ceil(w * padding)), int(math.ceil(h * padding))
        cw, ch = w + 2 * pad_x, h + 2 * pad_y
        f = max(1, round(target_size / max(cw, ch)))
        while (cw * f) % 16:
            cw += 1
        while (ch * f) % 16:
            ch += 1
        bg = parse_palette(bg_color)
        bg = bg[0] if bg is not None else pick_background(palette)
        canvas = np.zeros((ch, cw, 4), np.uint8)
        canvas[..., :3] = bg
        canvas[..., 3] = 255
        ox, oy = (cw - w) // 2, (ch - h) // 2
        m = native[..., 3] > 0
        canvas[oy:oy + h, ox:ox + w][m] = native[m]
        big = nearest_upscale(canvas, f)
        spec = {"factor": f, "canvas": (cw, ch), "offset": (ox, oy), "size": (w, h), "palette": palette.tolist(),
                "counts": counts.tolist(),
                "bg": bg.tolist(), "detected_input_scale": detected,
                "source": {"size": (int(src_w), int(src_h)), "bbox": (int(x0), int(y0), int(x1), int(y1))}}
        info = (f"sprite {w}x{h} art px ({len(palette)} colours, input scale x{detected}) -> canvas {cw}x{ch} x{f} = "
                f"{cw * f}x{ch * f}px, background #{int(bg[0]):02x}{int(bg[1]):02x}{int(bg[2]):02x}")
        img = torch.from_numpy(big[None, ..., :3].astype(np.float32) / 255.0)
        return (img, spec, cw * f, ch * f, info)


def frame_palette(spec, colors=0, palette_text="sprite"):
    """The palette the frames are snapped to: the sprite's own, merged down to `colors`, or a preset/hex list
    (optionally only its `colors` most fitting entries)."""
    sprite = np.array(spec["palette"], np.uint8)
    weights = np.array(spec.get("counts") or [1] * len(sprite), np.float64)
    fixed = parse_palette(palette_text) if palette_text.strip().lower() not in ("", "sprite", "auto") else None
    if fixed is None:
        return reduce_palette(sprite, weights, colors)
    use = np.bincount(nearest_index(rgb_to_lab(sprite), rgb_to_lab(fixed)), weights=weights, minlength=len(fixed))
    if colors > 0:
        return reduce_palette(fixed[use > 0], use[use > 0], colors) if (use > 0).sum() > colors else fixed[use > 0]
    return fixed


def _palette_with_new_colours(frames_rgb, palette, bg, max_new):
    """Sprite palette plus up to max_new colours that the (edited) frames introduce: flat pixels far (Lab > 14) from
    every sprite colour and from the background become extra palette entries (k-means)."""
    px = np.concatenate([palette_pixels(np.dstack([np.asarray(f)[..., :3], np.full(np.asarray(f).shape[:2], 255, np.uint8)]))
                         for f in frames_rgb[:4]])
    if len(px) > 60000:
        px = px[np.random.default_rng(0).choice(len(px), 60000, replace=False)]
    lab = rgb_to_lab(px)
    ref = rgb_to_lab(np.concatenate([palette, bg[None]], 0))
    d = np.sqrt(((lab[:, None] - ref[None]) ** 2).sum(-1)).min(1)
    new = px[d > 14]
    if len(new) < 0.002 * len(px) or len(new) < 20:
        return palette
    extra = kmeans_palette(new, max_new)
    keep = [c for c in extra if np.sqrt(((rgb_to_lab(c[None])[0] - ref) ** 2).sum(-1)).min() > 10]
    return np.concatenate([palette, np.array(keep, np.uint8)], 0) if keep else palette


def repixelate(frames_rgb, spec, height_match=False, bg_tolerance=18.0, method="dominant", colors=0,
               palette_text="sprite", stability=0.7, deflicker=True, cleanup=True, outline="none"):
    f = spec["factor"]
    cw, ch = spec["canvas"]
    add_new = palette_text.strip().lower().replace(" ", "") in ("sprite+new", "sprite+newcolours", "sprite+newcolors")
    palette = frame_palette(spec, 0 if add_new else colors, "sprite" if add_new else palette_text)
    bg = np.array(spec["bg"], np.uint8)
    if add_new and len(frames_rgb):
        palette = _palette_with_new_colours(frames_rgb, palette, bg, colors or 6)
    pal_bg = np.concatenate([palette, bg[None]], 0)
    bg_index = len(palette)
    frames = []
    for fr in frames_rgb:
        if fr.shape[:2] != (ch * f, cw * f):
            fr = np.asarray(Image.fromarray(fr[..., :3]).resize((cw * f, ch * f), Image.BILINEAR))
        frames.append(np.concatenate([_rekey_background(fr[..., :3], bg, bg_tolerance, palette), np.full(fr.shape[:2] + (1,), 255, np.uint8)], -1))
    if height_match and frames:
        # one scale for the whole clip, from the first frame (the driving video's start pose), about one fixed
        # anchor: the character gets the sprite's height and motion (steps, crouches, jumps) is kept as is
        x0, y0, x1, y1 = _subject_box(frames[0], bg, bg_tolerance, f)
        h = y1 - y0
        target = spec["size"][1] * f
        if h >= 8 and abs(h - target) >= f:
            anchor = ((x0 + x1) / 2, y1)
            frames = [_scale_about(fr, target / h, anchor, bg) for fr in frames]
    idxs, alphas = [], []
    for t, rgba in enumerate(frames):
        if method == "dominant":
            idx, alpha, votes = _grid_downsample_fast(rgba, f, 0, 0, cw, ch, pal_bg, 0.5, bg_index, return_counts=True)
            if stability > 0 and t > 0:
                # hysteresis: a cell keeps last frame's colour unless the new winner clearly beats it
                prev, pa = idxs[-1], alphas[-1]
                best = np.take_along_axis(votes, idx[..., None], -1)[..., 0]
                pv = np.take_along_axis(votes, prev[..., None], -1)[..., 0]
                keep = alpha & pa & (pv >= (1.0 - stability) * best) & (pv > 0)
                idx = np.where(keep, prev, idx)
        else:
            idx, alpha = grid_downsample(rgba, f, f, 0, 0, cw, ch, pal_bg, method, bg_index=bg_index)
        idxs.append(idx)
        alphas.append(alpha)
    if deflicker and len(idxs) > 2:
        # one-frame colour blinks: same colour before and after, a similar colour in between -> use that colour
        lab = rgb_to_lab(pal_bg)
        dist = np.sqrt(((lab[:, None] - lab[None]) ** 2).sum(-1))
        for _ in range(2):
            for t in range(1, len(idxs) - 1):
                a, b, c = idxs[t - 1], idxs[t], idxs[t + 1]
                solid = alphas[t - 1] & alphas[t] & alphas[t + 1]
                blink = solid & (a == c) & (b != a) & (dist[a, b] < 30)
                idxs[t] = np.where(blink, a, b)
    out = []
    for idx, alpha in zip(idxs, alphas):
        o = compose(np.where(alpha, idx, 0), alpha, pal_bg)
        if cleanup:
            o = cleanup_strays(smooth_silhouette(o))
        out.append(apply_outline(o, palette, outline) if outline not in ("none", "outside darkest") else o)
    return out


def _rekey_background(rgb, bg, tol, palette=None):
    """The frame's own flat background, painted back to the key colour. Video models drift the background they were
    given (green turning teal, blue, then purple over a clip); matched against the one key colour, the drifted
    background reads as sprite colours and comes out as a solid block. So each frame's border-connected background
    (close to that frame's own border colour) is found and set to the key first. A frame whose border is still the
    key colour is left as it is, and so is one with no figure in it (all one colour).
    The flood stops short of the sprite: its tolerance is kept under half the distance from the drifted background to
    the nearest sprite colour, so a background that drifted towards a body colour does not run into the body."""
    lab_bg = rgb_to_lab(np.asarray(bg, np.uint8)[None, None])[0, 0]
    lab = rgb_to_lab(np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])[None])[0]
    edge = np.median(lab, axis=0)
    if np.sqrt(((edge - lab_bg) ** 2).sum()) < tol:
        return rgb
    if palette is not None and len(palette):
        near = np.sqrt(((rgb_to_lab(np.asarray(palette, np.uint8)[None])[0] - edge) ** 2).sum(-1)).min()
        tol = max(3.0, min(tol, near / 2))
    mask = flood_background(rgb, tol)
    if mask.mean() < 0.05 or mask.mean() > 0.95:  # no flat border, or nothing but border: not a figure on a background
        return rgb
    out = rgb.copy()
    out[mask] = bg
    return out


def _subject_box(rgba, bg, tol, f):
    """Character box against the flat background, ignoring compression noise and small specks."""
    from scipy import ndimage
    lab = rgb_to_lab(rgba[..., :3])
    fg = np.sqrt(((lab - rgb_to_lab(bg[None, None])) ** 2).sum(-1)) > tol
    fg = ndimage.binary_opening(fg, structure=np.ones((3, 3), bool), iterations=max(1, f // 3))
    lbl, n = ndimage.label(fg)
    if n == 0:
        return subject_bbox(fg)
    areas = np.bincount(lbl.ravel())[1:]
    return subject_bbox(np.isin(lbl, np.nonzero(areas >= 0.05 * areas.max())[0] + 1))


def _scale_about(rgba, s, anchor, bg):
    """Scale a high-res frame by s about anchor (x, y), on the flat background."""
    H, W = rgba.shape[:2]
    big = Image.fromarray(rgba[..., :3]).resize((max(1, round(W * s)), max(1, round(H * s))), Image.BILINEAR)
    ax, ay = anchor
    ox, oy = round(ax - ax * s), round(ay - ay * s)          # where the scaled image's origin lands
    canvas = np.zeros_like(rgba)
    canvas[..., :3] = bg
    canvas[..., 3] = 255
    arr = np.asarray(big)
    xa, ya = max(0, ox), max(0, oy)
    xb, yb = min(W, ox + arr.shape[1]), min(H, oy + arr.shape[0])
    if xb > xa and yb > ya:
        canvas[ya:yb, xa:xb, :3] = arr[ya - oy:yb - oy, xa - ox:xb - ox]
    return canvas


class RepixelateFrames:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "frames": ("IMAGE",),
            "pixel_spec": ("PIXEL_SPEC",),
            "height_match": ("BOOLEAN", {"default": False, "tooltip": "Scale the clip (one factor, from the first frame) so the character has the sprite's height (for motion taken from a video)."}),
            "bg_tolerance": ("FLOAT", {"default": 18.0, "min": 1.0, "max": 80.0, "tooltip": "Colour distance used to find the subject against the flat background (height match)."}),
            "method": (["dominant", "median", "center", "mean"],),
            "colors": ("INT", {"default": 0, "min": 0, "max": 256, "tooltip": "0 = the sprite's own palette. N = merge it down to N of its own colours (fewer colours = less popping)."}),
            "palette": ("STRING", {"default": "sprite", "tooltip": "sprite = the sprite's colours, or a preset (PICO-8, DB32, Endesga-32, GameBoy) / hex colours. With colors > 0 only the N most used entries are kept."}),
            "stability": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 0.95, "step": 0.05, "tooltip": "Temporal colour stability: a pixel keeps last frame's colour unless the new colour clearly wins (0 = off, higher = steadier, may lag fast changes). Needs method dominant."}),
            "deflicker": ("BOOLEAN", {"default": True, "tooltip": "Remove one-frame colour blinks (a pixel that switches to a similar colour for a single frame)."}),
            "cleanup": ("BOOLEAN", {"default": True, "tooltip": "Remove isolated stray pixels and 1-pixel spurs/notches on the silhouette in each frame (distinct ones like eyes are kept)."}),
            "outline": (["none", "auto", "darker shade", "darkest"], {"tooltip": "Re-draw the silhouette outline in every frame (keeps it unbroken while moving). none = keep what the sprite/frames have."}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("pixel_frames", "info", "palette")
    FUNCTION = "run"
    CATEGORY = "comfyui_workflows/pixel"
    DESCRIPTION = "Bring video-model frames back to the sprite's exact pixel grid and palette (same pixel size), background keyed out."

    def run(self, frames, pixel_spec, height_match, bg_tolerance, method, colors, palette, stability, deflicker, cleanup,
            outline):
        fr = [f[..., :3] for f in to_rgba_np(frames)]
        out = repixelate(fr, pixel_spec, height_match, bg_tolerance, method, colors, palette, stability, deflicker, cleanup,
                         outline)
        pal = frame_palette(pixel_spec, colors, palette)
        cw, ch = pixel_spec["canvas"]
        info = (f"{len(out)} frames -> {cw}x{ch} art px, {len(pal)} colours (sprite has {len(pixel_spec['palette'])}), "
                f"colour popping {flicker_rate(out):.2f}% of pixels per frame")
        return (to_image_tensor(out), info, palette_to_text(pal))


def restore_canvas(frame, spec, mode="original canvas"):
    """Re-pixelated canvas frame -> the sprite's original canvas (same size and position as the input sprite), its
    trimmed bounding box, or the full padded canvas."""
    if mode == "full canvas":
        return frame
    if mode == "trim":
        x0, y0, x1, y1 = subject_bbox(frame[..., 3] > 0)
        return frame[y0:y1, x0:x1]
    src = spec.get("source")
    if not src:
        return frame
    (sw, sh), (bx0, by0, _, _) = src["size"], src["bbox"]
    ox, oy = spec["offset"]
    cx0, cy0 = ox - bx0, oy - by0                      # canvas position of the source image's (0, 0)
    out = np.zeros((sh, sw, 4), np.uint8)
    H, W = frame.shape[:2]
    xa, ya, xb, yb = max(0, cx0), max(0, cy0), min(W, cx0 + sw), min(H, cy0 + sh)
    if xb > xa and ya < yb:
        out[ya - cy0:yb - cy0, xa - cx0:xb - cx0] = frame[ya:yb, xa:xb]
    up = int(spec.get("detected_input_scale") or 1)          # the input was upscaled pixel art: give it back that way
    return nearest_upscale(out, up) if up > 1 else out


class RestoreSpriteCanvas:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "frames": ("IMAGE",),
            "pixel_spec": ("PIXEL_SPEC",),
            "mode": (["original canvas", "trim", "full canvas"], {"tooltip": "original canvas = same size and position as the input sprite (edits can be swapped into sheets). trim = just the subject. full canvas = keep the padding."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "run"
    CATEGORY = "comfyui_workflows/pixel"
    DESCRIPTION = "Put re-pixelated frames back onto the input sprite's original canvas."

    def run(self, frames, pixel_spec, mode):
        out = [restore_canvas(f, pixel_spec, mode) for f in to_rgba_np(frames)]
        return (to_image_tensor(pad_to_same_size(out)),)


# ----------------------------------------------------------------------------- seamless tiles
def _min_cut(err):
    """Vertical minimum-error seam through err (rows, cols): returns the cut column per row (DP, 8-connected)."""
    h, w = err.shape
    cost = err.astype(np.float64).copy()
    for y in range(1, h):
        prev = cost[y - 1]
        left = np.concatenate([[np.inf], prev[:-1]])
        right = np.concatenate([prev[1:], [np.inf]])
        cost[y] += np.minimum(np.minimum(left, prev), right)
    cut = np.zeros(h, int)
    cut[-1] = int(cost[-1].argmin())
    for y in range(h - 2, -1, -1):
        c = cut[y + 1]
        lo, hi = max(0, c - 1), min(w, c + 2)
        cut[y] = lo + int(cost[y, lo:hi].argmin())
    return cut, float(cost[-1].min())


def _wrap_columns(patch, t, o):
    """patch (rows, t+o): make it wrap horizontally at width t. Columns 0..o-1 become a min-cut join of the patch's
    right overlap (continuing past column t-1) and its left overlap (continuing into column o). Exact pixels only."""
    L, R = patch[:, :o], patch[:, t:t + o]
    err = np.sqrt(((rgb_to_lab(L[..., :3]) - rgb_to_lab(R[..., :3])) ** 2).sum(-1))
    cut, cost = _min_cut(err)
    out = patch[:, :t].copy()
    for y, c in enumerate(cut):
        out[y, :c] = R[y, :c]
    return out, cost


def make_tileable(img, t, overlap=0, tries=24, height=0, axis="both"):
    """Native pixel-art texture -> a tile that repeats seamlessly, using only the texture's own pixels
    (image-quilting minimum-error seams on the wrap). Picks the best position it can find.

    With axis="x" and an image already exactly `t` wide, there is nothing spare to cut from, so it is wrapped in
    place: its own left edge becomes the seam material, which is what the next repeat shows anyway. That is the
    normal case for a parallax layer, which is pixelized straight to its final width.

    axis "both" gives a t x t tile that wraps in both directions: the original behaviour, and what a terrain or
    ground texture wants.

    axis "x" gives a `t` wide by `height` tall strip that wraps sideways ONLY. That is what a parallax layer is:
    as tall as the screen, repeating left to right, with a sky at the top and ground at the bottom that must
    never meet. Wrapping such a strip vertically would fold one into the other.

    The seam work is identical either way - _wrap_columns is per-axis - so x-only is this function with the
    vertical pass left out.
    """
    t = int(t)
    o = overlap or max(2, t // 4)
    both = axis != "x"
    th = t if both else int(height or img.shape[0])
    H, W = img.shape[:2]
    need_h = th + o if both else th
    # An image already at the tile's width has no spare columns to cut a seam from, which is the normal case for a
    # parallax layer: it is pixelized straight to the width it will be in the game. Wrap it in place instead - the
    # seam material is its own left edge, because that is exactly what the next repeat puts there.
    self_wrap = not both and W == t and H >= th
    if self_wrap:
        # The seam material is the image's RIGHT edge, appended so _wrap_columns compares it against the left one
        # and cuts the better path between them: the left columns then continue what the right columns were
        # doing, which is exactly what the next repeat needs. Appending the LEFT edge instead compares a column
        # with itself, costs nothing anywhere, and leaves the image untouched - a seam that looks mended and is not.
        o = min(o, max(2, t // 4), W - 1)
        img = np.concatenate([img, img[:, -o:]], axis=1)
        W = img.shape[1]
    elif H < need_h or W < t + o:
        raise ValueError(f"texture {W}x{H} too small for a {t}x{th} tile with {o}px overlap "
                         f"(needs at least {t + o}x{need_h}); an image already {t} wide is wrapped in place "
                         f"instead, which needs axis=\"x\" and the full height")
    best = None
    y_room, x_room = H - need_h, W - t - o
    ys = np.linspace(0, y_room, min(tries, y_room + 1)).astype(int)
    xs = np.linspace(0, x_room, min(tries, x_room + 1)).astype(int)
    rng = np.random.default_rng(0)
    cands = {(int(y), int(x)) for y, x in zip(rng.choice(ys, tries), rng.choice(xs, tries))}
    cands.add((y_room // 2, x_room // 2))
    for y, x in cands:
        patch = img[y:y + need_h, x:x + t + o]
        h1, c1 = _wrap_columns(patch, t, o)                   # (need_h) x t, left/right edges now joined
        if not both:
            if best is None or c1 < best[0]:
                best = (c1, h1)
            continue
        v, c2 = _wrap_columns(h1.transpose(1, 0, 2), t, o)    # t x t (transposed)
        if best is None or c1 + c2 < best[0]:
            best = (c1 + c2, v.transpose(1, 0, 2))
    return best[1]


def busiest_band(img, height):
    """The top row of the `height`-tall band holding the most detail. Detail = how much neighbouring rows differ,
    summed across the width, so flat sky scores nothing and a horizon scores highly.

    Every band is scored directly rather than by smoothing and taking an argmax: a window as wide as the band makes
    every position that covers the detail score alike, and an argmax then resolves that tie towards row 0 - which
    is the top of the sky, the one place a layer does not want. Ties here resolve towards centring the detail.
    """
    H = img.shape[0]
    if H <= height:
        return 0
    g = img[..., :3].astype(np.float32).mean(axis=2)
    rows = np.abs(np.diff(g, axis=0)).sum(axis=1)                 # length H-1
    if rows.sum() <= 0:
        return (H - height) // 2                                  # nothing to find: the middle is as good as anywhere
    cum = np.concatenate([[0.0], np.cumsum(rows)])
    tops = np.arange(0, H - height + 1)
    score = cum[np.minimum(tops + height, len(rows))] - cum[tops]
    best = score.max()
    centroid = float((rows * np.arange(len(rows))).sum() / rows.sum())
    want = centroid - height / 2.0
    near = tops[score >= best * 0.99]                             # all the bands that are effectively as good
    return int(near[np.argmin(np.abs(near - want))])


def clear_sky(rgba, step=14.0, feather=2, keep=0.02):
    """Transparent from the top edge down to the skyline. `step`: how far a pixel may sit from the one above it,
    in Lab, and still count as sky. Returns (image, fraction cleared).

    Deliberately run before pixelizing. Afterwards a gradient is a handful of flat bands and every band edge reads
    like a skyline, so the region stops at the first one and most of the sky survives.
    """
    a = rgba.copy()
    h, w = a.shape[:2]
    if h < 2 or step <= 0:        # 0 = this slot has no sky to clear: the sky layer itself, or a set piece
        return a, 0.0
    lab = rgb_to_lab(a[..., :3])
    sky = np.zeros((h, w), bool)
    sky[0] = a[0, :, 3] >= 128                       # the top row is sky wherever it is not already transparent
    for y in range(1, h):
        if not sky[y - 1].any():
            break
        d = np.sqrt(((lab[y] - lab[y - 1]) ** 2).sum(-1))
        # a pixel joins if the one above it is sky, or a neighbour of it is: a skyline is rarely perfectly vertical
        above = sky[y - 1]
        near = above | np.roll(above, 1) | np.roll(above, -1)
        sky[y] = near & (d < step) & (a[y, :, 3] >= 128)
    # Two guards were tried here and both were wrong, for opposite reasons. "Too much of it was sky" refused a
    # far layer, which legitimately IS mostly sky. "It reached the bottom edge" refused a near layer, where the
    # sky genuinely shows between the trunks all the way down. Neither shape is a fault.
    # The only real fault is clearing the whole picture, so that is what is guarded: leave something behind.
    if 1.0 - sky.mean() < keep:
        return a, 0.0
    if feather > 0 and sky.any():                    # pull back from the skyline so no halo of sky clings to it
        from scipy import ndimage
        sky = ndimage.binary_erosion(sky, np.ones((1, 2 * feather + 1), bool))
        sky[0] = a[0, :, 3] >= 128
    a[sky, 3] = 0
    return a, float(sky.mean())


class ClearSky:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "step": ("FLOAT", {"default": 14.0, "min": 1.0, "max": 60.0, "step": 0.5, "tooltip": "How far a pixel may sit from the one above it, in Lab, and still count as sky. A dusk gradient moves 1-3 per row; a treeline moves 30. Raise it if sky survives, lower it if the skyline is eaten."}),
            "feather": ("INT", {"default": 2, "min": 0, "max": 16, "tooltip": "Pull the cleared area back from the skyline by this many pixels, so no fringe of sky clings to it."}),
            "keep": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.5, "step": 0.01, "tooltip": "Give up and change nothing unless at least this much of the layer survives. A far layer is mostly sky and a near layer shows sky between its trunks, so neither amount nor shape is a fault - only clearing everything is."}),
        }}

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("image", "cleared")
    FUNCTION = "run"
    CATEGORY = "comfyui_workflows/pixel"
    DESCRIPTION = ("Make a parallax layer's sky transparent down to its skyline, whatever colour it is. Run it on "
                   "the drawing, before Pixelize: afterwards the gradient is flat bands and each edge reads as a "
                   "skyline.")

    def run(self, images, step, feather, keep=0.02):
        out, frac = [], 0.0
        for f in to_rgba_np(images):
            img, got = clear_sky(f, step, feather, keep)
            out.append(img)
            frac = got
        return (to_image_tensor(pad_to_same_size(out)), frac)


class CropToStrip:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "height": ("INT", {"default": 576, "min": 8, "max": 8192, "tooltip": "How many rows to keep. The full width is always kept."}),
            "anchor": (["auto", "top", "centre", "bottom"], {"tooltip": "Which rows. auto = the band with the most detail in it, which for a landscape is the horizon and the ground rather than empty sky; where the skyline lands moves from seed to seed, so a fixed edge misses it."}),
        }}

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image", "top")
    FUNCTION = "run"
    CATEGORY = "comfyui_workflows/pixel"
    DESCRIPTION = ("Crop a full-width band out of a taller image, for a parallax layer. Z-Image stops composing a "
                   "scene that spans the width past about 2.4:1, so a layer is drawn on a squarer canvas and the "
                   "band is taken from it.")

    def run(self, images, height, anchor):
        out, tops = [], 0
        for f in to_rgba_np(images):
            H = f.shape[0]
            h = min(int(height), H)
            top = {"top": 0, "bottom": H - h, "centre": (H - h) // 2}.get(anchor, busiest_band(f, h))
            out.append(f[top:top + h])
            tops = int(top)
        return (to_image_tensor(pad_to_same_size(out)), tops)


class MakeTileable:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "tile_size": ("INT", {"default": 32, "min": 4, "max": 4096, "tooltip": "Tile width in art pixels (and its height, unless axis is x)."}),
            "overlap": ("INT", {"default": 0, "min": 0, "max": 256, "tooltip": "Seam search width in art pixels (0 = tile/4)."}),
            "preview_repeat": ("INT", {"default": 3, "min": 1, "max": 8, "tooltip": "Preview = the tile repeated N x N."}),
            "preview_scale": ("INT", {"default": 4, "min": 1, "max": 16}),
        }, "optional": {
            "axis": (["both", "x"], {"tooltip": "both = a square tile that wraps every way (terrain, ground). x = a strip that repeats sideways only, for a parallax layer: wrapping it vertically would fold the sky into the ground."}),
            "tile_height": ("INT", {"default": 0, "min": 0, "max": 4096, "tooltip": "axis x only: the strip's height in art pixels (0 = the whole input height)."}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("tile", "preview")
    FUNCTION = "run"
    CATEGORY = "comfyui_workflows/pixel"
    DESCRIPTION = "Cut a seamlessly repeating tile or sideways-repeating strip out of a pixel-art texture (exact pixels, minimum-error wrap seams)."

    def run(self, images, tile_size, overlap, preview_repeat, preview_scale, axis="both", tile_height=0):
        tiles, prevs = [], []
        for f in to_rgba_np(images):
            f = f.copy()
            if axis != "x":           # a layer keeps its alpha: the sky above its skyline is meant to be empty
                f[..., 3] = 255
            tile = make_tileable(f, tile_size, overlap, height=tile_height, axis=axis)
            tiles.append(tile)
            # a strip is previewed side by side, not in a grid: stacking it vertically shows a join it never makes
            reps = (1, preview_repeat, 1) if axis == "x" else (preview_repeat, preview_repeat, 1)
            prevs.append(nearest_upscale(np.tile(tile, reps), preview_scale))
        return (to_image_tensor(tiles), to_image_tensor(prevs))


# ----------------------------------------------------------------------------- sprite sheets
def save_gif(frames_rgba, path, duration):
    """Animated GIF with 1-bit transparency (pixel art has hard alpha, so nothing is lost)."""
    gif = []
    for f in frames_rgba:
        p = f.convert("RGB").quantize(colors=255, dither=Image.Dither.NONE)
        p.paste(255, mask=f.getchannel("A").point(lambda a: 255 if a == 0 else 0))
        gif.append(p)
    gif[0].save(path, save_all=True, append_images=gif[1:], duration=duration, loop=0, disposal=2, transparency=255,
                optimize=False)


class SaveSpriteSheet:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "frames": ("IMAGE",),
            "filename_prefix": ("STRING", {"default": "pixel/sprite"}),
            "frame_step": ("INT", {"default": 1, "min": 1, "max": 64, "tooltip": "Use every Nth frame (video models make many in-between frames)."}),
            "max_frames": ("INT", {"default": 0, "min": 0, "max": 1024, "tooltip": "Keep at most this many frames (0 = all)."}),
            "loop_trim": ("BOOLEAN", {"default": True, "tooltip": "Drop trailing frames identical to the first frame (seamless loops)."}),
            "columns": ("INT", {"default": 0, "min": 0, "max": 256, "tooltip": "0 = all frames in one row."}),
            "spacing": ("INT", {"default": 0, "min": 0, "max": 64}),
            "trim": ("BOOLEAN", {"default": True, "tooltip": "Crop all frames to their combined bounding box (same size for every frame)."}),
            "fps": ("INT", {"default": 10, "min": 1, "max": 60}),
            "tag": ("STRING", {"default": "anim", "tooltip": "Animation name stored in the Aseprite JSON (frameTags)."}),
            "preview_scale": ("INT", {"default": 4, "min": 1, "max": 16}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "INT")
    RETURN_NAMES = ("sheet", "frames", "info", "fps")
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "comfyui_workflows/pixel"
    DESCRIPTION = "Save frames as a sprite sheet PNG + Aseprite-compatible JSON + animated WebP and GIF previews."

    def run(self, frames, filename_prefix, frame_step, max_frames, loop_trim, columns, spacing, trim, fps, tag, preview_scale):
        import folder_paths
        fr = to_rgba_np(frames)[::frame_step]
        if loop_trim:                     # closed loops end on (almost) the first frame: it would play twice at the seam
            while len(fr) > 2 and frame_difference(fr[-1], fr[0]) < 0.02:
                fr.pop()
        if max_frames:
            fr = fr[:max_frames]
        if trim:
            union = np.zeros(fr[0].shape[:2], bool)
            for f in fr:
                union |= f[..., 3] > 0
            x0, y0, x1, y1 = subject_bbox(union)
            fr = [f[y0:y1, x0:x1] for f in fr]
        h, w = fr[0].shape[:2]
        n = len(fr)
        cols = columns or n
        rows = math.ceil(n / cols)
        sheet = np.zeros((rows * h + (rows - 1) * spacing, cols * w + (cols - 1) * spacing, 4), np.uint8)
        frames_json = {}
        for i, f in enumerate(fr):
            r, c = divmod(i, cols)
            x, y = c * (w + spacing), r * (h + spacing)
            sheet[y:y + h, x:x + w] = f
            frames_json[str(i)] = {"frame": {"x": x, "y": y, "w": w, "h": h}, "rotated": False, "trimmed": False,
                                          "spriteSourceSize": {"x": 0, "y": 0, "w": w, "h": h},
                                          "sourceSize": {"w": w, "h": h}, "duration": round(1000 / fps)}
        out_dir, name, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory(), w, h)
        base = f"{name}_{counter:05}_"
        Image.fromarray(sheet).save(os.path.join(out_dir, base + ".png"))
        meta = {"frames": frames_json, "meta": {"app": "comfyui_workflows", "version": "1.0", "image": base + ".png",
                                                 "format": "RGBA8888", "size": {"w": sheet.shape[1], "h": sheet.shape[0]},
                                                 "scale": "1", "frameTags": [{"name": tag, "from": 0, "to": n - 1, "direction": "forward"}]}}
        with open(os.path.join(out_dir, base + ".json"), "w") as fh:
            json.dump(meta, fh, indent=1)
        anim = [Image.fromarray(nearest_upscale(f, preview_scale), "RGBA") for f in fr]
        anim[0].save(os.path.join(out_dir, base + ".webp"), save_all=True, append_images=anim[1:],
                     duration=round(1000 / fps), loop=0, lossless=True)
        save_gif(anim, os.path.join(out_dir, base + ".gif"), round(1000 / fps))
        prev = nearest_upscale(sheet, preview_scale)
        Image.fromarray(prev).save(os.path.join(out_dir, base + "_preview.png"))
        info = f"{n} frames of {w}x{h} -> {base}.png ({sheet.shape[1]}x{sheet.shape[0]}), .json, .webp, .gif"
        ui = {"images": [{"filename": base + ".webp", "subfolder": subfolder, "type": "output"},
                         {"filename": base + "_preview.png", "subfolder": subfolder, "type": "output"}],
              "animated": (True,)}                              # the node shows the animation playing
        return {"ui": ui, "result": (to_image_tensor([sheet]), to_image_tensor(fr), info, fps)}


def frame_difference(a, b):
    """Share of pixels that differ between two RGBA frames (alpha or colour), 0..1; different sizes count as 1."""
    if a.shape != b.shape:
        return 1.0
    vis = (a[..., 3] > 0) | (b[..., 3] > 0)
    if not vis.any():
        return 0.0
    diff = (a[..., 3] > 0) != (b[..., 3] > 0)
    diff |= np.abs(a[..., :3].astype(np.int16) - b[..., :3].astype(np.int16)).max(-1) > 24
    return float((diff & vis).sum()) / float(vis.sum())


def checkerboard(h, w, cell=8, a=(205, 205, 205), b=(245, 245, 245)):
    yy, xx = np.mgrid[0:h, 0:w]
    board = ((yy // cell + xx // cell) % 2).astype(bool)
    out = np.empty((h, w, 4), np.uint8)
    out[..., :3] = np.where(board[..., None], np.array(a, np.uint8), np.array(b, np.uint8))
    out[..., 3] = 255
    return out


class SpriteAnimationPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "frames": ("IMAGE",),
            "fps": ("INT", {"default": 8, "min": 1, "max": 60, "tooltip": "Playback speed (connected to the sprite sheet's fps in the workflows)."}),
            "scale": ("INT", {"default": 6, "min": 1, "max": 32, "tooltip": "Nearest-neighbour zoom of the preview."}),
            "background": (["checker", "white", "black", "grey", "transparent"], {"tooltip": "What shows behind transparent pixels."}),
            "playback": (["loop", "ping-pong"], {"tooltip": "ping-pong plays forward then backward (preview only; the sheet is unchanged)."}),
        }}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "comfyui_workflows/pixel"
    DESCRIPTION = "Play sprite frames as a looping animation in the node (pixel-perfect zoom, transparency background)."

    def run(self, frames, fps, scale, background, playback):
        import folder_paths
        fr = pad_to_same_size(to_rgba_np(frames))
        if playback == "ping-pong" and len(fr) > 2:
            fr = fr + fr[-2:0:-1]
        h, w = fr[0].shape[0] * scale, fr[0].shape[1] * scale
        bg = {"checker": checkerboard(h, w, max(4, scale * 2)), "white": None, "black": None, "grey": None,
              "transparent": None}[background]
        flat = {"white": (255, 255, 255), "black": (0, 0, 0), "grey": (128, 128, 128)}.get(background)
        anim = []
        for f in fr:
            big = Image.fromarray(nearest_upscale(f, scale), "RGBA")
            if background == "transparent":
                anim.append(big)
                continue
            base = Image.fromarray(bg) if bg is not None else Image.new("RGBA", (w, h), flat + (255,))
            base.alpha_composite(big)
            anim.append(base)
        out_dir = folder_paths.get_temp_directory()
        os.makedirs(out_dir, exist_ok=True)
        name = f"sprite_preview_{os.getpid()}_{abs(hash((len(fr), w, h, fps, background, playback, fr[0].tobytes()[:4096]))) % 10**10}.webp"
        anim[0].save(os.path.join(out_dir, name), save_all=True, append_images=anim[1:], duration=round(1000 / fps),
                     loop=0, lossless=True)
        return {"ui": {"images": [{"filename": name, "subfolder": "", "type": "temp"}], "animated": (True,)}}


# ----------------------------------------------------------------------------- directional views
MIRRORS = {"front-right": "front-left", "right": "left", "back-right": "back-left",
           "front-left": "front-right", "left": "right", "back-left": "back-right"}
CLOCKWISE = ["front", "front-right", "right", "back-right", "back", "back-left", "left", "front-left"]


def direction_set(frames, names, mirror):
    """Name the views and add mirrored opposite sides (right -> left ...). Returns [(name, rgba)] in clockwise
    order (front, front-right, right, ..., front-left) for known names, unknown names after them."""
    views = dict(zip(names, frames))
    if mirror:
        for n, f in list(views.items()):
            m = MIRRORS.get(n)
            if m and m not in views:
                views[m] = f[:, ::-1].copy()
    known = [(n, views[n]) for n in CLOCKWISE if n in views]
    return known + [(n, f) for n, f in views.items() if n not in CLOCKWISE]


class SaveDirections:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "filename_prefix": ("STRING", {"default": "pixel/directions"}),
            "names": ("STRING", {"default": "front, front-right, right, back-right, back", "tooltip": "Direction name of each input image, in order (comma separated)."}),
            "mirror": ("BOOLEAN", {"default": True, "tooltip": "Add the opposite side by mirroring: front-right -> front-left, right -> left, back-right -> back-left (8 directions from 5 views)."}),
            "spacing": ("INT", {"default": 0, "min": 0, "max": 64}),
            "preview_scale": ("INT", {"default": 4, "min": 1, "max": 16}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("sheet", "directions", "info")
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "comfyui_workflows/pixel"
    DESCRIPTION = "Save directional views (front, right, back ...) as one PNG per direction + a sheet, mirroring the missing side."

    def run(self, images, filename_prefix, names, mirror, spacing, preview_scale):
        import folder_paths
        fr = pad_to_same_size(to_rgba_np(images))
        nm = [n.strip().lower() for n in names.split(",") if n.strip()]
        nm += [f"view{i}" for i in range(len(nm), len(fr))]
        views = direction_set(fr, nm[:len(fr)], mirror)
        h, w = views[0][1].shape[:2]
        sheet = np.zeros((h, len(views) * w + (len(views) - 1) * spacing, 4), np.uint8)
        frames_json = {}
        for i, (n, f) in enumerate(views):
            x = i * (w + spacing)
            sheet[:, x:x + w] = f
            frames_json[n] = {"frame": {"x": x, "y": 0, "w": w, "h": h}, "rotated": False, "trimmed": False,
                              "spriteSourceSize": {"x": 0, "y": 0, "w": w, "h": h}, "sourceSize": {"w": w, "h": h},
                              "duration": 100}
        out_dir, name, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory(), w, h)
        base = f"{name}_{counter:05}_"
        for n, f in views:
            Image.fromarray(f).save(os.path.join(out_dir, f"{base}{n}.png"))
        Image.fromarray(sheet).save(os.path.join(out_dir, base + ".png"))
        meta = {"frames": frames_json, "meta": {"app": "comfyui_workflows", "version": "1.0", "image": base + ".png",
                                                 "format": "RGBA8888", "size": {"w": sheet.shape[1], "h": h}, "scale": "1",
                                                 "frameTags": [{"name": n, "from": i, "to": i, "direction": "forward"}
                                                               for i, (n, _) in enumerate(views)]}}
        with open(os.path.join(out_dir, base + ".json"), "w") as fh:
            json.dump(meta, fh, indent=1)
        Image.fromarray(nearest_upscale(sheet, preview_scale)).save(os.path.join(out_dir, base + "_preview.png"))
        info = f"{len(views)} directions of {w}x{h}: " + ", ".join(n for n, _ in views) + f" -> {base}*.png, .json"
        ui = {"images": [{"filename": base + "_preview.png", "subfolder": subfolder, "type": "output"}]}
        return {"ui": ui, "result": (to_image_tensor([sheet]), to_image_tensor([f for _, f in views]), info)}


# ----------------------------------------------------------------------------- checks
def check_pixel_art(frames, max_colors=32, expected_height=0, consistent_height=True):
    lines, ok = [], True
    alphas = np.concatenate([f[..., 3].ravel() for f in frames])
    semi = int(((alphas > 0) & (alphas < 255)).sum())
    pal = np.unique(np.concatenate([f[f[..., 3] > 0][:, :3] for f in frames if (f[..., 3] > 0).any()] or [np.zeros((0, 3), np.uint8)]), axis=0)
    scales = [detect_scale(f)[0] for f in frames[:8]]
    heights = [subject_bbox(f[..., 3] > 0)[3] - subject_bbox(f[..., 3] > 0)[1] for f in frames]
    bodies = [body_bbox(f[..., 3] > 0)[3] - body_bbox(f[..., 3] > 0)[1] for f in frames]
    strays = sum(int((cleanup_strays(f, 1e9) != f).any(-1).sum()) for f in frames[:8])

    def chk(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        lines.append(("PASS  " if cond else "FAIL  ") + msg)

    chk(semi == 0, f"hard alpha: {semi} semi-transparent pixels")
    chk(max(scales) == 1, f"native resolution: detected upscale x{max(scales)} (must be 1)")
    chk(len(pal) <= max_colors, f"palette: {len(pal)} colours across {len(frames)} frame(s) (max {max_colors})")
    if consistent_height and len(frames) > 1:
        def steady(v):
            return max(v) - min(v) <= max(2, 0.06 * max(v))
        # either measure may be thrown off: props swinging (bbox), or tiny sprites where the opening eats limbs (body)
        chk(steady(bodies) or steady(heights),
            f"consistent height: body {min(bodies)}-{max(bodies)} px, with props {min(heights)}-{max(heights)} px")
    if expected_height:
        h = min((heights[0], bodies[0]), key=lambda v: abs(v - expected_height))
        chk(abs(h - expected_height) <= 1, f"subject height {h} px (expected {expected_height})")
    lines.append(f"info  size {frames[0].shape[1]}x{frames[0].shape[0]}, {strays} isolated pixels (first 8 frames)")
    if len(frames) > 1:
        lines.append(f"info  colour popping {flicker_rate(frames):.2f}% of pixels per frame (similar colour switching)")
    return ("PASS" if ok else "FAIL") + "\n" + "\n".join(lines), ok


class PixelArtCheck:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "max_colors": ("INT", {"default": 32, "min": 2, "max": 256}),
            "expected_height": ("INT", {"default": 0, "min": 0, "max": 2048, "tooltip": "Expected subject height in art pixels (0 = don't check)."}),
            "consistent_height": ("BOOLEAN", {"default": True, "tooltip": "Require the same height in every frame/view. Turn off for action clips where the pose changes the height (jumps, crouches, raised arms)."}),
        }}

    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("report", "passed")
    FUNCTION = "run"
    CATEGORY = "comfyui_workflows/pixel"
    DESCRIPTION = "Game-ready check: hard alpha, native resolution, palette size, consistent height across frames."

    def run(self, images, max_colors, expected_height, consistent_height):
        report, ok = check_pixel_art(to_rgba_np(images), max_colors, expected_height, consistent_height)
        return (report, ok)
