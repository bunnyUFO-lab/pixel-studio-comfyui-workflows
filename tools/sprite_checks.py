#!/usr/bin/env python3
"""First automatic checks for pixel-art sprites and sprite sheets (CPU only: numpy + Pillow).

Scores one image, a sprite sheet cut into a grid of frames, or a folder of frame images, with no person
looking and no model. The gate ideas (ground line, zoom, clean background, loop closes) follow the method of
agent-sprite-forge (MIT); the code here is our own.

    python tools/sprite_checks.py <image|sheet|folder> [--scale N] [--frames COLSxROWS] [--base PATH]
                                      [--background auto|none|#rrggbb] [--json]

Scores (see SCORE_HELP for the meaning and the good range of each):
    grid_purity, grid_scale, grid_stability, colour_flicker, same_character, palette_size, gates, overall.
"""
import argparse
import json
import os
import re
import sys

import numpy as np
from PIL import Image

IMAGE_EXT = (".png", ".webp", ".bmp", ".gif", ".jpg", ".jpeg")
GRID_ACCEPT = 0.97          # share of colour changes that must sit on the grid lines to call a grid found
PURITY_GOOD = 0.97          # public trainer threshold for grid purity
MAX_SCALE = 32
NATIVE_MAX_COLOURS = 64     # a sprite with no grid but at most this many colours counts as native resolution (scale 1)
BACKGROUND_SHARE = 0.9      # share of the border that must be one colour to call it a flat background

SCORE_HELP = {
    "grid_purity": "(scale 1 = native resolution: every block is one pixel, so purity is 1.0 by definition and palette size is the check) mean share of each sprite block's pixels equal to the block's main colour; good >= 0.97 (1.0 = clean grid)",
    "grid_scale": "detected pixel size in source pixels (1 = native resolution); null when no grid was found",
    "grid_stability": "share of frames whose grid (scale + offset) equals the sheet's most common grid; good = 1.0",
    "colour_flicker": "share of pixels that change colour between consecutive frames where the shape stays put; good <= 0.10",
    "same_character": "palette_overlap = share of a frame's pixels using a base-sprite colour (good >= 0.90); size_change = "
                      "largest relative height or width change from the base (good <= 0.10)",
    "palette_size": "distinct opaque colours after snapping to the grid, per frame and for the sheet; good <= 32 for the sheet",
    "gates": "ground_shift (px, good <= 1), zoom_variation (good <= 0.10), background_clean (good >= 0.98), "
             "loop_gap vs mean step (good: loop_ok true)",
    "overall": "mean of 0..1 quality values of everything above; only for ranking sheets against each other",
}
LEFT_OUT = [
    "identity by a learned model (face or body embedding): needs a model download",
    "turn-around check (front/side/back views agree)",
    "timing and spacing checks (arcs, holds): task ss-m11t6",
    "end-pose check beyond the loop-closes gate",
    "stray-pixel and disconnected-island count",
    "foot-contact detection (only the lowest-row ground line is checked)",
]


# ----------------------------------------------------------------------------- loading
def _natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def load_rgba(path):
    return np.array(Image.open(path).convert("RGBA"))


def split_sheet(rgba, cols, rows):
    h, w = rgba.shape[:2]
    if w % cols or h % rows:
        raise ValueError(f"sheet {w}x{h} does not divide into {cols}x{rows} frames")
    fw, fh = w // cols, h // rows
    return [rgba[r * fh:(r + 1) * fh, c * fw:(c + 1) * fw] for r in range(rows) for c in range(cols)]


def parse_frames(text):
    m = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", str(text))
    if not m or int(m.group(1)) < 1 or int(m.group(2)) < 1:
        raise ValueError(f"--frames must look like 4x1 (columns x rows, both at least 1), got {text!r}")
    return int(m.group(1)), int(m.group(2))


def load_frames(path, frames=None):
    """Return (list of RGBA arrays, source kind)."""
    if os.path.isdir(path):
        names = sorted((n for n in os.listdir(path) if n.lower().endswith(IMAGE_EXT)), key=_natural_key)
        if not names:
            raise ValueError(f"no images in folder {path}")
        return [load_rgba(os.path.join(path, n)) for n in names], "folder"
    img = load_rgba(path)
    if frames:
        cols, rows = parse_frames(frames)
        return split_sheet(img, cols, rows), "sheet"
    return [img], "image"


# ----------------------------------------------------------------------------- grid
def pack(rgba):
    """uint32 key per pixel; fully transparent pixels all map to 0."""
    a = rgba[..., 3].astype(np.uint32)
    k = (rgba[..., 0].astype(np.uint32) << 24) | (rgba[..., 1].astype(np.uint32) << 16) | \
        (rgba[..., 2].astype(np.uint32) << 8) | a
    k[a == 0] = 0
    return k


def parse_background(text):
    """'auto' | 'none' | '#rrggbb' -> ('auto', None) | ('none', None) | ('colour', key)."""
    t = (text or "auto").strip().lower()
    if t in ("auto", "none"):
        return t, None
    m = re.fullmatch(r"#?([0-9a-f]{6})", t)
    if not m:
        raise ValueError(f"--background must be auto, none or #rrggbb, got {text!r}")
    v = int(m.group(1), 16)
    return "colour", (v << 8) | 255


def _flood_from_border(match):
    """Pixels of `match` (bool array) connected (4-way) to a border pixel that is in `match`."""
    reach = np.zeros_like(match)
    reach[0], reach[-1], reach[:, 0], reach[:, -1] = match[0], match[-1], match[:, 0], match[:, -1]
    while True:
        new = reach.copy()
        new[1:] |= reach[:-1]
        new[:-1] |= reach[1:]
        new[:, 1:] |= reach[:, :-1]
        new[:, :-1] |= reach[:, 1:]
        new &= match
        if (new == reach).all():
            return reach
        reach = new


def remove_background(keys, mode="auto", colour=None):
    """Make a flat opaque background transparent (key 0). Returns (keys, background colour key or None).

    auto: the border's main colour is the background when it covers >= BACKGROUND_SHARE of the border and is not
    transparent already. colour: that colour. none: nothing. Only pixels of that colour connected to the border
    are removed, so the same colour inside the sprite (a white eye on a white background) stays.
    """
    if mode == "none":
        return keys, None
    if mode == "auto":
        border = np.concatenate([keys[0], keys[-1], keys[:, 0], keys[:, -1]])
        bg = int(_mode(border))
        if (bg & 255) == 0 or float((border == bg).mean()) < BACKGROUND_SHARE:
            return keys, None
    else:
        bg = int(colour)
    flood = _flood_from_border(keys == bg)
    if not flood.any():
        return keys, None
    out = keys.copy()
    out[flood] = 0
    return out, bg


def colour_hex(key):
    return "#%02x%02x%02x" % ((key >> 24) & 255, (key >> 16) & 255, (key >> 8) & 255)


def unpack(keys):
    out = np.empty(keys.shape + (4,), np.uint8)
    out[..., 0] = keys >> 24
    out[..., 1] = (keys >> 16) & 255
    out[..., 2] = (keys >> 8) & 255
    out[..., 3] = keys & 255
    return out


def change_positions(keys):
    cx = np.nonzero((keys[:, 1:] != keys[:, :-1]).any(axis=0))[0] + 1
    cy = np.nonzero((keys[1:, :] != keys[:-1, :]).any(axis=1))[0] + 1
    return cx, cy


def grid_for_scale(cx, cy, s):
    """Best offset for pixel size s and the share of colour changes that sit on its grid lines."""
    if s == 1:
        return 0, 0, 1.0
    ox = int(np.bincount(cx % s, minlength=s).argmax()) if len(cx) else 0
    oy = int(np.bincount(cy % s, minlength=s).argmax()) if len(cy) else 0
    sx = float(np.mean(cx % s == ox)) if len(cx) else 1.0
    sy = float(np.mean(cy % s == oy)) if len(cy) else 1.0
    return ox, oy, 0.5 * (sx + sy)


def detect_grid(keys, scale=None, max_scale=MAX_SCALE):
    """Return dict(scale, ox, oy, score, found). With `scale` given, only the offset is searched."""
    cx, cy = change_positions(keys)
    if scale is not None:
        ox, oy, sc = grid_for_scale(cx, cy, scale)
        return dict(scale=scale, ox=ox, oy=oy, score=sc, found=sc >= GRID_ACCEPT)
    native = dict(scale=1, ox=0, oy=0, score=1.0, found=True) if is_native(keys) else         dict(scale=None, ox=0, oy=0, score=0.0, found=False)
    if len(cx) < 2 or len(cy) < 2:
        return native
    for s in range(max_scale, 1, -1):
        if s * 4 > min(keys.shape):
            continue
        ox, oy, sc = grid_for_scale(cx, cy, s)
        if sc >= GRID_ACCEPT:
            return dict(scale=s, ox=ox, oy=oy, score=sc, found=True)
    return native


def is_native(keys):
    """No pixel-size grid, but a small palette: native resolution (scale 1). Blur and noise have hundreds of colours."""
    k = keys[(keys & 255) > 0]
    return bool(len(k)) and len(np.unique(k)) <= NATIVE_MAX_COLOURS


def block_mode(blocks):
    """blocks (nb, n) uint32 -> (mode key, mode count) per row."""
    nb, n = blocks.shape
    srt = np.sort(blocks, axis=1)
    new = np.ones(srt.shape, bool)
    new[:, 1:] = srt[:, 1:] != srt[:, :-1]
    runid = np.cumsum(new, axis=1) - 1
    counts = np.bincount((np.arange(nb)[:, None] * n + runid).ravel(), minlength=nb * n).reshape(nb, n)
    best = counts.argmax(axis=1)
    first = np.argmax(runid == best[:, None], axis=1)
    rows = np.arange(nb)
    return srt[rows, first], counts[rows, best]


def snap(keys, s, ox, oy):
    """Snap to the grid. Returns (native key array, purity over blocks that touch the sprite, native origin)."""
    if s == 1:
        return keys, 1.0
    h, w = keys.shape
    # leading partial blocks are dropped; the grid starts at the first line at or after the offset
    x0 = ox % s
    y0 = oy % s
    nx, ny = (w - x0) // s, (h - y0) // s
    if nx < 1 or ny < 1:
        return keys, 0.0
    crop = keys[y0:y0 + ny * s, x0:x0 + nx * s]
    blocks = crop.reshape(ny, s, nx, s).transpose(0, 2, 1, 3).reshape(ny * nx, s * s)
    mode, cnt = block_mode(blocks)
    touch = ((blocks & 255) > 0).any(axis=1)
    purity = float((cnt[touch] / (s * s)).mean()) if touch.any() else 1.0
    return mode.reshape(ny, nx), purity


def bbox(keys):
    m = (keys & 255) > 0
    if not m.any():
        return None
    ys = np.nonzero(m.any(axis=1))[0]
    xs = np.nonzero(m.any(axis=0))[0]
    return int(ys[0]), int(ys[-1]), int(xs[0]), int(xs[-1])


def pad_native(arrs):
    """Pad native key arrays to one canvas, bottom-centre anchored."""
    H = max(a.shape[0] for a in arrs)
    W = max(a.shape[1] for a in arrs)
    out = []
    for a in arrs:
        p = np.zeros((H, W), np.uint32)
        x = (W - a.shape[1]) // 2
        p[H - a.shape[0]:, x:x + a.shape[1]] = a
        out.append(p)
    return out


# ----------------------------------------------------------------------------- scores
def _clip(x):
    return float(max(0.0, min(1.0, x)))


def _opaque_colours(keys):
    k = keys[(keys & 255) > 0]
    return set(np.unique(k).tolist())


def _background_report(mode, colour, used):
    """JSON block saying which background was used. used = list of background keys (None = transparent) per frame."""
    found = [u for u in used if u is not None]
    if mode == "none":
        what = "none"
    elif found:
        what = colour_hex(max(set(found), key=found.count))
    else:
        what = "transparent"
    return {"mode": mode, "colour": what, "frames_with_opaque_background": len(found)}


def _check_scale(scale, keys):
    if scale is None:
        return
    if scale < 1:
        raise ValueError(f"--scale must be at least 1, got {scale}")
    h, w = keys[0].shape
    if scale > min(h, w):
        raise ValueError(f"--scale {scale} is larger than the frame ({w}x{h}); a pixel cannot be bigger than the image")
    boxes = [bbox(k) for k in keys]
    boxes = [b for b in boxes if b is not None]
    if boxes and scale > max(max(b[1] - b[0] + 1, b[3] - b[2] + 1) for b in boxes):
        raise ValueError(f"--scale {scale} is larger than the sprite itself; a pixel cannot be bigger than the sprite")


def _scale_warnings(scale, auto_scale, keys):
    out = []
    if scale is None:
        return out
    if auto_scale is not None and auto_scale != scale:
        rel = ""
        if scale > 1 and auto_scale % scale == 0:
            rel = f" ({scale} divides {auto_scale}: every {scale}x{scale} block is then pure, so purity reads 1.0 whatever the art)"
        elif scale > 1 and scale % auto_scale == 0:
            rel = f" ({scale} is a multiple of {auto_scale}: blocks straddle several art pixels, so purity drops)"
        out.append(f"--scale {scale} given but the grid detected in the image has scale {auto_scale}{rel}; "
                   f"the purity number is for scale {scale}")
    elif auto_scale is None and scale == 1:
        allk = np.concatenate([k[(k & 255) > 0] for k in keys])
        if len(np.unique(allk)) > NATIVE_MAX_COLOURS:
            out.append(f"--scale 1 given but the image has {len(np.unique(allk))} colours and no grid; "
                       f"purity 1.0 at scale 1 says nothing here")
    return out


def score_frames(frames, scale=None, base=None, background="auto"):
    bmode, bcol = parse_background(background)
    packed = [remove_background(pack(f), bmode, bcol) for f in frames]
    keys = [p[0] for p in packed]
    n = len(keys)
    _check_scale(scale, keys)
    warnings = []
    bg_report = _background_report(bmode, bcol, [p[1] for p in packed])
    if bmode == "colour" and not bg_report["frames_with_opaque_background"]:
        warnings.append(f"--background {background} given but no frame has that colour on its border; nothing was removed")
    if bg_report["frames_with_opaque_background"]:
        diff = {p[1] for p in packed if p[1] is not None}
        if len(diff) > 1:
            warnings.append("frames have different background colours: " + ", ".join(sorted(colour_hex(k) for k in diff)))
        if bg_report["frames_with_opaque_background"] < n:
            warnings.append("only some frames have an opaque background; the others are scored as transparent")
    dets = [detect_grid(k, scale) for k in keys]
    auto = dets if scale is None else [detect_grid(k) for k in keys]
    auto_found = [d["scale"] for d in auto if d["found"]]
    detected = None
    if auto_found:
        vals, counts = np.unique(auto_found, return_counts=True)
        detected = int(vals[counts.argmax()])
    warnings += _scale_warnings(scale, detected, keys)

    # sheet grid = most common (scale, offset-on-grid) among frames that found a grid
    found = [d for d in dets if d["found"]]
    if scale is not None:
        sheet_scale = scale
    elif found:
        vals, counts = np.unique([d["scale"] for d in found], return_counts=True)
        sheet_scale = int(vals[counts.argmax()])
    else:
        sheet_scale = None

    res = {"frames": n, "frame_size": [int(frames[0].shape[1]), int(frames[0].shape[0])],
           "background": bg_report, "detected_scale": detected, "scale_given": scale}
    q = {}

    # --- grid purity and scale
    if sheet_scale is None:
        res["grid_scale"] = None
        res["grid_found"] = False
        res["grid_purity"] = None
        res["grid_purity_per_frame"] = []
        res["grid_stability"] = 0.0
        snapped = keys  # nothing to snap to; later checks run on the source pixels
        q["grid_purity"] = 0.0
        q["grid_stability"] = 0.0
        sheet_scale_eff = 1
    else:
        per = []
        snapped = []
        offs = []
        for k, d in zip(keys, dets):
            if d["scale"] != sheet_scale:
                d = detect_grid(k, sheet_scale)  # own best offset at the sheet's scale
            nat, pur = snap(k, sheet_scale, d["ox"], d["oy"])
            snapped.append(nat)
            per.append(pur)
            offs.append((d["ox"] % sheet_scale, d["oy"] % sheet_scale, d["found"]))
        res["grid_scale"] = sheet_scale
        if sheet_scale == 1:
            res["grid_note"] = "scale 1 = native resolution: purity is 1.0 by definition (each block is one pixel); judge it by palette_size"
        res["grid_found"] = bool(len(found))
        res["grid_purity"] = round(float(np.mean(per)), 4)
        res["grid_purity_per_frame"] = [round(p, 4) for p in per]
        modal = max(set(offs), key=offs.count) if offs else None
        ok_offs = [o[:2] for o in offs if o[2]]
        if ok_offs:
            modal = max(set(ok_offs), key=ok_offs.count)
            stab = sum(1 for o in offs if o[2] and o[:2] == modal) / n
        else:
            stab = 0.0
        res["grid_stability"] = round(stab, 4)
        res["grid_offset"] = list(modal[:2]) if modal else None
        q["grid_purity"] = _clip((res["grid_purity"] - 0.85) / 0.15)
        q["grid_stability"] = stab
        sheet_scale_eff = sheet_scale

    nat = pad_native(snapped)

    # --- palette size (after snapping)
    cols = [_opaque_colours(a) for a in snapped]
    allc = set().union(*cols)
    res["palette_size"] = {"per_frame": [len(c) for c in cols], "max_frame": max(len(c) for c in cols), "sheet": len(allc)}
    q["palette_size"] = _clip(32 / max(1, len(allc)))

    # --- colour flicker: pixels opaque in both frames, only for frame pairs whose silhouette stays put
    if n > 1:
        changed = total = used = 0
        for a, b in zip(nat[:-1], nat[1:]):
            ma, mb = (a & 255) > 0, (b & 255) > 0
            both = ma & mb
            union = (ma | mb).sum()
            if union == 0 or both.sum() / union < 0.8:
                continue
            used += 1
            changed += int((a[both] != b[both]).sum())
            total += int(both.sum())
        res["colour_flicker"] = round(changed / total, 4) if total else None
        res["colour_flicker_pairs_used"] = used
        q["colour_flicker"] = _clip(1 - (res["colour_flicker"] or 0.0) / 0.2) if total else 1.0
    else:
        res["colour_flicker"] = None
        res["colour_flicker_pairs_used"] = 0

    # --- same character: drift from the base (first frame, or --base image)
    if base is not None:
        bk = remove_background(pack(base), bmode, bcol)[0]
        bd = detect_grid(bk, sheet_scale)
        bnat = snap(bk, sheet_scale, bd["ox"], bd["oy"])[0] if sheet_scale else bk
        base_cols = _opaque_colours(bnat)
        bbx = bbox(bk)
        compare = list(range(n))
    else:
        base_cols = cols[0]
        bbx = bbox(keys[0])
        compare = list(range(1, n))
    if compare and bbx is not None:
        bh, bw = bbx[1] - bbx[0] + 1, bbx[3] - bbx[2] + 1
        overlaps, size_changes = [], []
        for i in compare:
            a = snapped[i]
            op = a[(a & 255) > 0]
            overlaps.append(float(np.isin(op, list(base_cols)).mean()) if len(op) else 0.0)
            fb = bbox(keys[i])
            if fb is None:
                size_changes.append(1.0)
            else:
                size_changes.append(max(abs((fb[1] - fb[0] + 1) - bh) / bh, abs((fb[3] - fb[2] + 1) - bw) / bw))
        res["same_character"] = {
            "base": "image" if base is not None else "frame 1",
            "palette_overlap_min": round(min(overlaps), 4), "palette_overlap_mean": round(float(np.mean(overlaps)), 4),
            "size_change_max": round(max(size_changes), 4), "size_change_mean": round(float(np.mean(size_changes)), 4),
        }
        q["same_character"] = _clip((np.mean(overlaps) - 0.5) / 0.5) * _clip(1 - max(size_changes) / 0.2)
    else:
        res["same_character"] = None

    # --- take gates
    s = float(sheet_scale_eff)
    bbs = [bbox(k) for k in keys]
    gates = {}
    if all(b is not None for b in bbs):
        ground = np.array([(f.shape[0] - 1 - b[1]) / s for f, b in zip(frames, bbs)])      # distance of the lowest opaque row to the frame bottom
        height = np.array([(b[1] - b[0] + 1) / s for b in bbs])
        gates["ground_shift"] = round(float(ground.max() - ground.min()), 3)
        gates["zoom_variation"] = round(float((height.max() - height.min()) / np.median(height)), 4)
        q["ground"] = _clip(1 - gates["ground_shift"] / 4)
        q["zoom"] = _clip(1 - gates["zoom_variation"] / 0.25)
    else:
        gates["ground_shift"] = gates["zoom_variation"] = None
        q["ground"] = q["zoom"] = 0.0
    # clean background: border pixels of every frame share one colour (transparent in a cut-out sprite)
    shares = []
    for k in keys:
        border = np.concatenate([k[0], k[-1], k[:, 0], k[:, -1]])
        shares.append(float((border == _mode(border)).mean()))
    gates["background_clean"] = round(float(min(shares)), 4)
    q["background"] = _clip((gates["background_clean"] - 0.9) / 0.1)
    # loop closes: last frame about as close to the first as consecutive frames are to each other
    if n > 2:
        def diff(a, b):
            u = ((a & 255) > 0) | ((b & 255) > 0)
            return float((a[u] != b[u]).mean()) if u.any() else 0.0
        steps = [diff(a, b) for a, b in zip(nat[:-1], nat[1:])]
        gap = diff(nat[-1], nat[0])
        mean_step = float(np.mean(steps))
        gates["loop_gap"] = round(gap, 4)
        gates["loop_mean_step"] = round(mean_step, 4)
        gates["loop_ok"] = bool(gap <= max(1.5 * mean_step, 0.05))
        q["loop"] = _clip(1 - max(0.0, gap - max(1.5 * mean_step, 0.05)) / 0.2)
    else:
        gates["loop_gap"] = gates["loop_mean_step"] = gates["loop_ok"] = None
    res["gates"] = gates
    res["warnings"] = warnings

    res["quality"] = {k: round(v, 4) for k, v in q.items()}
    res["overall"] = round(float(np.mean(list(q.values()))), 4)
    return res


def _mode(a):
    v, c = np.unique(a, return_counts=True)
    return v[c.argmax()]


def run(path, scale=None, frames=None, base=None, background="auto"):
    fr, kind = load_frames(path, frames)
    sizes = {f.shape for f in fr}
    base_img = load_rgba(base) if base else None
    res = score_frames(fr, scale, base_img, background)
    res = {"input": path.replace("\\", "/"), "kind": kind, **res}
    if len(sizes) > 1:
        res["note"] = "frames differ in size; padded bottom-centre for pixel comparisons"
    return res


# ----------------------------------------------------------------------------- output
def format_table(r):
    g = r["gates"]
    sc = r["same_character"]
    f = lambda v, p=3: "n/a" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))
    rows = [
        ("grid purity", f(r["grid_purity"]), ">= 0.97"),
        ("grid scale", f(r["grid_scale"]), "found (not null)"),
        ("grid stability", f(r["grid_stability"]), "1.0"),
        ("colour flicker", f(r["colour_flicker"]), "<= 0.10"),
        ("palette overlap (min)", f(sc and sc["palette_overlap_min"]), ">= 0.90"),
        ("size change (max)", f(sc and sc["size_change_max"]), "<= 0.10"),
        ("palette size (sheet)", str(r["palette_size"]["sheet"]), "<= 32"),
        ("ground shift (px)", f(g["ground_shift"]), "<= 1"),
        ("zoom variation", f(g["zoom_variation"]), "<= 0.10"),
        ("background clean", f(g["background_clean"]), ">= 0.98"),
        ("loop closes", f(g["loop_ok"]), "True"),
        ("overall", f(r["overall"]), "higher is better"),
    ]
    w = max(len(a) for a, _, _ in rows)
    lines = [f"{r['input']}  ({r['kind']}, {r['frames']} frame(s), {r['frame_size'][0]}x{r['frame_size'][1]})"]
    lines += [f"  {a:<{w}}  {b:>8}   good: {c}" for a, b, c in rows]
    lines.append(f"  background used: {r['background']['colour']}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score a sprite, sprite sheet or folder of frames without looking at it.")
    ap.add_argument("path", help="image, sprite sheet, or folder of frame images")
    ap.add_argument("--scale", type=int, help="pixel size in source pixels (detected when left out)")
    ap.add_argument("--frames", help="sheet layout as COLSxROWS, e.g. 4x1 (a single image without it is one frame)")
    ap.add_argument("--base", help="base sprite image to compare frames against (default: the first frame)")
    ap.add_argument("--background", default="auto", metavar="auto|none|#rrggbb",
                    help="flat opaque background to ignore: auto = the border's one colour (default), none, or a colour")
    ap.add_argument("--json", action="store_true", help="print the scores as JSON only")
    a = ap.parse_args(argv)
    try:
        r = run(a.path, a.scale, a.frames, a.base, a.background)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(r, indent=2))
    else:
        print(format_table(r))
    for w in r.get("warnings", []):
        print(f"warning: {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
