"""Layered Vectorize: turn an illustration into an editable, layered SVG.

Instead of tracing every colour patch (thousands of shapes), the image is decomposed the way an
artist builds it, each layer traced with potrace:

  Flats       one base colour per region (palette of N auto colours, or your hex list), extended
              under the ink so there are no gaps
  Shadows     black at low opacity where pixels are darker than their region's base colour
  Highlights  white at low opacity where pixels are lighter than the base colour
  Ink         the dark outlines
  Details     small things the layers above don't explain (eyes, specks of colour), on top

Layers are <g> groups that Inkscape/Illustrator show as layers. Shapes are either one object per
region ("per region") or one compound path per colour/layer ("compound").
"""
import re

import cv2
import numpy as np
import potrace
from scipy import ndimage


INK_MAX_CHROMA = 8     # measured: real ink has chroma ~1 (90% < 5.4); dark spider brown ~10


def parse_palette(text):
    return [tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) for h in re.findall(r"#?\b([0-9a-fA-F]{6})\b", text or "")]


def hex_color(rgb):
    return "#%02x%02x%02x" % tuple(int(c) for c in rgb)


def rgb_to_lab(rgb):
    return cv2.cvtColor(np.asarray(rgb, np.uint8).reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)


def clean(mask, open_px, min_area):
    m = mask.astype(np.uint8)
    if open_px:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px * 2 + 1,) * 2))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lbl]


def disk(r):
    r = max(1, int(round(r)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def limit_thickness(mask, min_px, max_px):
    """Clamp stroke widths to [min_px, max_px] (0 = no limit) without a skeleton library.

    Uses the distance transform (each pixel's distance to the stroke edge; half the stroke width at
    its centre). Too-thin strokes get their centreline (distance ridge) redrawn at min_px. Too-thick
    strokes keep only the band within max_px / 2 of their centre, which stays continuous. Solid fills
    (areas that hold a disk 3 * max_px wide once small holes like eyes are closed) are left alone.
    """
    if not mask.any() or (min_px <= 0 and max_px <= 0):
        return mask
    m8 = mask.astype(np.uint8)
    dt = cv2.distanceTransform(m8, cv2.DIST_L2, 5)
    out = mask.copy()
    if max_px > 0:
        closed = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, disk(3))
        fills = cv2.morphologyEx(closed, cv2.MORPH_OPEN, disk(max_px * 1.5))
        fills = cv2.dilate(fills, disk(2)).astype(bool) & mask
        centre = cv2.dilate(dt, disk(max_px))                # ~ half-width of the stroke a pixel belongs to
        thick = mask & ~fills & (centre > max_px / 2 + 0.5)
        keep = dt >= centre - max_px / 2                     # within max_px / 2 of the centre
        out = (out & ~thick) | (thick & keep)
    if min_px > 0:
        dt2 = cv2.distanceTransform(out.astype(np.uint8), cv2.DIST_L2, 5)
        ridge = (dt2 > 0) & (dt2 >= cv2.dilate(dt2, np.ones((3, 3), np.uint8)) - 1e-3)
        thin = ridge & (dt2 < min_px / 2)
        if thin.any():
            out |= cv2.dilate(thin.astype(np.uint8), disk(min_px / 2)).astype(bool)
    return out


def trace_d(mask, offset, smoothness, turd):
    # this potrace port traces the False pixels, hence the inversion
    plist = potrace.Bitmap(~mask).trace(turdsize=turd, turnpolicy=potrace.POTRACE_TURNPOLICY_MINORITY,
                                        alphamax=smoothness, opticurve=True, opttolerance=0.4)
    ox, oy = offset
    parts = []
    for curve in plist:
        s = curve.start_point
        parts.append(f"M{s.x + ox:.1f} {s.y + oy:.1f}")
        for seg in curve.segments:
            e = seg.end_point
            if seg.is_corner:
                parts.append(f"L{seg.c.x + ox:.1f} {seg.c.y + oy:.1f}L{e.x + ox:.1f} {e.y + oy:.1f}")
            else:
                parts.append(f"C{seg.c1.x + ox:.1f} {seg.c1.y + oy:.1f} {seg.c2.x + ox:.1f} {seg.c2.y + oy:.1f} "
                             f"{e.x + ox:.1f} {e.y + oy:.1f}")
        parts.append("Z")
    return "".join(parts)


def shapes(mask, per_region, min_area, smoothness):
    """Path 'd' strings for a mask: one per connected region, or a single compound path."""
    if not mask.any():
        return []
    if not per_region:
        d = trace_d(mask, (0, 0), smoothness, min_area)
        return [d] if d else []
    n, cc, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = []
    for i in np.argsort(-stats[:, cv2.CC_STAT_AREA]):   # big regions first
        if i == 0 or stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        x, y, w, h = stats[i, :4]
        sub = np.zeros((h + 4, w + 4), bool)
        sub[2:-2, 2:-2] = cc[y:y + h, x:x + w] == i
        d = trace_d(sub, (x - 2, y - 2), smoothness, 0)
        if d:
            out.append(d)
    return out


def vectorize(rgb, alpha, colors, palette, min_region, ink_threshold, tone_threshold, detail_threshold,
              detail_min, shadow_opacity, highlight_opacity, layers, per_region, remove_background, smoothness,
              stacked=True, overlap_px=2, line_min=0, line_max=0, inset_px=2, tinted=True):
    H, W = rgb.shape[:2]
    lab = rgb_to_lab(rgb.reshape(-1, 3)).reshape(H, W, 3)
    L = lab[..., 0]                                        # OpenCV 8-bit Lab: L in 0..255
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)

    # Background (transparent): alpha if given, else bright low-chroma pixels touching the border.
    if alpha is not None:
        bg = alpha < 0.5
    elif remove_background:
        # plain background touching the border, plus large enclosed gaps of the same colour (e.g. between
        # an arm and a staff) - unless the palette itself asks for white
        lbl, n = ndimage.label((L > 235) & (chroma < 12))
        edge = np.unique(np.concatenate([lbl[0], lbl[-1], lbl[:, 0], lbl[:, -1]]))
        ids = set(edge[edge > 0].tolist())
        wants_white = any(min(c) > 235 for c in parse_palette(palette))
        if not wants_white and n:
            areas = ndimage.sum_labels(np.ones_like(L), lbl, index=np.arange(1, n + 1))
            ids |= set((np.where(areas >= min_region)[0] + 1).tolist())
        bg = np.isin(lbl, list(ids))
    else:
        bg = np.zeros((H, W), bool)

    ink_min = max(detail_min, min_region // 10)
    # ink is dark AND neutral: dark browns/blues (e.g. small spiders) stay colour
    ink = clean((L < ink_threshold * 2.55) & (chroma < INK_MAX_CHROMA) & ~bg, 1, ink_min)
    ink = limit_thickness(ink, line_min, line_max) & ~bg
    # Anti-aliased pixels along ink lines are darker than the flat next to them; they must not become
    # shadows or details. Near ink, only pixels LIGHTER than their flat count (e.g. eyes in a dark visor).
    ink_edge = cv2.dilate(ink.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))).astype(bool) & ~ink
    fg = ~bg & ~ink
    if not fg.any():
        fg = ~bg

    # Flats: smooth away shading, assign base colours, mode-filter, extend under the ink.
    smooth = cv2.pyrMeanShiftFiltering(np.ascontiguousarray(rgb[..., ::-1]), 12, 24)
    smooth_lab = rgb_to_lab(smooth[..., ::-1].reshape(-1, 3)).reshape(H, W, 3)
    pal = parse_palette(palette)
    if pal:
        centers = rgb_to_lab(pal)
    else:
        samples = smooth_lab[fg][:: max(1, int(fg.sum()) // 200000)]
        k = int(min(colors, len(samples)))
        _, _, centers = cv2.kmeans(samples, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5),
                                   4, cv2.KMEANS_PP_CENTERS)
    K = len(centers)
    labels = np.full((H, W), -1, np.int32)
    labels[fg] = ((smooth_lab[fg][:, None, :] - centers[None]) ** 2).sum(-1).argmin(-1)
    win = max(5, round(max(H, W) / 136)) | 1
    votes = np.stack([cv2.blur((labels == k).astype(np.float32), (win, win)) for k in range(K)], -1)
    labels = np.where(fg, votes.argmax(-1), -1).astype(np.int32)

    def fill_holes(holes):
        if holes.any() and (labels >= 0).any():
            _, (iy, ix) = ndimage.distance_transform_edt(labels < 0, return_indices=True)
            labels[holes] = labels[iy, ix][holes]

    fill_holes(~bg & (labels < 0))

    # Auto palette: a background-coloured flat (enclosed gaps) becomes transparent too.
    if not pal and alpha is None and remove_background:
        for k in range(K):
            if centers[k][0] > 235 and np.hypot(centers[k][1] - 128, centers[k][2] - 128) < 12:
                bg |= labels == k
                labels[labels == k] = -1
    # Merge flat regions smaller than min_region into their surroundings.
    small = np.zeros((H, W), bool)
    for k in range(K):
        n, cc, st, _ = cv2.connectedComponentsWithStats((labels == k).astype(np.uint8), 8)
        tiny = np.where(st[:, cv2.CC_STAT_AREA] < min_region)[0]
        small |= np.isin(cc, tiny[tiny > 0])
    labels[small] = -1
    fill_holes(small)

    present = [k for k in range(K) if (labels == k).any()]
    fills, base_lab = {}, np.zeros((K, 3), np.float32)
    for k in present:
        m = (labels == k) & ~ink
        m = m if m.any() else labels == k
        base_lab[k] = np.median(lab[m], axis=0)
        fills[k] = pal[k] if pal else tuple(np.median(rgb[m], axis=0).astype(int))

    # Keep fills, tones and details inside the outline: the anti-aliased rim between an outline and the
    # background ("halo") becomes background, and wherever there is ink, everything else stops inset_px
    # inside the outline's outer edge (still under the ink, so no gap shows; edges without ink are untouched).
    outside = bg.copy()
    if inset_px > 0 and ink.any():
        near_bg = cv2.dilate(bg.astype(np.uint8), disk(3)).astype(bool)
        near_ink = cv2.dilate(ink.astype(np.uint8), disk(3)).astype(bool)
        outside |= near_bg & near_ink & ~ink
        outside |= cv2.dilate(outside.astype(np.uint8), disk(inset_px)).astype(bool) & ink
    inside = ~outside

    # Tones: lighter / darker than the region's actual base colour.
    region_L = np.where(labels >= 0, base_lab[np.clip(labels, 0, None), 0], L)
    diff = L - region_L
    tone_min = max(detail_min, min_region // 4)
    shadow = clean((diff < -tone_threshold * 2.55) & ~ink & ~ink_edge & inside, 1, tone_min) if layers["shadows"] else None
    highlight = clean((diff > tone_threshold * 2.55) & ~ink & inside, 1, tone_min) if layers["highlights"] else None

    # Details: pixels the layers so far don't explain, in the nearest base colour (no extra colours).
    details = {}
    if layers["details"]:
        pred = base_lab[np.clip(labels, 0, None)].copy()
        if shadow is not None:
            pred[shadow, 0] *= 1 - shadow_opacity
        if highlight is not None:
            pred[highlight, 0] += (255 - pred[highlight, 0]) * highlight_opacity
        if layers["ink"]:
            pred[ink] = rgb_to_lab([[20, 20, 20]])[0]
        miss = (np.linalg.norm(lab - pred, axis=-1) > detail_threshold * 2.55) & ~bg & ~ink
        miss &= ~ink_edge | (diff > tone_threshold * 2.55)
        cand = [fills[k] for k in present]       # base colours only; dark details are covered by the ink layer
        cand_lab = rgb_to_lab(cand)
        idx = np.full((H, W), -1, np.int32)
        idx[miss] = ((lab[miss][:, None, :] - cand_lab[None]) ** 2).sum(-1).argmin(-1)
        same = np.zeros((H, W), bool)          # skip pixels already drawn in that colour
        for j, c in enumerate(cand[:len(present)]):
            same |= (idx == j) & (labels == present[j]) & ~ink
        idx[same] = -1
        for j, c in enumerate(cand):
            m = clean(idx == j, 0, detail_min)
            if m.any():
                details[c] = limit_thickness(m, line_min, line_max) & inside

    ink_rgb = tuple(np.median(rgb[ink], axis=0).astype(int)) if ink.any() else (20, 20, 20)
    # If a base colour is (near-)black, outlines use exactly that colour, so "black" counts once.
    darkest = min(present, key=lambda k: base_lab[k][0], default=None)
    if darkest is not None:
        dl = base_lab[darkest]
        if dl[0] < 18 * 2.55 and np.hypot(dl[1] - 128, dl[2] - 128) < 10:
            ink_rgb = tuple(int(v) for v in fills[darkest])

    def shade(rgb_c, amount):
        # amount < 0 darkens towards black, > 0 lightens towards white (same result as a black/white overlay)
        c = np.asarray(rgb_c, float)
        c = c * (1 + amount) if amount < 0 else c + (255 - c) * amount
        return tuple(int(round(v)) for v in np.clip(c, 0, 255))

    # Assemble the SVG: one <g> layer per role.
    def group(gid, label, attrs, ds, role=""):
        if not ds:
            return ""
        body = "".join(f'<path d="{d}"/>' for d in ds)
        role = f' data-role="{role}"' if role else ""
        return f'<g id="{gid}" inkscape:label="{label}"{role} {attrs}>{body}</g>'

    def tone_layer(name, mask, amount, opacity):
        # tinted: per base colour, a darker/lighter shade of that colour; overlay: black/white at an opacity
        role = name[:-1]
        if not tinted:
            colour = "#000000" if amount < 0 else "#ffffff"
            return group(name, name.title(), f'inkscape:groupmode="layer" fill="{colour}" fill-opacity="{opacity}" '
                         'fill-rule="evenodd"', shapes(mask, per_region, tone_min, smoothness), role)
        subs = []
        for k in order:
            c = hex_color(shade(fills[k], amount))
            subs.append(group(f"{role}-{c[1:]}", f"{name.title()[:-1]} of {hex_color(fills[k])}",
                              f'fill="{c}" fill-rule="evenodd"', shapes(mask & (labels == k), per_region, tone_min,
                                                                         smoothness), role))
        body = "".join(subs)
        return f'<g id="{name}" inkscape:groupmode="layer" inkscape:label="{name.title()}">{body}</g>' if body else ""

    out = []
    flat_groups = []
    order = sorted(present, key=lambda k: -(labels == k).sum())     # biggest colour at the bottom
    masks = {k: labels == k for k in order}
    if stacked:
        # Each colour also covers everything drawn above it, so the edge between two colours always sits on
        # a filled shape: no hairline gaps where independently traced edges don't line up, and the shapes
        # stay large and simple (no holes cut out for the colours on top).
        above = np.zeros((H, W), bool)
        for k in reversed(order):
            masks[k] = masks[k] | above
            above = masks[k]
    if overlap_px > 0:
        # grow each upper colour slightly into the ones below it (never into the background)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (overlap_px * 2 + 1,) * 2)
        for k in order[1:]:
            masks[k] = cv2.dilate(masks[k].astype(np.uint8), kern).astype(bool)
    for k in order:
        masks[k] &= inside
        ds = shapes(masks[k], per_region, min_region // 4, smoothness)
        c = hex_color(fills[k])
        flat_groups.append(group(f"flat-{c[1:]}", f"Flat {c}", f'fill="{c}" fill-rule="evenodd"', ds, "base"))
    out.append(f'<g id="flats" inkscape:groupmode="layer" inkscape:label="Flats">{"".join(flat_groups)}</g>')
    if shadow is not None:
        out.append(tone_layer("shadows", shadow, -shadow_opacity, shadow_opacity))
    if highlight is not None:
        out.append(tone_layer("highlights", highlight, highlight_opacity, highlight_opacity))
    if layers["ink"]:
        out.append(group("ink", "Ink", f'inkscape:groupmode="layer" fill="{hex_color(ink_rgb)}" fill-rule="evenodd"',
                         shapes(ink, per_region, ink_min, smoothness), "ink"))
    if details:
        dg = [group(f"detail-{hex_color(c)[1:]}", f"Detail {hex_color(c)}", f'fill="{hex_color(c)}" fill-rule="evenodd"',
                    shapes(m, per_region, detail_min, smoothness), "detail") for c, m in details.items()]
        out.append(f'<g id="details" inkscape:groupmode="layer" inkscape:label="Details">{"".join(dg)}</g>')

    return (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
            f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n' + "\n".join(o for o in out if o) + "\n</svg>\n")


class LayeredVectorize:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "colors": ("INT", {"default": 6, "min": 2, "max": 16, "tooltip": "Number of base colours (auto palette). Ignored when a palette is given."}),
            "palette": ("STRING", {"default": "", "multiline": False, "tooltip": "Optional hex colours, e.g. #4d4d50 #d9cbb0 #4a3a2e. Every flat snaps to the nearest one."}),
            "min_region_px": ("INT", {"default": 2000, "min": 1, "max": 100000, "tooltip": "Flat regions smaller than this many pixels merge into their surroundings. Higher = fewer shapes."}),
            "ink_threshold": ("INT", {"default": 22, "min": 0, "max": 100, "tooltip": "Pixels darker than this lightness (0-100) become ink."}),
            "tone_threshold": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 50.0, "step": 0.5, "tooltip": "How much darker/lighter than the base colour (lightness 0-100) a pixel must be to become shadow/highlight."}),
            "detail_threshold": ("FLOAT", {"default": 25.0, "min": 5.0, "max": 100.0, "step": 1.0, "tooltip": "Colour difference from the layers below needed to add a detail shape. Lower = more details."}),
            "detail_min_px": ("INT", {"default": 60, "min": 1, "max": 10000, "tooltip": "Smallest detail / ink speck kept, in pixels."}),
            "shadow_opacity": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 1.0, "step": 0.05, "tooltip": "Shadow strength: how much darker than its base colour (tinted) or the black opacity (overlay)."}),
            "highlight_opacity": ("FLOAT", {"default": 0.35, "min": 0.05, "max": 1.0, "step": 0.05, "tooltip": "Highlight strength: how much lighter than its base colour (tinted) or the white opacity (overlay)."}),
            "curve_smoothness": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 1.33, "step": 0.05, "tooltip": "Corner smoothing: 0 = all sharp corners, 1.33 = no corners."}),
            "objects": (["per region", "compound"], {"tooltip": "per region: each region is its own shape. compound: one shape per colour / layer."}),
            "shadows": ("BOOLEAN", {"default": True}),
            "highlights": ("BOOLEAN", {"default": True}),
            "details": ("BOOLEAN", {"default": True}),
            "ink": ("BOOLEAN", {"default": True}),
            "remove_background": ("BOOLEAN", {"default": True, "tooltip": "Opaque images: make the plain border-connected background transparent."}),
            "flat_stacking": (["stacked", "cutout"], {"tooltip": "stacked: biggest colour at the bottom, each colour extends under the ones above it (no gaps, larger simpler shapes). cutout: colours only cover their own area."}),
            "overlap_px": ("INT", {"default": 2, "min": 0, "max": 20, "tooltip": "Grow each flat colour this many pixels into the colour below it, to hide seams."}),
            "line_min_px": ("INT", {"default": 0, "min": 0, "max": 50, "tooltip": "Minimum stroke width (px, at the traced size) for ink and details. Thinner strokes are thickened. 0 = off."}),
            "line_max_px": ("INT", {"default": 0, "min": 0, "max": 100, "tooltip": "Maximum stroke width for ink and details. Thicker strokes are slimmed; solid fills wider than 3x this are kept. 0 = off."}),
            "inset_under_ink_px": ("INT", {"default": 2, "min": 0, "max": 20, "tooltip": "Keep fills, tones and details this many pixels inside the outer edge of outlines, so they never peek out past the ink. 0 = off."}),
            "tone_style": (["tinted", "overlay"], {"tooltip": "tinted: shadows/highlights are darker/lighter shades of each base colour (no extra colours). overlay: black/white shapes at shadow/highlight_opacity."}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("SVG_String",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "run"
    CATEGORY = "comfyui_workflows/svg"
    DESCRIPTION = "Layered vectorization: flats, shadows, highlights, ink and details as editable SVG layers."

    def run(self, image, colors, palette, min_region_px, ink_threshold, tone_threshold, detail_threshold,
            detail_min_px, shadow_opacity, highlight_opacity, curve_smoothness, objects, shadows, highlights,
            details, ink, remove_background, flat_stacking="stacked", overlap_px=2, line_min_px=0, line_max_px=0,
            inset_under_ink_px=2, tone_style="tinted"):
        out = []
        for img in image:
            arr = (img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            rgb, alpha = arr[..., :3].copy(), (arr[..., 3] / 255.0 if arr.shape[-1] == 4 else None)
            if alpha is not None and alpha.min() >= 0.5:
                alpha = None                      # fully opaque: treat as an opaque image
            out.append(vectorize(rgb, alpha, colors, palette, min_region_px, ink_threshold, tone_threshold,
                                 detail_threshold, detail_min_px, shadow_opacity, highlight_opacity,
                                 {"shadows": shadows, "highlights": highlights, "details": details, "ink": ink},
                                 objects == "per region", remove_background, curve_smoothness,
                                 flat_stacking == "stacked", overlap_px, line_min_px, line_max_px, inset_under_ink_px,
                                 tone_style == "tinted"))
        return (out,)
