"""Generates workflows/pixel-art/image-to-pixel.json.

Edit the constants below and re-run:  python tools/build_pixel_image_to_pixel.py

Image (object or character) -> true pixel art: optional BiRefNet background removal (core nodes), then the repo's
Pixelize node (exact grid at native resolution, 'pixels' = subject height, limited/locked palette, hard alpha),
saved at native size plus a nearest-neighbour preview, checked by Pixel Art Check.
"""
from comfy_workflow import BIREFNET, BIREFNET_URL, BLUE, GREEN, YELLOW, WORKFLOWS, Graph, inp, out

OUT = WORKFLOWS / "pixel-art" / "image-to-pixel.json"

PIXELS = 64            # subject height in art pixels
COLORS = 0             # 0 = automatic: the fewest colours within COLOR_ERROR
COLOR_ERROR = 12.0     # higher = fewer colours (12 faithful, 20 simplified, 30 very few); illustrations: 12
MAX_COLORS = 32        # Pixel Art Check limit (automatic palettes stay under it)
# images, pixels, measure, method, smooth, colors, color_error, palette, shared_palette, background, bg_tolerance, cleanup, edge_cleanup, outline,
# padding, preview_scale
SMOOTH = 0.6           # edge-preserving smoothing (art pixels); flattens texture into solid areas
CUTOUT = [True, False, 0.02, 8.0, 0.2]   # drop_shadow, enclosed_background, min_pocket, bg_tolerance, shadow_band
PIXELIZE = [PIXELS, "subject height", "pixeloe", SMOOTH, COLORS, COLOR_ERROR, "auto", True, "auto remove", 12.0, True, True, "auto", 1, 8]

NOTE = f"""Image -> true pixel art

1. Load an image (blue): a character or object. Any background: BiRefNet removes it.
   Art that's already on a plain/transparent background or already pixel art: bypass
   "Remove background" (select it, Ctrl+B) to skip the AI cut-out.
2. Pixelize (yellow):
   pixels  = size in art pixels (default: subject height {PIXELS})
   measure = what 'pixels' means; 'auto (detect grid)' recovers existing (upscaled) pixel art exactly
   colors = 0: automatic, the FEWEST colours that keep the look (color_error: higher = fewer); or a number,
   palette = a preset (PICO-8, DB32, Endesga-32, GameBoy) or hex list
   method: pixeloe = PixelOE's contrast-aware pixelization (cleanest for illustrations)
   smooth  = flattens texture/noise into solid areas first (0 = off, 1+ = flatter, fewer details)
   cleanup removes stray single pixels (eyes are kept); edge_cleanup defringes the edges (no light halo colours)
   and smooths the silhouette; outline: auto completes the dark outline (or uses darker shades), darkest = black
   Clean cutout removes a drop shadow under the subject and background seen through gaps (toggles).
3. Run. Saved to output/pixel/:
   image_to_pixel_*.png  native size (1 art pixel = 1 image pixel, transparent background) -> use in games
   image_to_pixel_x8_*.png  zoomed preview
"Pixel Art Check" must say PASS (hard alpha, native x1, palette size)."""

g = Graph()
node, link = g.node, g.link

node(1, "LoadImage", [0, 0], [400, 440], outputs=[out("IMAGE", "IMAGE"), out("MASK", "MASK")],
     widgets=["example.png", "image"], title="Input image", color=BLUE)
node(13, "JoinImageWithAlpha", [0, 440], [400, 46], inputs=[inp("image", "IMAGE"), inp("alpha", "MASK")],
     outputs=[out("IMAGE", "IMAGE")], title="Keep input transparency")
node(2, "LoadBackgroundRemovalModel", [440, 0], [320, 58], outputs=[out("bg_model", "BACKGROUND_REMOVAL")],
     widgets=[BIREFNET], title="BiRefNet",
     props={"Node name for S&R": "LoadBackgroundRemovalModel",
            "models": [{"name": BIREFNET, "url": BIREFNET_URL, "directory": "background_removal"}]})
node(3, "RemoveBackground", [440, 100], [320, 46], inputs=[inp("bg_removal_model", "BACKGROUND_REMOVAL"), inp("image", "IMAGE")],
     outputs=[out("mask", "MASK")])
node(4, "InvertMask", [440, 180], [320, 26], inputs=[inp("mask", "MASK")], outputs=[out("MASK", "MASK")])
node(5, "JoinImageWithAlpha", [440, 250], [320, 46], inputs=[inp("image", "IMAGE"), inp("alpha", "MASK")],
     outputs=[out("IMAGE", "IMAGE")], title="Remove background (Ctrl+B to skip)", color=YELLOW)
node(14, "CleanCutout", [440, 330], [320, 160], inputs=[inp("images", "IMAGE")],
     outputs=[out("image", "IMAGE"), out("removed", "MASK"), out("info", "STRING")], widgets=CUTOUT,
     title="Clean cutout (drop shadow, background pockets)", color=YELLOW)
node(6, "Pixelize", [800, 0], [360, 420], inputs=[inp("images", "IMAGE")],
     outputs=[out("pixel_art", "IMAGE"), out("preview", "IMAGE"), out("palette", "STRING"), out("info", "STRING")],
     widgets=PIXELIZE, title="Pixelize", color=YELLOW)
node(7, "SaveImage", [1200, 0], [360, 300], inputs=[inp("images", "IMAGE")], widgets=["pixel/image_to_pixel"],
     title="Save pixel art (native size)", color=GREEN)
node(8, "SaveImage", [1200, 340], [360, 420], inputs=[inp("images", "IMAGE")], widgets=["pixel/image_to_pixel_x8"],
     title="Save preview (x8)")
node(9, "PixelArtCheck", [800, 470], [360, 106], inputs=[inp("images", "IMAGE")],
     outputs=[out("report", "STRING"), out("passed", "BOOLEAN")], widgets=[MAX_COLORS, PIXELS, True], title="Pixel Art Check")
node(10, "PreviewAny", [800, 620], [360, 200], inputs=[inp("source", "*")], title="Check report")
node(11, "PreviewAny", [1200, 800], [360, 160], inputs=[inp("source", "*")], title="Palette")
node(12, "Note", [0, 530], [460, 470], widgets=[NOTE], title="READ ME", props={})

link(2, 0, 3, 0)
link(1, 0, 13, 0)            # LoadImage drops alpha into MASK: join it back (used when BiRefNet is bypassed)
link(1, 1, 13, 1)
link(1, 0, 3, 1)
link(3, 0, 4, 0)
link(13, 0, 5, 0)
link(4, 0, 5, 1)
link(5, 0, 14, 0)
link(14, 0, 6, 0)
link(6, 0, 7, 0)
link(6, 1, 8, 0)
link(6, 0, 9, 0)
link(9, 0, 10, 0)
link(6, 2, 11, 0)

g.save(OUT, "pixel-art-image-to-pixel", groups=[("Background removal (BiRefNet)", [420, -60, 360, 380], "#444")],
       scale=0.75, offset=(40, 80))
