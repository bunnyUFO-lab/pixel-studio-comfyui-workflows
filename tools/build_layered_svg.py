"""Generates workflows/image-to-svg/layered-svg.json.

Edit the constants below and re-run:  python tools/build_layered_svg.py

Same input stages as vtracer-svg (AI upscale, resize, hard transparency edge), then the repo's
Layered Vectorize node (custom_nodes/comfyui_workflows_utils/layered.py) instead of VTracer.
Defaults tested on a detailed 1408x768 character: ~430 shapes in 5 layers (vs ~2,900 from VTracer),
or ~11 in compound mode.
"""
from comfy_workflow import ALPHA_CUTOFF, BLUE, GREEN, UPSCALER, UPSCALER_URL, YELLOW, WORKFLOWS, Graph, inp, out

OUT = WORKFLOWS / "image-to-svg" / "layered-svg.json"

MAX_SIZE = 2048            # flats/ink trace cleanly at 2048; 4096 is ~4x slower for little gain
SAVE_PREFIX = "svg/layered"
# colors, palette, min_region_px, ink_threshold, tone_threshold, detail_threshold, detail_min_px,
# shadow_opacity, highlight_opacity, curve_smoothness, objects, shadows, highlights, details, ink,
# remove_background, flat_stacking, overlap_px, line_min_px, line_max_px, inset_under_ink_px, tone_style
LAYERED = [6, "", 2000, 22, 6.0, 25.0, 60, 0.3, 0.35, 1.1, "per region", True, True, True, True, True, "stacked", 2, 0, 0, 2, "tinted"]

NOTE = f"""Layered SVG - flats + shadows + highlights + ink + details

Builds the SVG the way an artist would, as named layers (Inkscape/Illustrator show them as layers):
  Flats       one base colour per region, extended under the ink (no gaps)
  Shadows     a darker shade of each base colour where the image is darker
  Highlights  a lighter shade of each base colour where it is lighter
  Ink         the dark outlines
  Details     small things the layers above miss (web strands, eyes...)
Saved to output/{SAVE_PREFIX}_NNNNN_.svg; "Shapes & file size" shows shapes and base colours
(shadow/highlight shades are not counted as colours).

Palette: leave empty for {LAYERED[0]} automatic colours, or type hex colours
(e.g. #4f4f52 #8c8c90 #e3d6bb #4b3426) - every flat snaps to the nearest one.
Ink (dark outlines) is detected separately, so you don't need black in the palette.

Fewer shapes: raise min_region_px (4000) and detail_min_px (100), or turn details off.
More detail: lower min_region_px (1000) and detail_min_px (30), or detail_threshold (18).
objects: 'per region' = each region its own shape; 'compound' = one shape per colour/layer.
Missing outlines: raise ink_threshold (30). Too much ink: lower it (15).
Line widths: line_min_px / line_max_px clamp ink + detail strokes (e.g. 4 / 8 at 2048px); 0 = off.
Flats are stacked (biggest colour at the bottom, extending under the others) so there are no gaps.
Tones too busy: raise tone_threshold (10); too flat: lower it (4).
Transparent PNGs keep transparency; plain opaque backgrounds are removed (remove_background)."""

g = Graph()
node, link = g.node, g.link

node(1, "LoadImage", [0, 0], [400, 420], outputs=[out("IMAGE", "IMAGE"), out("MASK", "MASK")],
     widgets=["example.png", "image"], title="Input image", color=BLUE)
node(12, "UpscaleModelLoader", [440, 0], [320, 58], outputs=[out("UPSCALE_MODEL", "UPSCALE_MODEL")],
     widgets=[UPSCALER], title="AI upscaler (Real-ESRGAN anime)",
     props={"Node name for S&R": "UpscaleModelLoader",
            "models": [{"name": UPSCALER, "url": UPSCALER_URL, "directory": "upscale_models"}]})
node(13, "ImageUpscaleWithModel", [440, 100], [320, 46],
     inputs=[inp("upscale_model", "UPSCALE_MODEL"), inp("image", "IMAGE")], outputs=[out("IMAGE", "IMAGE")],
     title="AI upscale x4 (Ctrl+B to skip)", color=YELLOW)
node(2, "ImageScaleToMaxDimension", [440, 190], [320, 82], inputs=[inp("image", "IMAGE")],
     outputs=[out("IMAGE", "IMAGE")], widgets=["lanczos", MAX_SIZE], title="Resize (longest side)", color=YELLOW)
node(3, "MaskToImage", [440, 340], [320, 26], inputs=[inp("mask", "MASK")], outputs=[out("IMAGE", "IMAGE")])
node(4, "ImageScaleToMaxDimension", [440, 410], [320, 82], inputs=[inp("image", "IMAGE")],
     outputs=[out("IMAGE", "IMAGE")], widgets=["lanczos", MAX_SIZE], title="Resize mask (keep = image)")
node(5, "ImageToMask", [440, 540], [320, 58], inputs=[inp("image", "IMAGE")], outputs=[out("MASK", "MASK")],
     widgets=["red"])
node(6, "ThresholdMask", [440, 640], [320, 58], inputs=[inp("mask", "MASK")], outputs=[out("MASK", "MASK")],
     widgets=[ALPHA_CUTOFF], title="Hard transparency edge")
node(7, "JoinImageWithAlpha", [440, 740], [320, 46], inputs=[inp("image", "IMAGE"), inp("alpha", "MASK")],
     outputs=[out("IMAGE", "IMAGE")], title="Apply transparency")

node(8, "LayeredVectorize", [800, 0], [360, 520], inputs=[inp("image", "IMAGE")],
     outputs=[out("SVG_String", "STRING")], widgets=LAYERED, title="Layered Vectorize", color=YELLOW)
node(14, "TS_SVGStringToSVGBytesIO", [1200, 0], [300, 26], inputs=[inp("SVG_String", "STRING")],
     outputs=[out("SVG", "SVG")], title="SVG text -> SVG")
node(9, "SaveSVGNode", [1200, 80], [420, 520], inputs=[inp("svg", "SVG")], outputs=[out("svg", "SVG")],
     widgets=[SAVE_PREFIX], title="Save SVG (output/svg/)", color=GREEN)
node(15, "SVGStats", [1660, 0], [300, 70], inputs=[inp("SVG_String", "STRING")],
     outputs=[out("info", "STRING"), out("shapes", "INT"), out("size_kb", "FLOAT")], title="SVG Stats")
node(16, "PreviewAny", [1660, 110], [340, 200], inputs=[inp("source", "*")], title="Shapes & file size")
node(11, "Note", [0, 470], [460, 620], widgets=[NOTE], title="READ ME", props={})

link(12, 0, 13, 0)
link(1, 0, 13, 1)
link(13, 0, 2, 0)
link(1, 1, 3, 0)
link(3, 0, 4, 0)
link(4, 0, 5, 0)
link(5, 0, 6, 0)
link(2, 0, 7, 0)
link(6, 0, 7, 1)
link(7, 0, 8, 0)
link(8, 0, 14, 0)
link(14, 0, 9, 0)
link(8, 0, 15, 0)
link(15, 0, 16, 0)

g.save(OUT, "image-to-svg-layered", groups=[("Transparency", [420, 310, 360, 500], "#444")],
       scale=0.8, offset=(40, 60))
