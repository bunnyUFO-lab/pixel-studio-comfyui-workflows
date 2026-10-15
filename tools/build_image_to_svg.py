"""Generates workflows/image-to-svg/vtracer-svg.json.

Edit the constants below and re-run:  python tools/build_image_to_svg.py

Defaults were tuned on a flat, anti-aliased character with transparency:
- No palette quantize step: PIL's median-cut palette is dominated by background and edge pixels,
  so small colored areas (eyes, cheeks) get merged into the wrong color. VTracer's own clustering
  keeps them when color_precision stays at 6; a high layer_difference keeps the path count low.
- Trace at 2048px: noticeably smoother curves than 1024px.
- Hard alpha edge: semi-transparent anti-aliased edge pixels otherwise become a fringe of tiny
  sliver paths around the subject (4x larger file).
Then refined on a detailed hand-drawn character (1408x768 JPEG):
- AI upscale (Real-ESRGAN anime 6B) before resizing: plain resizing keeps JPEG noise and soft
  edges, which trace as stair-stepped outlines. The AI upscale redraws crisp edges: smooth lines
  and half the file size at the same resolution.
- Smoother VTracer settings (corner 90, splice 60, length 6, speckle 10): fewer wobbles and corners
  without losing small details like spiders and web strands.
- Core Save SVG node: saves under output/ (subfolders allowed) and shows the result in the UI.
- Detail: at 2048px with layer_difference 40 / filter_speckle 10, small low-contrast details (dark-brown
  spiders on a black visor, their 3px eyes) merged into blobs. 4096px with layer_difference 24 and
  filter_speckle 6 keeps legs and eyes with smooth outlines (1.4 MB vs 0.55 MB for that image).
"""
from comfy_workflow import ALPHA_CUTOFF, BLUE, GREEN, UPSCALER, UPSCALER_URL, YELLOW, WORKFLOWS, Graph, inp, out

OUT = WORKFLOWS / "image-to-svg" / "vtracer-svg.json"

MAX_SIZE = 4096      # longest side in px before tracing (2048 for simple art / smaller files)
# VTracer color mode: hierarchical, mode, filter_speckle, color_precision, layer_difference,
# corner_threshold, length_threshold, max_iterations, splice_threshold, path_precision
VTRACER = ["stacked", "spline", 6, 6, 24, 90, 6.0, 20, 60, 3]
SAVE_PREFIX = "svg/image_to_svg"   # under output/; subfolders allowed



NOTE = f"""Image to SVG (local, VTracer) - logos, flat art, stylized characters, illustrations

Pipeline: image -> AI upscale x4 (crisp edges) -> resize (longest side {MAX_SIZE}px)
-> hard transparency edge -> VTracer -> SVG.
SVGs are saved to output/{SAVE_PREFIX}_NNNNN_.svg and shown in the Save SVG node.
"Shapes & file size" shows the shape count, colors and saved file size.
Defaults favour detail (4096px). Simple art / smaller files: set both resize nodes to 2048,
layer_difference 40 and filter_speckle 10 (about 1/3 the file size, loses tiny details).
Large, clean sources (or simple logos): bypass the AI upscale node (Ctrl+B) to save time.
Transparent PNGs keep their transparency. With an opaque image, a plain
background becomes one shape you can delete in your editor.

Tuning (yellow VTracer node):
- Small details lost (pupils, highlights): lower filter_speckle (8) and/or layer_difference (24-32).
- Too many shapes / big file: raise layer_difference (64) or filter_speckle (24).
- Colors wrong or merged: keep color_precision at 6+ (lowering it shifts colors).
- Jagged corners: raise corner_threshold (90-120); keep mode = spline.
- Shapes with holes behave oddly in editors: hierarchical = cutout.
- Simple flat logos: resize nodes down to 1024-2048, and the AI upscale can be bypassed.
- Still jagged: make sure mode = spline and the AI upscale is enabled; raise corner_threshold (120).
- Collapses into a few big blobs: input too soft/blurry - lower layer_difference (16).
Photos and soft gradients come out posterized - that is how tracing works.

VTracer parameters (full reference in README):
- hierarchical: stacked = shapes layered (small, no gaps); cutout = no overlaps (easier to edit).
- mode: spline = smooth curves; polygon = straight lines; none = pixel outlines.
- filter_speckle: drop patches smaller than N x N px (at the resized size).
- color_precision: bits per color channel for grouping (max 8); higher = more colors.
- layer_difference: how different neighbor colors must be to stay separate; higher = flatter.
- corner_threshold: degrees for a sharp corner; higher = smoother outlines.
- length_threshold: max segment length (px); higher = fewer points, smaller file.
- max_iterations: curve smoothing passes (rarely change).
- splice_threshold: degrees to start a new spline; lower = follows wiggles closer.
- path_precision: decimals in coordinates; lower = smaller file.
Hard transparency edge value: pixels more transparent than this are dropped."""

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

# Transparency: resize the mask the same way, then make its edge hard.
node(3, "MaskToImage", [440, 340], [320, 26], inputs=[inp("mask", "MASK")], outputs=[out("IMAGE", "IMAGE")])
node(4, "ImageScaleToMaxDimension", [440, 410], [320, 82], inputs=[inp("image", "IMAGE")],
     outputs=[out("IMAGE", "IMAGE")], widgets=["lanczos", MAX_SIZE], title="Resize mask (keep = image)")
node(5, "ImageToMask", [440, 540], [320, 58], inputs=[inp("image", "IMAGE")], outputs=[out("MASK", "MASK")],
     widgets=["red"])
node(6, "ThresholdMask", [440, 640], [320, 58], inputs=[inp("mask", "MASK")], outputs=[out("MASK", "MASK")],
     widgets=[ALPHA_CUTOFF], title="Hard transparency edge")
node(7, "JoinImageWithAlpha", [440, 740], [320, 46], inputs=[inp("image", "IMAGE"), inp("alpha", "MASK")],
     outputs=[out("IMAGE", "IMAGE")], title="Apply transparency")

node(8, "TS_ImageToSVGStringColor_Vtracer", [800, 0], [340, 320], inputs=[inp("image", "IMAGE")],
     outputs=[out("STRING", "STRING")], widgets=VTRACER, title="VTracer (color)", color=YELLOW)
node(14, "TS_SVGStringToSVGBytesIO", [1180, 0], [300, 26], inputs=[inp("SVG_String", "STRING")],
     outputs=[out("SVG", "SVG")], title="SVG text -> SVG")
node(9, "SaveSVGNode", [1180, 80], [420, 520], inputs=[inp("svg", "SVG")], outputs=[out("svg", "SVG")],
     widgets=[SAVE_PREFIX], title="Save SVG (output/svg/)", color=GREEN)
node(15, "SVGStats", [1640, 0], [300, 70], inputs=[inp("SVG_String", "STRING")],
     outputs=[out("info", "STRING"), out("shapes", "INT"), out("size_kb", "FLOAT")], title="SVG Stats")
node(16, "PreviewAny", [1640, 110], [340, 200], inputs=[inp("source", "*")], title="Shapes & file size")
node(11, "Note", [0, 470], [460, 700], widgets=[NOTE], title="READ ME", props={})

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

g.save(OUT, "image-to-svg-vtracer", groups=[("Transparency", [420, 310, 360, 500], "#444")],
       scale=0.8, offset=(40, 60))
