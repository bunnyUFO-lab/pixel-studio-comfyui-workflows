"""Generates workflows/image-to-svg/layered-svg-ai.json.

Edit the constants below and re-run:  python tools/build_layered_svg_ai.py

layered-svg with a Qwen-Image-2.1 repaint step first: the edit model redraws the image as flat,
cel-shaded art in your palette (it understands what things are: spiders brown, webs beige...), then
Layered Vectorize traces it. Node settings follow ComfyUI's official Qwen Image 2.1 edit template
(int8 model + Qwen3-VL 8B int8 text encoder, 25 steps, CFG 1, euler/simple), without its optional
prompt enhancer. First test (Tangle Heart, 45 s on an RTX 4090): composition and details kept, but
the prompt is the main thing to tune. The repaint prompt comes from the Vector Art Prompt template node.
"""
from comfy_workflow import (BLUE, GREEN, QWEN_HF as HF, QWEN_MODEL, QWEN_TE, QWEN_VAE, UPSCALER, UPSCALER_URL,
                            YELLOW, WORKFLOWS, Graph, inp, out)

OUT = WORKFLOWS / "image-to-svg" / "layered-svg-ai.json"

MAX_SIZE = 2048
SAVE_PREFIX = "svg/layered_ai"
# colors, palette, min_region_px, ink_threshold, tone_threshold, detail_threshold, detail_min_px,
# shadow_opacity, highlight_opacity, curve_smoothness, objects, shadows, highlights, details, ink, remove_background,
# flat_stacking, overlap_px, line_min_px, line_max_px, inset_under_ink_px, tone_style
LAYERED = [6, "", 2000, 22, 6.0, 25.0, 60, 0.3, 0.35, 1.1, "per region", True, True, True, True, True, "stacked", 2, 0, 0, 2, "tinted"]

NOTE = f"""Layered SVG with AI repaint (Qwen-Image-2.1)

1. Load your image (blue).
2. Vector Art Prompt (green) builds the repaint instruction. The default (faithful flat redraw) keeps every
   object recognizable, one flat base colour per element, the same colour for similar elements, simple
   hard-edged highlights/shadows, no gradients or texture, no lost details. Change style (cartoon, logo, icon,
   sticker, line art, game UI...), colours / palette (also used by Layered Vectorize), detail (fewer details =
   fewer shapes), outlines and shading; add specifics in 'subject' ("spiders dark brown"). See "Final prompt".
3. Run. The repainted image is saved (output/{SAVE_PREFIX}_repaint_*.png) and shown,
   then vectorized into layers (flats, shadows, highlights, ink, details).
4. Optional: to force exact colours in the SVG, type hex colours in Layered Vectorize -> palette
   (include one for every area, or areas snap to the nearest one). Empty = automatic colours.

Compare with / without the repaint: select "Qwen repaint" KSampler and press Ctrl+B
(bypass) - then the original image goes straight to the vectorizer.
Repaint takes ~45 s on an RTX 4090 the first time (model loading), less afterwards.
Different seed = different repaint; keep one you like (control_after_generate = fixed).

Models (~17.3 GB, MODELS preset env/svg.env): Qwen-Image-2.1 int8, Qwen3-VL 8B int8 text
encoder, Qwen-Image-2.1 VAE. Licence: Qwen Research (non-commercial)."""


def models(name, url, directory, type_):
    return {"Node name for S&R": type_, "models": [{"name": name, "url": url, "directory": directory}]}


g = Graph()
node, link = g.node, g.link

node(1, "LoadImage", [0, 0], [400, 420], outputs=[out("IMAGE", "IMAGE"), out("MASK", "MASK")],
     widgets=["example.png", "image"], title="Input image", color=BLUE)
node(11, "Note", [0, 470], [460, 560], widgets=[NOTE], title="READ ME", props={})

# Qwen-Image-2.1 repaint (edit) - same structure as the official ComfyUI template
node(20, "UNETLoader", [440, 0], [340, 82], outputs=[out("MODEL", "MODEL")], widgets=[QWEN_MODEL, "default"],
     title="Qwen-Image-2.1", props=models(QWEN_MODEL, f"{HF}/diffusion_models/{QWEN_MODEL}", "diffusion_models", "UNETLoader"))
node(21, "QwenImage21Cache", [440, 120], [340, 82], inputs=[inp("model", "MODEL")], outputs=[out("MODEL", "MODEL")],
     widgets=["auto", "default"])
node(22, "CLIPLoader", [440, 240], [340, 106], outputs=[out("CLIP", "CLIP")], widgets=[QWEN_TE, "qwen_image", "default"],
     title="Qwen3-VL text encoder", props=models(QWEN_TE, f"{HF}/text_encoders/{QWEN_TE}", "text_encoders", "CLIPLoader"))
node(23, "VAELoader", [440, 390], [340, 58], outputs=[out("VAE", "VAE")], widgets=[QWEN_VAE],
     props=models(QWEN_VAE, f"{HF}/vae/{QWEN_VAE}", "vae", "VAELoader"))
node(28, "VectorArtPrompt", [820, -520], [440, 420], outputs=[out("prompt", "STRING"), out("colors", "INT"),
     out("palette", "STRING")],
     widgets=["redraw <image1>", "", "faithful flat redraw", 6, "", "medium", "bold dark outlines",
              "one shadow + one highlight", "plain white", ""], title="Vector Art Prompt (repaint)", color=GREEN)
node(29, "PreviewAny", [1300, -520], [400, 240], inputs=[inp("source", "*")], title="Final prompt")
node(24, "TextEncodeQwenImage21", [820, 0], [440, 330],
     inputs=[inp("clip", "CLIP"), inp("vae", "VAE"), inp("images.image_1", "IMAGE"), inp("prompt", "STRING", "prompt")],
     outputs=[out("positive", "CONDITIONING"), out("negative", "CONDITIONING"), out("latent", "LATENT")],
     widgets=["", "", 1024], title="Encode repaint prompt")
node(25, "KSampler", [820, 370], [320, 262],
     inputs=[inp("model", "MODEL"), inp("positive", "CONDITIONING"), inp("negative", "CONDITIONING"),
             inp("latent_image", "LATENT")],
     outputs=[out("LATENT", "LATENT")], widgets=[1234, "randomize", 25, 1.0, "euler", "simple", 1.0],
     title="Qwen repaint (Ctrl+B to skip)", color=YELLOW)
node(26, "VAEDecode", [820, 670], [210, 46], inputs=[inp("samples", "LATENT"), inp("vae", "VAE")],
     outputs=[out("IMAGE", "IMAGE")])
node(27, "SaveImage", [1300, 0], [400, 420], inputs=[inp("images", "IMAGE")],
     widgets=[f"{SAVE_PREFIX}_repaint"], title="Repainted image")

# Vectorize
node(12, "UpscaleModelLoader", [1300, 460], [320, 58], outputs=[out("UPSCALE_MODEL", "UPSCALE_MODEL")],
     widgets=[UPSCALER], title="AI upscaler (Real-ESRGAN anime)",
     props=models(UPSCALER, UPSCALER_URL, "upscale_models", "UpscaleModelLoader"))
node(13, "ImageUpscaleWithModel", [1300, 560], [320, 46],
     inputs=[inp("upscale_model", "UPSCALE_MODEL"), inp("image", "IMAGE")], outputs=[out("IMAGE", "IMAGE")],
     title="AI upscale x4")
node(2, "ImageScaleToMaxDimension", [1300, 650], [320, 82], inputs=[inp("image", "IMAGE")],
     outputs=[out("IMAGE", "IMAGE")], widgets=["lanczos", MAX_SIZE], title="Resize (longest side)")
node(8, "LayeredVectorize", [1740, 0], [360, 520], inputs=[inp("image", "IMAGE"), inp("colors", "INT", "colors"),
                                                          inp("palette", "STRING", "palette")],
     outputs=[out("SVG_String", "STRING")], widgets=LAYERED, title="Layered Vectorize", color=YELLOW)
node(14, "TS_SVGStringToSVGBytesIO", [2140, 0], [300, 26], inputs=[inp("SVG_String", "STRING")],
     outputs=[out("SVG", "SVG")], title="SVG text -> SVG")
node(9, "SaveSVGNode", [2140, 80], [420, 520], inputs=[inp("svg", "SVG")], outputs=[out("svg", "SVG")],
     widgets=[SAVE_PREFIX], title="Save SVG (output/svg/)", color=GREEN)
node(15, "SVGStats", [2600, 0], [300, 70], inputs=[inp("SVG_String", "STRING")],
     outputs=[out("info", "STRING"), out("shapes", "INT"), out("size_kb", "FLOAT")], title="SVG Stats")
node(16, "PreviewAny", [2600, 110], [340, 200], inputs=[inp("source", "*")], title="Shapes & file size")

link(20, 0, 21, 0)
link(22, 0, 24, 0)
link(23, 0, 24, 1)
link(1, 0, 24, 2)
link(21, 0, 25, 0)
link(24, 0, 25, 1)
link(24, 1, 25, 2)
link(24, 2, 25, 3)
link(25, 0, 26, 0)
link(23, 0, 26, 1)
link(26, 0, 27, 0)
link(12, 0, 13, 0)
link(26, 0, 13, 1)
link(13, 0, 2, 0)
link(2, 0, 8, 0)
link(8, 0, 14, 0)
link(14, 0, 9, 0)
link(8, 0, 15, 0)
link(15, 0, 16, 0)
link(28, 0, 24, 3)           # template prompt -> Qwen edit encoder
link(28, 0, 29, 0)
link(28, 1, 8, 1)            # the same colours / palette drive the vectorizer
link(28, 2, 8, 2)

g.save(OUT, "image-to-svg-layered-ai", groups=[("Qwen-Image-2.1 repaint", [420, -60, 860, 800], "#3f789e"),
                                               ("Vectorize", [1280, -60, 1680, 820], "#8A8")],
       scale=0.55, offset=(40, 80))
