"""Generates workflows/image-to-svg/prompt-to-svg.json.

Edit the constants below and re-run:  python tools/build_prompt_to_svg.py

Prompt -> SVG: the Vector Art Prompt template turns a short subject into a prompt for flat, vector-friendly art
(logo, icon, sticker, cartoon character, line art, game UI...), Z-Image Turbo (Apache-2.0, 8 steps) draws it,
then the image is upscaled and traced by Layered Vectorize into named layers (flats, shadows, highlights, ink,
details). The template's colours / palette also drive the vectorizer, so drawing and tracing agree.
"""
from comfy_workflow import (BLUE, GREEN, UPSCALER, UPSCALER_URL, YELLOW, WORKFLOWS, ZIMAGE_HF, ZIMAGE_MODEL, ZIMAGE_TE,
                            ZIMAGE_VAE, Graph, inp, out)

OUT = WORKFLOWS / "image-to-svg" / "prompt-to-svg.json"

SUBJECT = "a friendly fox mascot head"
STYLE, COLORS, DETAIL, OUTLINE, SHADING = "cartoon character", 6, "low", "bold dark outlines", "one shadow + one highlight"
WIDTH, HEIGHT, STEPS, SHIFT, SEED = 1024, 1024, 8, 3.0, 7
MAX_SIZE = 2048
SAVE_PREFIX = "svg/prompt_to_svg"
# colors, palette, min_region_px, ink_threshold, tone_threshold, detail_threshold, detail_min_px,
# shadow_opacity, highlight_opacity, curve_smoothness, objects, shadows, highlights, details, ink, remove_background,
# flat_stacking, overlap_px, line_min_px, line_max_px, inset_under_ink_px, tone_style
LAYERED = [COLORS, "", 2000, 22, 6.0, 25.0, 60, 0.3, 0.35, 1.1, "per region", True, True, True, True, True, "stacked", 2, 0, 0, 2, "tinted"]

NOTE = f"""Prompt -> layered SVG

1. Vector Art Prompt (green): type just the SUBJECT ("a fox mascot head", "a treasure chest icon").
   style = flat vector illustration, cartoon character, logo / emblem, icon, sticker, line art, game item / UI art,
   stylized game character. colors / palette (hex or PICO-8, DB32...) are asked for in the prompt AND used by
   Layered Vectorize. detail: fewer details = fewer, bigger shapes. outlines become the ink layer; shading
   'one shadow + one highlight' fills the shadow/highlight layers, 'flat' gives the fewest shapes.
   See the full prompt in "Final prompt". Change the seed in the sampler for variations.
2. Z-Image Turbo (Apache-2.0) draws it at {WIDTH}x{HEIGHT} in ~10 s; the drawing is saved too
   (output/{SAVE_PREFIX}_image_*.png).
3. AI upscale x4 -> {MAX_SIZE} px -> Layered Vectorize (same settings as layered-svg) -> output/{SAVE_PREFIX}_*.svg.
"Shapes & file size" shows how simple the SVG came out."""


def models(name, url, directory, type_):
    return {"Node name for S&R": type_, "models": [{"name": name, "url": url, "directory": directory}]}


g = Graph()
node, link = g.node, g.link

node(1, "VectorArtPrompt", [0, 0], [420, 420], outputs=[out("prompt", "STRING"), out("colors", "INT"), out("palette", "STRING")],
     widgets=["generate", SUBJECT, STYLE, COLORS, "", DETAIL, OUTLINE, SHADING, "plain white", ""], title="Vector Art Prompt", color=GREEN)
node(2, "PreviewAny", [0, 460], [420, 240], inputs=[inp("source", "*")], title="Final prompt")
node(3, "Note", [0, 740], [460, 440], widgets=[NOTE], title="READ ME", props={})

node(10, "UNETLoader", [460, 0], [360, 82], outputs=[out("MODEL", "MODEL")], widgets=[ZIMAGE_MODEL, "default"],
     title="Z-Image Turbo", props=models(ZIMAGE_MODEL, f"{ZIMAGE_HF}/diffusion_models/{ZIMAGE_MODEL}", "diffusion_models", "UNETLoader"))
node(11, "ModelSamplingAuraFlow", [460, 120], [360, 58], inputs=[inp("model", "MODEL")], outputs=[out("MODEL", "MODEL")],
     widgets=[SHIFT])
node(12, "CLIPLoader", [460, 220], [360, 106], outputs=[out("CLIP", "CLIP")], widgets=[ZIMAGE_TE, "lumina2", "default"],
     props=models(ZIMAGE_TE, f"{ZIMAGE_HF}/text_encoders/{ZIMAGE_TE}", "text_encoders", "CLIPLoader"))
node(13, "VAELoader", [460, 370], [360, 58], outputs=[out("VAE", "VAE")], widgets=[ZIMAGE_VAE],
     props=models(ZIMAGE_VAE, f"{ZIMAGE_HF}/vae/{ZIMAGE_VAE}", "vae", "VAELoader"))
node(14, "CLIPTextEncode", [460, 470], [360, 100], inputs=[inp("clip", "CLIP"), inp("text", "STRING", "text")],
     outputs=[out("CONDITIONING", "CONDITIONING")], widgets=[""], title="Encode prompt")
node(15, "ConditioningZeroOut", [460, 610], [240, 26], inputs=[inp("conditioning", "CONDITIONING")],
     outputs=[out("CONDITIONING", "CONDITIONING")])
node(16, "EmptySD3LatentImage", [460, 680], [300, 106], outputs=[out("LATENT", "LATENT")], widgets=[WIDTH, HEIGHT, 1])
node(17, "KSampler", [860, 0], [320, 262],
     inputs=[inp("model", "MODEL"), inp("positive", "CONDITIONING"), inp("negative", "CONDITIONING"),
             inp("latent_image", "LATENT")],
     outputs=[out("LATENT", "LATENT")], widgets=[SEED, "randomize", STEPS, 1.0, "res_multistep", "simple", 1.0])
node(18, "VAEDecode", [860, 310], [200, 46], inputs=[inp("samples", "LATENT"), inp("vae", "VAE")], outputs=[out("IMAGE", "IMAGE")])
node(19, "SaveImage", [860, 400], [320, 360], inputs=[inp("images", "IMAGE")], widgets=[f"{SAVE_PREFIX}_image"],
     title="Generated image")

node(20, "UpscaleModelLoader", [1220, 0], [320, 58], outputs=[out("UPSCALE_MODEL", "UPSCALE_MODEL")],
     widgets=[UPSCALER], title="AI upscaler (Real-ESRGAN anime)", props=models(UPSCALER, UPSCALER_URL, "upscale_models", "UpscaleModelLoader"))
node(21, "ImageUpscaleWithModel", [1220, 100], [320, 46], inputs=[inp("upscale_model", "UPSCALE_MODEL"), inp("image", "IMAGE")],
     outputs=[out("IMAGE", "IMAGE")], title="AI upscale x4")
node(22, "ImageScaleToMaxDimension", [1220, 190], [320, 82], inputs=[inp("image", "IMAGE")],
     outputs=[out("IMAGE", "IMAGE")], widgets=["lanczos", MAX_SIZE], title="Resize (longest side)")
node(23, "LayeredVectorize", [1580, 0], [360, 520],
     inputs=[inp("image", "IMAGE"), inp("colors", "INT", "colors"), inp("palette", "STRING", "palette")],
     outputs=[out("SVG_String", "STRING")], widgets=LAYERED, title="Layered Vectorize", color=YELLOW)
node(24, "TS_SVGStringToSVGBytesIO", [1980, 0], [300, 26], inputs=[inp("SVG_String", "STRING")],
     outputs=[out("SVG", "SVG")], title="SVG text -> SVG")
node(25, "SaveSVGNode", [1980, 80], [420, 520], inputs=[inp("svg", "SVG")], outputs=[out("svg", "SVG")],
     widgets=[SAVE_PREFIX], title="Save SVG (output/svg/)", color=GREEN)
node(26, "SVGStats", [2440, 0], [300, 70], inputs=[inp("SVG_String", "STRING")],
     outputs=[out("info", "STRING"), out("shapes", "INT"), out("size_kb", "FLOAT")], title="SVG Stats")
node(27, "PreviewAny", [2440, 110], [340, 200], inputs=[inp("source", "*")], title="Shapes & file size")

link(1, 0, 2, 0)
link(1, 0, 14, 1)            # template prompt -> text encoder
link(1, 1, 23, 1)            # the same colours / palette drive the vectorizer
link(1, 2, 23, 2)
link(10, 0, 11, 0)
link(11, 0, 17, 0)
link(12, 0, 14, 0)
link(14, 0, 15, 0)
link(14, 0, 17, 1)
link(15, 0, 17, 2)
link(16, 0, 17, 3)
link(17, 0, 18, 0)
link(13, 0, 18, 1)
link(18, 0, 19, 0)
link(20, 0, 21, 0)
link(18, 0, 21, 1)
link(21, 0, 22, 0)
link(22, 0, 23, 0)
link(23, 0, 24, 0)
link(24, 0, 25, 0)
link(23, 0, 26, 0)
link(26, 0, 27, 0)

g.save(OUT, "image-to-svg-prompt-to-svg", groups=[("Z-Image Turbo", [440, -60, 760, 880], "#444"),
                                                  ("Vectorize", [1200, -60, 1600, 700], "#8A8")],
       scale=0.55, offset=(40, 80))
