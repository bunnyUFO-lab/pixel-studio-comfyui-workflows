"""Generates workflows/pixel-art/prompt-to-pixel.json.

Edit the constants below and re-run:  python tools/build_pixel_prompt_to_pixel.py

Prompt -> image -> true pixel art: Z-Image Turbo (Apache-2.0, 8 steps) with a pixel-art style LoRA draws the
character/object on a plain background, BiRefNet cuts it out, then the repo's Pixelize node puts it on an exact
grid at native resolution ('pixels' = subject height) with a limited palette and hard alpha.
"""
from comfy_workflow import (BIREFNET, BIREFNET_URL, BLUE, GREEN, PIXEL_LORA, PIXEL_LORA_URL, YELLOW, WORKFLOWS,
                            ZIMAGE_HF, ZIMAGE_MODEL, ZIMAGE_TE, ZIMAGE_VAE, Graph, inp, out)

OUT = WORKFLOWS / "pixel-art" / "prompt-to-pixel.json"

SUBJECT = "a fantasy knight with a blue cape and a sword, standing"
STYLE, VIEW, DETAIL, OUTLINE = "clean game sprite", "front three-quarter", "auto (from pixels)", "dark outline"
WIDTH, HEIGHT = 1024, 1024
LORA_STRENGTH = 1.0
STEPS, SHIFT = 8, 3.0
SEED = 42

PIXELS = 64            # subject height in art pixels
COLORS = 0             # 0 = automatic: the fewest colours within COLOR_ERROR
COLOR_ERROR = 20.0     # higher = fewer colours (12 faithful, 20 simplified, 30 very few)
MAX_COLORS = 32        # Pixel Art Check limit (automatic palettes stay under it)
SMOOTH = 0.0            # the LoRA already draws flat colours; smoothing would erase thin highlights
# images, pixels, measure, method, smooth, colors, color_error, palette, shared_palette, background, bg_tolerance, cleanup, edge_cleanup,
# outline, padding, preview_scale
CUTOUT = [True, False, 0.02, 8.0, 0.2]   # drop_shadow, enclosed_background, min_pocket, bg_tolerance, shadow_band
PIXELIZE = [PIXELS, "subject height", "pixeloe", SMOOTH, COLORS, COLOR_ERROR, "auto", True, "auto remove", 12.0, True, True, "auto", 1, 8]

NOTE = f"""Prompt -> pixel art

1. Pixel Art Prompt (green): type just the SUBJECT ("a swordsman in a red tunic"). The template adds the game-art
   style (clean sprite, 16-bit RPG, 8-bit, chibi, top-down, side-scroller, isometric, item/icon, monster, prop),
   the view, a detail level that fits 'pixels' (fewer details for small sprites), the colours, outline style,
   a plain background and the LoRA trigger. pixels / colors / palette set here also drive Pixelize and the check.
   See the full prompt in "Final prompt". Change the seed in the sampler for variations.
2. Model: Z-Image Turbo (Apache-2.0), 8 steps, CFG 1. "Pixel-art LoRA" gives the drawing a pixel-art look
   (flat colours, crisp outlines). Without it (Ctrl+B) you get a painted image that Pixelize still converts.
3. Pixelize (yellow): pixels / colors / palette come from the prompt node; colors 0 = automatic
   fewest colours (color_error {COLOR_ERROR}: higher = fewer), or a number / PICO-8, DB32, Endesga-32, GameBoy, hex list; smooth 0 (the art is already flat; raise it without the LoRA).
4. Run. Saved to output/pixel/:
   prompt_to_pixel_*.png      native size (1 art pixel = 1 image pixel, transparent) -> use in games
   prompt_to_pixel_x8_*.png   zoomed preview
   prompt_to_pixel_raw_*.png  the generated 1024px image (re-use it in image-to-pixel with other settings)
"Pixel Art Check" must say PASS."""

g = Graph()
node, link = g.node, g.link


def models(name, url, directory, type_):
    return {"Node name for S&R": type_, "models": [{"name": name, "url": url, "directory": directory}]}


node(1, "UNETLoader", [0, 0], [360, 82], outputs=[out("MODEL", "MODEL")], widgets=[ZIMAGE_MODEL, "default"],
     title="Z-Image Turbo", props=models(ZIMAGE_MODEL, f"{ZIMAGE_HF}/diffusion_models/{ZIMAGE_MODEL}", "diffusion_models", "UNETLoader"))
node(2, "LoraLoaderModelOnly", [0, 120], [360, 82], inputs=[inp("model", "MODEL")], outputs=[out("MODEL", "MODEL")],
     widgets=[PIXEL_LORA, LORA_STRENGTH], title="Pixel-art LoRA", color=YELLOW,
     props={"Node name for S&R": "LoraLoaderModelOnly", "models": [
         {"name": PIXEL_LORA, "url": PIXEL_LORA_URL, "directory": "loras"},
         {"name": "pixelart_redmond_zimage_turbo.safetensors", "directory": "loras",   # optional 'chunky' style (Apache-2.0)
          "url": "https://huggingface.co/artificialguybr/PIXELART-REDMOND-ZIMAGETURBO/resolve/main/%5BZImage.Turbo%5DPixelArt_Redmond.safetensors"}]})
node(3, "ModelSamplingAuraFlow", [0, 240], [360, 58], inputs=[inp("model", "MODEL")], outputs=[out("MODEL", "MODEL")],
     widgets=[SHIFT])
node(4, "CLIPLoader", [0, 340], [360, 106], outputs=[out("CLIP", "CLIP")], widgets=[ZIMAGE_TE, "lumina2", "default"],
     props=models(ZIMAGE_TE, f"{ZIMAGE_HF}/text_encoders/{ZIMAGE_TE}", "text_encoders", "CLIPLoader"))
node(5, "VAELoader", [0, 490], [360, 58], outputs=[out("VAE", "VAE")], widgets=[ZIMAGE_VAE],
     props=models(ZIMAGE_VAE, f"{ZIMAGE_HF}/vae/{ZIMAGE_VAE}", "vae", "VAELoader"))
node(23, "PixelArtPrompt", [400, -560], [420, 460], outputs=[out("prompt", "STRING"), out("pixels", "INT"),
     out("colors", "INT"), out("palette", "STRING")],
     widgets=[SUBJECT, STYLE, VIEW, PIXELS, DETAIL, COLORS, "auto", OUTLINE, "plain white", True, ""], title="Pixel Art Prompt", color=GREEN)
node(24, "PreviewAny", [860, -560], [420, 240], inputs=[inp("source", "*")], title="Final prompt")
node(6, "CLIPTextEncode", [400, 0], [420, 120], inputs=[inp("clip", "CLIP"), inp("text", "STRING", "text")],
     outputs=[out("CONDITIONING", "CONDITIONING")], widgets=[""], title="Encode prompt")
node(7, "ConditioningZeroOut", [400, 260], [240, 26], inputs=[inp("conditioning", "CONDITIONING")],
     outputs=[out("CONDITIONING", "CONDITIONING")])
node(8, "EmptySD3LatentImage", [400, 330], [300, 106], outputs=[out("LATENT", "LATENT")], widgets=[WIDTH, HEIGHT, 1])
node(9, "KSampler", [860, 0], [320, 262],
     inputs=[inp("model", "MODEL"), inp("positive", "CONDITIONING"), inp("negative", "CONDITIONING"),
             inp("latent_image", "LATENT")],
     outputs=[out("LATENT", "LATENT")], widgets=[SEED, "randomize", STEPS, 1.0, "res_multistep", "simple", 1.0])
node(10, "VAEDecode", [860, 310], [200, 46], inputs=[inp("samples", "LATENT"), inp("vae", "VAE")], outputs=[out("IMAGE", "IMAGE")])
node(11, "SaveImage", [860, 400], [320, 360], inputs=[inp("images", "IMAGE")], widgets=["pixel/prompt_to_pixel_raw"],
     title="Save generated image")
node(12, "LoadBackgroundRemovalModel", [1220, 0], [320, 58], outputs=[out("bg_model", "BACKGROUND_REMOVAL")],
     widgets=[BIREFNET], title="BiRefNet", props=models(BIREFNET, BIREFNET_URL, "background_removal", "LoadBackgroundRemovalModel"))
node(13, "RemoveBackground", [1220, 100], [320, 46], inputs=[inp("bg_removal_model", "BACKGROUND_REMOVAL"), inp("image", "IMAGE")],
     outputs=[out("mask", "MASK")])
node(14, "InvertMask", [1220, 180], [320, 26], inputs=[inp("mask", "MASK")], outputs=[out("MASK", "MASK")])
node(15, "JoinImageWithAlpha", [1220, 250], [320, 46], inputs=[inp("image", "IMAGE"), inp("alpha", "MASK")],
     outputs=[out("IMAGE", "IMAGE")], title="Remove background (Ctrl+B to skip)")
node(25, "CleanCutout", [1220, 330], [320, 160], inputs=[inp("images", "IMAGE")],
     outputs=[out("image", "IMAGE"), out("removed", "MASK"), out("info", "STRING")], widgets=CUTOUT,
     title="Clean cutout (drop shadow, background pockets)", color=YELLOW)
node(16, "Pixelize", [1580, 0], [360, 440], inputs=[inp("images", "IMAGE"), inp("pixels", "INT", "pixels"),
                                                   inp("colors", "INT", "colors"), inp("palette", "STRING", "palette")],
     outputs=[out("pixel_art", "IMAGE"), out("preview", "IMAGE"), out("palette", "STRING"), out("info", "STRING")],
     widgets=PIXELIZE, title="Pixelize", color=YELLOW)
node(17, "SaveImage", [1980, 0], [360, 300], inputs=[inp("images", "IMAGE")], widgets=["pixel/prompt_to_pixel"],
     title="Save pixel art (native size)", color=GREEN)
node(18, "SaveImage", [1980, 340], [360, 420], inputs=[inp("images", "IMAGE")], widgets=["pixel/prompt_to_pixel_x8"],
     title="Save preview (x8)")
node(19, "PixelArtCheck", [1580, 490], [360, 106], inputs=[inp("images", "IMAGE"), inp("expected_height", "INT", "expected_height")],
     outputs=[out("report", "STRING"), out("passed", "BOOLEAN")], widgets=[MAX_COLORS, PIXELS, True], title="Pixel Art Check")
node(20, "PreviewAny", [1580, 640], [360, 200], inputs=[inp("source", "*")], title="Check report")
node(21, "PreviewAny", [1980, 800], [360, 160], inputs=[inp("source", "*")], title="Palette")
node(22, "Note", [400, 480], [420, 460], widgets=[NOTE], title="READ ME", props={})

link(1, 0, 2, 0)
link(2, 0, 3, 0)
link(3, 0, 9, 0)
link(4, 0, 6, 0)
link(6, 0, 7, 0)
link(6, 0, 9, 1)
link(7, 0, 9, 2)
link(8, 0, 9, 3)
link(9, 0, 10, 0)
link(5, 0, 10, 1)
link(10, 0, 11, 0)
link(12, 0, 13, 0)
link(10, 0, 13, 1)
link(13, 0, 14, 0)
link(10, 0, 15, 0)
link(14, 0, 15, 1)
link(15, 0, 25, 0)
link(25, 0, 16, 0)
link(16, 0, 17, 0)
link(16, 1, 18, 0)
link(16, 0, 19, 0)
link(19, 0, 20, 0)
link(16, 2, 21, 0)
link(23, 0, 6, 1)            # template prompt -> text encoder
link(23, 0, 24, 0)
link(23, 1, 16, 1)           # the same pixels / colors / palette drive Pixelize and the check
link(23, 2, 16, 2)
link(23, 3, 16, 3)
link(23, 1, 19, 1)

g.save(OUT, "pixel-art-prompt-to-pixel",
       groups=[("Prompt template", [380, -620, 920, 540], "#a1309b"),
               ("Z-Image Turbo + pixel-art LoRA", [-20, -60, 1220, 860], "#444"),
               ("Background removal (BiRefNet)", [1200, -60, 360, 380], "#444")],
       scale=0.6, offset=(40, 80))
