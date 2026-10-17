"""Generates workflows/pixel-art/prompt-to-pixel-krea2.json.

Edit the constants below and re-run:  python tools/build_pixel_prompt_to_pixel_krea2.py

The same graph as prompt-to-pixel with Krea 2 Turbo instead of Z-Image Turbo: Krea 2 (8 steps, no CFG) with one of
e-n-v-y's k2-pixel LoRAs draws the subject on a plain background, BiRefNet cuts it out, then either the repo's
Pixelize (default) or the LoRA author's Krea 2 Pixel Art Refiner puts it on the grid. The k2-pixel LoRAs draw on a
fixed grid of 32, 64 or 128 cells across the 1024 px image; the refiner collapses exactly that grid (most common
colour per cell, OKLab palette), so with it the sprite is as tall as the subject is in grid cells, not 'pixels'.
"""
from comfy_workflow import (BIREFNET, BIREFNET_URL, GREEN, K2_PIXEL_HF, K2_PIXEL_LORAS, KREA2_HF, KREA2_MODEL, KREA2_TE,
                            KREA2_VAE, YELLOW, WORKFLOWS, Graph, inp, out)

OUT = WORKFLOWS / "pixel-art" / "prompt-to-pixel-krea2.json"

SUBJECT = "a fantasy knight with a blue cape and a sword, standing"
STYLE, VIEW, DETAIL, OUTLINE = "clean game sprite", "front three-quarter", "auto (from pixels)", "dark outline"
WIDTH, HEIGHT = 1024, 1024
GRID = 64              # k2-pixel LoRA: 32, 64 or 128 cells across the 1024 px image
LORA_STRENGTH = 1.0    # the author gives none; 1.0 is the LoRA's own default
TRIGGER = "pixel art"  # the LoRA's trigger; the author says k2-pixel64 works best without it
USE_TRIGGER = GRID != 64
FILL = 85              # % of the canvas the subject fills: the fixed grid needs a big subject for enough art pixels
STEPS, CFG = 8, 1.0    # Turbo is distilled: 8 steps, no CFG (cfg 1 in ComfyUI)
SEED = 42
REFINER = False        # Clean-up: False = Pixelize (finds the subject's grid), True = Krea 2 Pixel Art Refiner

PIXELS = 64            # subject height in art pixels (Pixelize); the refiner uses GRID
COLORS = 0             # 0 = automatic: the fewest colours within COLOR_ERROR
COLOR_ERROR = 20.0
MAX_COLORS = 32        # Pixel Art Check limit
REFINER_COLORS = 24    # the refiner's palette size (its default)
CUTOUT = [True, False, 0.02, 8.0, 0.2]   # drop_shadow, enclosed_background, min_pocket, bg_tolerance, shadow_band
PIXELIZE = [PIXELS, "subject height", "pixeloe", 0.0, COLORS, COLOR_ERROR, "auto", True, "auto remove", 12.0, True, True, "auto", 1, 8]

NOTE = f"""Prompt -> pixel art with Krea 2

1. Pixel Art Prompt (green): type just the SUBJECT. The template adds the game-art style, view, detail, colours,
   outline and a plain background. For Krea 2 it asks for a big subject (fill {FILL}%) and uses the k2-pixel
   trigger "{TRIGGER}" (lora_trigger off for k2-pixel64, which the author says works best without it).
2. Model: Krea 2 Turbo (Krea 2 Community Licence), 8 steps, no CFG, 1024 px. "Pixel-art LoRA" = k2-pixel32 / 64 / 128
   (MIT): the LoRA draws on a grid of that many cells across the image. Pick it from the size you want.
3. Clean-up, switch "Clean-up: author's refiner?":
   off (default) = Pixelize: finds the subject, makes it 'pixels' tall, automatic palette (like prompt-to-pixel).
   on = Krea 2 Pixel Art Refiner: collapses the LoRA's own grid (set its width/height = the LoRA's number), most
   common colour per cell, {REFINER_COLORS} colours, then Grid Cutout keeps the cells BiRefNet marks as subject.
   The sprite is then as tall as the subject is in grid cells (about {FILL}% of the grid), not 'pixels'.
4. Run. Saved to output/pixel/:
   prompt_to_pixel_krea2_*.png      native size, transparent -> use in games
   prompt_to_pixel_krea2_x8_*.png   zoomed preview
   prompt_to_pixel_krea2_raw_*.png  the generated 1024 px image
"Pixel Art Check" must say PASS (it does not check the height: the refiner's height follows the drawing)."""

g = Graph()
node, link = g.node, g.link


def models(name, url, directory, type_):
    return {"Node name for S&R": type_, "models": [{"name": name, "url": url, "directory": directory}]}


node(1, "UNETLoader", [0, 0], [360, 82], outputs=[out("MODEL", "MODEL")], widgets=[KREA2_MODEL, "default"],
     title="Krea 2 Turbo", props=models(KREA2_MODEL, f"{KREA2_HF}/diffusion_models/{KREA2_MODEL}", "diffusion_models", "UNETLoader"))
node(2, "LoraLoaderModelOnly", [0, 120], [360, 82], inputs=[inp("model", "MODEL")], outputs=[out("MODEL", "MODEL")],
     widgets=[K2_PIXEL_LORAS[GRID], LORA_STRENGTH], title="Pixel-art LoRA", color=YELLOW,
     props={"Node name for S&R": "LoraLoaderModelOnly", "models": [
         {"name": f, "url": f"{K2_PIXEL_HF}/{f}", "directory": "loras"} for f in K2_PIXEL_LORAS.values()]})
node(4, "CLIPLoader", [0, 340], [360, 106], outputs=[out("CLIP", "CLIP")], widgets=[KREA2_TE, "krea2", "default"],
     props=models(KREA2_TE, f"{KREA2_HF}/text_encoders/{KREA2_TE}", "text_encoders", "CLIPLoader"))
node(5, "VAELoader", [0, 490], [360, 58], outputs=[out("VAE", "VAE")], widgets=[KREA2_VAE],
     props=models(KREA2_VAE, f"{KREA2_HF}/vae/{KREA2_VAE}", "vae", "VAELoader"))
node(23, "PixelArtPrompt", [400, -620], [420, 520], outputs=[out("prompt", "STRING"), out("pixels", "INT"),
     out("colors", "INT"), out("palette", "STRING")],
     widgets=[SUBJECT, STYLE, VIEW, PIXELS, DETAIL, COLORS, "auto", OUTLINE, "plain white", USE_TRIGGER, "", "", TRIGGER, FILL],
     title="Pixel Art Prompt", color=GREEN)
node(24, "PreviewAny", [860, -620], [420, 240], inputs=[inp("source", "*")], title="Final prompt")
node(6, "CLIPTextEncode", [400, 0], [420, 120], inputs=[inp("clip", "CLIP"), inp("text", "STRING", "text")],
     outputs=[out("CONDITIONING", "CONDITIONING")], widgets=[""], title="Encode prompt")
node(7, "ConditioningZeroOut", [400, 260], [240, 26], inputs=[inp("conditioning", "CONDITIONING")],
     outputs=[out("CONDITIONING", "CONDITIONING")])
node(8, "EmptyLatentImage", [400, 330], [300, 106], outputs=[out("LATENT", "LATENT")], widgets=[WIDTH, HEIGHT, 1],
     title="Size / count")
node(9, "KSampler", [860, 0], [320, 262],
     inputs=[inp("model", "MODEL"), inp("positive", "CONDITIONING"), inp("negative", "CONDITIONING"),
             inp("latent_image", "LATENT")],
     outputs=[out("LATENT", "LATENT")], widgets=[SEED, "randomize", STEPS, CFG, "euler", "simple", 1.0])
node(10, "VAEDecode", [860, 310], [200, 46], inputs=[inp("samples", "LATENT"), inp("vae", "VAE")], outputs=[out("IMAGE", "IMAGE")])
node(11, "SaveImage", [860, 400], [320, 360], inputs=[inp("images", "IMAGE")], widgets=["pixel/prompt_to_pixel_krea2_raw"],
     title="Save generated image")
node(12, "LoadBackgroundRemovalModel", [1220, 0], [320, 58], outputs=[out("bg_model", "BACKGROUND_REMOVAL")],
     widgets=[BIREFNET], title="BiRefNet", props=models(BIREFNET, BIREFNET_URL, "background_removal", "LoadBackgroundRemovalModel"))
node(13, "RemoveBackground", [1220, 100], [320, 46], inputs=[inp("bg_removal_model", "BACKGROUND_REMOVAL"), inp("image", "IMAGE")],
     outputs=[out("mask", "MASK")])
node(14, "InvertMask", [1220, 180], [320, 26], inputs=[inp("mask", "MASK")], outputs=[out("MASK", "MASK")])
node(15, "JoinImageWithAlpha", [1220, 250], [320, 46], inputs=[inp("image", "IMAGE"), inp("alpha", "MASK")],
     outputs=[out("IMAGE", "IMAGE")], title="Remove background")
node(25, "CleanCutout", [1220, 330], [320, 160], inputs=[inp("images", "IMAGE")],
     outputs=[out("image", "IMAGE"), out("removed", "MASK"), out("info", "STRING")], widgets=CUTOUT,
     title="Clean cutout (drop shadow, background pockets)", color=YELLOW)
node(16, "Pixelize", [1580, 0], [360, 440], inputs=[inp("images", "IMAGE"), inp("pixels", "INT", "pixels"),
                                                   inp("colors", "INT", "colors"), inp("palette", "STRING", "palette")],
     outputs=[out("pixel_art", "IMAGE"), out("preview", "IMAGE"), out("palette", "STRING"), out("info", "STRING")],
     widgets=PIXELIZE, title="Pixelize", color=YELLOW)
node(26, "Krea2PixelArtRefiner", [1580, 500], [360, 180], inputs=[inp("image", "IMAGE")], outputs=[out("IMAGE", "IMAGE")],
     widgets=[GRID, GRID, REFINER_COLORS, True, False], title="Krea 2 Pixel Art Refiner (width/height = LoRA grid)", color=YELLOW)
node(27, "GridCutout", [1580, 720], [360, 160], inputs=[inp("pixel_art", "IMAGE"), inp("cutout", "IMAGE")],
     outputs=[out("pixel_art", "IMAGE"), out("preview", "IMAGE"), out("info", "STRING")], widgets=[0.5, 1, False, 8],
     title="Grid Cutout")
node(28, "ComfySwitchNode", [1980, 0], [320, 80], inputs=[inp("on_false", "IMAGE"), inp("on_true", "IMAGE")],
     outputs=[out("output", "IMAGE")], widgets=[REFINER], title="Clean-up: author's refiner?", color=YELLOW)
node(29, "ComfySwitchNode", [1980, 120], [320, 80], inputs=[inp("on_false", "IMAGE"), inp("on_true", "IMAGE")],
     outputs=[out("output", "IMAGE")], widgets=[REFINER], title="Preview: author's refiner?")
node(17, "SaveImage", [2340, 0], [360, 300], inputs=[inp("images", "IMAGE")], widgets=["pixel/prompt_to_pixel_krea2"],
     title="Save pixel art (native size)", color=GREEN)
node(18, "SaveImage", [2340, 340], [360, 420], inputs=[inp("images", "IMAGE")], widgets=["pixel/prompt_to_pixel_krea2_x8"],
     title="Save preview (x8)")
node(19, "PixelArtCheck", [1980, 260], [320, 106], inputs=[inp("images", "IMAGE")],
     outputs=[out("report", "STRING"), out("passed", "BOOLEAN")], widgets=[MAX_COLORS, 0, True], title="Pixel Art Check")
node(20, "PreviewAny", [1980, 410], [320, 200], inputs=[inp("source", "*")], title="Check report")
node(22, "Note", [400, 480], [420, 520], widgets=[NOTE], title="READ ME", props={})

link(1, 0, 2, 0)
link(2, 0, 9, 0)
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
link(10, 0, 26, 0)           # the refiner works on the whole generated image, background included
link(26, 0, 27, 0)
link(25, 0, 27, 1)           # ... and the cut-out decides which cells are the subject
link(16, 0, 28, 0)
link(27, 0, 28, 1)
link(16, 1, 29, 0)
link(27, 1, 29, 1)
link(28, 0, 17, 0)
link(29, 0, 18, 0)
link(28, 0, 19, 0)
link(19, 0, 20, 0)
link(23, 0, 6, 1)            # template prompt -> text encoder
link(23, 0, 24, 0)
link(23, 1, 16, 1)           # the same pixels / colors / palette drive Pixelize
link(23, 2, 16, 2)
link(23, 3, 16, 3)

g.save(OUT, "pixel-art-prompt-to-pixel-krea2",
       groups=[("Prompt template", [380, -680, 920, 600], "#a1309b"),
               ("Krea 2 Turbo + k2-pixel LoRA", [-20, -60, 1220, 860], "#444"),
               ("Background removal (BiRefNet)", [1200, -60, 360, 380], "#444"),
               ("Clean-up: Pixelize or the author's refiner", [1560, -60, 760, 960], "#444")],
       scale=0.55, offset=(40, 80))
