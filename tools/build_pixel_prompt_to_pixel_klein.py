"""Generates workflows/pixel-art/prompt-to-pixel-klein.json.

Edit the constants below and re-run:  python tools/build_pixel_prompt_to_pixel_klein.py

The same graph as prompt-to-pixel with Flux 2 Klein 4B (Black Forest Labs, Apache-2.0, 4 steps) instead of Z-Image
Turbo: Klein draws the subject on a plain magenta background with a pixel-art LoRA, BiRefNet cuts it out, then the repo's
Pixelize puts it on an exact grid at native resolution. Rendered the way round 6 of the owner's blind tests did it:
flux-2-klein-4b (the 4-step model), Qwen3-4B encoder, flux2 VAE, 4 steps, euler, CFG 1 with a zeroed negative, 1024 px,
LoRA strength 1.0. Our own LoRA ("klein-moss", trigger "mosspx", trained on the base 4B) scored 9.5 there against 9.8
for Krea 2 + our LoRA; it draws about 2 s a picture on a 128-cell grid (8 px). It is NOT published yet, so no URL can be
listed for it: the slot names its file (sprite-studio/klein-moss.safetensors) and a ComfyUI that does not have it
cannot run the workflow until the file is put in models/loras/sprite-studio/. The public adirik LoRA is listed as the
alternative (it needs its own prompt form, see the note).
"""
from comfy_workflow import (BIREFNET, BIREFNET_URL, GREEN, YELLOW, WORKFLOWS, Graph, inp, out)

OUT = WORKFLOWS / "pixel-art" / "prompt-to-pixel-klein.json"

KLEIN_HF = "https://huggingface.co/Comfy-Org/flux2-klein-4B/resolve/main/split_files"
KLEIN_MODEL = "flux-2-klein-4b.safetensors"      # 4-step distilled, 7.8 GB
KLEIN_TE = "qwen_3_4b.safetensors"                # 8.0 GB
KLEIN_VAE = "flux2-vae.safetensors"               # 0.3 GB
# Our own LoRA (step 1800 of the cloud run, 369,644,416 bytes, trained on the base 4B with 58 MOSS sprites on magenta,
# trigger "mosspx"). Not published: no URL. The app puts it in models/loras/sprite-studio/.
LORA = "sprite-studio/klein-moss.safetensors"
TRIGGER = "mosspx"
# Public alternative (adirik, Apache-2.0; scored 6.6 in the same blind round). It ignores the caption form below: use the
# page's form, "PXART4. A pixel art of <subject>, plain magenta background" (style "none", trigger "PXART4").
ALT_LORA = "adirik-pixelart-lora-klein-4b.safetensors"
ALT_LORA_URL = "https://huggingface.co/adirik/pixel-art-lora-flux.2-klein-4B/resolve/main/pixelart_lora.safetensors"

SUBJECT = "a fantasy knight with a blue cape and a sword, standing"
STYLE, VIEW, DETAIL, OUTLINE = "LoRA caption", "front three-quarter", "auto (from pixels)", "dark outline"
BACKGROUND = "solid magenta (chroma key)"        # both LoRAs obey magenta; Krea and Z-Image prefer white
WIDTH, HEIGHT = 1024, 1024
LORA_STRENGTH = 1.0
STEPS, CFG = 4, 1.0      # distilled: 4 steps, CFG 1 (the negative is zeroed out)
SEED = 42

PIXELS = 64            # subject height in art pixels (Pixelize)
COLORS = 0             # 0 = automatic: the fewest colours within COLOR_ERROR
COLOR_ERROR = 20.0     # higher = fewer colours (12 faithful, 20 simplified, 30 very few)
MAX_COLORS = 32        # Pixel Art Check limit (automatic palettes stay under it)
SMOOTH = 0.0
# images, pixels, measure, method, smooth, colors, color_error, palette, shared_palette, background, bg_tolerance, cleanup, edge_cleanup,
# outline, padding, preview_scale
CUTOUT = [True, False, 0.02, 8.0, 0.2]   # drop_shadow, enclosed_background, min_pocket, bg_tolerance, shadow_band
PIXELIZE = [PIXELS, "subject height", "pixeloe", SMOOTH, COLORS, COLOR_ERROR, "auto", True, "auto remove", 12.0, True, True, "auto", 1, 8]

NOTE = f"""Prompt -> pixel art with Flux 2 Klein 4B

1. Pixel Art Prompt (green): type just the SUBJECT. Style "LoRA caption" writes the short caption our Klein LoRA was
   trained on: "<subject>. Shown full body, <view>, on a plain magenta background, in <trigger> style." The other
   styles still work (they add the long template). 'trigger' is the LoRA's word: "{TRIGGER}" for our own LoRA.
   pixels / colors / palette also drive Pixelize and the check.
2. Model: Flux 2 Klein 4B (Apache-2.0), the 4-step version, 4 steps, CFG 1, 1024 px, ~2 s a picture. "Pixel-art LoRA"
   (yellow) is the slot: our own LoRA "klein-moss" (step 1800, trigger "mosspx", strength 1.0) scored 9.5 in the owner's
   blind round. It is NOT published yet, so it has no download URL: put the file in models/loras/sprite-studio/.
   The public adirik LoRA (Apache-2.0, scored 6.6; listed, downloadable) ignores the caption form: pick it, set the
   prompt style to "none", trigger "PXART4" and expect an illustration unless you use its page's form
   ("PXART4. A pixel art of <subject>, plain magenta background") in the subject.
3. Pixelize (yellow): pixels / colors / palette come from the prompt node. Klein draws on a grid of 8 px (128 cells) and
   its subjects come out about 93 art px tall, so Pixelize reads the drawing's own grid and scales to 'pixels'.
4. Run. Saved to output/pixel/:
   prompt_to_pixel_klein_*.png      native size (1 art pixel = 1 image pixel, transparent) -> use in games
   prompt_to_pixel_klein_x8_*.png   zoomed preview
   prompt_to_pixel_klein_raw_*.png  the generated 1024px image
"Pixel Art Check" must say PASS."""

g = Graph()
node, link = g.node, g.link


def models(name, url, directory, type_):
    return {"Node name for S&R": type_, "models": [{"name": name, "url": url, "directory": directory}]}


node(1, "UNETLoader", [0, 0], [360, 82], outputs=[out("MODEL", "MODEL")], widgets=[KLEIN_MODEL, "default"],
     title="Flux 2 Klein 4B (4 steps)", props=models(KLEIN_MODEL, f"{KLEIN_HF}/diffusion_models/{KLEIN_MODEL}", "diffusion_models", "UNETLoader"))
node(2, "LoraLoaderModelOnly", [0, 120], [360, 82], inputs=[inp("model", "MODEL")], outputs=[out("MODEL", "MODEL")],
     widgets=[LORA, LORA_STRENGTH], title="Pixel-art LoRA", color=YELLOW,
     props={"Node name for S&R": "LoraLoaderModelOnly", "models": [{"name": ALT_LORA, "url": ALT_LORA_URL, "directory": "loras"}]})
node(4, "CLIPLoader", [0, 340], [360, 106], outputs=[out("CLIP", "CLIP")], widgets=[KLEIN_TE, "flux2", "default"], title="Qwen3-4B text encoder",
     props=models(KLEIN_TE, f"{KLEIN_HF}/text_encoders/{KLEIN_TE}", "text_encoders", "CLIPLoader"))
node(5, "VAELoader", [0, 490], [360, 58], outputs=[out("VAE", "VAE")], widgets=[KLEIN_VAE], title="Flux 2 VAE",
     props=models(KLEIN_VAE, f"{KLEIN_HF}/vae/{KLEIN_VAE}", "vae", "VAELoader"))
node(23, "PixelArtPrompt", [400, -560], [420, 460], outputs=[out("prompt", "STRING"), out("pixels", "INT"),
     out("colors", "INT"), out("palette", "STRING")],
     widgets=[SUBJECT, STYLE, VIEW, PIXELS, DETAIL, COLORS, "auto", OUTLINE, BACKGROUND, True, "", "", TRIGGER, 60],
     title="Pixel Art Prompt", color=GREEN)
node(24, "PreviewAny", [860, -560], [420, 240], inputs=[inp("source", "*")], title="Final prompt")
node(6, "CLIPTextEncode", [400, 0], [420, 120], inputs=[inp("clip", "CLIP"), inp("text", "STRING", "text")],
     outputs=[out("CONDITIONING", "CONDITIONING")], widgets=[""], title="Encode prompt")
node(7, "ConditioningZeroOut", [400, 260], [240, 26], inputs=[inp("conditioning", "CONDITIONING")],
     outputs=[out("CONDITIONING", "CONDITIONING")])
node(8, "EmptyFlux2LatentImage", [400, 330], [300, 106], outputs=[out("LATENT", "LATENT")], widgets=[WIDTH, HEIGHT, 1])
node(30, "CFGGuider", [860, 0], [300, 106],
     inputs=[inp("model", "MODEL"), inp("positive", "CONDITIONING"), inp("negative", "CONDITIONING")],
     outputs=[out("GUIDER", "GUIDER")], widgets=[CFG])
node(31, "KSamplerSelect", [860, 150], [300, 58], outputs=[out("SAMPLER", "SAMPLER")], widgets=["euler"])
node(32, "Flux2Scheduler", [860, 250], [300, 106], outputs=[out("SIGMAS", "SIGMAS")], widgets=[STEPS, WIDTH, HEIGHT])
node(33, "RandomNoise", [860, 400], [300, 82], outputs=[out("NOISE", "NOISE")], widgets=[SEED, "randomize"],
     title="Noise (seed here)")
node(9, "SamplerCustomAdvanced", [860, 520], [300, 120],
     inputs=[inp("noise", "NOISE"), inp("guider", "GUIDER"), inp("sampler", "SAMPLER"), inp("sigmas", "SIGMAS"),
             inp("latent_image", "LATENT")], outputs=[out("output", "LATENT"), out("denoised_output", "LATENT")],
     title="Sampler")
node(10, "VAEDecode", [860, 680], [200, 46], inputs=[inp("samples", "LATENT"), inp("vae", "VAE")], outputs=[out("IMAGE", "IMAGE")])
node(11, "SaveImage", [860, 760], [320, 360], inputs=[inp("images", "IMAGE")], widgets=["pixel/prompt_to_pixel_klein_raw"],
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
node(17, "SaveImage", [1980, 0], [360, 300], inputs=[inp("images", "IMAGE")], widgets=["pixel/prompt_to_pixel_klein"],
     title="Save pixel art (native size)", color=GREEN)
node(18, "SaveImage", [1980, 340], [360, 420], inputs=[inp("images", "IMAGE")], widgets=["pixel/prompt_to_pixel_klein_x8"],
     title="Save preview (x8)")
node(19, "PixelArtCheck", [1580, 490], [360, 106], inputs=[inp("images", "IMAGE"), inp("expected_height", "INT", "expected_height")],
     outputs=[out("report", "STRING"), out("passed", "BOOLEAN")], widgets=[MAX_COLORS, PIXELS, True], title="Pixel Art Check")
node(20, "PreviewAny", [1580, 640], [360, 200], inputs=[inp("source", "*")], title="Check report")
node(21, "PreviewAny", [1980, 800], [360, 160], inputs=[inp("source", "*")], title="Palette")
node(22, "Note", [400, 480], [420, 460], widgets=[NOTE], title="READ ME", props={})

link(1, 0, 2, 0)
link(2, 0, 30, 0)
link(4, 0, 6, 0)
link(6, 0, 7, 0)
link(6, 0, 30, 1)
link(7, 0, 30, 2)
link(33, 0, 9, 0)
link(30, 0, 9, 1)
link(31, 0, 9, 2)
link(32, 0, 9, 3)
link(8, 0, 9, 4)
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

g.save(OUT, "pixel-art-prompt-to-pixel-klein",
       groups=[("Prompt template", [380, -620, 920, 540], "#a1309b"),
               ("Flux 2 Klein 4B + pixel-art LoRA", [-20, -60, 1220, 860], "#444"),
               ("Background removal (BiRefNet)", [1200, -60, 360, 380], "#444")],
       scale=0.6, offset=(40, 80))
