"""Generates workflows/pixel-art/image-to-directions.json.

Edit the constants below and re-run:  python tools/build_pixel_image_to_directions.py

Character image -> 5 game-direction views (front, front-right, right, back-right, back) drawn by Qwen-Image-2.1 edit
(one editable prompt per view, core StartLoop/EndLoop), background removed, pixelized TOGETHER (one grid size from
the front view, one shared palette), then mirrored into 8 directions and saved one PNG per direction + a sheet.
"""
from comfy_workflow import (BIREFNET, BIREFNET_URL, BLUE, GREEN, QWEN_HF as HF, QWEN_MODEL, QWEN_TE, QWEN_VAE,
                            WORKFLOWS, YELLOW, Graph, inp, out)

OUT = WORKFLOWS / "pixel-art" / "image-to-directions.json"

KEEP = ("Full body, standing pose, the exact same character, outfit, colours, proportions and art style, "
        "plain flat white background, no shadow.")
VIEWS = [
    ("front", "Show the character from <image1> in a front view, facing the camera directly."),
    ("front-right", "Show the character from <image1> in a three-quarter front view, body turned 45 degrees so it "
                    "faces the right side of the image."),
    ("right", "Show the character from <image1> in a side profile view, facing the right edge of the image."),
    ("back-right", "Show the character from <image1> in a three-quarter back view, seen from behind, turned 45 "
                   "degrees so it faces away from the camera and toward the right side of the image."),
    ("back", "Show the character from <image1> from directly behind (back view), facing away from the camera."),
]
PROMPTS = "\n".join(f"{p} {KEEP}" for _, p in VIEWS)
NAMES = ", ".join(n for n, _ in VIEWS)
STEPS, SEED = 25, 1234

PIXELS = 64            # body height in art pixels, every view (thin props like staffs not counted)
COLORS = 0             # 0 = automatic: the fewest colours within COLOR_ERROR
COLOR_ERROR = 20.0     # higher = fewer colours (12 faithful, 20 simplified, 30 very few)
MAX_COLORS = 32        # Pixel Art Check limit (automatic palettes stay under it)
# images, pixels, measure, method, smooth, colors, color_error, palette, shared_palette, background, bg_tolerance, cleanup, edge_cleanup,
# outline, padding, preview_scale
CUTOUT = [True, False, 0.02, 8.0, 0.2]   # drop_shadow, enclosed_background, min_pocket, bg_tolerance, shadow_band
PIXELIZE = [PIXELS, "body height (per frame)", "pixeloe", 0.6, COLORS, COLOR_ERROR, "auto", True, "auto remove", 12.0, True, True, "auto", 1, 8]

NOTE = f"""Character image -> 8-direction pixel sprites

1. Load a character image (blue), ideally full body on a plain background.
2. "View prompts" (green): ONE LINE PER VIEW, in the order of "names" on the Save node:
   {NAMES}
   Edit a line to fix a view. Add/remove lines and change "views" (= number of lines) + names to match.
3. Qwen-Image-2.1 edit draws each view ({STEPS} steps, ~30-40 s per view on an RTX 4090).
4. All views are pixelized together: each view's body becomes {PIXELS} art px tall (the model draws views at
   slightly different sizes; thin props like staffs aren't counted), feet on the bottom line, one shared
   palette (automatic fewest colours, or a number / preset), so they line up as game sprites.
5. Save directions: left-side views are the mirrored right-side ones (mirror = true) -> 8 directions.
"Turnaround preview" plays all directions in a loop (3 fps).
Saved to output/pixel/: directions_*_<direction>.png (native), directions_*.png (sheet), .json, _preview.png;
the raw generated views are in pixel/directions_raw_*.png.
Licence: Qwen-Image-2.1 model is non-commercial, but you own the images you generate (QwenDevs)."""


def models(name, url, directory, type_):
    return {"Node name for S&R": type_, "models": [{"name": name, "url": url, "directory": directory}]}


g = Graph()
node, link = g.node, g.link

node(1, "LoadImage", [0, 0], [400, 440], outputs=[out("IMAGE", "IMAGE"), out("MASK", "MASK")],
     widgets=["example.png", "image"], title="Character image", color=BLUE)
node(2, "UNETLoader", [440, 0], [340, 82], outputs=[out("MODEL", "MODEL")], widgets=[QWEN_MODEL, "default"],
     title="Qwen-Image-2.1", props=models(QWEN_MODEL, f"{HF}/diffusion_models/{QWEN_MODEL}", "diffusion_models", "UNETLoader"))
node(3, "QwenImage21Cache", [440, 120], [340, 82], inputs=[inp("model", "MODEL")], outputs=[out("MODEL", "MODEL")],
     widgets=["auto", "default"])
node(4, "CLIPLoader", [440, 240], [340, 106], outputs=[out("CLIP", "CLIP")], widgets=[QWEN_TE, "qwen_image", "default"],
     title="Qwen3-VL text encoder", props=models(QWEN_TE, f"{HF}/text_encoders/{QWEN_TE}", "text_encoders", "CLIPLoader"))
node(5, "VAELoader", [440, 390], [340, 58], outputs=[out("VAE", "VAE")], widgets=[QWEN_VAE],
     props=models(QWEN_VAE, f"{HF}/vae/{QWEN_VAE}", "vae", "VAELoader"))

# One iteration per view
node(10, "StartLoop", [820, 0], [300, 160],
     inputs=[inp("parent_iteration", "INT"), inp("initial_iteration_value", "*"),
             inp("mode.num_iterations", "INT", "mode.num_iterations")],
     outputs=[out("iteration_index", "INT"), out("is_first", "BOOLEAN"), out("is_last", "BOOLEAN"),
              out("list_item", "*"), out("current_iteration_value", "*")],
     widgets=["simple", len(VIEWS), False], title="Views (num_iterations = number of prompt lines)", color=YELLOW)
node(11, "SegmentPrompt", [820, 200], [460, 300], inputs=[inp("index", "INT")],
     outputs=[out("prompt", "STRING"), out("segment_seed", "INT")], widgets=[PROMPTS, SEED, "fixed"],
     title="View prompts (one line per view)", color=GREEN)
node(12, "TextEncodeQwenImage21", [1320, 0], [400, 200],
     inputs=[inp("clip", "CLIP"), inp("vae", "VAE"), inp("images.image_1", "IMAGE"), inp("prompt", "STRING", "prompt")],
     outputs=[out("positive", "CONDITIONING"), out("negative", "CONDITIONING"), out("latent", "LATENT")],
     widgets=["", "", 1024], title="Encode view")
node(13, "KSampler", [1320, 240], [320, 262],
     inputs=[inp("model", "MODEL"), inp("positive", "CONDITIONING"), inp("negative", "CONDITIONING"),
             inp("latent_image", "LATENT")],
     outputs=[out("LATENT", "LATENT")], widgets=[SEED, "fixed", STEPS, 1.0, "euler", "simple", 1.0], title="Qwen edit")
node(14, "VAEDecode", [1320, 540], [210, 46], inputs=[inp("samples", "LATENT"), inp("vae", "VAE")],
     outputs=[out("IMAGE", "IMAGE")])
node(15, "EndLoop", [1760, 0], [300, 140],
     inputs=[inp("output_value", "*"), inp("next_iteration_value", "*"), inp("terminations.termination0", "*")],
     outputs=[out("outputs", "IMAGE")], widgets=[True], title="End loop (collect views)")

# Cut out, batch, pixelize together
node(16, "SaveImage", [1760, 200], [320, 300], inputs=[inp("images", "IMAGE")], widgets=["pixel/directions_raw"],
     title="Save generated views")
node(17, "LoadBackgroundRemovalModel", [2100, 0], [320, 58], outputs=[out("bg_model", "BACKGROUND_REMOVAL")],
     widgets=[BIREFNET], title="BiRefNet", props=models(BIREFNET, BIREFNET_URL, "background_removal", "LoadBackgroundRemovalModel"))
node(18, "RemoveBackground", [2100, 100], [320, 46], inputs=[inp("bg_removal_model", "BACKGROUND_REMOVAL"), inp("image", "IMAGE")],
     outputs=[out("mask", "MASK")])
node(19, "InvertMask", [2100, 180], [320, 26], inputs=[inp("mask", "MASK")], outputs=[out("MASK", "MASK")])
node(20, "JoinImageWithAlpha", [2100, 250], [320, 46], inputs=[inp("image", "IMAGE"), inp("alpha", "MASK")],
     outputs=[out("IMAGE", "IMAGE")], title="Remove background")
node(28, "CleanCutout", [2100, 430], [320, 160], inputs=[inp("images", "IMAGE")],
     outputs=[out("image", "IMAGE"), out("removed", "MASK"), out("info", "STRING")], widgets=CUTOUT,
     title="Clean cutout (drop shadow, background pockets)", color=YELLOW)
node(21, "RebatchImages", [2100, 340], [320, 58], inputs=[inp("images", "IMAGE")], outputs=[out("IMAGE", "IMAGE")],
     widgets=[64], title="Views -> one batch")
node(22, "Pixelize", [2460, 0], [360, 440], inputs=[inp("images", "IMAGE")],
     outputs=[out("pixel_art", "IMAGE"), out("preview", "IMAGE"), out("palette", "STRING"), out("info", "STRING")],
     widgets=PIXELIZE, title="Pixelize (shared grid + palette)", color=YELLOW)
node(23, "SaveDirections", [2860, 0], [360, 200], inputs=[inp("images", "IMAGE")],
     outputs=[out("sheet", "IMAGE"), out("directions", "IMAGE"), out("info", "STRING")],
     widgets=["pixel/directions", NAMES, True, 2, 6], title="Save directions (mirror -> 8)", color=GREEN)
node(24, "PixelArtCheck", [2460, 490], [360, 106], inputs=[inp("images", "IMAGE")],
     outputs=[out("report", "STRING"), out("passed", "BOOLEAN")], widgets=[MAX_COLORS, PIXELS, True], title="Pixel Art Check")
node(25, "PreviewAny", [2460, 640], [360, 200], inputs=[inp("source", "*")], title="Check report")
node(26, "PreviewAny", [2860, 250], [360, 160], inputs=[inp("source", "*")], title="Directions info")
node(29, "SpriteAnimationPreview", [3260, 0], [520, 680], inputs=[inp("frames", "IMAGE"), inp("fps", "INT", "fps")],
     widgets=[3, 6, "checker", "loop"], title="Turnaround preview (all directions)", color=GREEN)
node(27, "Note", [0, 490], [460, 560], widgets=[NOTE], title="READ ME", props={})

link(2, 0, 3, 0)
link(1, 0, 10, 1)            # carried value: the input image (unchanged)
link(10, 0, 11, 0)
link(4, 0, 12, 0)
link(5, 0, 12, 1)
link(1, 0, 12, 2)
link(11, 0, 12, 3)
link(3, 0, 13, 0)
link(12, 0, 13, 1)
link(12, 1, 13, 2)
link(12, 2, 13, 3)
link(13, 0, 14, 0)
link(5, 0, 14, 1)
link(14, 0, 15, 0)           # output_value: this view, accumulated
link(10, 4, 15, 1)
link(15, 0, 16, 0)
link(17, 0, 18, 0)
link(15, 0, 18, 1)
link(18, 0, 19, 0)
link(15, 0, 20, 0)
link(19, 0, 20, 1)
link(20, 0, 28, 0)
link(28, 0, 21, 0)
link(21, 0, 22, 0)
link(22, 0, 23, 0)
link(22, 0, 24, 0)
link(24, 0, 25, 0)
link(23, 2, 26, 0)

link(23, 1, 29, 0)           # 8 directions -> turnaround playback

g.save(OUT, "pixel-art-image-to-directions", groups=[
    ("Qwen-Image-2.1", [420, -60, 380, 520], "#3f789e"),
    ("One edit per view (loop)", [800, -60, 1280, 620], "#a1309b"),
    ("Cut out + pixelize together", [2080, -60, 1160, 920], "#444"),
], scale=0.5, offset=(40, 80))
