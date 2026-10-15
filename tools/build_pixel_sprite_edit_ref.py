"""Generates workflows/pixel-art/sprite-edit-ref.json.

Edit the constants below and re-run:  python tools/build_pixel_sprite_edit_ref.py

Context-aware sprite edit: Qwen-Image-2.1 edits a full-resolution image of the sprite (e.g. the original drawing of one
direction) while seeing up to 3 reference images of the same character (other directions, the concept art), so a
fix like "hold the shovel like in the front view" is copied from the references instead of guessed. The result is
cut out (BiRefNet) and pixelized with the sprite set's own palette and body height, so it fits back into the set.
"""
from comfy_workflow import (BIREFNET, BIREFNET_URL, BLUE, GREEN, QWEN_HF as HF, QWEN_MODEL, QWEN_TE, QWEN_VAE, YELLOW,
                            WORKFLOWS, Graph, inp, out)

OUT = WORKFLOWS / "pixel-art" / "sprite-edit-ref.json"

PROMPT = ("<image1> is the side view (facing right) of the same character shown in <image2> and <image3>. Edit <image1> "
          "so he holds the long shovel exactly like in <image2>: vertical at his side, gripped near the top of the handle, "
          "the blade resting on the ground. Keep the side view, pose, proportions, colours and style of <image1>; plain "
          "white background.")
PIXELS, STEPS, SEED = 64, 25, 7
CUTOUT = [True, False, 0.02, 8.0, 0.2]
# images, pixels, measure, method, smooth, colors, color_error, palette, shared_palette, background, bg_tolerance, cleanup,
# edge_cleanup, outline, padding, preview_scale
PIXELIZE = [PIXELS, "body height (per frame)", "pixeloe", 0.0, 0, 20.0, "auto", True, "auto remove", 12.0, True, True, "auto", 1, 8]

NOTE = """Sprite edit with references (context-aware)

1. Target (blue): the image to change, best at full resolution (e.g. the original drawing of one direction).
2. References 1-3 (blue): other images of the same character (other directions, concept art). Unused ones can be
   bypassed (Ctrl+B).
3. Edit prompt (green): refer to the images as <image1> (target), <image2>, <image3>, <image4> (references):
   "make him hold the shovel exactly like in <image2>", "copy the hat from <image3>".
4. Pixelize (yellow): set pixels (body height) and palette to the sprite set's so the result fits back in.
Saved: output/pixel/sprite_edit_ref_*.png (native), _x8 preview, and the edited full-res image."""


def models(name, url, directory, type_):
    return {"Node name for S&R": type_, "models": [{"name": name, "url": url, "directory": directory}]}


g = Graph()
node, link = g.node, g.link

node(1, "LoadImage", [0, 0], [300, 320], outputs=[out("IMAGE", "IMAGE"), out("MASK", "MASK")], widgets=["example.png", "image"],
     title="Target image", color=BLUE)
node(2, "LoadImage", [0, 360], [300, 320], outputs=[out("IMAGE", "IMAGE"), out("MASK", "MASK")], widgets=["example.png", "image"],
     title="Reference 1", color=BLUE)
node(3, "LoadImage", [0, 720], [300, 320], outputs=[out("IMAGE", "IMAGE"), out("MASK", "MASK")], widgets=["example.png", "image"],
     title="Reference 2", color=BLUE)
node(4, "LoadImage", [0, 1080], [300, 320], outputs=[out("IMAGE", "IMAGE"), out("MASK", "MASK")], widgets=["example.png", "image"],
     title="Reference 3", color=BLUE)
node(20, "UNETLoader", [340, 0], [340, 82], outputs=[out("MODEL", "MODEL")], widgets=[QWEN_MODEL, "default"],
     title="Qwen-Image-2.1", props=models(QWEN_MODEL, f"{HF}/diffusion_models/{QWEN_MODEL}", "diffusion_models", "UNETLoader"))
node(21, "QwenImage21Cache", [340, 120], [340, 82], inputs=[inp("model", "MODEL")], outputs=[out("MODEL", "MODEL")],
     widgets=["auto", "default"])
node(22, "CLIPLoader", [340, 240], [340, 106], outputs=[out("CLIP", "CLIP")], widgets=[QWEN_TE, "qwen_image", "default"],
     title="Qwen3-VL text encoder", props=models(QWEN_TE, f"{HF}/text_encoders/{QWEN_TE}", "text_encoders", "CLIPLoader"))
node(23, "VAELoader", [340, 390], [340, 58], outputs=[out("VAE", "VAE")], widgets=[QWEN_VAE],
     props=models(QWEN_VAE, f"{HF}/vae/{QWEN_VAE}", "vae", "VAELoader"))
node(24, "TextEncodeQwenImage21", [720, 0], [440, 300],
     inputs=[inp("clip", "CLIP"), inp("vae", "VAE"), inp("images.image_1", "IMAGE"), inp("images.image_2", "IMAGE"),
             inp("images.image_3", "IMAGE"), inp("images.image_4", "IMAGE")],
     outputs=[out("positive", "CONDITIONING"), out("negative", "CONDITIONING"), out("latent", "LATENT")],
     widgets=[PROMPT, "", 1024], title="Edit prompt (with references)", color=GREEN)
node(25, "KSampler", [720, 340], [320, 262],
     inputs=[inp("model", "MODEL"), inp("positive", "CONDITIONING"), inp("negative", "CONDITIONING"), inp("latent_image", "LATENT")],
     outputs=[out("LATENT", "LATENT")], widgets=[SEED, "randomize", STEPS, 1.0, "euler", "simple", 1.0], title="Qwen edit")
node(26, "VAEDecode", [720, 640], [210, 46], inputs=[inp("samples", "LATENT"), inp("vae", "VAE")], outputs=[out("IMAGE", "IMAGE")])
node(27, "SaveImage", [720, 720], [320, 360], inputs=[inp("images", "IMAGE")], widgets=["pixel/sprite_edit_ref_full"],
     title="Save edited image (full res)")
node(30, "LoadBackgroundRemovalModel", [1200, 0], [320, 58], outputs=[out("bg_model", "BACKGROUND_REMOVAL")], widgets=[BIREFNET],
     title="BiRefNet", props=models(BIREFNET, BIREFNET_URL, "background_removal", "LoadBackgroundRemovalModel"))
node(31, "RemoveBackground", [1200, 100], [320, 46], inputs=[inp("bg_removal_model", "BACKGROUND_REMOVAL"), inp("image", "IMAGE")],
     outputs=[out("mask", "MASK")])
node(32, "InvertMask", [1200, 180], [320, 26], inputs=[inp("mask", "MASK")], outputs=[out("MASK", "MASK")])
node(33, "JoinImageWithAlpha", [1200, 250], [320, 46], inputs=[inp("image", "IMAGE"), inp("alpha", "MASK")],
     outputs=[out("IMAGE", "IMAGE")], title="Remove background")
node(34, "CleanCutout", [1200, 330], [320, 160], inputs=[inp("images", "IMAGE")],
     outputs=[out("image", "IMAGE"), out("removed", "MASK"), out("info", "STRING")], widgets=CUTOUT, title="Clean cutout")
node(35, "Pixelize", [1560, 0], [360, 440], inputs=[inp("images", "IMAGE")],
     outputs=[out("pixel_art", "IMAGE"), out("preview", "IMAGE"), out("palette", "STRING"), out("info", "STRING")],
     widgets=PIXELIZE, title="Pixelize (match the set)", color=YELLOW)
node(36, "SaveImage", [1960, 0], [320, 300], inputs=[inp("images", "IMAGE")], widgets=["pixel/sprite_edit_ref"],
     title="Save pixel sprite", color=GREEN)
node(37, "SaveImage", [1960, 340], [320, 400], inputs=[inp("images", "IMAGE")], widgets=["pixel/sprite_edit_ref_x8"],
     title="Save preview (x8)")
node(38, "PixelArtCheck", [1560, 490], [360, 106], inputs=[inp("images", "IMAGE")],
     outputs=[out("report", "STRING"), out("passed", "BOOLEAN")], widgets=[64, 0, True], title="Pixel Art Check")
node(39, "PreviewAny", [1560, 640], [360, 200], inputs=[inp("source", "*")], title="Check report")
node(40, "Note", [340, 500], [340, 420], widgets=[NOTE], title="READ ME", props={})

link(20, 0, 21, 0)
link(22, 0, 24, 0)
link(23, 0, 24, 1)
link(1, 0, 24, 2)
link(2, 0, 24, 3)
link(3, 0, 24, 4)
link(4, 0, 24, 5)
link(21, 0, 25, 0)
link(24, 0, 25, 1)
link(24, 1, 25, 2)
link(24, 2, 25, 3)
link(25, 0, 26, 0)
link(23, 0, 26, 1)
link(26, 0, 27, 0)
link(30, 0, 31, 0)
link(26, 0, 31, 1)
link(31, 0, 32, 0)
link(26, 0, 33, 0)
link(32, 0, 33, 1)
link(33, 0, 34, 0)
link(34, 0, 35, 0)
link(35, 0, 36, 0)
link(35, 1, 37, 0)
link(35, 0, 38, 0)
link(38, 0, 39, 0)

g.save(OUT, "pixel-art-sprite-edit-ref", groups=[("Qwen-Image-2.1 edit with references", [320, -60, 860, 1000], "#3f789e"),
                                                  ("Back to the set's pixels", [1180, -60, 1120, 920], "#444")],
       scale=0.5, offset=(40, 80),
       app=([(1, "image"), (2, "image"), (3, "image"), (24, "prompt"), (25, "seed"), (35, "pixels"), (35, "palette")], [36, 37, 39]))
