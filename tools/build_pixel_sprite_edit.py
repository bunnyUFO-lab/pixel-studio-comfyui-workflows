"""Generates workflows/pixel-art/sprite-edit.json.

Edit the constants below and re-run:  python tools/build_pixel_sprite_edit.py

Pixel sprite (or sprite-sheet frames) + edit prompt -> edited sprite with the SAME grid, size and position:
PrepareSpriteForVideo upscales it by an integer factor onto a flat background, Qwen-Image-2.1 edit applies the
instruction, RepixelateFrames brings it back onto exactly the same grid (palette: the sprite's, or sprite + a few
new colours the edit introduces), RestoreSpriteCanvas puts it back on the original canvas so it can replace the
original (e.g. a frame in a sheet). A batch of frames is edited one by one with the same prompt.
"""
from comfy_workflow import (BLUE, GREEN, QWEN_HF as HF, QWEN_MODEL, QWEN_TE, QWEN_VAE, YELLOW, WORKFLOWS, Graph, inp,
                            out)

OUT = WORKFLOWS / "pixel-art" / "sprite-edit.json"
OUT_REFS = WORKFLOWS / "pixel-art" / "sprite-edit-refs.json"   # used by Sprite Studio (frame edits with references)

PROMPT = "Give the character in <image1> a red scarf. Keep everything else exactly the same, same pose, same style."
TARGET, PADDING = 1024, 0.25
STEPS, SEED = 25, 7
# height_match, bg_tolerance, method, colors (max new colours), palette, stability, deflicker, cleanup, outline
REPIX = [False, 18.0, "dominant", 6, "sprite+new", 0.0, False, True, "auto"]

NOTE = f"""Sprite edit (keeps the exact pixel grid)

1. Load a pixel sprite (blue), native size or upscaled; a batch of frames works too (each edited the same way).
2. Edit prompt (green): an instruction about <image1> ("give him a red scarf", "make the cape torn",
   "remove the hat"). Say "keep everything else the same".
3. Qwen-Image-2.1 edits a x{TARGET // 64}-ish upscale, then the result is re-pixelated on the sprite's own grid:
   palette 'sprite+new' = the sprite's colours plus up to 'colors' new ones the edit adds; 'sprite' = locked.
4. Restore: 'original canvas' = same size and position as the input (drop-in replacement, e.g. sheet frames).
Saved to output/pixel/sprite_edit_*.png (+ _x8 preview). ~30-40 s per frame."""


def models(name, url, directory, type_):
    return {"Node name for S&R": type_, "models": [{"name": name, "url": url, "directory": directory}]}


def build(refs=False):
    """refs=True: the app's variant with up to 3 reference images (other frames / directions) as <image2>..<image4>."""
    g = Graph()
    node, link = g.node, g.link

    node(1, "LoadImage", [0, 0], [380, 400], outputs=[out("IMAGE", "IMAGE"), out("MASK", "MASK")],
         widgets=["example.png", "image"], title="Pixel sprite", color=BLUE)
    node(2, "JoinImageWithAlpha", [0, 440], [380, 46], inputs=[inp("image", "IMAGE"), inp("alpha", "MASK")],
         outputs=[out("IMAGE", "IMAGE")], title="Keep sprite transparency")
    node(3, "PrepareSpriteForVideo", [420, 0], [340, 130], inputs=[inp("sprite", "IMAGE")],
         outputs=[out("image", "IMAGE"), out("pixel_spec", "PIXEL_SPEC"), out("width", "INT"), out("height", "INT"),
                  out("info", "STRING")],
         widgets=[TARGET, PADDING, "auto"], title="Prepare sprite (integer upscale)")
    node(20, "UNETLoader", [420, 180], [340, 82], outputs=[out("MODEL", "MODEL")], widgets=[QWEN_MODEL, "default"],
         title="Qwen-Image-2.1", props=models(QWEN_MODEL, f"{HF}/diffusion_models/{QWEN_MODEL}", "diffusion_models", "UNETLoader"))
    node(21, "QwenImage21Cache", [420, 300], [340, 82], inputs=[inp("model", "MODEL")], outputs=[out("MODEL", "MODEL")],
         widgets=["auto", "default"])
    node(22, "CLIPLoader", [420, 420], [340, 106], outputs=[out("CLIP", "CLIP")], widgets=[QWEN_TE, "qwen_image", "default"],
         title="Qwen3-VL text encoder", props=models(QWEN_TE, f"{HF}/text_encoders/{QWEN_TE}", "text_encoders", "CLIPLoader"))
    node(23, "VAELoader", [420, 570], [340, 58], outputs=[out("VAE", "VAE")], widgets=[QWEN_VAE],
         props=models(QWEN_VAE, f"{HF}/vae/{QWEN_VAE}", "vae", "VAELoader"))
    node(24, "TextEncodeQwenImage21", [800, 0], [440, 260],
         inputs=[inp("clip", "CLIP"), inp("vae", "VAE"), inp("images.image_1", "IMAGE")]
         + ([inp(f"images.image_{k}", "IMAGE") for k in (2, 3, 4)] if refs else []),
         outputs=[out("positive", "CONDITIONING"), out("negative", "CONDITIONING"), out("latent", "LATENT")],
         widgets=[PROMPT, "", 1024], title="Edit prompt", color=GREEN)
    if refs:
        for k, nid in enumerate((40, 41, 42), 1):
            node(nid, "LoadImage", [0, 940 + 360 * (k - 1)], [380, 320], outputs=[out("IMAGE", "IMAGE"), out("MASK", "MASK")],
                 widgets=["example.png", "image"], title=f"Reference {k}", color=BLUE)
    node(25, "KSampler", [800, 300], [320, 262],
         inputs=[inp("model", "MODEL"), inp("positive", "CONDITIONING"), inp("negative", "CONDITIONING"),
                 inp("latent_image", "LATENT")],
         outputs=[out("LATENT", "LATENT")], widgets=[SEED, "randomize", STEPS, 1.0, "euler", "simple", 1.0], title="Qwen edit")
    node(26, "VAEDecode", [800, 600], [210, 46], inputs=[inp("samples", "LATENT"), inp("vae", "VAE")],
         outputs=[out("IMAGE", "IMAGE")])
    node(27, "RepixelateFrames", [1280, 0], [340, 250], inputs=[inp("frames", "IMAGE"), inp("pixel_spec", "PIXEL_SPEC")],
         outputs=[out("pixel_frames", "IMAGE"), out("info", "STRING"), out("palette", "STRING")], widgets=REPIX,
         title="Repixelate (same grid)", color=YELLOW)
    node(28, "RestoreSpriteCanvas", [1280, 300], [340, 82], inputs=[inp("frames", "IMAGE"), inp("pixel_spec", "PIXEL_SPEC")],
         outputs=[out("frames", "IMAGE")], widgets=["original canvas"], title="Restore canvas", color=YELLOW)
    node(29, "SaveImage", [1660, 0], [360, 320], inputs=[inp("images", "IMAGE")], widgets=["pixel/sprite_edit"],
         title="Save edited sprite (native size)", color=GREEN)
    node(30, "ImageScaleBy", [1660, 360], [300, 82], inputs=[inp("image", "IMAGE")], outputs=[out("IMAGE", "IMAGE")],
         widgets=["nearest-exact", 8.0], title="Zoom x8")
    node(31, "SaveImage", [1660, 480], [360, 400], inputs=[inp("images", "IMAGE")], widgets=["pixel/sprite_edit_x8"],
         title="Save preview (x8)")
    node(32, "PixelArtCheck", [1280, 430], [340, 106], inputs=[inp("images", "IMAGE")],
         outputs=[out("report", "STRING"), out("passed", "BOOLEAN")], widgets=[64, 0, True], title="Pixel Art Check")
    node(33, "PreviewAny", [1280, 580], [340, 200], inputs=[inp("source", "*")], title="Check report")
    node(34, "PreviewAny", [1280, 820], [340, 140], inputs=[inp("source", "*")], title="Palette")
    node(35, "Note", [0, 520], [380, 380], widgets=[NOTE], title="READ ME", props={})

    link(1, 0, 2, 0)
    link(1, 1, 2, 1)
    link(2, 0, 3, 0)
    link(20, 0, 21, 0)
    link(22, 0, 24, 0)
    link(23, 0, 24, 1)
    link(3, 0, 24, 2)
    if refs:
        for k, nid in enumerate((40, 41, 42)):
            link(nid, 0, 24, 3 + k)
    link(21, 0, 25, 0)
    link(24, 0, 25, 1)
    link(24, 1, 25, 2)
    link(24, 2, 25, 3)
    link(25, 0, 26, 0)
    link(23, 0, 26, 1)
    link(26, 0, 27, 0)
    link(3, 1, 27, 1)
    link(27, 0, 28, 0)
    link(3, 1, 28, 1)
    link(28, 0, 29, 0)
    link(28, 0, 30, 0)
    link(30, 0, 31, 0)
    link(28, 0, 32, 0)
    link(32, 0, 33, 0)
    link(27, 2, 34, 0)

    g.save(OUT_REFS if refs else OUT, "pixel-art-sprite-edit-refs" if refs else "pixel-art-sprite-edit", groups=[("Qwen-Image-2.1 edit", [400, -60, 860, 720], "#3f789e"),
                                                  ("Back to the same grid", [1260, -60, 780, 1040], "#444")],
           scale=0.6, offset=(40, 80),
           app=([(1, "image"), (24, "prompt"), (25, "seed"), (27, "palette"), (27, "colors"), (28, "mode")], [29, 31, 33]))


build()
build(refs=True)
