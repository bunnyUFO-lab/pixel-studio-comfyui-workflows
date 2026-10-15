"""Generates workflows/pixel-art/prompt-to-setpiece.json  (W2 of the pixel-scenery design).

Edit the constants below and re-run:  python tools/build_pixel_prompt_to_setpiece.py

Prompt -> one large background or foreground object (a tree, a cliff, a ruined tower) as true pixel art with a
transparent background and a ground line. The same chain as prompt-to-pixel.json - Z-Image Turbo with the pixel-art
LoRA, BiRefNet, CleanCutout, Pixelize - with two differences:

  * "Scenery Prompt" replaces "Pixel Art Prompt". PixelArtPrompt asks for one subject small in the middle with a
    wide empty margin; a set piece stands on its base at the bottom edge and fills the frame.
  * the canvas is settable and not square. Set pieces are tall (a tree: 832x1216) or wide (a cliff: 1216x832), and
    every one of those sizes is untested on Z-Image in this repo - it has only been run at 1024x1024 - so the size
    is a widget, not a constant to be trusted.

Pixelize measures "longest side" here, not subject height: a set piece's size is given by its longest side in the
design's table (a 64x96 tree, a 160x96 cliff), and the ground line is found afterwards as the lowest opaque row.
"""
from comfy_workflow import (BIREFNET, BIREFNET_URL, BLUE, GREEN, PIXEL_LORA, PIXEL_LORA_URL, YELLOW, WORKFLOWS,
                            ZIMAGE_HF, ZIMAGE_MODEL, ZIMAGE_TE, ZIMAGE_VAE, Graph, inp, out)

OUT = WORKFLOWS / "pixel-art" / "prompt-to-setpiece.json"

SUBJECT = "a big gnarled oak tree with a thick trunk and a wide canopy"
SLOT = "setpiece"
STYLE_TEXT = ""        # empty = the node's own generic scenery sentence; Sprite Studio sets its style's sentence here
KEY_COLOR = "white"
WIDTH, HEIGHT = 832, 1216      # tall object. Wide: 1216x832. Square-ish: 1024x1024. Multiples of 16, ~1.0-1.3 MP.
LORA_STRENGTH = 1.0
STEPS, SHIFT = 8, 3.0
SEED = 42

PIXELS = 96            # final size in art pixels, measured on the longest side
COLORS = 0             # 0 = the slot's default (set pieces: automatic fewest colours)
COLOR_ERROR = 20.0     # higher = fewer colours (12 faithful, 20 simplified, 30 very few)
MAX_COLORS = 32        # Pixel Art Check limit
SMOOTH = 0.0           # the LoRA already draws flat colours
# drop_shadow, enclosed_background, min_pocket, bg_tolerance, shadow_band.
# enclosed_background is ON here and off in prompt-to-pixel.json: a set piece is full of holes a character
# can see through - arches, windows, a broken roof - and BiRefNet keeps them, because they are inside the
# object. A "stone keep" came back as a frame around a solid white middle until this was turned on.
CUTOUT = [True, True, 0.02, 8.0, 0.2]
# images, pixels, measure, method, smooth, colors, color_error, palette, shared_palette, background, bg_tolerance,
# cleanup, edge_cleanup, outline, padding, preview_scale
PIXELIZE = [PIXELS, "longest side", "pixeloe", SMOOTH, COLORS, COLOR_ERROR, "auto", True, "auto remove", 12.0, True,
            True, "auto", 0, 8]

NOTE = f"""Prompt -> set piece (one large scenery object)

1. Scenery Prompt (green): type just the object ("a ruined stone watchtower"). slot = setpiece writes the set-piece
   template: one large object, side view, whole object in frame, standing on its base, on a plain flat key colour.
   The other slots (sky, clouds, far, mid, near, foreground) are for parallax layers and want a wide canvas instead.
   pixels / colors / palette set here also drive Pixelize and the check. See "Final prompt".
2. Canvas size (blue): {WIDTH}x{HEIGHT} for a tall object, 1216x832 for a wide one, 1024x1024 for a square one.
   Keep it a multiple of 16 and near 1.0-1.3 megapixels. These non-square sizes are UNTESTED on Z-Image Turbo in
   this repo - if the object comes out stretched or doubled, try 1024x1024 and crop, and note what happened.
3. Model: Z-Image Turbo (Apache-2.0), 8 steps, CFG 1, with the pixel-art LoRA (Ctrl+B to skip it).
4. BiRefNet cuts the object out; Clean cutout drops the drop shadow it tends to draw under it.
5. Pixelize (yellow): measure is "longest side" - 'pixels' is the longest side of the finished object, so a tree
   96 px tall and a cliff 160 px wide are both asked for the same way. padding 0: a set piece's base must sit on
   the bottom row, because the engine places it by its ground line.
6. Run. Saved to output/pixel/:
   prompt_to_setpiece_*.png      native size (transparent) -> use in games
   prompt_to_setpiece_x8_*.png   zoomed preview
   prompt_to_setpiece_raw_*.png  the generated {WIDTH}x{HEIGHT} image
"Pixel Art Check" must say PASS. The lowest opaque row of the result is its ground line."""

g = Graph()
node, link = g.node, g.link


def models(name, url, directory, type_):
    return {"Node name for S&R": type_, "models": [{"name": name, "url": url, "directory": directory}]}


node(1, "UNETLoader", [0, 0], [360, 82], outputs=[out("MODEL", "MODEL")], widgets=[ZIMAGE_MODEL, "default"],
     title="Z-Image Turbo", props=models(ZIMAGE_MODEL, f"{ZIMAGE_HF}/diffusion_models/{ZIMAGE_MODEL}", "diffusion_models", "UNETLoader"))
node(2, "LoraLoaderModelOnly", [0, 120], [360, 82], inputs=[inp("model", "MODEL")], outputs=[out("MODEL", "MODEL")],
     widgets=[PIXEL_LORA, LORA_STRENGTH], title="Pixel-art LoRA", color=YELLOW,
     props={"Node name for S&R": "LoraLoaderModelOnly", "models": [
         {"name": PIXEL_LORA, "url": PIXEL_LORA_URL, "directory": "loras"}]})
node(3, "ModelSamplingAuraFlow", [0, 240], [360, 58], inputs=[inp("model", "MODEL")], outputs=[out("MODEL", "MODEL")],
     widgets=[SHIFT])
node(4, "CLIPLoader", [0, 340], [360, 106], outputs=[out("CLIP", "CLIP")], widgets=[ZIMAGE_TE, "lumina2", "default"],
     props=models(ZIMAGE_TE, f"{ZIMAGE_HF}/text_encoders/{ZIMAGE_TE}", "text_encoders", "CLIPLoader"))
node(5, "VAELoader", [0, 490], [360, 58], outputs=[out("VAE", "VAE")], widgets=[ZIMAGE_VAE],
     props=models(ZIMAGE_VAE, f"{ZIMAGE_HF}/vae/{ZIMAGE_VAE}", "vae", "VAELoader"))
node(23, "SceneryPrompt", [400, -560], [420, 400], outputs=[out("prompt", "STRING"), out("pixels", "INT"),
     out("colors", "INT"), out("palette", "STRING")],
     widgets=[SUBJECT, SLOT, PIXELS, COLORS, "auto", KEY_COLOR, True, "", STYLE_TEXT],
     title="Scenery Prompt", color=GREEN)
node(24, "PreviewAny", [860, -560], [420, 240], inputs=[inp("source", "*")], title="Final prompt")
node(6, "CLIPTextEncode", [400, 0], [420, 120], inputs=[inp("clip", "CLIP"), inp("text", "STRING", "text")],
     outputs=[out("CONDITIONING", "CONDITIONING")], widgets=[""], title="Encode prompt")
node(7, "ConditioningZeroOut", [400, 260], [240, 26], inputs=[inp("conditioning", "CONDITIONING")],
     outputs=[out("CONDITIONING", "CONDITIONING")])
node(8, "EmptySD3LatentImage", [400, 330], [300, 106], outputs=[out("LATENT", "LATENT")], widgets=[WIDTH, HEIGHT, 1],
     title="Canvas size", color=BLUE)
node(9, "KSampler", [860, 0], [320, 262],
     inputs=[inp("model", "MODEL"), inp("positive", "CONDITIONING"), inp("negative", "CONDITIONING"),
             inp("latent_image", "LATENT")],
     outputs=[out("LATENT", "LATENT")], widgets=[SEED, "randomize", STEPS, 1.0, "res_multistep", "simple", 1.0])
node(10, "VAEDecode", [860, 310], [200, 46], inputs=[inp("samples", "LATENT"), inp("vae", "VAE")], outputs=[out("IMAGE", "IMAGE")])
node(11, "SaveImage", [860, 400], [320, 360], inputs=[inp("images", "IMAGE")], widgets=["pixel/prompt_to_setpiece_raw"],
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
node(17, "SaveImage", [1980, 0], [360, 300], inputs=[inp("images", "IMAGE")], widgets=["pixel/prompt_to_setpiece"],
     title="Save set piece (native size)", color=GREEN)
node(18, "SaveImage", [1980, 340], [360, 420], inputs=[inp("images", "IMAGE")], widgets=["pixel/prompt_to_setpiece_x8"],
     title="Save preview (x8)")
node(19, "PixelArtCheck", [1580, 490], [360, 106], inputs=[inp("images", "IMAGE")],
     outputs=[out("report", "STRING"), out("passed", "BOOLEAN")], widgets=[MAX_COLORS, 0, True], title="Pixel Art Check")
node(20, "PreviewAny", [1580, 640], [360, 200], inputs=[inp("source", "*")], title="Check report")
node(21, "PreviewAny", [1980, 800], [360, 160], inputs=[inp("source", "*")], title="Palette")
node(22, "Note", [400, 480], [420, 500], widgets=[NOTE], title="READ ME", props={})

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

g.save(OUT, "pixel-art-prompt-to-setpiece",
       groups=[("Scenery prompt template", [380, -620, 920, 480], "#a1309b"),
               ("Z-Image Turbo + pixel-art LoRA", [-20, -60, 1220, 900], "#444"),
               ("Background removal (BiRefNet)", [1200, -60, 360, 380], "#444")],
       scale=0.6, offset=(40, 80))
