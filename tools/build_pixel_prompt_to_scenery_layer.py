"""Generates workflows/pixel-art/prompt-to-scenery-layer.json  (W1 of the pixel-scenery design).

Edit the constants below and re-run:  python tools/build_pixel_prompt_to_scenery_layer.py

Prompt -> one parallax background layer: a strip as tall as the screen and twice as wide, with transparent pixels
above the skyline, that repeats sideways. Six of these stacked with different parallax speeds make a scene.

Three things make this different from prompt-to-setpiece.json, and all three were measured rather than assumed:

  * the canvas is NOT the layer's shape. Z-Image Turbo composes a scene that spans the full width up to about
    2.4:1 and gives up past it: asked for a mountain range across a 2048x576 frame (3.56:1) it drew one small peak
    dead centre with empty bands either side. A layer IS 3.56:1, so the shape cannot be negotiated - but the
    canvas can. Drawing at 2048x864 (2.37:1) and cropping the band gives the same pixels in one pass, with no
    seam. Measured 2026-10-09; see the Pixel Scenery card.
  * no BiRefNet. It looks for one main object, and a skyline running edge to edge is not one - run a layer
    through the set-piece chain and it keys out nearly everything, then Pixelize divides by an empty bounding box.
    The sky here is a flat colour connected to the top edge, which is exactly what Pixelize's own "auto remove"
    clears, so no cut-out model is involved at all.
  * it has to repeat sideways and must NOT repeat vertically: wrapping a layer top to bottom would fold the sky
    into the ground. Hence MakeTileable with axis="x".
"""
from comfy_workflow import (BLUE, GREEN, PIXEL_LORA, PIXEL_LORA_URL, YELLOW, WORKFLOWS,
                            ZIMAGE_HF, ZIMAGE_MODEL, ZIMAGE_TE, ZIMAGE_VAE, Graph, inp, out)

OUT = WORKFLOWS / "pixel-art" / "prompt-to-scenery-layer.json"

SUBJECT = "mountain range with snow on the peaks"
SLOT = "far"             # sky | clouds | far | mid | near | foreground - the depth this layer sits at
STYLE_TEXT = ""          # empty = the node's own generic scenery sentence; Sprite Studio sets its style's here
KEY_COLOR = "white"

# The drawing canvas: 2.37:1, which the model composes across. NOT the layer's shape.
WIDTH, HEIGHT = 2048, 864
BAND = 576               # rows kept out of it -> 2048x576, which is 3.56:1, the layer's real shape
LAYER_W, LAYER_H = 640, 180      # the finished layer in art pixels (320x180 screen; double for 640x360)
# A landscape model paints a sky however firmly the prompt asks for flat white, so the sky is cleared by
# what it is - a region that changes slowly downwards - rather than by its colour. Before Pixelize, while
# the gradient is still smooth: afterwards it is flat bands and every band edge reads as a skyline.
SKY_STEP, SKY_FEATHER, SKY_KEEP = 14.0, 2, 0.02

LORA_STRENGTH = 1.0
STEPS, SHIFT = 8, 3.0
SEED = 42

COLORS = 0               # 0 = the slot's default (far 5, mid 9, near 10, clouds and foreground 3)
COLOR_ERROR = 20.0
MAX_COLORS = 32
SMOOTH = 0.0
# images, pixels, measure, method, smooth, colors, color_error, palette, shared_palette, background, bg_tolerance,
# cleanup, edge_cleanup, outline, padding, preview_scale
# "whole frame", not "image height": every other measure places the grid on the subject's bounding box and trims
# to it, which is right for a sprite and wrong for a layer. A layer must keep its exact size and its transparent
# sky, or it will not line up with the layers behind it; and a clouds layer, which is mostly cleared key colour,
# has no bounding box left to measure at all. outline none and padding 0: a layer has no silhouette to outline
# and is cut off at both sides on purpose.
PIXELIZE = [LAYER_H, "whole frame", "pixeloe", SMOOTH, COLORS, COLOR_ERROR, "auto", True, "auto remove", 12.0,
            True, True, "none", 0, 4]

NOTE = f"""Prompt -> one parallax scenery layer

1. Scenery Prompt (green): type the layer's subject ("a distant mountain range"), and pick the SLOT it sits at.
   Each slot writes its own template: far is hazy and low contrast with pale colours near the sky, near is dark
   and high contrast along the bottom, foreground is almost black silhouettes. The slot also sets the colour
   count unless you override it. sky and clouds are slots too, though a plain sky is cheaper drawn as a gradient.
2. Canvas size (blue): {WIDTH}x{HEIGHT}. This is NOT the layer's shape and should not be set to it. Z-Image stops
   composing a full-width scene past about 2.4:1 - at {WIDTH}x{BAND} it draws one object in the middle of an empty
   frame. Draw at 2.37:1, crop the band. If you change the width, keep the ratio near 2.4:1.
3. Crop to strip (blue): keeps {BAND} rows of the full width. anchor "auto" picks the band with the most detail in
   it - the horizon and the ground under it - because where the skyline lands moves from seed to seed. Use top /
   centre / bottom when you want a particular part.
3b. Clear sky (blue): everything above the skyline becomes transparent, so the layers behind this one show
   through. It works by growing a region down from the top edge and stopping where the picture stops changing
   slowly - a skyline - rather than by looking for a key colour, because the model paints a sky whatever the
   prompt asks for. Raise `step` if sky survives; lower it if the skyline is being eaten.
4. Pixelize (yellow): measure "image height", so the strip becomes exactly {LAYER_H} art pixels tall and
   {LAYER_W} wide. background "auto remove" clears the flat sky colour connected to the edge - that is the whole
   sky-clearing step, and why no BiRefNet appears in this graph.
5. Make Tileable (yellow): axis "x". The layer repeats left to right; it must never wrap top to bottom.
   Its preview shows the strip side by side, which is how you check the seam.
6. Run. Saved to output/pixel/:
   prompt_to_scenery_layer_*.png       the layer, {LAYER_W}x{LAYER_H}, transparent above the skyline -> the game
   prompt_to_scenery_layer_tiled_*.png the same repeated, to check the seam
   prompt_to_scenery_layer_raw_*.png   the {WIDTH}x{HEIGHT} drawing, before the crop

Stack six of these - sky, clouds, far, mid, near, foreground - at different parallax speeds to make a scene."""

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
     out("colors", "INT"), out("palette", "STRING"), out("sky_step", "FLOAT")],
     widgets=[SUBJECT, SLOT, LAYER_H, COLORS, "auto", KEY_COLOR, True, "", STYLE_TEXT],
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
node(11, "SaveImage", [860, 400], [320, 300], inputs=[inp("images", "IMAGE")],
     widgets=["pixel/prompt_to_scenery_layer_raw"], title="Save drawing (before the crop)")
node(12, "CropToStrip", [1220, 0], [320, 106], inputs=[inp("images", "IMAGE")],
     outputs=[out("image", "IMAGE"), out("top", "INT")], widgets=[BAND, "auto"],
     title="Crop to strip", color=BLUE)
node(20, "ClearSky", [1220, 160], [320, 82], inputs=[inp("images", "IMAGE"), inp("step", "FLOAT", "step")],
     outputs=[out("image", "IMAGE"), out("cleared", "FLOAT")], widgets=[SKY_STEP, SKY_FEATHER, SKY_KEEP],
     title="Clear sky", color=BLUE)
node(13, "Pixelize", [1580, 0], [360, 440], inputs=[inp("images", "IMAGE"), inp("colors", "INT", "colors"),
                                                    inp("palette", "STRING", "palette")],
     outputs=[out("pixel_art", "IMAGE"), out("preview", "IMAGE"), out("palette", "STRING"), out("info", "STRING")],
     widgets=PIXELIZE, title="Pixelize", color=YELLOW)
node(14, "MakeTileable", [1980, 0], [360, 220], inputs=[inp("images", "IMAGE")],
     outputs=[out("tile", "IMAGE"), out("preview", "IMAGE")],
     widgets=[LAYER_W, 0, 3, 2, "x", LAYER_H], title="Make Tileable (sideways)", color=YELLOW)
node(15, "SaveImage", [2380, 0], [360, 260], inputs=[inp("images", "IMAGE")],
     widgets=["pixel/prompt_to_scenery_layer"], title="Save layer (native size)", color=GREEN)
node(16, "SaveImage", [2380, 300], [360, 300], inputs=[inp("images", "IMAGE")],
     widgets=["pixel/prompt_to_scenery_layer_tiled"], title="Save repeat (check the seam)")
node(17, "PixelArtCheck", [1980, 260], [360, 106], inputs=[inp("images", "IMAGE")],
     outputs=[out("report", "STRING"), out("passed", "BOOLEAN")], widgets=[MAX_COLORS, 0, True], title="Pixel Art Check")
node(18, "PreviewAny", [1980, 400], [360, 180], inputs=[inp("source", "*")], title="Check report")
node(19, "PreviewAny", [2380, 640], [360, 140], inputs=[inp("source", "*")], title="Palette")
node(22, "Note", [400, 480], [420, 520], widgets=[NOTE], title="READ ME", props={})

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
link(10, 0, 12, 0)           # the drawing -> crop the band
link(12, 0, 20, 0)           # the band -> clear its sky (while the gradient is still smooth)
link(20, 0, 13, 0)           # the cleared band -> pixelize to the layer's art size
link(13, 0, 14, 0)           # the layer -> make it repeat sideways
link(14, 0, 15, 0)
link(14, 1, 16, 0)
link(14, 0, 17, 0)
link(17, 0, 18, 0)
link(13, 2, 19, 0)
link(23, 0, 6, 1)            # template prompt -> text encoder
link(23, 0, 24, 0)
link(23, 2, 13, 1)           # the slot's colour count and palette drive Pixelize
link(23, 3, 13, 2)
link(23, 4, 20, 1)           # the slot decides how sensitive the sky clearing is

g.save(OUT, "pixel-art-prompt-to-scenery-layer",
       groups=[("Scenery prompt template", [380, -620, 920, 480], "#a1309b"),
               ("Z-Image Turbo + pixel-art LoRA", [-20, -60, 1220, 900], "#444"),
               ("Draw wide, crop the band", [1200, -60, 360, 200], "#2a6")],
       scale=0.55, offset=(40, 80))
