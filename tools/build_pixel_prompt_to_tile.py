"""Generates workflows/pixel-art/prompt-to-tile.json.

Edit the constants below and re-run:  python tools/build_pixel_prompt_to_tile.py

Prompt -> seamless pixel-art tile for tile maps: Z-Image Turbo + pixel-art LoRA draw a top-down terrain texture
(the Pixel Art Prompt "terrain tile (seamless)" style), Pixelize puts it on an exact grid with a small palette
(pass a palette to keep a whole tileset consistent), and MakeTileable cuts a tile that repeats seamlessly using
only the texture's own pixels (minimum-error wrap seams).
"""
from comfy_workflow import (GREEN, PIXEL_LORA, PIXEL_LORA_URL, YELLOW, WORKFLOWS, ZIMAGE_HF, ZIMAGE_MODEL, ZIMAGE_TE,
                            ZIMAGE_VAE, Graph, inp, out)

OUT = WORKFLOWS / "pixel-art" / "prompt-to-tile.json"

SUBJECT = "lush green grass with small flowers"
TILE = 32
TEXTURE = 64            # the texture is pixelized to TEXTURE x TEXTURE art px (2 tiles wide); the tile is cut from it
COLORS, COLOR_ERROR = 6, 16.0
SEED = 3
# images, pixels, measure, method, smooth, colors, color_error, palette, shared_palette, background, bg_tolerance, cleanup,
# edge_cleanup, outline, padding, preview_scale
PIXELIZE = [TEXTURE, "image height", "median", 0.6, COLORS, COLOR_ERROR, "auto", True, "keep", 12.0, True, False, "none", 0, 4]

NOTE = f"""Prompt -> seamless pixel tile

1. Pixel Art Prompt (green): describe the TERRAIN ("grass with flowers", "cobblestone path", "shallow water",
   "dungeon floor bricks"). Style is "terrain tile (seamless)".
2. Pixelize makes a {TEXTURE}x{TEXTURE} art-px texture (palette: auto, or paste the palette of another tile so a
   whole tileset matches). Make Tileable cuts a {TILE}x{TILE} tile that repeats without seams.
Saved to output/pixel/tile_*.png (tile) and tile_preview_*.png (3x3 repeat, x4)."""


def models(name, url, directory, type_):
    return {"Node name for S&R": type_, "models": [{"name": name, "url": url, "directory": directory}]}


g = Graph()
node, link = g.node, g.link

node(1, "UNETLoader", [0, 0], [360, 82], outputs=[out("MODEL", "MODEL")], widgets=[ZIMAGE_MODEL, "default"],
     title="Z-Image Turbo", props=models(ZIMAGE_MODEL, f"{ZIMAGE_HF}/diffusion_models/{ZIMAGE_MODEL}", "diffusion_models", "UNETLoader"))
node(2, "LoraLoaderModelOnly", [0, 120], [360, 82], inputs=[inp("model", "MODEL")], outputs=[out("MODEL", "MODEL")],
     widgets=[PIXEL_LORA, 0.0], title="Pixel-art LoRA", props=models(PIXEL_LORA, PIXEL_LORA_URL, "loras", "LoraLoaderModelOnly"))
# LoRA off for textures: it draws its own fine fake-pixel grid, which clashes (moire) with the tile grid
node(3, "ModelSamplingAuraFlow", [0, 240], [360, 58], inputs=[inp("model", "MODEL")], outputs=[out("MODEL", "MODEL")],
     widgets=[3.0])
node(4, "CLIPLoader", [0, 340], [360, 106], outputs=[out("CLIP", "CLIP")], widgets=[ZIMAGE_TE, "lumina2", "default"],
     props=models(ZIMAGE_TE, f"{ZIMAGE_HF}/text_encoders/{ZIMAGE_TE}", "text_encoders", "CLIPLoader"))
node(5, "VAELoader", [0, 490], [360, 58], outputs=[out("VAE", "VAE")], widgets=[ZIMAGE_VAE],
     props=models(ZIMAGE_VAE, f"{ZIMAGE_HF}/vae/{ZIMAGE_VAE}", "vae", "VAELoader"))
node(6, "PixelArtPrompt", [400, -480], [420, 460], outputs=[out("prompt", "STRING"), out("pixels", "INT"),
     out("colors", "INT"), out("palette", "STRING")],
     widgets=[SUBJECT, "terrain tile (seamless)", "top-down", TEXTURE, "auto (from pixels)", COLORS, "auto", "no outline",
              "plain white", False, ""], title="Pixel Art Prompt", color=GREEN)
node(7, "PreviewAny", [860, -480], [420, 200], inputs=[inp("source", "*")], title="Final prompt")
node(8, "CLIPTextEncode", [400, 0], [420, 120], inputs=[inp("clip", "CLIP"), inp("text", "STRING", "text")],
     outputs=[out("CONDITIONING", "CONDITIONING")], widgets=[""], title="Encode prompt")
node(9, "ConditioningZeroOut", [400, 160], [240, 26], inputs=[inp("conditioning", "CONDITIONING")],
     outputs=[out("CONDITIONING", "CONDITIONING")])
node(10, "EmptySD3LatentImage", [400, 230], [300, 106], outputs=[out("LATENT", "LATENT")], widgets=[1024, 1024, 1])
node(11, "KSampler", [860, 0], [320, 262],
     inputs=[inp("model", "MODEL"), inp("positive", "CONDITIONING"), inp("negative", "CONDITIONING"),
             inp("latent_image", "LATENT")],
     outputs=[out("LATENT", "LATENT")], widgets=[SEED, "randomize", 8, 1.0, "res_multistep", "simple", 1.0])
node(12, "VAEDecode", [860, 310], [200, 46], inputs=[inp("samples", "LATENT"), inp("vae", "VAE")], outputs=[out("IMAGE", "IMAGE")])
node(13, "Pixelize", [1220, 0], [360, 440], inputs=[inp("images", "IMAGE"), inp("colors", "INT", "colors"),
                                                   inp("palette", "STRING", "palette")],
     outputs=[out("pixel_art", "IMAGE"), out("preview", "IMAGE"), out("palette", "STRING"), out("info", "STRING")],
     widgets=PIXELIZE, title="Pixelize (texture)", color=YELLOW)
node(14, "MakeTileable", [1620, 0], [340, 160], inputs=[inp("images", "IMAGE")], outputs=[out("tile", "IMAGE"), out("preview", "IMAGE")],
     widgets=[TILE, 0, 3, 4], title="Make Tileable", color=YELLOW)
node(15, "SaveImage", [2000, 0], [300, 300], inputs=[inp("images", "IMAGE")], widgets=["pixel/tile"],
     title="Save tile (native size)", color=GREEN)
node(16, "SaveImage", [2000, 340], [400, 440], inputs=[inp("images", "IMAGE")], widgets=["pixel/tile_preview"],
     title="Save 3x3 preview")
node(17, "PreviewAny", [1620, 220], [340, 140], inputs=[inp("source", "*")], title="Palette")
node(18, "Note", [0, 600], [380, 300], widgets=[NOTE], title="READ ME", props={})
node(19, "SaveImage", [1220, 500], [360, 360], inputs=[inp("images", "IMAGE")], widgets=["pixel/tile_texture_raw"],
     title="Save raw texture")
node(20, "SaveImage", [1620, 400], [340, 360], inputs=[inp("images", "IMAGE")], widgets=["pixel/tile_texture"],
     title="Save pixel texture (x4)")

link(1, 0, 2, 0)
link(2, 0, 3, 0)
link(3, 0, 11, 0)
link(4, 0, 8, 0)
link(6, 0, 8, 1)
link(6, 0, 7, 0)
link(6, 2, 13, 1)
link(6, 3, 13, 2)
link(8, 0, 9, 0)
link(8, 0, 11, 1)
link(9, 0, 11, 2)
link(10, 0, 11, 3)
link(11, 0, 12, 0)
link(5, 0, 12, 1)
link(12, 0, 13, 0)
link(13, 0, 14, 0)
link(14, 0, 15, 0)
link(14, 1, 16, 0)
link(13, 2, 17, 0)
link(12, 0, 19, 0)
link(13, 1, 20, 0)

g.save(OUT, "pixel-art-prompt-to-tile", groups=[("Z-Image Turbo", [-20, -60, 1220, 700], "#444")], scale=0.6,
       offset=(40, 520),
       app=([(6, "subject"), (6, "palette"), (14, "tile_size"), (11, "seed")], [15, 16, 17]))
