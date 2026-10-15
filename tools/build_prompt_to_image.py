"""Generates workflows/image/prompt-to-image.json.

Edit the constants below and re-run:  python tools/build_prompt_to_image.py

Prompt -> N draft images in one run (concept drafts for a character or object, not pixel art yet). Z-Image Turbo
(Apache-2.0, 8 steps); the Pixel Art Prompt template keeps drafts simple and easy to pixelate later. Set the
pixel-art LoRA strength to 0 (and lora_trigger off) for plain concept art instead of pixel-styled drafts.
"""
from comfy_workflow import (GREEN, PIXEL_LORA, PIXEL_LORA_URL, YELLOW, WORKFLOWS, ZIMAGE_HF, ZIMAGE_MODEL, ZIMAGE_TE,
                            ZIMAGE_VAE, Graph, inp, out)

OUT = WORKFLOWS / "image" / "prompt-to-image.json"

SUBJECT = "a haunted grave digger with a shovel and a lantern"
STYLE, VIEW, PIXELS, DETAIL, OUTLINE = "clean game sprite", "front three-quarter", 64, "auto (from pixels)", "dark outline"
BATCH, WIDTH, HEIGHT, STEPS, SHIFT, SEED = 4, 1024, 1024, 8, 3.0, 1
LORA_STRENGTH = 1.0

NOTE = f"""Prompt -> draft images

Type a subject in "Pixel Art Prompt" (green); pick style, view and detail. "count" ({BATCH}) drafts are made in
one run, each with different noise. Pixel-art LoRA 1.0 = pixel-styled drafts (best for converting to sprites);
0 (and lora_trigger off) = plain concept art. Saved to output/draft/prompt_to_image_*.png.
Then use pixel-art/image-to-pixel (or the Sprite Studio app) to turn the one you like into a sprite."""


def models(name, url, directory, type_):
    return {"Node name for S&R": type_, "models": [{"name": name, "url": url, "directory": directory}]}


g = Graph()
node, link = g.node, g.link

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
node(6, "PixelArtPrompt", [400, -480], [420, 460], outputs=[out("prompt", "STRING"), out("pixels", "INT"),
     out("colors", "INT"), out("palette", "STRING")],
     widgets=[SUBJECT, STYLE, VIEW, PIXELS, DETAIL, 0, "auto", OUTLINE, "plain white", True, ""], title="Pixel Art Prompt", color=GREEN)
node(7, "PreviewAny", [860, -480], [420, 240], inputs=[inp("source", "*")], title="Final prompt")
node(8, "CLIPTextEncode", [400, 0], [420, 120], inputs=[inp("clip", "CLIP"), inp("text", "STRING", "text")],
     outputs=[out("CONDITIONING", "CONDITIONING")], widgets=[""], title="Encode prompt")
node(9, "ConditioningZeroOut", [400, 160], [240, 26], inputs=[inp("conditioning", "CONDITIONING")],
     outputs=[out("CONDITIONING", "CONDITIONING")])
node(10, "EmptySD3LatentImage", [400, 230], [300, 106], outputs=[out("LATENT", "LATENT")], widgets=[WIDTH, HEIGHT, BATCH],
     title="Size / count", color=YELLOW)
node(11, "KSampler", [860, 0], [320, 262],
     inputs=[inp("model", "MODEL"), inp("positive", "CONDITIONING"), inp("negative", "CONDITIONING"),
             inp("latent_image", "LATENT")],
     outputs=[out("LATENT", "LATENT")], widgets=[SEED, "randomize", STEPS, 1.0, "res_multistep", "simple", 1.0])
node(12, "VAEDecode", [860, 310], [200, 46], inputs=[inp("samples", "LATENT"), inp("vae", "VAE")], outputs=[out("IMAGE", "IMAGE")])
node(13, "SaveImage", [1220, 0], [520, 560], inputs=[inp("images", "IMAGE")], widgets=["draft/prompt_to_image"],
     title="Drafts", color=GREEN)
node(14, "Note", [0, 600], [380, 300], widgets=[NOTE], title="READ ME", props={})

link(1, 0, 2, 0)
link(2, 0, 3, 0)
link(3, 0, 11, 0)
link(4, 0, 8, 0)
link(6, 0, 8, 1)
link(6, 0, 7, 0)
link(8, 0, 9, 0)
link(8, 0, 11, 1)
link(9, 0, 11, 2)
link(10, 0, 11, 3)
link(11, 0, 12, 0)
link(5, 0, 12, 1)
link(12, 0, 13, 0)

g.save(OUT, "image-prompt-to-image", groups=[("Z-Image Turbo", [-20, -60, 1220, 700], "#444")], scale=0.6,
       offset=(40, 520),
       app=([(6, "subject"), (6, "style"), (6, "view"), (6, "pixels"), (10, "batch_size"), (2, "strength_model"),
             (11, "seed")], [13]))
