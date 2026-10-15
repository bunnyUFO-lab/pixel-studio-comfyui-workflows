"""Generates workflows/image-edit/qwen-edit.json.

Edit the constants below and re-run:  python tools/build_qwen_edit.py

Prompt-based image editing with Qwen-Image-2.1 (same nodes/settings as ComfyUI's official edit template,
without the optional prompt enhancer), with optional inpainting: paint a mask on the input image and only
that area changes; no mask = the whole image is edited. With a mask, the masked area (+context) is cropped
and enlarged to ~1 MP before editing (the edit model edits whole images at ~1 MP, so small details would
otherwise be too small to change), then stitched back inside the mask.
"""
from comfy_workflow import (BLUE, GREEN, QWEN_HF as HF, QWEN_MODEL, QWEN_TE, QWEN_VAE, YELLOW, WORKFLOWS, Graph,
                            inp, out)

OUT = WORKFLOWS / "image-edit" / "qwen-edit.json"

PROMPT = "Make the spiders on <image1> dark red, keep everything else exactly the same."
GROW_PX, FEATHER_PX = 8, 12
SAVE_PREFIX = "edit/qwen_edit"

NOTE = f"""Image edit with optional inpainting (Qwen-Image-2.1)

1. Load an image (blue).
2. Optional - inpaint: right-click the image node -> Open in MaskEditor, paint over the
   area to change, Save. Only that area changes; everything else stays pixel-identical.
   The masked area is cropped + enlarged first, so even small details (eyes) can be edited.
   No mask = the whole image is edited.
3. Write the edit as an instruction in the prompt (green). Refer to the image as <image1>.
   Good: "Replace the scythe with a spear, keep the pose". "Make the cape dark red".
   "Remove the spider on the helmet". "Change the background to a misty forest".
   With a mask, describe the change for that area and still say "keep everything else".
4. Run. Saved to output/{SAVE_PREFIX}_*.png; the Compare node shows before/after.
To keep editing: upload the result as the new input (or drag it from the queue onto
the Load Image node), paint a new mask, run again.

Mask edges: grow_px enlarges the painted area, feather_px softens the edge.
Crop: context = how much surrounding area the model sees (0.75 = 75% of the mask size).
~30-45 s per edit on an RTX 4090. Licence: Qwen Research (non-commercial)."""


def models(name, url, directory, type_):
    return {"Node name for S&R": type_, "models": [{"name": name, "url": url, "directory": directory}]}


g = Graph()
node, link = g.node, g.link

node(1, "LoadImage", [0, 0], [400, 440], outputs=[out("IMAGE", "IMAGE"), out("MASK", "MASK")],
     widgets=["example.png", "image"], title="Input image (paint a mask to inpaint)", color=BLUE)
node(20, "UNETLoader", [440, 0], [340, 82], outputs=[out("MODEL", "MODEL")], widgets=[QWEN_MODEL, "default"],
     title="Qwen-Image-2.1",
     props=models(QWEN_MODEL, f"{HF}/diffusion_models/{QWEN_MODEL}", "diffusion_models", "UNETLoader"))
node(21, "QwenImage21Cache", [440, 120], [340, 82], inputs=[inp("model", "MODEL")], outputs=[out("MODEL", "MODEL")],
     widgets=["auto", "default"])
node(22, "CLIPLoader", [440, 240], [340, 106], outputs=[out("CLIP", "CLIP")], widgets=[QWEN_TE, "qwen_image", "default"],
     title="Qwen3-VL text encoder", props=models(QWEN_TE, f"{HF}/text_encoders/{QWEN_TE}", "text_encoders", "CLIPLoader"))
node(23, "VAELoader", [440, 390], [340, 58], outputs=[out("VAE", "VAE")], widgets=[QWEN_VAE],
     props=models(QWEN_VAE, f"{HF}/vae/{QWEN_VAE}", "vae", "VAELoader"))
node(30, "CropForMaskedEdit", [440, 500], [340, 130], inputs=[inp("image", "IMAGE"), inp("mask", "MASK")],
     outputs=[out("image", "IMAGE"), out("crop_info", "CROP_INFO")], widgets=[0.75, 64, 1.0],
     title="Crop to mask + enlarge (small areas)", color=YELLOW)
node(24, "TextEncodeQwenImage21", [820, 0], [440, 300],
     inputs=[inp("clip", "CLIP"), inp("vae", "VAE"), inp("images.image_1", "IMAGE")],
     outputs=[out("positive", "CONDITIONING"), out("negative", "CONDITIONING"), out("latent", "LATENT")],
     widgets=[PROMPT, "", 1024], title="Edit prompt", color=GREEN)
node(25, "KSampler", [820, 340], [320, 262],
     inputs=[inp("model", "MODEL"), inp("positive", "CONDITIONING"), inp("negative", "CONDITIONING"),
             inp("latent_image", "LATENT")],
     outputs=[out("LATENT", "LATENT")], widgets=[1234, "randomize", 25, 1.0, "euler", "simple", 1.0],
     title="Qwen edit")
node(26, "VAEDecode", [820, 640], [210, 46], inputs=[inp("samples", "LATENT"), inp("vae", "VAE")],
     outputs=[out("IMAGE", "IMAGE")])
node(27, "MaskedEditComposite", [1300, 0], [320, 130],
     inputs=[inp("original", "IMAGE"), inp("edited", "IMAGE"), inp("mask", "MASK"), inp("crop_info", "CROP_INFO")],
     outputs=[out("image", "IMAGE"), out("used_mask", "MASK")], widgets=[GROW_PX, FEATHER_PX],
     title="Apply edit (inside mask, or everywhere)", color=YELLOW)
node(28, "SaveImage", [1660, 0], [420, 460], inputs=[inp("images", "IMAGE")], widgets=[SAVE_PREFIX],
     title="Result", color=GREEN)
node(29, "ImageCompare", [1300, 180], [340, 420], inputs=[inp("image_a", "IMAGE"), inp("image_b", "IMAGE")],
     widgets=[], title="Before / after")
node(11, "Note", [0, 490], [460, 520], widgets=[NOTE], title="READ ME", props={})

link(20, 0, 21, 0)
link(22, 0, 24, 0)
link(23, 0, 24, 1)
link(1, 0, 30, 0)
link(1, 1, 30, 1)
link(30, 0, 24, 2)
link(21, 0, 25, 0)
link(24, 0, 25, 1)
link(24, 1, 25, 2)
link(24, 2, 25, 3)
link(25, 0, 26, 0)
link(23, 0, 26, 1)
link(1, 0, 27, 0)
link(26, 0, 27, 1)
link(1, 1, 27, 2)
link(30, 1, 27, 3)
link(27, 0, 28, 0)
link(1, 0, 29, 0)
link(27, 0, 29, 1)

g.save(OUT, "image-edit-qwen-edit", groups=[("Qwen-Image-2.1 edit", [420, -60, 860, 780], "#3f789e")],
       scale=0.7, offset=(40, 80))
