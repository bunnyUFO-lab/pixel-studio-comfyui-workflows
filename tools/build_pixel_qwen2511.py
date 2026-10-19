"""Generates the Qwen-Image-Edit-2511 versions of the Qwen-Image-2.1 sprite workflows (public-release milestone S8, and
the Klein vs Qwen test):

  workflows/pixel-art/sprite-edit-2511.json          (= sprite-edit)
  workflows/pixel-art/image-to-directions-2511.json  (= image-to-directions, with fal's Multiple-Angles LoRA)

Edit the constants below and re-run:  python tools/build_pixel_qwen2511.py

Same pixel steps as the 2.1 versions; the model part is wired like Comfy's "Qwen Image Edit 2511" template:
2511 int8, Qwen2.5-VL 7B text encoder, AuraFlow shift 3.1, CFGNorm, Text Encode Qwen Image Edit Plus with the
"index_timestep_zero" reference method, empty negative. The lightx2v Lightning LoRA gives 4 steps at cfg 1; for the
full model set its strength to 0, steps 40, cfg 4. Everything here is Apache-2.0.
"""
from comfy_workflow import (BIREFNET, BIREFNET_URL, BLUE, GREEN, RED, WORKFLOWS, YELLOW, Graph, inp, out)

EDIT = WORKFLOWS / "pixel-art" / "sprite-edit-2511.json"
DIRECTIONS = WORKFLOWS / "pixel-art" / "image-to-directions-2511.json"

HF = "https://huggingface.co"
MODEL = "qwen_image_edit_2511_int8_convrot.safetensors"     # 20.5 GB
MODEL_URL = f"{HF}/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/{MODEL}"
TE = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
TE_URL = f"{HF}/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/text_encoders/{TE}"
VAE = "qwen_image_vae.safetensors"
VAE_URL = f"{HF}/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/{VAE}"
LIGHTNING = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
LIGHTNING_URL = f"{HF}/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/{LIGHTNING}"
ANGLES = "qwen-image-edit-2511-multiple-angles-lora.safetensors"
ANGLES_URL = f"{HF}/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA/resolve/main/{ANGLES}"

STEPS, CFG, SHIFT = 4, 1.0, 3.1
EDIT_PROMPT = "Give the character a red scarf. Keep everything else exactly the same, same pose, same pixel-art style."
TARGET, PADDING, EDIT_SEED = 1024, 0.25, 7
REPIX = [False, 18.0, "dominant", 6, "sprite+new", 0.0, False, True, "auto"]

# fal's camera tags (azimuth / elevation / distance). They name where the camera goes: "right side view" shows the
# character facing the LEFT edge, so our right-facing views use the left-side tags.
VIEWS = [
    ("front", "front view"),
    ("front-right", "front-left quarter view"),
    ("right", "left side view"),
    ("back-right", "back-left quarter view"),
    ("back", "back view"),
]
TAIL = "eye-level shot medium shot"
PROMPTS = "\n".join(f"<sks> {az} {TAIL}" for _, az in VIEWS)
NAMES = ", ".join(n for n, _ in VIEWS)
DIR_SEED = 1234
PIXELS, COLORS, COLOR_ERROR, MAX_COLORS = 64, 0, 20.0, 32
CUTOUT = [True, False, 0.02, 8.0, 0.2]
PIXELIZE = [PIXELS, "body height (per frame)", "pixeloe", 0.6, COLORS, COLOR_ERROR, "auto", True, "auto remove", 12.0,
            True, True, "auto", 1, 8]


def models(name, url, directory, type_):
    return {"Node name for S&R": type_, "models": [{"name": name, "url": url, "directory": directory}]}


def model_chain(g, x, y, angles=False):
    """Loader (60) -> [Angles LoRA (66)] -> Lightning LoRA (61) -> shift (62) -> CFGNorm (63); TE (64), VAE (65).
    The model output to sample with is node 63."""
    g.node(60, "UNETLoader", [x, y], [340, 82], outputs=[out("MODEL", "MODEL")], widgets=[MODEL, "default"],
           title="Qwen-Image-Edit-2511", props=models(MODEL, MODEL_URL, "diffusion_models", "UNETLoader"))
    if angles:
        g.node(66, "LoraLoaderModelOnly", [x, y + 120], [340, 82], inputs=[inp("model", "MODEL")],
               outputs=[out("MODEL", "MODEL")], widgets=[ANGLES, 1.0], title="Multiple-Angles LoRA",
               props=models(ANGLES, ANGLES_URL, "loras", "LoraLoaderModelOnly"))
    g.node(61, "LoraLoaderModelOnly", [x, y + 240], [340, 82], inputs=[inp("model", "MODEL")],
           outputs=[out("MODEL", "MODEL")], widgets=[LIGHTNING, 1.0],
           title="Lightning LoRA (4 steps; strength 0 for full model)",
           props=models(LIGHTNING, LIGHTNING_URL, "loras", "LoraLoaderModelOnly"))
    g.node(62, "ModelSamplingAuraFlow", [x, y + 360], [340, 58], inputs=[inp("model", "MODEL")],
           outputs=[out("MODEL", "MODEL")], widgets=[SHIFT])
    g.node(63, "CFGNorm", [x, y + 450], [340, 82], inputs=[inp("model", "MODEL")], outputs=[out("patched_model", "MODEL")],
           widgets=[1.0, False])
    g.node(64, "CLIPLoader", [x, y + 570], [340, 106], outputs=[out("CLIP", "CLIP")], widgets=[TE, "qwen_image", "default"],
           title="Qwen2.5-VL 7B text encoder", props=models(TE, TE_URL, "text_encoders", "CLIPLoader"))
    g.node(65, "VAELoader", [x, y + 710], [340, 58], outputs=[out("VAE", "VAE")], widgets=[VAE],
           props=models(VAE, VAE_URL, "vae", "VAELoader"))
    if angles:
        g.link(60, 0, 66, 0)
        g.link(66, 0, 61, 0)
    else:
        g.link(60, 0, 61, 0)
    g.link(61, 0, 62, 0)
    g.link(62, 0, 63, 0)


def encoders(g, x, y, pos_id, neg_id, prompt_widget):
    """Positive (pos_id) and empty negative (neg_id) Text Encode Qwen Image Edit Plus, each followed by the
    index_timestep_zero reference method (pos_id + 1, neg_id + 1)."""
    g.node(pos_id, "TextEncodeQwenImageEditPlus", [x, y], [420, 200],
           inputs=[inp("clip", "CLIP"), inp("vae", "VAE"), inp("image1", "IMAGE")]
           + ([inp("prompt", "STRING", "prompt")] if prompt_widget is None else []),
           outputs=[out("CONDITIONING", "CONDITIONING")], widgets=[prompt_widget or ""],
           title="Edit prompt" if prompt_widget else "Encode view", color=GREEN if prompt_widget else None)
    g.node(pos_id + 1, "FluxKontextMultiReferenceLatentMethod", [x, y + 240], [420, 58],
           inputs=[inp("conditioning", "CONDITIONING")], outputs=[out("CONDITIONING", "CONDITIONING")],
           widgets=["index_timestep_zero"])
    g.node(neg_id, "TextEncodeQwenImageEditPlus", [x, y + 330], [420, 120],
           inputs=[inp("clip", "CLIP"), inp("vae", "VAE"), inp("image1", "IMAGE")],
           outputs=[out("CONDITIONING", "CONDITIONING")], widgets=[""], title="Empty negative", color=RED)
    g.node(neg_id + 1, "FluxKontextMultiReferenceLatentMethod", [x, y + 490], [420, 58],
           inputs=[inp("conditioning", "CONDITIONING")], outputs=[out("CONDITIONING", "CONDITIONING")],
           widgets=["index_timestep_zero"])
    for e in (pos_id, neg_id):
        g.link(64, 0, e, 0)
        g.link(65, 0, e, 1)
    g.link(pos_id, 0, pos_id + 1, 0)
    g.link(neg_id, 0, neg_id + 1, 0)


def sampler(g, id_, x, y, seed, control):
    g.node(id_, "KSampler", [x, y], [320, 262],
           inputs=[inp("model", "MODEL"), inp("positive", "CONDITIONING"), inp("negative", "CONDITIONING"),
                   inp("latent_image", "LATENT")],
           outputs=[out("LATENT", "LATENT")], widgets=[seed, control, STEPS, CFG, "euler", "simple", 1.0],
           title="Qwen 2511 sample")


def build_edit():
    g = Graph()
    node, link = g.node, g.link
    note = f"""Sprite edit with Qwen-Image-Edit-2511 (keeps the exact pixel grid)

1. Load a pixel sprite (blue); a batch of frames works too.
2. Edit prompt (green): a short instruction ("give him a red scarf"); say what to keep.
3. 2511 edits a x{TARGET // 64}-ish upscale: Lightning LoRA, {STEPS} steps, cfg 1 (full model: LoRA strength 0, 40 steps,
   cfg 4), then it is re-pixelated on the sprite's own grid ('sprite+new' palette = sprite + up to 'colors' new).
4. Restore: 'original canvas' = same size and position as the input.
Saved to output/pixel/sprite_edit_2511_*.png (+ _x8 preview). Licence: Apache-2.0 (model and LoRA)."""

    node(1, "LoadImage", [0, 0], [380, 400], outputs=[out("IMAGE", "IMAGE"), out("MASK", "MASK")],
         widgets=["example.png", "image"], title="Pixel sprite", color=BLUE)
    node(2, "JoinImageWithAlpha", [0, 440], [380, 46], inputs=[inp("image", "IMAGE"), inp("alpha", "MASK")],
         outputs=[out("IMAGE", "IMAGE")], title="Keep sprite transparency")
    node(3, "PrepareSpriteForVideo", [420, 0], [340, 130], inputs=[inp("sprite", "IMAGE")],
         outputs=[out("image", "IMAGE"), out("pixel_spec", "PIXEL_SPEC"), out("width", "INT"), out("height", "INT"),
                  out("info", "STRING")],
         widgets=[TARGET, PADDING, "auto"], title="Prepare sprite (integer upscale)")
    model_chain(g, 420, 180)
    encoders(g, 800, 0, 70, 72, EDIT_PROMPT)
    node(74, "VAEEncode", [800, 600], [210, 46], inputs=[inp("pixels", "IMAGE"), inp("vae", "VAE")],
         outputs=[out("LATENT", "LATENT")])
    sampler(g, 25, 1260, 0, EDIT_SEED, "randomize")
    node(26, "VAEDecode", [1260, 300], [210, 46], inputs=[inp("samples", "LATENT"), inp("vae", "VAE")],
         outputs=[out("IMAGE", "IMAGE")])
    node(27, "RepixelateFrames", [1620, 0], [340, 250], inputs=[inp("frames", "IMAGE"), inp("pixel_spec", "PIXEL_SPEC")],
         outputs=[out("pixel_frames", "IMAGE"), out("info", "STRING"), out("palette", "STRING")], widgets=REPIX,
         title="Repixelate (same grid)", color=YELLOW)
    node(28, "RestoreSpriteCanvas", [1620, 300], [340, 82], inputs=[inp("frames", "IMAGE"), inp("pixel_spec", "PIXEL_SPEC")],
         outputs=[out("frames", "IMAGE")], widgets=["original canvas"], title="Restore canvas", color=YELLOW)
    node(29, "SaveImage", [2000, 0], [360, 320], inputs=[inp("images", "IMAGE")], widgets=["pixel/sprite_edit_2511"],
         title="Save edited sprite (native size)", color=GREEN)
    node(30, "ImageScaleBy", [2000, 360], [300, 82], inputs=[inp("image", "IMAGE")], outputs=[out("IMAGE", "IMAGE")],
         widgets=["nearest-exact", 8.0], title="Zoom x8")
    node(31, "SaveImage", [2000, 480], [360, 400], inputs=[inp("images", "IMAGE")], widgets=["pixel/sprite_edit_2511_x8"],
         title="Save preview (x8)")
    node(33, "PixelArtCheck", [1620, 430], [340, 106], inputs=[inp("images", "IMAGE")],
         outputs=[out("report", "STRING"), out("passed", "BOOLEAN")], widgets=[64, 0, True], title="Pixel Art Check")
    node(34, "PreviewAny", [1620, 580], [340, 200], inputs=[inp("source", "*")], title="Check report")
    node(35, "PreviewAny", [1620, 820], [340, 140], inputs=[inp("source", "*")], title="Palette")
    node(36, "Note", [0, 520], [380, 400], widgets=[note], title="READ ME", props={})

    link(1, 0, 2, 0)
    link(1, 1, 2, 1)
    link(2, 0, 3, 0)
    for e in (70, 72):
        link(3, 0, e, 2)
    link(3, 0, 74, 0)
    link(65, 0, 74, 1)
    link(63, 0, 25, 0)
    link(71, 0, 25, 1)
    link(73, 0, 25, 2)
    link(74, 0, 25, 3)
    link(25, 0, 26, 0)
    link(65, 0, 26, 1)
    link(26, 0, 27, 0)
    link(3, 1, 27, 1)
    link(27, 0, 28, 0)
    link(3, 1, 28, 1)
    link(28, 0, 29, 0)
    link(28, 0, 30, 0)
    link(30, 0, 31, 0)
    link(28, 0, 33, 0)
    link(33, 0, 34, 0)
    link(27, 2, 35, 0)

    g.save(EDIT, "pixel-art-sprite-edit-2511", groups=[
        ("Qwen-Image-Edit-2511", [400, -60, 380, 1040], "#3f789e"),
        ("Edit", [780, -60, 840, 760], "#a1309b"),
        ("Back to the same grid", [1600, -60, 780, 1040], "#444"),
    ], scale=0.5, offset=(40, 80),
        app=([(1, "image"), (70, "prompt"), (25, "seed"), (27, "palette"), (27, "colors"), (28, "mode")], [29, 31, 34]))


def build_directions():
    g = Graph()
    node, link = g.node, g.link
    note = f"""Character image -> 8-direction pixel sprites, with Qwen-Image-Edit-2511 + Multiple-Angles LoRA

1. Load a character image (blue); a pixel sprite is upscaled by an integer factor onto white.
2. "View prompts" (green): ONE CAMERA TAG PER VIEW (fal's Multiple-Angles LoRA), in the order of "names":
   {NAMES}
   Tags: <sks> <azimuth> <elevation> <distance>; azimuths front view, front-right quarter view, right side view,
   back-right quarter view, back view, back-left / left / front-left ...; elevations low-angle / eye-level / elevated /
   high-angle shot; distances close-up / medium shot / wide shot.
3. 2511 draws each view (Lightning, {STEPS} steps; full model: LoRA strength 0, 40 steps, cfg 4).
4. All views are pixelized together: body {PIXELS} art px tall, feet on the bottom line, one shared palette.
5. Save directions: left-side views are the mirrored right-side ones -> 8 directions.
Saved to output/pixel/: directions_2511_*_<direction>.png, sheet, .json; raw views in pixel/directions_2511_raw_*.png.
Licence: Apache-2.0 (model and both LoRAs)."""

    node(1, "LoadImage", [0, 0], [400, 440], outputs=[out("IMAGE", "IMAGE"), out("MASK", "MASK")],
         widgets=["example.png", "image"], title="Character image", color=BLUE)
    node(3, "JoinImageWithAlpha", [0, 470], [400, 46], inputs=[inp("image", "IMAGE"), inp("alpha", "MASK")],
         outputs=[out("IMAGE", "IMAGE")], title="Keep sprite transparency")
    node(2, "PrepareSpriteForVideo", [0, 530], [400, 130], inputs=[inp("sprite", "IMAGE")],
         outputs=[out("image", "IMAGE"), out("pixel_spec", "PIXEL_SPEC"), out("width", "INT"), out("height", "INT"),
                  out("info", "STRING")],
         widgets=[1024, 0.25, "#ffffff"], title="Upscale onto white (integer)")
    model_chain(g, 440, 0, angles=True)
    node(10, "StartLoop", [820, 0], [300, 160],
         inputs=[inp("parent_iteration", "INT"), inp("initial_iteration_value", "*"),
                 inp("mode.num_iterations", "INT", "mode.num_iterations")],
         outputs=[out("iteration_index", "INT"), out("is_first", "BOOLEAN"), out("is_last", "BOOLEAN"),
                  out("list_item", "*"), out("current_iteration_value", "*")],
         widgets=["simple", len(VIEWS), False], title="Views (num_iterations = number of prompt lines)", color=YELLOW)
    node(11, "SegmentPrompt", [820, 200], [460, 300], inputs=[inp("index", "INT")],
         outputs=[out("prompt", "STRING"), out("segment_seed", "INT")], widgets=[PROMPTS, DIR_SEED, "fixed"],
         title="View prompts (one line per view)", color=GREEN)
    encoders(g, 1320, 0, 70, 72, None)
    node(74, "VAEEncode", [1320, 600], [210, 46], inputs=[inp("pixels", "IMAGE"), inp("vae", "VAE")],
         outputs=[out("LATENT", "LATENT")])
    sampler(g, 15, 1780, 0, DIR_SEED, "fixed")
    node(16, "VAEDecode", [1780, 300], [210, 46], inputs=[inp("samples", "LATENT"), inp("vae", "VAE")],
         outputs=[out("IMAGE", "IMAGE")])
    node(17, "EndLoop", [2120, 0], [300, 140],
         inputs=[inp("output_value", "*"), inp("next_iteration_value", "*"), inp("terminations.termination0", "*")],
         outputs=[out("outputs", "IMAGE")], widgets=[True], title="End loop (collect views)")
    node(18, "SaveImage", [2120, 200], [320, 300], inputs=[inp("images", "IMAGE")], widgets=["pixel/directions_2511_raw"],
         title="Save generated views")
    node(30, "LoadBackgroundRemovalModel", [2460, 0], [320, 58], outputs=[out("bg_model", "BACKGROUND_REMOVAL")],
         widgets=[BIREFNET], title="BiRefNet", props=models(BIREFNET, BIREFNET_URL, "background_removal", "LoadBackgroundRemovalModel"))
    node(31, "RemoveBackground", [2460, 100], [320, 46], inputs=[inp("bg_removal_model", "BACKGROUND_REMOVAL"), inp("image", "IMAGE")],
         outputs=[out("mask", "MASK")])
    node(32, "InvertMask", [2460, 180], [320, 26], inputs=[inp("mask", "MASK")], outputs=[out("MASK", "MASK")])
    node(33, "JoinImageWithAlpha", [2460, 250], [320, 46], inputs=[inp("image", "IMAGE"), inp("alpha", "MASK")],
         outputs=[out("IMAGE", "IMAGE")], title="Remove background")
    node(34, "CleanCutout", [2460, 430], [320, 160], inputs=[inp("images", "IMAGE")],
         outputs=[out("image", "IMAGE"), out("removed", "MASK"), out("info", "STRING")], widgets=CUTOUT,
         title="Clean cutout (drop shadow, background pockets)", color=YELLOW)
    node(35, "RebatchImages", [2460, 340], [320, 58], inputs=[inp("images", "IMAGE")], outputs=[out("IMAGE", "IMAGE")],
         widgets=[64], title="Views -> one batch")
    node(36, "Pixelize", [2820, 0], [360, 440], inputs=[inp("images", "IMAGE")],
         outputs=[out("pixel_art", "IMAGE"), out("preview", "IMAGE"), out("palette", "STRING"), out("info", "STRING")],
         widgets=PIXELIZE, title="Pixelize (shared grid + palette)", color=YELLOW)
    node(37, "SaveDirections", [3220, 0], [360, 200], inputs=[inp("images", "IMAGE")],
         outputs=[out("sheet", "IMAGE"), out("directions", "IMAGE"), out("info", "STRING")],
         widgets=["pixel/directions_2511", NAMES, True, 2, 6], title="Save directions (mirror -> 8)", color=GREEN)
    node(38, "PixelArtCheck", [2820, 490], [360, 106], inputs=[inp("images", "IMAGE")],
         outputs=[out("report", "STRING"), out("passed", "BOOLEAN")], widgets=[MAX_COLORS, PIXELS, True], title="Pixel Art Check")
    node(39, "PreviewAny", [2820, 640], [360, 200], inputs=[inp("source", "*")], title="Check report")
    node(40, "PreviewAny", [3220, 250], [360, 160], inputs=[inp("source", "*")], title="Directions info")
    node(41, "SpriteAnimationPreview", [3620, 0], [520, 680], inputs=[inp("frames", "IMAGE"), inp("fps", "INT", "fps")],
         widgets=[3, 6, "checker", "loop"], title="Turnaround preview (all directions)", color=GREEN)
    node(42, "Note", [0, 700], [460, 560], widgets=[note], title="READ ME", props={})

    link(1, 0, 3, 0)
    link(1, 1, 3, 1)
    link(3, 0, 2, 0)
    link(2, 0, 10, 1)            # carried value: the upscaled input image (unchanged)
    link(10, 0, 11, 0)
    link(11, 0, 70, 3)
    for e in (70, 72):
        link(2, 0, e, 2)
    link(2, 0, 74, 0)
    link(65, 0, 74, 1)
    link(63, 0, 15, 0)
    link(71, 0, 15, 1)
    link(73, 0, 15, 2)
    link(74, 0, 15, 3)
    link(15, 0, 16, 0)
    link(65, 0, 16, 1)
    link(16, 0, 17, 0)
    link(10, 4, 17, 1)
    link(17, 0, 18, 0)
    link(30, 0, 31, 0)
    link(17, 0, 31, 1)
    link(31, 0, 32, 0)
    link(17, 0, 33, 0)
    link(32, 0, 33, 1)
    link(33, 0, 34, 0)
    link(34, 0, 35, 0)
    link(35, 0, 36, 0)
    link(36, 0, 37, 0)
    link(36, 0, 38, 0)
    link(38, 0, 39, 0)
    link(37, 2, 40, 0)
    link(37, 1, 41, 0)

    g.save(DIRECTIONS, "pixel-art-image-to-directions-2511", groups=[
        ("Qwen-Image-Edit-2511 + LoRAs", [420, -60, 380, 860], "#3f789e"),
        ("One view per loop", [800, -60, 1640, 760], "#a1309b"),
        ("Cut out + pixelize together", [2440, -60, 1160, 920], "#444"),
    ], scale=0.5, offset=(40, 80))


build_edit()
build_directions()
