"""Generates the FLUX 3 Image Klein versions of the Qwen-Image-2.1 sprite workflows (for the Klein vs Qwen test):

  workflows/pixel-art/sprite-edit-klein.json          (= sprite-edit)
  workflows/pixel-art/sprite-edit-refs-klein.json     (= sprite-edit-refs, up to 3 reference images)
  workflows/pixel-art/image-to-directions-klein.json  (= image-to-directions)

Edit the constants below and re-run:  python tools/build_pixel_klein.py

Same pixel steps as the Qwen versions (integer upscale in; Repixelate / Pixelize, Pixel Art Check out); only the
model part changes, wired like Comfy's own "FLUX 3 Image Klein distilled: image edit" template: Klein int8 (16 steps,
cfg 1, euler/simple), Qwen3-VL 4B text encoder, references through Reference Multi Latent (<ref_image_0> is the first
image), empty negative. Klein wants a full caption, not a one-line instruction, so the edit workflows run the short
request through BFL's prompt upsampler (Qwen3.5 9B, Text Generate) first; switch it off to send your own caption.
The direction views are written as full captions already, so they skip the upsampler.
Needs ComfyUI >= 0.41. Licence: FLUX Kommunity (commercial outputs only under US$5M revenue, with content review).
"""
from comfy_workflow import (BIREFNET, BIREFNET_URL, BLUE, GREEN, KLEIN_HF as HF, KLEIN_MODEL, KLEIN_TE,
                            KLEIN_TE_URL, KLEIN_UPSAMPLER, KLEIN_VAE, RED, WORKFLOWS, YELLOW, Graph, inp, out)

EDIT = WORKFLOWS / "pixel-art" / "sprite-edit-klein.json"
EDIT_REFS = WORKFLOWS / "pixel-art" / "sprite-edit-refs-klein.json"
DIRECTIONS = WORKFLOWS / "pixel-art" / "image-to-directions-klein.json"

STEPS, CFG = 16, 1.0
EDIT_PROMPT = "Give the character in the first image a red scarf. Keep everything else exactly the same."
TARGET, PADDING, EDIT_SEED = 1024, 0.25, 7
# height_match, bg_tolerance, method, colors (max new colours), palette, stability, deflicker, cleanup, outline
REPIX = [False, 18.0, "dominant", 6, "sprite+new", 0.0, False, True, "auto"]

# BFL's edit upsampler system prompt (from Comfy's Klein edit template), plus one line for pixel sprites.
UPSAMPLE_SYSTEM = (
    "You are the prompt upsampler of an image-editing model. You see the reference image(s), attached in order (the "
    "first image, the second image, ...), and the user's short request. Expand it into ONE detailed, self-contained "
    "instruction: the caption the editing model receives together with the same images. Output only the instruction "
    "as one continuous English prose block, with no greetings, explanations, Markdown, JSON, coordinates, ids or "
    "angle-bracket tokens. Read a vague or conversational request as the most plausible edit of the supplied images.\n\n"
    "Pick the voice by task shape. When one reference stays and gets bounded changes, write an imperative editing "
    "instruction on it (\"Remove the ...\", \"Change the ... to ...\", \"Zoom in on ...\"); a whole-image "
    "transformation opens with its scope (\"Convert the entire image into ...\", \"Apply a ... filter over the entire "
    "scene\", \"Horizontally flip the entire image so that ...\", \"Shift the camera viewpoint to ...\", then where the "
    "major elements should now be). When the references are drawn into a newly composed frame (combining subjects, a "
    "new scene, a garment, object or style donor), write a generation request (\"Create a photograph of ...\", "
    "\"Create an anime-style illustration of ...\", \"Generate a new image featuring ...\") that says what to take from "
    "each image, by ordinal, and what is newly generated.\n\n"
    "Keep every explicit edit, constraint, count, position, style cue and piece of wording, and decide what is left "
    "open as one concrete plan. Ground everything in what is actually visible: identify each referent unambiguously by "
    "its real appearance, never mention content that is not there, and use every image the user attached. State every "
    "end state concretely (which color, how large, which pose, which material) and give new content enough appearance "
    "to be generated. Cover the knock-on effects of the change (shadows, reflections, occlusion, perspective, scale, "
    "lighting) and say what fills a removed region. A zoom-in names what stays in frame and which neighbors are cropped "
    "out; an outpainting continues the scene in the same medium. Describe the change, not the whole scene, and close "
    "an editing instruction with one compact clause naming what stays unchanged; transferred people keep their "
    "identity, face and accessories unless asked otherwise. Transcribe on-image text verbatim in its original language "
    "inside double quotes, with line breaks as \\n, and match the original font, size and color when replacing it. "
    "Name recognizable public figures, brands and landmarks you can see. Directions are frame-relative; use no "
    "evaluative or quality words. Typical length is 50 to 150 words.\n\n"
    "The first image is an upscaled pixel-art game sprite: keep it pixel art with hard square pixels, flat colours from "
    "its palette, its outline, pose, framing and plain background.\n\n"
    "Output only the instruction."
)
# Text Generate: prompt, max_length, sampling_mode, temperature, top_k, top_p, min_p, repetition_penalty, seed,
# presence_penalty, thinking, use_default_template, mtp (the template's settings: top_k 1 = deterministic)
TEXTGEN = ["", 3584, "on", 1, 1, 0.95, 0, 1.05, 0, 0, False, True, "off"]

# Direction views: full captions (Klein's style), one line per view.
KEEP = ("Same character as <ref_image_0>: identical outfit, colours, proportions, props and art style. Full body, "
        "standing pose, centred, on a plain flat white background, no shadow, no text.")
VIEWS = [
    ("front", "Full-body game character turnaround, front view: the character from <ref_image_0> faces the camera "
              "directly."),
    ("front-right", "Full-body game character turnaround, three-quarter front view: the character from <ref_image_0> "
                    "is turned 45 degrees so it faces the right side of the image."),
    ("right", "Full-body game character turnaround, side profile view: the character from <ref_image_0> faces the "
              "right edge of the image."),
    ("back-right", "Full-body game character turnaround, three-quarter back view: the character from <ref_image_0> is "
                   "seen from behind, turned 45 degrees away from the camera toward the right side of the image."),
    ("back", "Full-body game character turnaround, back view: the character from <ref_image_0> is seen from directly "
             "behind, facing away from the camera."),
]
PROMPTS = "\n".join(f"{p} {KEEP}" for _, p in VIEWS)
NAMES = ", ".join(n for n, _ in VIEWS)
DIR_SEED = 1234
PIXELS, COLORS, COLOR_ERROR, MAX_COLORS = 64, 0, 20.0, 32
CUTOUT = [True, False, 0.02, 8.0, 0.2]
PIXELIZE = [PIXELS, "body height (per frame)", "pixeloe", 0.6, COLORS, COLOR_ERROR, "auto", True, "auto remove", 12.0,
            True, True, "auto", 1, 8]

LICENCE = "Licence: FLUX Kommunity v1.0 - commercial use of outputs only under US$5M revenue, with content review."


def models(name, url, directory, type_):
    return {"Node name for S&R": type_, "models": [{"name": name, "url": url, "directory": directory}]}


def loaders(g, x, y, upsampler=True):
    """Klein model (id 20), text encoder (22), VAE (23) and, for the edit workflows, the upsampler (46)."""
    g.node(20, "UNETLoader", [x, y], [340, 82], outputs=[out("MODEL", "MODEL")], widgets=[KLEIN_MODEL, "default"],
           title="FLUX 3 Image Klein (distilled)",
           props=models(KLEIN_MODEL, f"{HF}/diffusion_models/{KLEIN_MODEL}", "diffusion_models", "UNETLoader"))
    g.node(22, "CLIPLoader", [x, y + 120], [340, 106], outputs=[out("CLIP", "CLIP")], widgets=[KLEIN_TE, "flux3", "default"],
           title="Qwen3-VL 4B text encoder",
           props=models(KLEIN_TE, KLEIN_TE_URL, "text_encoders", "CLIPLoader"))
    g.node(23, "VAELoader", [x, y + 270], [340, 58], outputs=[out("VAE", "VAE")], widgets=[KLEIN_VAE],
           props=models(KLEIN_VAE, f"{HF}/vae/{KLEIN_VAE}", "vae", "VAELoader"))
    if upsampler:
        g.node(46, "CLIPLoader", [x, y + 370], [340, 106], outputs=[out("CLIP", "CLIP")],
               widgets=[KLEIN_UPSAMPLER, "flux3", "default"], title="Prompt upsampler (Qwen3.5 9B)",
               props=models(KLEIN_UPSAMPLER, f"{HF}/text_encoders/{KLEIN_UPSAMPLER}", "text_encoders", "CLIPLoader"))


def sampler(g, id_, x, y, seed, control):
    g.node(id_, "KSampler", [x, y], [320, 262],
           inputs=[inp("model", "MODEL"), inp("positive", "CONDITIONING"), inp("negative", "CONDITIONING"),
                   inp("latent_image", "LATENT")],
           outputs=[out("LATENT", "LATENT")], widgets=[seed, control, STEPS, CFG, "euler", "simple", 1.0],
           title="Klein sample")


def build_edit(refs=False):
    g = Graph()
    node, link = g.node, g.link
    n_refs = 3 if refs else 0
    note = f"""Sprite edit with FLUX 3 Image Klein (keeps the exact pixel grid)

1. Load a pixel sprite (blue); a batch of frames works too (each edited the same way).{'''
   References 1-3 (blue) are the second, third and fourth image.''' if refs else ''}
2. Edit request (green): a short instruction about the first image ("give him a red scarf"). The prompt
   upsampler (Qwen3.5 9B) turns it into the full caption Klein wants; "Upsampled prompt" shows it.
   Switch "Use prompt upsampler" off to send your own full caption instead (cite images as <ref_image_0>, ...).
3. Klein edits a x{TARGET // 64}-ish upscale ({STEPS} steps, cfg 1), then it is re-pixelated on the sprite's own grid:
   palette 'sprite+new' = the sprite's colours plus up to 'colors' new ones; 'sprite' = locked.
4. Restore: 'original canvas' = same size and position as the input.
Saved to output/pixel/sprite_edit_klein_*.png (+ _x8 preview).
{LICENCE}"""

    node(1, "LoadImage", [0, 0], [380, 400], outputs=[out("IMAGE", "IMAGE"), out("MASK", "MASK")],
         widgets=["example.png", "image"], title="Pixel sprite", color=BLUE)
    node(2, "JoinImageWithAlpha", [0, 440], [380, 46], inputs=[inp("image", "IMAGE"), inp("alpha", "MASK")],
         outputs=[out("IMAGE", "IMAGE")], title="Keep sprite transparency")
    node(3, "PrepareSpriteForVideo", [420, 0], [340, 130], inputs=[inp("sprite", "IMAGE")],
         outputs=[out("image", "IMAGE"), out("pixel_spec", "PIXEL_SPEC"), out("width", "INT"), out("height", "INT"),
                  out("info", "STRING")],
         widgets=[TARGET, PADDING, "auto"], title="Prepare sprite (integer upscale)")
    loaders(g, 420, 180)
    ref_ids = [40, 41, 42][:n_refs]
    for k, nid in enumerate(ref_ids):
        node(nid, "LoadImage", [0, 940 + 360 * k], [380, 320], outputs=[out("IMAGE", "IMAGE"), out("MASK", "MASK")],
             widgets=["example.png", "image"], title=f"Reference {k + 1}", color=BLUE)

    # Prompt: short request -> upsampler (optional) -> caption
    node(50, "PrimitiveStringMultiline", [800, 0], [440, 160], outputs=[out("STRING", "STRING")],
         widgets=[EDIT_PROMPT], title="Edit request", color=GREEN)
    node(48, "PrimitiveStringMultiline", [800, 200], [440, 160], outputs=[out("STRING", "STRING")],
         widgets=[UPSAMPLE_SYSTEM], title="Upsampler system prompt")
    node(53, "BatchImagesNode", [800, 400], [440, 46],
         inputs=[inp("images.image0", "IMAGE")] + [inp(f"images.image{k + 1}", "IMAGE") for k in range(n_refs)],
         outputs=[out("IMAGE", "IMAGE")], title="Images the upsampler sees")
    node(47, "TextGenerate", [800, 490], [440, 340],
         inputs=[inp("clip", "CLIP"), inp("image", "IMAGE"), inp("video", "IMAGE"), inp("audio", "AUDIO"),
                 inp("system_prompt", "STRING"), inp("prompt", "STRING", "prompt")],
         outputs=[out("generated_text", "STRING"), out("thinking", "STRING")], widgets=TEXTGEN,
         title="Prompt upsampler")
    node(51, "ComfySwitchNode", [1280, 0], [300, 82],
         inputs=[inp("on_false", "STRING"), inp("on_true", "STRING")], outputs=[out("output", "STRING")], widgets=[True],
         title="Use prompt upsampler")
    node(52, "PreviewAny", [1280, 120], [340, 200], inputs=[inp("source", "*")], title="Upsampled prompt")
    node(24, "CLIPTextEncode", [1280, 360], [340, 100], inputs=[inp("clip", "CLIP"), inp("text", "STRING", "text")],
         outputs=[out("CONDITIONING", "CONDITIONING")], widgets=[""], title="Encode caption")
    node(32, "ConditioningZeroOut", [1280, 500], [340, 26], inputs=[inp("conditioning", "CONDITIONING")],
         outputs=[out("CONDITIONING", "CONDITIONING")], title="Empty negative", color=RED)
    node(17, "ReferenceMultiLatent", [1280, 570], [340, 140],
         inputs=[inp("positive", "CONDITIONING"), inp("negative", "CONDITIONING"), inp("vae", "VAE"),
                 inp("images.image0", "IMAGE")] + [inp(f"images.image{k + 1}", "IMAGE") for k in range(n_refs)],
         outputs=[out("positive", "CONDITIONING"), out("negative", "CONDITIONING"), out("empty_latent", "LATENT")],
         widgets=["resize", 1], title="References (sprite = <ref_image_0>)")
    sampler(g, 25, 1660, 0, EDIT_SEED, "randomize")
    node(26, "VAEDecode", [1660, 300], [210, 46], inputs=[inp("samples", "LATENT"), inp("vae", "VAE")],
         outputs=[out("IMAGE", "IMAGE")])

    # Back to the same grid (as sprite-edit)
    node(27, "RepixelateFrames", [2020, 0], [340, 250], inputs=[inp("frames", "IMAGE"), inp("pixel_spec", "PIXEL_SPEC")],
         outputs=[out("pixel_frames", "IMAGE"), out("info", "STRING"), out("palette", "STRING")], widgets=REPIX,
         title="Repixelate (same grid)", color=YELLOW)
    node(28, "RestoreSpriteCanvas", [2020, 300], [340, 82], inputs=[inp("frames", "IMAGE"), inp("pixel_spec", "PIXEL_SPEC")],
         outputs=[out("frames", "IMAGE")], widgets=["original canvas"], title="Restore canvas", color=YELLOW)
    node(29, "SaveImage", [2400, 0], [360, 320], inputs=[inp("images", "IMAGE")], widgets=["pixel/sprite_edit_klein"],
         title="Save edited sprite (native size)", color=GREEN)
    node(30, "ImageScaleBy", [2400, 360], [300, 82], inputs=[inp("image", "IMAGE")], outputs=[out("IMAGE", "IMAGE")],
         widgets=["nearest-exact", 8.0], title="Zoom x8")
    node(31, "SaveImage", [2400, 480], [360, 400], inputs=[inp("images", "IMAGE")], widgets=["pixel/sprite_edit_klein_x8"],
         title="Save preview (x8)")
    node(33, "PixelArtCheck", [2020, 430], [340, 106], inputs=[inp("images", "IMAGE")],
         outputs=[out("report", "STRING"), out("passed", "BOOLEAN")], widgets=[64, 0, True], title="Pixel Art Check")
    node(34, "PreviewAny", [2020, 580], [340, 200], inputs=[inp("source", "*")], title="Check report")
    node(35, "PreviewAny", [2020, 820], [340, 140], inputs=[inp("source", "*")], title="Palette")
    node(36, "Note", [0, 520], [380, 400], widgets=[note], title="READ ME", props={})

    link(1, 0, 2, 0)
    link(1, 1, 2, 1)
    link(2, 0, 3, 0)
    link(3, 0, 53, 0)
    for k, nid in enumerate(ref_ids):
        link(nid, 0, 53, 1 + k)
    link(46, 0, 47, 0)
    link(53, 0, 47, 1)
    link(48, 0, 47, 4)
    link(50, 0, 47, 5)
    link(50, 0, 51, 0)           # switch off: the request as typed
    link(47, 0, 51, 1)           # switch on: the upsampled caption
    link(51, 0, 52, 0)
    link(22, 0, 24, 0)
    link(51, 0, 24, 1)
    link(24, 0, 32, 0)
    link(24, 0, 17, 0)
    link(32, 0, 17, 1)
    link(23, 0, 17, 2)
    link(3, 0, 17, 3)
    for k, nid in enumerate(ref_ids):
        link(nid, 0, 17, 4 + k)
    link(20, 0, 25, 0)
    link(17, 0, 25, 1)
    link(17, 1, 25, 2)
    link(17, 2, 25, 3)
    link(25, 0, 26, 0)
    link(23, 0, 26, 1)
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

    name = "pixel-art-sprite-edit-refs-klein" if refs else "pixel-art-sprite-edit-klein"
    g.save(EDIT_REFS if refs else EDIT, name, groups=[
        ("FLUX 3 Image Klein", [400, -60, 380, 940], "#3f789e"),
        ("Prompt (upsampler) + references", [780, -60, 860, 900], "#a1309b"),
        ("Back to the same grid", [2000, -60, 780, 1040], "#444"),
    ], scale=0.5, offset=(40, 80),
        app=([(1, "image"), (50, "value"), (51, "switch"), (25, "seed"), (27, "palette"), (27, "colors"), (28, "mode")],
             [29, 31, 34]))


def build_directions():
    g = Graph()
    node, link = g.node, g.link
    note = f"""Character image -> 8-direction pixel sprites, with FLUX 3 Image Klein

1. Load a character image (blue), ideally full body on a plain background (= <ref_image_0>).
2. "View prompts" (green): ONE FULL CAPTION PER VIEW, in the order of "names" on the Save node:
   {NAMES}
   Klein wants a full description, not an instruction, so each line says what the picture is.
3. Klein draws each view ({STEPS} steps, cfg 1).
4. All views are pixelized together: body {PIXELS} art px tall, feet on the bottom line, one shared palette.
5. Save directions: left-side views are the mirrored right-side ones -> 8 directions.
Saved to output/pixel/: directions_klein_*_<direction>.png, sheet, .json; raw views in pixel/directions_klein_raw_*.png.
{LICENCE}"""

    node(1, "LoadImage", [0, 0], [400, 440], outputs=[out("IMAGE", "IMAGE"), out("MASK", "MASK")],
         widgets=["example.png", "image"], title="Character image", color=BLUE)
    loaders(g, 440, 0, upsampler=False)
    node(10, "StartLoop", [820, 0], [300, 160],
         inputs=[inp("parent_iteration", "INT"), inp("initial_iteration_value", "*"),
                 inp("mode.num_iterations", "INT", "mode.num_iterations")],
         outputs=[out("iteration_index", "INT"), out("is_first", "BOOLEAN"), out("is_last", "BOOLEAN"),
                  out("list_item", "*"), out("current_iteration_value", "*")],
         widgets=["simple", len(VIEWS), False], title="Views (num_iterations = number of prompt lines)", color=YELLOW)
    node(11, "SegmentPrompt", [820, 200], [460, 300], inputs=[inp("index", "INT")],
         outputs=[out("prompt", "STRING"), out("segment_seed", "INT")], widgets=[PROMPTS, DIR_SEED, "fixed"],
         title="View prompts (one line per view)", color=GREEN)
    node(12, "CLIPTextEncode", [1320, 0], [340, 100], inputs=[inp("clip", "CLIP"), inp("text", "STRING", "text")],
         outputs=[out("CONDITIONING", "CONDITIONING")], widgets=[""], title="Encode view")
    node(13, "ConditioningZeroOut", [1320, 140], [340, 26], inputs=[inp("conditioning", "CONDITIONING")],
         outputs=[out("CONDITIONING", "CONDITIONING")], title="Empty negative", color=RED)
    node(14, "ReferenceMultiLatent", [1320, 210], [340, 140],
         inputs=[inp("positive", "CONDITIONING"), inp("negative", "CONDITIONING"), inp("vae", "VAE"),
                 inp("images.image0", "IMAGE")],
         outputs=[out("positive", "CONDITIONING"), out("negative", "CONDITIONING"), out("empty_latent", "LATENT")],
         widgets=["resize", 1], title="Reference (<ref_image_0>)")
    sampler(g, 15, 1320, 390, DIR_SEED, "fixed")
    node(16, "VAEDecode", [1320, 690], [210, 46], inputs=[inp("samples", "LATENT"), inp("vae", "VAE")],
         outputs=[out("IMAGE", "IMAGE")])
    node(17, "EndLoop", [1760, 0], [300, 140],
         inputs=[inp("output_value", "*"), inp("next_iteration_value", "*"), inp("terminations.termination0", "*")],
         outputs=[out("outputs", "IMAGE")], widgets=[True], title="End loop (collect views)")

    node(18, "SaveImage", [1760, 200], [320, 300], inputs=[inp("images", "IMAGE")], widgets=["pixel/directions_klein_raw"],
         title="Save generated views")
    node(30, "LoadBackgroundRemovalModel", [2100, 0], [320, 58], outputs=[out("bg_model", "BACKGROUND_REMOVAL")],
         widgets=[BIREFNET], title="BiRefNet", props=models(BIREFNET, BIREFNET_URL, "background_removal", "LoadBackgroundRemovalModel"))
    node(31, "RemoveBackground", [2100, 100], [320, 46], inputs=[inp("bg_removal_model", "BACKGROUND_REMOVAL"), inp("image", "IMAGE")],
         outputs=[out("mask", "MASK")])
    node(32, "InvertMask", [2100, 180], [320, 26], inputs=[inp("mask", "MASK")], outputs=[out("MASK", "MASK")])
    node(33, "JoinImageWithAlpha", [2100, 250], [320, 46], inputs=[inp("image", "IMAGE"), inp("alpha", "MASK")],
         outputs=[out("IMAGE", "IMAGE")], title="Remove background")
    node(34, "CleanCutout", [2100, 430], [320, 160], inputs=[inp("images", "IMAGE")],
         outputs=[out("image", "IMAGE"), out("removed", "MASK"), out("info", "STRING")], widgets=CUTOUT,
         title="Clean cutout (drop shadow, background pockets)", color=YELLOW)
    node(35, "RebatchImages", [2100, 340], [320, 58], inputs=[inp("images", "IMAGE")], outputs=[out("IMAGE", "IMAGE")],
         widgets=[64], title="Views -> one batch")
    node(36, "Pixelize", [2460, 0], [360, 440], inputs=[inp("images", "IMAGE")],
         outputs=[out("pixel_art", "IMAGE"), out("preview", "IMAGE"), out("palette", "STRING"), out("info", "STRING")],
         widgets=PIXELIZE, title="Pixelize (shared grid + palette)", color=YELLOW)
    node(37, "SaveDirections", [2860, 0], [360, 200], inputs=[inp("images", "IMAGE")],
         outputs=[out("sheet", "IMAGE"), out("directions", "IMAGE"), out("info", "STRING")],
         widgets=["pixel/directions_klein", NAMES, True, 2, 6], title="Save directions (mirror -> 8)", color=GREEN)
    node(38, "PixelArtCheck", [2460, 490], [360, 106], inputs=[inp("images", "IMAGE")],
         outputs=[out("report", "STRING"), out("passed", "BOOLEAN")], widgets=[MAX_COLORS, PIXELS, True], title="Pixel Art Check")
    node(39, "PreviewAny", [2460, 640], [360, 200], inputs=[inp("source", "*")], title="Check report")
    node(40, "PreviewAny", [2860, 250], [360, 160], inputs=[inp("source", "*")], title="Directions info")
    node(41, "SpriteAnimationPreview", [3260, 0], [520, 680], inputs=[inp("frames", "IMAGE"), inp("fps", "INT", "fps")],
         widgets=[3, 6, "checker", "loop"], title="Turnaround preview (all directions)", color=GREEN)
    node(42, "Note", [0, 490], [460, 560], widgets=[note], title="READ ME", props={})

    link(1, 0, 10, 1)            # carried value: the input image (unchanged)
    link(10, 0, 11, 0)
    link(22, 0, 12, 0)
    link(11, 0, 12, 1)
    link(12, 0, 13, 0)
    link(12, 0, 14, 0)
    link(13, 0, 14, 1)
    link(23, 0, 14, 2)
    link(1, 0, 14, 3)
    link(20, 0, 15, 0)
    link(14, 0, 15, 1)
    link(14, 1, 15, 2)
    link(14, 2, 15, 3)
    link(15, 0, 16, 0)
    link(23, 0, 16, 1)
    link(16, 0, 17, 0)           # output_value: this view, accumulated
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
    link(37, 1, 41, 0)           # 8 directions -> turnaround playback

    g.save(DIRECTIONS, "pixel-art-image-to-directions-klein", groups=[
        ("FLUX 3 Image Klein", [420, -60, 380, 400], "#3f789e"),
        ("One view per loop", [800, -60, 1280, 820], "#a1309b"),
        ("Cut out + pixelize together", [2080, -60, 1160, 920], "#444"),
    ], scale=0.5, offset=(40, 80))


build_edit()
build_edit(refs=True)
build_directions()
