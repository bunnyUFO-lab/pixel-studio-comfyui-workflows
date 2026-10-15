"""Generates workflows/pixel-art/sprite-inbetween.json.

Edit the constants below and re-run:  python tools/build_pixel_sprite_inbetween.py

Key frames (+ optional pose skeletons) -> the frames in between, on the sprite's own grid and palette: Wan 2.2 VACE
(Fun VACE A14B, HIGH + LOW noise with the 4-step lightx2v LoRAs) gets a control clip and its masks: the known frames
(start, end, kept or pose key frames) are upscaled sprite frames with mask 0, the frames to draw are grey (or an
OpenPose-style skeleton from a motion reference) with mask 1, and a part of a frame to redraw is the frame itself
with the masked pixels at 1. The first frame is also the reference image (who the character is). RepixelateFrames
brings every frame back onto the sprite's exact grid and palette and SaveSpriteSheet writes them.

Sprite Studio (op "inbetween") builds the clip itself: it uploads one picture per frame and replaces the two
"Control frame" / "Control mask" loaders by a batch of them, sets the size, length and the palette, and passes a
"Grid" picture (the sprite canvas with its corner pixels marked) so PrepareSpriteForVideo returns the exact grid.
"""
from comfy_workflow import (BLUE, GREEN, NEGATIVE, RED, TEXT_ENCODER, URLS, VAE, WORKFLOWS, YELLOW, Graph, inp, out)

OUT = WORKFLOWS / "pixel-art" / "sprite-inbetween.json"

WAN_HF = "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files"
VACE_HIGH = "wan2.2_fun_vace_high_noise_14B_fp8_scaled.safetensors"
VACE_LOW = "wan2.2_fun_vace_low_noise_14B_fp8_scaled.safetensors"
LIGHT_HIGH = "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors"
LIGHT_LOW = "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors"
MODEL_URLS = {VACE_HIGH: f"{WAN_HF}/diffusion_models/{VACE_HIGH}", VACE_LOW: f"{WAN_HF}/diffusion_models/{VACE_LOW}",
              LIGHT_HIGH: f"{WAN_HF}/loras/{LIGHT_HIGH}", LIGHT_LOW: f"{WAN_HF}/loras/{LIGHT_LOW}"}

PROMPT = ("Pixel art sprite animation of a game character. The character moves smoothly from the first pose to the last. "
          "Static camera, the character stays in place, flat solid background.")
NEG = NEGATIVE + ", camera movement, zoom, panning, background change, new objects, text"
SIZE, FRAMES, STEPS, SWITCH, SHIFT, SEED = 480, 17, 4, 2, 5.0, 7
VACE_STRENGTH = 1.0
# height_match, bg_tolerance, method, colors, palette, stability, deflicker, cleanup, outline
REPIX = [False, 18.0, "dominant", 0, "sprite", 0.6, True, True, "none"]
# filename_prefix, frame_step, max_frames, loop_trim, columns, spacing, trim, fps, tag, preview_scale
SHEET = ["pixel/sprite_inbetween", 1, 0, False, 0, 0, False, 12, "inbetween", 4]

NOTE = """Key frames -> in-betweens (Wan 2.2 VACE)

The control clip (blue) is one picture per video frame: known frames are the sprite frames upscaled on a flat
background, frames to draw are grey (or a pose skeleton). The control masks are white where the model draws and
black where it keeps the picture. The first frame also goes in as the reference image (keeps the character).
Grid (blue): the sprite canvas at its own pixel size; PrepareSpriteForVideo returns its grid for RepixelateFrames.
Sprite Studio builds the clip and masks (AI Animate in the pixel editor) and sets size, length and palette.
Models: Wan 2.2 Fun VACE A14B fp8 (HIGH + LOW noise, 17.4 GB each) + lightx2v 4-step T2V LoRAs (Comfy-Org repack)."""

g = Graph()
node, link = g.node, g.link


def model_props(type_, name, directory, url=None):
    return {"Node name for S&R": type_, "models": [{"name": name, "url": url or MODEL_URLS.get(name) or URLS[name], "directory": directory}]}


node(1, "UNETLoader", [0, 0], [420, 82], outputs=[out("MODEL", "MODEL")], widgets=[VACE_HIGH, "default"],
     title="Wan VACE HIGH-noise", props=model_props("UNETLoader", VACE_HIGH, "diffusion_models"))
node(2, "UNETLoader", [0, 130], [420, 82], outputs=[out("MODEL", "MODEL")], widgets=[VACE_LOW, "default"],
     title="Wan VACE LOW-noise", props=model_props("UNETLoader", VACE_LOW, "diffusion_models"))
node(3, "LoraLoaderModelOnly", [460, 0], [360, 82], inputs=[inp("model", "MODEL")], outputs=[out("MODEL", "MODEL")],
     widgets=[LIGHT_HIGH, 1.0], title="lightx2v 4-step (HIGH)", props=model_props("LoraLoaderModelOnly", LIGHT_HIGH, "loras"))
node(4, "LoraLoaderModelOnly", [460, 130], [360, 82], inputs=[inp("model", "MODEL")], outputs=[out("MODEL", "MODEL")],
     widgets=[LIGHT_LOW, 1.0], title="lightx2v 4-step (LOW)", props=model_props("LoraLoaderModelOnly", LIGHT_LOW, "loras"))
node(5, "ModelSamplingSD3", [860, 0], [240, 58], inputs=[inp("model", "MODEL")], outputs=[out("MODEL", "MODEL")], widgets=[SHIFT], title="Shift (HIGH)")
node(6, "ModelSamplingSD3", [860, 130], [240, 58], inputs=[inp("model", "MODEL")], outputs=[out("MODEL", "MODEL")], widgets=[SHIFT], title="Shift (LOW)")
node(7, "CLIPLoader", [0, 260], [420, 106], outputs=[out("CLIP", "CLIP")], widgets=[TEXT_ENCODER, "wan", "default"],
     title="Load UMT5 text encoder", props=model_props("CLIPLoader", TEXT_ENCODER, "text_encoders"))
node(8, "VAELoader", [0, 410], [420, 58], outputs=[out("VAE", "VAE")], widgets=[VAE], props=model_props("VAELoader", VAE, "vae"))

node(9, "CLIPTextEncode", [460, 260], [460, 180], inputs=[inp("clip", "CLIP")], outputs=[out("CONDITIONING", "CONDITIONING")],
     widgets=[PROMPT], title="Prompt", color=GREEN)
node(10, "CLIPTextEncode", [460, 480], [460, 160], inputs=[inp("clip", "CLIP")], outputs=[out("CONDITIONING", "CONDITIONING")],
     widgets=[NEG], title="Negative prompt", color=RED)

node(11, "LoadImage", [0, 520], [320, 320], outputs=[out("IMAGE", "IMAGE"), out("MASK", "MASK")], widgets=["example.png", "image"],
     title="Control frame", color=BLUE)
node(12, "LoadImage", [0, 880], [320, 320], outputs=[out("IMAGE", "IMAGE"), out("MASK", "MASK")], widgets=["example.png", "image"],
     title="Control mask", color=BLUE)
node(13, "ImageToMask", [360, 880], [240, 58], inputs=[inp("image", "IMAGE")], outputs=[out("MASK", "MASK")], widgets=["red"],
     title="Mask (white = draw)")
node(14, "LoadImage", [0, 1240], [320, 320], outputs=[out("IMAGE", "IMAGE"), out("MASK", "MASK")], widgets=["example.png", "image"],
     title="Reference", color=BLUE)
node(15, "LoadImage", [0, 1600], [320, 320], outputs=[out("IMAGE", "IMAGE"), out("MASK", "MASK")], widgets=["example.png", "image"],
     title="Grid", color=BLUE)
node(16, "JoinImageWithAlpha", [360, 1600], [240, 46], inputs=[inp("image", "IMAGE"), inp("alpha", "MASK")], outputs=[out("IMAGE", "IMAGE")],
     title="Keep grid transparency")
node(17, "PrepareSpriteForVideo", [640, 1600], [320, 130], inputs=[inp("sprite", "IMAGE")],
     outputs=[out("image", "IMAGE"), out("pixel_spec", "PIXEL_SPEC"), out("width", "INT"), out("height", "INT"), out("info", "STRING")],
     widgets=[SIZE, 0.0, "#00ff00"], title="Prepare grid", color=YELLOW)

# widgets: width, height, length, batch_size, strength
node(18, "WanVaceToVideo", [960, 260], [320, 260],
     inputs=[inp("positive", "CONDITIONING"), inp("negative", "CONDITIONING"), inp("vae", "VAE"),
             inp("control_video", "IMAGE"), inp("control_masks", "MASK"), inp("reference_image", "IMAGE")],
     outputs=[out("positive", "CONDITIONING"), out("negative", "CONDITIONING"), out("latent", "LATENT"), out("trim_latent", "INT")],
     widgets=[SIZE, SIZE, FRAMES, 1, VACE_STRENGTH], title="VACE (clip / length)", color=YELLOW)


def sampler_inputs():
    return [inp("model", "MODEL"), inp("positive", "CONDITIONING"), inp("negative", "CONDITIONING"), inp("latent_image", "LATENT")]


node(19, "KSamplerAdvanced", [1320, 0], [320, 334], inputs=sampler_inputs(), outputs=[out("LATENT", "LATENT")],
     widgets=["enable", SEED, "randomize", STEPS, 1, "euler", "simple", 0, SWITCH, "enable"], title="KSampler HIGH noise (seed here)")
node(20, "KSamplerAdvanced", [1320, 380], [320, 334], inputs=sampler_inputs(), outputs=[out("LATENT", "LATENT")],
     widgets=["disable", 0, "fixed", STEPS, 1, "euler", "simple", SWITCH, 10000, "disable"], title="KSampler LOW noise (refine)")
node(21, "TrimVideoLatent", [1320, 760], [240, 60], inputs=[inp("samples", "LATENT"), inp("trim_amount", "INT", "trim_amount")],
     outputs=[out("LATENT", "LATENT")], widgets=[0], title="Drop the reference frame")
node(22, "VAEDecode", [1320, 860], [210, 46], inputs=[inp("samples", "LATENT"), inp("vae", "VAE")], outputs=[out("IMAGE", "IMAGE")])
node(23, "VHS_VideoCombine", [1320, 960], [320, 420],
     inputs=[inp("images", "IMAGE"), inp("audio", "AUDIO"), inp("meta_batch", "VHS_BatchManager"), inp("vae", "VAE")],
     outputs=[out("Filenames", "VHS_FILENAMES")],
     widgets={"frame_rate": 16, "loop_count": 0, "filename_prefix": "pixel/sprite_inbetween_raw", "format": "video/h264-mp4",
              "pix_fmt": "yuv420p", "crf": 17, "save_metadata": True, "trim_to_audio": False, "pingpong": False, "save_output": True,
              "videopreview": {"hidden": False, "paused": False, "params": {}}},
     title="Raw video (motion check)", props={"Node name for S&R": "VHS_VideoCombine", "cnr_id": "comfyui-videohelpersuite"})
node(24, "RepixelateFrames", [1680, 0], [340, 250], inputs=[inp("frames", "IMAGE"), inp("pixel_spec", "PIXEL_SPEC")],
     outputs=[out("pixel_frames", "IMAGE"), out("info", "STRING"), out("palette", "STRING")], widgets=REPIX,
     title="Repixelate (same grid + palette)", color=YELLOW)
node(25, "SaveSpriteSheet", [1680, 300], [340, 330], inputs=[inp("frames", "IMAGE")],
     outputs=[out("sheet", "IMAGE"), out("frames", "IMAGE"), out("info", "STRING"), out("fps", "INT")], widgets=SHEET,
     title="Save sprite sheet", color=GREEN)
node(26, "PreviewAny", [2060, 0], [340, 160], inputs=[inp("source", "*")], title="Re-pixelate info")
node(27, "PreviewAny", [2060, 200], [340, 120], inputs=[inp("source", "*")], title="Grid info")
node(28, "Note", [1680, 700], [560, 300], widgets=[NOTE], title="READ ME", props={})

link(1, 0, 3, 0)
link(2, 0, 4, 0)
link(3, 0, 5, 0)
link(4, 0, 6, 0)
link(7, 0, 9, 0)
link(7, 0, 10, 0)
link(12, 0, 13, 0)
link(15, 0, 16, 0)
link(15, 1, 16, 1)
link(16, 0, 17, 0)
link(9, 0, 18, 0)
link(10, 0, 18, 1)
link(8, 0, 18, 2)
link(11, 0, 18, 3)
link(13, 0, 18, 4)
link(14, 0, 18, 5)
link(5, 0, 19, 0)
link(6, 0, 20, 0)
for ks in (19, 20):
    link(18, 0, ks, 1)
    link(18, 1, ks, 2)
link(18, 2, 19, 3)
link(19, 0, 20, 3)
link(20, 0, 21, 0)
link(18, 3, 21, 1)
link(21, 0, 22, 0)
link(8, 0, 22, 1)
link(22, 0, 23, 0)
link(22, 0, 24, 0)
link(17, 1, 24, 1)
link(24, 0, 25, 0)
link(24, 1, 26, 0)
link(17, 4, 27, 0)

g.save(OUT, "pixel-art-sprite-inbetween", groups=[
    ("Models", [-20, -60, 1140, 300], "#3f789e"),
    ("Control clip + prompts", [-20, 240, 1320, 1700], "#a1309b"),
    ("In-betweens (Wan 2.2 VACE)", [1300, -60, 360, 1460], "#88A"),
    ("Back to pixel art", [1660, -60, 760, 700], "#444"),
], scale=0.5, offset=(40, 80))
