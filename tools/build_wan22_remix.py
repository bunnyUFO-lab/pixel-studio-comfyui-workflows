"""Generates workflows/image-to-video/wan22-remix.json.

Edit the constants below and re-run:  python tools/build_wan22_remix.py
"""
from comfy_workflow import (GREEN, HIGH, LOW, NEGATIVE, RED, TEXT_ENCODER, URLS, VAE, WORKFLOWS, YELLOW,
                            Graph, inp, out)

OUT = WORKFLOWS / "image-to-video" / "wan22-remix.json"

WIDTH, HEIGHT, FRAMES, FPS = 720, 1280, 81, 16
TOTAL_STEPS, SWITCH_STEP, SHIFT = 8, 4, 8

POSITIVE = (
    "The woman slowly turns toward the camera and smiles, her hair moving gently in the breeze, "
    "natural body motion, soft cinematic lighting, smooth handheld camera push-in."
)
NOTE = f"""Wan 2.2 Remix v3.0 - Image to Video (RTX 4090 / 24 GB)

1. Load your start image (yellow node).
2. Set width/height to the SAME aspect ratio as the image (multiples of 16).
   The image is scaled + center-cropped to that size.
   Portrait 720x1280 / 480x832, landscape 1280x720 / 832x480, square 960x960 / 640x640.
3. Describe the MOTION and camera in the positive prompt, not what is already in the image.
4. Length is in frames (4n+1). {FRAMES} frames = {FRAMES // FPS} s at {FPS} fps. Keep <= 121.

Lightning (4-step distill) is baked into Remix: keep CFG 1, euler / simple,
{TOTAL_STEPS} total steps, switch HIGH->LOW at {SWITCH_STEP}. Do NOT add the lightx2v LoRAs on top.
Faster drafts: 6 steps / switch 3, or 480p resolution.

Seed is on the HIGH-noise sampler.
Optional LoRAs: select the node, Ctrl+B to un-bypass, pick a HIGH and LOW file."""

g = Graph()
node, link = g.node, g.link


def model_props(type_, name, directory):
    return {"Node name for S&R": type_, "models": [{"name": name, "url": URLS[name], "directory": directory}]}


# Models
node(1, "UNETLoader", [0, 0], [420, 82], outputs=[out("MODEL", "MODEL")],
     widgets=[HIGH, "default"], title="Load Remix HIGH-noise model",
     props=model_props("UNETLoader", HIGH, "diffusion_models"))
node(2, "UNETLoader", [0, 140], [420, 82], outputs=[out("MODEL", "MODEL")],
     widgets=[LOW, "default"], title="Load Remix LOW-noise model",
     props=model_props("UNETLoader", LOW, "diffusion_models"))
node(3, "LoraLoaderModelOnly", [460, 0], [340, 82], inputs=[inp("model", "MODEL")], outputs=[out("MODEL", "MODEL")],
     widgets=["your_lora_HIGH.safetensors", 1.0], title="Optional LoRA (HIGH) - bypassed", mode=4)
node(4, "LoraLoaderModelOnly", [460, 140], [340, 82], inputs=[inp("model", "MODEL")], outputs=[out("MODEL", "MODEL")],
     widgets=["your_lora_LOW.safetensors", 1.0], title="Optional LoRA (LOW) - bypassed", mode=4)
node(5, "ModelSamplingSD3", [840, 0], [260, 58], inputs=[inp("model", "MODEL")], outputs=[out("MODEL", "MODEL")],
     widgets=[SHIFT], title="Shift (HIGH)")
node(6, "ModelSamplingSD3", [840, 140], [260, 58], inputs=[inp("model", "MODEL")], outputs=[out("MODEL", "MODEL")],
     widgets=[SHIFT], title="Shift (LOW)")
node(7, "CLIPLoader", [0, 300], [420, 106], outputs=[out("CLIP", "CLIP")],
     widgets=[TEXT_ENCODER, "wan", "default"], title="Load UMT5 text encoder",
     props=model_props("CLIPLoader", TEXT_ENCODER, "text_encoders"))
node(8, "VAELoader", [0, 460], [420, 58], outputs=[out("VAE", "VAE")], widgets=[VAE],
     props=model_props("VAELoader", VAE, "vae"))

# Inputs
node(9, "LoadImage", [0, 620], [420, 420], outputs=[out("IMAGE", "IMAGE"), out("MASK", "MASK")],
     widgets=["example.png", "image"], title="Start image", color=YELLOW)
node(10, "CLIPTextEncode", [460, 300], [460, 200], inputs=[inp("clip", "CLIP")],
     outputs=[out("CONDITIONING", "CONDITIONING")], widgets=[POSITIVE],
     title="Positive prompt (describe the MOTION)", color=GREEN)
node(11, "CLIPTextEncode", [460, 540], [460, 180], inputs=[inp("clip", "CLIP")],
     outputs=[out("CONDITIONING", "CONDITIONING")], widgets=[NEGATIVE], title="Negative prompt", color=RED)
node(12, "WanImageToVideo", [960, 300], [320, 210],
     inputs=[inp("positive", "CONDITIONING"), inp("negative", "CONDITIONING"), inp("vae", "VAE"),
             inp("clip_vision_output", "CLIP_VISION_OUTPUT"), inp("start_image", "IMAGE")],
     outputs=[out("positive", "CONDITIONING"), out("negative", "CONDITIONING"), out("latent", "LATENT")],
     widgets=[WIDTH, HEIGHT, FRAMES, 1], title="Width / height / frames", color=YELLOW)
node(13, "PrimitiveNode", [960, 620], [260, 82], outputs=[out("INT", "INT", "steps")],
     widgets=[TOTAL_STEPS, "fixed"], title="Total steps", props={"Run widget replace on values": False}, color=YELLOW)
node(14, "PrimitiveNode", [960, 740], [260, 82], outputs=[out("INT", "INT", "end_at_step")],
     widgets=[SWITCH_STEP, "fixed"], title="Switch HIGH -> LOW at step",
     props={"Run widget replace on values": False}, color=YELLOW)


# Sampling + output
def sampler_inputs():
    return [inp("model", "MODEL"), inp("positive", "CONDITIONING"), inp("negative", "CONDITIONING"),
            inp("latent_image", "LATENT"), inp("steps", "INT", "steps"),
            inp("start_at_step", "INT", "start_at_step"), inp("end_at_step", "INT", "end_at_step")]


# widgets: add_noise, seed, control_after_generate, steps, cfg, sampler, scheduler, start, end, return_leftover_noise
node(15, "KSamplerAdvanced", [1320, 0], [320, 334], inputs=sampler_inputs(), outputs=[out("LATENT", "LATENT")],
     widgets=["enable", 123456789, "randomize", TOTAL_STEPS, 1, "euler", "simple", 0, SWITCH_STEP, "enable"],
     title="KSampler HIGH noise (seed here)")
node(16, "KSamplerAdvanced", [1320, 380], [320, 334], inputs=sampler_inputs(), outputs=[out("LATENT", "LATENT")],
     widgets=["disable", 0, "fixed", TOTAL_STEPS, 1, "euler", "simple", SWITCH_STEP, 10000, "disable"],
     title="KSampler LOW noise (refine)")
node(17, "VAEDecode", [1680, 0], [210, 46], inputs=[inp("samples", "LATENT"), inp("vae", "VAE")],
     outputs=[out("IMAGE", "IMAGE")])
node(18, "VHS_VideoCombine", [1680, 90], [400, 640],
     inputs=[inp("images", "IMAGE"), inp("audio", "AUDIO"), inp("meta_batch", "VHS_BatchManager"), inp("vae", "VAE")],
     outputs=[out("Filenames", "VHS_FILENAMES")],
     widgets={"frame_rate": FPS, "loop_count": 0, "filename_prefix": "wan22_remix_i2v/%date:yyyyMMdd_hhmmss%",
              "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 17, "save_metadata": True,
              "trim_to_audio": False, "pingpong": False, "save_output": True,
              "videopreview": {"hidden": False, "paused": False, "params": {}}},
     title=f"Save MP4 ({FPS} fps)",
     props={"Node name for S&R": "VHS_VideoCombine", "cnr_id": "comfyui-videohelpersuite"})
node(19, "Note", [1320, 780], [560, 360], widgets=[NOTE], title="READ ME", props={})

link(1, 0, 3, 0)
link(2, 0, 4, 0)
link(3, 0, 5, 0)
link(4, 0, 6, 0)
link(7, 0, 10, 0)
link(7, 0, 11, 0)
link(10, 0, 12, 0)
link(11, 0, 12, 1)
link(8, 0, 12, 2)
link(9, 0, 12, 4)
link(5, 0, 15, 0)
link(6, 0, 16, 0)
for ks in (15, 16):
    link(12, 0, ks, 1)
    link(12, 1, ks, 2)
    link(13, 0, ks, 4)
link(12, 2, 15, 3)
link(15, 0, 16, 3)
link(14, 0, 15, 6)
link(14, 0, 16, 5)
link(16, 0, 17, 0)
link(8, 0, 17, 1)
link(17, 0, 18, 0)

g.save(OUT, "wan22-remix", groups=[
    ("Models", [-20, -60, 1140, 300], "#3f789e"),
    ("Encoders + inputs", [-20, 260, 1300, 800], "#a1309b"),
    ("Sampling + output", [1300, -60, 800, 1220], "#88A"),
])
