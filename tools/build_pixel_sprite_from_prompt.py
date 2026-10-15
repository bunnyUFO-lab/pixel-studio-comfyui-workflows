"""Generates workflows/pixel-art/sprite-from-prompt.json.

Edit the constants below and re-run:  python tools/build_pixel_sprite_from_prompt.py

Pixel sprite + motion prompt -> animated sprite sheet at the sprite's own scale and palette:
PrepareSpriteForVideo upscales the sprite by an integer factor onto a flat background, Wan 2.2 Remix (first = last
frame, so the animation loops) animates it with the pixel-art animation LoRA, then RepixelateFrames brings every
frame back onto exactly the same grid with the sprite's exact palette and SaveSpriteSheet writes the sheet,
Aseprite JSON and an animated WebP.
"""
from comfy_workflow import (BLUE, GREEN, HIGH, LOW, NEGATIVE, PIXEL_ANIM_LORA, PIXEL_ANIM_LORA_URL, RED,
                            TEXT_ENCODER, URLS, VAE, WORKFLOWS, YELLOW, Graph, inp, out)

OUT = WORKFLOWS / "pixel-art" / "sprite-from-prompt.json"

PROMPT = ("Pixel art sprite animation. The character does a looping idle animation: breathing, a slight up and down "
          "bob, cape and hair swaying gently. Static camera, the character stays in place, flat solid background.")
NEG = NEGATIVE + ", camera movement, zoom, panning, background change, new objects, text"
TARGET_SIZE, PADDING = 640, 0.35       # video canvas (longest side), free space around the sprite for motion
FRAMES, FPS = 49, 16                   # 4n+1
TOTAL_STEPS, SWITCH_STEP, SHIFT = 8, 4, 5
LORA_STRENGTH = 1.0
SEED = 7
# frames, filename_prefix, frame_step, max_frames, loop_trim, columns, spacing, trim, fps, tag, preview_scale
SHEET = ["pixel/sprite_from_prompt", 2, 0, True, 0, 0, True, 8, "idle", 4]

# height_match, bg_tolerance, method, colors (0 = sprite palette), palette, stability, deflicker, cleanup, outline
REPIX = [False, 18.0, "dominant", 0, "sprite", 0.7, True, True, "none"]

NOTE = f"""Pixel sprite + prompt -> animated sprite sheet

1. Load your sprite (blue): native pixel art on a transparent or plain background (an upscaled copy is fine:
   its grid is detected). Output art pixels, size and palette stay exactly the sprite's.
2. Prompt (green): describe the MOTION ("walks in place", "swings the sword", "idle breathing").
   Keep "static camera" and "flat solid background".
3. The first and last frame are both your sprite, so the animation loops. For a one-shot action (attack, death)
   disconnect "end_image" on the yellow "Loop" node.
4. Settings: frames ({FRAMES} = ~{FRAMES // FPS} s at {FPS} fps), steps {TOTAL_STEPS} (switch at {SWITCH_STEP}), padding = room for motion.
   "Pixel-animate LoRA" (HIGH + LOW) is trained on pixel-art sprite animations; Ctrl+B both to compare without.
Colours: Re-pixelate 'colors' merges the sprite palette down to N of its own colours (fewer = steadier),
   'palette' swaps in a preset/hex list; 'stability' + 'deflicker' stop colours popping between frames.
5. Sprite sheet: frame_step {SHEET[1]} keeps every 2nd frame; loop_trim drops repeated end frames; fps for the WebP/JSON.
The "Animation preview" node plays the sheet frames at the sheet's fps (zoom, background, ping-pong).
Saved to output/pixel/: sprite_from_prompt_*.png (sheet), .json (Aseprite), .webp (animated), _preview.png.
The raw video is saved too (pixel/sprite_from_prompt_raw) to check the motion."""

g = Graph()
node, link = g.node, g.link


def model_props(type_, name, directory, url=None):
    return {"Node name for S&R": type_, "models": [{"name": name, "url": url or URLS[name], "directory": directory}]}


ATTACK_HF = "https://huggingface.co/Teluv/Wan_Video_Attack_Sprite_Animation/resolve/main/"
# Optional alternative LoRA (Apache-2.0) the Sprite Studio app swaps in for attacks with slash effects; listed so
# docker/models.py downloads it with this workflow (ComfyUI loads only the one the node names).
ATTACK = {"HIGH": ("wan_attack_sprite_v3_high_noise.safetensors", ATTACK_HF + "Wan_i14b_Attack_Sprite_Animation_ver3_high_noise.safetensors"),
          "LOW": ("wan_attack_sprite_v3_low_noise.safetensors", ATTACK_HF + "Wan_i14b_Attack_Sprite_Animation_ver3_low_noise.safetensors")}


def lora_props(which):
    pr = model_props("LoraLoaderModelOnly", PIXEL_ANIM_LORA, "loras", PIXEL_ANIM_LORA_URL)
    pr["models"].append({"name": ATTACK[which][0], "url": ATTACK[which][1], "directory": "loras"})
    return pr


# Models
node(1, "UNETLoader", [0, 0], [420, 82], outputs=[out("MODEL", "MODEL")], widgets=[HIGH, "default"],
     title="Load Remix HIGH-noise model", props=model_props("UNETLoader", HIGH, "diffusion_models"))
node(2, "UNETLoader", [0, 140], [420, 82], outputs=[out("MODEL", "MODEL")], widgets=[LOW, "default"],
     title="Load Remix LOW-noise model", props=model_props("UNETLoader", LOW, "diffusion_models"))
node(3, "LoraLoaderModelOnly", [460, 0], [340, 82], inputs=[inp("model", "MODEL")], outputs=[out("MODEL", "MODEL")],
     widgets=[PIXEL_ANIM_LORA, LORA_STRENGTH], title="Pixel-animate LoRA (HIGH)", props=lora_props("HIGH"))
node(4, "LoraLoaderModelOnly", [460, 140], [340, 82], inputs=[inp("model", "MODEL")], outputs=[out("MODEL", "MODEL")],
     widgets=[PIXEL_ANIM_LORA, LORA_STRENGTH], title="Pixel-animate LoRA (LOW)", props=lora_props("LOW"))
node(5, "ModelSamplingSD3", [840, 0], [260, 58], inputs=[inp("model", "MODEL")], outputs=[out("MODEL", "MODEL")],
     widgets=[SHIFT], title="Shift (HIGH)")
node(6, "ModelSamplingSD3", [840, 140], [260, 58], inputs=[inp("model", "MODEL")], outputs=[out("MODEL", "MODEL")],
     widgets=[SHIFT], title="Shift (LOW)")
node(7, "CLIPLoader", [0, 300], [420, 106], outputs=[out("CLIP", "CLIP")], widgets=[TEXT_ENCODER, "wan", "default"],
     title="Load UMT5 text encoder", props=model_props("CLIPLoader", TEXT_ENCODER, "text_encoders"))
node(8, "VAELoader", [0, 460], [420, 58], outputs=[out("VAE", "VAE")], widgets=[VAE],
     props=model_props("VAELoader", VAE, "vae"))

# Inputs
node(9, "LoadImage", [0, 600], [420, 420], outputs=[out("IMAGE", "IMAGE"), out("MASK", "MASK")],
     widgets=["example.png", "image"], title="Pixel sprite", color=BLUE)
node(10, "PrepareSpriteForVideo", [460, 600], [340, 130], inputs=[inp("sprite", "IMAGE")],
     outputs=[out("image", "IMAGE"), out("pixel_spec", "PIXEL_SPEC"), out("width", "INT"), out("height", "INT"),
              out("info", "STRING")],
     widgets=[TARGET_SIZE, PADDING, "auto"], title="Prepare sprite (integer upscale)", color=YELLOW)
node(11, "CLIPTextEncode", [460, 260], [460, 200], inputs=[inp("clip", "CLIP")],
     outputs=[out("CONDITIONING", "CONDITIONING")], widgets=[PROMPT], title="Motion prompt", color=GREEN)
node(12, "CLIPTextEncode", [460, 780], [340, 180], inputs=[inp("clip", "CLIP")],
     outputs=[out("CONDITIONING", "CONDITIONING")], widgets=[NEG], title="Negative prompt", color=RED)
node(13, "WanFirstLastFrameToVideo", [960, 260], [320, 230],
     inputs=[inp("positive", "CONDITIONING"), inp("negative", "CONDITIONING"), inp("vae", "VAE"),
             inp("clip_vision_start_image", "CLIP_VISION_OUTPUT"), inp("clip_vision_end_image", "CLIP_VISION_OUTPUT"),
             inp("start_image", "IMAGE"), inp("end_image", "IMAGE"),
             inp("width", "INT", "width"), inp("height", "INT", "height")],
     outputs=[out("positive", "CONDITIONING"), out("negative", "CONDITIONING"), out("latent", "LATENT")],
     widgets=[TARGET_SIZE, TARGET_SIZE, FRAMES, 1], title="Loop (first = last frame) / frames", color=YELLOW)
node(25, "JoinImageWithAlpha", [460, 540], [340, 46], inputs=[inp("image", "IMAGE"), inp("alpha", "MASK")],
     outputs=[out("IMAGE", "IMAGE")], title="Keep sprite transparency")
node(14, "PreviewAny", [460, 1000], [340, 120], inputs=[inp("source", "*")], title="Sprite info")


def sampler_inputs():
    return [inp("model", "MODEL"), inp("positive", "CONDITIONING"), inp("negative", "CONDITIONING"),
            inp("latent_image", "LATENT")]


# widgets: add_noise, seed, control_after_generate, steps, cfg, sampler, scheduler, start, end, return_leftover_noise
node(15, "KSamplerAdvanced", [1320, 0], [320, 334], inputs=sampler_inputs(), outputs=[out("LATENT", "LATENT")],
     widgets=["enable", SEED, "randomize", TOTAL_STEPS, 1, "euler", "simple", 0, SWITCH_STEP, "enable"],
     title="KSampler HIGH noise (seed here)")
node(16, "KSamplerAdvanced", [1320, 380], [320, 334], inputs=sampler_inputs(), outputs=[out("LATENT", "LATENT")],
     widgets=["disable", 0, "fixed", TOTAL_STEPS, 1, "euler", "simple", SWITCH_STEP, 10000, "disable"],
     title="KSampler LOW noise (refine)")
node(17, "VAEDecode", [1320, 760], [210, 46], inputs=[inp("samples", "LATENT"), inp("vae", "VAE")],
     outputs=[out("IMAGE", "IMAGE")])
node(18, "VHS_VideoCombine", [1320, 850], [320, 420],
     inputs=[inp("images", "IMAGE"), inp("audio", "AUDIO"), inp("meta_batch", "VHS_BatchManager"), inp("vae", "VAE")],
     outputs=[out("Filenames", "VHS_FILENAMES")],
     widgets={"frame_rate": FPS, "loop_count": 0, "filename_prefix": "pixel/sprite_from_prompt_raw",
              "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 17, "save_metadata": True,
              "trim_to_audio": False, "pingpong": False, "save_output": True,
              "videopreview": {"hidden": False, "paused": False, "params": {}}},
     title="Raw video (motion check)", props={"Node name for S&R": "VHS_VideoCombine", "cnr_id": "comfyui-videohelpersuite"})

# Back to pixel art
node(19, "RepixelateFrames", [1680, 0], [340, 250], inputs=[inp("frames", "IMAGE"), inp("pixel_spec", "PIXEL_SPEC")],
     outputs=[out("pixel_frames", "IMAGE"), out("info", "STRING"), out("palette", "STRING")], widgets=REPIX,
     title="Repixelate (same grid + palette)", color=YELLOW)
node(20, "SaveSpriteSheet", [1680, 180], [340, 330], inputs=[inp("frames", "IMAGE")],
     outputs=[out("sheet", "IMAGE"), out("frames", "IMAGE"), out("info", "STRING"), out("fps", "INT")], widgets=SHEET,
     title="Save sprite sheet", color=GREEN)
node(21, "PixelArtCheck", [2060, 0], [340, 106], inputs=[inp("images", "IMAGE")],
     outputs=[out("report", "STRING"), out("passed", "BOOLEAN")], widgets=[256, 0, True], title="Pixel Art Check")
node(22, "PreviewAny", [2060, 150], [340, 200], inputs=[inp("source", "*")], title="Check report")
node(23, "PreviewAny", [2060, 390], [340, 160], inputs=[inp("source", "*")], title="Sheet info")
node(27, "SpriteAnimationPreview", [2440, 0], [520, 680], inputs=[inp("frames", "IMAGE"), inp("fps", "INT", "fps")],
     widgets=[8, 6, "checker", "loop"], title="Animation preview (playback)", color=GREEN)
node(24, "Note", [1680, 560], [560, 520], widgets=[NOTE], title="READ ME", props={})

node(26, "PreviewAny", [2060, 600], [340, 160], inputs=[inp("source", "*")], title="Re-pixelate info (colours, popping)")

link(1, 0, 3, 0)
link(2, 0, 4, 0)
link(3, 0, 5, 0)
link(4, 0, 6, 0)
link(7, 0, 11, 0)
link(7, 0, 12, 0)
link(9, 0, 25, 0)            # LoadImage drops alpha into MASK: join it back
link(9, 1, 25, 1)
link(25, 0, 10, 0)
link(10, 4, 14, 0)
link(11, 0, 13, 0)
link(12, 0, 13, 1)
link(8, 0, 13, 2)
link(10, 0, 13, 5)
link(10, 0, 13, 6)
link(10, 2, 13, 7)
link(10, 3, 13, 8)
link(5, 0, 15, 0)
link(6, 0, 16, 0)
for ks in (15, 16):
    link(13, 0, ks, 1)
    link(13, 1, ks, 2)
link(13, 2, 15, 3)
link(15, 0, 16, 3)
link(16, 0, 17, 0)
link(8, 0, 17, 1)
link(17, 0, 18, 0)
link(17, 0, 19, 0)
link(10, 1, 19, 1)
link(19, 0, 20, 0)
link(20, 1, 21, 0)
link(21, 0, 22, 0)
link(20, 2, 23, 0)

link(19, 1, 26, 0)
link(20, 1, 27, 0)           # the sheet's frames + fps -> playback
link(20, 3, 27, 1)

g.save(OUT, "pixel-art-sprite-from-prompt", groups=[
    ("Models", [-20, -60, 1140, 300], "#3f789e"),
    ("Sprite + prompts", [-20, 240, 1320, 900], "#a1309b"),
    ("Animate (Wan 2.2 Remix)", [1300, -60, 360, 1360], "#88A"),
    ("Back to pixel art", [1660, -60, 760, 600], "#444"),
], scale=0.55, offset=(40, 80))
