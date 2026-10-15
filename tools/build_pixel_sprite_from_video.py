"""Generates workflows/pixel-art/sprite-from-video.json.

Edit the constants below and re-run:  python tools/build_pixel_sprite_from_video.py

Pixel sprite + driving video -> sprite sheet with the video's motion, at the sprite's own scale and palette:
PrepareSpriteForVideo upscales the sprite by an integer factor onto a flat background (the reference image), Wan
Animate 2 (core nodes, raw driving video, no pose preprocessing) moves the character like the person in the video,
then RepixelateFrames (height match on) scales the clip once so the first frame matches the sprite's height and brings it
back onto exactly the same grid with the sprite's exact palette. One fragment of up to 81 frames (~5 s at 16 fps).
"""
from comfy_workflow import (ANIMATE_HF, ANIMATE_LORA, ANIMATE_MODEL, BLUE, CLIP_VISION_H, GREEN, NEGATIVE, RED,
                            TEXT_ENCODER, URLS, VAE, WORKFLOWS, YELLOW, Graph, inp, out)

OUT = WORKFLOWS / "pixel-art" / "sprite-from-video.json"

PROMPT = ("Character Description: a pixel art game character sprite with crisp pixels and flat colours, moving like "
          "the person in the video.\nBackground description: a flat solid single-colour background, no shadows.")
NEG = NEGATIVE + ", camera movement, zoom, background change, shadow, text"
TARGET_SIZE, PADDING = 640, 0.6        # more room than for idle loops: driving videos move around
FPS, MAX_FRAMES = 16, 81
STEPS, SHIFT, SEED = 6, 5.0, 11
# frames, filename_prefix, frame_step, max_frames, loop_trim, columns, spacing, trim, fps, tag, preview_scale
SHEET = ["pixel/sprite_from_video", 2, 0, False, 0, 0, True, 8, "anim", 4]

# height_match, bg_tolerance, method, colors (0 = sprite palette), palette, stability, deflicker, cleanup, outline
REPIX = [True, 18.0, "dominant", 0, "sprite", 0.7, True, True, "none"]

NOTE = f"""Pixel sprite + driving video -> sprite sheet

1. Pixel sprite (blue): native pixel art (transparent or plain background); output keeps its pixel size + palette.
2. Driving video (blue): a person doing the motion (walk cycle, attack, jump), whole body visible, one person,
   static camera. Read at {FPS} fps, up to {MAX_FRAMES} frames ("frame_load_cap"; "skip_first_frames" to start later).
   The frame count is rounded down to 4n+1 automatically.
3. Wan Animate 2 moves your character like the video (no pose skeleton needed). {STEPS} steps with the lightx2v
   distill LoRA, CFG 1, ~1-2 min on an RTX 4090.
4. Re-pixelate: "height_match" scales the whole clip once so the character in the FIRST frame has the sprite's
   height (start the video in a neutral standing pose); the motion (steps, crouches, jumps) is kept as is.
Colours: Re-pixelate 'colors' merges the sprite palette down to N of its own colours (fewer = steadier),
   'palette' swaps in a preset/hex list; 'stability' + 'deflicker' stop colours popping between frames.
5. Sprite sheet: frame_step {SHEET[1]} keeps every 2nd frame; set max_frames / loop_trim for cycles.
The "Animation preview" node plays the sheet frames at the sheet's fps (zoom, background, ping-pong).
Saved to output/pixel/: sprite_from_video_*.png (sheet), .json (Aseprite), .webp, _preview.png,
and the raw video (pixel/sprite_from_video_raw) to check the motion."""

g = Graph()
node, link = g.node, g.link


def models(name, url, directory, type_):
    return {"Node name for S&R": type_, "models": [{"name": name, "url": url, "directory": directory}]}


# Models
node(1, "UNETLoader", [0, 0], [420, 82], outputs=[out("MODEL", "MODEL")], widgets=[ANIMATE_MODEL, "default"],
     title="Wan Animate 2", props=models(ANIMATE_MODEL, f"{ANIMATE_HF}/diffusion_models/{ANIMATE_MODEL}", "diffusion_models", "UNETLoader"))
node(2, "LoraLoaderModelOnly", [0, 130], [420, 82], inputs=[inp("model", "MODEL")], outputs=[out("MODEL", "MODEL")],
     widgets=[ANIMATE_LORA, 1.0], title="lightx2v distill LoRA (6 steps)",
     props=models(ANIMATE_LORA, f"{ANIMATE_HF}/loras/{ANIMATE_LORA}", "loras", "LoraLoaderModelOnly"))
node(3, "WanAnimate2Cache", [0, 260], [420, 82], inputs=[inp("model", "MODEL")], outputs=[out("MODEL", "MODEL")],
     widgets=["gpu", "int8"])
node(4, "ModelSamplingSD3", [0, 390], [420, 58], inputs=[inp("model", "MODEL")], outputs=[out("MODEL", "MODEL")],
     widgets=[SHIFT])
node(5, "CLIPLoader", [0, 490], [420, 106], outputs=[out("CLIP", "CLIP")], widgets=[TEXT_ENCODER, "wan", "default"],
     title="Load UMT5 text encoder", props=models(TEXT_ENCODER, URLS[TEXT_ENCODER], "text_encoders", "CLIPLoader"))
node(6, "VAELoader", [0, 640], [420, 58], outputs=[out("VAE", "VAE")], widgets=[VAE],
     props=models(VAE, URLS[VAE], "vae", "VAELoader"))
node(7, "CLIPVisionLoader", [0, 740], [420, 58], outputs=[out("CLIP_VISION", "CLIP_VISION")], widgets=[CLIP_VISION_H],
     props=models(CLIP_VISION_H, f"{ANIMATE_HF}/clip_vision/{CLIP_VISION_H}", "clip_vision", "CLIPVisionLoader"))

# Inputs
node(10, "LoadImage", [460, 0], [340, 400], outputs=[out("IMAGE", "IMAGE"), out("MASK", "MASK")],
     widgets=["example.png", "image"], title="Pixel sprite", color=BLUE)
node(11, "JoinImageWithAlpha", [460, 440], [340, 46], inputs=[inp("image", "IMAGE"), inp("alpha", "MASK")],
     outputs=[out("IMAGE", "IMAGE")], title="Keep sprite transparency")
node(12, "PrepareSpriteForVideo", [460, 530], [340, 130], inputs=[inp("sprite", "IMAGE")],
     outputs=[out("image", "IMAGE"), out("pixel_spec", "PIXEL_SPEC"), out("width", "INT"), out("height", "INT"),
              out("info", "STRING")],
     widgets=[TARGET_SIZE, PADDING, "auto"], title="Prepare sprite (integer upscale)", color=YELLOW)
node(13, "VHS_LoadVideo", [840, 0], [340, 520],
     outputs=[out("IMAGE", "IMAGE"), out("frame_count", "INT"), out("audio", "AUDIO"), out("video_info", "VHS_VIDEOINFO")],
     inputs=[inp("meta_batch", "VHS_BatchManager"), inp("vae", "VAE")],
     widgets={"video": "example.mp4", "force_rate": FPS, "custom_width": 0, "custom_height": 0,
              "frame_load_cap": MAX_FRAMES, "skip_first_frames": 0, "select_every_nth": 1, "format": "Wan",
              "videopreview": {"hidden": False, "paused": False, "params": {}}},
     title="Driving video", color=BLUE, props={"Node name for S&R": "VHS_LoadVideo", "cnr_id": "comfyui-videohelpersuite"})
node(14, "ImageScale", [840, 560], [340, 130],
     inputs=[inp("image", "IMAGE"), inp("width", "INT", "width"), inp("height", "INT", "height")],
     outputs=[out("IMAGE", "IMAGE")], widgets=["area", TARGET_SIZE, TARGET_SIZE, "center"], title="Video to sprite canvas")
node(15, "ComfyMathExpression", [840, 730], [340, 100], inputs=[inp("values.a", "FLOAT,INT,BOOLEAN")],
     outputs=[out("FLOAT", "FLOAT"), out("INT", "INT"), out("BOOL", "BOOLEAN")],
     widgets=[f"min({MAX_FRAMES}, 4 * floor((a - 1) / 4) + 1)"], title="Frames (4n+1)")
node(16, "PreviewAny", [460, 700], [340, 120], inputs=[inp("source", "*")], title="Sprite info")

# Wan Animate 2
node(20, "CLIPTextEncode", [1220, 0], [400, 180], inputs=[inp("clip", "CLIP")],
     outputs=[out("CONDITIONING", "CONDITIONING")], widgets=[PROMPT], title="Prompt", color=GREEN)
node(21, "CLIPTextEncode", [1220, 220], [400, 140], inputs=[inp("clip", "CLIP")],
     outputs=[out("CONDITIONING", "CONDITIONING")], widgets=[NEG], title="Negative prompt", color=RED)
node(22, "CLIPVisionEncode", [1220, 400], [400, 80], inputs=[inp("clip_vision", "CLIP_VISION"), inp("image", "IMAGE")],
     outputs=[out("CLIP_VISION_OUTPUT", "CLIP_VISION_OUTPUT")], widgets=["none"])
node(23, "WanAnimate2ToVideo", [1660, 0], [340, 330],
     inputs=[inp("positive", "CONDITIONING"), inp("negative", "CONDITIONING"), inp("vae", "VAE"),
             inp("reference_image", "IMAGE"), inp("pose_video", "IMAGE"), inp("clip_vision_output", "CLIP_VISION_OUTPUT"),
             inp("positive_pose", "CONDITIONING"), inp("clip_vision_output_pose", "CLIP_VISION_OUTPUT"),
             inp("continue_motion", "IMAGE"),
             inp("width", "INT", "width"), inp("height", "INT", "height"), inp("length", "INT", "length")],
     outputs=[out("positive", "CONDITIONING"), out("negative", "CONDITIONING"), out("latent", "LATENT"),
              out("trim_latent", "INT"), out("trim_image", "INT"), out("video_frame_offset", "INT")],
     widgets=[TARGET_SIZE, TARGET_SIZE, MAX_FRAMES, 1, 0, 1.0, 0.0, 1.0, 1.0], title="Wan Animate 2", color=YELLOW)
node(24, "KSamplerSelect", [1660, 380], [340, 58], outputs=[out("SAMPLER", "SAMPLER")], widgets=["lcm"])
node(25, "BasicScheduler", [1660, 480], [340, 106], inputs=[inp("model", "MODEL")], outputs=[out("SIGMAS", "SIGMAS")],
     widgets=["simple", STEPS, 1.0])
node(26, "SamplerCustom", [2040, 0], [320, 230],
     inputs=[inp("model", "MODEL"), inp("positive", "CONDITIONING"), inp("negative", "CONDITIONING"),
             inp("sampler", "SAMPLER"), inp("sigmas", "SIGMAS"), inp("latent_image", "LATENT")],
     outputs=[out("output", "LATENT"), out("denoised_output", "LATENT")], widgets=[True, SEED, "randomize", 1.0],
     title="Sampler (seed here)")
node(27, "TrimVideoLatent", [2040, 280], [320, 58], inputs=[inp("samples", "LATENT"), inp("trim_amount", "INT", "trim_amount")],
     outputs=[out("LATENT", "LATENT")], widgets=[0])
node(28, "VAEDecode", [2040, 380], [210, 46], inputs=[inp("samples", "LATENT"), inp("vae", "VAE")],
     outputs=[out("IMAGE", "IMAGE")])
node(29, "VHS_VideoCombine", [2040, 470], [320, 420],
     inputs=[inp("images", "IMAGE"), inp("audio", "AUDIO"), inp("meta_batch", "VHS_BatchManager"), inp("vae", "VAE")],
     outputs=[out("Filenames", "VHS_FILENAMES")],
     widgets={"frame_rate": FPS, "loop_count": 0, "filename_prefix": "pixel/sprite_from_video_raw",
              "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 17, "save_metadata": True,
              "trim_to_audio": False, "pingpong": False, "save_output": True,
              "videopreview": {"hidden": False, "paused": False, "params": {}}},
     title="Raw video (motion check)", props={"Node name for S&R": "VHS_VideoCombine", "cnr_id": "comfyui-videohelpersuite"})

# Back to pixel art
node(30, "RepixelateFrames", [2400, 0], [340, 250], inputs=[inp("frames", "IMAGE"), inp("pixel_spec", "PIXEL_SPEC")],
     outputs=[out("pixel_frames", "IMAGE"), out("info", "STRING"), out("palette", "STRING")], widgets=REPIX,
     title="Repixelate (height match, same grid + palette)", color=YELLOW)
node(31, "SaveSpriteSheet", [2400, 180], [340, 330], inputs=[inp("frames", "IMAGE")],
     outputs=[out("sheet", "IMAGE"), out("frames", "IMAGE"), out("info", "STRING"), out("fps", "INT")], widgets=SHEET,
     title="Save sprite sheet", color=GREEN)
node(32, "PixelArtCheck", [2780, 0], [340, 106], inputs=[inp("images", "IMAGE")],
     outputs=[out("report", "STRING"), out("passed", "BOOLEAN")], widgets=[256, 0, False], title="Pixel Art Check")
node(33, "PreviewAny", [2780, 150], [340, 200], inputs=[inp("source", "*")], title="Check report")
node(34, "PreviewAny", [2780, 390], [340, 160], inputs=[inp("source", "*")], title="Sheet info")
node(37, "SpriteAnimationPreview", [3160, 0], [520, 680], inputs=[inp("frames", "IMAGE"), inp("fps", "INT", "fps")],
     widgets=[8, 6, "checker", "loop"], title="Animation preview (playback)", color=GREEN)
node(35, "Note", [2400, 560], [560, 520], widgets=[NOTE], title="READ ME", props={})

node(36, "PreviewAny", [2780, 600], [340, 160], inputs=[inp("source", "*")], title="Re-pixelate info (colours, popping)")

link(1, 0, 2, 0)
link(2, 0, 3, 0)
link(3, 0, 4, 0)
link(10, 0, 11, 0)            # LoadImage drops alpha into MASK: join it back
link(10, 1, 11, 1)
link(11, 0, 12, 0)
link(12, 4, 16, 0)
link(13, 0, 14, 0)
link(12, 2, 14, 1)
link(12, 3, 14, 2)
link(13, 1, 15, 0)
link(5, 0, 20, 0)
link(5, 0, 21, 0)
link(7, 0, 22, 0)
link(12, 0, 22, 1)
link(20, 0, 23, 0)
link(21, 0, 23, 1)
link(6, 0, 23, 2)
link(12, 0, 23, 3)
link(14, 0, 23, 4)
link(22, 0, 23, 5)
link(12, 2, 23, 9)
link(12, 3, 23, 10)
link(15, 1, 23, 11)
link(4, 0, 25, 0)
link(4, 0, 26, 0)
link(23, 0, 26, 1)
link(23, 1, 26, 2)
link(24, 0, 26, 3)
link(25, 0, 26, 4)
link(23, 2, 26, 5)
link(26, 0, 27, 0)
link(23, 3, 27, 1)
link(27, 0, 28, 0)
link(6, 0, 28, 1)
link(28, 0, 29, 0)
link(28, 0, 30, 0)
link(12, 1, 30, 1)
link(30, 0, 31, 0)
link(31, 1, 32, 0)
link(32, 0, 33, 0)
link(31, 2, 34, 0)

link(30, 1, 36, 0)
link(31, 1, 37, 0)           # the sheet's frames + fps -> playback
link(31, 3, 37, 1)

g.save(OUT, "pixel-art-sprite-from-video", groups=[
    ("Models", [-20, -60, 460, 880], "#3f789e"),
    ("Sprite + driving video", [440, -60, 760, 900], "#a1309b"),
    ("Animate (Wan Animate 2)", [1200, -60, 1180, 960], "#88A"),
    ("Back to pixel art", [2380, -60, 760, 620], "#444"),
], scale=0.5, offset=(40, 80))
