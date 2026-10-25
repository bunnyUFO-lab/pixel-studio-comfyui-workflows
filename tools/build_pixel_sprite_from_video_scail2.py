"""Generates workflows/pixel-art/sprite-from-video-scail2.json.

Edit the constants below and re-run:  python tools/build_pixel_sprite_from_video_scail2.py

The SCAIL 2 arm of pixel-art/sprite-from-video (same inputs, same outputs, so the two can be compared clip for clip):
PrepareSpriteForVideo upscales the sprite by an integer factor onto a flat background (sizes divisible by 32, as
SCAIL 2 needs), SCAIL 2 (Wan 2.1 14B, core WanSCAILToVideo, animation mode) moves the character like the person in
the driving video, guided by two coloured masks from SCAIL2ColoredMask: SAM3 tracks the person in the video, and the
sprite's own silhouette (SpriteVideoMask) marks the reference, since SAM3 is unreliable on non-human sprites. Then
RepixelateFrames (height match on) brings the clip back onto exactly the same grid with the sprite's exact palette.
One 81-frame segment (~5 s at 16 fps). Settings follow Comfy's SCAIL-2 int8 template (DPO LoRA 1.0, lightx2v 0.8,
6 steps, euler, CFG 1, shift 5).
"""
from comfy_workflow import (ANIMATE_HF, ANIMATE_LORA, BLUE, CLIP_VISION_H, GREEN, NEGATIVE, RED, SAM3, SCAIL_DPO_LORA,
                            SCAIL_MODEL, TEXT_ENCODER, URLS, VAE, WORKFLOWS, YELLOW, Graph, inp, out)

OUT = WORKFLOWS / "pixel-art" / "sprite-from-video-scail2.json"
OUT_VIEWS = WORKFLOWS / "pixel-art" / "sprite-from-video-scail2-views.json"   # + back and side reference views

# SCAIL 2 leans on the video and the reference: one plain sentence (card "Updating the video generator", Prompts tab)
PROMPT = ("A pixel art game character with crisp pixels and flat colours performs the movement from the video on a "
          "plain flat background, static camera.")
NEG = NEGATIVE + ", camera movement, zoom, background change, shadow, text"
MASK_PROMPT = "human"                  # SAM3 text prompt for the person in the driving video
TARGET_SIZE, PADDING = 640, 0.6        # same as sprite-from-video
FPS, MAX_FRAMES = 16, 81
STEPS, SHIFT, SEED = 6, 5.0, 11
DPO_STRENGTH, DISTILL_STRENGTH = 1.0, 0.8
POSE_STRENGTH, POSE_START, POSE_END = 1.0, 0.0, 1.0
SHEET = ["pixel/sprite_from_video_scail2", 2, 0, False, 0, 0, True, 8, "anim", 4]
REPIX = [True, 18.0, "dominant", 0, "sprite", 0.7, True, True, "none"]

NOTE = f"""Pixel sprite + driving video -> sprite sheet (SCAIL 2)

The SCAIL 2 version of pixel-art/sprite-from-video: same inputs and outputs, a different motion model.
1. Pixel sprite (blue): native pixel art (transparent or plain background); output keeps its pixel size + palette.
2. Driving video (blue): one person doing the motion, whole body visible, static camera. Read at {FPS} fps, up to
   {MAX_FRAMES} frames ("frame_load_cap"; "skip_first_frames" to start later). Rounded down to 4n+1.
3. Masks: SAM3 finds the person in the video ("Driving video mask prompt", default "{MASK_PROMPT}"); the sprite's own
   silhouette is the reference mask, so creatures and robots work too. Check both in the mask previews.
4. SCAIL 2 (animation mode) moves your character like the video. DPO LoRA + lightx2v distill LoRA, {STEPS} steps,
   CFG 1. Keep the prompt to one plain sentence; the video and the sprite do the work.
5. Re-pixelate: "height_match" scales the clip once so the character in the FIRST frame has the sprite's height
   (start the video in a neutral standing pose). Colours as in sprite-from-video.
6. Sprite sheet: frame_step {SHEET[1]} keeps every 2nd frame; set max_frames / loop_trim for cycles.
Saved to output/pixel/: sprite_from_video_scail2_*.png (sheet), .json (Aseprite), .webp, _preview.png,
and the raw video (pixel/sprite_from_video_scail2_raw) to check the motion."""

def models(name, url, directory, type_):
    return {"Node name for S&R": type_, "models": [{"name": name, "url": url, "directory": directory}]}



def build(views):
    g = Graph()
    node, link = g.node, g.link


    # Models
    node(1, "UNETLoader", [0, 0], [420, 82], outputs=[out("MODEL", "MODEL")], widgets=[SCAIL_MODEL, "default"],
         title="SCAIL 2", props=models(SCAIL_MODEL, URLS[SCAIL_MODEL], "diffusion_models", "UNETLoader"))
    node(2, "LoraLoaderModelOnly", [0, 130], [420, 82], inputs=[inp("model", "MODEL")], outputs=[out("MODEL", "MODEL")],
         widgets=[SCAIL_DPO_LORA, DPO_STRENGTH], title="SCAIL 2 DPO LoRA",
         props=models(SCAIL_DPO_LORA, URLS[SCAIL_DPO_LORA], "loras", "LoraLoaderModelOnly"))
    node(3, "LoraLoaderModelOnly", [0, 260], [420, 82], inputs=[inp("model", "MODEL")], outputs=[out("MODEL", "MODEL")],
         widgets=[ANIMATE_LORA, DISTILL_STRENGTH], title="lightx2v distill LoRA (6 steps)",
         props=models(ANIMATE_LORA, f"{ANIMATE_HF}/loras/{ANIMATE_LORA}", "loras", "LoraLoaderModelOnly"))
    node(4, "ModelSamplingSD3", [0, 390], [420, 58], inputs=[inp("model", "MODEL")], outputs=[out("MODEL", "MODEL")],
         widgets=[SHIFT])
    node(5, "CLIPLoader", [0, 490], [420, 106], outputs=[out("CLIP", "CLIP")], widgets=[TEXT_ENCODER, "wan", "default"],
         title="Load UMT5 text encoder", props=models(TEXT_ENCODER, URLS[TEXT_ENCODER], "text_encoders", "CLIPLoader"))
    node(6, "VAELoader", [0, 640], [420, 58], outputs=[out("VAE", "VAE")], widgets=[VAE],
         props=models(VAE, URLS[VAE], "vae", "VAELoader"))
    node(7, "CLIPVisionLoader", [0, 740], [420, 58], outputs=[out("CLIP_VISION", "CLIP_VISION")], widgets=[CLIP_VISION_H],
         props=models(CLIP_VISION_H, f"{ANIMATE_HF}/clip_vision/{CLIP_VISION_H}", "clip_vision", "CLIPVisionLoader"))
    node(8, "CheckpointLoaderSimple", [0, 840], [420, 98], outputs=[out("MODEL", "MODEL"), out("CLIP", "CLIP"), out("VAE", "VAE")],
         widgets=[SAM3], title="SAM3.1 (masks)", props=models(SAM3, URLS[SAM3], "checkpoints", "CheckpointLoaderSimple"))

    # Inputs
    node(10, "LoadImage", [460, 0], [340, 400], outputs=[out("IMAGE", "IMAGE"), out("MASK", "MASK")],
         widgets=["example.png", "image"], title="Pixel sprite", color=BLUE)
    node(11, "JoinImageWithAlpha", [460, 440], [340, 46], inputs=[inp("image", "IMAGE"), inp("alpha", "MASK")],
         outputs=[out("IMAGE", "IMAGE")], title="Keep sprite transparency")
    node(12, "PrepareSpriteForVideo", [460, 530], [340, 154], inputs=[inp("sprite", "IMAGE")],
         outputs=[out("image", "IMAGE"), out("pixel_spec", "PIXEL_SPEC"), out("width", "INT"), out("height", "INT"),
                  out("info", "STRING")],
         widgets=[TARGET_SIZE, PADDING, "auto", 32], title="Prepare sprite (integer upscale, /32)", color=YELLOW)
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
    node(16, "PreviewAny", [460, 720], [340, 120], inputs=[inp("source", "*")], title="Sprite info")

    # Masks
    node(40, "SpriteVideoMask", [460, 880], [340, 46], inputs=[inp("image", "IMAGE"), inp("pixel_spec", "PIXEL_SPEC")],
         outputs=[out("mask", "MASK")], title="Sprite silhouette (reference mask)")
    node(41, "CLIPTextEncode", [840, 870], [340, 100], inputs=[inp("clip", "CLIP")],
         outputs=[out("CONDITIONING", "CONDITIONING")], widgets=[MASK_PROMPT], title="Driving video mask prompt", color=GREEN)
    node(42, "SAM3_VideoTrack", [840, 1010], [340, 130],
         inputs=[inp("images", "IMAGE"), inp("model", "MODEL"), inp("initial_mask", "MASK"), inp("conditioning", "CONDITIONING")],
         outputs=[out("track_data", "SAM3_TRACK_DATA")], widgets=[0.5, 1, 1], title="SAM3 track (driving video)")
    node(43, "SCAIL2ColoredMask", [1220, 870], [340, 130],
         inputs=[inp("driving_track_data", "SAM3_TRACK_DATA"), inp("ref_track_data", "SAM3_TRACK_DATA,MASK")],
         outputs=[out("pose_video_mask", "IMAGE"), out("reference_image_mask", "IMAGE")],
         widgets=["", "left_to_right", False], title="SCAIL 2 coloured masks (animation mode)")
    node(44, "PreviewImage", [1220, 1040], [200, 240], inputs=[inp("images", "IMAGE")], title="Video mask")
    node(45, "PreviewImage", [1440, 1040], [200, 240], inputs=[inp("images", "IMAGE")], title="Reference mask")

    # SCAIL 2
    node(20, "CLIPTextEncode", [1220, 0], [400, 180], inputs=[inp("clip", "CLIP")],
         outputs=[out("CONDITIONING", "CONDITIONING")], widgets=[PROMPT], title="Prompt", color=GREEN)
    node(21, "CLIPTextEncode", [1220, 220], [400, 140], inputs=[inp("clip", "CLIP")],
         outputs=[out("CONDITIONING", "CONDITIONING")], widgets=[NEG], title="Negative prompt", color=RED)
    node(22, "CLIPVisionEncode", [1220, 400], [400, 80], inputs=[inp("clip_vision", "CLIP_VISION"), inp("image", "IMAGE")],
         outputs=[out("CLIP_VISION_OUTPUT", "CLIP_VISION_OUTPUT")], widgets=["none"])
    node(23, "WanSCAILToVideo", [1660, 0], [340, 380],
         inputs=[inp("positive", "CONDITIONING"), inp("negative", "CONDITIONING"), inp("vae", "VAE"),
                 inp("pose_video", "IMAGE"), inp("pose_video_mask", "IMAGE"), inp("reference_image", "IMAGE"),
                 inp("reference_image_mask", "IMAGE"), inp("clip_vision_output", "CLIP_VISION_OUTPUT"),
                 inp("previous_frames", "IMAGE"),
                 inp("width", "INT", "width"), inp("height", "INT", "height"), inp("length", "INT", "length")],
         outputs=[out("positive", "CONDITIONING"), out("negative", "CONDITIONING"), out("latent", "LATENT"),
                  out("video_frame_offset", "INT")],
         widgets=[TARGET_SIZE, TARGET_SIZE, MAX_FRAMES, 1, POSE_STRENGTH, POSE_START, POSE_END, 0, 5, False],
         title="SCAIL 2", color=YELLOW)
    node(24, "KSamplerSelect", [1660, 430], [340, 58], outputs=[out("SAMPLER", "SAMPLER")], widgets=["euler"])
    node(25, "BasicScheduler", [1660, 530], [340, 106], inputs=[inp("model", "MODEL")], outputs=[out("SIGMAS", "SIGMAS")],
         widgets=["simple", STEPS, 1.0])
    node(26, "SamplerCustom", [2040, 0], [320, 230],
         inputs=[inp("model", "MODEL"), inp("positive", "CONDITIONING"), inp("negative", "CONDITIONING"),
                 inp("sampler", "SAMPLER"), inp("sigmas", "SIGMAS"), inp("latent_image", "LATENT")],
         outputs=[out("output", "LATENT"), out("denoised_output", "LATENT")], widgets=[True, SEED, "randomize", 1.0],
         title="Sampler (seed here)")
    node(28, "VAEDecode", [2040, 280], [210, 46], inputs=[inp("samples", "LATENT"), inp("vae", "VAE")],
         outputs=[out("IMAGE", "IMAGE")])
    node(29, "VHS_VideoCombine", [2040, 370], [320, 420],
         inputs=[inp("images", "IMAGE"), inp("audio", "AUDIO"), inp("meta_batch", "VHS_BatchManager"), inp("vae", "VAE")],
         outputs=[out("Filenames", "VHS_FILENAMES")],
         widgets={"frame_rate": FPS, "loop_count": 0, "filename_prefix": "pixel/sprite_from_video_scail2_raw",
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
    node(36, "PreviewAny", [2780, 600], [340, 160], inputs=[inp("source", "*")], title="Re-pixelate info (colours, popping)")
    node(37, "SpriteAnimationPreview", [3160, 0], [520, 680], inputs=[inp("frames", "IMAGE"), inp("fps", "INT", "fps")],
         widgets=[8, 6, "checker", "loop"], title="Animation preview (playback)", color=GREEN)
    node(35, "Note", [2400, 560], [560, 560], widgets=[NOTE], title="READ ME", props={})


    if views:
        # Extra reference views (SCAIL 2 takes a batch: the sprite first, then other views of it, each masked in the same
        # identity colour). MatchSpriteCanvas puts them on the sprite's canvas at the same scale and foot line.
        node(50, "LoadImage", [460, 1240], [340, 300], outputs=[out("IMAGE", "IMAGE"), out("MASK", "MASK")],
             widgets=["example.png", "image"], title="Back view", color=BLUE)
        node(51, "LoadImage", [460, 1580], [340, 300], outputs=[out("IMAGE", "IMAGE"), out("MASK", "MASK")],
             widgets=["example.png", "image"], title="Side view", color=BLUE)
        node(55, "JoinImageWithAlpha", [840, 1240], [340, 46], inputs=[inp("image", "IMAGE"), inp("alpha", "MASK")],
             outputs=[out("IMAGE", "IMAGE")], title="Back view transparency")
        node(56, "JoinImageWithAlpha", [840, 1300], [340, 46], inputs=[inp("image", "IMAGE"), inp("alpha", "MASK")],
             outputs=[out("IMAGE", "IMAGE")], title="Side view transparency")
        node(52, "ImageBatch", [840, 1360], [340, 46], inputs=[inp("image1", "IMAGE"), inp("image2", "IMAGE")],
             outputs=[out("IMAGE", "IMAGE")], title="Views")
        node(53, "MatchSpriteCanvas", [840, 1420], [340, 46], inputs=[inp("views", "IMAGE"), inp("pixel_spec", "PIXEL_SPEC")],
             outputs=[out("images", "IMAGE")], title="Views on the sprite canvas")
        node(54, "ImageBatch", [840, 1480], [340, 46], inputs=[inp("image1", "IMAGE"), inp("image2", "IMAGE")],
             outputs=[out("IMAGE", "IMAGE")], title="Sprite + views (references)")
        link(50, 0, 55, 0)
        link(50, 1, 55, 1)
        link(51, 0, 56, 0)
        link(51, 1, 56, 1)
        link(55, 0, 52, 0)
        link(56, 0, 52, 1)
        link(52, 0, 53, 0)
        link(12, 1, 53, 1)
        link(12, 0, 54, 0)
        link(53, 0, 54, 1)

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
    # masks
    link(12 if not views else 54, 0, 40, 0)   # sprite (+ views) -> masks
    link(12, 1, 40, 1)
    link(8, 1, 41, 0)
    link(14, 0, 42, 0)
    link(8, 0, 42, 1)
    link(41, 0, 42, 3)
    link(42, 0, 43, 0)
    link(40, 0, 43, 1)
    link(43, 0, 44, 0)
    link(43, 1, 45, 0)
    # SCAIL 2
    link(5, 0, 20, 0)
    link(5, 0, 21, 0)
    link(7, 0, 22, 0)
    link(12, 0, 22, 1)
    link(20, 0, 23, 0)
    link(21, 0, 23, 1)
    link(6, 0, 23, 2)
    link(14, 0, 23, 3)
    link(43, 0, 23, 4)
    link(12 if not views else 54, 0, 23, 5)   # reference image(s)
    link(43, 1, 23, 6)
    link(22, 0, 23, 7)
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
    link(26, 1, 28, 0)            # denoised output, as in Comfy's SCAIL-2 template
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

    g.save(OUT_VIEWS if views else OUT, "pixel-art-sprite-from-video-scail2" + ("-views" if views else ""), groups=[
        ("Models", [-20, -60, 460, 1020], "#3f789e"),
        ("Sprite + driving video", [440, -60, 760, 1220], "#a1309b"),
        ("Animate (SCAIL 2)", [1200, -60, 1180, 1360], "#88A"),
        ("Back to pixel art", [2380, -60, 760, 620], "#444"),
    ], scale=0.5, offset=(40, 80))


build(views=False)
build(views=True)
