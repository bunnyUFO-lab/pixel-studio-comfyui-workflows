"""Generates the fast local sprite-animation workflows:

  workflows/pixel-art/sprite-from-prompt-ti2v5b.json  Wan 2.2 TI2V-5B + FastWan LoRA, sprite + motion prompt
  workflows/pixel-art/sprite-from-prompt-vace.json    Wan 2.1 VACE 1.3B, first = last frame = the sprite (closed loop)
  workflows/pixel-art/sprite-from-video-vace.json     Wan 2.1 VACE 1.3B, pose skeletons from a driving video (SDPose)

Edit the constants below and re-run:  python tools/build_pixel_sprite_fast.py

Same pattern as sprite-from-prompt / sprite-from-video: PrepareSpriteForVideo upscales the sprite by an integer
factor onto a flat background, the video model animates it, RIFE doubles the frame rate (smoother motion), then
RepixelateFrames brings every frame back onto the sprite's grid and palette and SaveSpriteSheet writes the sheet.
All core ComfyUI nodes plus VideoHelperSuite and this repo's pixel nodes: no new custom nodes.
"""
from comfy_workflow import (BLUE, CAUSVID_1_3B_LORA, FASTWAN_5B_LORA, GREEN, NEGATIVE, RED, RIFE, TEXT_ENCODER, TI2V_5B,
                            URLS, VACE_1_3B, VAE, WAN22_VAE, WORKFLOWS, YELLOW, Graph, inp, out)

PROMPT = ("Pixel art sprite animation. The character does a looping idle animation: breathing, a slight up and down "
          "bob, cape and hair swaying gently. Static camera, the character stays in place, flat solid background.")
NEG = NEGATIVE + ", camera movement, zoom, panning, background change, new objects, text"
POSE_PROMPT = ("A pixel art game character sprite with crisp pixels and flat colours, moving like the pose skeleton. "
               "Static camera, flat solid single-colour background, no shadows.")
SEED = 7
RIFE_X = 2                              # frames per source frame after interpolation
RIFE_ON = False                         # RIFE node active (True) or bypassed (False); the sheet keeps every frame

# per workflow: model settings (picked in the fast-models benchmark, Sprite Studio card "Updating the video generator")
TI2V = dict(target=1024, padding=0.35, frames=49, fps=24, steps=4, cfg=1.0, sampler="euler", scheduler="simple",
            shift=5.0, lora=FASTWAN_5B_LORA, lora_strength=1.0)
VACE = dict(target=640, padding=0.35, frames=49, fps=16, steps=8, cfg=1.0, sampler="euler", scheduler="simple",
            shift=8.0, lora=CAUSVID_1_3B_LORA, lora_strength=0.7, lora_mode=0)
POSE = dict(VACE, padding=0.6)

# height_match, bg_tolerance, method, colors (0 = sprite palette), palette, stability, deflicker, cleanup, outline
REPIX = [False, 18.0, "dominant", 0, "sprite", 0.7, True, True, "none"]


def sheet(prefix, fps, loop=True, tag="idle"):
    # frames, filename_prefix, frame_step, max_frames, loop_trim, columns, spacing, trim, fps, tag, preview_scale
    return [prefix, 1, 0, loop, 0, 0, True, fps * (RIFE_X if RIFE_ON else 1), tag, 4]


def models(name, directory, type_, url=None):
    return {"Node name for S&R": type_, "models": [{"name": name, "url": url or URLS[name], "directory": directory}]}


def common_inputs(g, cfg, vae):
    """Text encoder, VAE, sprite -> PrepareSpriteForVideo. Node ids 4-16."""
    node, link = g.node, g.link
    node(4, "CLIPLoader", [0, 300], [420, 106], outputs=[out("CLIP", "CLIP")], widgets=[TEXT_ENCODER, "wan", "default"],
         title="Load UMT5 text encoder", props=models(TEXT_ENCODER, "text_encoders", "CLIPLoader"))
    node(5, "VAELoader", [0, 450], [420, 58], outputs=[out("VAE", "VAE")], widgets=[vae], props=models(vae, "vae", "VAELoader"))
    node(10, "LoadImage", [460, 0], [340, 400], outputs=[out("IMAGE", "IMAGE"), out("MASK", "MASK")],
         widgets=["example.png", "image"], title="Pixel sprite", color=BLUE)
    node(11, "JoinImageWithAlpha", [460, 440], [340, 46], inputs=[inp("image", "IMAGE"), inp("alpha", "MASK")],
         outputs=[out("IMAGE", "IMAGE")], title="Keep sprite transparency")
    node(12, "PrepareSpriteForVideo", [460, 530], [340, 130], inputs=[inp("sprite", "IMAGE")],
         outputs=[out("image", "IMAGE"), out("pixel_spec", "PIXEL_SPEC"), out("width", "INT"), out("height", "INT"),
                  out("info", "STRING")],
         widgets=[cfg["target"], cfg["padding"], "auto"], title="Prepare sprite (integer upscale)", color=YELLOW)
    node(16, "PreviewAny", [460, 700], [340, 120], inputs=[inp("source", "*")], title="Sprite info")
    link(10, 0, 11, 0)            # LoadImage drops alpha into MASK: join it back
    link(10, 1, 11, 1)
    link(11, 0, 12, 0)
    link(12, 4, 16, 0)


def model_chain(g, unet, cfg, lora_title):
    """UNET -> step-distill LoRA -> shift. Node ids 1-3."""
    g.node(1, "UNETLoader", [0, 0], [420, 82], outputs=[out("MODEL", "MODEL")], widgets=[unet, "default"],
           title="Video model", props=models(unet, "diffusion_models", "UNETLoader"))
    g.node(2, "LoraLoaderModelOnly", [0, 130], [420, 82], inputs=[inp("model", "MODEL")], outputs=[out("MODEL", "MODEL")],
           widgets=[cfg["lora"], cfg["lora_strength"]], title=lora_title, mode=cfg.get("lora_mode", 0),
           props=models(cfg["lora"], "loras", "LoraLoaderModelOnly"))
    g.node(3, "ModelSamplingSD3", [0, 250], [420, 58], inputs=[inp("model", "MODEL")], outputs=[out("MODEL", "MODEL")],
           widgets=[cfg["shift"]], title="Shift")
    g.link(1, 0, 2, 0)
    g.link(2, 0, 3, 0)


def sampler(g, cfg, x, latent_src, pos_src, neg_src):
    """KSampler (node 26) on the shifted model."""
    g.node(26, "KSampler", [x, 0], [320, 262],
           inputs=[inp("model", "MODEL"), inp("positive", "CONDITIONING"), inp("negative", "CONDITIONING"),
                   inp("latent_image", "LATENT")], outputs=[out("LATENT", "LATENT")],
           widgets=[SEED, "randomize", cfg["steps"], cfg["cfg"], cfg["sampler"], cfg["scheduler"], 1.0],
           title="Sampler (seed, steps)")
    g.link(3, 0, 26, 0)
    g.link(*pos_src, 26, 1)
    g.link(*neg_src, 26, 2)
    g.link(*latent_src, 26, 3)


def pixel_tail(g, x, decoded, prefix, fps, loop, tag, height_match=False):
    """Raw video, RIFE x2, Repixelate, sprite sheet, Pixel Art Check, previews. Node ids 29-41."""
    node, link = g.node, g.link
    node(29, "VHS_VideoCombine", [x - 360, 470], [320, 420],
         inputs=[inp("images", "IMAGE"), inp("audio", "AUDIO"), inp("meta_batch", "VHS_BatchManager"), inp("vae", "VAE")],
         outputs=[out("Filenames", "VHS_FILENAMES")],
         widgets={"frame_rate": fps, "loop_count": 0, "filename_prefix": prefix + "_raw",
                  "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 17, "save_metadata": True,
                  "trim_to_audio": False, "pingpong": False, "save_output": True,
                  "videopreview": {"hidden": False, "paused": False, "params": {}}},
         title="Raw video (motion check)", props={"Node name for S&R": "VHS_VideoCombine", "cnr_id": "comfyui-videohelpersuite"})
    node(40, "FrameInterpolationModelLoader", [x, 0], [340, 58], outputs=[out("INTERP_MODEL", "INTERP_MODEL")],
         widgets=[RIFE], title="RIFE model",
         props=models(RIFE, "frame_interpolation", "FrameInterpolationModelLoader"))
    node(41, "FrameInterpolate", [x, 100], [340, 82], inputs=[inp("interp_model", "INTERP_MODEL"), inp("images", "IMAGE")],
         outputs=[out("IMAGE", "IMAGE")], widgets=[RIFE_X], title=f"Smooth motion (RIFE x{RIFE_X}; Ctrl+B on/off)",
         color=YELLOW, mode=0 if RIFE_ON else 4)
    repix = list(REPIX)
    repix[0] = height_match
    node(30, "RepixelateFrames", [x, 230], [340, 250], inputs=[inp("frames", "IMAGE"), inp("pixel_spec", "PIXEL_SPEC")],
         outputs=[out("pixel_frames", "IMAGE"), out("info", "STRING"), out("palette", "STRING")], widgets=repix,
         title="Repixelate (same grid + palette)", color=YELLOW)
    node(31, "SaveSpriteSheet", [x, 520], [340, 330], inputs=[inp("frames", "IMAGE")],
         outputs=[out("sheet", "IMAGE"), out("frames", "IMAGE"), out("info", "STRING"), out("fps", "INT")],
         widgets=sheet(prefix, fps, loop, tag), title="Save sprite sheet", color=GREEN)
    node(32, "PixelArtCheck", [x + 380, 0], [340, 106], inputs=[inp("images", "IMAGE")],
         outputs=[out("report", "STRING"), out("passed", "BOOLEAN")], widgets=[256, 0, False], title="Pixel Art Check")
    node(33, "PreviewAny", [x + 380, 150], [340, 200], inputs=[inp("source", "*")], title="Check report")
    node(34, "PreviewAny", [x + 380, 390], [340, 160], inputs=[inp("source", "*")], title="Sheet info")
    node(36, "PreviewAny", [x + 380, 600], [340, 160], inputs=[inp("source", "*")], title="Re-pixelate info (colours, popping)")
    node(37, "SpriteAnimationPreview", [x + 760, 0], [520, 680], inputs=[inp("frames", "IMAGE"), inp("fps", "INT", "fps")],
         widgets=[8, 6, "checker", "loop"], title="Animation preview (playback)", color=GREEN)
    link(*decoded, 29, 0)
    link(40, 0, 41, 0)
    link(*decoded, 41, 1)
    link(41, 0, 30, 0)
    link(12, 1, 30, 1)
    link(30, 0, 31, 0)
    link(31, 1, 32, 0)
    link(32, 0, 33, 0)
    link(31, 2, 34, 0)
    link(30, 1, 36, 0)
    link(31, 1, 37, 0)           # the sheet's frames + fps -> playback
    link(31, 3, 37, 1)


def prompts(g, text, x=840):
    g.node(20, "CLIPTextEncode", [x, 0], [400, 180], inputs=[inp("clip", "CLIP")],
           outputs=[out("CONDITIONING", "CONDITIONING")], widgets=[text], title="Motion prompt", color=GREEN)
    g.node(21, "CLIPTextEncode", [x, 220], [400, 140], inputs=[inp("clip", "CLIP")],
           outputs=[out("CONDITIONING", "CONDITIONING")], widgets=[NEG], title="Negative prompt", color=RED)
    g.link(4, 0, 20, 0)
    g.link(4, 0, 21, 0)


def frames_node(g, n, pos):
    g.node(13, "PrimitiveInt", pos, [340, 82], outputs=[out("INT", "INT")], widgets=[n, "fixed"],
           title="Frames (4n+1)", color=YELLOW)


# --- 1. Wan 2.2 TI2V-5B: sprite + motion prompt (start frame only) --------------------------------------------------
def build_ti2v():
    c = TI2V
    g = Graph()
    model_chain(g, TI2V_5B, c, "FastWan LoRA (few steps)")
    common_inputs(g, c, WAN22_VAE)
    prompts(g, PROMPT)
    frames_node(g, c["frames"], [840, 400])
    g.node(22, "Wan22ImageToVideoLatent", [840, 520], [400, 170],
           inputs=[inp("vae", "VAE"), inp("start_image", "IMAGE"), inp("width", "INT", "width"),
                   inp("height", "INT", "height"), inp("length", "INT", "length")],
           outputs=[out("LATENT", "LATENT")], widgets=[c["target"], c["target"], c["frames"], 1],
           title="Start frame = sprite", color=YELLOW)
    g.link(5, 0, 22, 0)
    g.link(12, 0, 22, 1)
    g.link(12, 2, 22, 2)
    g.link(12, 3, 22, 3)
    g.link(13, 0, 22, 4)
    sampler(g, c, 1280, (22, 0), (20, 0), (21, 0))
    g.node(28, "VAEDecode", [1280, 300], [210, 46], inputs=[inp("samples", "LATENT"), inp("vae", "VAE")],
           outputs=[out("IMAGE", "IMAGE")])
    g.link(26, 0, 28, 0)
    g.link(5, 0, 28, 1)
    pixel_tail(g, 2000, (28, 0), "pixel/sprite_from_prompt_ti2v5b", c["fps"], loop=False, tag="anim")
    note = f"""Pixel sprite + motion prompt -> sprite sheet (Wan 2.2 TI2V-5B, fast)

1. Pixel sprite (blue): native pixel art; output keeps its pixel size + palette.
2. Motion prompt (green): describe the MOTION ("walks in place", "swings the sword", "casts a fire spell").
   Keep "static camera" and "flat solid background".
3. The sprite is the FIRST frame only (TI2V-5B has no end frame), so cycles do not close by themselves:
   use sprite-from-prompt-vace for loops.
4. {c['steps']} steps, CFG {c['cfg']}, FastWan LoRA (FastVideo, Apache-2.0). {c['frames']} frames at {c['fps']} fps.
5. RIFE x{RIFE_X} doubles the frames before Re-pixelate (smoother motion); Ctrl+B the yellow RIFE node to skip it.
6. Sprite sheet: keeps every {sheet('', c['fps'])[1]}th interpolated frame = {sheet('', c['fps'])[7]} fps.
Saved to output/pixel/: sprite_from_prompt_ti2v5b_*.png (sheet), .json (Aseprite), .webp, _raw.mp4."""
    g.node(35, "Note", [2000, 900], [560, 420], widgets=[note], title="READ ME", props={})
    g.save(WORKFLOWS / "pixel-art" / "sprite-from-prompt-ti2v5b.json", "pixel-art-sprite-from-prompt-ti2v5b", groups=[
        ("Models", [-20, -60, 460, 600], "#3f789e"),
        ("Sprite + prompt", [440, -60, 820, 900], "#a1309b"),
        ("Animate (Wan 2.2 TI2V-5B)", [1260, -60, 400, 960], "#88A"),
        ("Back to pixel art", [1980, -60, 1300, 960], "#444"),
    ], scale=0.5, offset=(40, 80))


# --- 2. Wan 2.1 VACE 1.3B: first = last frame = the sprite (closed loop) ---------------------------------------------
def vace_head(g, c, title):
    model_chain(g, VACE_1_3B, c, title)
    common_inputs(g, c, VAE)


def vace_tail(g, c, x, prefix, loop, tag, height_match=False):
    """WanVaceToVideo (22) -> KSampler -> trim the reference frames -> decode -> pixel tail."""
    sampler(g, c, x, (22, 2), (22, 0), (22, 1))
    g.node(27, "TrimVideoLatent", [x, 300], [320, 58], inputs=[inp("samples", "LATENT"), inp("trim_amount", "INT", "trim_amount")],
           outputs=[out("LATENT", "LATENT")], widgets=[0])
    g.node(28, "VAEDecode", [x, 400], [210, 46], inputs=[inp("samples", "LATENT"), inp("vae", "VAE")],
           outputs=[out("IMAGE", "IMAGE")])
    g.link(26, 0, 27, 0)
    g.link(22, 3, 27, 1)
    g.link(27, 0, 28, 0)
    g.link(5, 0, 28, 1)
    pixel_tail(g, x + 720, (28, 0), prefix, c["fps"], loop, tag, height_match)


def vace_node(g, pos):
    g.node(22, "WanVaceToVideo", pos, [400, 260],
           inputs=[inp("positive", "CONDITIONING"), inp("negative", "CONDITIONING"), inp("vae", "VAE"),
                   inp("control_video", "IMAGE"), inp("control_masks", "MASK"), inp("reference_image", "IMAGE"),
                   inp("width", "INT", "width"), inp("height", "INT", "height"), inp("length", "INT", "length")],
           outputs=[out("positive", "CONDITIONING"), out("negative", "CONDITIONING"), out("latent", "LATENT"),
                    out("trim_latent", "INT")],
           widgets=[512, 512, 49, 1, 1.0], title="VACE (control video, strength)", color=YELLOW)
    g.link(20, 0, 22, 0)
    g.link(21, 0, 22, 1)
    g.link(5, 0, 22, 2)
    g.link(12, 2, 22, 6)
    g.link(12, 3, 22, 7)


def build_vace_loop():
    c = VACE
    g = Graph()
    node, link = g.node, g.link
    vace_head(g, c, "CausVid LoRA (8 steps, CFG 1; non-commercial)")
    prompts(g, PROMPT)
    frames_node(g, c["frames"], [840, 400])
    # control video: sprite, grey x (frames - 2), sprite; mask 0 = keep, 1 = generate
    node(14, "ComfyMathExpression", [840, 520], [340, 100], inputs=[inp("values.a", "FLOAT,INT,BOOLEAN")],
         outputs=[out("FLOAT", "FLOAT"), out("INT", "INT"), out("BOOL", "BOOLEAN")], widgets=["a - 2"], title="Frames - 2")
    for nid, value, y in ((50, 0.5, 660), (55, 0.0, 760), (57, 1.0, 860)):
        node(nid, "SolidMask", [840, y], [200, 82], inputs=[inp("width", "INT", "width"), inp("height", "INT", "height")],
             outputs=[out("MASK", "MASK")], widgets=[value, 512, 512])
        node(nid + 1, "MaskToImage", [1060, y], [160, 26], inputs=[inp("mask", "MASK")], outputs=[out("IMAGE", "IMAGE")])
        link(12, 2, nid, 0)
        link(12, 3, nid, 1)
        link(nid, 0, nid + 1, 0)
    for nid, src, y in ((52, 51, 660), (59, 58, 860)):
        node(nid, "RepeatImageBatch", [1240, y], [200, 58], inputs=[inp("image", "IMAGE"), inp("amount", "INT", "amount")],
             outputs=[out("IMAGE", "IMAGE")], widgets=[c["frames"] - 2])
        link(src, 0, nid, 0)
        link(14, 1, nid, 1)
    for nid, a, b, y in ((53, 12, 52, 980), (54, 53, 12, 1040), (60, 56, 59, 1100), (61, 60, 56, 1160)):
        node(nid, "ImageBatch", [1240, y], [200, 46], inputs=[inp("image1", "IMAGE"), inp("image2", "IMAGE")],
             outputs=[out("IMAGE", "IMAGE")])
        link(a, 0, nid, 0)
        link(b, 0, nid, 1)
    node(62, "ImageToMask", [1240, 1220], [200, 58], inputs=[inp("image", "IMAGE")], outputs=[out("MASK", "MASK")],
         widgets=["red"])
    link(13, 0, 14, 0)
    link(61, 0, 62, 0)
    vace_node(g, [1480, 0])
    link(54, 0, 22, 3)
    link(62, 0, 22, 4)
    link(12, 0, 22, 5)                               # reference image = the sprite too: keeps the background
    link(13, 0, 22, 8)
    vace_tail(g, c, 1920, "pixel/sprite_from_prompt_vace", loop=True, tag="idle")
    note = f"""Pixel sprite + motion prompt -> looping sprite sheet (Wan 2.1 VACE 1.3B, fast)

1. Pixel sprite (blue): native pixel art; output keeps its pixel size + palette.
2. Motion prompt (green): describe the MOTION. Keep "static camera" and "flat solid background".
3. VACE gets a control video of [sprite, grey x (frames - 2), sprite] with a mask that keeps the first and last
   frame, so every animation starts and ends on your sprite and loops. Change frames on the yellow
   "Frames (4n+1)" node only.
4. {c['steps']} steps, CFG {c['cfg']} with the CausVid LoRA. CausVid is CC-BY-NC-4.0 (non-commercial): for a
   commercial build bypass it (Ctrl+B) and use ~20 steps, CFG 5, sampler uni_pc.
5. RIFE x{RIFE_X} doubles the frames before Re-pixelate; Ctrl+B the yellow RIFE node to skip it.
Saved to output/pixel/: sprite_from_prompt_vace_*.png (sheet), .json (Aseprite), .webp, _raw.mp4."""
    node(35, "Note", [2640, 900], [560, 420], widgets=[note], title="READ ME", props={})
    g.save(WORKFLOWS / "pixel-art" / "sprite-from-prompt-vace.json", "pixel-art-sprite-from-prompt-vace", groups=[
        ("Models", [-20, -60, 460, 600], "#3f789e"),
        ("Sprite + prompt + loop control", [440, -60, 1020, 1360], "#a1309b"),
        ("Animate (Wan 2.1 VACE 1.3B)", [1460, -60, 800, 960], "#88A"),
        ("Back to pixel art", [2620, -60, 1300, 960], "#444"),
    ], scale=0.5, offset=(40, 80))


# --- 3. Wan 2.1 VACE 1.3B: pose skeletons from a driving video ------------------------------------------------------
SDPOSE = "sdpose_wholebody_fp16.safetensors"
SDPOSE_URL = "https://huggingface.co/Comfy-Org/SDPose/resolve/main/checkpoints/" + SDPOSE


def build_vace_pose():
    c = POSE
    g = Graph()
    node, link = g.node, g.link
    vace_head(g, c, "CausVid LoRA (8 steps, CFG 1; non-commercial)")
    prompts(g, POSE_PROMPT)
    node(15, "VHS_LoadVideo", [460, 860], [340, 520],
         outputs=[out("IMAGE", "IMAGE"), out("frame_count", "INT"), out("audio", "AUDIO"), out("video_info", "VHS_VIDEOINFO")],
         inputs=[inp("meta_batch", "VHS_BatchManager"), inp("vae", "VAE")],
         widgets={"video": "example.mp4", "force_rate": c["fps"], "custom_width": 0, "custom_height": 0,
                  "frame_load_cap": c["frames"], "skip_first_frames": 0, "select_every_nth": 1, "format": "Wan",
                  "videopreview": {"hidden": False, "paused": False, "params": {}}},
         title="Driving video", color=BLUE, props={"Node name for S&R": "VHS_LoadVideo", "cnr_id": "comfyui-videohelpersuite"})
    node(13, "ComfyMathExpression", [840, 400], [340, 100], inputs=[inp("values.a", "FLOAT,INT,BOOLEAN")],
         outputs=[out("FLOAT", "FLOAT"), out("INT", "INT"), out("BOOL", "BOOLEAN")],
         widgets=[f"min({c['frames']}, 4 * floor((a - 1) / 4) + 1)"], title="Frames (4n+1)")
    node(17, "CheckpointLoaderSimple", [840, 540], [340, 98], outputs=[out("MODEL", "MODEL"), out("CLIP", "CLIP"), out("VAE", "VAE")],
         widgets=[SDPOSE], title="SDPose model",
         props={"Node name for S&R": "CheckpointLoaderSimple", "models": [{"name": SDPOSE, "url": SDPOSE_URL, "directory": "checkpoints"}]})
    node(18, "SDPoseKeypointExtractor", [840, 680], [340, 120],
         inputs=[inp("model", "MODEL"), inp("vae", "VAE"), inp("image", "IMAGE")], outputs=[out("POSE_KEYPOINT", "POSE_KEYPOINT")],
         widgets=[8], title="Find keypoints")
    # draw_body, draw_hands, draw_face, draw_feet, stick_width, face_point_size, score_threshold, draw_head
    node(19, "SDPoseDrawKeypoints", [840, 840], [340, 260], inputs=[inp("keypoints", "POSE_KEYPOINT")],
         outputs=[out("IMAGE", "IMAGE")], widgets=[True, True, False, True, 4, 3, 0.3, True], title="Draw skeletons")
    node(23, "ResizeAndPadImage", [840, 1140], [340, 130],
         inputs=[inp("image", "IMAGE"), inp("target_width", "INT", "target_width"), inp("target_height", "INT", "target_height")],
         outputs=[out("IMAGE", "IMAGE")], widgets=[512, 512, "black", "area"], title="Skeletons to sprite canvas")
    link(15, 1, 13, 0)
    link(17, 0, 18, 0)
    link(17, 2, 18, 1)
    link(15, 0, 18, 2)
    link(18, 0, 19, 0)
    link(19, 0, 23, 0)
    link(12, 2, 23, 1)
    link(12, 3, 23, 2)
    # control video: the sprite as frame 1 (kept, mask 0), then skeletons 2..n (generated, mask 1)
    node(24, "ImageFromBatch", [1240, 860], [200, 82], inputs=[inp("image", "IMAGE"), inp("length", "INT", "length")],
         outputs=[out("IMAGE", "IMAGE")], widgets=[1, 4096], title="Skeletons 2..n")
    node(25, "ImageBatch", [1240, 980], [200, 46], inputs=[inp("image1", "IMAGE"), inp("image2", "IMAGE")],
         outputs=[out("IMAGE", "IMAGE")], title="Sprite + skeletons")
    node(14, "ComfyMathExpression", [1240, 1060], [200, 100], inputs=[inp("values.a", "FLOAT,INT,BOOLEAN")],
         outputs=[out("FLOAT", "FLOAT"), out("INT", "INT"), out("BOOL", "BOOLEAN")], widgets=["a - 1"], title="Frames - 1")
    for nid, value, y in ((55, 0.0, 1180), (57, 1.0, 1280)):
        node(nid, "SolidMask", [1060, y], [160, 82], inputs=[inp("width", "INT", "width"), inp("height", "INT", "height")],
             outputs=[out("MASK", "MASK")], widgets=[value, 512, 512])
        node(nid + 1, "MaskToImage", [1240, y], [160, 26], inputs=[inp("mask", "MASK")], outputs=[out("IMAGE", "IMAGE")])
        link(12, 2, nid, 0)
        link(12, 3, nid, 1)
        link(nid, 0, nid + 1, 0)
    node(59, "RepeatImageBatch", [1240, 1380], [200, 58], inputs=[inp("image", "IMAGE"), inp("amount", "INT", "amount")],
         outputs=[out("IMAGE", "IMAGE")], widgets=[c["frames"] - 1])
    node(60, "ImageBatch", [1240, 1460], [200, 46], inputs=[inp("image1", "IMAGE"), inp("image2", "IMAGE")],
         outputs=[out("IMAGE", "IMAGE")])
    node(62, "ImageToMask", [1240, 1540], [200, 58], inputs=[inp("image", "IMAGE")], outputs=[out("MASK", "MASK")],
         widgets=["red"])
    link(23, 0, 24, 0)
    link(14, 1, 24, 1)
    link(13, 1, 14, 0)
    link(12, 0, 25, 0)
    link(24, 0, 25, 1)
    link(58, 0, 59, 0)
    link(14, 1, 59, 1)
    link(56, 0, 60, 0)
    link(59, 0, 60, 1)
    link(60, 0, 62, 0)
    vace_node(g, [1480, 0])
    link(25, 0, 22, 3)
    link(62, 0, 22, 4)
    link(12, 0, 22, 5)                               # reference image = the prepared sprite
    link(13, 1, 22, 8)
    vace_tail(g, c, 1920, "pixel/sprite_from_video_vace", loop=False, tag="anim", height_match=True)
    note = f"""Pixel sprite + driving video -> sprite sheet (Wan 2.1 VACE 1.3B pose control, fast) - EXPERIMENTAL

Tested 24 Oct 2026: with full-size human skeletons VACE draws a human in the sprite's colours. Scale the
skeletons to the sprite before relying on it.

1. Pixel sprite (blue): native pixel art; output keeps its pixel size + palette. It is VACE's reference image.
2. Driving video (blue): one person doing the motion, whole body visible, static camera. Read at {c['fps']} fps, up to
   {c['frames']} frames. SDPose turns it into skeletons, padded onto the sprite's canvas.
3. VACE draws the sprite following the skeletons. {c['steps']} steps, CFG {c['cfg']}, CausVid LoRA
   (CC-BY-NC-4.0, non-commercial: bypass it and use ~20 steps, CFG 5, uni_pc for a commercial build).
4. Re-pixelate "height_match" scales the clip once so the first frame has the sprite's height.
5. RIFE x{RIFE_X} doubles the frames before Re-pixelate; Ctrl+B the yellow RIFE node to skip it.
Saved to output/pixel/: sprite_from_video_vace_*.png (sheet), .json (Aseprite), .webp, _raw.mp4."""
    node(35, "Note", [2640, 900], [560, 420], widgets=[note], title="READ ME", props={})
    g.save(WORKFLOWS / "pixel-art" / "sprite-from-video-vace.json", "pixel-art-sprite-from-video-vace", groups=[
        ("Models", [-20, -60, 460, 600], "#3f789e"),
        ("Sprite + driving video -> skeletons", [440, -60, 1020, 1460], "#a1309b"),
        ("Animate (Wan 2.1 VACE 1.3B)", [1460, -60, 800, 960], "#88A"),
        ("Back to pixel art", [2620, -60, 1300, 960], "#444"),
    ], scale=0.5, offset=(40, 80))


if __name__ == "__main__":
    build_ti2v()
    build_vace_loop()
    build_vace_pose()
