"""Generates workflows/pixel-art/sprite-from-prompt-h3.json.

Edit the constants below and re-run:  python tools/build_pixel_sprite_h3.py

Pixel sprite + a description of the action -> animated sprite sheet at the sprite's own scale and palette, with
MiniMax H3 (first-and-last-frame model, 8-step turbo LoRA) as the video model. Same shape as sprite-from-prompt (Wan)
and sprite-from-prompt-ltx25: PrepareSpriteForVideo upscales the sprite by an integer factor onto a flat background,
the video model animates it, HoldFigure keeps the figure at one height / ground line / centre, RepixelateFrames brings
every frame back onto exactly the sprite's grid and palette, PickEvenFrames keeps N frames across the clip and
SaveSpriteSheet writes the sheet, the Aseprite JSON and an animated WebP.

H3 is wired like ComfyUI's own template "Image to Video (MiniMax H3)" (video_minimax_h3_i2v), flattened: the sprite is
the first frame, the audio branch is dropped, 8 turbo steps with res_multistep, BasicGuider (no CFG).
Licence: MiniMax H3 Community Licence Agreement ("license: other" on the model card).
"""
from build_pixel_sprite_fast import hold_figure_node
from comfy_workflow import BLUE, GREEN, WORKFLOWS, YELLOW, Graph, inp, out

OUT = WORKFLOWS / "pixel-art" / "sprite-from-prompt-h3.json"

H3_HF = "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main"
UNET = "minimax-h3/minimax_h3_fl2va_pruned_w6a8.safetensors"                    # 16.0 GB
TE = "minimax-h3/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"                  # 15.7 GB
VAE = "minimax-h3/minimax_h3_video_vae_int8_convrot.safetensors"                # 2.8 GB
LORA = "minimax-h3/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"   # 2.0 GB
URLS = {UNET: f"{H3_HF}/diffusion_models/minimax_h3_fl2va_pruned_w6a8.safetensors",
        TE: f"{H3_HF}/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        VAE: f"{H3_HF}/vae/minimax_h3_video_vae_int8_convrot.safetensors",
        LORA: f"{H3_HF}/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"}

# The owner's recipe, in three inputs: what the character is, the starting pose, the action. Defaults are the
# dwarf's hammer smash; the action is the text PixelLab's prompt fixer wrote for it (beats: wind-up, strike, impact,
# settle). The fixer is not available here: the app writes the action text in this form.
CHARACTER = ("a stocky dwarf blacksmith with a braided red beard, a brown leather apron and a heavy steel hammer")
POSE = "standing upright facing right with the hammer resting on his shoulder"
ACTION = ("Raise the hammer overhead with a quick, forceful arc, then slam the head down into the ground with heavy "
          "momentum. Pause momentarily for impact weight, then heave the hammer back up and settle it onto the "
          "shoulder to reset.")
USE_CHARACTER = False        # False = the action text alone (the form that was tested); True = character, pose, action

# canvas (longest side, whole art pixels), room around the sprite for motion, video frames, sheet frames
TARGET_SIZE, PADDING = 1024, 0.5
LENGTH = 124                 # video frames at 24 fps, on the model's 17k+5 grid (124 = ~5 s, the shortest trained)
FRAMES = 12                  # sprite frames kept across the clip
FPS = 12                     # playback of the kept frames (12 frames of a 5 s action: the hosted clips play at 12)
STEPS = 8
SEED = 6305
HOLD_STRENGTH = 0.7

# height_match, bg_tolerance, method, colors (0 = sprite palette), palette, stability, deflicker, cleanup, outline,
# hold_colors. hold_colors is OFF: on H3 attacks it turned an impact frame with a white slash effect flat orange.
REPIX = [False, 18.0, "dominant", 0, "sprite", 0.7, True, True, "none", False]
# frames, filename_prefix, frame_step, max_frames, loop_trim, columns, spacing, trim, fps, tag, preview_scale
SHEET = ["pixel/sprite_from_prompt_h3", 1, 0, False, 0, 0, True, FPS, "action", 4]

NOTE = f"""Pixel sprite + action description -> animated sprite sheet, with MiniMax H3

1. Load your sprite (blue): native pixel art on a transparent or plain background (an upscaled copy is fine: its
   grid is detected). Output art pixels, size and palette stay exactly the sprite's.
2. Describe the action (green), the way the owner's recipe does:
   - Character: one sentence saying what it is ("a stocky dwarf blacksmith with a braided red beard...").
   - Starting pose: "standing upright facing right with the hammer resting on his shoulder".
   - Action: two or three plain sentences written as BEATS: wind-up, strike, impact, settle. PixelLab's prompt fixer
     rewrites an ask like this; it is not available here, so write the beats yourself or let the app do it.
   The form that was tested is the ACTION text alone (switch "Use character and pose in the prompt" off): the model
   sees the sprite as its first frame, so it does not need to be told what the character looks like. Switch it on to
   put character, pose and action in one prompt (not tested).
3. The sprite is the first frame AND the last frame (a loop: idle, walk), like the Wan and LTX workflows. For a one-shot
   action (attack, cast, smash) disconnect "last_frame" on the yellow "Loop" node, as the app does. The action clips
   that were tested ran with the first frame only; a walk was tested both ways.
4. Video frames {LENGTH} at 24 fps (~5 s, the shortest the model was trained on; the length snaps to 17k+5). Frames kept
   = how many sprite frames to keep, spread across the whole clip. 8 turbo steps, no CFG. Seed in "Noise".
5. Canvas size matters. target_size {TARGET_SIZE} gives about 960x1120 for a 52x59 sprite: colours and the figure stay clean
   (a 480x640 canvas gave colour speckle and a hammer floating off the shoulder on a walk). Peak VRAM 21.4 GB of 24
   at 960x1120, 535 s; 1440x1248 peaked at 23.9 GB: keep the canvas under about 1.1 megapixels.
6. Hold figure (yellow, on) puts the figure at the clip's median height, ground line and centre on the raw frames.
   Re-pixelate then snaps every frame to the sprite's grid and palette. The colour hold is off here (hold_colors): it
   flattened an impact frame with a white slash effect. Turn it on if the clip's colours drift paler.
Good for action clips (attack, cast, smash). Walks work only on a big canvas and run livelier than the hosted tool's.
Models (~36 GB): H3 first-last-frame w6a8 16 GB, Qwen3-VL 32B encoder 15.7 GB, video VAE, 8-step turbo LoRA.
Licence: MiniMax H3 Community Licence Agreement."""

g = Graph()
node, link = g.node, g.link


def model_props(type_, name, directory):
    return {"Node name for S&R": type_, "models": [{"name": name, "url": URLS[name], "directory": directory}]}


# Models
node(1, "UNETLoader", [0, 0], [420, 82], outputs=[out("MODEL", "MODEL")], widgets=[UNET, "default"],
     title="Load H3 first-last-frame model", props=model_props("UNETLoader", UNET, "diffusion_models"))
node(2, "CLIPLoader", [0, 130], [420, 106], outputs=[out("CLIP", "CLIP")], widgets=[TE, "minimax", "default"],
     title="Load Qwen3-VL text encoder", props=model_props("CLIPLoader", TE, "text_encoders"))
node(3, "VAELoader", [0, 280], [420, 58], outputs=[out("VAE", "VAE")], widgets=[VAE], title="Load H3 video VAE",
     props=model_props("VAELoader", VAE, "vae"))
node(4, "LoraLoaderModelOnly", [0, 380], [420, 82], inputs=[inp("model", "MODEL")], outputs=[out("MODEL", "MODEL")],
     widgets=[LORA, 1.0], title="Turbo 8-step LoRA", props=model_props("LoraLoaderModelOnly", LORA, "loras"))

# Sprite
node(10, "LoadImage", [460, 0], [340, 400], outputs=[out("IMAGE", "IMAGE"), out("MASK", "MASK")],
     widgets=["example.png", "image"], title="Pixel sprite", color=BLUE)
node(11, "JoinImageWithAlpha", [460, 440], [340, 46], inputs=[inp("image", "IMAGE"), inp("alpha", "MASK")],
     outputs=[out("IMAGE", "IMAGE")], title="Keep sprite transparency")
node(12, "PrepareSpriteForVideo", [460, 530], [340, 160], inputs=[inp("sprite", "IMAGE")],
     outputs=[out("image", "IMAGE"), out("pixel_spec", "PIXEL_SPEC"), out("width", "INT"), out("height", "INT"),
              out("info", "STRING")],
     widgets=[TARGET_SIZE, PADDING, "auto", 32], title="Prepare sprite (integer upscale)", color=YELLOW)
node(13, "PreviewAny", [460, 730], [340, 120], inputs=[inp("source", "*")], title="Sprite info")

# The action, in the recipe's three parts
node(20, "PrimitiveStringMultiline", [840, 0], [420, 110], outputs=[out("STRING", "STRING")], widgets=[CHARACTER],
     title="Character (one sentence)", color=GREEN)
node(21, "PrimitiveStringMultiline", [840, 150], [420, 90], outputs=[out("STRING", "STRING")], widgets=[POSE],
     title="Starting pose", color=GREEN)
node(22, "PrimitiveStringMultiline", [840, 280], [420, 170], outputs=[out("STRING", "STRING")], widgets=[ACTION],
     title="Action (beats: wind-up, strike, impact, settle)", color=GREEN)
node(23, "StringConcatenate", [840, 490], [420, 130],
     inputs=[inp("string_a", "STRING", "string_a"), inp("string_b", "STRING", "string_b")],
     outputs=[out("STRING", "STRING")], widgets=["", "", ", "], title="Character, pose")
node(24, "StringConcatenate", [840, 660], [420, 130],
     inputs=[inp("string_a", "STRING", "string_a"), inp("string_b", "STRING", "string_b")],
     outputs=[out("STRING", "STRING")], widgets=["", "", ". "], title="Character, pose. Action")
node(25, "ComfySwitchNode", [840, 830], [300, 82], inputs=[inp("on_false", "STRING"), inp("on_true", "STRING")],
     outputs=[out("output", "STRING")], widgets=[USE_CHARACTER], title="Use character and pose in the prompt")
node(26, "PreviewAny", [840, 950], [420, 160], inputs=[inp("source", "*")], title="Prompt used")

# H3: the sprite is the first frame
node(30, "MiniMaxH3ImageToVideo", [1300, 0], [340, 250],
     inputs=[inp("clip", "CLIP"), inp("vae", "VAE"), inp("prompt", "STRING", "prompt"),
             inp("width", "INT", "width"), inp("height", "INT", "height"),
             inp("first_frame", "IMAGE"), inp("last_frame", "IMAGE")],
     outputs=[out("positive", "CONDITIONING"), out("latent", "LATENT")],
     widgets=["", TARGET_SIZE, TARGET_SIZE, LENGTH], title="Loop (first = last frame) / video frames", color=YELLOW)
node(40, "RandomNoise", [1300, 300], [300, 82], outputs=[out("NOISE", "NOISE")], widgets=[SEED, "randomize"],
     title="Noise (seed here)")
node(41, "KSamplerSelect", [1300, 420], [300, 58], outputs=[out("SAMPLER", "SAMPLER")], widgets=["res_multistep"])
node(42, "BasicScheduler", [1300, 520], [300, 106], inputs=[inp("model", "MODEL")], outputs=[out("SIGMAS", "SIGMAS")],
     widgets=["simple", STEPS, 1.0], title="Scheduler (8 turbo steps)")
node(43, "BasicGuider", [1300, 670], [300, 46], inputs=[inp("model", "MODEL"), inp("conditioning", "CONDITIONING")],
     outputs=[out("GUIDER", "GUIDER")])
node(44, "SamplerCustomAdvanced", [1300, 760], [300, 120],
     inputs=[inp("noise", "NOISE"), inp("guider", "GUIDER"), inp("sampler", "SAMPLER"), inp("sigmas", "SIGMAS"),
             inp("latent_image", "LATENT")], outputs=[out("output", "LATENT"), out("denoised_output", "LATENT")],
     title="Sampler")
node(45, "VAEDecode", [1300, 920], [210, 46], inputs=[inp("samples", "LATENT"), inp("vae", "VAE")],
     outputs=[out("IMAGE", "IMAGE")], title="Decode (video only)")
node(46, "VHS_VideoCombine", [1300, 1010], [320, 420],
     inputs=[inp("images", "IMAGE"), inp("audio", "AUDIO"), inp("meta_batch", "VHS_BatchManager"), inp("vae", "VAE")],
     outputs=[out("Filenames", "VHS_FILENAMES")],
     widgets={"frame_rate": 24, "loop_count": 0, "filename_prefix": "pixel/sprite_from_prompt_h3_raw",
              "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 17, "save_metadata": True,
              "trim_to_audio": False, "pingpong": False, "save_output": True,
              "videopreview": {"hidden": False, "paused": False, "params": {}}},
     title="Raw video (motion check)", props={"Node name for S&R": "VHS_VideoCombine", "cnr_id": "comfyui-videohelpersuite"})

# Back to pixel art
hold_figure_node(g, 50, [1680, 0], 45, 12, True, HOLD_STRENGTH)
node(51, "RepixelateFrames", [1680, 150], [340, 250], inputs=[inp("frames", "IMAGE"), inp("pixel_spec", "PIXEL_SPEC")],
     outputs=[out("pixel_frames", "IMAGE"), out("info", "STRING"), out("palette", "STRING")], widgets=REPIX,
     title="Repixelate (same grid + palette)", color=YELLOW)
node(52, "PickEvenFrames", [1680, 440], [340, 106], inputs=[inp("frames", "IMAGE")],
     outputs=[out("frames", "IMAGE"), out("info", "STRING")], widgets=[FRAMES, True],
     title="Frames to keep (across the whole clip)", color=GREEN)
node(53, "SaveSpriteSheet", [1680, 590], [340, 330], inputs=[inp("frames", "IMAGE")],
     outputs=[out("sheet", "IMAGE"), out("frames", "IMAGE"), out("info", "STRING"), out("fps", "INT")], widgets=SHEET,
     title="Save sprite sheet", color=GREEN)
node(54, "PixelArtCheck", [2060, 0], [340, 106], inputs=[inp("images", "IMAGE")],
     outputs=[out("report", "STRING"), out("passed", "BOOLEAN")], widgets=[256, 0, False], title="Pixel Art Check")
node(55, "PreviewAny", [2060, 150], [340, 200], inputs=[inp("source", "*")], title="Check report")
node(56, "PreviewAny", [2060, 390], [340, 160], inputs=[inp("source", "*")], title="Sheet info")
node(57, "PreviewAny", [2060, 600], [340, 160], inputs=[inp("source", "*")], title="Re-pixelate info (colours, popping)")
node(58, "PreviewAny", [2060, 800], [340, 160], inputs=[inp("source", "*")], title="Hold figure info")
node(59, "SpriteAnimationPreview", [2440, 0], [520, 680], inputs=[inp("frames", "IMAGE"), inp("fps", "INT", "fps")],
     widgets=[8, 6, "checker", "loop"], title="Animation preview (playback)", color=GREEN)
node(60, "Note", [0, 520], [420, 800], widgets=[NOTE], title="READ ME", props={})

link(1, 0, 4, 0)
link(10, 0, 11, 0)           # LoadImage drops alpha into MASK: join it back
link(10, 1, 11, 1)
link(11, 0, 12, 0)
link(12, 4, 13, 0)
link(20, 0, 23, 0)
link(21, 0, 23, 1)
link(23, 0, 24, 0)
link(22, 0, 24, 1)
link(22, 0, 25, 0)           # switch off: the action alone
link(24, 0, 25, 1)           # switch on: character, pose and action
link(25, 0, 26, 0)
link(25, 0, 30, 2)
link(2, 0, 30, 0)
link(3, 0, 30, 1)
link(12, 2, 30, 3)
link(12, 3, 30, 4)
link(12, 0, 30, 5)           # first frame = the sprite
link(12, 0, 30, 6)           # last frame = the sprite (loop); the app unlinks "last_frame" for a one-shot action
link(4, 0, 42, 0)
link(4, 0, 43, 0)
link(30, 0, 43, 1)
link(40, 0, 44, 0)
link(43, 0, 44, 1)
link(41, 0, 44, 2)
link(42, 0, 44, 3)
link(30, 1, 44, 4)
link(44, 0, 45, 0)
link(3, 0, 45, 1)
link(45, 0, 46, 0)
link(45, 0, 50, 0)
link(12, 1, 50, 1)
link(50, 0, 51, 0)
link(12, 1, 51, 1)
link(50, 1, 58, 0)
link(51, 0, 52, 0)
link(52, 0, 53, 0)
link(53, 1, 54, 0)
link(54, 0, 55, 0)
link(53, 2, 56, 0)
link(51, 1, 57, 0)
link(53, 1, 59, 0)           # the sheet's frames + fps -> playback
link(53, 3, 59, 1)

g.save(OUT, "pixel-art-sprite-from-prompt-h3", groups=[
    ("Models", [-20, -60, 460, 560], "#3f789e"),
    ("Sprite + action", [440, -60, 840, 1200], "#a1309b"),
    ("Animate (MiniMax H3, first frame = sprite)", [1280, -60, 380, 1480], "#88A"),
    ("Back to pixel art", [1660, -60, 760, 1000], "#444"),
], scale=0.5, offset=(40, 80))
