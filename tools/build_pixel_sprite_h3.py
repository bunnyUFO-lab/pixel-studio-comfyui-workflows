"""Generates workflows/pixel-art/sprite-from-prompt-h3.json.

Edit the constants below and re-run:  python tools/build_pixel_sprite_h3.py

Pixel sprite + a description of the action -> animated sprite sheet at the sprite's own scale and palette, with
MiniMax H3 (first-and-last-frame model, 8-step turbo LoRA) as the video model. Same shape as sprite-from-prompt (Wan)
and sprite-from-prompt-ltx25 (same inputs by title), with the recipe the h3-match session landed on 15 Nov 2026
(research/sprint2/h3match, card "h3-match-pixellab"; hand-over h3match-recipe.md):
- Prepare: the WHOLE sprite canvas enlarged x8 (96 -> 768) with nearest neighbour, on the neutral grey or white farthest
  from every sprite colour (never a magenta or green key: both bled into the body; x2 was mushy). x8 is the default since
  the x8-v1 round (16 Nov 2026, owner 9 to 10 on five of six sets; the x4 clips scored 7 to 9 and were called blurry):
  x4 smears interior detail (the first frame came back 75% correct at x4, 98.7% at x8). x4 stays one input value away.
- H3: 39 video frames (17k+5 grid; 22 is too short for a strike and a return), 8 turbo steps, res_multistep, no CFG,
  the action caption alone as the prompt; the last frame = the first for loops (walk, idle).
- PickEvenFrames keeps 13 frames at round(i * 38 / 12), the first being the input sprite.
- CleanVideoFrames (a port of h3match/_tools/clean.py, pixel-identical on its test clips), set to the x8-v1 recipe: each
  art pixel is the video pixel at the centre of its block (background by block majority), key-coloured regions cut only
  from 4 pixels up, one pass trimming blend pixels off the silhouette, effect clusters under 3 pixels are not effects,
  palette snap with up to four effect colours, no despeckle (it halved eyes and studs at x8), outline; then the x8-v3
  passes (round 6: 9.83 against 9.0 for x8-v1 on the same video frames): a part-following hold and the feet-band rule.
About 90 s a clip at x8 on the local card (42 to 89 s on a warm cloud GPU), about 55 s at x4; peak 21.7 GB. The colour hold of PR 28 is not used: it flattened an impact frame to orange. HoldFigure
is not used either: it shifts and scales raw frames by whole video pixels, which breaks the block grid the clean-up
votes on.
Licence: MiniMax H3 Community Licence Agreement ("license: other" on the model card).
"""
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

# whole canvas enlarged by FACTOR (target_size is only used when FACTOR is 0), video frames, sprite frames kept
FACTOR, TARGET_SIZE, PADDING = 8, 768, 0.0
LENGTH = 39                  # video frames at 24 fps on the model's 17k+5 grid (5, 22, 39, ...): 22 is too short
FRAMES = 13                  # sprite frames kept, the first being the input sprite: round(i * 38 / 12)
FPS = 12                     # playback of the kept frames
STEPS = 8
SEED = 6305

# bg_tolerance, effect_colours, effect_threshold, despeckle, outline, first_is_sprite, then the x8-v1 options:
# pick (centre), min_hole, edge_trim, effect_min, then the x8-v3 passes: follow_hold, its radius, mismatch and margin,
# feet_band. The node's own defaults (vote, 1, 0, 1, despeckle on, both passes off) are the older x4 recipe; this graph
# sets the x8-v3 values itself.
CLEAN = [60.0, 4, 40.0, False, True, True, "centre", 4, 1, 3, True, 3, 0.5, 40.0, 8]
# frames, filename_prefix, frame_step, max_frames, loop_trim, columns, spacing, trim, fps, tag, preview_scale
SHEET = ["pixel/sprite_from_prompt_h3", 1, 0, False, 0, 0, True, FPS, "action", 4]

NOTE = f"""Pixel sprite + action description -> animated sprite sheet, with MiniMax H3

Recipe of the h3-match session (15 Nov 2026) at the x8-v3 settings (17 Nov 2026): about 90 s a clip at x8 on the local card, 42 to 89 s on a warm cloud GPU, peak VRAM about 21.7 GB.
1. Load your sprite (blue): native pixel art on a transparent or plain background (an upscaled copy is fine: its
   grid is detected). Output art pixels, size and palette stay exactly the sprite's.
2. Describe the action (green). The prompt is the ACTION text alone (switch "Use character and pose in the prompt" off):
   the sprite is the model's first frame, so it does not need to be told what the character looks like. Structured
   layouts (character, pose and action in one prompt) made walks stop after one or two steps.
   Write the action as BEATS, two or three plain sentences, 30 to 50 words, imperative: wind-up, strike, impact, settle
   ("Raise the hammer overhead with a quick, forceful arc, then slam the head down into the ground with heavy momentum.
   Pause momentarily for impact weight, then heave the hammer back up and settle it onto the shoulder to reset.").
   PixelLab's prompt fixer writes text like this from a short ask; it is not part of this workflow (a later PR can add
   one with ComfyUI's TextGenerate), so write the beats or let the app do it.
   The LAST sentence matters: "return to the initial stance" makes a walk stop after one or two steps. For a walk or an
   idle end on something that keeps the motion going. The "Action (written as beats)" input fits attacks as written; for loops write the repeating motion.
3. Prepare: the WHOLE canvas enlarged x{FACTOR} (nearest neighbour), on the neutral grey or white farthest from the sprite's
   colours. Not a crop of the sprite plus padding, and never a magenta or green key (both bled into the body); x2 was
   mushy. The enlargement is the "factor" input of the yellow Prepare node (whole numbers, 4 to 8).
   x{FACTOR} is the default because x4 smears interior detail: the first frame, which should be the sprite itself, came back
   75% correct at x4 and 98.7% at x8, and the owner rated the x8 clips 9 to 10 and the x4 clips 7 to 9 ("blurry").
   x4 stays available and takes about half the time (about 55 s against about 90 s locally, 42 to 89 s at x8 on a warm
   cloud GPU): set "factor" to 4 and, if you like, the clean-up's pick to vote, despeckle on, min_hole 1, edge_trim 0,
   effect_min 1, follow_hold off and feet_band 0 (the node's own defaults). It was the clip LENGTH that decided whether a walk worked, not the size: at
   {LENGTH} frames the dwarf walks at x4, x6, x8 and x10 alike, landing back on frame 1; 124 frames failed from drift and
   dead holds. Keep the canvas under about 1.1 megapixels (1440x1248 peaked at 23.9 GB, so a 128 px sprite fits at x8
   with no padding, a bigger one needs a lower factor). Sprites above 128 px were not tried.
4. The sprite is the first frame AND the last frame (a loop: walk, idle), like the Wan and LTX workflows. For a one-shot
   action (attack, cast, smash) disconnect "last_frame" on the yellow "Loop" node, as the app does.
5. {LENGTH} video frames at 24 fps (the length snaps to 17k+5: 22 is too short for a strike and a return), 8 turbo steps,
   res_multistep, no CFG. 20 steps bought nothing. Seed in "Noise". "Frames to keep" takes {FRAMES} frames across the clip,
   the first being the input sprite.
6. Clean up (yellow), the x8-v3 settings (x8-v1 plus two passes, described after the list): each art pixel is the video pixel at the centre of its {FACTOR} x {FACTOR} block
   (the block's majority decides background or body); key-coloured regions are transparent only from 4 pixels up
   (smaller ones are body: a blue-grey arm band on a grey key); one pass removes silhouette pixels nearer the key than
   any sprite colour; each pixel is snapped to the sprite's palette; a pixel far from every sprite colour is an effect
   (spark, trail, flash) and keeps one of at most 4 effect colours, but a cluster under 3 pixels is not an effect; no
   despeckle (it halved eyes and studs at x8); a one-colour outline is redrawn; frame 1 is the input sprite itself.
   Then two passes on the finished frames: the hold (follow_hold) gives a pixel the colour its part had in the sprite or the
   frame before (found by matching a 5 x 5 patch up to 3 pixels away) when the video agrees, which stops colours churning on
   a moving part; the feet-band rule (feet_band 8) removes a group of effect colours that reaches the bottom 8 rows (a splat
   on the ground) unless its colours also show higher up in the clip (a trail or a cast). For x4, turn both off.
   Not used: the colour hold of PR 28 (it flattened an impact frame with a white slash to orange) and Hold Figure (it
   shifts raw frames by whole video pixels, which breaks the block grid).
Known gap: effects come out as video-style crescents and swooshes, not thin pixel lines; the model sometimes turns
towards the camera; some single-pixel noise remains.
Models (~36 GB): H3 first-last-frame w6a8 16 GB, Qwen3-VL 32B encoder 15.7 GB, video VAE, 8-step turbo LoRA.
Licence: MiniMax H3 Community Licence Agreement. Do not train on H3 output without asking the owner."""

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
     widgets=[TARGET_SIZE, PADDING, "auto", 32, "whole canvas", FACTOR], title="Prepare sprite (integer upscale)", color=YELLOW)
node(13, "PreviewAny", [460, 730], [340, 120], inputs=[inp("source", "*")], title="Sprite info")

# The action, in the recipe's three parts
node(20, "PrimitiveStringMultiline", [840, 0], [420, 110], outputs=[out("STRING", "STRING")], widgets=[CHARACTER],
     title="Character (one sentence)", color=GREEN)
node(21, "PrimitiveStringMultiline", [840, 150], [420, 90], outputs=[out("STRING", "STRING")], widgets=[POSE],
     title="Starting pose", color=GREEN)
node(22, "PrimitiveStringMultiline", [840, 280], [420, 170], outputs=[out("STRING", "STRING")], widgets=[ACTION],
     title="Action (written as beats)", color=GREEN)
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

# Back to pixel art: pick the frames, then clean them up (the first picked frame is the input sprite)
node(50, "PickEvenFrames", [1680, 0], [340, 106], inputs=[inp("frames", "IMAGE")],
     outputs=[out("frames", "IMAGE"), out("info", "STRING")], widgets=[FRAMES, False],
     title="Frames to keep (across the whole clip)", color=GREEN)
node(51, "CleanVideoFrames", [1680, 150], [340, 250], inputs=[inp("frames", "IMAGE"), inp("pixel_spec", "PIXEL_SPEC")],
     outputs=[out("pixel_frames", "IMAGE"), out("info", "STRING"), out("palette", "STRING")], widgets=CLEAN,
     title="Clean up (block vote, palette, effect colours)", color=YELLOW)
node(53, "SaveSpriteSheet", [1680, 440], [340, 330], inputs=[inp("frames", "IMAGE")],
     outputs=[out("sheet", "IMAGE"), out("frames", "IMAGE"), out("info", "STRING"), out("fps", "INT")], widgets=SHEET,
     title="Save sprite sheet", color=GREEN)
node(54, "PixelArtCheck", [2060, 0], [340, 106], inputs=[inp("images", "IMAGE")],
     outputs=[out("report", "STRING"), out("passed", "BOOLEAN")], widgets=[256, 0, False], title="Pixel Art Check")
node(55, "PreviewAny", [2060, 150], [340, 200], inputs=[inp("source", "*")], title="Check report")
node(56, "PreviewAny", [2060, 390], [340, 160], inputs=[inp("source", "*")], title="Sheet info")
node(57, "PreviewAny", [2060, 600], [340, 160], inputs=[inp("source", "*")], title="Clean-up info (colours, popping)")
node(58, "PreviewAny", [2060, 800], [340, 160], inputs=[inp("source", "*")], title="Frames kept")
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
link(50, 1, 58, 0)
link(50, 0, 51, 0)
link(12, 1, 51, 1)
link(51, 0, 53, 0)
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
