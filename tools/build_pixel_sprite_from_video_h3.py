"""Generates workflows/pixel-art/sprite-from-video-h3.json.

Edit the constants below and re-run:  python tools/build_pixel_sprite_from_video_h3.py

The MiniMax H3 arm of pixel-art/sprite-from-video (same inputs and outputs as the Wan Animate 2 and SCAIL 2 versions,
so the three can be compared clip for clip): PrepareSpriteForVideo upscales the sprite by an integer factor onto a flat
background (sizes divisible by 32), MiniMax H3 Ref2VA (core MiniMaxH3ReferenceToVideo) gets the sprite as <Picture 1>
and the driving video as <Video 1> ("performance only"), and an optional second view (e.g. the back from
image-to-directions) as <Picture 2>. Then RepixelateFrames (height match on) brings the clip back onto exactly the same
grid with the sprite's exact palette. H3 makes audio too; it is dropped. Settings follow Comfy's video_minimax_h3_r2v
template (20 steps, res_multistep, simple). 124 frames = ~5 s at 24 fps.

Licence: MiniMax H3's community licence excludes the US, EU, UK and South Korea; local use there needs a MiniMax
commercial licence.
"""
from comfy_workflow import (BLUE, GREEN, H3_AUDIO_VAE, H3_MODEL, H3_TE, H3_VAE, URLS, WORKFLOWS, YELLOW, Graph, inp,
                            out)

OUT = WORKFLOWS / "pixel-art" / "sprite-from-video-h3.json"

# H3 wants "a novel": six sections in a fixed order (card "Updating the video generator", Prompts tab; MiniMax's
# h3-prompt-writing skill). Replace the bracketed parts per sprite and clip; detailed_description should run
# 350-500 words, beat by beat, for best results.
PROMPT = """subject_definitions:
<Subject 1> is the pixel-art game character from <Picture 1>: keep its exact shape, colours, outfit and proportions.
<Subject 2> is the pixel-art rendering style of <Picture 1>: hard square pixels, a flat limited palette, thick dark outlines, no gradients, no blur, no anti-aliasing.
<Subject 3> is the performance of the person in <Video 1>: their body movement and timing.

summary:
[reference generation] <Subject 1> performs <Subject 3> in <Subject 2>, full body, centred, on the flat single-colour background of <Picture 1>, static front-on camera.

retention_analysis:
<Subject 1>: fully_preserved - shape, palette, outfit and size from <Picture 1>.
<Subject 2>: fully_preserved - the output stays pixel art in every frame.
<Subject 3>: attribute_transfer - <Video 1> supplies performance and timing only, not its photographic look, body, clothes or background.

detailed_description:
[Shot 1] A static, locked-off, full-body front view. The flat background colour of <Picture 1> fills the frame and never changes; there is no floor, no shadow, no scenery and no text. <Subject 1> stands centred at the same size as in <Picture 1>, drawn in <Subject 2> with crisp square pixels and the same few flat colours. Throughout the clip <Subject 1> copies every movement of the person in <Video 1> beat for beat and with the same timing: each step, swing, crouch, jump and arm movement happens at the same moment and to the same extent. The limbs, head and any props of <Subject 1> move as rigid pixel-art parts, keeping their outlines unbroken and their colours flat; nothing melts, blurs or smears, and no new colours, highlights or textures appear. The character keeps its own proportions instead of the person's, and never turns into a realistic human. The camera never moves, pans or zooms, and the character never leaves the frame. The clip ends with <Subject 1> back in the pose it started in.

overall_soundscape:
N/A

non_diegetic_music:
N/A"""
TARGET_SIZE, PADDING = 1024, 0.6       # H3 smears subjects under ~0.4 MP: larger than the Wan/SCAIL arms' 640
FPS, LENGTH = 24, 124                  # 17k+5 grid at 24 fps; 124 = ~5 s
STEPS, SEED = 20, 11
SHEET = ["pixel/sprite_from_video_h3", 3, 0, False, 0, 0, True, 8, "anim", 4]     # every 3rd of 24 fps = 8 fps
REPIX = [True, 18.0, "dominant", 0, "sprite", 0.7, True, True, "none"]

NOTE = f"""Pixel sprite + driving video -> sprite sheet (MiniMax H3)

The MiniMax H3 version of pixel-art/sprite-from-video: same inputs and outputs, a different motion model.
Licence: H3's community licence excludes the US, EU, UK and South Korea; use there needs a MiniMax commercial licence.
1. Pixel sprite (blue) = <Picture 1>. Optional "Second view" (unmute it, e.g. the back from image-to-directions)
   becomes <Picture 2>: name it in the prompt.
2. Driving video (blue) = <Video 1>: one person, whole body, static camera. Read at {FPS} fps, 2-15 s.
3. Prompt (green): H3 wants six sections in this order (subject_definitions, summary, retention_analysis,
   detailed_description, overall_soundscape, non_diegetic_music). Describe the clip beat by beat in
   detailed_description (350-500 words is best); keep "<Video 1> supplies performance and timing only".
4. MiniMax H3 Ref2VA, {STEPS} steps. length {LENGTH} = ~5 s (17k+5 frames at 24 fps). The sprite is upscaled to
   ~{TARGET_SIZE} px: H3 smears small subjects. The generated audio is dropped.
5. Re-pixelate + sprite sheet as in sprite-from-video; frame_step {SHEET[1]} keeps 8 fps of the 24.
Saved to output/pixel/: sprite_from_video_h3_*.png (sheet), .json (Aseprite), .webp, _preview.png,
and the raw video (pixel/sprite_from_video_h3_raw) to check the motion."""

g = Graph()
node, link = g.node, g.link


def models(name, directory, type_):
    return {"Node name for S&R": type_, "models": [{"name": name, "url": URLS[name], "directory": directory}]}


# Models
node(1, "UNETLoader", [0, 0], [420, 82], outputs=[out("MODEL", "MODEL")], widgets=[H3_MODEL, "default"],
     title="MiniMax H3 Ref2VA (w6a8)", props=models(H3_MODEL, "diffusion_models", "UNETLoader"))
node(2, "CLIPLoader", [0, 130], [420, 106], outputs=[out("CLIP", "CLIP")], widgets=[H3_TE, "minimax", "default"],
     title="Qwen3-VL 32B text encoder (NVFP4)", props=models(H3_TE, "text_encoders", "CLIPLoader"))
node(3, "VAELoader", [0, 280], [420, 58], outputs=[out("VAE", "VAE")], widgets=[H3_VAE], title="Video VAE",
     props=models(H3_VAE, "vae", "VAELoader"))
node(4, "VAELoader", [0, 380], [420, 58], outputs=[out("VAE", "VAE")], widgets=[H3_AUDIO_VAE], title="Audio VAE",
     props=models(H3_AUDIO_VAE, "vae", "VAELoader"))

# Inputs
node(10, "LoadImage", [460, 0], [340, 400], outputs=[out("IMAGE", "IMAGE"), out("MASK", "MASK")],
     widgets=["example.png", "image"], title="Pixel sprite", color=BLUE)
node(11, "JoinImageWithAlpha", [460, 440], [340, 46], inputs=[inp("image", "IMAGE"), inp("alpha", "MASK")],
     outputs=[out("IMAGE", "IMAGE")], title="Keep sprite transparency")
node(12, "PrepareSpriteForVideo", [460, 530], [340, 154], inputs=[inp("sprite", "IMAGE")],
     outputs=[out("image", "IMAGE"), out("pixel_spec", "PIXEL_SPEC"), out("width", "INT"), out("height", "INT"),
              out("info", "STRING")],
     widgets=[TARGET_SIZE, PADDING, "auto", 32], title="Prepare sprite (integer upscale, /32)", color=YELLOW)
node(16, "PreviewAny", [460, 720], [340, 120], inputs=[inp("source", "*")], title="Sprite info")
node(17, "LoadImage", [460, 880], [340, 340], outputs=[out("IMAGE", "IMAGE"), out("MASK", "MASK")],
     widgets=["example.png", "image"], title="Second view (optional, <Picture 2>)", color=BLUE, mode=2)
node(13, "VHS_LoadVideo", [840, 0], [340, 520],
     outputs=[out("IMAGE", "IMAGE"), out("frame_count", "INT"), out("audio", "AUDIO"), out("video_info", "VHS_VIDEOINFO")],
     inputs=[inp("meta_batch", "VHS_BatchManager"), inp("vae", "VAE")],
     widgets={"video": "example.mp4", "force_rate": FPS, "custom_width": 0, "custom_height": 0,
              "frame_load_cap": 15 * FPS, "skip_first_frames": 0, "select_every_nth": 1, "format": "None",
              "videopreview": {"hidden": False, "paused": False, "params": {}}},
     title="Driving video", color=BLUE, props={"Node name for S&R": "VHS_LoadVideo", "cnr_id": "comfyui-videohelpersuite"})

# MiniMax H3
node(20, "MiniMaxH3ReferenceToVideo", [1220, 0], [460, 620],
     inputs=[inp("clip", "CLIP"), inp("vae", "VAE"), inp("audio_vae", "VAE"),
             inp("ref_images.ref_image_0", "IMAGE"), inp("ref_images.ref_image_1", "IMAGE"),
             inp("ref_videos.ref_video_0", "IMAGE"),
             inp("width", "INT", "width"), inp("height", "INT", "height")],
     outputs=[out("positive", "CONDITIONING"), out("LATENT", "LATENT")],
     widgets=[PROMPT, TARGET_SIZE, TARGET_SIZE, LENGTH, "match"], title="Prompt + MiniMax H3 Ref2VA", color=GREEN)
node(21, "RandomNoise", [1720, 0], [300, 82], outputs=[out("NOISE", "NOISE")], widgets=[SEED, "randomize"],
     title="Noise (seed here)")
node(22, "KSamplerSelect", [1720, 120], [300, 58], outputs=[out("SAMPLER", "SAMPLER")], widgets=["res_multistep"])
node(23, "BasicScheduler", [1720, 220], [300, 106], inputs=[inp("model", "MODEL")], outputs=[out("SIGMAS", "SIGMAS")],
     widgets=["simple", STEPS, 1.0], color=YELLOW)
node(24, "BasicGuider", [1720, 370], [300, 46], inputs=[inp("model", "MODEL"), inp("conditioning", "CONDITIONING")],
     outputs=[out("GUIDER", "GUIDER")])
node(25, "SamplerCustomAdvanced", [2060, 0], [300, 106],
     inputs=[inp("noise", "NOISE"), inp("guider", "GUIDER"), inp("sampler", "SAMPLER"), inp("sigmas", "SIGMAS"),
             inp("latent_image", "LATENT")],
     outputs=[out("output", "LATENT"), out("denoised_output", "LATENT")])
node(28, "VAEDecode", [2060, 150], [210, 46], inputs=[inp("samples", "LATENT"), inp("vae", "VAE")],
     outputs=[out("IMAGE", "IMAGE")])
node(29, "VHS_VideoCombine", [2060, 240], [320, 420],
     inputs=[inp("images", "IMAGE"), inp("audio", "AUDIO"), inp("meta_batch", "VHS_BatchManager"), inp("vae", "VAE")],
     outputs=[out("Filenames", "VHS_FILENAMES")],
     widgets={"frame_rate": FPS, "loop_count": 0, "filename_prefix": "pixel/sprite_from_video_h3_raw",
              "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 17, "save_metadata": True,
              "trim_to_audio": False, "pingpong": False, "save_output": True,
              "videopreview": {"hidden": False, "paused": False, "params": {}}},
     title="Raw video (motion check)", props={"Node name for S&R": "VHS_VideoCombine", "cnr_id": "comfyui-videohelpersuite"})

# Back to pixel art
node(30, "RepixelateFrames", [2420, 0], [340, 250], inputs=[inp("frames", "IMAGE"), inp("pixel_spec", "PIXEL_SPEC")],
     outputs=[out("pixel_frames", "IMAGE"), out("info", "STRING"), out("palette", "STRING")], widgets=REPIX,
     title="Repixelate (height match, same grid + palette)", color=YELLOW)
node(31, "SaveSpriteSheet", [2420, 180], [340, 330], inputs=[inp("frames", "IMAGE")],
     outputs=[out("sheet", "IMAGE"), out("frames", "IMAGE"), out("info", "STRING"), out("fps", "INT")], widgets=SHEET,
     title="Save sprite sheet", color=GREEN)
node(32, "PixelArtCheck", [2800, 0], [340, 106], inputs=[inp("images", "IMAGE")],
     outputs=[out("report", "STRING"), out("passed", "BOOLEAN")], widgets=[256, 0, False], title="Pixel Art Check")
node(33, "PreviewAny", [2800, 150], [340, 200], inputs=[inp("source", "*")], title="Check report")
node(34, "PreviewAny", [2800, 390], [340, 160], inputs=[inp("source", "*")], title="Sheet info")
node(36, "PreviewAny", [2800, 600], [340, 160], inputs=[inp("source", "*")], title="Re-pixelate info (colours, popping)")
node(37, "SpriteAnimationPreview", [3180, 0], [520, 680], inputs=[inp("frames", "IMAGE"), inp("fps", "INT", "fps")],
     widgets=[8, 6, "checker", "loop"], title="Animation preview (playback)", color=GREEN)
node(35, "Note", [2420, 560], [560, 560], widgets=[NOTE], title="READ ME", props={})

link(10, 0, 11, 0)            # LoadImage drops alpha into MASK: join it back
link(10, 1, 11, 1)
link(11, 0, 12, 0)
link(12, 4, 16, 0)
link(2, 0, 20, 0)
link(3, 0, 20, 1)
link(4, 0, 20, 2)
link(12, 0, 20, 3)            # <Picture 1>
link(17, 0, 20, 4)            # <Picture 2> (muted by default)
link(13, 0, 20, 5)            # <Video 1>
link(12, 2, 20, 6)
link(12, 3, 20, 7)
link(1, 0, 23, 0)
link(1, 0, 24, 0)
link(20, 0, 24, 1)
link(21, 0, 25, 0)
link(24, 0, 25, 1)
link(22, 0, 25, 2)
link(23, 0, 25, 3)
link(20, 1, 25, 4)
link(25, 0, 28, 0)
link(3, 0, 28, 1)
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

g.save(OUT, "pixel-art-sprite-from-video-h3", groups=[
    ("Models", [-20, -60, 460, 520], "#3f789e"),
    ("Sprite + driving video", [440, -60, 760, 1300], "#a1309b"),
    ("Animate (MiniMax H3)", [1200, -60, 1200, 760], "#88A"),
    ("Back to pixel art", [2400, -60, 760, 620], "#444"),
], scale=0.5, offset=(40, 80))
