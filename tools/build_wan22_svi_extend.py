"""Generates workflows/video-to-video/wan22-svi-extend.json.

Edit the constants below and re-run:  python tools/build_wan22_svi_extend.py

Extends a video with Wan 2.2 Remix + SVI 2.0 Pro LoRAs (Stable Video Infinity): each segment continues
from the previous segment's last latent (motion context) and is anchored to a reference frame. The number
of segments is a setting (core StartLoop/EndLoop), with one prompt per line. Outputs the full video
(original + extensions, cross-faded at the joins) and the new frames on their own.
"""
from comfy_workflow import (BLUE, GREEN, HIGH, LOW, NEGATIVE, RED, TEXT_ENCODER, URLS, VAE, WORKFLOWS, YELLOW,
                            Graph, inp, out)

OUT = WORKFLOWS / "video-to-video" / "wan22-svi-extend.json"

SVI = "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/LoRAs/Stable-Video-Infinity/v2.0"
SVI_HIGH = "SVI_v2_PRO_Wan2.2-I2V-A14B_HIGH_lora_rank_128_fp16.safetensors"
SVI_LOW = "SVI_v2_PRO_Wan2.2-I2V-A14B_LOW_lora_rank_128_fp16.safetensors"
SVI_STRENGTH = 1.0

SEGMENTS = 2               # loops: each adds ~(FRAMES - 1 - OVERLAP) frames (~4.75 s at 16 fps)
FRAMES = 81                # frames generated per segment (4n+1)
MAX_SIDE = 832             # 832 = 480p (fast); 1280 = 720p (~2.5x slower)
FPS = 16
TOTAL_STEPS, SWITCH_STEP, SHIFT = 8, 4, 8   # Remix has the Lightning distill baked in
OVERLAP = 4
PROMPTS = ("The camera slowly pulls back as the character keeps moving naturally, smooth continuous motion.\n"
           "The character turns to face the camera, natural body motion, soft cinematic lighting.")

NOTE = f"""Extend a video (Wan 2.2 Remix + SVI 2.0 Pro)

1. Load a video (blue). It is resampled to {FPS} fps and resized (longest side {MAX_SIDE}).
2. segments (Start Loop -> num_iterations): how many times to extend.
   Each segment adds ~{(FRAMES - 1 - OVERLAP) / FPS:.1f} s.
3. Prompts (green): one line per segment, describing what happens next.
   More segments than lines -> the last line repeats.
4. Run. Two MP4s are saved in output/video/:
   wan22_extend_full_*  original + all new segments (cross-faded at each join)
   wan22_extend_new_*   only the new frames
To extend further: load the *_full video and run again with new prompts.

Anchor (Video Anchor + Tail): SVI keeps returning towards this frame.
  first frame = keeps the original identity; last frame = follows where the video ended.
Join: overlap = frames cross-faded at each join; color_match limits colour drift.
Time on an RTX 4090 at 480p: ~1.75 min per segment (measured: 2 segments in 3.5 min).
Models: Remix v3.0 high/low + SVI 2.0 Pro LoRAs (2 x 1.23 GB) + NSFW UMT5 + Wan 2.1 VAE."""


def models(name, url, directory, type_):
    return {"Node name for S&R": type_, "models": [{"name": name, "url": url, "directory": directory}]}


VHS_COMBINE = {"frame_rate": FPS, "loop_count": 0, "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 17,
               "save_metadata": True, "trim_to_audio": False, "pingpong": False, "save_output": True,
               "videopreview": {"hidden": False, "paused": False, "params": {}}}

g = Graph()
node, link = g.node, g.link

# Input video
node(1, "VHS_LoadVideo", [0, 0], [380, 520], outputs=[out("IMAGE", "IMAGE"), out("frame_count", "INT"),
                                                       out("audio", "AUDIO"), out("video_info", "VHS_VIDEOINFO")],
     inputs=[inp("meta_batch", "VHS_BatchManager"), inp("vae", "VAE")],
     widgets={"video": "example.mp4", "force_rate": FPS, "custom_width": 0, "custom_height": 0, "frame_load_cap": 0,
              "skip_first_frames": 0, "select_every_nth": 1, "format": "AnimateDiff",
              "videopreview": {"hidden": False, "paused": False, "params": {}}},
     title="Input video", color=BLUE, props={"Node name for S&R": "VHS_LoadVideo", "cnr_id": "comfyui-videohelpersuite"})
node(2, "ResizeForWan", [420, 0], [300, 82], inputs=[inp("images", "IMAGE")],
     outputs=[out("images", "IMAGE"), out("width", "INT"), out("height", "INT")], widgets=[MAX_SIDE],
     title="Resize (longest side)", color=YELLOW)
node(3, "VideoAnchorAndTail", [420, 130], [300, 82], inputs=[inp("images", "IMAGE")],
     outputs=[out("anchor", "IMAGE"), out("tail", "IMAGE")], widgets=["first frame", 5], title="Anchor + tail frames",
     color=YELLOW)
node(4, "VAEEncode", [420, 260], [210, 46], inputs=[inp("pixels", "IMAGE"), inp("vae", "VAE")],
     outputs=[out("LATENT", "LATENT")], title="Encode anchor")
node(5, "VAEEncode", [420, 340], [210, 46], inputs=[inp("pixels", "IMAGE"), inp("vae", "VAE")],
     outputs=[out("LATENT", "LATENT")], title="Encode tail (motion context)")

# Models
node(10, "UNETLoader", [0, 580], [380, 82], outputs=[out("MODEL", "MODEL")], widgets=[HIGH, "default"],
     title="Remix HIGH-noise", props=models(HIGH, URLS[HIGH], "diffusion_models", "UNETLoader"))
node(11, "UNETLoader", [0, 700], [380, 82], outputs=[out("MODEL", "MODEL")], widgets=[LOW, "default"],
     title="Remix LOW-noise", props=models(LOW, URLS[LOW], "diffusion_models", "UNETLoader"))
node(12, "LoraLoaderModelOnly", [420, 580], [340, 82], inputs=[inp("model", "MODEL")], outputs=[out("MODEL", "MODEL")],
     widgets=[SVI_HIGH, SVI_STRENGTH], title="SVI 2.0 Pro LoRA (HIGH)",
     props=models(SVI_HIGH, f"{SVI}/{SVI_HIGH}", "loras", "LoraLoaderModelOnly"))
node(13, "LoraLoaderModelOnly", [420, 700], [340, 82], inputs=[inp("model", "MODEL")], outputs=[out("MODEL", "MODEL")],
     widgets=[SVI_LOW, SVI_STRENGTH], title="SVI 2.0 Pro LoRA (LOW)",
     props=models(SVI_LOW, f"{SVI}/{SVI_LOW}", "loras", "LoraLoaderModelOnly"))
node(14, "ModelSamplingSD3", [800, 580], [260, 58], inputs=[inp("model", "MODEL")], outputs=[out("MODEL", "MODEL")],
     widgets=[SHIFT], title="Shift (HIGH)")
node(15, "ModelSamplingSD3", [800, 700], [260, 58], inputs=[inp("model", "MODEL")], outputs=[out("MODEL", "MODEL")],
     widgets=[SHIFT], title="Shift (LOW)")
node(16, "CLIPLoader", [0, 820], [380, 106], outputs=[out("CLIP", "CLIP")], widgets=[TEXT_ENCODER, "wan", "default"],
     title="NSFW UMT5 text encoder", props=models(TEXT_ENCODER, URLS[TEXT_ENCODER], "text_encoders", "CLIPLoader"))
node(17, "VAELoader", [0, 970], [380, 58], outputs=[out("VAE", "VAE")], widgets=[VAE],
     props=models(VAE, URLS[VAE], "vae", "VAELoader"))
node(18, "CLIPTextEncode", [420, 820], [400, 160], inputs=[inp("clip", "CLIP")],
     outputs=[out("CONDITIONING", "CONDITIONING")], widgets=[NEGATIVE], title="Negative prompt", color=RED)

# Loop: one iteration per segment
node(20, "StartLoop", [1120, 0], [300, 160],
     inputs=[inp("parent_iteration", "INT"), inp("initial_iteration_value", "*"),
             inp("mode.num_iterations", "INT", "mode.num_iterations")],
     outputs=[out("iteration_index", "INT"), out("is_first", "BOOLEAN"), out("is_last", "BOOLEAN"),
              out("list_item", "*"), out("current_iteration_value", "*")],
     widgets=["simple", SEGMENTS, False], title="Segments (num_iterations)", color=YELLOW)
node(21, "SegmentPrompt", [1120, 200], [420, 220], inputs=[inp("index", "INT")],
     outputs=[out("prompt", "STRING"), out("segment_seed", "INT")], widgets=[PROMPTS, 1234, "randomize"],
     title="Prompts (one line per segment)", color=GREEN)
node(22, "CLIPTextEncode", [1580, 200], [300, 90], inputs=[inp("clip", "CLIP"), inp("text", "STRING", "text")],
     outputs=[out("CONDITIONING", "CONDITIONING")], widgets=[""], title="Encode segment prompt")
node(23, "WanSVIProConditioning", [1580, 330], [320, 150],
     inputs=[inp("positive", "CONDITIONING"), inp("negative", "CONDITIONING"), inp("anchor_samples", "LATENT"),
             inp("prev_samples", "LATENT")],
     outputs=[out("positive", "CONDITIONING"), out("negative", "CONDITIONING"), out("latent", "LATENT")],
     widgets=[FRAMES, 1], title="SVI conditioning")
node(24, "KSamplerAdvanced", [1940, 0], [320, 334],
     inputs=[inp("model", "MODEL"), inp("positive", "CONDITIONING"), inp("negative", "CONDITIONING"),
             inp("latent_image", "LATENT"), inp("noise_seed", "INT", "noise_seed")],
     outputs=[out("LATENT", "LATENT")],
     widgets=["enable", 0, "fixed", TOTAL_STEPS, 1, "euler", "simple", 0, SWITCH_STEP, "enable"],
     title="KSampler HIGH noise")
node(25, "KSamplerAdvanced", [1940, 380], [320, 334],
     inputs=[inp("model", "MODEL"), inp("positive", "CONDITIONING"), inp("negative", "CONDITIONING"),
             inp("latent_image", "LATENT")],
     outputs=[out("LATENT", "LATENT")],
     widgets=["disable", 0, "fixed", TOTAL_STEPS, 1, "euler", "simple", SWITCH_STEP, 10000, "disable"],
     title="KSampler LOW noise")
node(26, "VAEDecode", [2300, 0], [210, 46], inputs=[inp("samples", "LATENT"), inp("vae", "VAE")],
     outputs=[out("IMAGE", "IMAGE")])
node(27, "EndLoop", [2300, 100], [300, 140],
     inputs=[inp("output_value", "*"), inp("next_iteration_value", "*"), inp("terminations.termination0", "*")],
     outputs=[out("outputs", "IMAGE")], widgets=[True], title="End loop (collect segments)")

# Join + save
node(28, "JoinVideoSegments", [2640, 0], [320, 106], inputs=[inp("original", "IMAGE"), inp("segments", "IMAGE")],
     outputs=[out("full_video", "IMAGE"), out("new_frames", "IMAGE")], widgets=[OVERLAP, True],
     title="Join segments", color=YELLOW)
node(29, "VHS_VideoCombine", [3000, 0], [380, 560],
     inputs=[inp("images", "IMAGE"), inp("audio", "AUDIO"), inp("meta_batch", "VHS_BatchManager"), inp("vae", "VAE")],
     outputs=[out("Filenames", "VHS_FILENAMES")],
     widgets={**VHS_COMBINE, "filename_prefix": "video/wan22_extend_full"}, title="Save full video", color=GREEN,
     props={"Node name for S&R": "VHS_VideoCombine", "cnr_id": "comfyui-videohelpersuite"})
node(30, "VHS_VideoCombine", [3420, 0], [380, 560],
     inputs=[inp("images", "IMAGE"), inp("audio", "AUDIO"), inp("meta_batch", "VHS_BatchManager"), inp("vae", "VAE")],
     outputs=[out("Filenames", "VHS_FILENAMES")],
     widgets={**VHS_COMBINE, "filename_prefix": "video/wan22_extend_new"}, title="Save new frames only", color=GREEN,
     props={"Node name for S&R": "VHS_VideoCombine", "cnr_id": "comfyui-videohelpersuite"})
node(31, "Note", [0, 1080], [460, 560], widgets=[NOTE], title="READ ME", props={})

link(1, 0, 2, 0)
link(2, 0, 3, 0)
link(3, 0, 4, 0)
link(17, 0, 4, 1)
link(3, 1, 5, 0)
link(17, 0, 5, 1)
link(10, 0, 12, 0)
link(11, 0, 13, 0)
link(12, 0, 14, 0)
link(13, 0, 15, 0)
link(16, 0, 18, 0)
link(5, 0, 20, 1)            # initial carried value = encoded tail of the input video
link(20, 0, 21, 0)           # iteration_index -> prompt line + seed
link(16, 0, 22, 0)
link(21, 0, 22, 1)
link(22, 0, 23, 0)
link(18, 0, 23, 1)
link(4, 0, 23, 2)
link(20, 4, 23, 3)           # current_iteration_value = previous segment's latent
link(14, 0, 24, 0)
link(23, 0, 24, 1)
link(23, 1, 24, 2)
link(23, 2, 24, 3)
link(21, 1, 24, 4)
link(15, 0, 25, 0)
link(23, 0, 25, 1)
link(23, 1, 25, 2)
link(24, 0, 25, 3)
link(25, 0, 26, 0)
link(17, 0, 26, 1)
link(26, 0, 27, 0)           # output_value: decoded frames, accumulated
link(25, 0, 27, 1)           # next_iteration_value: this segment's latent
link(2, 0, 28, 0)
link(27, 0, 28, 1)
link(28, 0, 29, 0)
link(28, 1, 30, 0)

g.save(OUT, "video-to-video-wan22-svi-extend",
       groups=[("Input", [-20, -60, 760, 480], "#a1309b"), ("Models", [-20, 520, 1100, 520], "#3f789e"),
               ("Loop: one iteration per segment", [1100, -60, 1520, 820], "#8A8"),
               ("Join + save", [2620, -60, 1200, 660], "#88A")],
       scale=0.45, offset=(40, 80))
