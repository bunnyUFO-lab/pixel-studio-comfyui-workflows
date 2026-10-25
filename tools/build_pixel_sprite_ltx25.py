"""Generates workflows/pixel-art/sprite-from-prompt-ltx25.json: LTX-2.5 (Lightricks, 22B, distilled) animates a pixel
sprite from a motion prompt, with the sprite as both the first and the last frame so the animation loops.

Edit the constants below and re-run:  python tools/build_pixel_sprite_ltx25.py

Wired like ComfyUI's own "First & Last Frame to Video (LTX-2.5)" template, flattened (comfy.py does not expand
subgraphs): LTXVAddGuide puts the sprite at frame 0 and at the last frame, 8 distilled steps (manual sigmas), Euler
ancestral, CFG 1. LTX-2.5 is a joint audio/video model, so an empty audio latent rides along and is discarded.
The pixel steps are the same as the other sprite workflows (build_pixel_sprite_fast.pixel_tail): Re-pixelate onto
the sprite's grid and palette, sprite sheet, Pixel Art Check. An optional prompt enhancer (Gemma 4 E2B) rewrites the
motion prompt into the long scene caption LTX prefers. Licence: LTX-2 Community (free under US$10M revenue).
"""
from build_pixel_sprite_fast import pixel_tail
from comfy_workflow import BLUE, GREEN, RED, WORKFLOWS, YELLOW, Graph, inp, out

LTX_HF = "https://huggingface.co/Lightricks/LTX-2.5/resolve/main"
UNET = "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"   # 21.5 GB
TE = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"          # 15.4 GB
VIDEO_VAE = "ltx-2.5-video-vae-bf16.safetensors"
AUDIO_VAE = "ltx-2.5-audio-vae-bf16.safetensors"
ENHANCER = "gemma4_e2b_it_int8_convrot.safetensors"
URLS = {UNET: f"{LTX_HF}/diffusion_models/{UNET}", TE: f"{LTX_HF}/text_encoders/{TE}",
        VIDEO_VAE: f"{LTX_HF}/vae/{VIDEO_VAE}", AUDIO_VAE: f"{LTX_HF}/vae/{AUDIO_VAE}",
        ENHANCER: f"https://huggingface.co/Comfy-Org/gemma-4/resolve/main/text_encoders/{ENHANCER}"}

PROMPT = ("Pixel art sprite animation. The character does a looping idle animation: breathing, a slight up and down "
          "bob, cape and hair swaying gently. Static camera, the character stays in place, flat solid background.")
NEG = ("blurry, out of focus, overexposed, low contrast, washed out colors, excessive noise, flickering, motion blur, "
       "distorted proportions, deformed features, camera movement, zoom, panning, background change, new objects, text, "
       "watermark")
TARGET, PADDING = 640, 0.35
FRAMES, FPS = 73, 24                 # 8n+1; ~3 s like the other sprite workflows
GUIDE_STRENGTH = 0.9                 # first and last frame (the template uses 0.7; higher holds the sprite closer)
SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"   # distilled, 8 steps
SEED = 7
ENHANCE = False                      # prompt enhancer on (True) or the motion prompt as typed (False)

NOTE = f"""Pixel sprite + motion prompt -> looping sprite sheet, with LTX-2.5 (22B, distilled)

1. Load your sprite (blue). It is upscaled by a whole number onto a flat background (sides divisible by 32).
2. Motion prompt (green): describe the motion. LTX likes a full description; switch "Use prompt enhancer" on to
   have Gemma 4 E2B write one from your short prompt and the sprite.
3. The sprite is the first AND the last frame (guide strength {GUIDE_STRENGTH}), so the animation loops. For a one-shot
   action, bypass "Last frame = sprite".
4. {FRAMES} frames at {FPS} fps (~3 s), 8 distilled steps, CFG 1. Re-pixelate brings every frame back onto the sprite's
   grid and palette; the sheet keeps every frame.
Models (~39 GB): LTX-2.5 distilled int8 (21.5 GB), Gemma 4 12B int8 encoder (15.4 GB), video + audio VAE, Gemma 4 E2B
(enhancer). The LTX-2.5 repo is gated: accept the licence on Hugging Face, HF_TOKEN in .secrets.env.
Licence: LTX-2 Community, free commercial use under US$10M revenue."""


def models(name, directory, type_):
    return {"Node name for S&R": type_, "models": [{"name": name, "url": URLS[name], "directory": directory}]}


def build():
    g = Graph()
    node, link = g.node, g.link
    # models
    node(1, "UNETLoader", [0, 0], [420, 82], outputs=[out("MODEL", "MODEL")], widgets=[UNET, "default"],
         title="Video model (LTX-2.5 distilled)", props=models(UNET, "diffusion_models", "UNETLoader"))
    node(4, "CLIPLoader", [0, 130], [420, 106], outputs=[out("CLIP", "CLIP")], widgets=[TE, "ltxv", "default"],
         title="Gemma 4 12B text encoder", props=models(TE, "text_encoders", "CLIPLoader"))
    node(5, "VAELoader", [0, 280], [420, 58], outputs=[out("VAE", "VAE")], widgets=[VIDEO_VAE],
         title="Video VAE", props=models(VIDEO_VAE, "vae", "VAELoader"))
    node(6, "VAELoader", [0, 380], [420, 58], outputs=[out("VAE", "VAE")], widgets=[AUDIO_VAE],
         title="Audio VAE (unused audio track)", props=models(AUDIO_VAE, "vae", "VAELoader"))
    node(7, "CLIPLoader", [0, 480], [420, 106], outputs=[out("CLIP", "CLIP")], widgets=[ENHANCER, "ltxv", "default"],
         title="Prompt enhancer model (Gemma 4 E2B)", mode=0 if ENHANCE else 4,
         props=models(ENHANCER, "text_encoders", "CLIPLoader"))
    # sprite
    node(10, "LoadImage", [460, 0], [340, 400], outputs=[out("IMAGE", "IMAGE"), out("MASK", "MASK")],
         widgets=["example.png", "image"], title="Pixel sprite", color=BLUE)
    node(11, "JoinImageWithAlpha", [460, 440], [340, 46], inputs=[inp("image", "IMAGE"), inp("alpha", "MASK")],
         outputs=[out("IMAGE", "IMAGE")], title="Keep sprite transparency")
    node(12, "PrepareSpriteForVideo", [460, 530], [340, 160], inputs=[inp("sprite", "IMAGE")],
         outputs=[out("image", "IMAGE"), out("pixel_spec", "PIXEL_SPEC"), out("width", "INT"), out("height", "INT"),
                  out("info", "STRING")],
         widgets=[TARGET, PADDING, "auto", 32], title="Prepare sprite (integer upscale)", color=YELLOW)
    node(16, "PreviewAny", [460, 730], [340, 120], inputs=[inp("source", "*")], title="Sprite info")
    node(13, "LTXVPreprocess", [460, 880], [340, 58], inputs=[inp("image", "IMAGE")],
         outputs=[out("output_image", "IMAGE")], widgets=[18], title="LTX image prep")
    # prompt
    node(20, "PrimitiveStringMultiline", [840, 0], [420, 160], outputs=[out("STRING", "STRING")], widgets=[PROMPT],
         title="Motion prompt", color=GREEN)
    node(21, "TextGenerateLTX2Prompt", [840, 200], [420, 220],
         inputs=[inp("clip", "CLIP"), inp("image", "IMAGE"), inp("video", "IMAGE"), inp("audio", "AUDIO"),
                 inp("prompt", "STRING", "prompt")],
         outputs=[out("generated_text", "STRING")],
         widgets=["", 600, "on", 0.7, 64, 0.95, 0.05, 1.15, 0, 0, False, True], title="Prompt enhancer",
         mode=0 if ENHANCE else 4)
    node(22, "ComfySwitchNode", [840, 460], [300, 82], inputs=[inp("on_false", "STRING"), inp("on_true", "STRING")],
         outputs=[out("output", "STRING")], widgets=[ENHANCE], title="Use prompt enhancer")
    node(23, "PreviewAny", [840, 580], [420, 160], inputs=[inp("source", "*")], title="Prompt used")
    node(24, "CLIPTextEncode", [840, 780], [420, 100], inputs=[inp("clip", "CLIP"), inp("text", "STRING", "text")],
         outputs=[out("CONDITIONING", "CONDITIONING")], widgets=[""], title="Encode prompt")
    node(25, "CLIPTextEncode", [840, 920], [420, 140], inputs=[inp("clip", "CLIP")],
         outputs=[out("CONDITIONING", "CONDITIONING")], widgets=[NEG], title="Negative prompt", color=RED)
    node(26, "LTXVConditioning", [840, 1100], [300, 82],
         inputs=[inp("positive", "CONDITIONING"), inp("negative", "CONDITIONING")],
         outputs=[out("positive", "CONDITIONING"), out("negative", "CONDITIONING")], widgets=[FPS])
    # latents and guides (first frame, last frame = the sprite)
    node(60, "EmptyLTXVLatentVideo", [1300, 0], [300, 130],
         inputs=[inp("width", "INT", "width"), inp("height", "INT", "height")], outputs=[out("LATENT", "LATENT")],
         widgets=[TARGET, TARGET, FRAMES, 1], title=f"Frames ({FRAMES} = 8n+1, {FPS} fps)", color=YELLOW)

    def guide(id_, y, idx, title):
        node(id_, "LTXVAddGuide", [1300, y], [300, 150],
             inputs=[inp("positive", "CONDITIONING"), inp("negative", "CONDITIONING"), inp("vae", "VAE"),
                     inp("latent", "LATENT"), inp("image", "IMAGE")],
             outputs=[out("positive", "CONDITIONING"), out("negative", "CONDITIONING"), out("latent", "LATENT")],
             widgets=[idx, GUIDE_STRENGTH], title=title, color=YELLOW)

    guide(61, 170, 0, "First frame = sprite")
    guide(62, 360, -1, "Last frame = sprite (loop; bypass for one-shot)")
    node(63, "LTXVEmptyLatentAudio", [1300, 550], [300, 106], inputs=[inp("audio_vae", "VAE")],
         outputs=[out("Latent", "LATENT")], widgets=[FRAMES, FPS, 1], title="Empty audio (discarded)")
    node(64, "LTXVConcatAVLatent", [1300, 700], [300, 58],
         inputs=[inp("video_latent", "LATENT"), inp("audio_latent", "LATENT")], outputs=[out("latent", "LATENT")])
    # sampling
    node(70, "RandomNoise", [1640, 0], [300, 82], outputs=[out("NOISE", "NOISE")], widgets=[SEED, "randomize"],
         title="Seed")
    node(71, "LTXVDualCFGGuider", [1640, 120], [300, 106],
         inputs=[inp("model", "MODEL"), inp("positive", "CONDITIONING"), inp("negative", "CONDITIONING")],
         outputs=[out("GUIDER", "GUIDER")], widgets=[1, 1])
    node(72, "SamplerEulerAncestral", [1640, 260], [300, 82], outputs=[out("SAMPLER", "SAMPLER")], widgets=[0, 1])
    node(73, "ManualSigmas", [1640, 380], [300, 82], outputs=[out("SIGMAS", "SIGMAS")], widgets=[SIGMAS],
         title="Sigmas (distilled, 8 steps)")
    node(74, "SamplerCustomAdvanced", [1640, 500], [300, 120],
         inputs=[inp("noise", "NOISE"), inp("guider", "GUIDER"), inp("sampler", "SAMPLER"), inp("sigmas", "SIGMAS"),
                 inp("latent_image", "LATENT")], outputs=[out("output", "LATENT"), out("denoised_output", "LATENT")],
         title="Sampler")
    node(75, "LTXVSeparateAVLatent", [1640, 660], [300, 58], inputs=[inp("av_latent", "LATENT")],
         outputs=[out("video_latent", "LATENT"), out("audio_latent", "LATENT")])
    node(76, "LTXVCropGuides", [1640, 760], [300, 82],
         inputs=[inp("positive", "CONDITIONING"), inp("negative", "CONDITIONING"), inp("latent", "LATENT")],
         outputs=[out("positive", "CONDITIONING"), out("negative", "CONDITIONING"), out("latent", "LATENT")])
    node(77, "VAEDecodeTiled", [1640, 880], [300, 150], inputs=[inp("samples", "LATENT"), inp("vae", "VAE")],
         outputs=[out("IMAGE", "IMAGE")], widgets=[512, 64, 64, 16])
    node(50, "Note", [0, 640], [420, 420], widgets=[NOTE], title="READ ME", props={})

    link(10, 0, 11, 0)            # LoadImage drops alpha into MASK: join it back
    link(10, 1, 11, 1)
    link(11, 0, 12, 0)
    link(12, 4, 16, 0)
    link(12, 0, 13, 0)
    link(7, 0, 21, 0)
    link(13, 0, 21, 1)
    link(20, 0, 21, 4)
    link(20, 0, 22, 0)            # switch off: the prompt as typed
    link(21, 0, 22, 1)            # switch on: the enhanced prompt
    link(22, 0, 23, 0)
    link(4, 0, 24, 0)
    link(22, 0, 24, 1)
    link(4, 0, 25, 0)
    link(24, 0, 26, 0)
    link(25, 0, 26, 1)
    link(12, 2, 60, 0)
    link(12, 3, 60, 1)
    link(26, 0, 61, 0)            # first-frame guide on the empty latent
    link(26, 1, 61, 1)
    link(5, 0, 61, 2)
    link(60, 0, 61, 3)
    link(13, 0, 61, 4)
    link(61, 0, 62, 0)            # last-frame guide chained after it
    link(61, 1, 62, 1)
    link(5, 0, 62, 2)
    link(61, 2, 62, 3)
    link(13, 0, 62, 4)
    link(6, 0, 63, 0)
    link(62, 2, 64, 0)
    link(63, 0, 64, 1)
    link(1, 0, 71, 0)
    link(62, 0, 71, 1)
    link(62, 1, 71, 2)
    link(70, 0, 74, 0)
    link(71, 0, 74, 1)
    link(72, 0, 74, 2)
    link(73, 0, 74, 3)
    link(64, 0, 74, 4)
    link(74, 1, 75, 0)
    link(62, 0, 76, 0)
    link(62, 1, 76, 1)
    link(75, 0, 76, 2)
    link(76, 2, 77, 0)
    link(5, 0, 77, 1)
    pixel_tail(g, 2000, (77, 0), "pixel/sprite_from_prompt_ltx25", FPS, True, "idle")
    g.save(WORKFLOWS / "pixel-art" / "sprite-from-prompt-ltx25.json", "pixel-art-sprite-from-prompt-ltx25", groups=[
        ("Models", [-20, -60, 460, 1140], "#3f789e"),
        ("Sprite + prompt", [440, -60, 840, 1240], "#a1309b"),
        ("Animate (LTX-2.5, first = last frame)", [1280, -60, 680, 1140], "#88A"),
        ("Back to pixel art", [1980, -60, 1300, 960], "#444"),
    ], scale=0.5, offset=(40, 80))


if __name__ == "__main__":
    build()
