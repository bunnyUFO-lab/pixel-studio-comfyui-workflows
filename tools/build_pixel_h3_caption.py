"""Generates workflows/pixel-art/h3-caption.json.

Edit the constants below and re-run:  python tools/build_pixel_h3_caption.py

Our own MiniMax H3 prompt fixer, the model half: a short ask ("hammer smash") plus the sprite -> the action caption H3
responds to (beats, 30 to 50 words). Qwen3-VL 8B (Apache 2.0, the text encoder the Qwen edit workflows already load) looks
at the sprite through ComfyUI's TextGenerate and writes the caption twice, with two seeds. The checks and repairs that
follow (banned words, the closing return) are code in the app, not graph: the model ignores some word bans, so the app
looks at both captions and picks or rewords one (webapp/app/h3_fixer.py in pixel-studio; rules v6 of the h3-match session,
research/sprint2/h3match/FIXER-PORT.md).

Inputs by title: "Sprite (on white, enlarged x4)" (the app prepares it: RGBA composited on white, 96 -> 384 px nearest),
"System prompt", "User prompt" (examples, then the case), "Caption writer 1 (first seed)" and "Caption writer 2 (second seed)"
(their sampling_mode.seed). Outputs by title: "Caption" (first seed) and "Caption 2" (second seed).
No video model is loaded: about 25 to 60 s and 2 to 5 cents on the cloud worker, mostly the model load.
"""
from comfy_workflow import BLUE, GREEN, QWEN_HF, QWEN_TE, WORKFLOWS, Graph, inp, out

OUT = WORKFLOWS / "pixel-art" / "h3-caption.json"

# CLIPLoader type "ideogram4": H3's own text encoder cannot generate text (TextGenerate returns only commas), and the
# Qwen3-VL 8B file loads under this type in the cloud runs of the h3-match session.
CLIP_TYPE = "ideogram4"
MAX_LENGTH = 160
# sampling_mode "on" + temperature, top_k, top_p, min_p, repetition_penalty, seed
SAMPLING = ["on", 0.3, 64, 0.95, 0.05, 1.05]
SEEDS = (7, 8)

SYSTEM = """You write action captions for a video model that animates one pixel-art game sprite. The first frame of the video is the sprite itself, so the caption describes only MOVEMENT, never appearance, colours, style or background.
Rules:
- Two or three imperative sentences, 30 to 50 words in all. Start with a verb ("Raise", "Lift", "Crouch", "Stride"). Do not name the character.
- For an attack or other one-off action, walk through the beats in order: the anticipation or wind-up, the main action with its weight, the follow-through, then the return to the starting stance.
- Give each beat its speed: a quick raise or wind-up, a fast strike, a short hold, a quick return, so the whole action ends inside the clip.
- Keep any effect small and tight to what makes it: a thin trail following the weapon, a few sparks where it hits, a bolt straight out from the hand no longer than the arm's reach. Never a ring, burst or crescent around the body.
- Keep the body and face turned the same way as in the sprite throughout; pull a held object or hand back toward the chest or shoulder, never behind the body.
- When a weapon is drawn back for a swing, pull it back over the shoulder with a slight lean, the body and face staying side-on.
- For an attack or cast, the last beat brings the body, the weapon or the hand back to where it was in the first frame; do not end with an arm or weapon still held out.
- Name the body parts and held objects that move, and how (arc, thrust, bob). Say what stays steady.
- For a walk or run, describe one stride as its beats in order: the lead foot steps down with the body's bob, then the trailing leg swings through and plants. Name what stays steady. End on a step that carries on into the next one; do not end with a return to the stance or to the starting position. Never write stride forward, sway, turn or travel.
- For an idle or other loop, describe the repeating motion and end by returning to the initial stance.
- Drop instructions about the canvas, frames, staying in place or looping; the model does not need them.
Answer with the caption only."""

USER = """Subject: a stocky dwarf blacksmith with a braided red beard, a brown leather apron and a heavy steel hammer
Starting pose: standing upright facing right with the hammer resting on his shoulder
Request: Overhead hammer smash.
Caption:"""

NOTE = """Prompt fixer for MiniMax H3: the caption writer

1. Load the sprite prepared the way the app does it (blue): RGBA composited on opaque white, enlarged x4 with nearest
   neighbour (96 -> 384 px), saved as an RGB PNG.
2. System prompt (green): the rules (v6). User prompt (green): a few examples of short asks and their captions, then
   the sprite's subject, its starting pose and the ask, ending with "Caption:".
3. Two TextGenerate runs, identical except the seed (7 and 8): Qwen3-VL 8B looks at the sprite, 160 tokens,
   temperature 0.3, top_k 64, top_p 0.95, min_p 0.05, repetition penalty 1.05. Caption = first seed, Caption 2 = second.
4. What comes next is not in this graph: the app (or you) checks the first caption against the banned words, falls back
   to the second, and last of all rewords the first by code. The model ignores some word bans, so the check is code.
No video model is loaded. About 25 to 60 s and 2 to 5 cents on the cloud worker.
Model: Qwen3-VL 8B int8 (Apache 2.0), the one the Qwen edit workflows load."""

g = Graph()
node, link = g.node, g.link

node(1, "CLIPLoader", [0, 0], [420, 106], outputs=[out("CLIP", "CLIP")], widgets=[QWEN_TE, CLIP_TYPE, "default"],
     title="Load Qwen3-VL text model",
     props={"Node name for S&R": "CLIPLoader",
            "models": [{"name": QWEN_TE, "url": f"{QWEN_HF}/text_encoders/{QWEN_TE}", "directory": "text_encoders"}]})
node(2, "LoadImage", [0, 150], [340, 400], outputs=[out("IMAGE", "IMAGE"), out("MASK", "MASK")],
     widgets=["example.png", "image"], title="Sprite (on white, enlarged x4)", color=BLUE)
node(3, "PrimitiveStringMultiline", [460, 0], [520, 360], outputs=[out("STRING", "STRING")], widgets=[SYSTEM],
     title="System prompt", color=GREEN)
node(4, "PrimitiveStringMultiline", [460, 400], [520, 220], outputs=[out("STRING", "STRING")], widgets=[USER],
     title="User prompt", color=GREEN)
for nid, y, seed, title in ((5, 0, SEEDS[0], "Caption writer 1 (first seed)"), (6, 400, SEEDS[1], "Caption writer 2 (second seed)")):
    node(nid, "TextGenerate", [1020, y], [420, 360],
         inputs=[inp("clip", "CLIP"), inp("image", "IMAGE"), inp("prompt", "STRING", "prompt"), inp("system_prompt", "STRING")],
         outputs=[out("generated_text", "STRING")],
         widgets=["", MAX_LENGTH] + SAMPLING + [seed, False, True], title=title, color=GREEN)
node(7, "PreviewAny", [1480, 0], [420, 200], inputs=[inp("source", "*")], title="Caption")
node(8, "PreviewAny", [1480, 400], [420, 200], inputs=[inp("source", "*")], title="Caption 2")
node(9, "Note", [0, 600], [420, 380], widgets=[NOTE], title="READ ME", props={})

link(1, 0, 5, 0)
link(1, 0, 6, 0)
link(2, 0, 5, 1)
link(2, 0, 6, 1)
link(4, 0, 5, 2)
link(4, 0, 6, 2)
link(3, 0, 5, 3)
link(3, 0, 6, 3)
link(5, 0, 7, 0)
link(6, 0, 8, 0)

g.save(OUT, "pixel-art-h3-caption", groups=[
    ("Model + sprite", [-20, -60, 460, 1100], "#3f789e"),
    ("Prompts", [440, -60, 560, 700], "#a1309b"),
    ("Write the caption twice (seeds 7 and 8)", [1000, -60, 920, 700], "#88A"),
], scale=0.5, offset=(40, 80))
