"""CPU tests for the MiniMax H3 sprite workflow (workflows/pixel-art/sprite-from-prompt-h3.json), its builder and the
PickEvenFrames node. No ComfyUI or GPU needed.

    python tests/test_sprite_h3_workflow.py
"""
import json
import os
import subprocess
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "custom_nodes", "comfyui_workflows_utils"))
import pixel as px  # noqa: E402

RESULTS = []


def check(cond, msg):
    RESULTS.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + msg)


# --- PickEvenFrames
P = px.PickEvenFrames
idx = P.indices(124, 12)
check(len(idx) == 12 and idx[-1] == 123 and idx[0] > 0, f"124 frames, 12 kept, the input frame dropped, ends on the last: {idx}")
check(idx == sorted(set(idx)) and max(b - a for a, b in zip(idx, idx[1:])) - min(b - a for a, b in zip(idx, idx[1:])) <= 1,
      "the frames are evenly spaced (gaps differ by at most 1)")
check(P.indices(124, 12, False)[0] == 0 and len(P.indices(124, 12, False)) == 12, "skip_first off: starts on the first frame")
check(P.indices(5, 12) == [0, 1, 2, 3, 4], "a clip shorter than the count is kept whole")
t = torch.arange(124, dtype=torch.float32).reshape(124, 1, 1, 1).expand(124, 2, 2, 3)
kept, info = P().run(t, 12, True)
check(kept.shape == (12, 2, 2, 3) and [int(v) for v in kept[:, 0, 0, 0]] == idx, "the node keeps exactly those frames")
check(P.INPUT_TYPES()["required"]["count"][1]["default"] == 12, "12 frames by default")

# --- the workflow
path = os.path.join(ROOT, "workflows", "pixel-art", "sprite-from-prompt-h3.json")
before = open(path, "rb").read()
subprocess.run([sys.executable, os.path.join(ROOT, "tools", "build_pixel_sprite_h3.py")], cwd=os.path.join(ROOT, "tools"),
               check=True, capture_output=True)
check(open(path, "rb").read() == before, "the committed workflow is what the builder writes")
wf = json.loads(before.decode("utf-8"))
nodes = {n["id"]: n for n in wf["nodes"]}
by_title = {n.get("title"): n for n in wf["nodes"] if n.get("title")}
links = {l[0]: l for l in wf["links"]}


def source(node, input_name):
    inp = next(i for i in node["inputs"] if i["name"] == input_name)
    return nodes[links[inp["link"]][1]] if inp.get("link") is not None else None


# inputs the app sets by title
for title in ("Pixel sprite", "Character (one sentence)", "Starting pose", "Action (beats: wind-up, strike, impact, settle)",
              "Use character and pose in the prompt", "Frames to keep (across the whole clip)", "Noise (seed here)",
              "Loop (first = last frame) / video frames", "Prepare sprite (integer upscale)", "Hold figure (height, ground line, centre)",
              "Save sprite sheet"):
    check(title in by_title, f"input by title: {title}")
check(by_title["Noise (seed here)"]["widgets_values"][0] == 6305, "seed is the first widget of the noise node")
h3 = by_title["Loop (first = last frame) / video frames"]
check(h3["widgets_values"][3] == 124 and (h3["widgets_values"][3] - 5) % 17 == 0, "video length 124 sits on the 17k+5 grid")
check(source(h3, "first_frame")["type"] == "PrepareSpriteForVideo" and source(h3, "last_frame") is source(h3, "first_frame"),
      "the sprite is the first and the last frame (a loop); the app unlinks last_frame for a one-shot, as it does for Wan")
check(source(h3, "prompt")["type"] == "ComfySwitchNode", "the prompt comes from the character/pose switch")
check(by_title["Use character and pose in the prompt"]["widgets_values"][0] is False, "the tested form (action alone) is the default")

# the finishing chain: decode -> hold figure -> repixelate -> pick frames -> sheet
hold = by_title["Hold figure (height, ground line, centre)"]
repix = by_title["Repixelate (same grid + palette)"]
check(source(hold, "frames")["type"] == "VAEDecode" and hold["widgets_values"][0] is True, "hold figure is on, straight after the decode")
check(source(repix, "frames") is hold, "re-pixelate works on the held frames")
check(source(by_title["Frames to keep (across the whole clip)"], "frames") is repix, "frames are picked after re-pixelate")
check(source(by_title["Save sprite sheet"], "frames")["type"] == "PickEvenFrames", "the sheet is made from the picked frames")
check(repix["widgets_values"][9] is False, "the colour hold is off (it flattened a slash-effect frame on H3)")

# models: each loader names a file with a URL, and the names match the widget values
need = {"UNETLoader": "diffusion_models", "CLIPLoader": "text_encoders", "VAELoader": "vae", "LoraLoaderModelOnly": "loras"}
for n in wf["nodes"]:
    if n["type"] in need:
        ms = n["properties"].get("models", [])
        check(len(ms) == 1 and ms[0]["name"] == n["widgets_values"][0] and ms[0]["directory"] == need[n["type"]]
              and ms[0]["url"].startswith("https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/"),
              f"{n['type']}: {n['widgets_values'][0]} is listed with its URL")

print(f"{sum(RESULTS)}/{len(RESULTS)} passed")
sys.exit(0 if all(RESULTS) else 1)
