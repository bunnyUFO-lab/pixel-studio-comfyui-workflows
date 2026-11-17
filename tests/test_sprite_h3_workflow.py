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
for title in ("Pixel sprite", "Character (one sentence)", "Starting pose", "Action (written as beats)",
              "Use character and pose in the prompt", "Frames to keep (across the whole clip)", "Noise (seed here)",
              "Loop (first = last frame) / video frames", "Prepare sprite (integer upscale)", "Clean up (block vote, palette, effect colours)",
              "Save sprite sheet"):
    check(title in by_title, f"input by title: {title}")
check(by_title["Noise (seed here)"]["widgets_values"][0] == 6305, "seed is the first widget of the noise node")
h3 = by_title["Loop (first = last frame) / video frames"]
check(source(h3, "first_frame")["type"] == "PrepareSpriteForVideo" and source(h3, "last_frame") is source(h3, "first_frame"),
      "the sprite is the first and the last frame (a loop); the app unlinks last_frame for a one-shot, as it does for Wan")
check(source(h3, "prompt")["type"] == "ComfySwitchNode", "the prompt comes from the character/pose switch")
check(by_title["Use character and pose in the prompt"]["widgets_values"][0] is False, "the tested form (action alone) is the default")

# the recipe of the h3-match session
prep = by_title["Prepare sprite (integer upscale)"]
check(prep["widgets_values"][4:] == ["whole canvas", 8], "preparation: the whole canvas enlarged x8 (the x8-v1 default; x4 is one value away), not a crop plus padding")
check(h3["widgets_values"][3] == 39 and (h3["widgets_values"][3] - 5) % 17 == 0, "39 video frames, on the 17k+5 grid")
sched = next(n for n in wf["nodes"] if n["type"] == "BasicScheduler")
check(sched["widgets_values"][:2] == ["simple", 8], "8 steps, simple schedule")
check(next(n for n in wf["nodes"] if n["type"] == "KSamplerSelect")["widgets_values"][0] == "res_multistep", "res_multistep")
check(any(n["type"] == "BasicGuider" for n in wf["nodes"]) and not any(n["type"] in ("CFGGuider", "KSampler") for n in wf["nodes"]), "no CFG")
check(next(n for n in wf["nodes"] if n["type"] == "LoraLoaderModelOnly")["widgets_values"][1] == 1.0, "turbo LoRA at 1.0")

# the finishing chain: decode -> pick 13 -> clean up -> sheet. No colour hold, no hold figure, no re-pixelate.
pick = by_title["Frames to keep (across the whole clip)"]
clean = by_title["Clean up (block vote, palette, effect colours)"]
check(source(pick, "frames")["type"] == "VAEDecode" and pick["widgets_values"] == [13, False], "13 frames are picked from the decoded video, the first being the input sprite")
check(source(clean, "frames") is pick and source(clean, "pixel_spec") is by_title["Prepare sprite (integer upscale)"], "the clean-up works on the picked frames and the preparation's spec")
check(clean["widgets_values"] == [60.0, 4, 40.0, False, True, True, "centre", 4, 1, 3, True, 3, 0.5, 40.0, 8],
      "clean-up = x8-v3: tolerance 60, 4 effect colours, threshold 40, no despeckle, outline, frame 1 = sprite, pick centre, min_hole 4, edge_trim 1, effect_min 3, follow_hold on (3, 0.5, 40), feet_band 8")
px_inputs = px.CleanVideoFrames.INPUT_TYPES()
names = list(px_inputs["required"]) + list(px_inputs["optional"])
check(names[5:] == ["despeckle", "outline", "first_is_sprite", "pick", "min_hole", "edge_trim", "effect_min", "follow_hold", "follow_radius", "follow_mismatch", "follow_margin", "feet_band"],
      "the graph's widget order matches the node's inputs")
check(source(by_title["Save sprite sheet"], "frames") is clean, "the sheet is made from the cleaned frames")
check(not any(n["type"] in ("HoldFigure", "RepixelateFrames") for n in wf["nodes"]), "the colour hold (a RepixelateFrames option) and Hold Figure are not used")

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
