"""CPU tests for the H3 prompt-fixer workflow (workflows/pixel-art/h3-caption.json) and its builder: inputs and outputs by
title, the two seeds, the sampling settings of the h3-match session (rules v6), the model list. No ComfyUI or GPU needed.

    python tests/test_h3_caption_workflow.py
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = []


def check(cond, msg):
    RESULTS.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + msg)


path = os.path.join(ROOT, "workflows", "pixel-art", "h3-caption.json")
before = open(path, "rb").read()
subprocess.run([sys.executable, os.path.join(ROOT, "tools", "build_pixel_h3_caption.py")], cwd=os.path.join(ROOT, "tools"),
               check=True, capture_output=True)
check(open(path, "rb").read() == before, "the committed workflow is what the builder writes")
wf = json.loads(before.decode("utf-8"))
nodes = {n["id"]: n for n in wf["nodes"]}
by_title = {n.get("title"): n for n in wf["nodes"] if n.get("title")}
links = {l[0]: l for l in wf["links"]}


def source(node, input_name):
    inp = next(i for i in node["inputs"] if i["name"] == input_name)
    return nodes[links[inp["link"]][1]] if inp.get("link") is not None else None


for title in ("Sprite (on white, enlarged x4)", "System prompt", "User prompt", "Caption writer 1 (first seed)",
              "Caption writer 2 (second seed)", "Caption", "Caption 2"):
    check(title in by_title, f"found by title: {title}")
check(len(by_title) == len([n for n in wf["nodes"] if n.get("title")]), "titles are unique")

# the model: Qwen3-VL 8B as a text model, loaded the way the cloud runs loaded it
clip = by_title["Load Qwen3-VL text model"]
check(clip["type"] == "CLIPLoader" and clip["widgets_values"] == ["qwen3vl_8b_int8_convrot.safetensors", "ideogram4", "default"],
      "CLIPLoader: qwen3vl_8b_int8_convrot, type ideogram4")
m = clip["properties"]["models"][0]
check(m["name"] == clip["widgets_values"][0] and m["directory"] == "text_encoders" and m["url"].startswith("https://huggingface.co/")
      and m["url"].endswith("/text_encoders/qwen3vl_8b_int8_convrot.safetensors"), "the text model is listed with its URL")
check(not any(n["type"] in ("UNETLoader", "VAELoader", "LoraLoaderModelOnly") for n in wf["nodes"]), "no video or image model is loaded")

# two TextGenerate runs, identical except the seed
a, b = by_title["Caption writer 1 (first seed)"], by_title["Caption writer 2 (second seed)"]
check(a["type"] == b["type"] == "TextGenerate", "two TextGenerate nodes")
va, vb = a["widgets_values"], b["widgets_values"]
check(va[:8] == vb[:8] and va[9:] == vb[9:], "identical except the seed")
check(va[1] == 160 and va[2] == "on" and va[3:8] == [0.3, 64, 0.95, 0.05, 1.05], "max_length 160, sampling on: temperature 0.3, top_k 64, top_p 0.95, min_p 0.05, repetition penalty 1.05")
check(va[8] == 7 and vb[8] == 8, "seeds 7 and 8 (sampling_mode.seed is the ninth widget)")
for node in (a, b):
    check(source(node, "clip") is clip and source(node, "image") is by_title["Sprite (on white, enlarged x4)"]
          and source(node, "prompt") is by_title["User prompt"] and source(node, "system_prompt") is by_title["System prompt"],
          f"{node['title']}: model, sprite, user prompt and system prompt come from the titled nodes")

# outputs by title (the job result returns PreviewAny text under the node's title)
check(source(by_title["Caption"], "source") is a and source(by_title["Caption 2"], "source") is b and by_title["Caption"]["type"] == "PreviewAny",
      "Caption = first seed, Caption 2 = second seed, both PreviewAny")

# the rules (v6) are the default system prompt
sysp = by_title["System prompt"]["widgets_values"][0]
check(sysp.startswith("You write action captions for a video model") and sysp.rstrip().endswith("Answer with the caption only."), "system prompt: the v6 text")
for rule in ("never behind the body", "pull it back over the shoulder with a slight lean", "brings the body, the weapon or the hand back to where it was",
             "Never write stride forward, sway, turn or travel", "Never a ring, burst or crescent around the body", "30 to 50 words"):
    check(rule in sysp, f"system prompt has the rule: {rule}")
check("thin trail following the weapon" in sysp and "thin trail behind" not in sysp, "v5/v6 wording: trail following, not behind")
user = by_title["User prompt"]["widgets_values"][0]
check(user.rstrip().endswith("Caption:") and "Subject:" in user and "Starting pose:" in user and "Request:" in user, "user prompt: the four lines, ending with Caption:")

# the repo's own lists
env = open(os.path.join(ROOT, "env", "pixel.env"), encoding="utf-8").read()
models_line = next(l for l in env.splitlines() if l.startswith("MODELS="))
check("pixel-art/h3-caption" in models_line.split("=", 1)[1].split(","), "env/pixel.env MODELS lists pixel-art/h3-caption")
readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
check("pixel-art/h3-caption" in readme, "README lists the workflow")

print(f"{sum(RESULTS)}/{len(RESULTS)} passed")
sys.exit(0 if all(RESULTS) else 1)
