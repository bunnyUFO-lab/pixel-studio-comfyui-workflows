"""CPU tests for the Flux 2 Klein 4B still-sprite workflow (workflows/pixel-art/prompt-to-pixel-klein.json) and its builder.
No ComfyUI or GPU needed.

    python tests/test_prompt_to_pixel_klein.py
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


path = os.path.join(ROOT, "workflows", "pixel-art", "prompt-to-pixel-klein.json")
before = open(path, "rb").read()
subprocess.run([sys.executable, os.path.join(ROOT, "tools", "build_pixel_prompt_to_pixel_klein.py")], cwd=os.path.join(ROOT, "tools"),
               check=True, capture_output=True)
check(open(path, "rb").read() == before, "the committed workflow is what the builder writes")
wf = json.loads(before.decode("utf-8"))
nodes = {n["id"]: n for n in wf["nodes"]}
by_title = {n.get("title"): n for n in wf["nodes"] if n.get("title")}
links = {l[0]: l for l in wf["links"]}


def source(node, input_name):
    inp = next(i for i in node["inputs"] if i["name"] == input_name)
    return nodes[links[inp["link"]][1]] if inp.get("link") is not None else None


# inputs the app sets by title, as in prompt-to-pixel and prompt-to-pixel-krea2
for title in ("Pixel Art Prompt", "Pixel-art LoRA", "Pixelize", "Pixel Art Check", "Noise (seed here)"):
    check(title in by_title, f"input by title: {title}")
check(wf["nodes"][0] is not None and {n["type"] for n in wf["nodes"]} >= {"PixelArtPrompt", "Pixelize", "CleanCutout", "PixelArtCheck"},
      "the same prompt, cut-out, pixelize and check nodes as the other prompt-to-pixel workflows")
prompt_node = by_title["Pixel Art Prompt"]
check(prompt_node["widgets_values"][1] == "LoRA caption", "the prompt node writes the caption form the Klein LoRA was trained on")
check(prompt_node["widgets_values"][8] == "solid magenta (chroma key)", "background is magenta (both Klein LoRAs obey it)")

# the rated recipe: 4-step Klein, Qwen3-4B (flux2), flux2 VAE, 4 steps, euler, CFG 1 with a zeroed negative, 1024 px
unet = next(n for n in wf["nodes"] if n["type"] == "UNETLoader")
check(unet["widgets_values"][0] == "flux-2-klein-4b.safetensors", "the 4-step Klein 4B model")
check(next(n for n in wf["nodes"] if n["type"] == "CLIPLoader")["widgets_values"][:2] == ["qwen_3_4b.safetensors", "flux2"], "Qwen3-4B text encoder, type flux2")
check(next(n for n in wf["nodes"] if n["type"] == "Flux2Scheduler")["widgets_values"][0] == 4, "4 steps")
check(next(n for n in wf["nodes"] if n["type"] == "KSamplerSelect")["widgets_values"][0] == "euler", "euler")
guider = next(n for n in wf["nodes"] if n["type"] == "CFGGuider")
check(guider["widgets_values"][0] == 1.0 and source(guider, "negative")["type"] == "ConditioningZeroOut", "CFG 1, negative zeroed out")
check(source(guider, "model") is by_title["Pixel-art LoRA"], "the LoRA slot sits between the model and the guider")
check(by_title["Pixel-art LoRA"]["widgets_values"][1] == 1.0, "LoRA strength 1.0")
check(next(n for n in wf["nodes"] if n["type"] == "EmptyFlux2LatentImage")["widgets_values"][:2] == [1024, 1024], "1024 x 1024")

# models: each loader names a file with a URL, and the names match the widget values. One known exception: our own
# LoRA is not published, so it has no URL; the public alternative is listed instead.
UNPUBLISHED = "sprite-studio/klein-moss.safetensors"
need = {"UNETLoader": "diffusion_models", "CLIPLoader": "text_encoders", "VAELoader": "vae", "LoraLoaderModelOnly": "loras",
        "LoadBackgroundRemovalModel": "background_removal"}
for n in wf["nodes"]:
    if n["type"] in need:
        ms = n["properties"].get("models", [])
        names = [m["name"] for m in ms]
        urls_ok = all(m.get("url", "").startswith("https://huggingface.co/") and m["directory"] == need[n["type"]] for m in ms)
        if n["widgets_values"][0] == UNPUBLISHED:
            check(urls_ok and names == ["adirik-pixelart-lora-klein-4b.safetensors"],
                  f"{UNPUBLISHED} is not published (no URL possible); the public alternative is listed with its URL")
        else:
            check(n["widgets_values"][0] in names and urls_ok, f"{n['type']}: {n['widgets_values'][0]} is listed with a URL")
check(by_title["Pixel-art LoRA"]["widgets_values"][0] == UNPUBLISHED, "the LoRA slot holds our own Klein LoRA (the 9.5 recipe)")
check(prompt_node["widgets_values"][12] == "mosspx", "its trigger word is mosspx")

print(f"{sum(RESULTS)}/{len(RESULTS)} passed")
sys.exit(0 if all(RESULTS) else 1)
