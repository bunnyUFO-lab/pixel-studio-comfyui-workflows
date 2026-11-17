"""Every model a workflow loads must be listed in its properties.models with a URL, because the cloud worker and
docker/models.py download only what a workflow lists. Checks the loader nodes of every workflow (bypassed ones are skipped).

    python tests/test_model_lists.py
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# loader node type -> its model folder
LOADERS = {"UNETLoader": "diffusion_models", "CLIPLoader": "text_encoders", "DualCLIPLoader": "text_encoders",
           "VAELoader": "vae", "LoraLoaderModelOnly": "loras", "LoraLoader": "loras", "CLIPVisionLoader": "clip_vision",
           "ControlNetLoader": "controlnet", "UpscaleModelLoader": "upscale_models",
           "FrameInterpolationModelLoader": "frame_interpolation", "LoadBackgroundRemovalModel": "background_removal"}
# files a workflow names that cannot be downloaded: not published yet. Report them, do not hide them.
UNPUBLISHED = {"sprite-studio/klein-moss.safetensors": "our own Flux 2 Klein LoRA (prompt-to-pixel-klein): upload it, add its URL"}

RESULTS = []


def check(cond, msg):
    RESULTS.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + msg)


for path in sorted((ROOT / "workflows").rglob("*.json")):
    wf = json.loads(path.read_text(encoding="utf-8"))
    name = path.relative_to(ROOT / "workflows").with_suffix("").as_posix()
    for n in wf.get("nodes", []):
        folder = LOADERS.get(n["type"])
        if not folder or n.get("mode") in (2, 4):
            continue
        wv = n.get("widgets_values") or []
        if not wv or not isinstance(wv[0], str):
            continue
        models = n.get("properties", {}).get("models", []) or []
        listed = {m["name"]: m for m in models if m.get("name")}
        if wv[0] in UNPUBLISHED:
            print(f"NOTE {name}: {wv[0]} is not published: {UNPUBLISHED[wv[0]]}")
            continue
        m = listed.get(wv[0])
        check(m is not None and m.get("url", "").startswith("https://") and m.get("directory") == folder,
              f"{name}: {n['type']} {wv[0]} is listed with a URL in {folder}")

print(f"{sum(RESULTS)}/{len(RESULTS)} passed")
sys.exit(0 if all(RESULTS) else 1)
