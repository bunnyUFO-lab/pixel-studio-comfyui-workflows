"""Benchmark the sprite-animation workflows: every arm x sprite x motion on the local ComfyUI.

  python tools/bench_video_models.py --sprites DIR --drivers DIR --out DIR [--arms a,b] [--motions idle,walk]
                                     [--sprite-names pyra,golem] [--seed 7] [--dry-run]

Sprites are DIR/<name>.png (native pixel art); drivers are DIR/driver_<motion>*.mp4 (a person doing the motion,
used by the video arms). Each clip gets two sheets from one generation: every model frame at the model's fps, and a
RIFE x2 branch added to the API prompt (both with Re-pixelate's deflicker off, so the model's own steadiness shows).
Writes OUT/<arm>/<sprite>_<motion>[_rife]_*.png/.webp/.json and OUT/results.json (time from the server's execution
timestamps, whole-GPU peak VRAM sampled with nvidia-smi, Pixel Art Check, colour popping). Re-running skips clips
already in results.json. Arms run one after another, so the first clip of each arm loads its models (cold) and the
rest are warm. Server: COMFY_URL (default http://127.0.0.1:8188).
"""
import argparse
import copy
import glob
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, ".claude", "skills", "comfyui", "scripts"))
import comfy  # noqa: E402

WF = os.path.join(ROOT, "workflows", "pixel-art")
STYLE = "Static camera, the character stays in place, flat solid background."
MOTIONS = {
    "idle": "The character does a looping idle animation: breathing, a slight up and down bob, clothes and hair "
            "swaying gently.",
    "walk": "The character walks in place with a steady walk cycle, legs stepping and arms swinging, facing the "
            "same direction.",
    "attack": "The character attacks: it winds up and swings its weapon in a powerful strike, then returns to its "
              "stance.",
    "spell": "The character casts a spell: it raises its arms, glowing magic energy and sparkling particles gather "
             "and burst outward in a bright flash, then it returns to its stance.",
}
# LTX-2 prompting guide (Lightricks): one flowing present-tense paragraph, the action first, then the movements in
# order, the character's look, the setting and the camera, under 200 words. {char} is the sprite's caption
# (DIR/<name>.txt next to the sprite), which a text-only arm needs and an image arm uses to hold the character.
LTX_LOOK = ("Its face, outfit and proportions stay exactly the same, it keeps facing the same way and stays in the "
            "centre of the frame. The camera is locked off and static, with flat even lighting, crisp pixel edges and a "
            "limited colour palette, like a classic 16-bit video game sprite animation.")
LTX_MOTIONS = {
    "idle": "A 2D pixel art game sprite of {char}, full body in three-quarter view on a flat solid background. The "
            "character stands in a relaxed looping idle: its chest rises and falls with slow breaths, its body bobs "
            "gently up and down, and loose cloth and hair sway a little, returning to the starting pose. ",
    "walk": "A 2D pixel art game sprite of {char}, full body in three-quarter view on a flat solid background. The "
            "character walks in place in a smooth looping walk cycle: one leg steps forward while the opposite arm "
            "swings forward, then the other leg steps forward while the other arm swings, the body rising and falling "
            "slightly with each step. It does not move across the frame. ",
    "attack": "A 2D pixel art game sprite of {char}, full body in three-quarter view on a flat solid background. The "
              "character draws its weapon back over its shoulder, swings it forward in one fast, powerful strike, "
              "then pulls it back and settles into its starting stance. ",
    "spell": "A 2D pixel art game sprite of {char}, full body in three-quarter view on a flat solid background. The "
             "character raises both arms, a small ball of glowing magic gathers between its hands and bursts outward "
             "into sparks that fade quickly, then it lowers its arms back to its starting stance. ",
}
# arm -> workflow, input kind, model fps, UI node mode changes {title: mode}, API overrides {title.input: value}
ARMS = {
    "remix": dict(wf="sprite-from-prompt", kind="prompt", fps=16),
    "animate2": dict(wf="sprite-from-video", kind="video", fps=16),
    "ti2v-fastwan": dict(wf="sprite-from-prompt-ti2v5b", kind="prompt", fps=24),
    "ti2v-turbo": dict(wf="sprite-from-prompt-ti2v5b", kind="prompt", fps=24,
                       set={"FastWan LoRA (few steps).lora_name": "Wan22_TI2V_5B_Turbo_lora_rank_64_fp16.safetensors"}),
    "ti2v-base": dict(wf="sprite-from-prompt-ti2v5b", kind="prompt", fps=24,
                      modes={"FastWan LoRA (few steps)": 4},
                      set={"Sampler (seed, steps).steps": 20, "Sampler (seed, steps).cfg": 5.0,
                           "Sampler (seed, steps).sampler_name": "uni_pc",
                           "Prepare sprite (integer upscale).target_size": 1024}),
    "vace-causvid": dict(wf="sprite-from-prompt-vace", kind="prompt", fps=16),
    "vace-plain": dict(wf="sprite-from-prompt-vace", kind="prompt", fps=16,
                       modes={"CausVid LoRA (8 steps, CFG 1; non-commercial)": 4},
                       set={"Sampler (seed, steps).steps": 20, "Sampler (seed, steps).cfg": 5.0,
                            "Sampler (seed, steps).sampler_name": "uni_pc"}),
    "vace-pose": dict(wf="sprite-from-video-vace", kind="video", fps=16),
    "ltx25": dict(wf="sprite-from-prompt-ltx25", kind="prompt", fps=24),
    "ltx25-enh": dict(wf="sprite-from-prompt-ltx25", kind="prompt", fps=24,
                      modes={"Prompt enhancer": 0, "Prompt enhancer model (Gemma 4 E2B)": 0},
                      set={"Use prompt enhancer.switch": True}),
    "ltx25-tp": dict(wf="sprite-from-prompt-ltx25", kind="prompt", fps=24, prompt="ltx"),
    # text only: the sprite is not a guide frame, it only sets the grid, palette and canvas for Re-pixelate
    "ltx25-t2v": dict(wf="sprite-from-prompt-ltx25", kind="prompt", fps=24, prompt="ltx", no_guides=True),
}
RIFE = "rife_v4.26.safetensors"


class VramSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self.stop = 0, threading.Event()

    def run(self):
        while not self.stop.is_set():
            try:
                used = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                          capture_output=True, text=True, timeout=5).stdout.split()[0])
                self.peak = max(self.peak, used)
            except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                pass
            self.stop.wait(0.5)


def retry(fn, *args, tries=30):
    """Polling survives a dropped connection (the server is busy loading a model)."""
    for i in range(tries):
        try:
            return fn(*args)
        except (ConnectionError, OSError):
            if i == tries - 1:
                raise
            time.sleep(5)


def node_by_title(api, title):
    ids = comfy.find_nodes(api, title)
    if not ids:
        raise SystemExit(f"no node '{title}' ({comfy.node_titles(api)})")
    return ids


def drop_guides(api):
    """Remove LTXVAddGuide / LTXVCropGuides, wiring their (positive, negative, latent) inputs straight through."""
    for nid in [k for k, v in api.items() if v["class_type"] in ("LTXVAddGuide", "LTXVCropGuides")]:
        ins = api.pop(nid)["inputs"]
        through = [ins["positive"], ins["negative"], ins["latent"]]
        for v in api.values():
            for k, x in v["inputs"].items():
                if isinstance(x, list) and len(x) == 2 and x[0] == nid:
                    v["inputs"][k] = through[x[1]]


def caption(sprite_file):
    path = os.path.splitext(sprite_file)[0] + ".txt"
    if not os.path.exists(path):
        raise SystemExit(f"no caption {path} (needed by the LTX-prompt arms)")
    return open(path, encoding="utf-8").read().strip()


def build_prompt(arm, sprite_file, motion, driver_file, prefix, seed, object_info):
    cfg = ARMS[arm]
    wf = comfy.load_workflow(os.path.join(WF, cfg["wf"] + ".json"))
    for n in wf["nodes"]:
        if n.get("title") in cfg.get("modes", {}):
            n["mode"] = cfg["modes"][n["title"]]
    api = comfy.to_api(wf, object_info)
    api[node_by_title(api, "Pixel sprite")[0]]["inputs"]["image"] = comfy.upload(sprite_file)
    if cfg["kind"] == "video":
        api[node_by_title(api, "Driving video")[0]]["inputs"]["video"] = comfy.upload(driver_file)
    else:
        mp = api[node_by_title(api, "Motion prompt")[0]]["inputs"]
        text = f"Pixel art sprite animation. {MOTIONS[motion]} {STYLE}"
        if cfg.get("prompt") == "ltx":
            text = LTX_MOTIONS[motion].format(char=caption(sprite_file)) + LTX_LOOK
        mp["text" if "text" in mp else "value"] = text
    if cfg.get("no_guides"):
        drop_guides(api)
    for key, value in cfg.get("set", {}).items():
        title, widget = key.rsplit(".", 1)
        for i in node_by_title(api, title):
            api[i]["inputs"][widget] = value
    for n in api.values():
        for k in ("seed", "noise_seed"):
            if not isinstance(n["inputs"].get(k, []), list) and n["inputs"].get("add_noise", True) in ("enable", True):
                n["inputs"][k] = seed
        if n["class_type"] == "VHS_VideoCombine":
            n["inputs"]["filename_prefix"] = prefix + "_raw"
        if n["class_type"] == "PixelArtCheck":
            n["inputs"]["consistent_height"] = False               # the same check on every arm (attacks change height)
        if n["class_type"] == "SpriteAnimationPreview":
            n["_drop"] = True
    api = {k: v for k, v in api.items() if not v.pop("_drop", False)}
    # one sheet with every model frame, one through RIFE x2 (RIFE nodes in the workflow are bypassed here)
    rep_id = next(k for k, v in api.items() if v["class_type"] == "RepixelateFrames")
    sheet_id = next(k for k, v in api.items() if v["class_type"] == "SaveSpriteSheet")
    rep, sheet = api[rep_id], api[sheet_id]
    src = rep["inputs"]["frames"]
    if api[src[0]]["class_type"] == "FrameInterpolate":
        src = api[src[0]]["inputs"]["images"]
        rep["inputs"]["frames"] = src
    rep["inputs"]["deflicker"] = False
    sheet["inputs"].update(filename_prefix=prefix, frame_step=1, fps=cfg["fps"], max_frames=0)
    api["9001"] = {"class_type": "FrameInterpolationModelLoader", "inputs": {"model_name": RIFE}, "_meta": {"title": "RIFE model"}}
    api["9002"] = {"class_type": "FrameInterpolate", "inputs": {"interp_model": ["9001", 0], "images": src, "multiplier": 2},
                   "_meta": {"title": "RIFE x2"}}
    api["9003"] = copy.deepcopy(rep)
    api["9003"]["inputs"]["frames"] = ["9002", 0]
    api["9003"]["_meta"] = {"title": "Re-pixelate info RIFE"}
    api["9004"] = copy.deepcopy(sheet)
    api["9004"]["inputs"].update(frames=["9003", 0], filename_prefix=prefix + "_rife", fps=cfg["fps"] * 2)
    api["9005"] = {"class_type": "PixelArtCheck", "inputs": {"images": ["9004", 1], "max_colors": 256, "expected_height": 0,
                                                              "consistent_height": False}, "_meta": {"title": "Check RIFE"}}
    api["9006"] = {"class_type": "PreviewAny", "inputs": {"source": ["9005", 0]}, "_meta": {"title": "Check report RIFE"}}
    api["9007"] = {"class_type": "PreviewAny", "inputs": {"source": ["9003", 1]}, "_meta": {"title": "Re-pixelate info RIFE"}}
    for k, v in api.items():                                           # previews of the plain branch, stable titles
        if v["class_type"] == "PreviewAny" and v["inputs"].get("source") == [rep_id, 1]:
            v["_meta"]["title"] = "Re-pixelate info"
    return api


def exec_seconds(prompt_id):
    h = comfy.http(f"/history/{prompt_id}")[prompt_id]
    ts = {m[0]: m[1].get("timestamp") for m in h.get("status", {}).get("messages", [])}
    if ts.get("execution_start") and (ts.get("execution_success") or ts.get("execution_error")):
        return ((ts.get("execution_success") or ts.get("execution_error")) - ts["execution_start"]) / 1000.0
    return None


def summarize(text):
    passed = "PASS" in text.upper().split("\n")[0] if text else None
    m = re.search(r"popping ([\d.]+)%", text or "")
    return passed, float(m.group(1)) if m else None


def run_clip(api, out_dir):
    res = comfy.http("/prompt", {"prompt": api, "client_id": "bench-video-models"})
    if res.get("node_errors"):
        raise SystemExit("node errors: " + json.dumps(res["node_errors"])[:2000])
    pid, vram = res["prompt_id"], VramSampler()
    while pid not in [j[1] for j in retry(comfy.http, "/queue")["queue_running"]] and not retry(comfy.collect, pid):
        time.sleep(1)                                  # wait for earlier jobs (other sessions) to finish
    vram.start()
    while True:
        r = retry(comfy.collect, pid)
        if r:
            break
        time.sleep(2)
    vram.stop.set()
    os.makedirs(out_dir, exist_ok=True)
    files = []
    for f in r["files"]:
        if f["type"] != "output":
            continue
        q = urllib.parse.urlencode({"filename": f["filename"], "subfolder": f["subfolder"], "type": "output"})
        dest = os.path.join(out_dir, f["filename"])
        with open(dest, "wb") as fh:
            fh.write(retry(lambda: comfy.http("/view?" + q, raw=True)))
        files.append(os.path.relpath(dest, os.path.dirname(out_dir)))
    return pid, r, vram.peak, files


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sprites", required=True)
    p.add_argument("--drivers", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--motions", default=",".join(MOTIONS))
    p.add_argument("--sprite-names", default="")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    sprites = sorted(glob.glob(os.path.join(a.sprites, "*.png")))
    if a.sprite_names:
        keep = a.sprite_names.split(",")
        sprites = [s for s in sprites if os.path.splitext(os.path.basename(s))[0] in keep]
    results_path = os.path.join(a.out, "results.json")
    results = json.load(open(results_path, encoding="utf-8")) if os.path.exists(results_path) else []
    done = {(r["arm"], r["sprite"], r["motion"]) for r in results if r.get("status") == "success"}
    oi = comfy.http("/object_info")
    for arm in a.arms.split(","):
        for motion in a.motions.split(","):
            drivers = sorted(glob.glob(os.path.join(a.drivers, f"driver_{motion}*.mp4")))
            for sprite in sprites:
                name = os.path.splitext(os.path.basename(sprite))[0]
                if (arm, name, motion) in done:
                    continue
                prefix = f"bench_fast/{arm}/{name}_{motion}"
                api = build_prompt(arm, sprite, motion, drivers[0] if drivers else None, prefix, a.seed, oi)
                if a.dry_run:
                    print(arm, name, motion, len(api), "nodes")
                    continue
                t0 = time.time()
                pid, r, peak, files = run_clip(api, os.path.join(a.out, arm))
                plain = summarize(r["text"].get("Check report", ""))
                rife = summarize(r["text"].get("Check report RIFE", ""))
                pop = summarize(r["text"].get("Re-pixelate info", ""))[1]
                pop_rife = summarize(r["text"].get("Re-pixelate info RIFE", ""))[1]
                row = dict(arm=arm, sprite=name, motion=motion, status=r["status"], prompt_id=pid,
                           seconds=exec_seconds(pid), wall=round(time.time() - t0, 1), peak_vram_mib=peak,
                           check=plain[0], check_rife=rife[0], popping=pop, popping_rife=pop_rife,
                           errors=r["errors"], files=files)
                results = [x for x in results if (x["arm"], x["sprite"], x["motion"]) != (arm, name, motion)] + [row]
                with open(results_path, "w", encoding="utf-8") as fh:
                    json.dump(results, fh, indent=1)
                print(f"{arm:13} {name:15} {motion:7} {r['status']:8} {row['seconds'] or 0:6.1f}s  "
                      f"peak {peak} MiB  check {plain[0]}/{rife[0]}  popping {pop}/{pop_rife}", flush=True)


if __name__ == "__main__":
    main()
