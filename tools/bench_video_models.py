"""Benchmark the sprite-from-video motion models: every arm x sprite x driving clip x seed, on the local ComfyUI.

    python tools/bench_video_models.py --sprites DIR --clips DIR [--arms A,B] [--seeds 11] [--out output/bench-video]

Runs arm by arm (so each model loads once), skips jobs already in results.json (re-run to resume), and records for
each job: execution time (from ComfyUI's own timestamps), peak GPU memory over the job (nvidia-smi, with the memory
in use just before it as the baseline), Pixel Art Check, colour popping and the output files. Repixelate's deflicker
is off in every arm, so flicker shows. Arms are defined in ARMS below.
"""
import argparse
import contextlib
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
from argparse import Namespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, ".claude", "skills", "comfyui", "scripts"))
import comfy  # noqa: E402

WAN = os.path.join(ROOT, "workflows", "pixel-art", "sprite-from-video.json")
SCAIL = os.path.join(ROOT, "workflows", "pixel-art", "sprite-from-video-scail2.json")
WAN_PREP, SCAIL_PREP = "Prepare sprite (integer upscale)", "Prepare sprite (integer upscale, /32)"
REPIX = "Repixelate (height match, same grid + palette)"
ZH_PROMPT = ("人物外观描述：一个像素风游戏角色精灵，清晰的像素和平涂颜色。背景描述：背景为纯色，光线均匀，无阴影，无其他物体。"
             "人物动作的参考视频。")

# arm -> (workflow, title of its sprite-prep node, extra "Node.widget=value" settings)
ARMS = {
    "A": (WAN, WAN_PREP, []),                                                   # today: 6 steps + lightx2v
    "A-zh": (WAN, WAN_PREP, [f"Prompt.text={json.dumps(ZH_PROMPT, ensure_ascii=False)}"]),
    "A-d": (WAN, WAN_PREP, ['#1.unet_name="wan_animate_2_distill_int8_convrot.safetensors"',
                            "lightx2v distill LoRA (6 steps).strength_model=0", "BasicScheduler.steps=10"]),
    "B": (SCAIL, SCAIL_PREP, []),                                               # SCAIL 2: 6 steps + DPO + lightx2v
    "A-fast": (WAN, WAN_PREP, ["BasicScheduler.steps=4", f"{WAN_PREP}.target_size=512"]),
    "B-fast": (SCAIL, SCAIL_PREP, ["BasicScheduler.steps=4", f"{SCAIL_PREP}.target_size=512"]),
}


def gpu_used_mb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.split()[0])


class PeakMeter(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self.stop = 0, threading.Event()

    def run(self):
        while not self.stop.is_set():
            self.peak = max(self.peak, gpu_used_mb())
            self.stop.wait(0.5)


def queue_idle():
    q = comfy.http("/queue")
    return not (q["queue_running"] or q["queue_pending"])


def run_job(arm, sprite, clip, seed, out_dir):
    workflow, _, sets = ARMS[arm]
    args = Namespace(workflow=workflow, input=[f"Pixel sprite={sprite}", f"Driving video={clip}"],
                     set=sets + [f"Sampler (seed here).noise_seed={seed}", f"{REPIX}.deflicker=false"],
                     no_wait=True, dump_api=None, server=None, out=None, timeout=0)
    base = gpu_used_mb()
    meter = PeakMeter()
    meter.start()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        comfy.cmd_run(args)
    pid = re.search(r"queued (\S+)", buf.getvalue()).group(1)
    while not (h := comfy.http(f"/history/{pid}").get(pid)) or not h.get("status", {}).get("status_str"):
        time.sleep(5)
    meter.stop.set()
    res = comfy.collect(pid, out_dir)
    stamps = {m[0]: m[1].get("timestamp") for m in h["status"].get("messages", [])}
    end = stamps.get("execution_success") or stamps.get("execution_error") or stamps.get("execution_interrupted")
    report = res["text"].get("Check report", "")
    popping = re.search(r"popping ([\d.]+)%", res["text"].get("Re-pixelate info (colours, popping)", ""))
    sheet = next((f.get("local") for f in res["files"] if f["node"] == "Save sprite sheet" and f["filename"].endswith(".png")
                  and not f["filename"].endswith("_preview.png")), None)
    return {"arm": arm, "sprite": os.path.basename(sprite), "clip": os.path.basename(clip), "seed": seed,
            "prompt_id": pid, "status": res["status"], "errors": res["errors"],
            "seconds": round((end - stamps["execution_start"]) / 1000, 1) if end and "execution_start" in stamps else None,
            "gpu_base_mb": base, "gpu_peak_mb": meter.peak,
            "check_passed": report.startswith("PASS"), "popping_pct": float(popping.group(1)) if popping else None,
            "sheet": os.path.relpath(sheet, out_dir) if sheet else None,
            "files": [os.path.relpath(f["local"], out_dir) for f in res["files"] if f.get("local")],
            "texts": res["text"]}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sprites", required=True, help="folder of sprite PNGs")
    p.add_argument("--clips", required=True, help="folder of driving videos (.mp4)")
    p.add_argument("--arms", default=",".join(ARMS), help=f"comma-separated, from {', '.join(ARMS)}")
    p.add_argument("--seeds", default="11", help="comma-separated seeds")
    p.add_argument("--out", default=os.path.join(ROOT, "output", "bench-video"))
    p.add_argument("--wait-idle", type=int, default=0, help="seconds to wait for an empty ComfyUI queue before each job")
    a = p.parse_args()
    sprites = sorted(os.path.join(a.sprites, f) for f in os.listdir(a.sprites) if f.lower().endswith(".png"))
    clips = sorted(os.path.join(a.clips, f) for f in os.listdir(a.clips) if f.lower().endswith((".mp4", ".webm", ".mov")))
    seeds = [int(s) for s in a.seeds.split(",")]
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "results.json")
    results = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else []
    done = {(r["arm"], r["sprite"], r["clip"], r["seed"]) for r in results if r["status"] == "success"}
    jobs = [(arm, s, c, seed) for arm in a.arms.split(",") for s in sprites for c in clips for seed in seeds
            if (arm, os.path.basename(s), os.path.basename(c), seed) not in done]
    print(f"{len(jobs)} jobs ({len(done)} already done) -> {a.out}")
    for i, (arm, s, c, seed) in enumerate(jobs, 1):
        t0 = time.time()
        while a.wait_idle and not queue_idle() and time.time() - t0 < a.wait_idle:
            time.sleep(10)
        r = run_job(arm, s, c, seed, os.path.join(a.out, arm))
        r["files"] = [os.path.join(arm, f) for f in r["files"]]
        r["sheet"] = r["sheet"] and os.path.join(arm, r["sheet"])
        results = [x for x in results if (x["arm"], x["sprite"], x["clip"], x["seed"]) != (arm, r["sprite"], r["clip"], seed)]
        results.append(r)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=1, ensure_ascii=False)
        print(f"[{i}/{len(jobs)}] {arm} {r['sprite']} {r['clip']} seed {seed}: {r['status']} {r['seconds']}s "
              f"peak {r['gpu_peak_mb'] - r['gpu_base_mb']} MB check {'PASS' if r['check_passed'] else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
