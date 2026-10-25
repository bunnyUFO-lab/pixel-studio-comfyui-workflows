"""SCAIL 2 tuning test (Sprite Studio card "Updating the video generator", tab "SCAIL tuning").

    python tools/tune_scail2.py prep --sprites DIR --clips DIR --masks DIR --out DIR [--target 640]
    python tools/tune_scail2.py run --sprites DIR --prepped DIR --views DIR --variants A,B,... --out DIR

prep: for each sprite x driving clip, a "clean" driving clip on the sprite's own video canvas: the actor cut out with
      per-frame masks (BiRefNet, one PNG per frame in <masks>/<clip>/), put on the sprite's flat background colour,
      scaled so the actor in frame 1 has the sprite's height, centred on the sprite with its feet on the sprite's foot
      line. Writes <out>/<target>/<sprite>__<clip>.mp4 and <out>/<target>/<sprite>.json (canvas, background colour).
run:  the variants below through the SCAIL 2 workflows, recording the same fields as bench_scail_wan.py.
"""
import argparse
import glob
import json
import os
import subprocess
import threading
import time
import sys

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "custom_nodes", "comfyui_workflows_utils"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
FPS = 16


def prepared(sprite_path, target):
    import pixel
    rgba = np.asarray(Image.open(sprite_path).convert("RGBA"), np.float32) / 255.0
    img, spec, w, h, _ = pixel.PrepareSpriteForVideo().run(torch.from_numpy(rgba[None]), target, 0.6, "auto", 32)
    return spec, w, h


def frames(clip):
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", clip, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                            "-of", "csv=p=0", clip], capture_output=True, text=True, check=True).stdout
    w, h = map(int, probe.strip().split(",")[:2])
    return np.frombuffer(out, np.uint8).reshape(-1, h, w, 3)


def prep(a):
    for sprite in sorted(glob.glob(os.path.join(a.sprites, "*.png"))):
        sname = os.path.splitext(os.path.basename(sprite))[0]
        spec, W, H = prepared(sprite, a.target)
        f = spec["factor"]
        (ox, oy), (sw, sh) = spec["offset"], spec["size"]
        bg = np.array(spec["bg"], np.float32)
        s_h, s_foot, s_cx = sh * f, (oy + sh) * f, (ox + sw / 2) * f
        d = os.path.join(a.out, str(a.target))
        os.makedirs(d, exist_ok=True)
        json.dump({"canvas": [W, H], "bg": spec["bg"], "factor": f}, open(os.path.join(d, sname + ".json"), "w"))
        for clip in sorted(glob.glob(os.path.join(a.clips, "*.mp4"))):
            cname = os.path.splitext(os.path.basename(clip))[0]
            masks = sorted(glob.glob(os.path.join(a.masks, cname, "*.png")))
            if not masks:
                continue
            fr = frames(clip)[:len(masks)]
            m0 = np.asarray(Image.open(masks[0]).convert("L")) > 127
            ys, xs = np.nonzero(m0)
            a_h, a_foot, a_cx = ys.max() - ys.min() + 1, ys.max() + 1, (xs.min() + xs.max() + 1) / 2
            k = s_h / a_h                                       # one scale for the clip, from frame 1
            dx, dy = s_cx - a_cx * k, s_foot - a_foot * k
            out = []
            for rgb, mp in zip(fr, masks):
                m = np.asarray(Image.open(mp).convert("L"), np.float32) / 255.0
                rgba = np.dstack([rgb.astype(np.float32), m * 255])
                big = Image.fromarray(rgba.astype(np.uint8), "RGBA").resize(
                    (max(1, round(rgb.shape[1] * k)), max(1, round(rgb.shape[0] * k))), Image.LANCZOS)
                canvas = Image.new("RGBA", (W, H), tuple(int(c) for c in bg) + (255,))
                canvas.alpha_composite(big, (int(round(dx)), int(round(dy)))) if dx >= 0 and dy >= 0 else \
                    canvas.paste(big, (int(round(dx)), int(round(dy))), big)
                out.append(np.asarray(canvas.convert("RGB")))
            dst = os.path.join(d, f"{sname}__{cname}.mp4")
            p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                                  "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "8", "-pix_fmt", "yuv444p", dst],
                                 stdin=subprocess.PIPE)
            p.communicate(np.stack(out).tobytes())
            print(f"{dst}: {len(out)} frames {W}x{H}, actor x{k:.2f}")


# --------------------------------------------------------------------------------------------------- run
SCAIL = os.path.join(ROOT, "workflows", "pixel-art", "sprite-from-video-scail2.json")
VIEWS = os.path.join(ROOT, "workflows", "pixel-art", "sprite-from-video-scail2-views.json")
PREP = "Prepare sprite (integer upscale, /32)"
COLOURS = {"blue": (0, 0, 255), "green": (0, 255, 0), "red": (255, 0, 0), "magenta": (255, 0, 255), "cyan": (0, 255, 255),
           "yellow": (255, 255, 0), "white": (255, 255, 255), "black": (0, 0, 0), "grey": (128, 128, 128),
           "orange": (255, 128, 0), "purple": (128, 0, 255), "pink": (255, 128, 192)}
# per sprite: (short name, long caption: art style, palette, outfit, proportions)
CAPTIONS = {
    "WiseBeard": ("old wizard", "an old wizard with a long white beard, a tall pointed grey hat, a long grey-blue robe that "
                  "reaches his feet, a wooden staff in one hand; short stocky body, head and hat about a third of his height, "
                  "short legs hidden under the robe"),
    "Adventurer": ("young adventurer", "a young adventurer with dark brown skin, a green cap, a red scarf, a knee-length green "
                   "tunic with a brown belt, green trousers and brown boots; chunky proportions, big head, short legs"),
    "GoblinWarrior": ("goblin warrior", "a lean green goblin warrior with long pointed ears, a dark leather vest, a loincloth, "
                      "bare green arms and legs and a short blade; hunched stance, long arms, short bowed legs"),
}
BASE_PROMPT = ("A pixel art game character with crisp pixels and flat colours performs the movement from the video on a "
               "plain flat background, static camera.")


def colour_name(rgb):
    return min(COLOURS, key=lambda n: sum((a - b) ** 2 for a, b in zip(COLOURS[n], rgb)))


def prompt(kind, sprite, bg):
    short, long_ = CAPTIONS[sprite]
    if kind == "short":
        return f"Full body view of a pixel-art {short} starting to move, on a plain flat {bg} background."
    if kind == "long":
        return (f"Pixel art game sprite of {long_}. Drawn in a clean 16-bit style: hard square pixels, a small flat palette, "
                f"thick dark outlines, no gradients, no blur, no anti-aliasing. The character keeps exactly these proportions "
                f"and outfit while performing the movement from the video, full body, centred, on a plain flat {bg} "
                f"background with no floor and no shadow, static camera.")
    return BASE_PROMPT[:-len(" plain flat background, static camera.")] + f" plain flat {bg} background, static camera."


def variants(best_prompt="long", best_after_c="E"):
    """Each variant: (workflow, uses the cleaned clips, prompt kind, extra settings, target)."""
    c = ["WanSCAILToVideo.pose_strength=0.7", "WanSCAILToVideo.pose_end=0.8"]
    d = c + ["SCAIL 2 DPO LoRA.strength_model=0.4"]
    e = d + ["lightx2v distill LoRA (6 steps).strength_model=0.6", "BasicScheduler.steps=10"]
    chain = {"C": c, "D": d, "E": e}
    return {
        "B": (SCAIL, True, "base", [], 640),
        "B-long": (SCAIL, True, "long", [], 640),
        "B-short": (SCAIL, True, "short", [], 640),
        "C": (SCAIL, True, best_prompt, c, 640),
        "D": (SCAIL, True, best_prompt, d, 640),
        "E": (SCAIL, True, best_prompt, e, 640),
        "F": (VIEWS, True, best_prompt, chain[best_after_c], 640),
        "G": (VIEWS, True, best_prompt, chain[best_after_c], 896),
    }


def _local_job(comfy, bench, workflow, inputs, job_sets, odir):
    """One job on the local ComfyUI: (collect() result, seconds, base MB, peak MB, extra fields)."""
    from argparse import Namespace
    import contextlib
    import io
    import re
    import time
    args = Namespace(workflow=workflow, input=inputs, set=job_sets, no_wait=True, dump_api=None, server=None, out=None,
                     timeout=0)
    base = bench.gpu_used_mb()
    meter = bench.PeakMeter()
    meter.start()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        comfy.cmd_run(args)
    pid = re.search(r"queued (\S+)", buf.getvalue()).group(1)
    while not (h := comfy.http(f"/history/{pid}").get(pid)) or not h.get("status", {}).get("status_str"):
        time.sleep(5)
    meter.stop.set()
    res = comfy.collect(pid, odir)
    st = {m[0]: m[1].get("timestamp") for m in h["status"].get("messages", [])}
    end = st.get("execution_success") or st.get("execution_error")
    secs = round((end - st["execution_start"]) / 1000, 1) if end else None
    return res, secs, base, meter.peak, {"prompt_id": pid, "where": "local RTX 4090"}


_MODAL_INFO = {}
_API_LOCK = threading.Lock()


def _modal_http(url, path, data=None):
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url.rstrip("/") + path, data=json.dumps(data).encode() if data is not None else None,
                                 method="POST" if data is not None else "GET")
    req.add_header("Modal-Key", os.environ["MODAL_KEY"])
    req.add_header("Modal-Secret", os.environ["MODAL_SECRET"])
    if data is not None:
        req.add_header("Content-Type", "application/json")
    for _ in range(20):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code < 500:
                raise SystemExit(f"Modal {path}: HTTP {e.code} {e.read()[:500]!r}")
        except (ConnectionError, TimeoutError, OSError):
            pass
        time.sleep(10)
    raise SystemExit(f"Modal {path}: no answer")


def _modal_job(comfy, url, workflow, inputs, job_sets, odir):
    """One job through the psc-job/v1 API: the UI workflow is converted with the deployment's object_info, inputs go
    inline. Returns the same shape as _local_job (no local GPU numbers)."""
    import base64
    import contextlib
    import io
    from argparse import Namespace
    if "oi" not in _MODAL_INFO:
        _MODAL_INFO["oi"] = _modal_http(url, "/v1/info")["object_info"]
    with _API_LOCK:                                   # cmd_run goes through module globals: one conversion at a time
        files, seen = [], {}
        real_http, real_upload = comfy.http, comfy.upload

        def fake_http(path, data=None, method=None, raw=False):
            if path == "/object_info":
                return _MODAL_INFO["oi"]
            if path == "/prompt":
                seen["api"] = data["prompt"]
                return {"prompt_id": "modal"}
            return real_http(path, data, method, raw)

        def fake_upload(path):
            files.append(path)
            return os.path.basename(path)
        comfy.http, comfy.upload = fake_http, fake_upload
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                comfy.cmd_run(Namespace(workflow=workflow, input=inputs, set=job_sets, no_wait=True, dump_api=None,
                                        server=None, out=None, timeout=0))
        finally:
            comfy.http, comfy.upload = real_http, real_upload
    doc = {"protocol": "psc-job/v1", "workflow": seen["api"], "timeout": 3600,
           "inputs": [{"name": os.path.basename(f), "data": base64.b64encode(open(f, "rb").read()).decode()} for f in files]}
    jid = _modal_http(url, "/v1/jobs", doc)["id"]
    while True:
        time.sleep(10)
        st = _modal_http(url, f"/v1/jobs/{jid}")
        if st["status"] in ("done", "failed", "cancelled"):
            break
    if st["status"] != "done":
        return ({"status": "error", "errors": [str(st.get("error"))], "files": [], "text": {}}, None, None, None,
                {"modal_job": jid, "where": "Modal"})
    r = st["result"]
    os.makedirs(odir, exist_ok=True)
    local = []
    for f in r["files"]:
        dst = os.path.join(odir, f["filename"])
        open(dst, "wb").write(base64.b64decode(f["data"]))
        local.append({"node": f["node"], "filename": f["filename"], "local": dst})
    res = {"status": r["status"], "errors": r["errors"], "files": local, "text": r["text"]}
    return res, round(r["timing"]["run"], 1), None, None, {
        "modal_job": jid, "where": f"Modal {r['cost'].get('gpu')}", "boot": r["timing"]["boot"],
        "usd": r["cost"].get("usd"), "worker": r.get("worker")}


def run(a):
    import re
    from concurrent.futures import ThreadPoolExecutor
    import bench_scail_wan as bench
    comfy = bench.comfy
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "results.json")
    state = {"results": json.load(open(path, encoding="utf-8")) if os.path.exists(path) else []}
    done = {(r["arm"], r["sprite"], r["clip"]) for r in state["results"] if r["status"] == "success"}
    table = variants(a.best_prompt, a.best_after_c)
    jobs = []
    for v in a.variants.split(","):
        workflow, clean, kind, sets, target = table[v]
        for sname in a.sprite_names.split(","):
            bg = colour_name(json.load(open(os.path.join(a.prepped, str(target), sname + ".json")))["bg"])
            for cname in a.clip_names.split(","):
                if (v, sname + ".png", cname + ".mp4") in done:
                    continue
                clip = (os.path.join(a.prepped, str(target), f"{sname}__{cname}.mp4") if clean
                        else os.path.join(a.clips, cname + ".mp4"))
                inputs = [f"Pixel sprite={os.path.join(a.sprites, sname + '.png')}", f"Driving video={clip}"]
                if workflow == VIEWS:
                    inputs += [f"Back view={os.path.join(a.views, sname, 'back.png')}",
                               f"Side view={os.path.join(a.views, sname, 'side.png')}"]
                job_sets = sets + [f"Prompt.text={json.dumps(prompt(kind, sname, bg))}", f"{PREP}.target_size={target}",
                                   "Sampler (seed here).noise_seed=11", f"{bench.REPIX}.deflicker=false"]
                jobs.append((v, sname, cname, workflow, inputs, job_sets, prompt(kind, sname, bg), sets))
    lock = threading.Lock()
    print(f"{len(jobs)} jobs on {'Modal' if a.modal else 'the local ComfyUI'}", flush=True)

    def one(job):
        v, sname, cname, workflow, inputs, job_sets, text, sets = job
        odir = os.path.join(a.out, v)
        if a.modal:
            res, secs, base, peak, extra = _modal_job(comfy, a.modal, workflow, inputs, job_sets, odir)
        else:
            res, secs, base, peak, extra = _local_job(comfy, bench, workflow, inputs, job_sets, odir)
        pop = re.search(r"popping ([\d.]+)%", res["text"].get("Re-pixelate info (colours, popping)", ""))
        iso = re.search(r"(\d+) isolated pixels", res["text"].get("Check report", ""))
        r = {"arm": v, "sprite": sname + ".png", "clip": cname + ".mp4", "seed": 11, "status": res["status"],
             "errors": res["errors"], "prompt": text, "sets": sets, "seconds": secs, "gpu_base_mb": base,
             "gpu_peak_mb": peak, "check_passed": res["text"].get("Check report", "").startswith("PASS"),
             "popping_pct": float(pop.group(1)) if pop else None, "isolated_px": int(iso.group(1)) if iso else None,
             "files": [os.path.join(v, os.path.relpath(f["local"], odir)) for f in res["files"] if f.get("local")],
             "texts": res["text"], **extra}
        with lock:
            state["results"] = [x for x in state["results"]
                                if (x["arm"], x["sprite"], x["clip"]) != (v, r["sprite"], r["clip"])] + [r]
            json.dump(state["results"], open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
            print(f"{v} {sname} {cname}: {r['status']} {secs}s check {'PASS' if r['check_passed'] else 'FAIL'} "
                  f"isolated {r['isolated_px']} popping {r['popping_pct']} ({extra['where']})", flush=True)

    with ThreadPoolExecutor(a.parallel if a.modal else 1) as pool:
        list(pool.map(one, jobs))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("prep")
    q.add_argument("--sprites", required=True)
    q.add_argument("--clips", required=True)
    q.add_argument("--masks", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--target", type=int, default=640)
    r = sub.add_parser("run")
    r.add_argument("--sprites", required=True)
    r.add_argument("--clips", required=True)
    r.add_argument("--prepped", required=True)
    r.add_argument("--views", required=True, help="<views>/<sprite>/back.png and side.png")
    r.add_argument("--variants", required=True)
    r.add_argument("--sprite-names", default="WiseBeard,Adventurer,GoblinWarrior")
    r.add_argument("--clip-names", default="walk,attack,turn_360")
    r.add_argument("--best-prompt", default="long", help="long or short: the prompt C onwards builds on")
    r.add_argument("--best-after-c", default="E", help="C, D or E: the settings F and G build on")
    r.add_argument("--out", required=True)
    r.add_argument("--modal", default="", help="jobs URL of a psc-job/v1 deployment: run there instead of locally")
    r.add_argument("--parallel", type=int, default=3, help="Modal jobs at once")
    a = p.parse_args()
    (prep if a.cmd == "prep" else run)(a)


if __name__ == "__main__":
    main()
