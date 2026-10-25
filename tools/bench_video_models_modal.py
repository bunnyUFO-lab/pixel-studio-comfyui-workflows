"""Run the bench_video_models.py arms on the Modal job API (psc-job/v1) instead of a local ComfyUI, several clips at once.

  python tools/bench_video_models_modal.py --jobs-url https://<ws>--<app>-jobs.modal.run --sprites DIR --drivers DIR
                                           --out DIR [--arms a,b] [--motions idle,walk] [--sprite-names a,b]
                                           [--parallel 3] [--seed 7]

Same graphs as the local runner (build_prompt: every model frame, plus a RIFE x2 branch), converted with the
deployment's own object_info (GET /v1/info). Each clip is one job; its inputs travel inline and every file comes back
in the result. Writes OUT/<arm>/... and OUT/results.json in the local runner's format, plus the job's billed time,
GPU and cost. `seconds` is the job's run time (the first job of a container also pays `boot`, kept separately), so it
is comparable between clips, not with the local 4090. Needs MODAL_KEY / MODAL_SECRET (a Modal proxy auth token).
"""
import argparse
import base64
import concurrent.futures
import glob
import json
import os
import threading
import time
import urllib.error
import urllib.request

import bench_video_models as bench

comfy = bench.comfy
LOCK = threading.Lock()


def call(url, path, data=None, method=None, timeout=120):
    req = urllib.request.Request(url.rstrip("/") + path, data=json.dumps(data).encode() if data is not None else None,
                                 method=method or ("POST" if data is not None else "GET"))
    req.add_header("Modal-Key", os.environ["MODAL_KEY"])
    req.add_header("Modal-Secret", os.environ["MODAL_SECRET"])
    if data is not None:
        req.add_header("Content-Type", "application/json")
    for i in range(20):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code < 500:
                raise SystemExit(f"HTTP {e.code} on {path}: {e.read().decode(errors='replace')[:2000]}")
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(min(30, 3 * (i + 1)))
    raise SystemExit(f"no answer from {path}")


def build_job(arm, sprite, motion, driver, prefix, seed, object_info):
    files = []

    def inline(path):                                  # stands in for comfy.upload: the file travels in the job
        files.append(path)
        return os.path.basename(path)

    comfy.upload = inline
    api = bench.build_prompt(arm, sprite, motion, driver, prefix, seed, object_info)
    inputs = [{"name": os.path.basename(f), "data": base64.b64encode(open(f, "rb").read()).decode()} for f in files]
    return {"protocol": "psc-job/v1", "workflow": api, "inputs": inputs, "timeout": 3600}


def run(url, job, out_dir):
    jid = call(url, "/v1/jobs", job)["id"]
    t0 = time.time()
    while True:
        st = call(url, f"/v1/jobs/{jid}", timeout=300)
        if st["status"] in ("done", "failed", "cancelled"):
            break
        time.sleep(5)
    if st["status"] != "done":
        return jid, None, round(time.time() - t0, 1), [st.get("error") or st["status"]]
    res = st["result"]
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for f in res.get("files", []):
        if f.get("type") != "output":
            continue
        dest = os.path.join(out_dir, f["filename"])
        with open(dest, "wb") as fh:
            fh.write(base64.b64decode(f["data"]))
        saved.append(os.path.relpath(dest, os.path.dirname(out_dir)))
    res["saved"] = saved
    return jid, res, round(time.time() - t0, 1), res.get("errors", [])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jobs-url", required=True)
    p.add_argument("--sprites", required=True)
    p.add_argument("--drivers", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--arms", default=",".join(bench.ARMS))
    p.add_argument("--motions", default=",".join(bench.MOTIONS))
    p.add_argument("--sprite-names", default="")
    p.add_argument("--parallel", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    sprites = sorted(glob.glob(os.path.join(a.sprites, "*.png")))
    if a.sprite_names:
        keep = a.sprite_names.split(",")
        sprites = [s for s in sprites if os.path.splitext(os.path.basename(s))[0] in keep]
    results_path = os.path.join(a.out, "results.json")
    results = json.load(open(results_path, encoding="utf-8")) if os.path.exists(results_path) else []
    done = {(r["arm"], r["sprite"], r["motion"]) for r in results if r.get("status") == "success"}
    oi = call(a.jobs_url, "/v1/info", timeout=300)["object_info"]
    jobs = []
    for arm in a.arms.split(","):
        for motion in a.motions.split(","):
            drivers = sorted(glob.glob(os.path.join(a.drivers, f"driver_{motion}*.mp4")))
            for sprite in sprites:
                name = os.path.splitext(os.path.basename(sprite))[0]
                if (arm, name, motion) in done:
                    continue
                prefix = f"bench_fast/{arm}/{name}_{motion}"
                jobs.append((arm, name, motion, build_job(arm, sprite, motion, drivers[0] if drivers else None, prefix,
                                                          a.seed, oi)))
    print(f"{len(jobs)} jobs, {a.parallel} at a time", flush=True)

    def one(item):
        arm, name, motion, job = item
        jid, res, wall, errors = run(a.jobs_url, job, os.path.join(a.out, arm))
        text = (res or {}).get("text", {})
        plain, rife = bench.summarize(text.get("Check report", "")), bench.summarize(text.get("Check report RIFE", ""))
        timing, cost = (res or {}).get("timing", {}), (res or {}).get("cost", {})
        row = dict(arm=arm, sprite=name, motion=motion, status=(res or {}).get("status", "failed"), prompt_id=jid,
                   seconds=timing.get("run"), boot=timing.get("boot"), billed=timing.get("billed"), wall=wall,
                   gpu=cost.get("gpu"), cost_usd=cost.get("usd"), worker=(res or {}).get("worker"),
                   peak_vram_mib=None, check=plain[0], check_rife=rife[0],
                   popping=bench.summarize(text.get("Re-pixelate info", ""))[1],
                   popping_rife=bench.summarize(text.get("Re-pixelate info RIFE", ""))[1],
                   errors=errors, files=(res or {}).get("saved", []), where="modal")
        with LOCK:
            nonlocal_results = [x for x in json.load(open(results_path, encoding="utf-8"))] if os.path.exists(results_path) else []
            nonlocal_results = [x for x in nonlocal_results if (x["arm"], x["sprite"], x["motion"]) != (arm, name, motion)]
            nonlocal_results.append(row)
            with open(results_path, "w", encoding="utf-8") as fh:
                json.dump(nonlocal_results, fh, indent=1)
        print(f"{arm:13} {name:15} {motion:7} {row['status']:8} run {row['seconds'] or 0:6.1f}s boot {row['boot'] or 0:5.1f}s "
              f"${row['cost_usd'] or 0:.3f} {row['gpu'] or ''} check {plain[0]}/{rife[0]} {'; '.join(map(str, errors))[:120]}",
              flush=True)

    os.makedirs(a.out, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(a.parallel) as ex:
        list(ex.map(one, jobs))


if __name__ == "__main__":
    main()
