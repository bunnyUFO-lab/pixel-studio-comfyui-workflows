"""CPU tests for the RunPod worker (cloud/runpod/handler.py): a fake ComfyUI on 127.0.0.1, no runpod, boto3 or GPU.
    python tests/test_runpod_handler.py"""
import asyncio
import base64
import http.server
import json
import os
import sys
import tempfile
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "cloud"))
sys.path.insert(0, os.path.join(ROOT, "cloud", "runpod"))
import job_runner as jr  # noqa: E402
import handler as h  # noqa: E402

RESULTS = []
REAL_PROCESS = jr.ComfyProcess


def check(cond, msg):
    RESULTS.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + msg)


def raises(fn, kind, text):
    try:
        fn()
    except Exception as e:
        return type(e) is kind and text in str(e)
    return False


def b64(data):
    return base64.b64encode(data).decode()


# ----------------------------------------------------------------------------- fake ComfyUI (as in test_job_runner.py)
class Fake(http.server.BaseHTTPRequestHandler):
    server_version = "fake"
    mode = "ok"                  # ok | big | hang (the prompt ends only when interrupted)
    outdir = tempdir = indir = None
    history = {}
    posts = []
    interrupted = threading.Event()

    def log_message(self, *a):
        pass

    def send(self, code, obj):
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/system_stats":
            return self.send(200, {"system": {"comfyui_version": "0.38.2"},
                                   "devices": [{"name": "cuda:0 NVIDIA GeForce RTX 4090 : cudaMallocAsync"}]})
        if self.path.startswith("/history/"):
            pid = self.path.rsplit("/", 1)[1]
            return self.send(200, {pid: Fake.history[pid]} if pid in Fake.history else {})
        if self.path == "/models":
            return self.send(200, ["checkpoints"])
        if self.path == "/models/checkpoints":
            return self.send(200, ["a.safetensors"])
        if self.path == "/object_info":
            return self.send(200, {"LoadImage": {}})
        self.send(404, {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        Fake.posts.append((self.path, body))
        if self.path == "/history":
            Fake.history.clear()
            return self.send(200, {})
        if self.path == "/interrupt":
            Fake.interrupted.set()
            if Fake.mode == "hang":
                Fake.history["p2"] = {"status": {"status_str": "error", "completed": False, "messages": []}, "outputs": {}}
            return self.send(200, {})
        if self.path == "/queue":
            return self.send(200, {})
        if self.path != "/prompt":
            return self.send(404, {})
        if Fake.mode == "hang":
            return self.send(200, {"prompt_id": "p2", "number": 2, "node_errors": {}})
        time.sleep(0.05)
        with open(os.path.join(Fake.outdir, "pixel_00001_.webp"), "wb") as fh:
            fh.write(b"WEBP" * 10)
        with open(os.path.join(Fake.outdir, "pixel_00001_.json"), "wb") as fh:
            fh.write(b'{"k": 1}')
        os.makedirs(os.path.join(Fake.outdir, "sub"), exist_ok=True)
        with open(os.path.join(Fake.outdir, "sub", "other.png"), "wb") as fh:
            fh.write(b"\x89PNG" * 5 if Fake.mode != "big" else b"x" * 2000)
        Fake.history["p1"] = {"status": {"status_str": "success", "completed": True, "messages": []}, "outputs": {
            "7": {"images": [{"filename": "pixel_00001_.webp", "subfolder": "", "type": "output"}]}}}
        self.send(200, {"prompt_id": "p1", "number": 1, "node_errors": {}})


class FakeProc:
    def __init__(self, root, url):
        self.url = url
        self.input_dir, self.output_dir, self.temp_dir = (os.path.join(root, d) for d in ("input", "output", "temp"))
        for d in (self.input_dir, self.output_dir, self.temp_dir):
            os.makedirs(d, exist_ok=True)

    stopped = 0
    up = True

    def gpu_name(self):
        return REAL_PROCESS.gpu_name(self)

    def alive(self):
        return self.up

    def stop(self):
        self.stopped += 1

    def info(self):
        return {"object_info": {"LoadImage": {}}, "models": {"checkpoints": ["a.safetensors"]}, "comfyui_version": "0.38.2"}


GRAPH = {"1": {"class_type": "LoadImage", "inputs": {"image": "ss_cat.png"}, "_meta": {"title": "Input image"}},
         "7": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}, "_meta": {"title": "Save pixel art"}}}


def request(**kw):
    doc = {"protocol": jr.PROTOCOL, "workflow": GRAPH, "inputs": [{"name": "ss_cat.png", "data": b64(b"PNGDATA")}]}
    doc.update(kw)
    return {"id": "job-1", "input": doc}


class StubProgress:
    """Stands in for job_runner._Progress (which needs ComfyUI's WebSocket): reports one update when it starts."""
    shape = {"node": "7", "value": 3, "max": 10, "done": ["1"], "total": 2}

    def __init__(self, url, client_id, total, callback):
        self.callback = callback

    def start(self):
        self.callback(dict(StubProgress.shape))

    def close(self):
        pass


# ----------------------------------------------------------------------------- gpu_rates.json
with open(os.path.join(ROOT, "cloud", "runpod", "gpu_rates.json"), encoding="utf-8") as fh:
    rates = json.load(fh)
for name, key in (("NVIDIA GeForce RTX 4090", "RTX-4090"), ("NVIDIA L40S", "L40S"), ("NVIDIA L40", "L40"),
                  ("NVIDIA RTX 6000 Ada Generation", "RTX-6000-ADA"),
                  ("NVIDIA RTX PRO 6000 Blackwell Server Edition", "RTX-PRO-6000"), ("NVIDIA RTX A6000", "A6000"),
                  ("NVIDIA GeForce RTX 5090", "RTX-5090"), ("NVIDIA RTX 4000 Ada Generation", "RTX-4000-ADA"),
                  ("NVIDIA A40", "A40"), ("NVIDIA L4", "L4"), ("NVIDIA A100-SXM4-80GB", "A100-80GB"),
                  ("NVIDIA H100 80GB HBM3", "H100")):
    check(jr.gpu_rate(name, rates)[0] == key, f"gpu_rates: {name!r} -> {key}")
check(jr.gpu_rate("NVIDIA GeForce RTX 4090", rates) == ("RTX-4090", 1.10), "gpu_rates: RTX-4090 costs 1.10 an hour")
check(all(k in rates["rates"] for k in rates["match"].values()), "gpu_rates: every match points at a rate")

# ----------------------------------------------------------------------------- the handler
server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
threading.Thread(target=server.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{server.server_address[1]}"
os.environ["RUNPOD_POD_ID"] = "pod-abc"
for var in ("PSC_S3_BUCKET", "PSC_INLINE_MAX_BYTES", "PSC_S3_PREFIX"):
    os.environ.pop(var, None)

with tempfile.TemporaryDirectory() as root:
    proc = FakeProc(root, url)
    Fake.indir, Fake.outdir, Fake.tempdir = proc.input_dir, proc.output_dir, proc.temp_dir
    h.PROC, h.BOOT, h.BOOT_PENDING = proc, 7.5, True

    # success, first job carries the boot
    seen = []
    real_progress = jr._Progress
    jr._Progress = StubProgress
    try:
        r = h.handler(request(), progress=seen.append)
        check(seen == [StubProgress.shape], f"progress: the callback got the protocol's shape ({seen})")
        r_bad = h.handler(request(), progress=lambda p: 1 / 0)
        check(r_bad["status"] == "success", "progress: a raising callback does not fail the job")
    finally:
        jr._Progress = real_progress
    by = {f["filename"]: f for f in r["files"]}
    check(r["status"] == "success" and not r["errors"] and "error" not in r, "handler: success, no top-level 'error' key")
    check(base64.b64decode(by["pixel_00001_.webp"]["data"]) == b"WEBP" * 10 and by["pixel_00001_.webp"]["node"] == "Save pixel art",
          "handler: the listed file, base64 in the result")
    check(by["pixel_00001_.json"]["node"] == "" and by["other.png"]["subfolder"] == "sub", "handler: the unlisted sibling and the subfolder file")
    json.dumps(r)
    check(r["worker"] == "pod-abc", "handler: worker is RUNPOD_POD_ID")
    check(r["cost"]["gpu"] == "NVIDIA GeForce RTX 4090" and r["cost"]["rate_per_hour"] == 1.10
          and abs(r["cost"]["usd"] - r["timing"]["billed"] * 1.10 / 3600) < 1e-5, "handler: cost from the RunPod rates")
    check(r["timing"]["boot"] == 7.5, "handler: the first job carries the boot seconds")
    r2 = h.handler(request())
    check(r2["timing"]["boot"] == 0, "handler: the second job's boot is 0")
    check(h.JOB_LOCK.acquire(blocking=False) and not h.JOB_LOCK.release(), "handler: the job lock is free after a job")

    # an infrastructure failure raises as RuntimeError; a bad document as ValueError
    class Boom(Exception):
        pass

    def boom(*a, **k):
        raise Boom("no gpu")
    real_run = jr.run_job
    jr.run_job = boom
    try:
        check(raises(lambda: h.handler(request()), RuntimeError, "Boom: no gpu"), "handler: infrastructure errors raise RuntimeError(Type: message)")
        check(h.PROC is None and proc.stopped == 1, "handler: an infrastructure error recycles ComfyUI (stopped, next job starts a new one)")
    finally:
        jr.run_job = real_run
        h.PROC = proc
    # a ComfyUI that died between jobs is replaced before the next job, which then carries the new boot time
    class Fresh(FakeProc):
        def __init__(self, comfy_dir, port=8188):
            super().__init__(root, url)

        def start(self, extra_args=()):
            return self

        def wait_ready(self, timeout=600):
            pass
    proc.up = False
    real_cls = jr.ComfyProcess
    jr.ComfyProcess = Fresh
    try:
        r = h.handler(request())
    finally:
        jr.ComfyProcess = real_cls
    check(r["status"] == "success" and proc.stopped == 2 and h.PROC is not proc and h.BOOT_PENDING is False and r["timing"]["boot"] >= 0,
          "handler: a ComfyUI that died between jobs is replaced and the job runs")
    proc.up = True
    h.PROC = proc
    check(raises(lambda: h.handler(request(workflow={})), ValueError, "workflow"), "handler: a bad document raises ValueError")
    check(h.JOB_LOCK.acquire(blocking=False) and not h.JOB_LOCK.release(), "handler: the job lock is free after failures")

    # input over the budget
    os.environ["PSC_INLINE_MAX_BYTES"] = "100"
    check(raises(lambda: h.handler(request()), ValueError, "over the inline limit"), "handler: input over PSC_INLINE_MAX_BYTES raises ValueError")
    try:
        h.handler(request())
    except ValueError as e:
        check(str(len(json.dumps(request()["input"]))) in str(e), f"handler: the error names the size ({e})")

    # result over the budget
    in_size = len(json.dumps(request()["input"]))
    os.environ["PSC_INLINE_MAX_BYTES"] = str(in_size + 300)
    Fake.mode = "big"
    up = []

    def uploader(key, data):
        up.append((key, data))
        return "https://example.test/" + key, 1900000000
    r = h.handler(request(), uploader=uploader)
    check(r["status"] == "success" and r["files"] and all("url" in f and "expires" in f and "data" not in f for f in r["files"]),
          "big result: every file has url and expires, no data")
    check(len(up) == len(r["files"]) and all(k.startswith("psc-jobs/job-1/") for k, _ in up), f"big result: uploads under <prefix><job id>/ ({[k for k, _ in up]})")
    check(dict(up).get("psc-jobs/job-1/output/sub/other.png") == b"x" * 2000, "big result: the uploader got the bytes, key <type>/<subfolder>/<filename>")
    check(next(f for f in r["files"] if f["filename"] == "other.png")["size"] == 2000, "big result: size stays the decoded size")
    os.environ["PSC_S3_PREFIX"] = "p/"
    up.clear()
    h.handler(request(), uploader=uploader)
    check(up and all(k.startswith("p/job-1/") for k, _ in up), "big result: PSC_S3_PREFIX is the key prefix")
    os.environ.pop("PSC_S3_PREFIX")
    r = h.handler(request())
    check(r["status"] == "error" and r["files"] == [] and any("over RunPod's inline limit" in e and "PSC_S3_" in e for e in r["errors"]),
          f"big result without a bucket: status error naming the limit ({r['errors']})")
    check(os.listdir(proc.output_dir) == [] and os.listdir(proc.input_dir) == [], "big result: the worker's folders are cleaned")
    Fake.mode = "ok"
    os.environ.pop("PSC_INLINE_MAX_BYTES")

    # actions
    check(h.handler({"input": {"action": "info"}})["comfyui_version"] == "0.38.2", "action info: ComfyProcess.info()")
    for bad in ({"input": {"action": "nope"}}, {"input": {}}, {"input": "x"}, {}):
        check(raises(lambda bad=bad: h.handler(bad), ValueError, "unknown input"), f"unknown input {json.dumps(bad)} raises ValueError")
    check(h._workflows("a/b, c") == ["a/b", "c"] and h._workflows("starter")[0] == "pixel-art/image-to-pixel", "download: workflow list expansion")
    pix = os.path.join(root, "pixel.env")
    with open(pix, "w") as fh:
        fh.write("# x\nMODELS=a/one,b/two\n")
    old = h.PIXEL_ENV
    h.PIXEL_ENV = pix
    try:
        check(h._workflows("pixel") == ["a/one", "b/two"] and h._workflows(None) == ["a/one", "b/two"], "download: 'pixel' reads MODELS from pixel.env")
    finally:
        h.PIXEL_ENV = old

    # _settle: an orphaned job (the lock is held) is interrupted, then the job runs
    Fake.posts.clear()
    Fake.interrupted.clear()
    h.JOB_LOCK.acquire()

    def orphan():
        Fake.interrupted.wait(10)
        time.sleep(0.2)
        h.JOB_LOCK.release()
    threading.Thread(target=orphan, daemon=True).start()
    r = h.handler(request())
    paths = [p for p, _ in Fake.posts]
    check(r["status"] == "success", "settle: the job ran once the orphan let go")
    check("/interrupt" in paths and ("/queue", {"clear": True}) in Fake.posts and paths.index("/interrupt") < paths.index("/prompt"),
          f"settle: /interrupt and the queue clear came before the prompt ({paths})")
    # ... and gives up with a plain error when the orphan never lets go
    h.JOB_LOCK.acquire()
    h.SETTLE_SECONDS = 0.3
    try:
        check(raises(lambda: h.handler(request()), RuntimeError, "previous, cancelled job"), "settle: an orphan that never ends -> RuntimeError")
    finally:
        h.SETTLE_SECONDS = 120
        h.JOB_LOCK.release()

    # _settle before _proc: an orphan that recycles ComfyUI must not leave the next job a dead process
    old_proc_fn, started = h._proc, []
    h.JOB_LOCK.acquire()

    def orphan_recycles():
        Fake.interrupted.wait(10)
        h._recycle("orphan failed after the interrupt")          # stops and forgets PROC
        h.JOB_LOCK.release()
    Fake.interrupted.clear()
    h.PROC, h.BOOT, h.BOOT_PENDING = proc, 0.0, False
    threading.Thread(target=orphan_recycles, daemon=True).start()

    def fresh_proc():
        if h.PROC is None:
            started.append(1)
            h.PROC, h.BOOT, h.BOOT_PENDING = FakeProc(root, url), 4.0, True
        return h.PROC
    h._proc = fresh_proc
    try:
        r = h.handler(request())
    finally:
        h._proc = old_proc_fn
    check(r["status"] == "success" and started == [1] and r["timing"]["boot"] == 4.0,
          "settle: the orphan recycled ComfyUI, the next job started a new process and carried its boot")

    # _serve: cancelling the task interrupts ComfyUI and the job's thread frees the lock
    async def cancel_run():
        Fake.mode = "hang"
        Fake.posts.clear()
        Fake.interrupted.clear()
        h.PROC, h.BOOT_PENDING = proc, False
        task = asyncio.ensure_future(h._serve(request()))
        for _ in range(100):
            await asyncio.sleep(0.05)
            if any(p == "/prompt" for p, _ in Fake.posts):
                break
        task.cancel()
        try:
            await task
            cancelled = False
        except asyncio.CancelledError:
            cancelled = True
        return cancelled
    try:
        cancelled = asyncio.run(cancel_run())
    finally:
        Fake.mode = "ok"
    check(cancelled, "serve: cancelling the task raises CancelledError")
    check(Fake.interrupted.is_set() and ("/queue", {"clear": True}) in Fake.posts, "serve: cancel sent /interrupt and cleared the queue")
    freed = h.JOB_LOCK.acquire(timeout=10)
    check(freed, "serve: the job's thread ended and released the lock")
    if freed:
        h.JOB_LOCK.release()
    check(asyncio.run(h._serve(request()))["status"] == "success", "serve: an uncancelled job returns its result")

server.shutdown()
print(f"\n{sum(RESULTS)}/{len(RESULTS)} passed")
sys.exit(0 if all(RESULTS) else 1)
