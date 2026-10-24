"""CPU tests for the job runner (no network beyond a fake ComfyUI on 127.0.0.1, no Modal): python tests/test_job_runner.py"""
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
import job_runner as jr  # noqa: E402

RESULTS = []


def check(cond, msg):
    RESULTS.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + msg)


def raises(fn, text):
    try:
        fn()
    except ValueError as e:
        return text in str(e)
    return False


def b64(data):
    return base64.b64encode(data).decode()


# ----------------------------------------------------------------------------- fake ComfyUI
class Fake(http.server.BaseHTTPRequestHandler):
    """POST /prompt "executes" at once: writes files into output/ (and temp/), then GET /history/<id> answers."""
    server_version = "fake"
    mode = "ok"          # ok | node_errors | exec_error | big | hang (files written, history never appears)
    simulate_cache = False      # True: an identical prompt writes no files, as ComfyUI does without --cache-none
    seen = set()
    interrupts = 0
    died = False
    outdir = tempdir = None
    history = {}
    seen_inputs = {}
    cleared = 0

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
                                   "devices": [{"name": "cuda:0 NVIDIA L40S : cudaMallocAsync"}]})
        if self.path == "/queue":
            return self.send(200, {"queue_running": [], "queue_pending": []})
        if self.path.startswith("/history/"):
            pid = self.path.rsplit("/", 1)[1]
            return self.send(200, {pid: Fake.history[pid]} if pid in Fake.history else {})
        self.send(404, {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path == "/history":
            Fake.history.clear()
            Fake.cleared += 1
            return self.send(200, {})
        if self.path == "/interrupt":
            Fake.interrupts += 1
            return self.send(200, {})
        if self.path != "/prompt":
            return self.send(404, {})
        graph = body["prompt"]
        Fake.seen_inputs = {n: open(os.path.join(Fake.indir, n), "rb").read() for n in os.listdir(Fake.indir)}
        if Fake.mode == "node_errors":
            return self.send(400, {"error": {"type": "prompt_outputs_failed_validation", "message": "Prompt outputs failed validation"},
                                   "node_errors": {"7": {"class_type": "SaveImage", "errors": [
                                       {"message": "Required input is missing", "details": "images"}]}}})
        time.sleep(0.05)
        status = {"status_str": "success", "completed": True, "messages": []}
        if Fake.mode == "exec_error":
            status = {"status_str": "error", "completed": False, "messages": [
                ["execution_start", {}], ["execution_error", {"node_type": "Pixelize", "exception_message": "boom"}]]}
            Fake.history["p1"] = {"status": status, "outputs": {}}
            return self.send(200, {"prompt_id": "p1", "number": 1, "node_errors": {}})
        key = json.dumps([graph, sorted((n, d.decode("latin1")) for n, d in Fake.seen_inputs.items())], sort_keys=True)
        cached = Fake.simulate_cache and key in Fake.seen
        Fake.seen.add(key)
        if Fake.mode == "hang":
            with open(os.path.join(Fake.outdir, "late.png"), "wb") as fh:
                fh.write(b"late")
            return self.send(200, {"prompt_id": "never", "number": 1, "node_errors": {}})
        if not cached:
            with open(os.path.join(Fake.outdir, "pixel_00001_.webp"), "wb") as fh:
                fh.write(b"WEBP" * 10)
            with open(os.path.join(Fake.outdir, "pixel_00001_.json"), "wb") as fh:          # unlisted sibling
                fh.write(b'{"k": 1}')
            os.makedirs(os.path.join(Fake.outdir, "sub"), exist_ok=True)
            with open(os.path.join(Fake.outdir, "sub", "other.png"), "wb") as fh:
                fh.write(b"\x89PNG" * 5 if Fake.mode != "big" else b"x" * 2000)
            with open(os.path.join(Fake.tempdir, "preview.png"), "wb") as fh:
                fh.write(b"prev")
        Fake.history["p1"] = {"status": status, "outputs": {
            "7": {"images": [{"filename": "pixel_00001_.webp", "subfolder": "", "type": "output"}]},
            "9": {"text": ["PASS 3 colours"]},
            "8": {"images": [{"filename": "preview.png", "subfolder": "", "type": "temp"}]}}}
        self.send(200, {"prompt_id": "p1", "number": 1, "node_errors": {}})


class FakeProc:
    def __init__(self, root, url):
        self.url = url
        self.input_dir, self.output_dir, self.temp_dir = (os.path.join(root, d) for d in ("input", "output", "temp"))
        for d in (self.input_dir, self.output_dir, self.temp_dir):
            os.makedirs(d, exist_ok=True)

    up = True

    def gpu_name(self):
        return jr.ComfyProcess.gpu_name(self)

    def alive(self):
        return self.up


GRAPH = {"1": {"class_type": "LoadImage", "inputs": {"image": "ss_cat.png"}, "_meta": {"title": "Input image"}},
         "7": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}, "_meta": {"title": "Save pixel art"}},
         "8": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}, "_meta": {"title": "Save preview"}},
         "9": {"class_type": "PixelArtCheck", "inputs": {}, "_meta": {"title": "Check report"}}}


def job(**kw):
    doc = {"protocol": jr.PROTOCOL, "workflow": GRAPH, "inputs": [{"name": "ss_cat.png", "data": b64(b"PNGDATA")}]}
    doc.update(kw)
    return doc


# 1. validate_job
check(jr.validate_job(job()) == 3600, "validate_job: a good document passes, default timeout 3600")
check(raises(lambda: jr.validate_job(job(workflow={})), "workflow"), "validate_job: empty workflow")
check(raises(lambda: jr.validate_job({"inputs": []}), "workflow"), "validate_job: missing workflow")
check(raises(lambda: jr.validate_job(job(workflow={"1": {"inputs": {}}})), "class_type"), "validate_job: node without class_type")
for bad in ("a/b.png", "..\\x.png", "x..png", ".hidden", "", "a\0b", "a" * 256):
    check(raises(lambda bad=bad: jr.validate_job(job(inputs=[{"name": bad, "data": ""}])), "plain file name"),
          f"validate_job: input name {bad[:12]!r} rejected")
big = [{"name": "big.bin", "data": "A" * (jr.MAX_INPUT_BYTES * 4 // 3 + 400)}]
check(raises(lambda: jr.validate_job(job(inputs=big)), "MiB"), "validate_job: inputs over 64 MiB")
check(raises(lambda: jr.validate_job(job(timeout=0)), "timeout") and raises(lambda: jr.validate_job(job(timeout=3601)), "timeout"),
      "validate_job: timeout 0 and 3601")
check(raises(lambda: jr.validate_job(job(protocol="psc-job/v9")), "protocol"), "validate_job: other protocol version")

# 2. snapshot / new_files
with tempfile.TemporaryDirectory() as d:
    for n in ("keep.txt", "rewrite.txt"):
        with open(os.path.join(d, n), "w") as fh:
            fh.write("one")
    before = jr.snapshot([d])
    with open(os.path.join(d, "rewrite.txt"), "w") as fh:
        fh.write("changed!")
    os.makedirs(os.path.join(d, "s"))
    with open(os.path.join(d, "s", "new.txt"), "w") as fh:
        fh.write("n")
    got = [os.path.relpath(p, d).replace("\\", "/") for p in jr.new_files(before, jr.snapshot([d]))]
    check(got == ["rewrite.txt", "s/new.txt"], f"new_files: rewritten and new reported, untouched not ({got})")

# 5. encode_result / decode_job round trip
payload = bytes(range(256)) * 3
enc = jr.encode_result({"files": [{"filename": "a.bin", "data": payload}], "status": "success"})
check(json.loads(json.dumps(enc))["files"][0]["data"] == b64(payload), "encode_result: base64, JSON-safe")
check(jr.decode_result(enc)["files"][0]["data"] == payload, "encode_result / decode_result round-trip bytes exactly")
check(jr.decode_job({"inputs": [{"name": "x", "data": b64(payload)}]}) == [("x", payload)], "decode_job: base64 -> bytes exactly")
check(raises(lambda: jr.decode_job({"inputs": [{"name": "x", "data": "@@@"}]}), "base64"), "decode_job: bad base64 -> ValueError")

# gpu_rate against the real rates file
rates = json.load(open(os.path.join(ROOT, "cloud", "gpu_rates.json")))
check(jr.gpu_rate("NVIDIA L40S", rates) == ("L40S", 1.95), "gpu_rate: L40S")
check(jr.gpu_rate("NVIDIA RTX PRO 6000 Blackwell Server Edition", rates) == ("RTX-PRO-6000", 3.03), "gpu_rate: RTX PRO 6000")
check(jr.gpu_rate("Mystery GPU", rates) == (None, None), "gpu_rate: unknown GPU -> (None, None)")

# 3 + 4. run_job against the fake
server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
threading.Thread(target=server.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{server.server_address[1]}"
with tempfile.TemporaryDirectory() as root:
    proc = FakeProc(root, url)
    Fake.indir, Fake.outdir, Fake.tempdir = proc.input_dir, proc.output_dir, proc.temp_dir
    with open(os.path.join(proc.output_dir, "old.png"), "wb") as fh:       # a leftover that must not be reported
        fh.write(b"old")
    check(jr.ComfyProcess.gpu_name(proc) == "NVIDIA L40S", "gpu_name: parsed from /system_stats")

    Fake.mode = "ok"
    r = jr.run_job(job(), proc, boot=10.0, rate=1.95, worker="w1")
    print(json.dumps({k: v for k, v in jr.encode_result(r).items() if k != "files"}, indent=1)[:600])
    check(Fake.seen_inputs == {"ss_cat.png": b"PNGDATA"}, "run_job: input landed in input/ under its name")
    check(r["status"] == "success" and not r["errors"], "run_job: success")
    by = {f["filename"]: f for f in r["files"]}
    check(by["pixel_00001_.webp"]["node"] == "Save pixel art" and by["pixel_00001_.webp"]["data"] == b"WEBP" * 10,
          "run_job: listed file with its node title and bytes")
    check(by["pixel_00001_.json"]["node"] == "" and by["pixel_00001_.json"]["data"] == b'{"k": 1}',
          "run_job: unlisted sibling (x.json next to x.webp) returned with node ''")
    check(by["other.png"]["subfolder"] == "sub" and by["preview.png"]["type"] == "temp" and by["preview.png"]["node"] == "Save preview",
          "run_job: subfolder file and temp preview")
    check("old.png" not in by, "run_job: untouched file not reported")
    check(r["text"] == {"Check report": "PASS 3 colours"}, "run_job: text collected by node title")
    check(os.listdir(proc.input_dir) == [] and sorted(os.listdir(proc.output_dir)) == ["old.png"]
          and os.listdir(proc.temp_dir) == [], "run_job: inputs and outputs deleted afterwards (leftover kept, empty subfolder removed)")
    check(Fake.cleared >= 1, "run_job: history cleared")
    t = r["timing"]
    check(t["run"] > 0 and t["boot"] == 10.0 and abs(t["billed"] - (t["boot"] + t["run"])) < 1e-6, "run_job: timing.run > 0, billed == boot + run")
    check(r["cost"]["rate_per_hour"] == 1.95 and r["cost"]["gpu"] == "NVIDIA L40S"
          and abs(r["cost"]["usd"] - t["billed"] * 1.95 / 3600) < 1e-5, "run_job: cost = billed * rate / 3600")
    check(r["worker"] == "w1" and r["protocol"] == jr.PROTOCOL, "run_job: worker and protocol")
    r2 = jr.run_job(job(), proc, rate=rates)
    check(r2["cost"]["rate_per_hour"] == 1.95 and r2["timing"]["boot"] == 0, "run_job: rates document matched on the GPU name; boot 0")
    r3 = jr.run_job(job(), proc)
    check(r3["cost"]["usd"] is None, "run_job: no rate -> usd None")

    Fake.mode = "node_errors"
    r = jr.run_job(job(), proc)
    check(r["status"] == "error" and any(e.startswith("SaveImage: Required input is missing") for e in r["errors"])
          and os.listdir(proc.input_dir) == [], f"error path: /prompt node_errors -> status error ({r['errors']})")
    Fake.mode = "exec_error"
    r = jr.run_job(job(), proc)
    check(r["status"] == "error" and r["errors"] == ["Pixelize: boom"], f"error path: execution_error -> 'Pixelize: boom' ({r['errors']})")
    Fake.mode = "big"
    old_cap = jr.MAX_OUTPUT_BYTES
    jr.MAX_OUTPUT_BYTES = 1000
    try:
        r = jr.run_job(job(), proc)
    finally:
        jr.MAX_OUTPUT_BYTES = old_cap
    check(r["status"] == "error" and r["files"] == [] and "larger than" in r["errors"][0]
          and os.listdir(proc.temp_dir) == [], "error path: outputs over the cap -> status error, no files, cleaned up")
    try:
        jr.run_job(job(inputs=[{"name": "../x", "data": ""}]), proc)
        check(False, "run_job validates the document")
    except ValueError:
        check(True, "run_job validates the document")

    # A1: an identical job on a warm container. Without --cache-none ComfyUI re-runs nothing: history lists files
    # that were not written. That must never come back as a quiet success without files.
    Fake.mode, Fake.simulate_cache, Fake.seen = "ok", True, set()
    first = jr.run_job(job(), proc)
    second = jr.run_job(job(), proc)
    check(first["status"] == "success" and len(first["files"]) == 4, "A1: first job on the container returns its files")
    check(second["status"] == "error" and any("did not write" in e and "pixel_00001_.webp" in e for e in second["errors"]),
          f"A1: identical second job whose outputs came from ComfyUI's cache -> status error naming the file ({second['errors']})")
    check(os.listdir(proc.input_dir) == [] and os.listdir(proc.temp_dir) == [], "A1: the cached job still cleans up")
    Fake.simulate_cache = False
    again = [jr.run_job(job(), proc) for _ in range(2)]
    check(all(r["status"] == "success" and len(r["files"]) == 4 for r in again),
          "A1: two identical jobs that both execute (what --cache-none gives) both return their files")
    popen_calls = []
    real_popen = jr.subprocess.Popen
    jr.subprocess.Popen = lambda cmd, **kw: popen_calls.append(cmd) or type("P", (), {"poll": lambda self: None})()
    try:
        cp = jr.ComfyProcess(root, 8199)
        check(not cp.alive(), "A2: alive() is False before start()")
        cp.start()
        jr.ComfyProcess(root, 8198).start(["--cpu"])
    finally:
        jr.subprocess.Popen = real_popen
    check("--cache-none" in popen_calls[0] and "--cache-none" not in popen_calls[1],
          "A1: ComfyProcess.start() adds --cache-none by default; explicit args replace it (refresh_info --cpu)")
    check(cp.alive(), "A2: alive() is True while the process runs")
    cp.proc = type("P", (), {"poll": lambda self: 137})()
    check(not cp.alive(), "A2: alive() is False once ComfyUI exited")

    # A2: a crash or a timeout stops the container taking jobs; a timeout leaves no outputs behind; A4: clamp.
    recycled = []
    proc.up = True
    ok = jr.run_job_guarded(job(), proc, recycled.append)
    check(ok["status"] == "success" and recycled == [], "A2: a good job recycles nothing")
    Fake.mode = "node_errors"
    r = jr.run_job_guarded(job(), proc, recycled.append)
    check(r["status"] == "error" and recycled == [], "A2: a rejected graph is a result, not a reason to recycle")
    try:
        jr.run_job_guarded(job(inputs=[{"name": "../x", "data": ""}]), proc, recycled.append)
        check(False, "A2: a bad document raises ValueError")
    except ValueError:
        check(recycled == [], "A2: a bad document (ValueError) recycles nothing")
    Fake.mode, Fake.interrupts = "hang", 0
    t0 = time.time()
    try:
        jr.run_job_guarded(job(timeout=3600), proc, recycled.append, max_timeout=1)
        check(False, "A4: the deployment ceiling times the job out")
    except RuntimeError as e:
        took = time.time() - t0
        check(took < 15 and "TimeoutError" in str(e) and Fake.interrupts == 1,
              f"A4: timeout 3600 clamped to max_timeout=1 -> timed out and interrupted after {took:.1f}s ({e})")
    check(len(recycled) == 1 and "TimeoutError" in recycled[0], f"A2: a timeout recycles the container ({recycled})")
    check(os.listdir(proc.output_dir) == ["old.png"] and os.listdir(proc.input_dir) == [],
          "A2: the timed-out job's outputs and inputs are deleted (no leak into the next job)")
    Fake.mode = "ok"
    nxt = jr.run_job_guarded(job(), proc, [].append)
    check(nxt["status"] == "success" and not any(f["filename"] == "late.png" for f in nxt["files"]),
          "A2: the next job does not return the timed-out job's file")
    recycled.clear()
    proc.up = False
    try:
        jr.run_job_guarded(job(), proc, recycled.append)
        check(False, "A2: dead ComfyUI refuses the job")
    except RuntimeError as e:
        check(len(recycled) == 1 and "not running" in recycled[0] and "no more jobs" in str(e),
              "A2: ComfyUI already dead at the start -> recycle and fail the job")
    recycled.clear()

    class Dying(FakeProc):
        def alive(self):
            return not Fake.died
    dying = Dying(root, url)
    real_run = jr.run_job

    def run_then_die(*a, **kw):
        out = real_run(*a, **kw)
        Fake.died = True          # ComfyUI is killed right after the job (OOM), before the result is sent
        return out
    jr.run_job = run_then_die
    try:
        res = jr.run_job_guarded(job(), dying, recycled.append)
    finally:
        jr.run_job = real_run
    check(res["status"] == "success" and len(recycled) == 1 and "exited during the job" in recycled[0],
          "A2: ComfyUI gone after a finished job -> result returned, container recycled")
server.shutdown()

print(f"\n{sum(RESULTS)}/{len(RESULTS)} passed")
sys.exit(0 if all(RESULTS) else 1)
