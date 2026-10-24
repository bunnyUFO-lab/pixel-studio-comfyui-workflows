"""RunPod Serverless worker for the job protocol psc-job/v1 (docs/job-api.md, "RunPod transport").

RunPod carries the job document as `input` of `POST /run` and the result document as `output` of `GET /status`.
job_runner.py (copied next to this file in the image) does the work; this file only starts ComfyUI once per worker,
moves documents in and out, and keeps one job at a time on the process.

    handler({"input": <psc-job/v1 document>})        -> the result document
    handler({"input": {"action": "info"}})           -> ComfyProcess.info()
    handler({"input": {"action": "download", "workflows": "pixel"}})      -> models.py download (fills the volume)
    handler({"input": {"action": "list_models", "workflows": "pixel"}})   -> models.py list

Infrastructure problems raise (RunPod marks the job FAILED with the message); a rejected graph or a failed node is a
result with status "error", as on Modal. The `runpod` and `boto3` packages are imported only where they are used, so
the tests import this module without them. Local test: python handler.py --rp_serve_api (flags parsed by the SDK).
"""
import collections
import json
import mimetypes
import os
import subprocess
import sys
import threading
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.dirname(HERE)):          # the image has both files in /opt/psc; the repo has job_runner in cloud/
    if _p not in sys.path:
        sys.path.insert(0, _p)
import job_runner  # noqa: E402

COMFY_DIR = os.environ.get("PSC_COMFY_DIR", "/opt/ComfyUI")
VOLUME = os.environ.get("PSC_VOLUME_DIR", "/runpod-volume")
PIXEL_ENV = os.environ.get("PSC_PIXEL_ENV", "/opt/env/pixel.env")
STARTER = ",".join([
    "pixel-art/image-to-pixel", "pixel-art/prompt-to-pixel", "pixel-art/sprite-edit", "pixel-art/sprite-edit-ref",
    "pixel-art/sprite-edit-refs", "image-edit/qwen-edit",
])
DEFAULT_INLINE_MAX = 9 * 1024 * 1024         # RunPod's /run limit is 10 MB; the same budget is assumed for /status output
SETTLE_SECONDS = 120

JOB_LOCK = threading.Lock()                   # held for the whole life of a job, also by a thread RunPod gave up on
_PROC_LOCK = threading.Lock()
PROC = None
BOOT = 0.0
BOOT_PENDING = False
try:
    with open(os.path.join(HERE, "gpu_rates.json"), encoding="utf-8") as _fh:
        RATES = json.load(_fh)
except OSError:
    RATES = {}


# ----------------------------------------------------------------------------- the ComfyUI process
def _models_dir():
    return os.path.join(VOLUME, "models") if os.path.isdir(VOLUME) else os.path.join(COMFY_DIR, "models")


def _link_models():
    """With a network volume attached, ComfyUI's models folder becomes a link to it (models.py follows MODELS_DIR)."""
    if not os.path.isdir(VOLUME):
        return
    target = os.path.join(VOLUME, "models")
    os.makedirs(target, exist_ok=True)
    link = os.path.join(COMFY_DIR, "models")
    if os.path.islink(link) and os.path.realpath(link) == os.path.realpath(target):
        return
    if os.path.islink(link):
        os.remove(link)
    elif os.path.isdir(link):
        import shutil
        shutil.rmtree(link)
    os.symlink(target, link)


def _proc():
    """The worker's ComfyUI, started on first use. Tests set PROC / BOOT / BOOT_PENDING instead."""
    global PROC, BOOT, BOOT_PENDING
    with _PROC_LOCK:
        if PROC is None:
            _link_models()
            os.environ["MODELS_DIR"] = _models_dir()
            started = time.time()
            proc = job_runner.ComfyProcess(COMFY_DIR, 8188).start()
            try:
                proc.wait_ready(600)
            except BaseException:
                proc.stop()
                raise
            BOOT, BOOT_PENDING, PROC = round(time.time() - started, 3), True, proc
            print(f"ComfyUI ready after {BOOT}s", flush=True)
        return PROC


def _settle(proc):
    """Take JOB_LOCK. If it is held, a cancelled or timed-out job's thread is still running (the SDK moved on): stop
    its prompt, then wait for the thread to let go. The caller releases the lock."""
    if JOB_LOCK.acquire(blocking=False):
        return
    for path, body in (("/interrupt", {}), ("/queue", {"clear": True})):
        try:
            job_runner._request(proc.url + path, body, timeout=10)
        except (OSError, ValueError):
            pass
    if not JOB_LOCK.acquire(timeout=SETTLE_SECONDS):
        raise RuntimeError("the previous, cancelled job is still running on this worker")


# ----------------------------------------------------------------------------- progress and uploads
def _safe(callback):
    def call(p):
        try:
            callback(p)
        except Exception:        # progress is best effort; it never fails the job
            pass
    return call


def _sdk_progress(job, every=2.0):
    """progress_update(job, p), at most once every `every` seconds."""
    last = [0.0]

    def send(p):
        if time.time() - last[0] < every:
            return
        last[0] = time.time()
        from runpod.serverless import progress_update
        progress_update(job, p)
    return send


def s3_uploader(environ=None):
    """upload(key, bytes) -> (presigned GET url, expires epoch) to the S3-compatible bucket in PSC_S3_*, or None when
    PSC_S3_BUCKET is not set. Credentials come from the usual AWS_* variables (RunPod secrets on the endpoint)."""
    env = os.environ if environ is None else environ
    bucket = env.get("PSC_S3_BUCKET")
    if not bucket:
        return None
    import boto3
    client = boto3.client("s3", endpoint_url=env.get("PSC_S3_ENDPOINT") or None, region_name=env.get("AWS_REGION") or None)
    seconds = int(env.get("PSC_S3_URL_SECONDS") or 3600)

    def upload(key, data):
        client.put_object(Bucket=bucket, Key=key, Body=data,
                          ContentType=mimetypes.guess_type(key)[0] or "application/octet-stream")
        url = client.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=seconds)
        return url, int(time.time()) + seconds
    return upload


# ----------------------------------------------------------------------------- actions
def _inline_max():
    return int(os.environ.get("PSC_INLINE_MAX_BYTES") or DEFAULT_INLINE_MAX)


def _size(n):
    return f"{n} bytes ({n / 2**20:.1f} MiB)"


def _run_job(doc, job_id, progress, uploader):
    global BOOT_PENDING
    budget = _inline_max()
    n = len(json.dumps(doc))
    if n > budget:
        raise ValueError(f"the job is {_size(n)}, over the inline limit of {_size(budget)}: RunPod takes at most 10 MB "
                         "per request; use the Modal source for inputs this large")
    proc = _proc()
    _settle(proc)
    try:
        try:
            result = job_runner.run_job(doc, proc, on_progress=progress, boot=BOOT if BOOT_PENDING else 0.0, rate=RATES,
                                        worker=os.environ.get("RUNPOD_POD_ID", ""))
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"{type(e).__name__}: {e}") from e
        BOOT_PENDING = False
    finally:
        JOB_LOCK.release()
    encoded = job_runner.encode_result(result)
    total = len(json.dumps(encoded))
    if total <= budget:
        return encoded
    if uploader is None:
        result["status"] = "error"
        result["files"] = []
        result["errors"].append(f"outputs are {total / 2**20:.1f} MB, over RunPod's inline limit of "
                                f"{budget / 2**20:.1f} MB; configure PSC_S3_* on the endpoint (cloud/runpod/README.md) "
                                "or use the Modal source")
        return job_runner.encode_result(result)
    prefix = os.environ.get("PSC_S3_PREFIX", "psc-jobs/")
    files = []
    for f in result["files"]:
        key = prefix + "/".join(p for p in (job_id, f["type"], f["subfolder"], f["filename"]) if p)
        url, expires = uploader(key, f["data"])
        files.append({**{k: v for k, v in f.items() if k != "data"}, "url": url, "expires": expires})
    return job_runner.encode_result(dict(result, files=files))


def _workflows(value):
    if value in (None, "", "pixel"):
        with open(PIXEL_ENV, encoding="utf-8") as fh:
            line = next(l for l in fh.read().splitlines() if l.startswith("MODELS="))
        value = line.split("=", 1)[1]
    elif value == "starter":
        value = STARTER
    return [w.strip() for w in str(value).split(",") if w.strip()]


def _models_py(verb, workflows):
    """Run docker/models.py; returns (exit code, last 40 lines). Output also goes to the worker log."""
    cmd = [sys.executable, "/usr/local/bin/models.py", verb, *_workflows(workflows)]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                         env=dict(os.environ, MODELS_DIR=_models_dir()))
    tail = collections.deque(maxlen=40)
    for line in p.stdout:
        print(line, end="", flush=True)
        tail.append(line.rstrip("\n"))
    return p.wait(), "\n".join(tail)


UNKNOWN = "unknown input: expected a psc-job/v1 document or an action (info, download, list_models)"


def handler(job, progress=None, uploader=None):
    inp = (job or {}).get("input")
    if not isinstance(inp, dict):
        raise ValueError(UNKNOWN)
    if "protocol" in inp:
        if progress is None:
            progress = _sdk_progress(job)
        return _run_job(inp, str(job.get("id") or uuid.uuid4().hex), _safe(progress),
                        uploader if uploader is not None else s3_uploader())
    action = inp.get("action")
    if action == "info":
        return _proc().info()
    if action == "download":
        rc, tail = _models_py("download", inp.get("workflows"))
        if rc:
            raise RuntimeError(f"models.py download exited with {rc} (run it again to resume):\n{tail}")
        return {"rc": rc, "tail": tail}
    if action == "list_models":
        rc, text = _models_py("list", inp.get("workflows"))
        return {"rc": rc, "text": text}
    raise ValueError(UNKNOWN)


def _serve(job):
    # Results are documents with `errors`, never a top-level "error" key: the SDK reads that key as a failure.
    return handler(job)


if __name__ == "__main__":
    import runpod
    _proc()                  # report ready only with ComfyUI up
    runpod.serverless.start({"handler": _serve})
