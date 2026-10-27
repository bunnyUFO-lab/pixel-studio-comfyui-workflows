"""Smoke client for the job API (docs/job-api.md), stdlib only.

    python cloud/job_smoke.py --url https://<workspace>--<app>-jobs.modal.run [--n 3] [--out DIR]

Headers come from the environment: MODAL_KEY / MODAL_SECRET (a Modal proxy auth token), sent as Modal-Key /
Modal-Secret. It reads /v1/info, converts workflows/pixel-art/image-to-pixel.json to an API graph with
.claude/skills/comfyui/scripts/comfy.py (to_api), submits N jobs at once with a generated 64x64 PNG, polls them,
writes every returned file under --out, and prints one line per job plus how much the run windows overlapped.
`make_job` and `summary_line` are also used by `modal run cloud/modal_comfy.py::smoke`.
"""
import argparse
import base64
import importlib.util
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request
import uuid
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
WORKFLOW = os.path.join(REPO, "workflows", "pixel-art", "image-to-pixel.json")
COMFY_PY = os.path.join(REPO, ".claude", "skills", "comfyui", "scripts", "comfy.py")


def png(size=64):
    """A size x size RGBA PNG (a red disc on a transparent background), stdlib only."""
    rows = bytearray()
    c = (size - 1) / 2
    for y in range(size):
        rows.append(0)
        for x in range(size):
            inside = (x - c) ** 2 + (y - c) ** 2 <= (size * 0.35) ** 2
            rows += bytes((200, 40, 40, 255)) if inside else bytes((0, 0, 0, 0))

    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(rows))) + chunk(b"IEND", b""))


def to_api(workflow, object_info):
    spec = importlib.util.spec_from_file_location("comfy_cli", COMFY_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.to_api(workflow, object_info)


def make_job(object_info, timeout=1800):
    """A job document for image-to-pixel with a fresh input name (so parallel jobs never share a file name)."""
    with open(WORKFLOW, encoding="utf-8") as fh:
        api = to_api(json.load(fh), object_info)
    name = f"ss_{uuid.uuid4().hex[:10]}_cat.png"
    node = next(n for n in api.values() if n["class_type"] == "LoadImage")
    node["inputs"]["image"] = name
    return {"protocol": "psc-job/v1", "workflow": api, "timeout": timeout,
            "inputs": [{"name": name, "data": base64.b64encode(png()).decode()}]}


def summary_line(job_id, result):
    t, c = result.get("timing", {}), result.get("cost", {})
    usd = c.get("usd")
    return (f"{job_id}  {result.get('status')}  worker={result.get('worker')}  gpu={c.get('gpu')}  boot={t.get('boot')}s  "
            f"run={t.get('run')}s  billed={t.get('billed')}s  usd={'?' if usd is None else format(usd, '.4f')}  "
            f"files={len(result.get('files', []))}")


class Client:
    def __init__(self, url):
        self.url = url.rstrip("/")
        self.headers = {"Modal-Key": os.environ.get("MODAL_KEY", ""), "Modal-Secret": os.environ.get("MODAL_SECRET", "")}

    def call(self, method, path, body=None, timeout=300):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data, method=method, headers=dict(self.headers))
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read() or b"null")
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                return e.code, json.loads(raw)
            except ValueError:
                return e.code, {"error": raw.decode("utf-8", "replace")[:300]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", required=True, help="the jobs URL (the deploy prints it)")
    ap.add_argument("--n", type=int, default=1, help="jobs to submit at once")
    ap.add_argument("--out", default="job-smoke-out", help="where to write the returned files")
    ap.add_argument("--timeout", type=int, default=1800, help="seconds to wait for all jobs")
    a = ap.parse_args()
    if not (os.environ.get("MODAL_KEY") and os.environ.get("MODAL_SECRET")):
        sys.exit("set MODAL_KEY and MODAL_SECRET (a Modal proxy auth token) in the environment")
    cl = Client(a.url)
    code, health = cl.call("GET", "/v1/health")
    print(f"health {code}: {json.dumps(health)}")
    t0 = time.time()
    code, info = cl.call("GET", "/v1/info", timeout=900)
    if code != 200:
        sys.exit(f"/v1/info -> {code}: {info}")
    print(f"info: {len(info['object_info'])} node classes, {len(info['models'])} model folders, "
          f"ComfyUI {info.get('comfyui_version')} ({time.time() - t0:.1f}s)")
    ids = []
    for _ in range(a.n):
        code, res = cl.call("POST", "/v1/jobs", make_job(info["object_info"]), timeout=300)
        if code != 202:
            sys.exit(f"POST /v1/jobs -> {code}: {res}")
        ids.append(res["id"])
    print(f"submitted {len(ids)} job(s) at {time.strftime('%H:%M:%S')}")
    pending, done, last = set(ids), {}, None
    deadline = time.time() + a.timeout
    while pending and time.time() < deadline:
        for jid in sorted(pending):
            code, st = cl.call("GET", f"/v1/jobs/{jid}")
            if code == 404:
                done[jid] = {"status": "lost", "error": st}
                pending.discard(jid)
            elif st.get("status") in ("done", "failed", "cancelled"):
                done[jid] = st
                pending.discard(jid)
            elif st.get("progress") and st["progress"] != last:
                last = st["progress"]
                print(f"  {jid[-8:]} {st['status']}: node {last.get('node')} {last.get('value')}/{last.get('max')}")
        time.sleep(2)
    windows, ok = [], True
    for jid in ids:
        st = done.get(jid)
        if st is None:
            print(f"{jid}  TIMEOUT after {a.timeout}s")
            ok = False
        elif st["status"] != "done":
            print(f"{jid}  {st['status']}: {st.get('error')}")
            ok = False
        else:
            r = st["result"]
            print(summary_line(jid[-8:], r))
            for e in r.get("errors", []):
                print("   ERROR", e)
            if r["status"] != "success":
                ok = False
            d = os.path.join(a.out, jid[-8:])
            os.makedirs(d, exist_ok=True)
            for f in r["files"]:
                with open(os.path.join(d, f["filename"]), "wb") as fh:
                    fh.write(base64.b64decode(f["data"]))
            windows.append((r["timing"]["started_at"], r["timing"]["finished_at"]))
    if len(windows) > 1:
        overlap = min(w[1] for w in windows) - max(w[0] for w in windows)
        print(f"run windows overlap: {max(overlap, 0):.1f}s (all {len(windows)} jobs ran at once: {overlap > 0})")
    print(f"total {time.time() - t0:.1f}s; files under {a.out}")
    sys.exit(0 if ok and len(done) == len(ids) else 1)


if __name__ == "__main__":
    main()
