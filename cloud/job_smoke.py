"""Smoke client for the job API (docs/job-api.md), stdlib only.

    python cloud/job_smoke.py --url https://<workspace>--<app>-jobs.modal.run [--n 3] [--out DIR]
    python cloud/job_smoke.py --runpod <endpoint id> [--n 3]       # RunPod: /run + /status, needs RUNPOD_API_KEY
    python cloud/job_smoke.py --local http://127.0.0.1:8010        # the RunPod handler's own server (/runsync)

Headers come from the environment: MODAL_KEY / MODAL_SECRET (a Modal proxy auth token), sent as Modal-Key /
Modal-Secret; RUNPOD_API_KEY, sent as a Bearer token. It reads the node and model lists (/v1/info, or an `info` job
on RunPod), converts workflows/pixel-art/image-to-pixel.json to an API graph with
.claude/skills/comfyui/scripts/comfy.py (to_api), submits N jobs at once with a generated 64x64 PNG, polls them,
writes every returned file under --out (fetching `url` files with a plain GET), and prints one line per job plus how much the run windows overlapped.
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


def http_json(method, url, headers, body=None, timeout=300):
    """(status code, parsed JSON) of one request; an error status is returned, not raised."""
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers=dict(headers))
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


def file_bytes(f):
    """The bytes of a result file: inline `data`, or a plain GET of its presigned `url` (protocol extension)."""
    if f.get("data") is not None:
        return base64.b64decode(f["data"])
    with urllib.request.urlopen(f["url"], timeout=300) as r:
        return r.read()


class Client:
    """The Modal transport: POST /v1/jobs, poll GET /v1/jobs/{id}."""

    def __init__(self, url):
        self.url = url.rstrip("/")
        self.headers = {"Modal-Key": os.environ.get("MODAL_KEY", ""), "Modal-Secret": os.environ.get("MODAL_SECRET", "")}

    def call(self, method, path, body=None, timeout=300):
        return http_json(method, self.url + path, self.headers, body, timeout)

    def health(self):
        return self.call("GET", "/v1/health")

    def info(self):
        code, info = self.call("GET", "/v1/info", timeout=900)
        if code != 200:
            sys.exit(f"/v1/info -> {code}: {info}")
        return info

    def submit(self, doc):
        code, res = self.call("POST", "/v1/jobs", doc, timeout=300)
        if code != 202:
            sys.exit(f"POST /v1/jobs -> {code}: {res}")
        return res["id"]

    def poll(self, jid):
        """(state, payload): state queued | running | done | failed | cancelled | lost; payload = progress, result or error."""
        code, st = self.call("GET", f"/v1/jobs/{jid}")
        if code == 404:
            return "lost", st
        if st.get("status") == "done":
            return "done", st["result"]
        if st.get("status") in ("failed", "cancelled"):
            return st["status"], st.get("error")
        return st.get("status"), st.get("progress")


class RunpodClient:
    """The RunPod transport (docs/job-api.md): POST /run {"input": doc}, poll GET /status/{id}. With local=True it
    talks to the SDK's own server (`python handler.py --rp_serve_api`), whose /run never runs anything: it uses
    /runsync and the job is finished when submit returns."""
    STATES = {"IN_QUEUE": "queued", "IN_PROGRESS": "running", "COMPLETED": "done", "FAILED": "failed",
              "TIMED_OUT": "failed", "CANCELLED": "cancelled"}

    def __init__(self, target, local=False):
        target = target.rstrip("/")
        self.local = local
        self.url = target if local or target.startswith("http") else "https://api.runpod.ai/v2/" + target
        self.headers = {} if local else {"Authorization": "Bearer " + os.environ.get("RUNPOD_API_KEY", "")}
        self.finished = {}

    def call(self, method, path, body=None, timeout=300):
        return http_json(method, self.url + path, self.headers, body, timeout)

    def health(self):
        return self.call("GET", "/health") if not self.local else (200, {"local": self.url})

    def submit(self, doc):
        if self.local:
            code, res = self.call("POST", "/runsync", {"input": doc}, timeout=3600)
            jid = (res or {}).get("id") or uuid.uuid4().hex
            self.finished[jid] = (code, res)
            return jid
        code, res = self.call("POST", "/run", {"input": doc}, timeout=300)
        if code != 200 or "id" not in (res or {}):
            sys.exit(f"POST /run -> {code}: {res}")
        return res["id"]

    def poll(self, jid):
        code, st = self.finished.pop(jid) if jid in self.finished else self.call("GET", f"/status/{jid}")
        if code == 404:
            return "lost", st
        state = self.STATES.get(st.get("status"))
        if state is None:
            return "failed", st.get("error") or st
        if state == "done":
            return "done", st.get("output")
        if state in ("failed", "cancelled"):
            return state, st.get("error") or st.get("output")
        return state, st.get("output")

    def info(self):
        jid = self.submit({"action": "info"})
        deadline = time.time() + 1800
        while time.time() < deadline:
            state, payload = self.poll(jid)
            if state == "done":
                return payload
            if state in ("failed", "cancelled", "lost"):
                sys.exit(f"info job {state}: {payload}")
            time.sleep(3)
        sys.exit("info job did not finish in 30 minutes (a cold start pulls the image, then ComfyUI boots)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--url", help="Modal: the jobs URL (the deploy prints it)")
    where.add_argument("--runpod", metavar="ENDPOINT", help="RunPod: endpoint id or https://api.runpod.ai/v2/<id> (needs RUNPOD_API_KEY)")
    where.add_argument("--local", metavar="URL", help="the RunPod handler's own server (handler.py --rp_serve_api), e.g. http://127.0.0.1:8010")
    ap.add_argument("--n", type=int, default=1, help="jobs to submit at once")
    ap.add_argument("--out", default="job-smoke-out", help="where to write the returned files")
    ap.add_argument("--timeout", type=int, default=1800, help="seconds to wait for all jobs")
    a = ap.parse_args()
    if a.url:
        if not (os.environ.get("MODAL_KEY") and os.environ.get("MODAL_SECRET")):
            sys.exit("set MODAL_KEY and MODAL_SECRET (a Modal proxy auth token) in the environment")
        cl = Client(a.url)
    else:
        if a.runpod and not os.environ.get("RUNPOD_API_KEY"):
            sys.exit("set RUNPOD_API_KEY (console > Settings > API Keys) in the environment")
        cl = RunpodClient(a.runpod or a.local, local=bool(a.local))
    code, health = cl.health()
    print(f"health {code}: {json.dumps(health)}")
    t0 = time.time()
    info = cl.info()
    print(f"info: {len(info['object_info'])} node classes, {len(info['models'])} model folders, "
          f"ComfyUI {info.get('comfyui_version')} ({time.time() - t0:.1f}s)")
    ids = [cl.submit(make_job(info["object_info"])) for _ in range(a.n)]
    print(f"submitted {len(ids)} job(s) at {time.strftime('%H:%M:%S')}")
    pending, done, last = set(ids), {}, None
    deadline = time.time() + a.timeout
    while pending and time.time() < deadline:
        for jid in sorted(pending):
            state, payload = cl.poll(jid)
            if state in ("done", "failed", "cancelled", "lost"):
                done[jid] = (state, payload)
                pending.discard(jid)
            elif state == "running" and isinstance(payload, dict) and payload != last and "node" in payload:
                last = payload
                print(f"  {jid[-8:]} {state}: node {last.get('node')} {last.get('value')}/{last.get('max')}")
        if pending:
            time.sleep(2)
    windows, ok = [], True
    for jid in ids:
        state, payload = done.get(jid, (None, None))
        if state is None:
            print(f"{jid}  TIMEOUT after {a.timeout}s")
            ok = False
        elif state != "done":
            print(f"{jid}  {state}: {payload}")
            ok = False
        else:
            r = payload
            print(summary_line(jid[-8:], r))
            for e in r.get("errors", []):
                print("   ERROR", e)
            if r["status"] != "success":
                ok = False
            d = os.path.join(a.out, jid[-8:])
            os.makedirs(d, exist_ok=True)
            for f in r["files"]:
                with open(os.path.join(d, f["filename"]), "wb") as fh:
                    fh.write(file_bytes(f))
            windows.append((r["timing"]["started_at"], r["timing"]["finished_at"]))
    if len(windows) > 1:
        overlap = min(w[1] for w in windows) - max(w[0] for w in windows)
        print(f"run windows overlap: {max(overlap, 0):.1f}s (all {len(windows)} jobs ran at once: {overlap > 0})")
    print(f"total {time.time() - t0:.1f}s; files under {a.out}")
    sys.exit(0 if ok and len(done) == len(ids) else 1)


if __name__ == "__main__":
    main()
