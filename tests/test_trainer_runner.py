"""CPU tests for trainer/runner.py: token, upload, inline config, time and idle limits.
No GPU, no network beyond 127.0.0.1. Runners start as subprocesses on 127.0.0.1 (never 0.0.0.0) with a fake trainer.

    python tests/test_trainer_runner.py        (also collected by pytest as test_trainer_runner)
"""
import hashlib
import http.client
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER = os.path.join(ROOT, "trainer", "runner.py")
sys.path.insert(0, os.path.join(ROOT, "trainer"))
import runner  # noqa: E402

RESULTS = []
TOKEN = "T" * 8 + "k3y-" + "abcdefghijklmnopqrstuvwxyz0123456789"      # 48 characters, made up for the tests
WRONG = "W" * 8 + "rong-" + "zyxwvutsrqponmlkjihgfedcba9876543210"

FAKE = r'''
import os, sys, time
cfg = sys.argv[-1]
name = os.path.splitext(os.path.basename(cfg))[0]
steps, step_s = int(os.environ.get("FAKE_STEPS", "5")), float(os.environ.get("FAKE_STEP_S", "0.1"))
hb = os.environ.get("FAKE_HEARTBEAT")
bad = [k for k in ("RUNNER_TOKEN", "HF_WRITE_TOKEN", "RUNPOD_API_KEY") if k in os.environ]
print("token-in-env: " + ("yes" if bad else "no"), flush=True)
for i in range(1, steps + 1):
    if hb:
        with open(hb, "w") as f:
            f.write(str(time.time()))
    print("%s: %3d%%|#####     | %d/%d [00:0%d<00:01, 1.00s/it, lr: 1.0e-04 loss: 4.1e-01]" % (name, 100 * i // steps, i, steps, i % 10), flush=True)
    time.sleep(step_s)
'''


def check(cond, msg):
    RESULTS.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + msg)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Box:
    """A temp folder with the runner's folders and the fake trainer."""

    def __init__(self):
        self.dir = Path(tempfile.mkdtemp(prefix="trainer-test-"))
        self.datasets, self.output = self.dir / "datasets", self.dir / "output"
        self.configs, self.state = self.dir / "configs", self.dir / "state"
        self.fake = self.dir / "fake_trainer.py"
        self.fake.write_text(FAKE, encoding="utf-8")
        self.heartbeat = self.dir / "heartbeat.txt"
        self.procs = []

    def env(self, extra=None):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("RUNNER_", "RUN_", "FAKE_")) and k not in ("RUNPOD_API_KEY", "RUNPOD_POD_ID")}
        env.update(RUNNER_HOST="127.0.0.1", RUN_CMD=json.dumps([sys.executable, str(self.fake)]), RUN_CWD=str(self.dir),
                   CONFIG_ROOT=str(self.configs), OUTPUT_ROOT=str(self.output), DATASETS_ROOT=str(self.datasets),
                   RUNNER_STATE_DIR=str(self.state), RUNNER_WATCH_EVERY_S="0.2", FAKE_HEARTBEAT=str(self.heartbeat),
                   PYTHONUNBUFFERED="1")
        env.update(extra or {})
        return env

    def start(self, token=None, **extra):
        port = free_port()
        env = self.env(dict(extra, RUNNER_PORT=str(port)))
        if token:
            env["RUNNER_TOKEN"] = token
        n = len(self.procs)
        out, err = open(self.dir / f"out{n}.txt", "wb"), open(self.dir / f"err{n}.txt", "wb")
        proc = subprocess.Popen([sys.executable, RUNNER], env=env, stdout=out, stderr=err, stdin=subprocess.DEVNULL)
        proc.port, proc.out_path, proc.err_path = port, self.dir / f"out{n}.txt", self.dir / f"err{n}.txt"
        self.procs.append(proc)
        return proc

    def wait_up(self, proc, secs=10):
        end = time.time() + secs
        while time.time() < end and proc.poll() is None:
            try:
                socket.create_connection(("127.0.0.1", proc.port), 0.5).close()
                return True
            except OSError:
                time.sleep(0.1)
        return False

    def logs(self, proc):
        return proc.out_path.read_bytes().decode("utf-8", "replace") + proc.err_path.read_bytes().decode("utf-8", "replace")

    def close(self):
        for p in self.procs:
            if p.poll() is None:
                p.kill()
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                pass
        time.sleep(0.2)
        shutil.rmtree(self.dir, ignore_errors=True)


def call(proc, method, path, body=None, token=None, headers=None, raw=None):
    """-> (status, parsed JSON or None, response headers)."""
    h = dict(headers or {})
    if token:
        h["Authorization"] = "Bearer " + token
    data = raw
    if body is not None:
        data = json.dumps(body).encode()
        h["Content-Type"] = "application/json"
    conn = http.client.HTTPConnection("127.0.0.1", proc.port, timeout=30)
    try:
        conn.request(method, path, body=data, headers=h)
        r = conn.getresponse()
        payload = r.read()
        try:
            parsed = json.loads(payload)
        except ValueError:
            parsed = None
        return r.status, parsed, dict(r.getheaders())
    finally:
        conn.close()


def wait_status(proc, name, token, want=("done", "failed", "cancelled"), secs=20):
    end = time.time() + secs
    st = None
    while time.time() < end:
        code, js, _ = call(proc, "GET", f"/jobs/{name}", token=token)
        st = js.get("status") if code == 200 else None
        if st in want:
            return st
        time.sleep(0.2)
    return st


def wait_exit(proc, secs):
    try:
        return proc.wait(secs)
    except subprocess.TimeoutExpired:
        return None


def make_config(box, name, steps=5):
    """A config file on disk for the config_path route (JSON text is valid YAML)."""
    box.configs.mkdir(parents=True, exist_ok=True)
    path = box.configs / f"{name}.yaml"
    path.write_text(json.dumps(inline_config(box, name, steps)), encoding="utf-8")
    return path


def inline_config(box, name, steps=5, folder=None, out=None):
    """The config object the app sends as `config` (same shape as webapp/app/training.py config_dict)."""
    return {"job": "extension", "config": {"name": name, "process": [{
        "type": "sd_trainer", "training_folder": str(out or box.output), "device": "cuda:0",
        "datasets": [{"folder_path": str(folder or box.datasets / name), "caption_ext": "txt"}],
        "train": {"steps": steps}}]}}


def with_process(box, name, **changes):
    """inline_config with extra / replaced keys in the one process entry."""
    cfg = inline_config(box, name)
    cfg["config"]["process"][0].update(changes)
    return cfg


PNG = b"\x89PNG\r\n\x1a\n" + b"x" * 40
JPG = b"\xff\xd8\xff\xe0" + b"x" * 40
WEBP = b"RIFF" + b"\x10\x00\x00\x00" + b"WEBP" + b"x" * 20


def make_zip(entries, path=None):
    """entries: [(name or ZipInfo, bytes)] -> zip bytes (or written to path)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for n, data in entries:
            z.writestr(n, data)
    data = buf.getvalue()
    if path:
        Path(path).write_bytes(data)
    return data


def only_empty_incoming(root):
    root = Path(root)
    names = sorted(os.listdir(root)) if root.is_dir() else []
    return names in ([], [".incoming"]) and not (names and any((root / ".incoming").iterdir()))


# ============================================================================= tests
def test_bind_and_config():
    check(runner.bind_refusal("127.0.0.1", "", False) is None, "bind: loopback without a token is fine")
    check(runner.bind_refusal("::1", "", False) is None and runner.bind_refusal("localhost", "", False) is None,
          "bind: ::1 and localhost without a token are fine")
    msg = runner.bind_refusal("0.0.0.0", "", False)
    check(msg and "RUNNER_TOKEN" in msg and "RUNNER_ALLOW_OPEN" in msg, "bind: 0.0.0.0 without a token is refused (message names both variables)")
    check(runner.bind_refusal("0.0.0.0", "", True) is None, "bind: 0.0.0.0 with RUNNER_ALLOW_OPEN is allowed")
    check(runner.bind_refusal("0.0.0.0", TOKEN, False) is None, "bind: 0.0.0.0 with a token is allowed")
    box = Box()
    try:
        p = box.start(token="shorttoken1234")
        code = wait_exit(p, 10)
        err = box.logs(p)
        check(code == 2 and "RUNNER_TOKEN" in err and "shorttoken1234" not in err, f"config: a 14 character token exits 2 naming the variable, never its value (code {code})")
        p = box.start(RUNNER_TIME_LIMIT_S="100000")
        code = wait_exit(p, 10)
        check(code == 2 and "RUNNER_TIME_LIMIT_MAX_S" in box.logs(p), f"config: a time limit over the maximum exits 2 (code {code})")
        p = box.start(token=TOKEN, RUNNER_TIME_LIMIT_S="0")
        code = wait_exit(p, 10)
        check(code == 2, f"config: a token with the time limit switched off exits 2 (code {code})")
        check(runner.TIME_LIMIT_MAX_S == 10800.0, f"config: the default ceiling is 3 hours ({runner.TIME_LIMIT_MAX_S})")
        p = box.start(token=TOKEN, RUNNER_TIME_LIMIT_S="10801")
        code = wait_exit(p, 10)
        check(code == 2 and "RUNNER_TIME_LIMIT_MAX_S" in box.logs(p), f"config: 3 h + 1 s with no RUNNER_TIME_LIMIT_MAX_S exits 2 (code {code})")
        p = box.start(token=TOKEN, RUNNER_TIME_LIMIT_S="10800")
        up = box.wait_up(p)
        check(up and p.poll() is None, "config: a limit of exactly 3 h starts")
        p = box.start(token=TOKEN, RUNNER_TIME_LIMIT_S="10801", RUNNER_TIME_LIMIT_MAX_S="14400")
        up = box.wait_up(p)
        check(up and p.poll() is None, "config: a higher ceiling works when RUNNER_TIME_LIMIT_MAX_S is set at creation")
        entry = (Path(ROOT) / "trainer" / "cloud" / "entrypoint.sh").read_text(encoding="utf-8")
        check("RUNNER_TIME_LIMIT_MAX_S:-10800" in entry, "config: the hung-runner backstop defaults to 3 h + 10 min")
    finally:
        box.close()


def test_auth_and_secrets():
    box = Box()
    try:
        p = box.start(token=TOKEN)
        check(box.wait_up(p), "auth: the runner with a token starts")
        make_config(box, "authjob")
        routes = [("GET", "/jobs", None), ("GET", "/jobs/authjob", None), ("GET", "/datasets/authjob", None),
                  ("GET", "/files/authjob/a.txt", None), ("GET", "/nothing", None),
                  ("POST", "/jobs", {"name": "authjob", "config_path": str(box.configs / "authjob.yaml")}),
                  ("POST", "/jobs/authjob/cancel", {}), ("POST", "/deadline", {"seconds": 600}),
                  ("PUT", "/datasets/authjob", None)]
        bad = []
        for method, path, body in routes:
            raw = b"x" if method == "PUT" else None
            for label, kw in (("no header", {}), ("wrong token", {"token": WRONG}),
                              ("query token", {"alt": path + "?token=" + TOKEN})):
                url = kw.get("alt", path)
                s, js, hd = call(p, method, url, body=body, token=kw.get("token"), raw=raw)
                if s != 401 or js != {"error": "token required"} or hd.get("WWW-Authenticate") != "Bearer":
                    bad.append((method, path, label, s))
        check(not bad, f"auth: every route answers 401 {{error: token required}} + WWW-Authenticate without a header, with a wrong token and with ?token= ({bad})")
        s, js, _ = call(p, "GET", "/health")
        check(s == 200 and js == {"ok": True, "version": runner.VERSION, "auth": "required"}, f"auth: /health without a token is exactly {{ok, version, auth}} ({js})")
        s, js, _ = call(p, "GET", "/health", token=WRONG)
        check(s == 200 and set(js) == {"ok", "version", "auth"}, "auth: /health with a wrong token is the reduced body")
        s, js, _ = call(p, "GET", "/health", token=TOKEN)
        check(s == 200 and "busy" in js and js["deadline_in_s"] > 0 and js["idle_in_s"] > 0, f"auth: /health with the token has busy and both limits ({js})")
        s, js, _ = call(p, "GET", "/jobs", token=TOKEN)
        check(s == 200 and js == [], "auth: GET /jobs with the token -> 200")
        s, js, _ = call(p, "POST", "/jobs", {"name": "authjob", "config_path": str(box.configs / "authjob.yaml")}, token=TOKEN)
        check(s == 201, f"auth: POST /jobs with the token -> 201 ({s})")
        st = wait_status(p, "authjob", TOKEN)
        check(st == "done", f"secrets: the fake run finishes ({st})")
        s, js, _ = call(p, "GET", "/jobs/authjob", token=TOKEN)
        tail = "\n".join(js["log_tail"])
        check("token-in-env: no" in tail, "secrets: the training child sees no RUNNER_TOKEN, HF_WRITE_TOKEN or RUNPOD_API_KEY")
        # the child environment drops the two write keys even when they are set for the runner
        call(p, "GET", "/jobs", token=WRONG)
        p.terminate()
        wait_exit(p, 10)
        texts = [box.logs(p)]
        for f in (box.output / "authjob").glob("*"):
            if f.is_file():
                texts.append(f.read_text(encoding="utf-8", errors="replace"))
        blob = "\n".join(texts)
        check(TOKEN not in blob and WRONG not in blob and TOKEN[:20] not in blob and WRONG[:20] not in blob,
              "secrets: neither token appears in the runner's stdout, stderr, runner.log or runner_status.json")
        check("token: set" in blob and "401" in blob, "secrets: the start-up line says 'token: set' and 401s are logged without headers")
        check((box.output / "authjob" / "runner.log").exists() and (box.output / "authjob" / "runner_status.json").exists(),
              "secrets: runner.log and runner_status.json exist (so the grep above looked at something)")
    finally:
        box.close()
    # the child scrub also covers the write keys
    box = Box()
    try:
        # since ss-m10t8 a HF_WRITE_TOKEN needs HF_REPO, the token form and RUNNER_TOKEN (else the runner exits 2)
        p = box.start(token=TOKEN, HF_WRITE_TOKEN="hf_" + "x" * 30, HF_REPO="test/sprite-studio-loras",
                      RUNPOD_API_KEY="rpa_" + "y" * 20, FAKE_STEPS="3",
                      RUNNER_UPLOAD_CMD=json.dumps([sys.executable, "-c", "pass"]))     # never the real uploader, whatever the image has
        box.wait_up(p)
        make_config(box, "scrub")
        call(p, "POST", "/jobs", {"name": "scrub", "config_path": str(box.configs / "scrub.yaml")}, token=TOKEN)
        wait_status(p, "scrub", TOKEN)
        s, js, _ = call(p, "GET", "/jobs/scrub", token=TOKEN)
        check("token-in-env: no" in "\n".join(js["log_tail"]), "secrets: HF_WRITE_TOKEN and RUNPOD_API_KEY are not passed to the child")
    finally:
        box.close()


def test_no_token_local():
    box = Box()
    try:
        p = box.start()
        box.wait_up(p)
        make_config(box, "local")
        s, js, _ = call(p, "GET", "/health")
        check(s == 200 and "busy" in js and js["deadline_in_s"] is None and js["idle_in_s"] is None, f"no token: /health is today's body plus null limits ({js})")
        s, js, _ = call(p, "POST", "/jobs", {"name": "local", "config_path": str(box.configs / "local.yaml")})
        check(s == 201, "no token: POST /jobs works without a header")
        check(wait_status(p, "local", None) == "done", "no token: a fake run reaches done")
        s, js, _ = call(p, "POST", "/deadline", {"seconds": 600})
        check(s == 409, f"no token: POST /deadline -> 409 when no limit is active ({s})")
        check("token: none" in box.logs(p), "no token: the start-up line says 'token: none'")
    finally:
        box.close()


def test_extract_dataset():
    tmp = Path(tempfile.mkdtemp(prefix="trainer-zip-"))
    root = tmp / "datasets"
    saved = (runner.IMAGE_MAX, runner.TEXT_MAX, runner.TOTAL_MAX, runner.MAX_ENTRIES)
    try:
        runner.IMAGE_MAX, runner.TEXT_MAX, runner.TOTAL_MAX = 1 << 20, 64 << 10, 1 << 30

        def try_zip(entries, label, expect, name="ds", keep=False):
            zp = tmp / "in.zip"
            make_zip(entries, zp)
            try:
                runner.extract_dataset(zp, root, name)
                check(False, f"zip: {label} must be refused")
            except ValueError as e:
                check(expect in str(e) and (only_empty_incoming(root) or (keep and not any((root / ".incoming").iterdir()))),
                      f"zip: {label} -> refused ({e}) and nothing is kept")

        zp = tmp / "good.zip"
        make_zip([("a.png", PNG), ("a.txt", "a cat".encode()), ("b.JPG", JPG), ("c.webp", WEBP), ("__MACOSX/a.png", b"junk"),
                  ("._a.png", b"junk"), ("folder/", b""), ("folder/._b.png", b"junk")], zp)
        info = runner.extract_dataset(zp, root, "ds")
        files = sorted(os.listdir(root / "ds"))
        check(info["images"] == 3 and info["captions"] == 1 and files == ["a.png", "a.txt", "b.JPG", "c.webp"]
              and info["dataset_dir"] == str(root / "ds") and info["bytes"] > 0 and only_empty_incoming(root / "ds") is not None,
              f"zip: a good flat zip lands in <root>/<name>/ and __MACOSX, ._ and directory entries are skipped ({files}, {info})")
        check(sorted(os.listdir(root)) == [".incoming", "ds"] and not any((root / ".incoming").iterdir()), "zip: .incoming is empty afterwards")
        shutil.rmtree(root)
        for bad in ("../a.png", "/a.png", "sub/a.png", "..\\a.png", "C:\\a.png", "a.py", "a.png.exe"):
            try_zip([("good.png", PNG), (bad, PNG)], f"entry {bad!r}", "entry not allowed")
        link = zipfile.ZipInfo("a.png")
        link.external_attr = 0o120777 << 16
        try_zip([("ok.png", PNG), (link, b"/etc/passwd")], "a symlink entry", "symbolic link")
        try_zip([("a.png", b"this is not a png at all")], "a .png that is not a picture", "not a png")
        try_zip([("a.png", PNG), ("a.txt", b"\xff\xfe\x00bad")], "a caption that is not UTF-8", "not UTF-8")
        try_zip([("a.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * (40 << 20))], "a 40 MiB zero-filled a.png against a 1 MiB cap", "too large")
        try_zip([("a.txt", b"x" * 70000), ("b.png", PNG)], "a caption over 64 KiB", "too large")
        runner.MAX_ENTRIES = 1000
        try_zip([(f"{i}.png", PNG) for i in range(1001)], "1001 entries", "too many entries")
        try_zip([("a.png", PNG), ("A.PNG", PNG)], "a case-insensitive duplicate", "duplicate")
        try_zip([("a.txt", b"only text")], "no image", "no images")
        try_zip([("__MACOSX/a.png", PNG), ("._a.png", PNG)], "only skipped entries", "no images")
        zp.write_bytes(b"PK this is not a zip")
        try:
            runner.extract_dataset(zp, root, "ds")
            check(False, "zip: garbage must be refused")
        except ValueError as e:
            check("not a readable zip" in str(e) and only_empty_incoming(root), f"zip: a file that is not a zip -> refused ({e})")
        enc = tmp / "enc.zip"
        make_zip([("a.png", PNG)], enc)
        raw = bytearray(enc.read_bytes())
        for sig in (b"PK\x03\x04", b"PK\x01\x02"):          # set the encrypted flag in the local and central headers
            i = raw.find(sig)
            raw[i + (6 if sig == b"PK\x03\x04" else 8)] |= 1
        enc.write_bytes(bytes(raw))
        try:
            runner.extract_dataset(enc, root, "ds")
            check(False, "zip: an encrypted entry must be refused")
        except ValueError as e:
            check("encrypted" in str(e) and only_empty_incoming(root), f"zip: an encrypted entry -> refused ({e})")
        # a failed upload leaves the earlier dataset untouched
        zp = tmp / "first.zip"
        make_zip([("a.png", PNG), ("old.png", PNG)], zp)
        runner.extract_dataset(zp, root, "keep")
        try_zip([("zzz.exe", b"x")], "a bad zip over an existing dataset", "entry not allowed", name="keep", keep=True)
        check(sorted(os.listdir(root / "keep")) == ["a.png", "old.png"], "zip: a refused upload leaves the existing dataset as it was")
    finally:
        runner.IMAGE_MAX, runner.TEXT_MAX, runner.TOTAL_MAX, runner.MAX_ENTRIES = saved
        shutil.rmtree(tmp, ignore_errors=True)


def test_upload_http():
    box = Box()
    try:
        p = box.start(token=TOKEN, RUNNER_UPLOAD_MAX_BYTES="200000", FAKE_STEPS="400", FAKE_STEP_S="0.1")
        box.wait_up(p)
        z1 = make_zip([("a.png", PNG), ("a.txt", b"one"), ("old.png", PNG)])
        sha1 = hashlib.sha256(z1).hexdigest()
        s, js, _ = call(p, "PUT", "/datasets/up", raw=z1, token=TOKEN, headers={"Content-Type": "application/zip", "X-Content-SHA256": sha1})
        check(s == 201 and js["sha256"] == sha1 and js["images"] == 2 and js["captions"] == 1 and js["name"] == "up"
              and js["dataset_dir"].replace("\\", "/").endswith("/datasets/up"), f"http: 201 with matching sha256 and counts ({s} {js})")
        s, js, _ = call(p, "GET", "/datasets/up", token=TOKEN)
        check(s == 200 and js["images"] == 2 and js["captions"] == 1 and js["bytes"] > 0, f"http: GET /datasets/<name> ({js})")
        s, js, _ = call(p, "GET", "/datasets/nothere", token=TOKEN)
        check(s == 404, "http: GET /datasets/<missing> -> 404")
        # missing Content-Length
        conn = http.client.HTTPConnection("127.0.0.1", p.port, timeout=10)
        conn.putrequest("PUT", "/datasets/up")
        conn.putheader("Authorization", "Bearer " + TOKEN)
        conn.endheaders()
        r = conn.getresponse()
        check(r.status == 411, f"http: no Content-Length -> 411 ({r.status})")
        conn.close()
        # over the limit, answered without the body being sent
        conn = http.client.HTTPConnection("127.0.0.1", p.port, timeout=10)
        conn.putrequest("PUT", "/datasets/up")
        conn.putheader("Authorization", "Bearer " + TOKEN)
        conn.putheader("Content-Length", str(10 ** 9))
        conn.endheaders()
        r = conn.getresponse()
        check(r.status == 413 and r.getheader("Connection", "").lower() == "close", f"http: over the limit -> 413 + Connection: close, answered without the body ({r.status})")
        conn.close()
        s, js, _ = call(p, "PUT", "/datasets/up", raw=z1, token=TOKEN, headers={"X-Content-SHA256": "0" * 64})
        check(s == 400 and "SHA256" in js["error"], f"http: a wrong X-Content-SHA256 -> 400 ({s})")
        s, js, _ = call(p, "PUT", "/datasets/bad%20name", raw=z1, token=TOKEN)
        check(s == 400, "http: a bad dataset name -> 400")
        s, js, _ = call(p, "PUT", "/datasets/up", raw=make_zip([("x.py", b"print(1)")]), token=TOKEN)
        check(s == 400 and "entry not allowed" in js["error"], f"http: a bad zip -> 400 naming the entry ({js})")
        # short body
        conn = http.client.HTTPConnection("127.0.0.1", p.port, timeout=10)
        conn.putrequest("PUT", "/datasets/up")
        conn.putheader("Authorization", "Bearer " + TOKEN)
        conn.putheader("Content-Length", str(len(z1) + 500))
        conn.endheaders()
        conn.send(z1)
        conn.sock.shutdown(socket.SHUT_WR)
        r = conn.getresponse()
        check(r.status == 400, f"http: a body shorter than Content-Length -> 400 ({r.status})")
        conn.close()
        check(sorted(os.listdir(box.datasets / "up")) == ["a.png", "a.txt", "old.png"] and not any((box.datasets / ".incoming").iterdir()),
              "http: after the refused uploads the first dataset is intact and .incoming is empty")
        # second upload replaces the first
        z2 = make_zip([("a.png", PNG), ("b.webp", WEBP)])
        s, js, _ = call(p, "PUT", "/datasets/up", raw=z2, token=TOKEN, headers={"X-Content-SHA256": hashlib.sha256(z2).hexdigest().upper()})
        check(s == 201 and sorted(os.listdir(box.datasets / "up")) == ["a.png", "b.webp"], f"http: a second upload replaces the first (old extra file gone) ({s})")
        # upload while a job of that name runs
        s, js, _ = call(p, "POST", "/jobs", {"name": "up", "config": inline_config(box, "up", 400)}, token=TOKEN)
        check(s == 201, f"http: a run from the uploaded dataset starts ({s} {js})")
        s, js, _ = call(p, "PUT", "/datasets/up", raw=z1, token=TOKEN)
        check(s == 409, f"http: upload while a job of that name runs -> 409 ({s})")
        call(p, "POST", "/jobs/up/cancel", {}, token=TOKEN)
        check(wait_status(p, "up", TOKEN) == "cancelled", "http: the run is cancelled")
        s, js, _ = call(p, "PUT", "/datasets/up", raw=z1, token=TOKEN)
        check(s == 201, f"http: after the run ended the upload works again ({s})")
    finally:
        box.close()


def test_inline_config():
    box = Box()
    try:
        p = box.start()
        box.wait_up(p)
        st, js, _ = call(p, "POST", "/jobs", {"name": "inl", "config": inline_config(box, "inl")})
        check(st == 201, f"config: a run from `config` (a JSON object) starts ({st} {js})")
        check(wait_status(p, "inl", None) == "done", "config: it reaches done")
        written = json.loads((box.configs / "inl.json").read_text(encoding="utf-8"))
        check(written == inline_config(box, "inl"), "config: written to CONFIG_ROOT/<name>.json from the parsed object")
        check(js["total_steps"] == 5, f"config: total steps come from the inline object ({js['total_steps']})")
        out = Path(tempfile.gettempdir()) / "elsewhere"
        one = inline_config(box, "bad")["config"]["process"][0]
        bad = [
            ("folder_path outside", inline_config(box, "bad", folder=out)),
            ("folder_path with ..", inline_config(box, "bad", folder=box.datasets / ".." / "x")),
            ("folder_path is the datasets root", inline_config(box, "bad", folder=box.datasets)),
            ("folder_path in .incoming", inline_config(box, "bad", folder=box.datasets / ".incoming" / "x")),
            ("training_folder outside", inline_config(box, "bad", out=out)),
            ("config.name that climbs out of the output folder", inline_config(box, "../../etc/cron.d")),
            ("config.name that is not the job name", inline_config(box, "other")),
            ("control_path", with_process(box, "bad", control_path="/root")),
            ("mask_path in a dataset", with_process(box, "bad", datasets=[{"folder_path": str(box.datasets / "bad"), "mask_path": "/root"}])),
            ("clip_image_path in a dataset", with_process(box, "bad", datasets=[{"folder_path": str(box.datasets / "bad"), "clip_image_path": "/etc"}])),
            ("a remote model that is not on the list", with_process(box, "bad", model={"name_or_path": "someone/any-repo"})),
            ("a local model path", with_process(box, "bad", model={"name_or_path": "/etc"})),
            ("an adapter that is not on the list", with_process(box, "bad", model={"assistant_lora_path": "someone/x/y.safetensors"})),
            ("push_to_hub", with_process(box, "bad", save={"push_to_hub": True, "hf_repo_id": "someone/x"})),
            ("hf_repo_id alone", with_process(box, "bad", save={"hf_repo_id": "someone/x"})),
            ("an unknown top-level key", {**inline_config(box, "bad"), "extra": 1}),
            ("an unknown process key", with_process(box, "bad", python_hook="x")),
            ("two process entries", {"job": "extension", "config": {"name": "bad", "process": [one, one]}}),
            ("text instead of an object", "job: extension\nconfig:\n  name: bad\n"),
            ("a list", [1]),
            ("a number", 5),
            ("more than 256 KiB", with_process(box, "bad", trigger_word="x" * 262200)),
        ]
        for label, cfg in bad:
            st, js, _ = call(p, "POST", "/jobs", {"name": "bad", "config": cfg})
            check(st == 400 and not (box.configs / "bad.json").exists(), f"config: {label} -> 400, nothing written ({st} {js.get('error') if js else ''})")
        # The old bypasses as raw bodies: a key and a value written with \u escapes, and NaN.
        escaped_key = '{"name":"bad","config":{"job":"extension","config":{"name":"bad","process":[{"datasets":[{"folder\\u005fpath":"/etc"}]}]}}}'
        escaped_val = '{"name":"bad","config":{"job":"extension","config":{"name":"bad","process":[{"type":"sd_trainer","training_folder":"/tmp/\\u002e\\u002e/x"}]}}}'
        nan_body = '{"name":"bad","config":{"job":"extension","config":{"name":"bad","process":[{"train":{"lr":NaN}}]}}}'
        for label, raw in (("an escaped key", escaped_key), ("an escaped value", escaped_val), ("NaN", nan_body)):
            st, js, _ = call(p, "POST", "/jobs", raw=raw.encode(), headers={"Content-Type": "application/json"})
            check(st == 400 and not (box.configs / "bad.json").exists(), f"config: {label} -> 400 ({st})")
        st, js, _ = call(p, "POST", "/jobs", {"name": "both", "config": inline_config(box, "both"), "config_path": str(box.configs / "inl.json")})
        check(st == 400 and "not both" in js["error"], f"config: config and config_path together -> 400 ({st})")
        for name in ("job\n", "a b", "../x"):
            st, js, _ = call(p, "POST", "/jobs", {"name": name, "config": inline_config(box, "job")})
            check(st == 400, f"names: {name!r} -> 400")
        check(not runner.NAME_RE.match("job\n") and not runner.ENTRY_RE.match("a.png\n"), "names: a trailing newline does not match NAME_RE / ENTRY_RE")
    finally:
        box.close()


def test_check_inline_config_in_process():
    """The checker alone (no HTTP)."""
    box = Box()
    old = (runner.OUTPUT_ROOT, runner.DATASETS_ROOT)
    try:
        runner.OUTPUT_ROOT, runner.DATASETS_ROOT = box.output, box.datasets
        check(runner.check_inline_config(inline_config(box, "ok"), "ok") is None, "checker: the app's shape passes")
        good_model = with_process(box, "ok", model={"name_or_path": "Tongyi-MAI/Z-Image-Turbo", "arch": "zimage"})
        check(runner.check_inline_config(good_model, "ok") is None, "checker: the allowed base model passes")
        why = runner.check_inline_config(with_process(box, "ok", save={"push_to_hub": True}), "ok")
        check(why and "push_to_hub" in why, f"checker: the refusal names the key ({why})")
        why = runner.check_inline_config(with_process(box, "ok", train={"steps": True}), "ok")
        check(why and "whole number" in why, "checker: a boolean is not a number")
    finally:
        runner.OUTPUT_ROOT, runner.DATASETS_ROOT = old
        box.close()


def raw_request(p, method, path, content_length):
    conn = http.client.HTTPConnection("127.0.0.1", p.port, timeout=10)
    conn.putrequest(method, path)
    conn.putheader("Authorization", "Bearer " + TOKEN)
    conn.putheader("Content-Length", content_length)
    conn.endheaders()
    try:
        return conn.getresponse().status
    except (http.client.HTTPException, OSError) as e:
        return type(e).__name__
    finally:
        conn.close()


def test_small_hardening():
    box = Box()
    try:
        p = box.start(token=TOKEN)
        box.wait_up(p)
        for label, cl in (("a negative", "-1"), ("a non-number", "abc"), ("a unicode digit", "²")):
            r = raw_request(p, "POST", "/jobs", cl)
            check(r in (400, 411), f"hardening: {label} Content-Length on a JSON route -> 400/411 ({r})")
        r = raw_request(p, "PUT", "/datasets/up", "²")
        check(r == 411, f"hardening: a unicode digit as the upload length -> 411 ({r})")
        check("Traceback" not in box.logs(p), "hardening: no traceback in the logs")
        st, js, _ = call(p, "PUT", "/datasets/job%0A", raw=make_zip([("a.png", PNG)]), token=TOKEN)
        folders = [d for d in os.listdir(box.datasets) if d != ".incoming"] if box.datasets.exists() else []
        check(st == 400 and not folders, f"hardening: PUT /datasets/job%0A -> 400, no folder ({st})")
        p2 = box.start(token=" " * 40)
        check(wait_exit(p2, 10) == 2, "hardening: a token of spaces exits 2")
        # one upload at a time
        big = make_zip([("a.png", PNG + os.urandom(150000))])
        slow = http.client.HTTPConnection("127.0.0.1", p.port, timeout=20)
        slow.putrequest("PUT", "/datasets/slow")
        slow.putheader("Authorization", "Bearer " + TOKEN)
        slow.putheader("Content-Length", str(len(big)))
        slow.endheaders()
        slow.send(big[:1000])
        time.sleep(0.5)
        st, js, _ = call(p, "PUT", "/datasets/other", raw=make_zip([("a.png", PNG)]), token=TOKEN)
        check(st == 409 and "another upload" in js["error"], f"hardening: a second upload while one is running -> 409 ({st})")
        slow.send(big[1000:])
        slow.getresponse().read()
        slow.close()
        st, js, _ = call(p, "PUT", "/datasets/other", raw=make_zip([("a.png", PNG)]), token=TOKEN)
        check(st == 201, f"hardening: ... and the next one works ({st})")
    finally:
        box.close()


def write_boot(box, **d):
    box.state.mkdir(parents=True, exist_ok=True)
    (box.state / "boot.json").write_text(json.dumps(d), encoding="utf-8")


def test_boot_json_is_per_pod():
    # another pod's deadline, already past, on a reused volume: the new pod starts a fresh clock
    box = Box()
    try:
        write_boot(box, pod="oldpod123456", first_start=time.time() - 100, deadline=time.time() - 10)
        p = box.start(token=TOKEN, RUNNER_TIME_LIMIT_S="3600", RUNPOD_POD_ID="newpod654321")
        up = box.wait_up(p)
        st, js, _ = call(p, "GET", "/health", token=TOKEN) if up else (0, {}, {})
        check(up and p.poll() is None and st == 200 and 3500 < js["deadline_in_s"] <= 3600,
              f"boot: another pod's expired boot.json is ignored, the new pod gets a fresh clock ({js.get('deadline_in_s')})")
        boot = json.loads((box.state / "boot.json").read_text(encoding="utf-8"))
        check(boot["pod"] == "newpod654321" and boot["deadline"] > time.time() + 3000, "boot: boot.json now belongs to the new pod")
    finally:
        box.close()
    # another pod's unexpired deadline must not shorten the new pod either
    box = Box()
    try:
        write_boot(box, pod="oldpod123456", first_start=time.time() - 100, deadline=time.time() + 30)
        p = box.start(token=TOKEN, RUNNER_TIME_LIMIT_S="3600", RUNPOD_POD_ID="newpod654321")
        box.wait_up(p)
        st, js, _ = call(p, "GET", "/health", token=TOKEN)
        check(js.get("deadline_in_s", 0) > 3000, f"boot: another pod's unexpired deadline does not shorten the new pod ({js.get('deadline_in_s')})")
    finally:
        box.close()
    # the same pod restarting keeps its clock; a file with no pod id binds nobody; RUNNER_BOOT_ID wins
    box = Box()
    try:
        write_boot(box, pod="samepod123456", first_start=time.time() - 100, deadline=time.time() - 10)
        t0 = time.time()
        p = box.start(token=TOKEN, RUNNER_TIME_LIMIT_S="3600", RUNPOD_POD_ID="samepod123456")
        code = wait_exit(p, 10)
        check(code == 3 and time.time() - t0 < 6, f"boot: the same pod restarting past its deadline still exits 3 ({code})")
        write_boot(box, first_start=time.time() - 100, deadline=time.time() - 10)
        p = box.start(token=TOKEN, RUNNER_TIME_LIMIT_S="3600", RUNPOD_POD_ID="samepod123456")
        box.wait_up(p)
        check(p.poll() is None, "boot: a boot.json with no pod id does not bind a pod (fresh clock)")
        write_boot(box, pod="bootid-1", first_start=time.time() - 100, deadline=time.time() - 10)
        p = box.start(token=TOKEN, RUNNER_TIME_LIMIT_S="3600", RUNPOD_POD_ID="whatever12345", RUNNER_BOOT_ID="bootid-1")
        check(wait_exit(p, 10) == 3, "boot: RUNNER_BOOT_ID is the id when given")
    finally:
        box.close()


def test_watchdog_fails_closed():
    box = Box()
    try:
        p = box.start(token=TOKEN, RUNNER_WATCH_EVERY_S="inf")
        check(wait_exit(p, 10) == 2 and "RUNNER_WATCH_EVERY_S" in box.logs(p), "watchdog: RUNNER_WATCH_EVERY_S=inf exits 2 naming the variable")
        p = box.start(token=TOKEN, RUNNER_TIME_LIMIT_MAX_S="inf")
        check(wait_exit(p, 10) == 2 and "RUNNER_TIME_LIMIT_MAX_S" in box.logs(p), "watchdog: RUNNER_TIME_LIMIT_MAX_S=inf exits 2")
        p = box.start(token=TOKEN, RUNNER_WATCH_EVERY_S="nan")
        check(wait_exit(p, 10) == 2, "watchdog: nan exits 2")
    finally:
        box.close()
    # an exception inside the loop ends the runner; it must not go on with no limit
    saved = (runner.current, runner._exit, runner.IDLE_LIMIT_S, runner.WATCH_EVERY_S, runner.DEADLINE)
    exits = []

    def boom():
        raise RuntimeError("test")
    try:
        runner.current, runner._exit = boom, exits.append
        runner.IDLE_LIMIT_S, runner.WATCH_EVERY_S, runner.DEADLINE = 1.0, 0.01, None
        t = threading.Thread(target=runner.watchdog, daemon=True)
        t.start()
        t.join(5)
        check(exits == [3] and not t.is_alive(), f"watchdog: an exception in the loop calls exit(3) ({exits})")
    finally:
        runner.current, runner._exit, runner.IDLE_LIMIT_S, runner.WATCH_EVERY_S, runner.DEADLINE = saved


STUBBORN = r"""
import os, signal, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)          # ignores the polite stop: only SIGKILL ends it
open(os.environ["FAKE_PIDFILE"], "w").write(str(os.getpid()))
hb = os.environ["FAKE_HEARTBEAT"]
while True:
    open(hb, "w").write(str(time.time()))
    time.sleep(0.1)
"""


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_pid(path, secs=10):
    end = time.time() + secs
    while time.time() < end:
        try:
            return int(path.read_text())
        except (OSError, ValueError):
            time.sleep(0.1)
    return 0


def test_linux_kill_path():
    """killpg(SIGTERM), the grace period, killpg(SIGKILL) and the runner's SIGTERM handler, with a trainer that ignores
    SIGTERM. POSIX only: on Windows _kill uses taskkill and there are no signals, so it is skipped there with this
    reason and runs in a Linux container:
        docker run --rm -v <repo>:/w:ro python:3.12-slim python /w/tests/test_trainer_runner.py kill"""
    if os.name == "nt":
        print("SKIP linux kill path: needs POSIX signals and process groups (Windows uses taskkill); run it in a Linux container")
        return
    import signal
    box = Box()
    pids = []
    try:
        pidfile = box.dir / "child.pid"
        stub = box.dir / "stubborn.py"
        stub.write_text(STUBBORN, encoding="utf-8")
        extra = dict(RUN_CMD=json.dumps([sys.executable, str(stub)]), FAKE_PIDFILE=str(pidfile), RUNNER_KILL_GRACE_S="1")
        # 1. the time limit
        p = box.start(token=TOKEN, RUNNER_TIME_LIMIT_S="3", **extra)
        box.wait_up(p)
        t0 = time.time()
        call(p, "POST", "/jobs", {"name": "stub", "config": inline_config(box, "stub", 2000)}, token=TOKEN)
        pid = read_pid(pidfile)
        pids.append(pid)
        code = wait_exit(p, 25)
        took = time.time() - t0
        time.sleep(0.5)
        check(pid and code == 3 and not alive(pid), f"kill: a trainer that ignores SIGTERM is SIGKILLed at the limit, the runner exits 3 ({code}, child alive: {alive(pid)})")
        check(took < 3 + 1 + 5 + 4, f"kill: it took the limit + grace + slack ({took:.1f}s)")
        m1 = box.heartbeat.stat().st_mtime
        time.sleep(0.6)
        check(m1 == box.heartbeat.stat().st_mtime, "kill: the heartbeat stops")
        st = json.loads((box.output / "stub" / "runner_status.json").read_text(encoding="utf-8"))
        check(st["status"] == "cancelled" and "time limit" in (st["error"] or ""), f"kill: status is cancelled with the limit as the reason ({st['status']})")
        # 2. a cancel through the API (the same _kill)
        pidfile.unlink()
        p = box.start(token=TOKEN, RUNNER_TIME_LIMIT_S="600", RUNNER_STATE_DIR=str(box.dir / "state2"), **extra)
        box.wait_up(p)
        call(p, "POST", "/jobs", {"name": "stub2", "config": inline_config(box, "stub2", 2000)}, token=TOKEN)
        pid = read_pid(pidfile)
        pids.append(pid)
        call(p, "POST", "/jobs/stub2/cancel", {}, token=TOKEN)
        check(pid and wait_status(p, "stub2", TOKEN, secs=15) == "cancelled" and not alive(pid), "kill: cancel also reaches SIGKILL for a trainer that ignores SIGTERM")
        # 3. the runner gets SIGTERM (a RunPod stop): it leaves with code 0
        pidfile.unlink()
        call(p, "POST", "/jobs", {"name": "stub3", "config": inline_config(box, "stub3", 2000)}, token=TOKEN)
        pids.append(read_pid(pidfile))
        p.send_signal(signal.SIGTERM)
        check(wait_exit(p, 15) == 0, "kill: SIGTERM to the runner -> exit code 0")
    finally:
        for pid in pids:
            if pid:
                try:
                    os.kill(pid, 9)
                except OSError:
                    pass
        box.close()


def test_readme_is_honest_about_the_token():
    text = (Path(ROOT) / "trainer" / "README.md").read_text(encoding="utf-8")
    check("/proc/1/environ" in text and "accidental" in text.lower(), "readme: says what the token scrub covers (accidental dumps, /proc/1/environ)")
    check("training process never sees it" not in text, "readme: no longer claims the training process never sees the token")
    check("JSON object" in text and "push_to_hub" in text, "readme: the config route describes the JSON object and the refusals")


def test_time_and_idle_limits():
    box = Box()
    try:
        p = box.start(token=TOKEN, RUNNER_TIME_LIMIT_S="3", FAKE_STEPS="2000", FAKE_STEP_S="0.1")
        box.wait_up(p)
        t0 = time.time()
        s, js, _ = call(p, "POST", "/jobs", {"name": "long", "config": inline_config(box, "long", 2000)}, token=TOKEN)
        code = wait_exit(p, 25)
        took = time.time() - t0
        check(s == 201 and code == 3 and took < 25, f"limit: the runner exits with code 3 at its time limit, with no client ({code}, {took:.1f}s)")
        st = json.loads((box.output / "long" / "runner_status.json").read_text(encoding="utf-8"))
        check(st["status"] == "cancelled" and "time limit" in (st["error"] or ""), f"limit: runner_status.json says cancelled with 'time limit' ({st['status']}, {st['error']})")
        check("time limit reached" in box.logs(p), "limit: 'runner: time limit reached' is printed")
        m1 = box.heartbeat.stat().st_mtime
        time.sleep(0.8)
        m2 = box.heartbeat.stat().st_mtime
        check(m1 == m2, "limit: the heartbeat file stops changing (the training child is gone)")
        boot = json.loads((box.state / "boot.json").read_text(encoding="utf-8"))
        check(abs(boot["deadline"] - boot["first_start"] - 3) < 0.01, "limit: boot.json holds first_start and deadline")
    finally:
        box.close()
    # a restart keeps the first start
    box = Box()
    try:
        box.state.mkdir(parents=True)
        (box.state / "boot.json").write_text(json.dumps({"first_start": time.time() - 100, "deadline": time.time() - 10}), encoding="utf-8")
        t0 = time.time()
        p = box.start(token=TOKEN, RUNNER_TIME_LIMIT_S="3600")
        code = wait_exit(p, 10)
        check(code == 3 and time.time() - t0 < 3 + 3, f"limit: a boot.json already past its deadline -> exit 3 at once ({code}, {time.time() - t0:.1f}s)")
        check("time limit reached" in box.logs(p), "limit: ... and says so")
    finally:
        box.close()
    # idle limit
    box = Box()
    try:
        p = box.start(token=TOKEN, RUNNER_TIME_LIMIT_S="3600", RUNNER_IDLE_LIMIT_S="2")
        box.wait_up(p)
        t0 = time.time()
        code = wait_exit(p, 12)
        check(code == 3 and "idle limit reached" in box.logs(p), f"idle: no job and no request for 2 s -> exit 3 ({code}, {time.time() - t0:.1f}s)")
    finally:
        box.close()
    box = Box()
    try:
        # A 5 s idle limit and a request every second: a busy PC would have to stall this test for 4 s to lose the race
        # (with a 2 s limit a 1 s stall was enough). The check is as strict as before: the runner must still be up after
        # more than the idle limit has passed, and must stop after the requests stop.
        p = box.start(token=TOKEN, RUNNER_TIME_LIMIT_S="3600", RUNNER_IDLE_LIMIT_S="5")
        box.wait_up(p)
        t0 = time.time()
        refused = None
        while time.time() - t0 < 7.5:
            try:
                call(p, "GET", "/jobs", token=TOKEN)
                call(p, "GET", "/jobs", token=WRONG)                  # a refused request does not count
            except OSError as e:                                      # the runner went away: record it, do not abort the file
                refused = e
                break
            time.sleep(1)
        check(refused is None and p.poll() is None, f"idle: an authenticated GET /jobs every second keeps it up for 7 s, past the 5 s limit ({refused!r})")
        code = wait_exit(p, 20)
        check(code == 3, f"idle: ... and it stops about 5 s after the last request ({code})")
    finally:
        box.close()
    box = Box()
    try:
        p = box.start(token=TOKEN, RUNNER_TIME_LIMIT_S="3600", RUNNER_IDLE_LIMIT_S="2", FAKE_STEPS="40", FAKE_STEP_S="0.1")
        box.wait_up(p)
        call(p, "POST", "/jobs", {"name": "busy", "config": inline_config(box, "busy", 40)}, token=TOKEN)
        time.sleep(3.2)
        check(p.poll() is None, "idle: a running job is not idle")
    finally:
        box.close()
    # POST /deadline
    box = Box()
    try:
        p = box.start(token=TOKEN, RUNNER_TIME_LIMIT_S="300", RUNNER_TIME_LIMIT_MAX_S="600")
        box.wait_up(p)
        s, js, _ = call(p, "POST", "/deadline", {"seconds": 600})
        check(s == 401, "deadline: without the token -> 401")
        s, js, _ = call(p, "GET", "/health", token=TOKEN)
        check(240 < js["deadline_in_s"] <= 300, f"deadline: starts near the limit ({js['deadline_in_s']})")
        s, js, _ = call(p, "POST", "/deadline", {"seconds": 400}, token=TOKEN)
        check(s == 200 and 395 <= js["deadline_in_s"] <= 400, f"deadline: with the token sets deadline_in_s near the value ({s} {js})")
        s, js, _ = call(p, "POST", "/deadline", {"seconds": 86400}, token=TOKEN)
        check(s == 200 and 500 < js["deadline_in_s"] <= 600, f"deadline: a request beyond the maximum is capped at first start + max ({js})")
        s, js, _ = call(p, "POST", "/deadline", {"seconds": 120}, token=TOKEN)
        check(s == 200 and 115 <= js["deadline_in_s"] <= 120, f"deadline: it can shorten the deadline too ({js})")
        boot = json.loads((box.state / "boot.json").read_text(encoding="utf-8"))
        check(abs(boot["deadline"] - time.time() - 120) < 5, "deadline: the new deadline is saved in boot.json")
        for bad in (59, 86401, "600", 1.5, True, None):
            s, js, _ = call(p, "POST", "/deadline", {"seconds": bad}, token=TOKEN)
            if s != 400:
                check(False, f"deadline: seconds={bad!r} -> 400 ({s})")
                break
        else:
            check(True, "deadline: seconds outside 60-86400 or not an integer -> 400")
    finally:
        box.close()


def run(only=None):
    del RESULTS[:]
    if only:                                          # python tests/test_trainer_runner.py kill  -> runs test_*kill*
        for name, fn in sorted(globals().items()):
            if name.startswith("test_") and name != "test_trainer_runner" and callable(fn) and any(o in name for o in only):
                fn()
        print("\n%d/%d passed" % (sum(RESULTS), len(RESULTS)))
        return all(RESULTS)
    test_bind_and_config()
    test_auth_and_secrets()
    test_no_token_local()
    test_extract_dataset()
    test_upload_http()
    test_inline_config()
    test_check_inline_config_in_process()
    test_small_hardening()
    test_boot_json_is_per_pod()
    test_watchdog_fails_closed()
    test_linux_kill_path()
    test_readme_is_honest_about_the_token()
    test_time_and_idle_limits()
    print("\n%d/%d passed" % (sum(RESULTS), len(RESULTS)))
    return all(RESULTS)


def test_trainer_runner():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run(sys.argv[1:]) else 1)
