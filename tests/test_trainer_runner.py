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
        env = {k: v for k, v in os.environ.items() if not k.startswith(("RUNNER_", "RUN_", "FAKE_")) and k != "RUNPOD_API_KEY"}
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
    box.configs.mkdir(parents=True, exist_ok=True)
    path = box.configs / f"{name}.yaml"
    path.write_text(inline_config(box, name, steps), encoding="utf-8")
    return path


def inline_config(box, name, steps=5, folder=None, out=None):
    return (f'job: extension\nconfig:\n  name: "{name}"\n  process:\n    - type: sd_trainer\n'
            f'      training_folder: "{(out or box.output).as_posix()}"\n      datasets:\n'
            f'        - folder_path: "{(folder or box.datasets / name).as_posix()}"\n      train:\n        steps: {steps}\n')


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
        p = box.start(HF_WRITE_TOKEN="hf_write_" + "x" * 20, RUNPOD_API_KEY="rpa_" + "y" * 20, FAKE_STEPS="3")
        box.wait_up(p)
        make_config(box, "scrub")
        call(p, "POST", "/jobs", {"name": "scrub", "config_path": str(box.configs / "scrub.yaml")})
        wait_status(p, "scrub", None)
        s, js, _ = call(p, "GET", "/jobs/scrub")
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
        s, js, _ = call(p, "POST", "/jobs", {"name": "inl", "config": inline_config(box, "inl")})
        check(s == 201, f"config: a run from `config` starts ({s} {js})")
        check(wait_status(p, "inl", None) == "done", "config: it reaches done")
        check((box.configs / "inl.yaml").read_text(encoding="utf-8") == inline_config(box, "inl"), "config: written to CONFIG_ROOT/<name>.yaml")
        check(js["total_steps"] == 5, f"config: total steps come from the inline text ({js['total_steps']})")
        out = Path(tempfile.gettempdir()) / "elsewhere"
        for label, text in (("folder_path outside", inline_config(box, "x", folder=out)),
                            ("folder_path with ..", inline_config(box, "x", folder=box.datasets / ".." / "x")),
                            ("training_folder outside", inline_config(box, "x", out=out)),
                            ("a flow-style folder_path", "datasets: [{folder_path: /etc}]\n"),
                            ("more than 256 KiB", "# " + "x" * 262200 + "\n")):
            s, js, _ = call(p, "POST", "/jobs", {"name": "bad", "config": text})
            check(s == 400 and not (box.configs / "bad.yaml").exists(), f"config: {label} -> 400, nothing written ({s} {js.get('error') if js else ''})")
        s, js, _ = call(p, "POST", "/jobs", {"name": "both", "config": inline_config(box, "both"), "config_path": str(box.configs / "inl.yaml")})
        check(s == 400 and "not both" in js["error"], f"config: config and config_path together -> 400 ({s})")
        s, js, _ = call(p, "POST", "/jobs", {"name": "num", "config": 5})
        check(s == 400, "config: a config that is not a string -> 400")
    finally:
        box.close()


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
        p = box.start(token=TOKEN, RUNNER_TIME_LIMIT_S="3600", RUNNER_IDLE_LIMIT_S="2")
        box.wait_up(p)
        t0 = time.time()
        while time.time() - t0 < 4.5:
            s, _, _ = call(p, "GET", "/jobs", token=TOKEN)
            call(p, "GET", "/jobs", token=WRONG)                  # a refused request does not count
            time.sleep(1)
        check(p.poll() is None, "idle: an authenticated GET /jobs every second keeps it up for 4 s")
        code = wait_exit(p, 12)
        check(code == 3, f"idle: ... and it stops about 2 s after the last request ({code})")
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


def run():
    del RESULTS[:]
    test_bind_and_config()
    test_auth_and_secrets()
    test_no_token_local()
    test_extract_dataset()
    test_upload_http()
    test_inline_config()
    test_time_and_idle_limits()
    print("\n%d/%d passed" % (sum(RESULTS), len(RESULTS)))
    return all(RESULTS)


def test_trainer_runner():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
