"""Tests for .claude/skills/pixel-art/scripts/studio.py against a fake Pixel Studio (no server, no GPU needed).

Run from the repo root:  python tests/test_studio_helper.py
"""
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDIO = os.path.join(ROOT, ".claude", "skills", "pixel-art", "scripts", "studio.py")
PNG = b"\x89PNG\r\n\x1a\n" + b"fakepixels" * 4

RESULTS = []
SEEN = {"imports": [], "origins": [], "waits": 0}


def check(cond, msg):
    RESULTS.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + msg)


def parse_multipart(ctype, body):
    boundary = re.search(r"boundary=(\S+)", ctype).group(1).encode()
    parts = {}
    for chunk in body.split(b"--" + boundary):
        chunk = chunk.strip(b"\r\n")
        if not chunk or chunk == b"--":
            continue
        head, _, data = chunk.partition(b"\r\n\r\n")
        name = re.search(rb'name="([^"]+)"', head).group(1).decode()
        fn = re.search(rb'filename="([^"]*)"', head)
        parts[name] = (fn.group(1).decode() if fn else None, data)
    return parts


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, obj=None, raw=None):
        body = raw if raw is not None else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlsplit(self.path)
        if u.path == "/api/status":
            return self.send(200, {"run_on": "auto", "sources": [{"name": "pc", "label": "PC", "state": "ready", "url": "x"},
                                                                  {"name": "cloud", "label": "Cloud", "state": "asleep"}]})
        if u.path == "/api/resolve":
            p = parse_qs(u.query)["path"][0]
            if p == "generate/3":   # like the real store.resolve: only generate/<number>, not generate/characters
                return self.send(200, {"subject": {"id": "subj1"}, "asset": None})
            if p == "asset9":
                return self.send(200, {"subject": {"id": "subj1"}, "asset": {"id": "asset9", "ref": "character/1/image/1"}})
            return self.send(404, {"detail": "nothing at " + p})
        if u.path == "/files/sprite.png":
            return self.send(200, raw=PNG)
        self.send(404, {})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n)
        u = urlsplit(self.path)
        SEEN["origins"].append(self.headers.get("Origin"))
        if u.path == "/api/subjects/subj1/import":
            SEEN["imports"].append(parse_multipart(self.headers["Content-Type"], body))
            k = SEEN["imports"][-1].get("kind", (0, b""))[1]
            return self.send(200, {"id": "asset9", "kind": "reference" if k == b"auto" else "image"})
        if u.path == "/mcp":
            m = json.loads(body)
            name, args = m["params"]["name"], m["params"]["arguments"]
            if name == "boom":
                res = {"content": [{"type": "text", "text": "unknown argument source"}], "isError": True}
            elif name == "get" and args["ref"] == "generate/characters":
                out = {"ref": "generate/3", "name": "Characters", "kind": "playground"}
                res = {"content": [{"type": "text", "text": json.dumps(out)}], "structuredContent": out}
            elif name == "get":
                out = {"ref": args["ref"], "files": {"sprite.png": "http://192.0.2.1:9/files/sprite.png"}}
                res = {"content": [{"type": "text", "text": json.dumps(out)}], "structuredContent": out}
            elif name == "wait_for_job":
                SEEN["waits"] += 1
                out = {"job_id": args["job_id"], "status": "running" if SEEN["waits"] < 2 else "done"}
                res = {"content": [{"type": "text", "text": json.dumps(out)}], "structuredContent": out}
            else:
                out = {"tool": name, "args": args}
                res = {"content": [{"type": "text", "text": json.dumps(out)}], "structuredContent": out}
            return self.send(200, {"jsonrpc": "2.0", "id": m["id"], "result": res})
        self.send(404, {})


def run(*argv, env_url=None):
    env = {**os.environ, "PIXEL_STUDIO_URL": env_url or URL}
    p = subprocess.run([sys.executable, STUDIO, *argv], capture_output=True, text=True, env=env, timeout=60)
    return p.returncode, p.stdout, p.stderr


srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
URL = "http://127.0.0.1:%d" % srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()

# status
rc, out, _ = run("status")
j = json.loads(out)
check(rc == 0 and j["ok"] and j["url"] == URL and j["run_on"] == "auto"
      and j["sources"] == [{"name": "pc", "label": "PC", "state": "ready"}, {"name": "cloud", "label": "Cloud", "state": "asleep"}],
      "status: ok, url, run_on and sources (name, label, state only)")

# status against a closed port: exit 2 and ok:false, within about 6 s
s = socket.socket()
s.bind(("127.0.0.1", 0))
dead = "http://127.0.0.1:%d" % s.getsockname()[1]
s.close()
t = time.time()
rc, out, _ = run("status", env_url=dead)
j = json.loads(out)
check(rc == 2 and j["ok"] is False and j["url"] == dead and j["error"] and time.time() - t < 6,
      "status on a closed port: exit 2, ok:false, error, fast")

# call: values parsed as JSON, plain strings kept, --json merged, no Origin header
rc, out, _ = run("call", "generate_sprite", "--arg", "prompt=a knight", "--arg", "pixels=48", "--arg", "variants=[1,2]",
                 "--arg", "remove_background=true", "--json", '{"target": "generate/characters"}')
a = json.loads(out)["args"]
check(rc == 0 and a == {"target": "generate/characters", "prompt": "a knight", "pixels": 48, "variants": [1, 2],
                        "remove_background": True}, "call: --arg parsed as JSON where possible, --json merged")
check(SEEN["origins"] and all(o is None for o in SEEN["origins"]), "call: no Origin header sent")

# call: isError -> exit 1 with the message
rc, out, err = run("call", "boom")
check(rc == 1 and "unknown argument source" in err, "call: isError exits 1 and prints the message")

# import
png = os.path.join(tempfile.mkdtemp(), "art.png")
with open(png, "wb") as f:
    f.write(PNG)
rc, out, _ = run("import", "generate/characters", png)
j = json.loads(out)
parts = SEEN["imports"][0] if SEEN["imports"] else {}
check(rc == 0 and j == {"id": "asset9", "kind": "image", "ref": "character/1/image/1"}, "import: prints id, kind and ref")
check(parts.get("file") == ("art.png", PNG) and parts.get("kind", (0, b""))[1] == b"image",
      "import: well-formed multipart with file (name and bytes) and kind=image by default")
gif = os.path.join(tempfile.mkdtemp(), "anim.gif")
with open(gif, "wb") as f:
    f.write(b"GIF89a")
rc, out, _ = run("import", "generate/characters", gif)
j = json.loads(out)
check(rc == 0 and SEEN["imports"][-1]["kind"][1] == b"auto" and j["kind"] == "reference" and "reference" in j["note"],
      "import: .gif goes as kind=auto and the note names the kind the server chose")
rc, out, err = run("import", "nowhere/at/all", png)
check(rc == 1 and "404" in err, "import: an unknown subject fails with exit 1")

# download: re-bases the file URL onto our own server
dest = tempfile.mkdtemp()
rc, out, _ = run("download", "character/1/image/1", "--out", dest)
j = json.loads(out)
check(rc == 0 and len(j["saved"]) == 1 and open(os.path.join(dest, "sprite.png"), "rb").read() == PNG,
      "download: fetches each file into DIR (via our base URL, not the one the server reports)")

# wait: loops until the job is over
rc, out, _ = run("wait", "job1", "--timeout", "60")
check(rc == 0 and json.loads(out)["status"] == "done" and SEEN["waits"] == 2, "wait: loops wait_for_job until done")

# the default URL is localhost:8190 and nothing else is hard-coded
src = open(STUDIO).read()
check(src.count("localhost:8190") == 1 and not re.search(r"\b(?:10|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d+\.\d+", src),
      "studio.py: localhost:8190 only as the default, no private addresses")

print("\n%d/%d passed" % (sum(RESULTS), len(RESULTS)))
sys.exit(0 if all(RESULTS) else 1)
