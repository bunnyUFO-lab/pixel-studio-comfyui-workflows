"""CPU tests for the trainer image's helpers: `pixel-studio-comfy cloud trainer build --dry-run` and
trainer/cloud/shutdown.py against a fake RunPod on 127.0.0.1 (no network beyond that, no keys, no docker).

    python tests/test_trainer_cloud.py        (also collected by pytest as test_trainer_cloud)
"""
import contextlib
import http.server
import io
import os
import re
import subprocess
import sys
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from psc import cli  # noqa: E402

SHUTDOWN = os.path.join(ROOT, "trainer", "cloud", "shutdown.py")
KEY = "rpa_FAKEKEYFORTESTS0123456789abcdef"
POD = "abc123def456"
RESULTS = []


def check(cond, msg):
    RESULTS.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + msg)


def main_capture(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


class FakeRunPod(http.server.BaseHTTPRequestHandler):
    seen = []
    status = 200

    def log_message(self, *a):
        pass

    def handle_any(self):
        FakeRunPod.seen.append((self.command, self.path, self.headers.get("Authorization")))
        body = b'{"secret": "response body that must never be printed"}'
        self.send_response(FakeRunPod.status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_DELETE = do_POST = do_GET = handle_any


def run_shutdown(base, **env):
    full = {k: v for k, v in os.environ.items() if not k.startswith(("RUNPOD_", "RUNNER_"))}
    full.update(RUNNER_SHUTDOWN_RETRY_S="0", RUNPOD_REST_BASE=base)
    full.update({k: v for k, v in env.items() if v is not None})
    p = subprocess.run([sys.executable, SHUTDOWN], env=full, capture_output=True, text=True, timeout=60)
    return p.returncode, p.stdout + p.stderr


def run():
    del RESULTS[:]
    # ---- cloud trainer build --dry-run
    code, out, err = main_capture(["cloud", "trainer", "build", "--dry-run"])
    lines = out.strip().splitlines()
    m = re.fullmatch(r"docker build -f trainer/cloud/Dockerfile --build-arg BASE=ostris/aitoolkit:0\.13\.23 "
                     r"-t ghcr\.io/bunnyufo-lab/psc-trainer:aitk-0\.13\.23-([0-9a-f]{4,40}|dev) "
                     r"-t ghcr\.io/bunnyufo-lab/psc-trainer:latest \.", lines[0] if lines else "")
    check(code == 0 and m and len(lines) == 1, f"build --dry-run prints the build line with both tags and nothing else ({lines})")
    check("Dry run" in err, "build --dry-run says nothing was built (stderr)")
    code, out, err = main_capture(["cloud", "trainer", "build", "--dry-run", "--push"])
    lines = out.strip().splitlines()
    check(code == 0 and len(lines) == 3 and lines[1].startswith("docker push ghcr.io/bunnyufo-lab/psc-trainer:aitk-0.13.23-")
          and lines[2] == "docker push ghcr.io/bunnyufo-lab/psc-trainer:latest", f"--push adds one docker push per tag ({lines[1:]})")
    code, out, err = main_capture(["cloud", "trainer", "build", "--dry-run", "--base", "python:3.12-slim", "--tag", "psc-trainer:smoke"])
    check(out.strip() == "docker build -f trainer/cloud/Dockerfile --build-arg BASE=python:3.12-slim -t psc-trainer:smoke .",
          f"--base and --tag are used ({out.strip()})")
    check(cli.cloud_trainer in cli.COMMAND_MODULES, "the module is registered in COMMAND_MODULES")

    # ---- files of the image
    for rel in ("Dockerfile", "Dockerfile.dockerignore", "entrypoint.sh", "shutdown.py"):
        check(os.path.isfile(os.path.join(ROOT, "trainer", "cloud", rel)), f"trainer/cloud/{rel} exists")
    with open(os.path.join(ROOT, "trainer", "cloud", "entrypoint.sh"), "rb") as f:
        sh = f.read()
    check(b"\r" not in sh, "entrypoint.sh has LF line endings")
    with open(os.path.join(ROOT, "trainer", "cloud", "Dockerfile"), encoding="utf-8") as f:
        dockerfile = f.read()
    check("RUNNER_TOKEN" not in dockerfile and "RUNPOD_API_KEY" not in dockerfile and "HF_WRITE_TOKEN" not in dockerfile,
          "the Dockerfile sets no token and no key")

    # ---- shutdown.py
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeRunPod)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = "http://127.0.0.1:%d/v1" % srv.server_address[1]
    try:
        env = {"RUNPOD_POD_ID": POD, "RUNPOD_API_KEY": KEY}
        FakeRunPod.seen.clear()
        code, text = run_shutdown(base, RUNNER_ON_EXIT="runpod-terminate", **env)
        check(code == 0 and FakeRunPod.seen == [("DELETE", f"/v1/pods/{POD}", f"Bearer {KEY}")], f"terminate sends DELETE /pods/<id> with the Bearer header ({FakeRunPod.seen})")
        check(KEY not in text and "response body" not in text and "terminate" in text and "200" in text, f"terminate prints the mode and the status only ({text.strip()})")
        FakeRunPod.seen.clear()
        code, text = run_shutdown(base, RUNNER_ON_EXIT="runpod-stop", **env)
        check(code == 0 and FakeRunPod.seen == [("POST", f"/v1/pods/{POD}/stop", f"Bearer {KEY}")], f"stop sends POST /pods/<id>/stop ({FakeRunPod.seen})")
        FakeRunPod.seen.clear()
        code, text = run_shutdown(base, **env)
        check(code == 0 and not FakeRunPod.seen, "mode none (the default) sends nothing")
        code, text = run_shutdown(base, RUNNER_ON_EXIT="none", **env)
        check(code == 0 and not FakeRunPod.seen, "RUNNER_ON_EXIT=none sends nothing")
        code, text = run_shutdown(base, RUNNER_ON_EXIT="runpod-terminate", RUNPOD_POD_ID=POD)
        check(code == 1 and "RUNPOD_API_KEY" in text and not FakeRunPod.seen, f"a missing RUNPOD_API_KEY exits 1 and names it ({text.strip()})")
        code, text = run_shutdown(base, RUNNER_ON_EXIT="runpod-terminate", RUNPOD_API_KEY=KEY)
        check(code == 1 and "RUNPOD_POD_ID" in text and KEY not in text and not FakeRunPod.seen, f"a missing RUNPOD_POD_ID exits 1 and names it ({text.strip()})")
        code, text = run_shutdown(base, RUNNER_ON_EXIT="runpod-terminate", RUNPOD_POD_ID="../x", RUNPOD_API_KEY=KEY)
        check(code == 1 and not FakeRunPod.seen, "a pod id with odd characters is refused before any request")
        code, text = run_shutdown("http://example.com/v1", RUNNER_ON_EXIT="runpod-terminate", **env)
        check(code == 1 and "https://" in text and KEY not in text, "a plain http base that is not 127.0.0.1 is refused")
        code, text = run_shutdown(base, RUNNER_ON_EXIT="runpod-bogus", **env)
        check(code == 1 and not FakeRunPod.seen, "an unknown mode exits 1")
        FakeRunPod.status = 500
        FakeRunPod.seen.clear()
        code, text = run_shutdown(base, RUNNER_ON_EXIT="runpod-terminate", **env)
        check(code == 1 and len(FakeRunPod.seen) == 3 and KEY not in text and "response body" not in text,
              f"three tries on HTTP 500, then exit 1 ({len(FakeRunPod.seen)} requests)")
        FakeRunPod.status = 200
    finally:
        srv.shutdown()
        srv.server_close()
    print("\n%d/%d passed" % (sum(RESULTS), len(RESULTS)))
    return all(RESULTS)


def test_trainer_cloud():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
