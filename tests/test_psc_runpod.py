"""CPU tests for `pixel-studio-comfy cloud runpod` (no network, no Docker, no RunPod): python tests/test_psc_runpod.py"""
import contextlib
import io
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from psc import cli, cloud_runpod as cr  # noqa: E402

RESULTS = []


def check(cond, msg):
    RESULTS.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + msg)


def main_capture(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = cli.main(argv)
        except SystemExit as e:
            code = e.code
    return code, out.getvalue(), err.getvalue()


class Calls:
    """Replaces cr.run: records the argument lists and answers like git, docker image inspect, build and push."""

    def __init__(self, push_output="", push_code=0, have_image=True):
        self.calls, self.push_output, self.push_code, self.have_image = [], push_output, push_code, have_image

    def __call__(self, args, cwd=None, capture=False, timeout=None):
        self.calls.append(list(args))
        code, out = 0, ""
        if args[:2] == ["git", "rev-parse"]:
            out = "abc1234\n"
        elif args[:2] == ["docker", "push"]:
            code, out = self.push_code, self.push_output
        return subprocess.CompletedProcess(args, code, stdout=out)


def patched(calls):
    import psc.local as local
    saved = cr.run, local.run
    cr.run = local.run = calls
    return saved


def restore(saved):
    import psc.local as local
    cr.run, local.run = saved


# ---- parser
p = cli.build_parser().parse_args(["cloud", "runpod", "build", "--dry-run", "--push", "--tag", "x:y", "--base", "b:1"])
check(p.func is cr.cmd_build and p.dry_run and p.push and (p.tag, p.base) == ("x:y", "b:1"), "parser: cloud runpod build flags")
p = cli.build_parser().parse_args(["cloud", "runpod", "snippet", "--endpoint", "abc123"])
check(p.func is cr.cmd_snippet and p.name == "runpod" and p.label == "Cloud (RunPod)" and p.concurrency == 3 and p.gpu == "RTX-4090",
      "parser: cloud runpod snippet defaults")
check(cli.build_parser().parse_args(["cloud", "runpod", "settings"]).func is cr.cmd_settings, "parser: cloud runpod settings")
code, out, err = main_capture(["cloud", "runpod", "snippet"])
check(code == 2 and "--endpoint" in err, "parser: snippet without --endpoint exits 2")
code, out, err = main_capture(["cloud", "modal", "snippet", "--help"])
check(code == 0 and "usage:" in out, "parser: the modal group still works next to runpod")

# ---- snippet
with open(os.path.join(ROOT, "cloud", "runpod", "gpu_rates.json"), encoding="utf-8") as fh:
    rates = json.load(fh)
want = {"name": "runpod", "kind": "runpod", "label": "Cloud (RunPod)", "url": "https://api.runpod.ai/v2/abc123", "probe": 0,
        "concurrency": 3, "cost_per_hour": 1.1, "headers": {"Authorization": "$RUNPOD_API_KEY"}}
check(cr.snippet("abc123", rates) == want, "snippet: exact COMFY_SOURCES entry (kind runpod, URL, header reference, cost 1.10)")
check(cr.snippet("https://api.runpod.ai/v2/abc123/run", rates) == want, "snippet: a full endpoint URL is accepted")
check(cr.snippet("abc123", rates, "r2", "Two", 2, "l40s")["cost_per_hour"] == 1.75, "snippet: --gpu l40s -> 1.75, case-insensitive")
check(cr.snippet("abc123", rates, gpu="Mystery")["cost_per_hour"] == 0, "snippet: an unknown GPU -> 0")
code, out, err = main_capture(["--dir", ROOT, "cloud", "runpod", "snippet", "--endpoint", "abc123"])
check(code == 0 and json.loads(out) == want, "snippet: stdout is only the JSON")
check("Execution timeout    3900" in err and "RUNPOD_API_KEY" in err, "snippet: the checklist and the key note go to stderr")
code, out, err = main_capture(["--dir", ROOT, "cloud", "runpod", "snippet", "--endpoint", "../x"])
check(code == 1 and "endpoint" in err and out == "", "snippet: a bad endpoint id -> plain error")
code, out, err = main_capture(["cloud", "runpod", "settings"])
check(code == 0 and "3900 s" in out and "300 s" in out and "200 GB" in out and "RTX 4090" in out, "settings: the checklist")
check("sk-" not in out and "rpa_" not in out, "settings: no key value")

# ---- build --dry-run (git is mocked: the sha is abc1234)
calls = Calls()
saved = patched(calls)
try:
    code, out, err = main_capture(["--dir", ROOT, "cloud", "runpod", "build", "--dry-run", "--base", "comfyui-local:v0.38.2"])
finally:
    restore(saved)
tag = "ghcr.io/bunnyufo-lab/psc-comfy-runpod"
check(code == 0 and out.strip() == f"docker build -f cloud/runpod/Dockerfile --build-arg BASE=comfyui-local:v0.38.2 "
                                    f"-t {tag}:v0.38.2-abc1234 -t {tag}:latest .", f"build --dry-run: the docker build line, repo root context ({out.strip()})")
check(not any(c[:2] in (["docker", "build"], ["docker", "push"]) for c in calls.calls), "build --dry-run: nothing is built")
saved = patched(Calls())
try:
    code, out, err = main_capture(["--dir", ROOT, "cloud", "runpod", "build", "--dry-run", "--push", "--tag", "psc-comfy-runpod:dev", "--base", "b:1"])
finally:
    restore(saved)
lines = out.strip().splitlines()
check(lines == ["docker build -f cloud/runpod/Dockerfile --build-arg BASE=b:1 -t psc-comfy-runpod:dev .", "docker push psc-comfy-runpod:dev"],
      f"build --dry-run --push --tag: one tag, build then push ({lines})")
saved = patched(Calls())
try:
    code, out, err = main_capture(["--dir", ROOT, "cloud", "runpod", "build", "--dry-run"])
finally:
    restore(saved)
check(code == 0 and "--build-arg BASE=comfyui-local:" in out, "build --dry-run: the base defaults to the stack's comfyui-local image")

# ---- build --push without login (docker mocked)
calls = Calls(push_output="denied: permission_denied: The token provided does not match expected scopes.", push_code=1)
saved = patched(calls)
try:
    code, out, err = main_capture(["--dir", ROOT, "cloud", "runpod", "build", "--push", "--base", "comfyui-local:v0.38.2"])
finally:
    restore(saved)
check(code == 1 and "docker login ghcr.io" in err and "Traceback" not in err, f"build --push without login: plain message ({err.strip().splitlines()[-1] if err.strip() else ''})")
check([c[:2] for c in calls.calls].count(["docker", "build"]) == 1 and ["docker", "push", f"{tag}:v0.38.2-abc1234"] in calls.calls
      and ["docker", "push", f"{tag}:latest"] not in calls.calls, "build --push: the build ran, the first push stopped it")

# ---- build without the base image
calls = Calls()
orig = cr.image_exists
cr.image_exists = lambda ref: False
saved = patched(calls)
try:
    code, out, err = main_capture(["--dir", ROOT, "cloud", "runpod", "build", "--base", "comfyui-local:v0.38.2"])
finally:
    restore(saved)
    cr.image_exists = orig
check(code == 1 and "docker compose build comfyui" in err and not any(c[:2] == ["docker", "build"] for c in calls.calls),
      "build: no base image -> says to build it first, builds nothing")

print(f"\n{sum(RESULTS)}/{len(RESULTS)} passed")
sys.exit(0 if all(RESULTS) else 1)
