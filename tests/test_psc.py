"""CPU tests for the pixel-studio-comfy CLI (no Docker, no network): python tests/test_psc.py"""
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from psc import __version__, cli, core, local  # noqa: E402

RESULTS = []


def check(cond, msg):
    RESULTS.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + msg)


# ---- parse_env
env = core.parse_env(
    "# comment\r\n\r\nPORT=8189\r\nexport MODELS=a/b,c/d\r\nQ1=\"two words\"\r\nQ2='x y'\r\nEMPTY=\r\n"
    "NOTE=value # trailing\r\nBAD LINE\r\n  SPACED = padded  \r\n")
check(env["PORT"] == "8189" and env["MODELS"] == "a/b,c/d", "parse_env: CRLF, export prefix")
check(env["Q1"] == "two words" and env["Q2"] == "x y", "parse_env: double and single quotes stripped")
check(env["EMPTY"] == "" and "EMPTY" in env, "parse_env: empty value kept as empty string")
check(env["NOTE"] == "value" and env["SPACED"] == "padded", "parse_env: trailing comment, padding")
check("BAD LINE" not in env and "# comment" not in env and len(env) == 7, "parse_env: comments and junk skipped")

# ---- base_url precedence: override > PORT > 8188
check(core.base_url({}) == "http://127.0.0.1:8188", "base_url: default 8188")
check(core.base_url({"PORT": "8189"}) == "http://127.0.0.1:8189", "base_url: PORT from env")
check(core.base_url({"PORT": "8189"}, "http://10.0.0.5:9000/") == "http://10.0.0.5:9000", "base_url: override wins, no trailing slash")
check(core.base_url({"PORT": "abc"}) == "http://127.0.0.1:8188", "base_url: bad PORT falls back")

# ---- dockerfile_base_image on the real Dockerfile
with open(os.path.join(ROOT, "docker", "Dockerfile"), encoding="utf-8") as f:
    dockerfile = f.read()
base = core.dockerfile_base_image(dockerfile)
check(base is not None and base.startswith("nvidia/cuda:"), "dockerfile_base_image: real Dockerfile -> %s" % base)
check(core.dockerfile_base_image("FROM --platform=linux/amd64 python:3.12 AS b\n") == "python:3.12", "dockerfile_base_image: skips flags")
check(core.dockerfile_base_image("RUN true\n") is None, "dockerfile_base_image: none without FROM")
with open(os.path.join(ROOT, "docker-compose.yml"), encoding="utf-8") as f:
    compose_text = f.read()
check(core.compose_default(compose_text, "COMFYUI_REF", "x").startswith("v0."), "compose_default: COMFYUI_REF from the real compose file")

# ---- payloads captured from a live ComfyUI 0.38.2 (trimmed)
QUEUE = {
    "queue_running": [[60, "9f680554-7e2b-466d-a263-36d9669bcbf7", {"1": {"class_type": "UNETLoader"}}, {"client_id": "c"}, ["9"]]],
    "queue_pending": [[61, "aaaaaaaa-0000-0000-0000-000000000001", {}, {}, []], [62, "bbbbbbbb-0000", {}, {}, []]],
}
q = core.summarize_queue(QUEUE)
check((q["running"], q["pending"]) == (1, 2), "summarize_queue: counts")
check(q["running_ids"] == ["9f680554"] and q["pending_ids"] == ["aaaaaaaa", "bbbbbbbb"], "summarize_queue: shortened prompt ids")
many = {"queue_running": [], "queue_pending": [[i, "id%07d" % i] for i in range(9)]}
check(len(core.summarize_queue(many)["pending_ids"]) == 5 and core.summarize_queue(many)["pending"] == 9, "summarize_queue: first 5 ids, true count")
check(core.summarize_queue({}) == {"running": 0, "pending": 0, "running_ids": [], "pending_ids": []}, "summarize_queue: empty payload")

STATS = {
    "system": {"os": "linux", "comfyui_version": "0.38.2", "pytorch_version": "2.11.0+cu128"},
    "devices": [{"name": "cuda:0 NVIDIA GeForce RTX 4090 : cudaMallocAsync", "type": "cuda", "index": 0,
                 "vram_total": 25756565504, "vram_free": 10588102726}],
}
s = core.summarize_system_stats(STATS)
check(s["version"] == "0.38.2" and s["gpu"] == "NVIDIA GeForce RTX 4090", "summarize_system_stats: version, GPU name cleaned")
check(s["vram_total"] == 25756565504 and core.gb(s["vram_free"]) == "9.9 GB", "summarize_system_stats: VRAM")
check(core.summarize_system_stats({"system": {}, "devices": []})["gpu"] is None, "summarize_system_stats: no device")

# ---- parse_models_list
LISTING = """pixel-art/image-to-pixel  (2 models)
  [ok     ] checkpoints/a.safetensors   0.44 GB
  [missing] checkpoints/b.safetensors
pixel-art/sprite-edit  (2 models)
  [partial] loras/c.safetensors
  [ok     ] vae/d.safetensors   0.30 GB
  (no downloadable models listed - fine)
"""
ok, missing, partial, names = core.parse_models_list(LISTING)
check((ok, missing, partial) == (2, 1, 1), "parse_models_list: ok/missing/partial counts")
check(names == ["checkpoints/b.safetensors", "loras/c.safetensors"], "parse_models_list: names of models that are not ok")
check(core.parse_models_list("") == (0, 0, 0, []), "parse_models_list: empty")

# ---- doctor exit code and formatting
C = core.Check
check(core.doctor_exit_code([C("OK", "a", "x"), C("WARN", "b", "y"), C("SKIP", "c", "z")]) == 0, "doctor_exit_code: WARN and SKIP -> 0")
check(core.doctor_exit_code([C("OK", "a", "x"), C("FAIL", "docker daemon", "start Docker Desktop")]) == 1, "doctor_exit_code: FAIL -> 1")
block = core.format_checks([C("OK", "python", "3.11"), C("FAIL", "docker daemon", "start Docker Desktop")])
check(block.splitlines()[1] == "[FAIL] docker daemon: start Docker Desktop" and block.isascii(), "format_checks: one ASCII line per check")

# ---- local helpers
check(local.vram_mib("24564 MiB") == 24564 and local.vram_mib("") is None and local.vram_mib("N/A") is None, "vram_mib")
check(local.docker_ready([C("OK", "docker", ""), C("FAIL", "docker daemon", "")]) is False, "docker_ready: a failed docker check blocks")
check(local.docker_ready([C("OK", "docker", ""), C("OK", "docker compose", ""), C("WARN", ".env", "")]) is True, "docker_ready: other names ignored")

# ---- parser: the five commands, and the global options
parser = cli.build_parser()
for cmd in ("doctor", "status", "up", "down", "logs"):
    check(parser.parse_args([cmd]).command == cmd, "parser: command %s" % cmd)
a = parser.parse_args(["--dir", "x", "--url", "http://h:1", "up", "--wait", "0"])
check((a.dir, a.url, a.wait) == ("x", "http://h:1", 0), "parser: global options before the command, up --wait 0")
check(parser.parse_args(["logs", "-f", "--tail", "5"]).follow and parser.parse_args(["logs"]).tail == 200, "parser: logs -f, --tail default 200")
check(parser.parse_args(["up"]).wait == 120, "parser: up --wait default 120")
check(not hasattr(parser.parse_args(["down"]), "volumes"), "parser: down offers no volumes option")
check(__version__ == "0.1.0", "version string")

# ---- the extension pattern M3 will use: a throwaway group registered through the same mechanism
CALLS = []


def grp_handler(args, stack):
    CALLS.append((args.name, stack))
    return 7


def register_group(sub):
    g = core.group(sub, "grp", help="throwaway")
    p = g.add_parser("sub")
    p.add_argument("name")
    p.set_defaults(func=grp_handler)


def register_group_again(sub):  # a second module adding to the same group
    p = core.group(sub, "grp").add_parser("other")
    p.set_defaults(func=lambda args, stack: 0)


p2 = cli.build_parser([local, types.SimpleNamespace(register=register_group), types.SimpleNamespace(register=register_group_again)])
args = p2.parse_args(["grp", "sub", "hello"])
check(args.func(args, "STACK") == 7 and CALLS == [("hello", "STACK")], "registry: grp sub dispatches to its handler")
check(p2.parse_args(["grp", "other"]).func(None, None) == 0, "registry: two modules share one group")
try:
    p2.parse_args(["grp"])
    check(False, "registry: group without a subcommand is an error")
except SystemExit as e:
    check(e.code == 2, "registry: group without a subcommand is an error")

# ---- 1b: presets, .env and .secrets.env text, update refusal
from psc import presets  # noqa: E402

listed = presets.list_presets(ROOT)
byname = {p.name: p for p in listed}
check([p.name for p in listed][:5] == ["pixel", "svg", "edit", "video", "all"], "list_presets: pixel, svg, edit, video, all in order")
check(all(p.description for p in listed), "list_presets: every description is non-empty")
check(byname["pixel"].description == "pixel-art workflows", "list_presets: description without the Use-with hint")
check(byname["all"].models == ["all"] and byname["all"].count == len(presets.workflow_names(ROOT)) > 5, "list_presets: all = every workflow")
check(byname["video"].count == 2 and byname["edit"].models == ["image-edit/qwen-edit"], "list_presets: counts and models")

with open(os.path.join(ROOT, "env", "pixel.env"), encoding="utf-8") as f:
    pixel_text = f.read()
pm = presets.preset_models(pixel_text)
check(pm[0] == "pixel-art/image-to-pixel" and "pose/extract-pose" in pm and all("/" in m for m in pm), "preset_models: env/pixel.env")
check(presets.preset_models("# MODELS=a\nMODELS=\n") == [] and presets.preset_models("MODELS= a , b \r\n") == ["a", "b"], "preset_models: empty and padded")

rendered = presets.render_env(pixel_text, {"MODELS": "x/y", "PORT": "8189", "BIND_ADDRESS": "127.0.0.1"})
parsed = core.parse_env(rendered)
check(parsed["MODELS"] == "x/y" and parsed["PORT"] == "8189" and parsed["BIND_ADDRESS"] == "127.0.0.1", "render_env: overrides applied")
check([ln for ln in rendered.splitlines() if ln.startswith("MODELS=")] == ["MODELS=x/y"], "render_env: the preset MODELS line is replaced, so MODELS appears once")
check(rendered.startswith("# Preset: pixel-art") and rendered.endswith("\n") and "\r" not in rendered, "render_env: preset text kept, LF only")
check(core.parse_env(presets.render_env(pixel_text, {}))["MODELS"] == pm[0] + "," + ",".join(pm[1:]) and "Set by" not in presets.render_env(pixel_text, {}),
      "render_env: no overrides = the preset unchanged")

SECRETS = "# header\r\nCIVITAI_TOKEN=\r\n# HF_TOKEN=commented\r\nHF_TOKEN=\r\nOTHER=1\r\n"
new, filled = presets.fill_secrets(SECRETS, {"CIVITAI_TOKEN": "abc", "HF_TOKEN": "xyz", "OTHER": "2"})
check(filled == ["CIVITAI_TOKEN", "HF_TOKEN"], "fill_secrets: reports the filled keys")
check(new == "# header\r\nCIVITAI_TOKEN=abc\r\n# HF_TOKEN=commented\r\nHF_TOKEN=xyz\r\nOTHER=1\r\n", "fill_secrets: fills empty, keeps comments, CRLF and other keys")
again, filled2 = presets.fill_secrets(new, {"CIVITAI_TOKEN": "other", "HF_TOKEN": "other"})
check(again == new and filled2 == [], "fill_secrets: a non-empty value is never changed")
same, none = presets.fill_secrets(SECRETS, {"CIVITAI_TOKEN": ""})
check(same == SECRETS and none == [], "fill_secrets: empty value fills nothing")
check(presets.fill_secrets("HF_TOKEN=\"\"\n", {"HF_TOKEN": "t"})[0] == "HF_TOKEN=t\n", "fill_secrets: empty quotes count as empty")

check(presets.update_refusal(" M README.md\n", "origin/main") and "uncommitted" in presets.update_refusal(" M README.md\n", "origin/main"), "update_refusal: dirty")
check("no upstream" in presets.update_refusal("", ""), "update_refusal: no upstream")
check(presets.update_refusal("", "origin/main") is None, "update_refusal: clean with upstream -> None")

p3 = cli.build_parser()
check({"setup", "models", "update"} <= set(p3._subparsers._group_actions[0].choices), "parser: setup, models and update are registered")
check(p3.parse_args(["models", "get", "pixel"]).func is p3.parse_args(["models", "pull", "pixel"]).func, "parser: models get is an alias of models pull")
sa = p3.parse_args(["setup", "--yes", "--no-up", "--port", "8189", "--models", "a/b", "--project-name", "psc-smoke", "--lan"])
check((sa.yes, sa.no_up, sa.port, sa.models, sa.project_name, sa.lan) == (True, True, 8189, "a/b", "psc-smoke", True), "parser: setup options")
check(p3.parse_args(["update", "--no-up"]).no_up is True, "parser: update --no-up")

# ---- 1c: gateway detection, token helpers, doctor gateway checks
from psc import token_cmd  # noqa: E402
from psc.stack import Stack  # noqa: E402

tok = token_cmd.new_token()
check(len(tok) >= 43 and all(ch.isalnum() or ch in "-_" for ch in tok), "new_token: 32 random bytes, URL-safe characters")
check(token_cmd.new_token() != token_cmd.new_token(), "new_token: differs each time")
check(token_cmd.set_secret("A=1\nPSC_TOKEN=\nB=2\n", "PSC_TOKEN", "x") == "A=1\nPSC_TOKEN=x\nB=2\n", "set_secret: replaces an empty value")
check(token_cmd.set_secret("PSC_TOKEN=old\n", "PSC_TOKEN", "new") == "PSC_TOKEN=new\n", "set_secret: replaces a set value (rotate)")
check(token_cmd.set_secret("# PSC_TOKEN=comment\nA=1\n", "PSC_TOKEN", "n") == "# PSC_TOKEN=comment\nA=1\nPSC_TOKEN=n\n", "set_secret: appends when only a comment mentions the key")
check(token_cmd.set_secret("A=1\r\nPSC_TOKEN=\r\n", "PSC_TOKEN", "x") == "A=1\r\nPSC_TOKEN=x\r\n", "set_secret: keeps CRLF")
check(token_cmd.set_secret("A=1", "PSC_TOKEN", "x") == "A=1\nPSC_TOKEN=x\n", "set_secret: file without a final newline")

check(core.gateway_enabled({"PSC_GATEWAY": "1"}) and core.gateway_enabled({"PSC_GATEWAY": "true"}), "gateway_enabled: 1 and true")
check(not core.gateway_enabled({}) and not core.gateway_enabled({"PSC_GATEWAY": "0"}) and not core.gateway_enabled({"PSC_GATEWAY": ""}), "gateway_enabled: off by default, 0, empty")
check(core.compose_files({}) == [] and core.compose_files({"PSC_GATEWAY": "1"}) == ["docker-compose.yml", "docker/gateway.compose.yml"], "compose_files: none without the gateway, two files with it")
check(core.bearer_headers("abc") == {"Authorization": "Bearer abc"} and core.bearer_headers("") == {}, "bearer_headers")


GOOD = "t" * 40


class FakeStack:
    """Just enough of Stack for check_gateway: statuses by (path, authorised)."""

    url = "http://127.0.0.1:8189"
    sends_token = True

    def __init__(self, headers, statuses):
        self.headers, self.statuses = headers, statuses

    def psc_token(self):
        return self.headers.get("Authorization", "")[len("Bearer "):]

    def api_headers(self):
        return self.headers

    def http_status(self, path, headers=None, timeout=5):
        code = self.statuses[(path, bool(headers))]
        if code is None:
            raise OSError("down")
        return code


good = FakeStack({"Authorization": "Bearer " + GOOD}, {("/system_stats", False): 401, ("/manager/queue/status", True): 404})
check(local.check_gateway(good).status == "OK", "doctor gateway: 401 without a token and 404 for /manager -> OK")
open_gw = FakeStack({"Authorization": "Bearer " + GOOD}, {("/system_stats", False): 200, ("/manager/queue/status", True): 404})
check(local.check_gateway(open_gw).status == "FAIL", "doctor gateway: answering without a token -> FAIL")
manager_open = FakeStack({"Authorization": "Bearer " + GOOD}, {("/system_stats", False): 401, ("/manager/queue/status", True): 200})
check(local.check_gateway(manager_open).status == "FAIL", "doctor gateway: /manager reachable -> FAIL")
check(local.check_gateway(FakeStack({}, {})).status == "FAIL", "doctor gateway: empty PSC_TOKEN -> FAIL")
down = FakeStack({"Authorization": "Bearer " + GOOD}, {("/system_stats", False): None, ("/manager/queue/status", True): None})
check(local.check_gateway(down).status == "WARN", "doctor gateway: no answer -> WARN")

import tempfile  # noqa: E402
from psc.stack import CliError  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    st = Stack(tmp)
    check(st.compose_files() == [] and st.api_headers() == {} and not st.gateway, "Stack: gateway off by default, no header")
    with open(os.path.join(tmp, ".env"), "w") as f:
        f.write("PSC_GATEWAY=1\n")
    check(st.compose_files() == ["docker-compose.yml", "docker/gateway.compose.yml"] and st.gateway, "Stack: PSC_GATEWAY=1 adds the override file")
    check(st.api_headers() == {}, "Stack: gateway on but no .secrets.env -> no header")
    with open(os.path.join(tmp, ".secrets.env"), "w") as f:
        f.write("PSC_TOKEN=" + GOOD + "\n")
    check(st.api_headers() == {"Authorization": "Bearer " + GOOD}, "Stack: header from PSC_TOKEN in .secrets.env")
    ensured = token_cmd.ensure_token(st)
    check(ensured == GOOD, "ensure_token: keeps an existing token")
    for bad in ("abc*", "secret123", "x" * 31, "a b" + "c" * 40, "a\"" + "c" * 40, "a}" + "c" * 40, ""):
        check(not core.valid_token(bad), "valid_token: rejects %r" % bad[:12])
    check(core.valid_token(GOOD) and core.valid_token(token_cmd.new_token()) and core.valid_token("A-_" + "z" * 29), "valid_token: accepts generated and 32+ of A-Za-z0-9_-")
    check(not core.valid_token(GOOD + "\n"), "valid_token: no trailing newline")
    with open(os.path.join(tmp, ".secrets.env"), "w") as f:
        f.write("PSC_TOKEN=abc*\n")
    check(st.api_headers() == {}, "Stack: an invalid PSC_TOKEN is never sent")
    try:
        token_cmd.ensure_token(st)
        check(False, "ensure_token: refuses an invalid token")
    except CliError as e:
        check("token rotate" in str(e) and "abc*" not in str(e), "ensure_token: refuses an invalid token, names token rotate, does not print it")
    check(local.check_gateway(st).status == "FAIL" and "rotate" in local.check_gateway(st).detail, "doctor gateway: invalid token -> FAIL")
    with open(os.path.join(tmp, ".secrets.env"), "w") as f:
        f.write("PSC_TOKEN=" + GOOD + "\n")
    for u, sends in (("http://127.0.0.1:8189", True), ("http://localhost:9", True), ("http://other-host:8188", False), ("https://pod.example.com", False)):
        check(Stack(tmp, u).sends_token is sends and (Stack(tmp, u).api_headers() != {}) is sends, "Stack: token %s for --url %s" % ("sent" if sends else "withheld", u))
    check(Stack(tmp, None).sends_token and local.check_gateway(Stack(tmp, "http://other-host:8188")).status == "SKIP", "doctor gateway: SKIP for a non-local --url")
    os.remove(os.path.join(tmp, ".secrets.env"))
    created = token_cmd.ensure_token(st)
    check(len(created) >= 43 and token_cmd.current_token(st) == created, "ensure_token: creates the file and the token")

check(cli.build_parser().parse_args(["token", "show"]).func is token_cmd.cmd_show and cli.build_parser().parse_args(["token", "rotate"]).func is token_cmd.cmd_rotate, "parser: token show and rotate")
check(cli.build_parser().parse_args(["setup", "--gateway"]).gateway is True, "parser: setup --gateway")
with open(os.path.join(ROOT, "docker", "Caddyfile"), encoding="utf-8") as f:
    caddyfile = f.read()
check("{$PSC_TOKEN}" in caddyfile and "respond 401" in caddyfile and "respond 404" in caddyfile and caddyfile.index("handle @authed") < caddyfile.rindex("respond 401"),
      "Caddyfile: token placeholder, 404 inside the authed block, 401 last")
with open(os.path.join(ROOT, "docker", "gateway.compose.yml"), encoding="utf-8") as f:
    override = f.read()
check("!reset" in override and ":latest" not in override and "caddy:2." in override, "gateway.compose.yml: ports reset, pinned caddy tag")

# The allow list is the security contract of the gateway: exactly these paths, each also under /api.
ALLOWED = ["/prompt", "/history", "/history/*", "/view", "/upload/image", "/object_info", "/object_info/*", "/ws", "/queue",
           "/interrupt", "/system_stats", "/pixelstudio/*"]
allowed_lines = [ln.split() for ln in caddyfile.splitlines() if ln.strip().startswith("@allowed")]
listed = allowed_lines[0][2:] if len(allowed_lines) == 1 and allowed_lines[0][:2] == ["@allowed", "path"] else None
check(listed is not None and sorted(listed) == sorted(ALLOWED + ["/api" + p for p in ALLOWED]), "Caddyfile: @allowed is exactly the 12 paths and their /api twins")
check(listed is not None and not any(p.rstrip("/*") in ("", "/manager", "/userdata", "/settings", "/internal", "/extensions", "/upload/mask", "/api")
                                     or p.startswith(("/manager", "/userdata", "/settings", "/internal", "/api/manager", "/api/userdata")) or p in ("/*", "/api/*")
                                     for p in listed), "Caddyfile: no /, /manager, /userdata, /settings or /internal in the allow list")
check(len(allowed_lines) == 1 and caddyfile.count("reverse_proxy") == 1, "Caddyfile: one allow-list matcher, one proxy")
check("case \"$$PSC_TOKEN\" in *[!A-Za-z0-9_-]*)" in override and 'if [ -z "$$PSC_TOKEN" ]' in override, "gateway.compose.yml: guard rejects an empty or out-of-charset token")

# ---- gateway review: token rotate only recreates a running gateway; setup --gateway writes COMPOSE_FILE
class RotateStack:
    def __init__(self, root, state):
        self.inner, self.state, self.calls, self.gateway = Stack(root), state, [], True

    def read_text(self, *a):
        return self.inner.read_text(*a)

    def write_text(self, *a, **k):
        return self.inner.write_text(*a, **k)

    def service_state(self, service="comfyui"):
        return self.state

    def compose(self, *args, **k):
        self.calls.append(args)
        return types.SimpleNamespace(returncode=0)


import contextlib  # noqa: E402
import io  # noqa: E402

for state, expect in (("running", 1), (None, 0), ("exited", 0)):
    with tempfile.TemporaryDirectory() as rtmp:
        rs = RotateStack(rtmp, state)
        with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
            token_cmd.cmd_rotate(None, rs)
        saved = rs.inner.read_text(".secrets.env")
        check(len(rs.calls) == expect and "PSC_TOKEN=" + out.getvalue().strip() in saved, "token rotate: gateway %s -> %d recreate call(s), token saved" % (state, expect))

# ---- 1b review: private writes, and setup never overwriting (temp folder, step_checks not called)
import shutil  # noqa: E402
import stat  # noqa: E402
import tempfile  # noqa: E402
from psc import setup as psc_setup  # noqa: E402
from psc.stack import Stack  # noqa: E402

tmp = tempfile.mkdtemp(prefix="psc-test-")
saved_env = {k: os.environ.pop(k, None) for k in ("CIVITAI_TOKEN", "HF_TOKEN")}
try:
    st = Stack(tmp)
    st.write_text("a.txt", "x\ny\n", private=True)
    with open(os.path.join(tmp, "a.txt"), "rb") as f:
        check(f.read() == b"x\ny\n", "write_text private: LF endings kept")
    if os.name != "nt":
        check(stat.S_IMODE(os.stat(os.path.join(tmp, "a.txt")).st_mode) == 0o600, "write_text private: mode 0600")
        os.chmod(os.path.join(tmp, "a.txt"), 0o644)
        st.write_text("a.txt", "z\n", private=True)
        check(stat.S_IMODE(os.stat(os.path.join(tmp, "a.txt")).st_mode) == 0o600, "write_text private: an existing file is tightened")
    shutil.copytree(os.path.join(ROOT, "env"), os.path.join(tmp, "env"))
    shutil.copytree(os.path.join(ROOT, "workflows"), os.path.join(tmp, "workflows"))
    shutil.copy(os.path.join(ROOT, ".secrets.env.example"), os.path.join(tmp, ".secrets.env.example"))
    sargs = types.SimpleNamespace(port=None, lan=False, project_name=None, output_dir=None, models=None, preset="pixel",
                                  civitai_token=None, hf_token=None, gateway=False)
    psc_setup.say = lambda text="": None
    psc_setup.step_env(sargs, st, False, None)
    psc_setup.step_secrets(sargs, st, False)
    first = (st.read_text(".env"), st.read_text(".secrets.env"))
    check(first[0] and "BIND_ADDRESS=127.0.0.1" in first[0] and first[1] is not None, "setup: first run writes .env and .secrets.env")
    check("COMPOSE_FILE" not in first[0], "setup: no COMPOSE_FILE without --gateway")
    gtmp = tempfile.mkdtemp(prefix="psc-test-gw-")
    try:
        for name in ("env", "workflows"):
            shutil.copytree(os.path.join(ROOT, name), os.path.join(gtmp, name))
        gst = Stack(gtmp)
        psc_setup.step_env(types.SimpleNamespace(**dict(vars(sargs), gateway=True)), gst, False, None)
        genv = core.parse_env(gst.read_text(".env"))
        check(genv["PSC_GATEWAY"] == "1" and genv["COMPOSE_FILE"] == "docker-compose.yml;docker/gateway.compose.yml" and genv["COMPOSE_PATH_SEPARATOR"] == ";",
              "setup --gateway: writes PSC_GATEWAY, COMPOSE_FILE and COMPOSE_PATH_SEPARATOR")
    finally:
        shutil.rmtree(gtmp, ignore_errors=True)
    sargs2 = types.SimpleNamespace(**dict(vars(sargs), port=9999, lan=True, gateway=True))
    psc_setup.step_env(sargs2, st, False, None)
    psc_setup.step_secrets(sargs2, st, False)
    check((st.read_text(".env"), st.read_text(".secrets.env")) == first, "setup: a second run with other flags leaves both files byte-identical")
    st.write_text(".secrets.env", "CIVITAI_TOKEN=mine\nHF_TOKEN=\n", private=True)
    os.environ["HF_TOKEN"] = "fromenv"
    psc_setup.step_secrets(sargs, st, False)
    check(st.read_text(".secrets.env") == "CIVITAI_TOKEN=mine\nHF_TOKEN=fromenv\n", "setup: an empty token is filled from the environment, a set one survives")
finally:
    for k, v in saved_env.items():
        os.environ.pop(k, None)
        if v is not None:
            os.environ[k] = v
    shutil.rmtree(tmp, ignore_errors=True)

print("\n%d/%d passed" % (sum(RESULTS), len(RESULTS)))
sys.exit(0 if all(RESULTS) else 1)
