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

print("\n%d/%d passed" % (sum(RESULTS), len(RESULTS)))
sys.exit(0 if all(RESULTS) else 1)
