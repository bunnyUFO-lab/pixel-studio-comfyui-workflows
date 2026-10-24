"""CPU tests for `pixel-studio-comfy cloud modal` (no network, no Modal): python tests/test_psc_cloud.py"""
import contextlib
import io
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from psc import cli, cloud_modal as cm  # noqa: E402
from psc.stack import CliError  # noqa: E402

RESULTS = []


def check(cond, msg):
    RESULTS.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + msg)


def raises_cli(fn):
    try:
        fn()
    except CliError as e:
        return str(e)
    return None


def main_capture(argv, env=None):
    """cli.main(argv) with stdout and stderr captured and os.environ patched: (exit code, stdout, stderr)."""
    saved = dict(os.environ)
    out, err = io.StringIO(), io.StringIO()
    try:
        for k, v in (env or {}).items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = cli.main(argv)
            except SystemExit as e:
                code = e.code
    finally:
        os.environ.clear()
        os.environ.update(saved)
    return code, out.getvalue(), err.getvalue()


# The output of `modal deploy cloud/modal_comfy.py` (Modal client 1.6.1, 24 Oct 2026, upload lines trimmed).
DEPLOY_OUTPUT = (
    '├── ✓ Created Function jobs\n'
    '│   └── https://danieldlgrzz--sprite-studio-comfy-m3-jobs.us-east.modal.run \U0001f511\n'
    '├── ✓ Created Function download.\n'
    '├── ✓ Created Function list_models.\n'
    '├── ✓ Created Function comfyui\n'
    '│   └── https://danieldlgrzz--sprite-studio-comfy-m3-comfyui.us-east.modal.run\n'
    '│       \U0001f511\n'
    '├── ✓ Created Function ComfyJob.*.\n'
    '└── ✓ Created Function refresh_info.\n'
    '✓ App deployed in 5.499s! \U0001f389\n'
    '\n'
    'View Deployment:\n'
    'https://modal.com/apps/danieldlgrzz/main/deployed/sprite-studio-comfy-m3\n'
)

# ---- 1. parser
args = cli.build_parser().parse_args(["cloud", "modal", "deploy", "--dry-run", "--gpu", "L40S", "--max-containers", "2"])
check(args.func is cm.cmd_deploy and args.dry_run and args.gpu == "L40S" and args.max_containers == 2, "parser: cloud modal deploy flags")
args = cli.build_parser().parse_args(["cloud", "modal", "snippet", "--name", "x", "--label", "Y"])
check(args.func is cm.cmd_snippet and (args.name, args.label) == ("x", "Y"), "parser: cloud modal snippet")
for argv in (["cloud", "modal", "deploy", "--help"], ["cloud", "modal", "snippet", "--help"]):
    code, out, err = main_capture(argv)
    check(code == 0 and "usage:" in out, "parser: %s prints usage, exit 0" % " ".join(argv[1:]))
for argv in (["cloud"], ["cloud", "modal"]):
    code, out, err = main_capture(argv)
    check(code == 2 and "usage:" in err, "parser: %s alone exits 2 with usage" % " ".join(argv))

# ---- 2. merge_env_text
src = "# my settings\r\nMODAL_GPU=L40S\r\nOTHER=keep me # note\r\n\r\n# tail comment\r\n"
merged = cm.merge_env_text(src, {"MODAL_GPU": "H100", "MODAL_IDLE_SECONDS": "60"})
check("\r" not in merged and merged.endswith("\n"), "merge_env_text: CRLF in, LF out")
check(merged.splitlines()[:5] == ["# my settings", "MODAL_GPU=H100", "OTHER=keep me # note", "", "# tail comment"],
      "merge_env_text: comments, unknown key and order kept, MODAL_GPU rewritten in place")
check(merged.splitlines()[-1] == "MODAL_IDLE_SECONDS=60", "merge_env_text: a new key is appended")
check(cm.merge_env_text("# MODAL_GPU=old\nMODAL_GPU=a\n", {"MODAL_GPU": "b"}) == "# MODAL_GPU=old\nMODAL_GPU=b\n",
      "merge_env_text: a commented-out key is not touched")
check(cm.merge_env_text("", {"A": "1"}) == "A=1\n" and cm.merge_env_text("A=1", {}) == "A=1\n", "merge_env_text: empty and no-op")

# ---- 3. effective_config
check(cm.effective_config({}, {}) == cm.DEFAULTS, "effective_config: defaults")
keys = {"app": ("MODAL_APP", "a-file", "a-env"), "gpu": ("MODAL_GPU", "L40S", "H100"),
        "max_containers": ("MODAL_MAX_CONTAINERS", "2", "4"), "idle": ("MODAL_IDLE_SECONDS", "90", "30"),
        "preset": ("MODAL_PRESET", "starter", "all"), "job_timeout": ("MODAL_JOB_TIMEOUT", "100", "200")}
for key, (name, from_file, from_env) in keys.items():
    is_int = isinstance(cm.DEFAULTS[key], int)
    want_file = int(from_file) if is_int else from_file
    want_env = int(from_env) if is_int else from_env
    check(cm.effective_config({name: from_file}, {})[key] == want_file
          and cm.effective_config({name: from_file}, {name: from_env})[key] == want_env
          and cm.effective_config({}, {})[key] == cm.DEFAULTS[key], "effective_config: %s env > file > default" % name)
for bad in ("0", "abc", "33", "-1"):
    err = raises_cli(lambda bad=bad: cm.effective_config({"MODAL_MAX_CONTAINERS": bad}, {}))
    err2 = raises_cli(lambda bad=bad: cm.effective_config({}, {"MODAL_MAX_CONTAINERS": bad}))
    check(err is not None and "MODAL_MAX_CONTAINERS" in err and err2 is not None, "effective_config: MODAL_MAX_CONTAINERS=%r -> CliError" % bad)
check(raises_cli(lambda: cm.effective_config({"MODAL_MAX_CONTAINERS": ""}, {})) is None, "effective_config: an empty value means the default")
check(raises_cli(lambda: cm.effective_config({}, {"MODAL_JOB_TIMEOUT": "3601"})) is not None, "effective_config: job timeout above 3600 -> CliError")

# ---- 4. find_urls on the real deploy output
jobs, comfy = cm.find_urls(DEPLOY_OUTPUT)
check(jobs == "https://danieldlgrzz--sprite-studio-comfy-m3-jobs.us-east.modal.run", "find_urls: jobs URL from the real deploy output")
check(comfy == "https://danieldlgrzz--sprite-studio-comfy-m3-comfyui.us-east.modal.run", "find_urls: comfyui URL (the key emoji on the next line does not matter)")
check(cm.find_urls("\x1b[32mhttps://w--app-jobs.modal.run\x1b[0m done") == ("https://w--app-jobs.modal.run", None), "find_urls: ANSI codes stripped, no comfyui URL")
check(cm.find_urls("https://danieldlgrzz--sprite-studio-comfy-m3-comfyui.us-east.modal.run\nhttps://modal.com/apps/x") is None, "find_urls: no jobs URL -> None")
check(cm.find_urls("") is None and cm.find_urls(None) is None, "find_urls: empty output -> None")

# ---- 5. snippet
with open(os.path.join(ROOT, "cloud", "gpu_rates.json"), encoding="utf-8") as f:
    rates = json.load(f)
cfg = cm.effective_config({"MODAL_GPU": "RTX-PRO-6000,L40S"}, {})
snip = json.loads(json.dumps(cm.snippet(cfg, jobs, rates, "modal", "Cloud")))
check(snip["kind"] == "comfy-job" and snip["concurrency"] == cfg["max_containers"] == 3 and snip["idle"] == 300 and snip["probe"] == 0
      and snip["url"] == jobs and snip["name"] == "modal", "snippet: kind comfy-job, concurrency = max containers, idle, probe 0")
check(snip["cost_per_hour"] == 3.03, "snippet: cost_per_hour 3.03 for RTX-PRO-6000,L40S")
check(cm.snippet(cm.effective_config({"MODAL_GPU": "L40S"}, {}), jobs, rates, "m", "L")["cost_per_hour"] == 1.95, "snippet: cost_per_hour 1.95 for L40S")
check(cm.snippet(cm.effective_config({"MODAL_GPU": "Mystery"}, {}), jobs, rates, "m", "L")["cost_per_hour"] == 0, "snippet: unknown GPU -> 0")
check(snip["headers"] == {"Modal-Key": "$MODAL_KEY", "Modal-Secret": "$MODAL_SECRET"}, "snippet: headers are the $ references")
others = json.dumps({k: v for k, v in snip.items() if k != "headers"})
check("$" not in others and "ak-" not in others and "as-" not in others, "snippet: no other key holds a $ or a token")
check(cm.gpu_rate("l40s", rates) == 1.95 and cm.gpu_rate("", rates) is None and cm.gpu_rate("H100", {}) is None, "gpu_rate: case-insensitive, empty, no table")

# ---- 6. Modal client missing
with tempfile.TemporaryDirectory() as tmp:
    empty_path = os.path.join(tmp, "nothing-here")
    os.makedirs(empty_path)
    for argv in (["--dir", tmp, "cloud", "modal", "snippet"], ["--dir", tmp, "cloud", "modal", "deploy", "--app", "try-it"]):
        code, out, err = main_capture(argv, {"PSC_MODAL": None, "PATH": empty_path})
        check(code == 1 and "Modal client not found" in err and "PSC_MODAL" in err and "Traceback" not in err and out == "",
              "no Modal client: %s -> exit 1, plain message, nothing on stdout" % " ".join(argv[2:]))
    check(not os.path.exists(os.path.join(tmp, "cloud", "modal.env")), "no Modal client: deploy wrote no file")

# ---- dry run: effective config, no file, no deploy
with tempfile.TemporaryDirectory() as tmp:
    os.makedirs(os.path.join(tmp, "cloud"))
    with open(os.path.join(tmp, "cloud", "modal.env.example"), "w") as f:
        f.write("# template\nMODAL_GPU=L40S\n")
    code, out, err = main_capture(["--dir", tmp, "cloud", "modal", "deploy", "--dry-run", "--app", "try-it", "--max-containers", "2", "--idle", "60"],
                                  {"PSC_MODAL": None, "MODAL_APP": "from-shell", "MODAL_MAX_CONTAINERS": "9", "PATH": ""})
    check(code == 0 and out == "" and "MODAL_APP=try-it" in err and "MODAL_MAX_CONTAINERS=2" in err and "MODAL_IDLE_SECONDS=60" in err,
          "dry run: flags beat the shell, config on stderr, nothing on stdout, no Modal client needed")
    check("modal deploy cloud/modal_comfy.py" in err and "modal run cloud/modal_comfy.py::refresh_info" in err and "::download" not in err,
          "dry run: lists the commands")
    check(not os.path.exists(os.path.join(tmp, "cloud", "modal.env")), "dry run: writes nothing")
    code, out, err = main_capture(["--dir", tmp, "cloud", "modal", "deploy", "--dry-run", "--download", "--no-warm", "--preset", "starter"], {"PATH": ""})
    check("::download --workflows starter" in err and "refresh_info" not in err, "dry run: --download uses the preset, --no-warm drops the info run")
    code, out, err = main_capture(["--dir", tmp, "cloud", "modal", "deploy", "--dry-run", "--max-containers", "0"], {"PATH": ""})
    check(code == 1 and "MODAL_MAX_CONTAINERS" in err, "dry run: --max-containers 0 -> exit 1")

# ---- B1: deploying the live app needs --live
with tempfile.TemporaryDirectory() as tmp:
    os.makedirs(os.path.join(tmp, "cloud"))
    with open(os.path.join(tmp, "cloud", "modal.env.example"), "w") as f:
        f.write("MODAL_GPU=RTX-PRO-6000,L40S\n")
    env = {"PSC_MODAL": None, "MODAL_APP": None, "PATH": ""}
    base = ["--dir", tmp, "cloud", "modal", "deploy"]
    code, out, err = main_capture(base, env)
    check(code == 1 and "sprite-studio-comfy" in err and "refusing" in err and "GPU" in err and "RTX-PRO-6000" in err
          and "cost" in err and "--live" in err and "Modal client not found" not in err and out == "" and "Traceback" not in err,
          "live guard: the default app (the live one) is refused, naming the app, the GPU and the cost change")
    check(not os.path.exists(os.path.join(tmp, "cloud", "modal.env")), "live guard: nothing was written")
    code, out, err = main_capture(base, dict(env, MODAL_APP="sprite-studio-comfy"))
    check(code == 1 and "refusing" in err, "live guard: a stale MODAL_APP=sprite-studio-comfy in the shell is not consent")
    code, out, err = main_capture(base + ["--live"], env)
    check("refusing" not in err and "Modal client not found" in err, "live guard: --live passes the guard")
    code, out, err = main_capture(base + ["--app", "sprite-studio-comfy"], env)
    check("refusing" not in err and "Modal client not found" in err, "live guard: --app sprite-studio-comfy spelled out passes the guard")
    code, out, err = main_capture(base + ["--app", "sprite-studio-comfy-m3"], env)
    check("refusing" not in err and "Modal client not found" in err, "live guard: another app name needs nothing")
    code, out, err = main_capture(base + ["--dry-run"], env)
    check(code == 0 and "NOTE" in err and "--live" in err and out == "", "live guard: --dry-run on the live app prints a note and writes nothing")

# ---- the config keys agree with cloud/modal_comfy.py (3a)
with open(os.path.join(ROOT, "cloud", "modal_comfy.py"), encoding="utf-8") as f:
    modal_comfy = f.read()
check(all('"%s"' % name in modal_comfy for name in cm.ENV_KEYS.values()), "config keys: every MODAL_* key is read by cloud/modal_comfy.py")
check(all(('"%s": %s' % (k, json.dumps(v))) in modal_comfy for k, v in cm.DEFAULTS.items()), "config keys: defaults match cloud/modal_comfy.py")

print("\n%d/%d passed" % (sum(RESULTS), len(RESULTS)))
sys.exit(0 if all(RESULTS) else 1)
