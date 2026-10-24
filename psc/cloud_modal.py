"""cloud modal: deploy the Modal job API (cloud/modal_comfy.py) and print the COMFY_SOURCES entry for Pixel Studio.

  pixel-studio-comfy cloud modal deploy [--gpu G] [--max-containers N] [--idle S] [--preset P] [--app NAME]
                                        [--download] [--no-warm] [--dry-run]
  pixel-studio-comfy cloud modal snippet [--name NAME] [--label LABEL]

Settings live in cloud/modal.env (git-ignored; cloud/modal.env.example is the tracked template). Flags are merged
into that file; precedence at deploy time is flag > process environment > file > default. stdout carries only the
snippet JSON (so `> file` works); progress and notes go to stderr. The Modal client is run as a subprocess
(PSC_MODAL, else `modal` on PATH): this CLI stays stdlib-only and never imports modal.
"""
import json
import os
import re
import shutil
import subprocess
import sys

from psc import core
from psc.stack import CliError

ENV_KEYS = {"app": "MODAL_APP", "gpu": "MODAL_GPU", "max_containers": "MODAL_MAX_CONTAINERS",
            "idle": "MODAL_IDLE_SECONDS", "preset": "MODAL_PRESET", "job_timeout": "MODAL_JOB_TIMEOUT"}
DEFAULTS = {"app": "sprite-studio-comfy", "gpu": "RTX-PRO-6000,L40S", "max_containers": 3, "idle": 300,
            "preset": "pixel", "job_timeout": 3600}
INT_RANGES = {"max_containers": (1, 32), "idle": (1, 3600 * 24), "job_timeout": (1, 3600)}
SCRIPT = "cloud/modal_comfy.py"
ENV_FILE = "cloud/modal.env"
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
URL = re.compile(r"https://[A-Za-z0-9.-]+\.modal\.run")


# ----------------------------------------------------------------------------- pure helpers
def merge_env_text(text, updates):
    """Rewrite only the given KEY=VALUE lines in place; keep comments, order and unknown keys; append missing keys.
    Output uses LF endings and ends with a newline."""
    lines = (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    pending = dict(updates)
    for i, line in enumerate(lines):
        m = re.match(r"^(\s*(?:export\s+)?)([A-Za-z_][A-Za-z0-9_]*)(\s*=)", line)
        if m and m.group(2) in pending:
            lines[i] = "%s%s=%s" % (m.group(1), m.group(2), pending.pop(m.group(2)))
    for key, value in pending.items():
        lines.append("%s=%s" % (key, value))
    return "\n".join(lines) + "\n"


def effective_config(file_env, process_env):
    """The six settings: process env > file > defaults. Integers are checked (CliError naming the key)."""
    cfg = {}
    for key, default in DEFAULTS.items():
        name = ENV_KEYS[key]
        raw = (process_env.get(name) or "").strip() or (file_env.get(name) or "").strip()
        if not raw:
            cfg[key] = default
        elif isinstance(default, int):
            low, high = INT_RANGES[key]
            if not raw.isdigit() or not low <= int(raw) <= high:
                raise CliError("%s must be a whole number from %d to %d (got %r)" % (name, low, high, raw))
            cfg[key] = int(raw)
        else:
            cfg[key] = raw
    if not cfg["gpu"].replace(",", "").strip():
        raise CliError("MODAL_GPU is empty")
    return cfg


def find_urls(output):
    """(jobs url, comfyui url or None) from `modal deploy` output, or None when there is no jobs URL."""
    urls = []
    for url in URL.findall(ANSI.sub("", output or "")):
        if url not in urls:
            urls.append(url)

    def label(url):
        return url[len("https://"):].split(".", 1)[0]
    jobs = ([u for u in urls if label(u).endswith("-jobs")] or [u for u in urls if "-jobs" in u] or [None])[0]
    if jobs is None:
        return None
    comfy = ([u for u in urls if label(u).endswith("-comfyui")] or [None])[0]
    return jobs, comfy


def gpu_rate(gpu_list, rates):
    """USD per hour of the first GPU in a comma list, from the rates document, or None when it is not listed."""
    first = [g.strip() for g in (gpu_list or "").split(",") if g.strip()]
    if not first:
        return None
    table = {k.lower(): v for k, v in ((rates or {}).get("rates") or {}).items()}
    return table.get(first[0].lower())


def snippet(cfg, url, rates, name="modal", label="Cloud"):
    """The COMFY_SOURCES object. Headers are always the $MODAL_KEY / $MODAL_SECRET references, never values."""
    rate = gpu_rate(cfg["gpu"], rates)
    return {"name": name, "kind": "comfy-job", "label": label, "url": url, "probe": 0, "idle": cfg["idle"],
            "concurrency": cfg["max_containers"], "cost_per_hour": 0 if rate is None else rate,
            "headers": {"Modal-Key": "$MODAL_KEY", "Modal-Secret": "$MODAL_SECRET"}}


def ascii_line(text):
    return ANSI.sub("", text).encode("ascii", "replace").decode("ascii").rstrip()


# ----------------------------------------------------------------------------- the Modal client
def say(text=""):
    print(text, file=sys.stderr, flush=True)


def find_modal(environ=None):
    environ = os.environ if environ is None else environ
    path = environ.get("PSC_MODAL") or shutil.which("modal")
    if not path:
        raise CliError("Modal client not found: pip install modal && modal setup, or set PSC_MODAL to its path")
    return path


def modal_env(cfg):
    env = dict(os.environ)
    env["PYTHONWARNINGS"] = "ignore"       # the client prints asyncio deprecation warnings on Python 3.14
    env["PYTHONIOENCODING"] = "utf-8"      # and box-drawing characters that a cp1252 console cannot take
    for key, name in ENV_KEYS.items():     # the resolved settings win over a stale shell value
        env[name] = str(cfg[key])
    return env


def run_modal(args, stack, cfg, stream=True):
    """Run the Modal client in the stack folder. Output is read as UTF-8 and echoed ASCII-safe to stderr line by line;
    returns (exit code, all output)."""
    cmd = [find_modal()] + list(args)
    try:
        proc = subprocess.Popen(cmd, cwd=str(stack.root), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, env=modal_env(cfg))
    except OSError as e:
        raise CliError("could not run %s: %s" % (cmd[0], e))
    lines = []
    for raw in iter(proc.stdout.readline, b""):
        text = raw.decode("utf-8", "replace")
        lines.append(text)
        if stream:
            say(ascii_line(text))
    proc.wait()
    return proc.returncode, "".join(lines)


# ----------------------------------------------------------------------------- commands
def read_rates(stack):
    text = stack.read_text("cloud", "gpu_rates.json")
    try:
        return json.loads(text) if text else {}
    except ValueError:
        return {}


def file_env(stack):
    return core.parse_env(stack.read_text(*ENV_FILE.split("/")) or "")


def flag_updates(args):
    given = {"app": args.app, "gpu": args.gpu, "max_containers": args.max_containers, "idle": args.idle,
             "preset": args.preset}
    return {ENV_KEYS[k]: str(v) for k, v in given.items() if v is not None}


def print_snippet(cfg, url, rates, name, label):
    print(json.dumps(snippet(cfg, url, rates, name, label), indent=1), flush=True)
    if gpu_rate(cfg["gpu"], rates) is None:
        say("WARN: GPU %r is not in cloud/gpu_rates.json; cost_per_hour is 0." % cfg["gpu"].split(",")[0])
    say("Add this object to COMFY_SOURCES in Pixel Studio's env; MODAL_KEY / MODAL_SECRET are a Modal proxy auth "
        "token (dashboard > Settings > Proxy Auth Tokens).")


def cmd_deploy(args, stack):
    updates = flag_updates(args)
    process_env = {k: v for k, v in os.environ.items() if k not in updates}     # a flag beats the shell value
    text = stack.read_text(*ENV_FILE.split("/"))
    new_text = merge_env_text(text if text is not None else (stack.read_text("cloud", "modal.env.example") or ""),
                              updates)
    cfg = effective_config(core.parse_env(new_text), process_env)
    commands = []
    if args.download:
        commands.append(["run", SCRIPT + "::download", "--workflows", cfg["preset"]])
    commands.append(["deploy", SCRIPT])
    if not args.no_warm:
        commands.append(["run", SCRIPT + "::refresh_info"])
    if args.dry_run:
        say("Effective config (%s would %s):" % (ENV_FILE, "change" if updates else "stay as it is"))
        for key, name in ENV_KEYS.items():
            say("  %s=%s" % (name, cfg[key]))
        for c in commands:
            say("  modal " + " ".join(c))
        say("Dry run: nothing was written or deployed.")
        return 0
    find_modal()
    if text is None or updates:
        stack.write_text(ENV_FILE, new_text)
        say("wrote %s" % ENV_FILE)
    say("deploying app %s: GPU %s, up to %d containers, idle %d s" % (cfg["app"], cfg["gpu"], cfg["max_containers"], cfg["idle"]))
    if args.download:
        code, _ = run_modal(commands[0], stack, cfg)
        if code:
            raise CliError("the model download failed (modal exit code %d); see the output above" % code)
    code, out = run_modal(["deploy", SCRIPT], stack, cfg)
    if code:
        raise CliError("modal deploy failed (exit code %d). Last lines:\n%s" % (code, "\n".join(ascii_line(l) for l in out.splitlines()[-20:])))
    urls = find_urls(out)
    if urls is None:
        raise CliError("modal deploy finished but printed no jobs URL. Last lines:\n%s"
                       % "\n".join(ascii_line(l) for l in out.splitlines()[-20:]))
    jobs, comfy = urls
    if comfy:
        say("single-container ComfyUI endpoint (unchanged): " + comfy)
    if not args.no_warm:
        say("caching the node and model lists (CPU, no GPU) ...")
        code, _ = run_modal(["run", SCRIPT + "::refresh_info"], stack, cfg)
        if code:
            say("WARN: the info refresh failed (exit code %d); the first request to /v1/info will do it." % code)
    print_snippet(cfg, jobs, read_rates(stack), "modal", "Cloud")
    return 0


def cmd_snippet(args, stack):
    cfg = effective_config(file_env(stack), os.environ)
    code, out = run_modal(["profile", "current"], stack, cfg, stream=False)
    workspace = ascii_line(out.strip().splitlines()[-1]) if code == 0 and out.strip() else ""
    if not workspace or " " in workspace:
        raise CliError("could not read the Modal workspace from `modal profile current`; run `modal setup`")
    url = "https://%s--%s-jobs.modal.run" % (workspace, cfg["app"])
    say("note: the URL is derived from the Modal profile %r and the app name; a deploy prints the real one." % workspace)
    print_snippet(cfg, url, read_rates(stack), args.name, args.label)
    return 0


def register(sub):
    cloud = core.group(sub, "cloud", help="cloud GPU providers")
    modal = cloud.add_parser("modal", help="Modal: deploy the job API, print the Pixel Studio source").add_subparsers(
        dest="modal_cmd", metavar="ACTION")
    modal.required = True
    p = modal.add_parser("deploy", help="write cloud/modal.env, deploy to Modal, print the COMFY_SOURCES entry")
    p.add_argument("--gpu", help="GPU list, comma separated, tried in order (default RTX-PRO-6000,L40S)")
    p.add_argument("--max-containers", type=int, help="job containers at once, 1..32 (default 3)")
    p.add_argument("--idle", type=int, help="seconds a warm container waits for the next job (default 300)")
    p.add_argument("--preset", help="models `--download` fetches: pixel, starter, all or workflow names (default pixel)")
    p.add_argument("--app", help="Modal app name (default sprite-studio-comfy, the live app: pick another to try)")
    p.add_argument("--download", action="store_true", help="download the models into the Modal Volume first (can take long)")
    p.add_argument("--no-warm", action="store_true", help="skip the CPU run that caches the node and model lists")
    p.add_argument("--dry-run", action="store_true", help="print the effective config and commands; write and deploy nothing")
    p.set_defaults(func=cmd_deploy)
    p = modal.add_parser("snippet", help="print the COMFY_SOURCES entry for the current config without deploying")
    p.add_argument("--name", default="modal", help="source name (default modal)")
    p.add_argument("--label", default="Cloud", help="label shown in Pixel Studio (default Cloud)")
    p.set_defaults(func=cmd_snippet)
