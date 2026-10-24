"""doctor, status, up, down and logs: the commands that work on the local Docker stack."""
import platform
import shutil
import sys
import time
import urllib.error

from psc import __version__, core
from psc.core import Check
from psc.stack import CliError, run

SMI_QUERY = ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
NET_ERRORS = (OSError, ValueError, urllib.error.URLError)


def say(text=""):
    print(text, flush=True)


# ---------------------------------------------------------------- checks (doctor; setup reuses them)

def check_python():
    v = sys.version_info
    ok = v >= (3, 9)
    return Check("OK" if ok else "FAIL", "python", "%d.%d.%d" % v[:3] + ("" if ok else " (3.9 or newer needed)"))


def check_stack_files(stack):
    checks = []
    if stack.path("docker-compose.yml").is_file():
        checks.append(Check("OK", "stack folder", "docker-compose.yml found in %s" % stack.root))
    else:
        checks.append(Check("FAIL", "stack folder", "no docker-compose.yml in %s (use --dir)" % stack.root))
    env_text = stack.read_text(".env")
    if env_text is None:
        checks.append(Check("WARN", ".env", "missing: run the installer or copy a preset from env/"))
    else:
        models = core.parse_env(env_text).get("MODELS", "")
        checks.append(Check("OK", ".env", "MODELS=%s" % (models if models else "(empty)")))
    sec_text = stack.read_text(".secrets.env")
    if sec_text is None:
        checks.append(Check("WARN", ".secrets.env", "missing: optional, copy .secrets.env.example for Civitai or Hugging Face tokens"))
    else:
        keys = core.parse_env(sec_text)
        shown = ", ".join("%s %s" % (k, "set" if v else "empty") for k, v in sorted(keys.items()))
        checks.append(Check("OK", ".secrets.env", shown or "no keys"))
    return checks


def docker_ready(checks):
    return all(c.status == "OK" for c in checks if c.name.startswith("docker"))


def check_docker(stack):
    if not shutil.which("docker"):
        return [Check("FAIL", "docker", "not on PATH: install Docker Desktop (Windows) or Docker Engine (Linux)")]
    checks = [Check("OK", "docker", "CLI found")]
    try:
        info = run(["docker", "info", "--format", "{{.ServerVersion}}"], capture=True, timeout=30)
    except CliError as e:
        info = None
        checks.append(Check("FAIL", "docker daemon", "%s Start Docker Desktop." % e))
    if info is not None:
        if info.returncode == 0:
            checks.append(Check("OK", "docker daemon", "engine %s" % info.stdout.strip()))
        else:
            checks.append(Check("FAIL", "docker daemon", "does not answer: start Docker Desktop"))
    comp = run(["docker", "compose", "version", "--short"], capture=True, timeout=30)
    ver = comp.stdout.strip().lstrip("v") if comp.returncode == 0 else ""
    if ver[:1].isdigit() and int(ver.split(".")[0]) >= 2:
        checks.append(Check("OK", "docker compose", "v" + ver))
    else:
        checks.append(Check("FAIL", "docker compose", "Compose v2 not found (docker compose version): update Docker"))
    return checks


def host_gpu():
    """(name, driver, memory) from the host nvidia-smi, or None."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        proc = run(SMI_QUERY, capture=True, timeout=20)
    except CliError:
        return None
    line = proc.stdout.strip().splitlines()[0] if proc.returncode == 0 and proc.stdout.strip() else ""
    parts = [p.strip() for p in line.split(",")]
    return tuple(parts[:3]) if len(parts) >= 3 else None


def vram_mib(memory):
    """A string like 24564 MiB becomes 24564; anything else is None."""
    first = memory.split()[0] if memory and memory.split() else ""
    return int(first) if first.isdigit() else None


def check_host_gpu():
    gpu = host_gpu()
    if gpu is None:
        return Check("WARN", "nvidia-smi", "not answering on the host (fine on a Linux host where it is not on PATH)")
    return Check("OK", "nvidia-smi", "%s, driver %s, %s" % gpu)


def image_exists(ref):
    return run(["docker", "image", "inspect", ref], capture=True, timeout=30).returncode == 0


def stack_image(stack):
    ref = stack.read_env().get("COMFYUI_REF") or core.compose_default(
        stack.read_text("docker-compose.yml"), "COMFYUI_REF", core.DEFAULT_COMFYUI_REF)
    return "comfyui-local:%s" % ref


def base_image(stack):
    return core.dockerfile_base_image(stack.read_text("docker", "Dockerfile") or "")


def check_gpu_in_docker(stack, state, pull=False):
    """GPU visible inside Docker. Doctor never pulls or builds; setup passes pull=True for the CUDA base image."""
    if state == "running":
        proc = stack.compose("exec", "-T", "comfyui", *SMI_QUERY, capture=True, timeout=60)
        source = "running comfyui container"
    else:
        candidates = [stack_image(stack), base_image(stack)]
        image = next((i for i in candidates if i and image_exists(i)), None)
        if image is None and pull and candidates[1]:
            say("Checking the GPU inside Docker (downloads the CUDA base image the build needs anyway)")
            image = candidates[1]
            if run(["docker", "pull", image]).returncode != 0:
                return Check("FAIL", "gpu in docker", "could not pull %s" % image)
        if image is None:
            return Check("SKIP", "gpu in docker", "build the image first (pixel-studio-comfy up)")
        proc = run(["docker", "run", "--rm", "--gpus", "all", "--entrypoint", SMI_QUERY[0], image] + SMI_QUERY[1:],
                   capture=True, timeout=180)
        source = "image " + image
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    if proc.returncode == 0 and "," in line:
        return Check("OK", "gpu in docker", "%s (%s)" % (line, source))
    return Check("FAIL", "gpu in docker", "no GPU visible in a container (%s): check the NVIDIA driver, Docker Desktop "
                 "WSL2 GPU support or the NVIDIA Container Toolkit" % source)


def check_disk(stack):
    out = stack.read_env().get("COMFY_OUTPUT_DIR") or ""
    target = stack.path(out) if out else stack.root
    while not target.exists() and target != target.parent:
        target = target.parent
    free = shutil.disk_usage(str(target)).free
    status = "OK" if free >= 50 * 1024 ** 3 else "WARN"
    return Check(status, "disk", "%s free where outputs go (%s)%s" % (
        core.gb(free), target, "" if status == "OK" else ": under 50 GB, models are large"))


def check_container(stack):
    state = stack.service_state()
    if state == "running":
        return state, Check("OK", "container", "comfyui running")
    if state == "restarting":
        return state, Check("FAIL", "container", "comfyui keeps restarting: see logs -f")
    if state is None:
        return state, Check("WARN", "container", "comfyui is not created (pixel-studio-comfy up)")
    return state, Check("WARN", "container", "comfyui is %s (pixel-studio-comfy up)" % state)


def api_headers(stack):
    return {}


def check_api(stack, state):
    try:
        stats = core.summarize_system_stats(stack.http_json("/system_stats", timeout=5, headers=api_headers(stack)))
        return Check("OK", "comfyui api", "%s answers, version %s" % (stack.url, stats["version"]))
    except NET_ERRORS:
        if state == "running":
            return Check("WARN", "comfyui api", "no answer at %s: starting, or still downloading models (see logs -f)" % stack.url)
        return Check("FAIL", "comfyui api", "no answer at %s and no container running" % stack.url)


def check_models(stack, state):
    if not stack.read_env().get("MODELS"):
        return Check("SKIP", "models", "no model preset in .env")
    if state != "running":
        return Check("SKIP", "models", "container not running")
    # The entrypoint maps "models" to models.py, but exec bypasses the entrypoint: call the script itself.
    proc = stack.compose("exec", "-T", "comfyui", "python", "/usr/local/bin/models.py", "list", capture=True, timeout=120)
    ok, missing, partial, names = core.parse_models_list(proc.stdout or "")
    total = ok + missing + partial
    if proc.returncode != 0 or total == 0:
        return Check("WARN", "models", "could not read the model list from the container")
    if missing + partial == 0:
        return Check("OK", "models", "%d/%d present" % (ok, total))
    shown = ", ".join(names[:5]) + (" ..." if len(names) > 5 else "")
    return Check("WARN", "models", "%d/%d present, %d missing: %s" % (ok, total, missing + partial, shown))


def run_doctor(stack):
    checks = [check_python()]
    checks += check_stack_files(stack)
    docker = check_docker(stack)
    checks += docker
    checks.append(check_host_gpu())
    usable = docker_ready(docker) and stack.path("docker-compose.yml").is_file()
    state = container = None
    if usable:
        state, container = check_container(stack)

    def skipped(name):
        return Check("SKIP", name, "docker is not usable")

    checks.append(check_gpu_in_docker(stack, state) if usable else skipped("gpu in docker"))
    checks.append(check_disk(stack))
    checks.append(container or skipped("container"))
    checks.append(check_api(stack, state))
    checks.append(check_models(stack, state) if usable else skipped("models"))
    return checks


def footer(stack):
    sha = ""
    if shutil.which("git"):
        proc = run(["git", "-C", str(stack.root), "rev-parse", "--short", "HEAD"], capture=True, timeout=10)
        sha = proc.stdout.strip() if proc.returncode == 0 else ""
    return "pixel-studio-comfy %s | repo %s | %s" % (__version__, sha or "no git", platform.platform())


# ---------------------------------------------------------------- commands

def cmd_doctor(args, stack):
    checks = run_doctor(stack)
    say(core.format_checks(checks))
    say(footer(stack))
    return core.doctor_exit_code(checks)


def cmd_status(args, stack):
    state = stack.service_state()
    say("container: %s" % (state or "not created"))
    try:
        stats = core.summarize_system_stats(stack.http_json("/system_stats", headers=api_headers(stack)))
        queue = core.summarize_queue(stack.http_json("/queue", headers=api_headers(stack)))
    except NET_ERRORS:
        say("comfyui: not reachable at %s" % stack.url)
        return 1
    say("comfyui: reachable at %s, version %s" % (stack.url, stats["version"]))
    if stats["gpu"]:
        vram = ""
        if stats["vram_total"]:
            vram = ", VRAM %s free of %s" % (core.gb(stats["vram_free"] or 0), core.gb(stats["vram_total"]))
        say("gpu: %s%s" % (stats["gpu"], vram))
    else:
        say("gpu: none reported")
    say("queue: %d running, %d pending" % (queue["running"], queue["pending"]))
    if queue["running_ids"]:
        say("  running: " + " ".join(queue["running_ids"]))
    if queue["pending_ids"]:
        say("  pending: " + " ".join(queue["pending_ids"]))
    return 0


def wait_for_api(stack, seconds):
    deadline = time.time() + seconds
    while True:
        try:
            stack.http_json("/system_stats", timeout=3, headers=api_headers(stack))
            return True
        except NET_ERRORS:
            pass
        if time.time() >= deadline:
            return False
        time.sleep(2)


def cmd_up(args, stack):
    code = stack.compose("up", "-d").returncode
    if code:
        return code
    if args.wait <= 0:
        say("started; ComfyUI will be at %s" % stack.url)
        return 0
    say("waiting up to %d s for ComfyUI at %s ..." % (args.wait, stack.url))
    if wait_for_api(stack, args.wait):
        say("ComfyUI is up at %s" % stack.url)
    else:
        say("WARN: no answer from %s after %d s. The first start may still be downloading models: "
            "follow it with: pixel-studio-comfy logs -f" % (stack.url, args.wait))
    return 0


def cmd_down(args, stack):
    return stack.compose("down").returncode  # never -v: the named volumes hold the models


def cmd_logs(args, stack):
    cmd = ["logs", "--tail", str(args.tail)] + (["-f"] if args.follow else []) + ["comfyui"]
    return stack.compose(*cmd).returncode


def register(sub):
    p = sub.add_parser("doctor", help="check Python, Docker, GPU, disk, the container and the API")
    p.set_defaults(func=cmd_doctor)
    p = sub.add_parser("status", help="container, ComfyUI version, GPU memory and queue")
    p.set_defaults(func=cmd_status)
    p = sub.add_parser("up", help="start the stack (docker compose up -d) and wait for ComfyUI")
    p.add_argument("--wait", type=int, default=120, metavar="N",
                   help="seconds to wait for the API, 0 = do not wait (default 120)")
    p.set_defaults(func=cmd_up)
    p = sub.add_parser("down", help="stop and remove the containers (models are kept)")
    p.set_defaults(func=cmd_down)
    p = sub.add_parser("logs", help="show the ComfyUI container log")
    p.add_argument("--tail", type=int, default=200, metavar="N", help="lines to show (default 200)")
    p.add_argument("-f", "--follow", action="store_true", help="keep following the log")
    p.set_defaults(func=cmd_logs)
