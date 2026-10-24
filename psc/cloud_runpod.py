"""cloud runpod: build the RunPod Serverless worker image and print the Pixel Studio source for the endpoint.

  pixel-studio-comfy cloud runpod build [--base IMAGE] [--tag IMAGE] [--push] [--dry-run]
  pixel-studio-comfy cloud runpod snippet --endpoint ID [--name NAME] [--label LABEL] [--concurrency N] [--gpu GPU]
  pixel-studio-comfy cloud runpod settings

The endpoint itself is created by hand in the RunPod console (its API is changing and needs a key): `settings` prints
what to enter there, `snippet` prints the COMFY_SOURCES entry. stdout carries only the commands (build --dry-run) or
the snippet JSON / settings text; progress and notes go to stderr. See cloud/runpod/README.md.
"""
import json
import re

from psc import core
from psc.cloud_modal import say
from psc.local import image_exists, stack_image
from psc.stack import CliError, run

REGISTRY_IMAGE = "ghcr.io/bunnyufo-lab/psc-comfy-runpod"
DOCKERFILE = "cloud/runpod/Dockerfile"
RATES_FILE = ("cloud", "runpod", "gpu_rates.json")
API_BASE = "https://api.runpod.ai/v2/"
NOT_LOGGED_IN = re.compile(r"denied|unauthorized|authentication required|requested access", re.I)


def read_rates(stack):
    text = stack.read_text(*RATES_FILE)
    try:
        return json.loads(text) if text else {}
    except ValueError:
        return {}


def gpu_rate(gpu, rates):
    """USD per hour of a rates key such as RTX-4090 (case-insensitive), or None when it is not listed."""
    table = {k.lower(): v for k, v in ((rates or {}).get("rates") or {}).items()}
    return table.get((gpu or "").strip().lower())


def endpoint_id(value):
    """The endpoint id from an id or a https://api.runpod.ai/v2/<id>[/...] URL."""
    value = (value or "").strip()
    if value.startswith(API_BASE):
        value = value[len(API_BASE):].split("/", 1)[0]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise CliError("--endpoint must be the endpoint id from the RunPod console (letters and digits), got %r" % value)
    return value


def snippet(endpoint, rates, name="runpod", label="Cloud (RunPod)", concurrency=3, gpu="RTX-4090"):
    """The COMFY_SOURCES object. The header is always the $RUNPOD_API_KEY reference, never a value."""
    rate = gpu_rate(gpu, rates)
    return {"name": name, "kind": "runpod", "label": label, "url": API_BASE + endpoint_id(endpoint), "probe": 0,
            "concurrency": concurrency, "cost_per_hour": 0 if rate is None else rate,
            "headers": {"Authorization": "$RUNPOD_API_KEY"}}


def settings_text(image=REGISTRY_IMAGE + ":latest"):
    return "\n".join([
        "RunPod endpoint settings (console > Serverless > New Endpoint > Import from Docker Registry):",
        "  Container image      %s   (a public GHCR package: no registry credential)" % image,
        "  GPU priority         24 GB (RTX 4090) -> 48 GB PRO (L40S / RTX 6000 Ada) -> 32 GB (RTX 5090)",
        "  Active workers       0           Max workers  3 (= the source's concurrency)    GPUs per worker  1",
        "  Idle timeout         300 s       FlashBoot    on",
        "  Execution timeout    3900 s      (a job may run 3600 s; the default 600 s would kill video jobs)",
        "  Scaling              request count, scaler value 1",
        "  Container disk       20 GB",
        "  Network volume       100 GB standard, in a data center that has both 4090 and L40S; attach it to the endpoint",
        "                       (mounted at /runpod-volume; the worker links ComfyUI's models folder to it)",
        "  Environment          optional, for outputs over 9 MB: PSC_S3_BUCKET, PSC_S3_ENDPOINT, PSC_S3_PREFIX,",
        "                       AWS_REGION, and AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY as RunPod secrets",
        "                       ({{ RUNPOD_SECRET_<name> }}). Details: cloud/runpod/README.md",
        "After creating it: send the download job once to fill the volume (README step 4), then run",
        "  python cloud/job_smoke.py --runpod <endpoint id> --n 1",
    ])


# ----------------------------------------------------------------------------- build
def git_sha(stack):
    proc = run(["git", "rev-parse", "--short", "HEAD"], cwd=str(stack.root), capture=True, timeout=30)
    sha = (proc.stdout or "").strip() if proc.returncode == 0 else ""
    return sha if re.fullmatch(r"[0-9a-f]{4,40}", sha) else "dev"


def build_plan(stack, base=None, tag=None, push=False):
    """(base image, tags, [argv]) for docker build and, with push, one docker push per tag."""
    base = base or stack_image(stack)
    if tag:
        tags = [tag]
    else:
        ref = base.rsplit(":", 1)[1] if ":" in base.rsplit("/", 1)[-1] else "latest"
        tags = ["%s:%s-%s" % (REGISTRY_IMAGE, ref, git_sha(stack)), REGISTRY_IMAGE + ":latest"]
    build = ["docker", "build", "-f", DOCKERFILE, "--build-arg", "BASE=" + base]
    for t in tags:
        build += ["-t", t]
    commands = [build + ["."]]
    if push:
        commands += [["docker", "push", t] for t in tags]
    return base, tags, commands


def cmd_build(args, stack):
    base, tags, commands = build_plan(stack, args.base, args.tag, args.push)
    if args.dry_run:
        for c in commands:
            print(" ".join(c))
        say("Dry run: nothing was built or pushed (the build context is %s)." % stack.root)
        return 0
    if not image_exists(base):
        raise CliError("the base image %s does not exist; build it first with `docker compose build comfyui` "
                       "(or pass --base)" % base)
    say("building %s from %s (the first build copies about 15 GB of layers)" % (tags[0], base))
    code = run(commands[0], cwd=str(stack.root)).returncode
    if code:
        raise CliError("docker build failed (exit code %d); see the output above" % code)
    for c in commands[1:]:
        proc = run(c, cwd=str(stack.root), capture=True)
        say((proc.stdout or "").rstrip())
        if proc.returncode:
            if NOT_LOGGED_IN.search(proc.stdout or ""):
                raise CliError("docker push was refused: log in first with `docker login ghcr.io` (user name and a "
                               "personal access token with write:packages), then run this command again")
            raise CliError("docker push failed (exit code %d); see the output above" % proc.returncode)
    say("built %s" % ", ".join(tags))
    return 0


def cmd_snippet(args, stack):
    rates = read_rates(stack)
    entry = snippet(args.endpoint, rates, args.name, args.label, args.concurrency, args.gpu)
    print(json.dumps(entry, indent=1), flush=True)
    if gpu_rate(args.gpu, rates) is None:
        say("WARN: GPU %r is not in cloud/runpod/gpu_rates.json; cost_per_hour is 0." % args.gpu)
    say("Add this object to COMFY_SOURCES in Pixel Studio's env; RUNPOD_API_KEY is a RunPod API key "
        "(console > Settings > API Keys) in the environment Pixel Studio runs with.")
    say(settings_text())
    return 0


def cmd_settings(args, stack):
    print(settings_text())
    return 0


def register(sub):
    cloud = core.group(sub, "cloud", help="cloud GPU providers")
    runpod = cloud.add_parser("runpod", help="RunPod Serverless: build the worker image, print the Pixel Studio source"
                              ).add_subparsers(dest="runpod_cmd", metavar="ACTION")
    runpod.required = True
    p = runpod.add_parser("build", help="docker build the worker image from the local ComfyUI image (and push it)")
    p.add_argument("--base", help="ComfyUI image to build on (default: the stack's comfyui-local:<ref>)")
    p.add_argument("--tag", help="image tag (default %s:<ref>-<git sha> and :latest)" % REGISTRY_IMAGE)
    p.add_argument("--push", action="store_true", help="docker push the tags (needs `docker login ghcr.io`)")
    p.add_argument("--dry-run", action="store_true", help="print the docker commands; run nothing")
    p.set_defaults(func=cmd_build)
    p = runpod.add_parser("snippet", help="print the COMFY_SOURCES entry for an endpoint, then the settings checklist")
    p.add_argument("--endpoint", required=True, help="endpoint id from the RunPod console")
    p.add_argument("--name", default="runpod", help="source name (default runpod)")
    p.add_argument("--label", default="Cloud (RunPod)", help="label shown in Pixel Studio (default 'Cloud (RunPod)')")
    p.add_argument("--concurrency", type=int, default=3, help="jobs at once = the endpoint's max workers (default 3)")
    p.add_argument("--gpu", default="RTX-4090", help="GPU the cost is taken from (default RTX-4090)")
    p.set_defaults(func=cmd_snippet)
    p = runpod.add_parser("settings", help="print the recommended endpoint settings")
    p.set_defaults(func=cmd_settings)
