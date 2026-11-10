"""cloud trainer: build the style-trainer image for rented machines (trainer/cloud/Dockerfile).

  pixel-studio-comfy cloud trainer build [--base IMAGE] [--tag IMAGE] [--push] [--dry-run]

The image is ostris ai-toolkit plus trainer/runner.py; it ends the machine when the runner exits (see
trainer/README.md, "On a rented machine"). It is published as a public GHCR package, so RunPod needs no registry
login. stdout carries only the docker commands of --dry-run; progress and notes go to stderr.
"""
import re

from psc import core
from psc.cloud_modal import say
from psc.stack import CliError, run

REGISTRY_IMAGE = "ghcr.io/bunnyufo-lab/psc-trainer"
DEFAULT_BASE = "ostris/aitoolkit:0.13.23@sha256:35e46171bc6def5e745fc478befec2ab0a0551fe78ce3d5f76cb22271795bb52"
DOCKERFILE = "trainer/cloud/Dockerfile"
NOT_LOGGED_IN = re.compile(r"denied|unauthorized|authentication required|requested access", re.I)


def git_sha(stack):
    proc = run(["git", "rev-parse", "--short", "HEAD"], cwd=str(stack.root), capture=True, timeout=30)
    sha = (proc.stdout or "").strip() if proc.returncode == 0 else ""
    return sha if re.fullmatch(r"[0-9a-f]{4,40}", sha) else "dev"


def build_plan(stack, base=None, tag=None, push=False):
    """(base image, tags, [argv]) for docker build and, with push, one docker push per tag."""
    base = base or DEFAULT_BASE
    if tag:
        tags = [tag]
    else:
        named = base.split("@", 1)[0]          # drop a @sha256:... digest; the tag is what names the result
        ref = named.rsplit(":", 1)[1] if ":" in named.rsplit("/", 1)[-1] else "latest"
        tags = ["%s:aitk-%s-%s" % (REGISTRY_IMAGE, ref, git_sha(stack)), REGISTRY_IMAGE + ":latest"]
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
    say("the base image %s is about 6.6 GB to download if it is not on this machine yet" % base)
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


def register(sub):
    cloud = core.group(sub, "cloud", help="cloud GPU providers")
    trainer = cloud.add_parser("trainer", help="the style trainer image for rented machines"
                               ).add_subparsers(dest="trainer_cmd", metavar="ACTION")
    trainer.required = True
    p = trainer.add_parser("build", help="docker build the trainer image on top of ostris/aitoolkit (and push it)")
    p.add_argument("--base", help="base image (default %s)" % DEFAULT_BASE)
    p.add_argument("--tag", help="image tag (default %s:aitk-<base tag>-<git sha> and :latest)" % REGISTRY_IMAGE)
    p.add_argument("--push", action="store_true", help="docker push the tags (needs `docker login ghcr.io`)")
    p.add_argument("--dry-run", action="store_true", help="print the docker commands; run nothing")
    p.set_defaults(func=cmd_build)
