"""update: fast-forward the checkout, then restart the stack. Never resets, stashes or switches branches."""
import types
from pathlib import Path

from psc import local
from psc.presets import update_refusal
from psc.stack import CliError, run


def git(stack, *args, capture=True):
    return run(["git", "-C", str(stack.root)] + list(args), capture=capture, timeout=300)


def is_checkout(stack):
    proc = git(stack, "rev-parse", "--show-toplevel")
    return proc.returncode == 0 and Path(proc.stdout.strip()).resolve() == stack.root


def cmd_update(args, stack):
    if not is_checkout(stack):
        raise CliError("%s is not a git checkout, so there is nothing to update from. "
                       "Install from a clone of the repository." % stack.root)
    porcelain = git(stack, "status", "--porcelain", "--untracked-files=no").stdout
    up = git(stack, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    reason = update_refusal(porcelain, up.stdout.strip() if up.returncode == 0 else "")
    if reason:
        raise CliError("cannot update: " + reason)
    old = git(stack, "rev-parse", "--short", "HEAD").stdout.strip()
    code = git(stack, "pull", "--ff-only", capture=False).returncode
    if code:
        raise CliError("git pull --ff-only failed (the branch has diverged from %s?); nothing was changed." % up.stdout.strip())
    new = git(stack, "rev-parse", "--short", "HEAD").stdout.strip()
    changed = 0
    if old != new:
        changed = len([p for p in git(stack, "diff", "--name-only", old, new).stdout.splitlines() if p.strip()])
    local.say("%s -> %s, %d changed paths" % (old, new, changed))
    if args.no_up:
        return 0
    return local.cmd_up(types.SimpleNamespace(wait=120), stack)


def register(sub):
    p = sub.add_parser("update", help="git pull --ff-only, then restart the stack")
    p.add_argument("--no-up", action="store_true", help="only pull; do not run up afterwards")
    p.set_defaults(func=cmd_update)
