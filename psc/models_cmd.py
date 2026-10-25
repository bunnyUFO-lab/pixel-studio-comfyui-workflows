"""models: list what the workflows need, pull what is missing, show the presets."""
from psc import core, local
from psc.presets import list_presets
from psc.stack import CliError


def expand(stack, names):
    """Preset names become their workflow names; anything else is taken as a workflow name or path."""
    presets = {p.name: p for p in list_presets(stack.root)}
    out = []
    for name in names:
        out += presets[name].models if name in presets else [name]
    return out


def run_models(stack, verb, names):
    if stack.service_running():
        # exec bypasses the entrypoint, so call the script the entrypoint would call
        proc = stack.compose("exec", "-T", "comfyui", "python", "/usr/local/bin/models.py", verb, *names)
    else:
        proc = stack.compose("run", "--rm", "-T", "comfyui", "models", verb, *names)
    return proc.returncode


def targets(stack, names):
    if names:
        return expand(stack, names)
    current = [m.strip() for m in stack.read_env().get("MODELS", "").split(",") if m.strip()]
    if not current:
        raise CliError("no model preset in .env; name a preset or a workflow, e.g. models list pixel "
                       "(see: pixel-studio-comfy models presets)")
    return current


def cmd_list(args, stack):
    return run_models(stack, "list", targets(stack, args.names))


def cmd_pull(args, stack):
    if not args.names:
        raise CliError("name a preset or a workflow to pull, e.g. models pull pixel (see: models presets)")
    return run_models(stack, "download", targets(stack, args.names))


def cmd_presets(args, stack):
    presets = list_presets(stack.root)
    width = max([len(p.name) for p in presets] + [4])
    local.say("%-*s  %9s  %s" % (width, "name", "workflows", "description"))
    for p in presets:
        local.say("%-*s  %9d  %s" % (width, p.name, p.count, p.description))
    local.say("none is also accepted by setup (no models downloaded).")
    return 0


def register(sub):
    g = core.group(sub, "models", help="list, pull and describe the models the workflows need")
    p = g.add_parser("list", help="show required models and whether they are present (default: the .env preset)")
    p.add_argument("names", nargs="*", metavar="PRESET|WORKFLOW")
    p.set_defaults(func=cmd_list)
    p = g.add_parser("pull", aliases=["get"],
                     help="download missing models; does NOT edit .env, so the next start only fetches the .env preset")
    p.add_argument("names", nargs="*", metavar="PRESET|WORKFLOW")
    p.set_defaults(func=cmd_pull)
    p = g.add_parser("presets", help="the preset table: name, workflows, description")
    p.set_defaults(func=cmd_presets)
