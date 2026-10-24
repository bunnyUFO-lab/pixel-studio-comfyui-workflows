"""setup: check the machine, choose models, write .env and .secrets.env (never overwriting), start, run doctor."""
import getpass
import os
import sys
import types

from psc import core, local
from psc.presets import SECRET_KEYS, fill_secrets, list_presets, render_env, workflow_names
from psc.stack import CliError

DEFAULT_PRESET = "pixel"
VIDEO_VRAM_MIB = 23000  # a 24 GB card reports about 24500 MiB; anything below 23000 is a smaller card


def say(text=""):
    local.say(text)


def step_checks(stack):
    """Stop with one plain sentence on a FAIL. Returns the host GPU tuple (or None)."""
    checks = [local.check_python()]
    if not stack.path("docker-compose.yml").is_file():
        checks.append(core.Check("FAIL", "stack folder", "no docker-compose.yml in %s (use --dir)" % stack.root))
    checks += local.check_docker(stack)
    host = local.check_host_gpu()
    checks.append(host)
    say(core.format_checks(checks))
    failed = [c for c in checks if c.status == "FAIL"]
    if failed:
        raise CliError("%s: %s" % (failed[0].name, failed[0].detail))
    state = stack.service_state()
    gpu = local.check_gpu_in_docker(stack, state, pull=True)
    say(core.format_checks([gpu]))
    if gpu.status == "FAIL":
        raise CliError("%s: %s" % (gpu.name, gpu.detail))
    return local.host_gpu()


def choose_models(args, stack, interactive):
    """Returns (label, models override or None, preset text). The override is set only for --models and none."""
    presets = {p.name: p for p in list_presets(stack.root)}
    if args.models:
        names = [m.strip() for m in args.models.split(",") if m.strip()]
        known = set(workflow_names(stack.root))
        unknown = [n for n in names if n not in known and n != "all"]
        if unknown:
            raise CliError("unknown workflow(s): %s (see the names under workflows/, without .json)" % ", ".join(unknown))
        return "custom", ",".join(names), preset_text(stack, DEFAULT_PRESET)
    name = args.preset
    if not name:
        name = DEFAULT_PRESET
        if interactive:
            say("Model presets (what gets downloaded):")
            for p in presets.values():
                say("  %-6s %2d workflows  %s" % (p.name, p.count, p.description))
            say("  none   no models")
            answer = input("Preset [%s]: " % DEFAULT_PRESET).strip()
            name = answer or DEFAULT_PRESET
    if name == "none":
        return "none", "", "# Preset: none\n"
    if name not in presets:
        raise CliError("unknown preset %s; choose one of: %s, none" % (name, ", ".join(presets)))
    return name, None, preset_text(stack, name)


def preset_text(stack, name):
    path = stack.path(".env.all") if name == "all" else stack.path("env", name + ".env")
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def step_env(args, stack, interactive, host):
    existing = stack.read_text(".env")
    flags = [f for f, v in (("--port", args.port), ("--lan", args.lan), ("--project-name", args.project_name),
                            ("--output-dir", args.output_dir), ("--models", args.models), ("--preset", args.preset)) if v]
    if existing is not None:
        models = core.parse_env(existing).get("MODELS", "")
        say("keeping existing .env (MODELS=%s)" % (models or "(empty)"))
        if flags:
            say("%s ignored: setup never edits an existing .env. Edit %s yourself to change it." % (
                ", ".join(flags), stack.path(".env")))
        return
    label, models, text = choose_models(args, stack, interactive)
    overrides = {}
    if args.port:
        overrides["PORT"] = str(args.port)
    overrides["BIND_ADDRESS"] = "0.0.0.0" if args.lan else "127.0.0.1"
    if args.project_name:
        overrides["COMPOSE_PROJECT_NAME"] = args.project_name
    if args.output_dir:
        overrides["COMFY_OUTPUT_DIR"] = args.output_dir
    if models is not None:
        overrides["MODELS"] = models
    stack.write_text(".env", render_env(text, overrides))
    say("wrote .env (preset: %s)" % label)
    if args.lan:
        say("NOTE: ComfyUI is reachable from your network (BIND_ADDRESS=0.0.0.0) and has no authentication until the gateway.")
    else:
        say("ComfyUI listens on 127.0.0.1 only (this PC). Use --lan to open it to the network; "
            "there is no authentication until the gateway.")
    mib = local.vram_mib(host[2]) if host else None
    if mib is not None and mib < VIDEO_VRAM_MIB:
        say("NOTE: this GPU has %.0f GB; the video and animation workflows need 24 GB." % (mib / 1024.0))


def read_token(name, flag_value, interactive):
    if flag_value:
        return flag_value
    if os.environ.get(name):
        return os.environ[name]
    if interactive:
        return getpass.getpass("%s (Enter to skip): " % name).strip()
    return ""


def step_secrets(args, stack, interactive):
    existing = stack.read_text(".secrets.env")
    text = existing if existing is not None else (stack.read_text(".secrets.env.example") or "")
    current = core.parse_env(text)
    wanted = {"CIVITAI_TOKEN": args.civitai_token, "HF_TOKEN": args.hf_token}
    values = {}
    for key in SECRET_KEYS:
        if key in current and not current[key]:  # only an empty value is ever filled
            token = read_token(key, wanted[key], interactive)
            if token:
                values[key] = token
    new_text, filled = fill_secrets(text, values)
    if existing is None or new_text != existing:
        stack.write_text(".secrets.env", new_text, private=True)
    keys = core.parse_env(new_text)
    shown = ", ".join("%s %s" % (k, "set" if keys.get(k) else "empty") for k in SECRET_KEYS if k in keys)
    say("%s %s (%s)" % (".secrets.env", "written" if existing is None else "kept", shown or "no keys"))


def cmd_setup(args, stack):
    interactive = not args.yes and sys.stdin.isatty()
    host = step_checks(stack)
    step_env(args, stack, interactive, host)
    step_secrets(args, stack, interactive)
    code = 0
    if not args.no_up:
        code = local.cmd_up(types.SimpleNamespace(wait=600), stack)
        if code:
            return code
        say("")
        code = local.cmd_doctor(args, stack)
    say("")
    say("Next:")
    say("  ComfyUI:  %s" % stack.url)
    say("  pixel-studio-comfy status")
    say("  pixel-studio-comfy models pull <preset>   (see: pixel-studio-comfy models presets)")
    say("  Pixel Studio: add a ComfyUI source with the URL %s" % stack.url)
    return code


def register(sub):
    p = sub.add_parser("setup", help="check this machine, write .env and .secrets.env, start ComfyUI, run doctor")
    p.add_argument("--preset", metavar="NAME", help="model preset: pixel (default), svg, edit, video, all, none")
    p.add_argument("--models", metavar="a,b,c", help="workflow names to download models for (wins over --preset)")
    p.add_argument("--port", type=int, metavar="N", help="host port (PORT)")
    p.add_argument("--lan", action="store_true", help="listen on the network, not only this PC (BIND_ADDRESS=0.0.0.0)")
    p.add_argument("--project-name", metavar="NAME", help="Compose project name (COMPOSE_PROJECT_NAME)")
    p.add_argument("--output-dir", metavar="PATH", help="where outputs go (COMFY_OUTPUT_DIR)")
    p.add_argument("--civitai-token", metavar="TOKEN", help="fills an empty CIVITAI_TOKEN (else the environment variable)")
    p.add_argument("--hf-token", metavar="TOKEN", help="fills an empty HF_TOKEN (else the environment variable)")
    p.add_argument("--no-up", action="store_true", help="write the files, do not start the stack or run doctor")
    p.add_argument("-y", "--yes", action="store_true", help="never prompt; use the defaults")
    p.set_defaults(func=cmd_setup)
