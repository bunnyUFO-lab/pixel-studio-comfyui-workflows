"""Pure helpers: no I/O beyond what is passed in. Everything here is unit-tested without Docker or a network."""
import re
from collections import namedtuple

DEFAULT_PORT = 8188
DEFAULT_COMFYUI_REF = "v0.38.2"
QUOTES = chr(34) + chr(39)

Check = namedtuple("Check", "status name detail")  # status: OK / WARN / FAIL / SKIP


def parse_env(text):
    """Parse KEY=VALUE lines: comments, blank lines, optional quotes, CRLF, an export prefix."""
    env = {}
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in QUOTES:
            value = value[1:-1]
        elif " #" in value:  # trailing comment on an unquoted value
            value = value.split(" #", 1)[0].rstrip()
        if key:
            env[key] = value
    return env


def base_url(env, override=None):
    """ComfyUI base URL: override > PORT from env > 8188. Always 127.0.0.1 (Windows tries ::1 first for localhost)."""
    if override:
        return override.rstrip("/")
    port = (env.get("PORT") or "").strip()
    if not port.isdigit():
        port = str(DEFAULT_PORT)
    return "http://127.0.0.1:%s" % port


def dockerfile_base_image(text):
    """The image of the first FROM line (skipping --platform flags)."""
    for line in text.replace("\r\n", "\n").split("\n"):
        parts = line.split()
        if len(parts) >= 2 and parts[0].upper() == "FROM":
            rest = [p for p in parts[1:] if not p.startswith("--")]
            if rest:
                return rest[0]
    return None


def compose_default(text, var, fallback=None):
    """The ${VAR:-default} default written in a compose file."""
    m = re.search(r"\$\{%s:-([^}]*)\}" % re.escape(var), text or "")
    return m.group(1) if m else fallback


def gateway_enabled(env):
    """PSC_GATEWAY=1 in .env turns on the Bearer-token gateway."""
    return (env.get("PSC_GATEWAY") or "").strip().lower() in ("1", "true", "yes", "on")


def compose_files(env):
    """The -f files for every compose call: none normally, the gateway override when the gateway is on."""
    return ["docker-compose.yml", "docker/gateway.compose.yml"] if gateway_enabled(env) else []


TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{32,}")


def valid_token(token):
    """A gateway token Caddy can match literally: 32+ of A-Za-z0-9_- (no wildcard, space, quote or brace)."""
    return bool(TOKEN_RE.fullmatch(token or ""))


def bearer_headers(token):
    return {"Authorization": "Bearer " + token} if token else {}


def summarize_queue(data):
    """From a /queue payload: counts and the first 5 shortened prompt ids of each list."""
    def ids(items):
        return [str(i[1])[:8] for i in items or [] if isinstance(i, (list, tuple)) and len(i) > 1]

    return {"running": len(data.get("queue_running") or []), "pending": len(data.get("queue_pending") or []),
            "running_ids": ids(data.get("queue_running"))[:5], "pending_ids": ids(data.get("queue_pending"))[:5]}


def summarize_system_stats(data):
    """From a /system_stats payload: ComfyUI version and the first device."""
    system = data.get("system") or {}
    devices = data.get("devices") or []
    out = {"version": system.get("comfyui_version"), "gpu": None, "vram_total": None, "vram_free": None}
    if devices:
        dev = devices[0]
        name = re.sub(r"^cuda:\d+\s+", "", str(dev.get("name") or "")).split(" : ")[0].strip()
        out.update(gpu=name or None, vram_total=dev.get("vram_total"), vram_free=dev.get("vram_free"))
    return out


def gb(n):
    return "%.1f GB" % (n / 1024.0 ** 3)


_MODEL_LINE = re.compile(r"^\s*\[(ok|missing|partial)\s*\]\s+(\S+)")


def parse_models_list(text):
    """Parse `models list` output into (ok, missing, partial, names of the models that are not ok)."""
    ok = missing = partial = 0
    names = []
    for line in text.replace("\r\n", "\n").split("\n"):
        m = _MODEL_LINE.match(line)
        if not m:
            continue
        state, name = m.groups()
        if state == "ok":
            ok += 1
        else:
            names.append(name)
            if state == "missing":
                missing += 1
            else:
                partial += 1
    return ok, missing, partial, names


def doctor_exit_code(checks):
    return 1 if any(c.status == "FAIL" for c in checks) else 0


def format_checks(checks):
    return "\n".join("[%-4s] %s: %s" % (c.status, c.name, c.detail) for c in checks)


def group(subparsers, name, help=None):
    """Return the subparser group `name`, creating it on first use (so several modules can add to one group)."""
    cache = subparsers.__dict__.setdefault("_psc_groups", {})
    if name not in cache:
        parser = subparsers.add_parser(name, help=help)
        cache[name] = parser.add_subparsers(dest=name + "_cmd", metavar="SUBCOMMAND")
        cache[name].required = True
    return cache[name]
