"""token: show or rotate the gateway Bearer token (PSC_TOKEN in .secrets.env)."""
import re
import secrets
import sys

from psc import core, local
from psc.stack import CliError

KEY = "PSC_TOKEN"


def new_token():
    return secrets.token_urlsafe(32)


def set_secret(text, key, value):
    """Replace the value of an active KEY= line, or append the line. Comments and line endings are kept."""
    pattern = re.compile(r"^(\s*(?:export\s+)?%s\s*=).*$" % re.escape(key))
    ending = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines()
    replaced = False
    for i, line in enumerate(lines):
        m = pattern.match(line)
        if m and not replaced:
            lines[i] = "%s%s" % (m.group(1), value)
            replaced = True
    if not replaced:
        lines.append("%s=%s" % (key, value))
    return ending.join(lines) + ending


def current_token(stack):
    text = stack.read_text(".secrets.env")
    return core.parse_env(text).get(KEY, "") if text else ""


def ensure_token(stack):
    """The token from .secrets.env; created (and the file too, from the example) when empty."""
    token = current_token(stack)
    if token:
        if not core.valid_token(token):
            raise CliError("PSC_TOKEN in .secrets.env is not valid (it needs 32 or more of A-Z a-z 0-9 _ -): "
                           "run pixel-studio-comfy token rotate")
        return token
    text = stack.read_text(".secrets.env")
    if text is None:
        text = stack.read_text(".secrets.env.example") or ""
    token = new_token()
    stack.write_text(".secrets.env", set_secret(text, KEY, token), private=True)
    return token


def cmd_show(args, stack):
    print(ensure_token(stack), flush=True)
    return 0


def cmd_rotate(args, stack):
    text = stack.read_text(".secrets.env")
    if text is None:
        text = stack.read_text(".secrets.env.example") or ""
    token = new_token()
    stack.write_text(".secrets.env", set_secret(text, KEY, token), private=True)
    if stack.gateway and stack.service_state("gateway") == "running":
        # restart would keep the old environment: the container must be recreated to read .secrets.env again
        code = stack.compose("up", "-d", "--no-deps", "--force-recreate", "gateway", capture=True).returncode
        if code:
            raise CliError("the new token is saved but the gateway could not be restarted; run: pixel-studio-comfy up")
    elif stack.gateway:
        print("note: the gateway is not running; the new token is saved and the next `pixel-studio-comfy up` uses it.", file=sys.stderr)
    else:
        print("note: the gateway is off (PSC_GATEWAY=1 is not in .env); the new token is saved for when it is on.", file=sys.stderr)
    print(token, flush=True)
    return 0


def register(sub):
    g = core.group(sub, "token", help="show or rotate the gateway Bearer token")
    p = g.add_parser("show", help="print PSC_TOKEN (created when empty)")
    p.set_defaults(func=cmd_show)
    p = g.add_parser("rotate", help="replace PSC_TOKEN and restart the gateway; prints the new token")
    p.set_defaults(func=cmd_rotate)
