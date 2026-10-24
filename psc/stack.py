"""The I/O side: running docker compose against the stack folder and asking ComfyUI over HTTP."""
import json
import os
import subprocess
import urllib.request
from pathlib import Path

from psc import core

# Compose gives the shell environment priority over .env, so these count when set there.
ENV_KEYS = ("PORT", "MODELS", "COMPOSE_PROJECT_NAME", "COMFYUI_REF", "COMFY_OUTPUT_DIR", "BIND_ADDRESS")


class CliError(Exception):
    pass


def run(args, cwd=None, capture=False, timeout=None):
    """subprocess.run with an argument list (never a shell). stdin is never inherited."""
    try:
        return subprocess.run(args, cwd=cwd, stdin=subprocess.DEVNULL, timeout=timeout,
                              stdout=subprocess.PIPE if capture else None,
                              stderr=subprocess.STDOUT if capture else None,
                              universal_newlines=True, encoding="utf-8", errors="replace")
    except FileNotFoundError:
        hint = " Install Docker Desktop (Windows) or Docker Engine (Linux)." if args[0] == "docker" else ""
        raise CliError("%s was not found on PATH.%s" % (args[0], hint))
    except subprocess.TimeoutExpired:
        raise CliError("%s did not answer within %s s." % (" ".join(args[:3]), timeout))


class Stack:
    def __init__(self, root, url=None):
        self.root = Path(root).resolve()
        self._url = url

    def path(self, *parts):
        return self.root.joinpath(*parts)

    def read_text(self, *parts):
        try:
            return self.path(*parts).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    def write_text(self, name, text, private=False):
        """Write a file in the stack folder with LF endings (Python 3.9 has no newline= on Path.write_text).
        private=True creates it readable by the owner only (0600; no effect on Windows) and tightens an existing file."""
        path = str(self.path(name))
        if private:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            f = os.fdopen(fd, "w", encoding="utf-8", newline="\n")
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
        else:
            f = open(path, "w", encoding="utf-8", newline="\n")
        with f:
            f.write(text)

    def read_env(self):
        """The stack .env, with the shell values for the keys compose also takes from the shell."""
        text = self.read_text(".env")
        env = core.parse_env(text) if text else {}
        for key in ENV_KEYS:
            if key in os.environ:
                env[key] = os.environ[key]
        return env

    @property
    def url(self):
        return core.base_url(self.read_env(), self._url)

    def compose_files(self):
        return []  # extra -f files (an override) go here

    def compose(self, *args, capture=False, timeout=None):
        cmd = ["docker", "compose"]
        for f in self.compose_files():
            cmd += ["-f", f]
        return run(cmd + list(args), cwd=str(self.root), capture=capture, timeout=timeout)

    def ps(self):
        """Rows of `docker compose ps --all --format json` (JSON object per line, or one array on older Compose)."""
        proc = self.compose("ps", "--all", "--format", "json", capture=True, timeout=30)
        text = proc.stdout.strip() if proc.returncode == 0 else ""
        if not text:
            return []
        try:
            data = json.loads(text)
            return data if isinstance(data, list) else [data]
        except ValueError:
            rows = []
            for line in text.splitlines():
                if line.strip().startswith("{"):
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        pass
            return rows

    def service_state(self, service="comfyui"):
        for row in self.ps():
            if row.get("Service") == service:
                return (row.get("State") or "").lower()
        return None

    def service_running(self, service="comfyui"):
        return self.service_state(service) == "running"

    def http_json(self, path, timeout=5, headers=None):
        return http_json(self.url, path, timeout, headers)


def http_json(url, path, timeout=5, headers=None):
    req = urllib.request.Request(url.rstrip("/") + path, headers=headers or {})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # local server: never via a proxy
    with opener.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))
