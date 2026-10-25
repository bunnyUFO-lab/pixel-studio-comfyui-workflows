"""Pure helpers for presets, the .env and .secrets.env text, and the update refusal. Unit-tested without Docker."""
import re
from collections import namedtuple
from pathlib import Path

from psc import core

Preset = namedtuple("Preset", "name description models count")

ORDER = ["pixel", "svg", "edit", "video", "all"]
SECRET_KEYS = ("CIVITAI_TOKEN", "HF_TOKEN")


def preset_description(text):
    """The words after "# Preset:" on the first line, without the trailing "Use with:" hint."""
    first = (text or "").replace("\r\n", "\n").split("\n", 1)[0]
    m = re.match(r"\s*#\s*Preset:\s*(.*)", first)
    if not m:
        return ""
    desc = re.split(r"\s*Use with:", m.group(1))[0].strip()
    return desc.rstrip(".").strip()


def preset_models(text):
    """Workflow names listed in the MODELS line of a preset (["all"] for MODELS=all)."""
    value = core.parse_env(text or "").get("MODELS", "")
    return [m.strip() for m in value.split(",") if m.strip()]


def workflow_names(root):
    """Workflow names (path under workflows/ without .json), sorted."""
    base = Path(root) / "workflows"
    if not base.is_dir():
        return []
    return sorted(p.relative_to(base).with_suffix("").as_posix() for p in base.rglob("*.json")
                  if not p.name.startswith("."))


def list_presets(root):
    """Presets of a repo: each env/*.env by file stem, plus "all" (.env.all). Known names first, in a fixed order."""
    root = Path(root)
    found = {}
    files = list((root / "env").glob("*.env")) if (root / "env").is_dir() else []
    files = [(p.stem, p) for p in files]
    if (root / ".env.all").is_file():
        files.append(("all", root / ".env.all"))
    everything = len(workflow_names(root))
    for name, path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        models = preset_models(text)
        count = everything if models == ["all"] else len(models)
        found[name] = Preset(name, preset_description(text), models, count)
    names = [n for n in ORDER if n in found] + sorted(n for n in found if n not in ORDER)
    return [found[n] for n in names]


def render_env(preset_text, overrides):
    """A new .env: the preset text, with the overrides in a block at the end.

    An active assignment of an overridden key in the preset is dropped first, so each key appears once and the
    result does not depend on which duplicate Compose would honour.
    """
    lines = (preset_text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    keep = []
    for line in lines:
        body = line.strip()
        if body.startswith("export "):
            body = body[len("export "):].lstrip()
        key = body.split("=", 1)[0].strip() if "=" in body and not body.startswith("#") else None
        if key in overrides:
            continue
        keep.append(line)
    text = "\n".join(keep).rstrip("\n") + "\n"
    if overrides:
        text += "\n# Set by pixel-studio-comfy setup\n"
        for key, value in overrides.items():
            text += "%s=%s\n" % (key, value)
    return text


_EMPTY_ASSIGN = re.compile(r"^(\s*(?:export\s+)?)(\w+)(\s*=\s*)(?:\"\"|'')?\s*$")


def fill_secrets(text, values):
    """Fill only empty KEY= lines whose key has a non-empty value. Returns (new text, filled keys).

    Comments, other lines and the line endings (LF or CRLF) are kept. A non-empty value is never changed.
    """
    filled = []
    out = []
    for line in text.splitlines(True):
        ending = line[len(line.rstrip("\r\n")):]
        m = _EMPTY_ASSIGN.match(line.rstrip("\r\n"))
        if m and values.get(m.group(2)):
            out.append("%s%s%s%s%s" % (m.group(1), m.group(2), m.group(3), values[m.group(2)], ending))
            filled.append(m.group(2))
        else:
            out.append(line)
    return "".join(out), filled


def update_refusal(porcelain, upstream):
    """Why `update` must not run (a sentence), or None. porcelain: `git status --porcelain --untracked-files=no`."""
    if (porcelain or "").strip():
        return "the checkout has uncommitted changes to tracked files; commit or stash them first, then run update again."
    if not (upstream or "").strip():
        return "the current branch has no upstream to pull from; check out a tracking branch such as main."
    return None
