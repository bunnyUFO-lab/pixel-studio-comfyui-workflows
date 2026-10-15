"""List and download the models that workflows need.

Model requirements are read from the workflow JSON files themselves: each loader node
carries a `properties.models` list of {name, url, directory} (the same metadata ComfyUI's
"missing models" dialog uses).

    models [list] [WORKFLOW...]   show required models and whether they are present
    models download [WORKFLOW...]  download missing models (resumes partial files)

WORKFLOW is a path under workflows/ without ".json" (e.g. image-to-video/wan22-remix),
a bare file name, or "all". With no WORKFLOW, the MODELS env var is used (comma separated).
"""
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

WORKFLOWS = Path(os.environ.get("WORKFLOWS_DIR", "/opt/ComfyUI/user/default/workflows"))
MODELS = Path(os.environ.get("MODELS_DIR", "/opt/ComfyUI/models"))


def all_workflows():
    return sorted(p for p in WORKFLOWS.rglob("*.json") if not p.name.startswith("."))


def workflow_name(path):
    return path.relative_to(WORKFLOWS).with_suffix("").as_posix()


def select(names):
    if not names or names == ["all"]:
        return all_workflows()
    found = []
    for name in names:
        name = name.strip().removesuffix(".json")
        matches = [p for p in all_workflows() if workflow_name(p) == name or p.stem == name]
        if not matches:
            sys.exit(f"models: no workflow named '{name}' under workflows/")
        found += matches
    return found


def iter_nodes(wf):
    yield from wf.get("nodes", [])
    for sub in wf.get("definitions", {}).get("subgraphs", []):
        yield from sub.get("nodes", [])


def required_models(path):
    wf = json.loads(path.read_text(encoding="utf-8"))
    seen = {}
    for node in iter_nodes(wf):
        if node.get("mode") in (2, 4):  # muted / bypassed nodes don't need their models
            continue
        for m in node.get("properties", {}).get("models", []) or []:
            if not (m.get("name") and m.get("url") and m.get("directory")):
                continue
            rel = Path(m["directory"]) / m["name"]
            if rel.is_absolute() or ".." in rel.parts:
                sys.exit(f"models: unsafe path '{rel}' in {path.name}")
            seen[rel.as_posix()] = m["url"]
    return seen


def status(rel):
    target = MODELS / rel
    if target.with_name(target.name + ".aria2").exists() or target.with_name(target.name + ".part").exists():
        return "partial"
    return "ok" if target.exists() else "missing"


def cmd_list(paths):
    for path in paths:
        reqs = required_models(path)
        print(f"{workflow_name(path)}  ({len(reqs)} models)")
        if not reqs:
            print("  (no downloadable models listed - fine if it needs none; otherwise add properties.models to its loader nodes)")
        for rel in reqs:
            st = status(rel)
            size = f"{(MODELS / rel).stat().st_size / 1e9:6.2f} GB" if st == "ok" else ""
            print(f"  [{st:7}] {rel} {size}")


def cmd_download(paths):
    todo = {}
    for path in paths:
        todo.update(required_models(path))
    missing = {rel: url for rel, url in todo.items() if status(rel) != "ok"}
    print(f"models: {len(todo) - len(missing)}/{len(todo)} present, {len(missing)} to download")
    failed = []
    for rel, url in missing.items():
        target = MODELS / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"models: downloading {rel}", flush=True)
        if not download(url, target):
            print(f"models: FAILED {rel}", flush=True)
            failed.append(rel)
    if failed:
        sys.exit(f"models: {len(failed)} download(s) failed: {', '.join(failed)}")
    print("models: all present")


def download(url, target):
    host = urlparse(url).hostname or ""
    if host.endswith("civitai.com"):
        # Civitai needs an API key and redirects to a signed storage URL, so the key goes in the query
        # string (an auth header would be forwarded to the storage host). curl doesn't echo URLs on errors,
        # which keeps the key out of the logs.
        token = os.environ.get("CIVITAI_TOKEN")
        if not token:
            print("models: Civitai downloads need an API key - set CIVITAI_TOKEN in .secrets.env "
                  "(civitai.com > Account settings > API Keys)", flush=True)
            return False
        sep = "&" if "?" in url else "?"
        part = target.with_name(target.name + ".part")
        ok = subprocess.call(["curl", "-fL", "-sS", "-C", "-", "-o", str(part), f"{url}{sep}token={token}"]) == 0
        if ok:
            part.rename(target)
        return ok
    args = ["aria2c", "--console-log-level=warn", "--summary-interval=30", "-x", "8", "-s", "8", "-c",
            "-d", str(target.parent), "-o", target.name]
    token = os.environ.get("HF_TOKEN")
    if token and host in ("huggingface.co", "hf.co"):
        args.append(f"--header=Authorization: Bearer {token}")
    return subprocess.call(args + [url]) == 0


def main(argv):
    cmd = argv[0] if argv and argv[0] in ("list", "download") else "list"
    names = argv[1:] if argv and argv[0] == cmd else argv
    if not names:
        names = [n for n in os.environ.get("MODELS", "").split(",") if n.strip()] or ["all"]
    paths = select(names)
    (cmd_download if cmd == "download" else cmd_list)(paths)


if __name__ == "__main__":
    main(sys.argv[1:])
