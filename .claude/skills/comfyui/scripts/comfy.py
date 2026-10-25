#!/usr/bin/env python3
"""Drive a local ComfyUI from the command line (stdlib only; works on Windows/macOS/Linux).

  python comfy.py status                         server, GPU/VRAM, queue
  python comfy.py list                           workflows saved in ComfyUI (= repo workflows/)
  python comfy.py inspect <workflow>             editable settings per node title (and the node's id/type)
  python comfy.py run <workflow> [--set "Node title.widget=value" ...] [--input "Node title=path/to/file" ...]
                     [--out DIR] [--timeout SEC] [--no-wait] [--dump-api FILE]
  python comfy.py queue | interrupt | free        queue state, stop the running job, unload models / free VRAM
  python comfy.py result <prompt_id> [--out DIR]  outputs (files + text) of a finished job

<workflow> is a path under workflows/ without .json (e.g. pixel-art/image-to-pixel) or a local .json file.
Nodes are matched by title first, then by type ("KSampler") or id ("#12"). Values are parsed as JSON when
possible (numbers, true/false), otherwise used as strings. Server: --server or COMFY_URL (default http://127.0.0.1:8188).
"""
import argparse
import copy
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

SERVER = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
FRONTEND_ONLY = {"Note", "MarkdownNote", "PrimitiveNode", "Reroute", "PrimitiveStringMultiline_Frontend"}
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
FILE_WIDGETS = ("image", "video", "audio", "file")       # set with --input "Node title=path" (uploads the file)
UI_ONLY_KEYS = {"videopreview", "choose video to upload", "choose file to upload", "upload"}
SPECIAL_DEFAULTS = {"IMAGECOMPARE": {"__value__": ["", ""]}}     # frontend-only widget types and their default value


def format_dates(value):
    """Replace %date:yyyyMMdd_hhmmss% placeholders like the frontend does before queueing."""
    if not isinstance(value, str) or "%date:" not in value:
        return value
    import re
    now = time.localtime()
    rep = {"yyyy": "%04d" % now.tm_year, "yy": "%02d" % (now.tm_year % 100), "MM": "%02d" % now.tm_mon, "dd": "%02d" % now.tm_mday,
           "hh": "%02d" % now.tm_hour, "mm": "%02d" % now.tm_min, "ss": "%02d" % now.tm_sec}

    def fmt(m):
        out = m.group(1)
        for k in ("yyyy", "yy", "MM", "dd", "hh", "mm", "ss"):
            out = out.replace(k, rep[k])
        return out
    return re.sub(r"%date:([^%]+)%", fmt, value)


# ----------------------------------------------------------------------------- HTTP
def http(path, data=None, method=None, raw=False):
    req = urllib.request.Request(SERVER + path, data=data, method=method)
    if data is not None and not isinstance(data, bytes):
        req.data = json.dumps(data).encode()
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            body = r.read()
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code} on {path}: {e.read().decode(errors='replace')[:3000]}")
    except urllib.error.URLError as e:
        raise SystemExit(f"cannot reach ComfyUI at {SERVER}: {e.reason}")
    return body if raw else (json.loads(body) if body else None)


def upload(path):
    boundary = uuid.uuid4().hex
    name = os.path.basename(path)
    ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
    with open(path, "rb") as fh:
        content = fh.read()
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
            f"Content-Type: {ctype}\r\n\r\n").encode() + content + \
           f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(SERVER + "/upload/image", data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(req, timeout=300) as r:
        res = json.loads(r.read())
    return (res["subfolder"] + "/" if res.get("subfolder") else "") + res["name"]


# ----------------------------------------------------------------------------- workflows
def load_workflow(ref):
    if os.path.isfile(ref):
        with open(ref, encoding="utf-8") as fh:
            return json.load(fh)
    rel = ref if ref.endswith(".json") else ref + ".json"
    return http("/api/userdata/" + urllib.parse.quote("workflows/" + rel, safe=""))


def input_specs(info):
    """Ordered [(name, type, options)] of a node class's inputs from /object_info."""
    out = []
    for section in ("required", "optional"):
        for name, spec in (info["input"].get(section) or {}).items():
            typ = spec[0] if isinstance(spec[0], str) else "COMBO"
            opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            out.append((name, typ, opts))
    return out


def widget_names(info):
    """Widget slots in the order the frontend stores widgets_values (incl. control_after_generate and dynamic combos)."""
    names = []
    for name, typ, opts in input_specs(info):
        if opts.get("forceInput"):
            continue
        if typ == "COMFY_DYNAMICCOMBO_V3":
            names.append((name, "dynamic", opts))
            continue
        if typ in WIDGET_TYPES or typ.startswith("COMBO") or isinstance(typ, list) or opts.get("widgetType") in WIDGET_TYPES:
            names.append((name, typ, opts))
            if opts.get("control_after_generate") or (typ == "INT" and name in ("seed", "noise_seed")):
                names.append((None, "control", {}))
    return names


def map_widgets(node, info):
    """widgets_values -> {input_name: value} for one UI node."""
    vals = node.get("widgets_values")
    if vals is None:
        return {}
    if isinstance(vals, dict):                                   # e.g. VideoHelperSuite (format-dependent keys too)
        return {k: v for k, v in vals.items() if k not in UI_ONLY_KEYS}
    out, i = {}, 0
    for name, typ, opts in widget_names(info):
        if i >= len(vals):
            break
        if typ == "control":
            i += 1
            continue
        if typ == "dynamic":
            choice = vals[i]
            out[name] = choice
            i += 1
            for opt in opts.get("options", []):
                if opt.get("key") == choice:
                    for sub_section in ("required", "optional"):
                        for sub, sspec in (opt.get("inputs", {}).get(sub_section) or {}).items():
                            if i < len(vals):
                                out[f"{name}.{sub}"] = vals[i]
                                i += 1
            continue
        out[name] = vals[i]
        i += 1
    return out


def to_api(wf, object_info):
    nodes = {n["id"]: n for n in wf["nodes"]}
    links = {}
    for l in wf.get("links", []):
        if isinstance(l, dict):
            links[l["id"]] = (l["origin_id"], l["origin_slot"], l.get("type"))
        else:
            links[l[0]] = (l[1], l[2], l[5] if len(l) > 5 else None)
    if wf.get("definitions", {}).get("subgraphs"):
        print("warning: subgraphs are not expanded by comfy.py; use a workflow without subgraphs", file=sys.stderr)

    def resolve(link_id, want_type=None, depth=0):
        """Follow a link to a real (executing) node output: through Reroute, bypassed nodes and PrimitiveNode."""
        if link_id is None or link_id not in links or depth > 50:
            return None
        src, slot, typ = links[link_id]
        n = nodes.get(src)
        if n is None:
            return None
        if n["type"] == "Reroute":
            return resolve(n["inputs"][0].get("link"), typ, depth + 1)
        if n["type"] == "PrimitiveNode":
            return ("literal", (n.get("widgets_values") or [None])[0])
        if n.get("mode") == 2:                                       # muted: no value
            return None
        if n.get("mode") == 4:                                       # bypassed: pass the matching input through
            out_type = (n.get("outputs") or [{}])[slot].get("type") if slot < len(n.get("outputs") or []) else typ
            for inp in n.get("inputs", []):
                if inp.get("link") is not None and inp.get("type") == out_type:
                    return resolve(inp["link"], out_type, depth + 1)
            return None
        return ("link", [str(src), slot])

    api = {}
    for nid, n in nodes.items():
        if n["type"] in FRONTEND_ONLY or n.get("mode") in (2, 4):
            continue
        info = object_info.get(n["type"])
        if info is None:
            raise SystemExit(f"node type '{n['type']}' (#{nid} {n.get('title', '')}) is not installed in ComfyUI")
        inputs = {k: format_dates(v) for k, v in map_widgets(n, info).items()}
        for name, typ, opts in input_specs(info):
            if typ in SPECIAL_DEFAULTS and name not in inputs:
                inputs[name] = copy.deepcopy(SPECIAL_DEFAULTS[typ])
        for inp in n.get("inputs", []):
            if inp.get("link") is None:
                continue
            r = resolve(inp["link"], inp.get("type"))
            name = inp.get("widget", {}).get("name") or inp["name"]
            if r is None:
                inputs.pop(name, None)
            elif r[0] == "literal":
                inputs[name] = r[1]
            else:
                inputs[name] = r[1]
        api[str(nid)] = {"class_type": n["type"], "inputs": inputs, "_meta": {"title": n.get("title") or n["type"]}}
    return api


def find_nodes(api_or_wf_nodes, ref):
    """Match by title, then type, then '#id', then title ignoring case."""
    items = api_or_wf_nodes
    if ref.startswith("#"):
        return [k for k in items if k == ref[1:]]
    by_title = [k for k, v in items.items() if v["_meta"]["title"] == ref]
    by_type = [k for k, v in items.items() if v["class_type"] == ref]
    return by_title or by_type or [k for k, v in items.items() if v["_meta"]["title"].casefold() == ref.strip().casefold()]


def node_titles(api, keys=None):
    """Quoted titles of the nodes that have literal inputs (optionally only those with one of `keys`), for errors."""
    return ", ".join(f"'{n['_meta']['title']}'" for n in api.values()
                     if any(not isinstance(v, list) and (keys is None or k in keys) for k, v in n["inputs"].items()))


def parse_value(text):
    try:
        return json.loads(text)
    except ValueError:
        return text


# ----------------------------------------------------------------------------- outputs
def collect(prompt_id, out_dir=None):
    h = http(f"/history/{prompt_id}").get(prompt_id)
    if not h:
        return None
    status = h.get("status", {})
    result = {"status": status.get("status_str"), "files": [], "text": {}, "errors": []}
    for m in status.get("messages", []):
        if m[0] == "execution_error":
            result["errors"].append(f"{m[1].get('node_type')}: {m[1].get('exception_message')}")
    titles = {k: v.get("_meta", {}).get("title", k) for k, v in (h.get("prompt", [None, None, {}])[2] or {}).items()}
    for nid, out in h.get("outputs", {}).items():
        for key in ("images", "gifs", "videos", "audio"):
            for f in out.get(key, []) or []:
                if not isinstance(f, dict) or "filename" not in f:
                    continue
                entry = {"node": titles.get(nid, nid), "filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": f.get("type", "output")}
                if out_dir and entry["type"] in ("output", "temp"):      # temp = previews (e.g. animation playback)
                    os.makedirs(out_dir, exist_ok=True)
                    q = urllib.parse.urlencode({"filename": entry["filename"], "subfolder": entry["subfolder"], "type": entry["type"]})
                    dest = os.path.join(out_dir, entry["filename"])
                    with open(dest, "wb") as fh:
                        fh.write(http("/view?" + q, raw=True))
                    entry["local"] = dest
                result["files"].append(entry)
        if out.get("text"):
            result["text"][titles.get(nid, nid)] = "\n".join(map(str, out["text"]))
    return result


# ----------------------------------------------------------------------------- commands
def cmd_status(_):
    s = http("/system_stats")
    q = http("/queue")
    d = (s.get("devices") or [{}])[0]
    print(f"ComfyUI {s['system'].get('comfyui_version')} | {d.get('name', '?')[:48]} | "
          f"VRAM free {d.get('vram_free', 0) / 1e9:.1f}/{d.get('vram_total', 0) / 1e9:.1f} GB | "
          f"queue: {len(q['queue_running'])} running, {len(q['queue_pending'])} pending")


def cmd_list(_):
    for f in http("/api/userdata?dir=workflows&recurse=true"):
        if f.endswith(".json"):
            print(f[:-5])


def cmd_inspect(a):
    wf, oi = load_workflow(a.workflow), http("/object_info")
    api = to_api(wf, oi)
    for nid, n in sorted(api.items(), key=lambda kv: int(kv[0])):
        lits = {k: v for k, v in n["inputs"].items() if not (isinstance(v, list) and len(v) == 2 and isinstance(v[0], str))}
        if not lits:
            continue
        title = n["_meta"]["title"]
        print(f"#{nid} [{n['class_type']}] {title}      (--set \"{title}.<widget>=value\")")
        for k, v in lits.items():
            s = json.dumps(v)
            hint = f'   <- a file: --input "{title}=path/to/file"' if k in FILE_WIDGETS else ""
            print(f"    {k} = {s if len(s) < 100 else s[:97] + '...'}{hint}")


def cmd_run(a):
    if a.server:
        global SERVER
        SERVER = a.server
    wf, oi = load_workflow(a.workflow), http("/object_info")
    api = to_api(wf, oi)
    for spec in a.input or []:
        if "=" not in spec:
            raise SystemExit(f'--input: use --input "Node title=path" (got {spec!r})')
        ref, path = spec.split("=", 1)
        ids = find_nodes(api, ref)
        if not ids and ref.rsplit(".", 1)[-1] in FILE_WIDGETS:      # "Input image.image=path"
            ref = ref.rsplit(".", 1)[0]
            ids = find_nodes(api, ref)
        if not ids:
            raise SystemExit(f"--input: no node '{ref}'. Use --input \"Node title=path\"; file inputs here: "
                             f"{node_titles(api, FILE_WIDGETS)}")
        node = api[ids[0]]
        key = next((k for k in FILE_WIDGETS if k in node["inputs"]), None)
        if key is None:
            raise SystemExit(f"--input: node '{ref}' has no image/video/audio input")
        name = upload(path)
        node["inputs"][key] = name
        print(f"uploaded {path} -> {ref}.{key} = {name}")
    for spec in a.set or []:
        if "=" not in spec:
            raise SystemExit(f'--set: use --set "Node title.widget=value" (got {spec!r})')
        left, value = spec.split("=", 1)
        parts, ids, near = left.split("."), [], None
        for n in range(len(parts) - 1, 0, -1):          # dotted widgets too: "Segments.mode.num_iterations"
            ref, widget = ".".join(parts[:n]), ".".join(parts[n:])
            found = find_nodes(api, ref)
            if found and widget in api[found[0]]["inputs"]:
                ids = found
                break
            near = near or (found and ref)
        if near and not ids:
            widgets = ", ".join(k for k, v in api[find_nodes(api, near)[0]]["inputs"].items() if not isinstance(v, list))
            raise SystemExit(f"--set: node '{near}' has no widget '{left[len(near) + 1:]}'. Its widgets: {widgets} "
                             f"(e.g. --set \"{near}.{widgets.split(', ')[0]}=value\")")
        if not ids:
            raise SystemExit(f"--set: no node matches '{left}'. Use \"Node title.widget=value\" with a widget "
                             f"as `inspect` prints it under one of these titles: {node_titles(api)}")
        path = parse_value(value)
        if widget in FILE_WIDGETS and isinstance(path, str) and os.path.isfile(path):   # a local file: upload it
            name = upload(path)
            print(f"uploaded {path} -> {ref}.{widget} = {name}")
            value = json.dumps(name)
        for i in ids:
            api[i]["inputs"][widget] = parse_value(value)
    if a.dump_api:
        with open(a.dump_api, "w", encoding="utf-8") as fh:
            json.dump(api, fh, indent=1)
    res = http("/prompt", {"prompt": api, "client_id": uuid.uuid4().hex})
    if res.get("node_errors"):
        raise SystemExit("node errors: " + json.dumps(res["node_errors"], indent=1)[:3000])
    pid = res["prompt_id"]
    print(f"queued {pid}")
    if a.no_wait:
        return
    t0 = time.time()
    while time.time() - t0 < a.timeout:
        time.sleep(2)
        r = collect(pid, a.out)
        if r:
            print(f"{r['status']} in {time.time() - t0:.0f}s")
            for e in r["errors"]:
                print("ERROR", e)
            for f in r["files"]:
                print(f"file  [{f['node']}] {f.get('local') or f['subfolder'] + '/' + f['filename']}")
            for title, text in r["text"].items():
                print(f"text  [{title}]\n" + "\n".join("      " + line for line in text.splitlines()))
            sys.exit(0 if r["status"] == "success" else 1)
    raise SystemExit(f"timed out after {a.timeout}s (job {pid} may still be running)")


def cmd_result(a):
    r = collect(a.prompt_id, a.out)
    print(json.dumps(r, indent=1) if r else "not finished / unknown id")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("list")
    sub.add_parser("queue")
    sub.add_parser("interrupt")
    sub.add_parser("free")
    s = sub.add_parser("inspect"); s.add_argument("workflow")
    s = sub.add_parser("run"); s.add_argument("workflow")
    s.add_argument("--set", action="append"); s.add_argument("--input", action="append")
    s.add_argument("--out"); s.add_argument("--timeout", type=int, default=3600)
    s.add_argument("--no-wait", action="store_true"); s.add_argument("--dump-api")
    s.add_argument("--server", default=None)
    s = sub.add_parser("result"); s.add_argument("prompt_id"); s.add_argument("--out")
    a = p.parse_args()
    if a.server:
        global SERVER
        SERVER = a.server
    if a.cmd == "status":
        cmd_status(a)
    elif a.cmd == "list":
        cmd_list(a)
    elif a.cmd == "inspect":
        cmd_inspect(a)
    elif a.cmd == "run":
        cmd_run(a)
    elif a.cmd == "result":
        cmd_result(a)
    elif a.cmd == "queue":
        q = http("/queue")
        print(f"running: {[x[1] for x in q['queue_running']]}\npending: {[x[1] for x in q['queue_pending']]}")
    elif a.cmd == "interrupt":
        http("/interrupt", {}); print("interrupted")
    elif a.cmd == "free":
        http("/free", {"unload_models": True, "free_memory": True}); print("models unloaded, memory freed")


if __name__ == "__main__":
    main()
