#!/usr/bin/env python3
"""Small client for Pixel Studio (its UI may still say Sprite Studio). Python standard library only.

Base URL: --server, else $PIXEL_STUDIO_URL, else DEFAULT_URL below. JSON on stdout.
Exit codes: 0 ok, 1 error (tool or HTTP), 2 Pixel Studio not reachable.

  studio.py status
  studio.py call <tool> [--arg k=v ...] [--json JSON]        # MCP tools/call (same tools as the MCP server)
  studio.py import <subject-ref> <file> [--kind auto|image|sprite|sheet]   # local file -> asset (REST import)
  studio.py download <asset-ref> --out DIR                   # MCP get, then fetch each file
  studio.py wait <job_id> [--timeout 900]                    # loop wait_for_job until the job is over
"""
import argparse
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

DEFAULT_URL = "http://localhost:8190"


class Unreachable(Exception):
    pass


class Failed(Exception):
    pass


def base_url(args):
    return (args.server or os.environ.get("PIXEL_STUDIO_URL") or DEFAULT_URL).rstrip("/")


def http(url, data=None, headers=None, timeout=60):
    """Return (status, body bytes). No Origin header is sent: the MCP origin guard refuses foreign origins."""
    req = urllib.request.Request(url, data=data, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except (urllib.error.URLError, OSError) as e:
        raise Unreachable(str(getattr(e, "reason", e)))


def get_json(url, timeout=60):
    code, body = http(url, timeout=timeout)
    if code >= 400:
        raise Failed("HTTP %s from %s: %s" % (code, url, body[:300].decode("utf-8", "replace")))
    return json.loads(body)


def rpc_call(base, tool, arguments, timeout=960):
    msg = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": tool, "arguments": arguments}}
    code, body = http(base + "/mcp", json.dumps(msg).encode(), {"Content-Type": "application/json"}, timeout)
    try:
        r = json.loads(body)
    except ValueError:
        raise Failed("HTTP %s from /mcp: %s" % (code, body[:300].decode("utf-8", "replace")))
    if "error" in r:
        raise Failed(r["error"].get("message", str(r["error"])))
    res = r["result"]
    text = "\n".join(c.get("text", "") for c in res.get("content", []) if c.get("type") == "text")
    if res.get("isError"):
        raise Failed(text or "tool error")
    return res.get("structuredContent"), text


def parse_value(v):
    try:
        return json.loads(v)
    except ValueError:
        return v


def multipart(fields, files):
    """Build a multipart/form-data body by hand: fields {name: str}, files {name: (filename, bytes, type)}."""
    b = uuid.uuid4().hex
    out = []
    for k, v in fields.items():
        out.append(('--%s\r\nContent-Disposition: form-data; name="%s"\r\n\r\n%s\r\n' % (b, k, v)).encode())
    for k, (fn, data, ct) in files.items():
        out.append(('--%s\r\nContent-Disposition: form-data; name="%s"; filename="%s"\r\nContent-Type: %s\r\n\r\n'
                    % (b, k, fn, ct)).encode() + data + b"\r\n")
    out.append(("--%s--\r\n" % b).encode())
    return b"".join(out), "multipart/form-data; boundary=" + b


def cmd_status(args):
    base = base_url(args)
    try:
        s = get_json(base + "/api/status", timeout=5)
    except Unreachable as e:
        print(json.dumps({"ok": False, "url": base, "error": e.args[0]}))
        return 2
    srcs = [{"name": x.get("name"), "label": x.get("label"), "state": x.get("state")} for x in s.get("sources") or []]
    print(json.dumps({"ok": True, "url": base, "run_on": s.get("run_on"), "sources": srcs}))
    return 0


def cmd_call(args):
    base = base_url(args)
    a = json.loads(args.json) if args.json else {}
    for kv in args.arg or []:
        k, _, v = kv.partition("=")
        a[k] = parse_value(v)
    _, text = rpc_call(base, args.tool, a)
    print(text)
    return 0


def cmd_import(args):
    base = base_url(args)
    sub = get_json(base + "/api/resolve?path=" + urllib.parse.quote(args.subject, safe="/"))["subject"]["id"]
    with open(args.file, "rb") as f:
        data = f.read()
    name = os.path.basename(args.file)
    ct = mimetypes.guess_type(name)[0] or "application/octet-stream"
    body, ctype = multipart({"kind": args.kind}, {"file": (name, data, ct)})
    code, resp = http("%s/api/subjects/%s/import" % (base, sub), body, {"Content-Type": ctype})
    if code >= 400:
        raise Failed("import failed (HTTP %s): %s" % (code, resp[:300].decode("utf-8", "replace")))
    asset = json.loads(resp)
    ref = None
    try:
        ref = (get_json(base + "/api/resolve?path=" + urllib.parse.quote(asset["id"], safe="/")).get("asset") or {}).get("ref")
    except Failed:
        pass
    out = {"id": asset["id"], "kind": asset.get("kind"), "ref": ref}
    if name.lower().endswith((".gif", ".webp")) and asset.get("kind") != "video":
        out["note"] = "an animated picture lands as a reference, not a video: convert a motion clip to mp4 and import that"
    print(json.dumps(out))
    return 0


def cmd_download(args):
    base = base_url(args)
    data, text = rpc_call(base, "get", {"ref": args.ref})
    data = data or json.loads(text)
    os.makedirs(args.out, exist_ok=True)
    saved = []
    for name, url in (data.get("files") or {}).items():
        path = urllib.parse.urlsplit(url).path       # re-base on our own URL: the server may not know its public address
        code, body = http(base + path)
        if code >= 400:
            raise Failed("download of %s failed (HTTP %s)" % (name, code))
        dest = os.path.join(args.out, os.path.basename(name))
        with open(dest, "wb") as f:
            f.write(body)
        saved.append(dest.replace("\\", "/"))
    print(json.dumps({"ref": args.ref, "saved": saved}))
    return 0


def cmd_wait(args):
    base = base_url(args)
    end = time.time() + args.timeout
    while True:
        left = int(end - time.time())
        data, text = rpc_call(base, "wait_for_job", {"job_id": args.job_id, "timeout_seconds": max(1, min(300, left))})
        data = data or json.loads(text)
        if data.get("status") not in ("queued", "running"):
            print(text)
            return 0 if data.get("status") == "done" else 1
        if time.time() >= end:
            print(text)
            print("timeout: the job is still running; do not resubmit it, wait again", file=sys.stderr)
            return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", help="Pixel Studio URL (default $PIXEL_STUDIO_URL or " + DEFAULT_URL + ")")
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("status").set_defaults(fn=cmd_status)
    c = sp.add_parser("call")
    c.add_argument("tool")
    c.add_argument("--arg", action="append", help="k=v (v parsed as JSON when possible)")
    c.add_argument("--json", help="all arguments as a JSON object")
    c.set_defaults(fn=cmd_call)
    i = sp.add_parser("import")
    i.add_argument("subject")
    i.add_argument("file")
    i.add_argument("--kind", default="image", choices=["auto", "image", "sprite", "sheet"])
    i.set_defaults(fn=cmd_import)
    d = sp.add_parser("download")
    d.add_argument("ref")
    d.add_argument("--out", required=True)
    d.set_defaults(fn=cmd_download)
    w = sp.add_parser("wait")
    w.add_argument("job_id")
    w.add_argument("--timeout", type=int, default=900)
    w.set_defaults(fn=cmd_wait)
    args = ap.parse_args()
    try:
        return args.fn(args)
    except Unreachable as e:
        print(json.dumps({"ok": False, "url": base_url(args), "error": e.args[0]}))
        return 2
    except Failed as e:
        print("error: %s" % e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
