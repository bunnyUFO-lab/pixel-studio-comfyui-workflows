#!/usr/bin/env python3
"""End the rented machine when the runner has exited. Standard library only.

RUNNER_ON_EXIT = none (default) | runpod-terminate | runpod-stop

For RunPod it needs RUNPOD_POD_ID and RUNPOD_API_KEY (RunPod puts both into every pod; the key is limited to that
pod). RUNPOD_REST_BASE defaults to https://rest.runpod.io/v1; only https:// or http://127.0.0.1: (tests) is accepted.
Prints only the mode and the HTTP status, never the key or a response body.

UNVERIFIED: the two RunPod calls below (DELETE /pods/{id} to terminate, POST /pods/{id}/stop to stop) were written
from memory and have not run against RunPod. ss-m10t10 and ss-m10t4 check them against RunPod's docs once a key exists.
"""
import os
import re
import sys
import time
import urllib.error
import urllib.request

MODES = ("none", "runpod-terminate", "runpod-stop")
DEFAULT_BASE = "https://rest.runpod.io/v1"
TRIES = 3


def call(mode, base, pod, key):
    """One request. Returns the HTTP status, or None when the server could not be reached."""
    if mode == "runpod-terminate":
        method, url = "DELETE", f"{base}/pods/{pod}"
    else:
        method, url = "POST", f"{base}/pods/{pod}/stop"
    req = urllib.request.Request(url, method=method, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, OSError):
        return None


def main():
    mode = os.environ.get("RUNNER_ON_EXIT", "none").strip() or "none"
    if mode not in MODES:
        print(f"shutdown: unknown RUNNER_ON_EXIT {mode!r} (use one of: {', '.join(MODES)})", file=sys.stderr)
        return 1
    if mode == "none":
        print("shutdown: mode none, nothing to do")
        return 0
    pod, key = os.environ.get("RUNPOD_POD_ID", ""), os.environ.get("RUNPOD_API_KEY", "")
    for name, value in (("RUNPOD_POD_ID", pod), ("RUNPOD_API_KEY", key)):
        if not value:
            print(f"shutdown: {name} is missing; the machine was not ended", file=sys.stderr)
            return 1
    if not re.fullmatch(r"[A-Za-z0-9]{6,40}", pod):
        print("shutdown: RUNPOD_POD_ID does not look like a pod id; the machine was not ended", file=sys.stderr)
        return 1
    base = (os.environ.get("RUNPOD_REST_BASE") or DEFAULT_BASE).rstrip("/")
    if not (base.startswith("https://") or base.startswith("http://127.0.0.1:")):
        print("shutdown: RUNPOD_REST_BASE must start with https:// (or http://127.0.0.1:)", file=sys.stderr)
        return 1
    try:
        wait = float(os.environ.get("RUNNER_SHUTDOWN_RETRY_S", "10"))
    except ValueError:
        wait = 10.0
    status = None
    for attempt in range(1, TRIES + 1):
        status = call(mode, base, pod, key)
        print(f"shutdown: mode {mode}, try {attempt}, HTTP {status if status is not None else 'no answer'}", flush=True)
        if status is not None and 200 <= status < 300:
            return 0
        if attempt < TRIES:
            time.sleep(wait)
    print("shutdown: the machine was not ended; end it in the RunPod console", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
