#!/usr/bin/env python3
"""End the rented machine when the runner has exited. Standard library only.

RUNNER_ON_EXIT = none (default) | runpod-terminate | runpod-stop

For RunPod it needs RUNPOD_POD_ID and RUNPOD_API_KEY (RunPod puts both into every pod; the key is limited to that
pod). RUNPOD_REST_BASE defaults to https://rest.runpod.io/v1; only https:// or http://127.0.0.1: (tests) is accepted.
Prints only the mode and the HTTP status, never the key or a response body.

Two ways to end the machine, tried one after the other: the REST call, then `runpodctl` (RUNNER_RUNPODCTL, default
`runpodctl`, which ships in RunPod pods and uses the pod's own key from its environment; nothing secret goes on a
command line). `runpodctl` is tried in its current form (`runpodctl pod remove <id>`, `runpodctl pod stop <id>`) and, if
that fails, in the legacy form (`runpodctl remove pod <id>`, `runpodctl stop pod <id>`). If both methods fail this script
does NOT exit: it keeps trying for ever, first after RUNNER_SHUTDOWN_HOLD_S seconds (default 60), then with the wait
doubling up to RUNNER_SHUTDOWN_HOLD_MAX_S (default 300). A container that exits is restarted by RunPod and would loop
(and bill) for ever; one that never exits cannot, and retries cost nothing next to a GPU hour. The same holds for a missing key or pod id. The app is
the third line of defence (it ends any pod of ours that is past its deadline). RUNNER_SHUTDOWN_HOLD=0 turns the holding
off (the script then exits 1 as soon as the tries are used up; for tests and a local `docker run`).

UNVERIFIED: the RunPod calls below (DELETE /pods/{id} to terminate, POST /pods/{id}/stop to stop, and
`runpodctl pod remove <id>` / `runpodctl pod stop <id>` and the legacy forms) were written from memory and have not run against RunPod.
ss-m10t10 and ss-m10t4 check them against RunPod's docs once a key exists.
"""
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

MODES = ("none", "runpod-terminate", "runpod-stop")
DEFAULT_BASE = "https://rest.runpod.io/v1"
TRIES = 3


def number(name, default):
    try:
        v = float(os.environ.get(name, "") or default)
        return v if 0 <= v < 1e9 else default
    except ValueError:
        return default


def call(mode, base, pod, key):
    """One REST request. Returns the HTTP status, or None when the server could not be reached."""
    if mode == "runpod-terminate":
        method, url = "DELETE", f"{base}/pods/{pod}"
    else:
        method, url = "POST", f"{base}/pods/{pod}/stop"
    req = urllib.request.Request(url, method=method, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=number("RUNNER_SHUTDOWN_TIMEOUT_S", 30)) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, OSError):
        return None


def call_ctl(mode, pod):
    """The second method: runpodctl. Returns its exit code, or None when it is not there. No secret on the command line."""
    exe = os.environ.get("RUNNER_RUNPODCTL") or "runpodctl"
    path = shutil.which(exe)
    if not path:
        return None
    verb = "remove" if mode == "runpod-terminate" else "stop"
    code = None
    # current form first, the legacy verb-first form second (it may be dropped by a newer runpodctl)
    for argv in ([path, "pod", verb, pod], [path, verb, "pod", pod]):
        try:
            code = subprocess.run(argv, capture_output=True, timeout=60).returncode
        except (OSError, subprocess.SubprocessError):
            code = None
        if code == 0:
            return 0
    return code


def try_both(mode, base, pod, key, label):
    """One round: the REST call, then runpodctl. True when either one worked."""
    status = call(mode, base, pod, key)
    print(f"shutdown: mode {mode}, {label}, HTTP {status if status is not None else 'no answer'}", flush=True)
    if status is not None and 200 <= status < 300:
        return True
    code = call_ctl(mode, pod)
    print(f"shutdown: mode {mode}, {label}, runpodctl {'not found' if code is None else 'exit ' + str(code)}", flush=True)
    return code == 0


def backoff(n, first, cap):
    """Seconds to wait before hold retry number n (1, 2, ...): `first`, doubling, never above `cap` (and never below `first`)."""
    return min(first * (2 ** min(n - 1, 30)), max(cap, first))


def hold(message):
    """Say why, then never exit (unless RUNNER_SHUTDOWN_HOLD=0): an exited container is restarted and bills again."""
    print(message, file=sys.stderr, flush=True)
    if os.environ.get("RUNNER_SHUTDOWN_HOLD", "1") == "0":
        return 1
    print("shutdown: holding (not exiting) so RunPod does not restart this container; end the pod in the console", file=sys.stderr, flush=True)
    while True:
        time.sleep(3600)


def main():
    mode = os.environ.get("RUNNER_ON_EXIT", "none").strip() or "none"
    if mode not in MODES:
        return hold(f"shutdown: unknown RUNNER_ON_EXIT {mode!r} (use one of: {', '.join(MODES)})")
    if mode == "none":
        print("shutdown: mode none, nothing to do")
        return 0
    pod, key = os.environ.get("RUNPOD_POD_ID", ""), os.environ.get("RUNPOD_API_KEY", "")
    for name, value in (("RUNPOD_POD_ID", pod), ("RUNPOD_API_KEY", key)):
        if not value:
            return hold(f"shutdown: {name} is missing; the machine was not ended")
    if not re.fullmatch(r"[A-Za-z0-9]{6,40}", pod):
        return hold("shutdown: RUNPOD_POD_ID does not look like a pod id; the machine was not ended")
    base = (os.environ.get("RUNPOD_REST_BASE") or DEFAULT_BASE).rstrip("/")
    if not (base.startswith("https://") or base.startswith("http://127.0.0.1:")):
        return hold("shutdown: RUNPOD_REST_BASE must start with https:// (or http://127.0.0.1:)")
    wait = number("RUNNER_SHUTDOWN_RETRY_S", 10)
    for attempt in range(1, TRIES + 1):
        if try_both(mode, base, pod, key, f"try {attempt}"):
            return 0
        if attempt < TRIES:
            time.sleep(wait)
    if os.environ.get("RUNNER_SHUTDOWN_HOLD", "1") == "0":
        print("shutdown: the machine was not ended; end it in the RunPod console", file=sys.stderr)
        return 1
    # Both methods failed: keep trying for ever, with a capped back-off. Never exit.
    first, cap, n = number("RUNNER_SHUTDOWN_HOLD_S", 60), number("RUNNER_SHUTDOWN_HOLD_MAX_S", 300), 0
    print("shutdown: the machine was not ended; trying again until it is (this script does not exit)", file=sys.stderr, flush=True)
    while True:
        n += 1
        time.sleep(backoff(n, first, cap))
        if try_both(mode, base, pod, key, f"hold try {n}"):
            return 0

if __name__ == "__main__":
    sys.exit(main())
