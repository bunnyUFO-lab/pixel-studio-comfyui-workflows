"""ComfyUI job protocol "psc-job/v1": run one job on a ComfyUI process and return everything it wrote.

Provider-neutral: no `modal` import. The Modal app (cloud/modal_comfy.py) calls it today; a RunPod handler can import
this file unchanged. The protocol is described in docs/job-api.md.

A job is one ComfyUI API-format graph plus the input files it references (base64). `run_job` writes the inputs into
ComfyUI's input/ folder, queues the graph, waits for it, and returns the result document: the files ComfyUI's history
lists AND every other file that appeared or changed under output/ and temp/ during the job. Inputs and outputs are
deleted afterwards.

ONE JOB AT A TIME PER COMFYUI PROCESS. The new-file detection compares output/ and temp/ before and after the job, so
it would attribute another job's files to this one on a shared ComfyUI. Callers must not run two jobs on one process
(on Modal: no @modal.concurrent on the job class).

Only the standard library is required. Progress over ComfyUI's WebSocket needs `websockets` and is skipped without it
(the job still runs; the result is the same).
"""
import base64
import copy
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

PROTOCOL = "psc-job/v1"
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_OUTPUT_BYTES = 256 * 1024 * 1024
MAX_TIMEOUT = 3600
HISTORY_KEYS = ("images", "gifs", "videos", "audio")
# ComfyUI keeps an execution cache between prompts: an identical graph re-runs nothing, so its save nodes write no
# files while the history still lists them. A job must not get that. The cache stays on (loaders, encoders and
# samplers keep their models in VRAM); instead run_job adds a per-job MARKER_KEY input to every output node, which
# changes only that node's cache key, so output nodes always re-run. ComfyUI passes a node only the inputs it declares.
JOB_ARGS = ()
MARKER_KEY = "psc_job"


class InputTooLarge(ValueError):
    """The inputs exceed MAX_INPUT_BYTES (a transport answers 413 instead of 400)."""


# ----------------------------------------------------------------------------- documents
def valid_name(name):
    """A plain file name: no path separators, no '..', no NUL, no leading dot, at most 255 characters."""
    return (isinstance(name, str) and 0 < len(name) <= 255 and not name.startswith(".")
            and not any(c in name for c in "/\\\0") and ".." not in name)


def decode_job(doc):
    """The inputs of a job document as [(name, bytes)]. Raises ValueError on bad base64."""
    out = []
    for item in doc.get("inputs") or []:
        try:
            out.append((item["name"], base64.b64decode(item["data"], validate=True)))
        except (ValueError, TypeError, KeyError) as e:
            raise ValueError(f"input {item.get('name')!r}: data is not valid base64") from e
    return out


def validate_job(doc):
    """Raise ValueError naming the first problem; return the effective timeout in seconds."""
    if not isinstance(doc, dict):
        raise ValueError("job must be a JSON object")
    if doc.get("protocol") not in (None, PROTOCOL):
        raise ValueError(f"unsupported protocol {doc.get('protocol')!r} (this server speaks {PROTOCOL})")
    wf = doc.get("workflow")
    if not isinstance(wf, dict) or not wf:
        raise ValueError("workflow is missing or empty (an API-format graph is required)")
    for nid, node in wf.items():
        if not isinstance(node, dict) or not node.get("class_type"):
            raise ValueError(f"workflow node {nid!r} has no class_type (is this an API-format graph?)")
    total = 0
    names = set()
    inputs = doc.get("inputs") or []
    if not isinstance(inputs, list):
        raise ValueError("inputs must be a list")
    for item in inputs:
        if not isinstance(item, dict) or not isinstance(item.get("data"), str):
            raise ValueError("each input needs a name and base64 data")
        name = item.get("name")
        if not valid_name(name):
            raise ValueError(f"input name {name!r} is not a plain file name (no / \\ .. NUL, no leading dot, max 255)")
        if name in names:
            raise ValueError(f"input name {name!r} appears twice")
        names.add(name)
        total += len(item["data"]) * 3 // 4          # decoded size, estimated from the base64 length
    if total > MAX_INPUT_BYTES:
        raise InputTooLarge(f"inputs are {total // 2**20} MiB decoded; the limit is {MAX_INPUT_BYTES // 2**20} MiB")
    timeout = doc.get("timeout", MAX_TIMEOUT)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 1 <= timeout <= MAX_TIMEOUT:
        raise ValueError(f"timeout must be 1..{MAX_TIMEOUT} seconds")
    return timeout


def encode_result(result):
    """Result with file bytes -> JSON-ready result: each file's `data` becomes a base64 string."""
    out = dict(result)
    out["files"] = [dict(f, data=base64.b64encode(f["data"]).decode("ascii")) if isinstance(f.get("data"), bytes) else f
                    for f in result.get("files", [])]
    return out


def decode_result(result):
    """Inverse of encode_result."""
    out = dict(result)
    out["files"] = [dict(f, data=base64.b64decode(f["data"])) for f in result.get("files", [])]
    return out


# ----------------------------------------------------------------------------- files
def snapshot(dirs):
    """{path: (size, mtime_ns)} of every file under the given directories."""
    state = {}
    for d in dirs:
        for root, _, files in os.walk(d):
            for f in files:
                p = os.path.join(root, f)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                state[p] = (st.st_size, st.st_mtime_ns)
    return state


def new_files(before, after):
    """Paths in `after` that are new or changed since `before`, sorted."""
    return sorted(p for p, sig in after.items() if before.get(p) != sig)


def gpu_rate(gpu_name, rates):
    """(key, usd per hour) for a CUDA device name from the rates document (cloud/gpu_rates.json), or (None, None)."""
    for needle, key in (rates or {}).get("match", {}).items():
        if needle.lower() in (gpu_name or "").lower() and key in rates.get("rates", {}):
            return key, rates["rates"][key]
    return None, None


# ----------------------------------------------------------------------------- ComfyUI process
def _request(url, data=None, timeout=30, method=None):
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, method=method,
                                 headers={"Content-Type": "application/json"} if body is not None else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:           # ComfyUI answers 400 with a JSON body for a rejected graph
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, {"error": {"message": raw.decode("utf-8", "replace")[:500]}}


class ComfyProcess:
    """A headless ComfyUI: start it, wait for it, ask it for what it has. run_job only needs `url`, `input_dir`,
    `output_dir`, `temp_dir` and `gpu_name()`, so tests can point it at a fake server."""

    def __init__(self, comfy_dir, port=8188):
        self.comfy_dir = comfy_dir
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        self.input_dir = os.path.join(comfy_dir, "input")
        self.output_dir = os.path.join(comfy_dir, "output")
        self.temp_dir = os.path.join(comfy_dir, "temp")
        self.proc = None

    def start(self, extra_args=JOB_ARGS):
        cmd = [sys.executable, os.path.join(self.comfy_dir, "main.py"), "--listen", "127.0.0.1", "--port",
               str(self.port), "--disable-auto-launch", *extra_args]
        self.proc = subprocess.Popen(cmd, cwd=self.comfy_dir)
        return self

    def alive(self):
        """False once the ComfyUI process has exited (crash, OOM kill); also False before start()."""
        return self.proc is not None and self.proc.poll() is None

    def wait_ready(self, timeout=600):
        end = time.time() + timeout
        while time.time() < end:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(f"ComfyUI exited with code {self.proc.returncode} before it was ready")
            try:
                if _request(self.url + "/system_stats", timeout=5)[0] == 200:
                    return
            except (OSError, ValueError):
                pass
            time.sleep(0.5)
        raise TimeoutError(f"ComfyUI was not ready after {timeout}s")

    def gpu_name(self):
        """CUDA device name as ComfyUI reports it (e.g. "NVIDIA L40S"); "" on CPU or when unknown."""
        try:
            devices = _request(self.url + "/system_stats", timeout=10)[1].get("devices") or []
            return re.sub(r"^cuda:\d+\s+", "", devices[0]["name"]).split(" : ")[0] if devices else ""
        except (OSError, ValueError, KeyError, IndexError, AttributeError):
            return ""

    def info(self):
        """What a client needs before it can submit: node definitions, model files per folder, version."""
        def get(path):
            return _request(self.url + path, timeout=120)[1]
        models = {}
        for folder in get("/models") or []:
            try:
                models[folder] = get("/models/" + urllib.parse.quote(folder))
            except (OSError, ValueError):
                models[folder] = []
        return {"object_info": get("/object_info"), "models": models,
                "comfyui_version": (get("/system_stats").get("system") or {}).get("comfyui_version", ""),
                "updated": time.time()}

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(20)
            except subprocess.TimeoutExpired:
                self.proc.kill()


# ----------------------------------------------------------------------------- running a job
class _Progress(threading.Thread):
    """Follows ComfyUI's WebSocket for one client id and reports {node, value, max, done, total}. Needs `websockets`;
    does nothing without it."""

    def __init__(self, url, client_id, total, callback):
        super().__init__(daemon=True)
        self.url, self.client_id, self.callback = url, client_id, callback
        self.state = {"node": None, "value": 0, "max": 0, "done": [], "total": total}
        self.stop_flag = threading.Event()
        self.ws = None
        try:
            from websockets.sync.client import connect
            self.connect = connect
        except ImportError:
            self.connect = None

    def run(self):
        if self.connect is None:
            return
        try:
            self.ws = self.connect(self.url.replace("http", "ws", 1) + "/ws?clientId=" + self.client_id,
                                   open_timeout=10)
            while not self.stop_flag.is_set():
                try:
                    msg = self.ws.recv(timeout=1)
                except TimeoutError:
                    continue
                if isinstance(msg, bytes):
                    continue
                m = json.loads(msg)
                d, s = m.get("data") or {}, self.state
                if m.get("type") == "executing":
                    if s["node"] and s["node"] not in s["done"]:
                        s["done"].append(s["node"])
                    s["node"], s["value"], s["max"] = d.get("node"), 0, 0
                elif m.get("type") == "progress":
                    s["node"], s["value"], s["max"] = d.get("node") or s["node"], d.get("value", 0), d.get("max", 0)
                else:
                    continue
                self.callback(dict(s, done=list(s["done"])))
        except Exception:        # progress is best effort; the job result does not depend on it
            pass

    def close(self):
        self.stop_flag.set()
        try:
            if self.ws is not None:
                self.ws.close()
        except Exception:
            pass


def _history_files(history, titles):
    files = []
    for nid, out in (history.get("outputs") or {}).items():
        for key in HISTORY_KEYS:
            for f in out.get(key) or []:
                if isinstance(f, dict) and "filename" in f:
                    files.append((titles.get(nid, nid), f["filename"], f.get("subfolder", "") or "",
                                  f.get("type", "output")))
    return files


def _history_text(history, titles):
    text = {}
    for nid, out in (history.get("outputs") or {}).items():
        if out.get("text"):
            text[titles.get(nid, nid)] = "\n".join(map(str, out["text"]))
    return text


def _within(root, *parts):
    """root/parts as a real path that is certainly inside root (None if it is not)."""
    base = os.path.realpath(root)
    p = os.path.realpath(os.path.join(root, *parts))
    return p if p.startswith(base + os.sep) else None


def _output_classes(proc):
    """The class_types with output_node true in /object_info, fetched once per process; None if unreadable."""
    cached = getattr(proc, "_psc_output_classes", None)
    if cached is not None:
        return cached
    try:
        code, info = _request(proc.url + "/object_info", timeout=120)
    except (OSError, ValueError):
        return None
    if code != 200 or not isinstance(info, dict) or not info:
        return None
    classes = frozenset(k for k, v in info.items() if isinstance(v, dict) and v.get("output_node"))
    try:
        proc._psc_output_classes = classes
    except AttributeError:
        pass
    return classes


def output_node_ids(graph, classes=None):
    """Ids of the output nodes of an API graph: classes with output_node true, or, when `classes` is None
    (object_info unreadable), the nodes no other node links to."""
    if classes is not None:
        return [nid for nid, n in graph.items() if n.get("class_type") in classes]
    linked = set()
    for node in graph.values():
        for v in (node.get("inputs") or {}).values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], (str, int)) and str(v[0]) in graph:
                linked.add(str(v[0]))
    return [nid for nid in graph if str(nid) not in linked]


def mark_outputs(graph, classes=None, job_id=None):
    """A deep copy of `graph` with a unique MARKER_KEY input on each output node (an existing input is kept)."""
    marked = copy.deepcopy(graph)
    marker = job_id or uuid.uuid4().hex
    for nid in output_node_ids(marked, classes):
        inputs = marked[nid].setdefault("inputs", {})
        if isinstance(inputs, dict):
            inputs.setdefault(MARKER_KEY, marker)
    return marked


def run_job(doc, proc, on_progress=None, boot=0.0, rate=None, worker="", max_timeout=None):
    """Run one job document on `proc` and return the result document (file data as bytes: encode_result for JSON).

    boot:  seconds this container spent starting ComfyUI, to charge to this job (0 for later jobs of a container).
    rate:  usd per hour, or the rates document of cloud/gpu_rates.json (matched on the GPU name); None = no cost.
    max_timeout: the deployment's ceiling in seconds; the job's own timeout is cut to it, so the runner times the
           job out cleanly before the platform kills the whole call.
    Infrastructure problems (ComfyUI not answering, timeout) raise; a rejected graph or a failed node is a result
    with status "error"."""
    timeout = validate_job(doc)
    if max_timeout:
        timeout = min(timeout, max_timeout)
    inputs = decode_job(doc)
    graph = doc["workflow"]
    titles = {k: (v.get("_meta") or {}).get("title", k) for k, v in graph.items()}
    started = time.time()
    written, listed, produced = [], [], []
    before = snapshot([proc.output_dir, proc.temp_dir])
    result = {"protocol": PROTOCOL, "status": "success", "files": [], "text": {}, "errors": []}
    try:
        os.makedirs(proc.input_dir, exist_ok=True)
        for name, data in inputs:
            path = os.path.join(proc.input_dir, name)
            with open(path, "wb") as fh:
                fh.write(data)
            written.append(path)
        client_id = uuid.uuid4().hex
        submitted = mark_outputs(graph, _output_classes(proc))
        progress = None
        if on_progress is not None:
            progress = _Progress(proc.url, client_id, len(graph), on_progress)
            progress.start()
        try:
            code, res = _request(proc.url + "/prompt", {"prompt": submitted, "client_id": client_id})
            if code != 200 or not isinstance(res, dict) or "prompt_id" not in res or res.get("node_errors"):
                result["status"] = "error"
                result["errors"] += _prompt_errors(res)
            else:
                history = _wait_history(proc, res["prompt_id"], started + timeout)
                status = history.get("status") or {}
                for m in status.get("messages", []):
                    if m[0] == "execution_error":
                        result["errors"].append(f"{m[1].get('node_type')}: {m[1].get('exception_message')}")
                if status.get("status_str") != "success" or result["errors"]:
                    result["status"] = "error"
                    if not result["errors"]:
                        result["errors"].append(f"ComfyUI reported status {status.get('status_str')!r}")
                result["text"] = _history_text(history, titles)
                listed = _history_files(history, titles)
        finally:
            if progress is not None:
                progress.close()
        produced = new_files(before, snapshot([proc.output_dir, proc.temp_dir]))
        result["files"] = _gather_files(proc, listed, produced, result)
    finally:
        # Also on a timeout or a crash: whatever the job wrote must not show up in the next job's result.
        try:
            produced = new_files(before, snapshot([proc.output_dir, proc.temp_dir]))
        except OSError:
            pass
        _cleanup(proc, written, produced)
        try:
            _request(proc.url + "/history", {"clear": True}, timeout=10)
        except (OSError, ValueError):
            pass
    finished = time.time()
    run = finished - started
    result["timing"] = {"boot": round(boot, 3), "run": round(run, 3), "billed": round(boot + run, 3),
                        "started_at": started, "finished_at": finished}
    gpu = proc.gpu_name()
    per_hour = gpu_rate(gpu, rate)[1] if isinstance(rate, dict) else rate
    result["cost"] = {"gpu": gpu, "rate_per_hour": per_hour,
                      "usd": None if per_hour is None else round((boot + run) * per_hour / 3600, 6)}
    result["worker"] = worker
    return result


def run_job_guarded(doc, proc, recycle, **kw):
    """run_job for a container that serves several jobs one after another (Modal's ComfyJob, a RunPod worker).

    `recycle(reason)` must stop the container from taking more jobs (Modal: modal.experimental.stop_fetching_inputs).
    It is called when ComfyUI is not running at the start, when the job raises an infrastructure error (ComfyUI
    died, did not answer, timed out: its state is then unknown) and when ComfyUI is gone after the job. The next job
    then gets a fresh container, at the price of one cold start. A bad document (ValueError) is the caller's fault
    and recycles nothing. Infrastructure errors are re-raised as RuntimeError (a TimeoutError would read as "still
    running" to a status poll)."""
    validate_job(doc)
    decode_job(doc)
    if not proc.alive():
        recycle("ComfyUI is not running")
        raise RuntimeError("ComfyUI is not running; this container takes no more jobs")
    try:
        result = run_job(doc, proc, **kw)
    except Exception as e:
        recycle(f"{type(e).__name__}: {e}")
        raise RuntimeError(f"{type(e).__name__}: {e}") from None
    if not proc.alive():
        recycle("ComfyUI exited during the job")
    return result


def _prompt_errors(res):
    if not isinstance(res, dict):
        return ["ComfyUI rejected the graph"]
    errors = []
    err = res.get("error")
    if isinstance(err, dict):
        errors.append(f"{err.get('type', 'error')}: {err.get('message', '')}".strip())
    elif err:
        errors.append(str(err))
    for nid, ne in (res.get("node_errors") or {}).items():
        for e in ne.get("errors", []):
            errors.append(f"{ne.get('class_type', nid)}: {e.get('message', '')} {e.get('details', '')}".strip())
    return errors or ["ComfyUI rejected the graph"]


def _wait_history(proc, pid, deadline):
    """ComfyUI adds the history entry when the prompt has finished (or failed)."""
    while True:
        code, hist = _request(proc.url + "/history/" + pid, timeout=30)
        if code == 200 and isinstance(hist, dict) and pid in hist:
            return hist[pid]
        if time.time() > deadline:
            try:
                _request(proc.url + "/interrupt", {}, timeout=10)
            except (OSError, ValueError):
                pass
            _wait_idle(proc)
            raise TimeoutError("job did not finish before its timeout")
        time.sleep(0.5)


def _wait_idle(proc, seconds=20):
    """After /interrupt: wait until ComfyUI is no longer running a prompt, so a late save does not hit the next job."""
    end = time.time() + seconds
    while time.time() < end:
        try:
            code, q = _request(proc.url + "/queue", timeout=5)
        except (OSError, ValueError):
            return
        if code != 200 or not isinstance(q, dict) or not q.get("queue_running"):
            return
        time.sleep(0.25)


def _gather_files(proc, listed, produced, result):
    """File entries (with bytes) for the history-listed files (node title) and the other new files (node "")."""
    roots = {"output": proc.output_dir, "temp": proc.temp_dir}
    entries, seen = [], set()
    for node, filename, subfolder, ftype in listed:
        root = roots.get(ftype)
        path = _within(root, subfolder, filename) if root else None
        if path and os.path.isfile(path):
            if path not in seen:
                seen.add(path)
                entries.append((node, filename, subfolder, ftype, path))
        else:       # listed but not written by this job: ComfyUI served the node from its cache. Never pass silently.
            result["status"] = "error"
            result["errors"].append(f"{node}: ComfyUI lists {filename} but did not write it (output served from "
                                    "its execution cache); run the job again")
    for p in produced:
        rp = os.path.realpath(p)
        if rp in seen:
            continue
        for ftype, root in roots.items():
            rel = os.path.relpath(rp, os.path.realpath(root))
            if not rel.startswith(".."):
                sub, fn = os.path.split(rel)
                seen.add(rp)
                entries.append(("", fn, sub.replace("\\", "/"), ftype, rp))
                break
    if sum(os.path.getsize(e[4]) for e in entries) > MAX_OUTPUT_BYTES:
        result["status"] = "error"
        result["errors"].append(f"outputs are larger than {MAX_OUTPUT_BYTES // 2**20} MiB; none are returned")
        return []
    files = []
    for node, filename, subfolder, ftype, path in entries:
        with open(path, "rb") as fh:
            data = fh.read()
        files.append({"node": node, "filename": filename, "subfolder": subfolder, "type": ftype,
                      "size": len(data), "data": data})
    return files


def _cleanup(proc, written, produced):
    for p in list(written) + list(produced):
        try:
            os.remove(p)
        except OSError:
            pass
    for root in (proc.output_dir, proc.temp_dir):       # drop subfolders the job created and left empty
        for d, _, _ in sorted(os.walk(root), reverse=True):
            if d != root:
                try:
                    os.rmdir(d)
                except OSError:
                    pass
