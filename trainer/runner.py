#!/usr/bin/env python3
"""Sprite Studio trainer runner: a tiny HTTP front for ostris ai-toolkit. Standard library only (it runs inside the
ostris/aitoolkit image with that image's python; nothing to install).

    python runner.py            # listens on 0.0.0.0:8675

API (JSON):
  GET  /health                    -> {ok, busy, job, version}
  GET  /jobs                      -> [{name, status, step, total_steps}]
  POST /jobs {name, config_path, output_dir?, total_steps?}
                                  starts `RUN_CMD <config_path>` (one job at a time; 409 while another runs)
  GET  /jobs/<name>               -> {name, status: queued|running|done|failed|cancelled, phase, step, total_steps,
                                      it_per_s, eta_s, loss, log_tail: [lines], checkpoints: [files], samples: [files],
                                      started, finished, returncode, error}
  POST /jobs/<name>/cancel        terminates the whole process group
  GET  /files/<name>/<path>       a checkpoint / sample from the job's output folder (no path traversal)

Environment:
  RUNNER_HOST / RUNNER_PORT   bind address (0.0.0.0 / 8675)
  RUN_CMD       command the config path is appended to: "python /app/ai-toolkit/run.py" (shell words, or a JSON array)
  RUN_CWD       working directory for it (/app/ai-toolkit when it exists)
  CONFIG_ROOT   configs must live under this folder (/workspace/configs)
  OUTPUT_ROOT   ai-toolkit's training_folder: job outputs are OUTPUT_ROOT/<name> (/workspace/output)

Progress comes from ai-toolkit's tqdm bar ("<name>:  12%|#2   | 300/2500 [05:01<36:00,  1.02s/it, lr: 1.0e-04
loss: 4.1e-01]"), checkpoints and samples from the output folder (<name>_000000250.safetensors, <name>.safetensors,
samples/<time>_000000250_0.jpg). A copy of the log is kept in OUTPUT_ROOT/<name>/runner.log and the final status in
runner_status.json, so a restarted runner still reports finished runs.
"""
import collections
import json
import mimetypes
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

VERSION = "1"
HOST = os.environ.get("RUNNER_HOST", "0.0.0.0")
PORT = int(os.environ.get("RUNNER_PORT", "8675"))
RUN_CMD = os.environ.get("RUN_CMD", "python /app/ai-toolkit/run.py")
RUN_CWD = os.environ.get("RUN_CWD") or ("/app/ai-toolkit" if os.path.isdir("/app/ai-toolkit") else None)
CONFIG_ROOT = Path(os.environ.get("CONFIG_ROOT", "/workspace/configs"))
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", "/workspace/output"))
LOG_LINES = 400                       # kept in memory per job
TAIL = 40                             # lines returned in log_tail
KILL_GRACE = 15                       # seconds between SIGTERM and SIGKILL on cancel

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
BAR_RE = re.compile(r"^(?P<desc>.*?):?\s*(?P<pct>\d{1,3})%\|(?P<bar>[^|]*)\|\s*(?P<n>\d+)/(?P<total>\d+)\s*(?:\[(?P<info>[^\]]*)\]?)?")
RATE_RE = re.compile(r"(?P<v>\d+(?:\.\d+)?)\s*(?P<u>s/it|it/s)")
LOSS_RE = re.compile(r"loss:\s*(?P<v>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
CKPT_RE = re.compile(r"_(\d{6,})\.safetensors$")
SAMPLE_EXT = (".jpg", ".jpeg", ".png", ".webp")
SERVE_EXT = {".safetensors", ".jpg", ".jpeg", ".png", ".webp", ".txt", ".log", ".yaml", ".yml", ".json"}
PHASES = [  # (pattern in a log line or a progress bar's description, phase)
    (re.compile(r"generating images|sampling", re.I), "drawing samples"),
    (re.compile(r"caching latents|caching text|cache latents", re.I), "caching latents"),
    (re.compile(r"fetching \d+ files|download|\.safetensors:\s*\d+%|\.(?:json|txt|model):\s*\d+%", re.I), "downloading models"),
    (re.compile(r"quantiz", re.I), "quantizing the model"),
    (re.compile(r"loading|load checkpoint|loading checkpoint shards", re.I), "loading the model"),
    (re.compile(r"saving|saved", re.I), "saving a checkpoint"),
]


def parse_interval(text):
    """tqdm intervals '05:01', '1:02:03' -> seconds (None for '?')."""
    text = (text or "").strip()
    if not re.fullmatch(r"\d+(?::\d{1,2}){1,2}", text):
        return None
    secs = 0
    for part in text.split(":"):
        secs = secs * 60 + int(part)
    return secs


def parse_bar(line):
    """A tqdm progress line -> {desc, step, total, it_per_s, eta_s, loss} or None."""
    m = BAR_RE.search(line)
    if not m:
        return None
    info = m.group("info") or ""
    out = {"desc": m.group("desc").strip(), "pct": int(m.group("pct")), "step": int(m.group("n")), "total": int(m.group("total")),
           "it_per_s": None, "eta_s": None, "loss": None}
    r = RATE_RE.search(info)
    if r:
        v = float(r.group("v"))
        out["it_per_s"] = (1.0 / v if v > 0 else None) if r.group("u") == "s/it" else v
    head = info.split(",", 1)[0]
    if "<" in head:
        out["eta_s"] = parse_interval(head.split("<", 1)[1])
    if out["eta_s"] is None and out["it_per_s"]:
        out["eta_s"] = round(max(0, out["total"] - out["step"]) / out["it_per_s"])
    lm = LOSS_RE.search(info)
    if lm:
        out["loss"] = float(lm.group("v"))
    return out


ANY_BAR_RE = re.compile(r"^(?P<desc>.*?):?\s*(?P<pct>\d{1,3})%\|[^|]*\|")


def any_bar(line):
    """Any tqdm-style bar (also '4.49G/9.97G' downloads) -> {desc, pct} or None."""
    m = ANY_BAR_RE.search(line or "")
    return {"desc": m.group("desc").strip(), "pct": int(m.group("pct"))} if m else None


def command(config_path):
    raw = RUN_CMD.strip()
    words = json.loads(raw) if raw.startswith("[") else shlex.split(raw)
    return [*words, str(config_path)]


def config_steps(path):
    """Total steps from the config text (no YAML parser needed: the `steps:` line of the train section)."""
    try:
        m = re.search(r"^\s*steps:\s*(\d+)\s*$", Path(path).read_text(encoding="utf-8", errors="replace"), re.M)
        return int(m.group(1)) if m else None
    except OSError:
        return None


def inside(root, path):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def list_outputs(out_dir):
    out_dir = Path(out_dir)
    ck, sm = [], []
    if out_dir.is_dir():
        ck = [p.name for p in out_dir.iterdir() if p.is_file() and p.suffix == ".safetensors"]

        def key(n):
            m = CKPT_RE.search(n)
            return (0, int(m.group(1)), n) if m else (1, 0, n)
        ck.sort(key=key)
        sdir = out_dir / "samples"
        if sdir.is_dir():
            sm = sorted(p.name for p in sdir.iterdir() if p.is_file() and p.suffix.lower() in SAMPLE_EXT)
    return ck, sm


# ----------------------------------------------------------------- jobs
class Job:
    def __init__(self, name, config_path, output_dir, total_steps):
        self.name, self.config_path, self.output_dir = name, str(config_path), Path(output_dir)
        self.total_steps = total_steps
        self.status, self.phase = "queued", "starting"
        self.step, self.it_per_s, self.eta_s, self.loss = 0, None, None, None
        self.log = collections.deque(maxlen=LOG_LINES)
        self.started = self.finished = self.returncode = None
        self.error = None
        self.proc = None
        self.cancel_requested = False
        self.lock = threading.Lock()
        self._last_logged_step = -1
        self._last_bar_written = None
        self._logfile = None

    # ---- output parsing
    def feed(self, segment, newline):
        """One piece of output ending in \\n (newline=True) or \\r (a progress update). Progress bars redraw many
        times a second: the latest state of each bar replaces its previous line instead of piling up."""
        line = segment.rstrip()
        if not line.strip():
            return
        bar = parse_bar(line)
        anybar = bar or any_bar(line)
        with self.lock:
            prev = any_bar(self.log[-1]) if self.log else None
            if bar and self._is_training_bar(bar):
                self.phase = "training"
                self.step, self.total_steps = bar["step"], bar["total"]
                self.it_per_s = bar["it_per_s"] if bar["it_per_s"] is not None else self.it_per_s
                self.eta_s = bar["eta_s"]
                self.loss = bar["loss"] if bar["loss"] is not None else self.loss
                if prev and prev["desc"] == bar["desc"]:
                    self.log[-1] = line
                else:
                    self.log.append(line)
                if self.step - self._last_logged_step >= 50 or (newline and self.step != self._last_logged_step):
                    self._last_logged_step = self.step
                    self._write_log(line)
                return
            if anybar:                                       # another bar: downloads, loading shards, caching, sampling
                self._phase_from(anybar["desc"] or line)
                if prev and prev["desc"] == anybar["desc"]:
                    self.log[-1] = line
                else:
                    self.log.append(line)
                if anybar["pct"] >= 100 and line != self._last_bar_written:
                    self._last_bar_written = line
                    self._write_log(line)
                return
            self._phase_from(line)
            self.log.append(line)
            self._last_bar_written = None                    # the same bar text later (next sample round) is logged again
            self._write_log(line)

    def _is_training_bar(self, bar):
        if bar["desc"] == self.name or bar["desc"].endswith(self.name):
            return True
        if self.total_steps and bar["total"] == self.total_steps:
            return not any(p.search(bar["desc"]) for p, _ in PHASES)
        return False

    def _phase_from(self, text):
        for pat, phase in PHASES:
            if pat.search(text):
                if not (self.phase == "training" and phase in ("loading the model", "saving a checkpoint")):
                    self.phase = phase
                elif phase == "saving a checkpoint":
                    self.phase = phase
                return

    def _write_log(self, line):
        try:
            if self._logfile is None:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                self._logfile = open(self.output_dir / "runner.log", "a", encoding="utf-8")
            self._logfile.write(line + "\n")
            self._logfile.flush()
        except OSError:
            pass

    # ---- status
    def snapshot(self):
        ck, sm = list_outputs(self.output_dir)
        with self.lock:
            return {"name": self.name, "status": self.status, "phase": self.phase, "step": self.step,
                    "total_steps": self.total_steps, "it_per_s": self.it_per_s, "eta_s": self.eta_s, "loss": self.loss,
                    "log_tail": list(self.log)[-TAIL:], "checkpoints": ck, "samples": sm, "started": self.started,
                    "finished": self.finished, "returncode": self.returncode, "error": self.error,
                    "config_path": self.config_path}

    def persist(self):
        snap = self.snapshot()
        snap.pop("checkpoints"), snap.pop("samples")
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.output_dir / "runner_status.json.tmp"
            tmp.write_text(json.dumps(snap, indent=1), encoding="utf-8")
            tmp.replace(self.output_dir / "runner_status.json")
        except OSError:
            pass

    # ---- process
    def start(self):
        cmd = command(self.config_path)
        env = dict(os.environ, PYTHONUNBUFFERED="1", TQDM_MININTERVAL=os.environ.get("TQDM_MININTERVAL", "1"))
        kw = {"start_new_session": True} if os.name != "nt" else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        self._write_log(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} start: {' '.join(cmd)}")
        try:
            self.proc = subprocess.Popen(cmd, cwd=RUN_CWD, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, **kw)
        except OSError as e:
            with self.lock:
                self.status, self.phase, self.error = "failed", "failed", f"could not start the trainer: {e}"
                self.finished = time.time()
                self.log.append(self.error)
            self._write_log(self.error)
            self.persist()
            return
        with self.lock:
            self.status, self.started = "running", time.time()
        self.persist()
        threading.Thread(target=self._pump, daemon=True, name=f"pump-{self.name}").start()

    def _pump(self):
        buf = ""
        try:
            out = self.proc.stdout
            while True:
                chunk = out.read1(65536) if hasattr(out, "read1") else out.read(4096)
                if not chunk:
                    break
                buf += chunk.decode("utf-8", errors="replace")
                while True:
                    m = re.search(r"\r\n|\r(?=.)|\n", buf, re.S)   # a trailing "\r" waits: it may be half of "\r\n"
                    if not m:
                        break
                    seg, sep = buf[:m.start()], m.group()
                    buf = buf[m.end():]
                    self.feed(seg, sep != "\r")
                # tqdm writes "\r<bar>" and only ends it with the next redraw: show a complete bar right away
                # (it is fed again when its terminator arrives, which just replaces it in place)
                pending = buf.rstrip("\r")
                if pending.endswith("]") and any_bar(pending):
                    self.feed(pending, False)
            if buf.strip("\r\n"):
                self.feed(buf.strip("\r\n"), True)
        except Exception as e:                       # noqa: BLE001 - never leave a job stuck in "running"
            with self.lock:
                self.log.append(f"runner: output reader failed: {e}")
        rc = self.proc.wait()
        with self.lock:
            self.returncode, self.finished = rc, time.time()
            if self.cancel_requested:
                self.status, self.phase = "cancelled", "cancelled"
            elif rc == 0:
                self.status, self.phase, self.eta_s = "done", "done", 0
                if self.total_steps and self.step < self.total_steps and self.step:
                    self.log.append(f"runner: finished at step {self.step} of {self.total_steps}")
            else:
                self.status, self.phase = "failed", "failed"
                errs = [ln for ln in self.log if re.search(r"error|exception|traceback|out of memory|killed", ln, re.I)]
                self.error = (errs[-1] if errs else (self.log[-1] if self.log else "")) or f"exit code {rc}"
                if rc in (-9, 137):
                    self.error += " (killed: out of memory? give WSL/Docker more RAM)"
            line = f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} {self.status} (exit code {rc})"
        self._write_log(line)
        self.persist()
        if self._logfile:
            try:
                self._logfile.close()
            except OSError:
                pass

    def cancel(self):
        with self.lock:
            if self.status not in ("queued", "running"):
                return False
            self.cancel_requested = True
            self.phase = "cancelling"               # status stays "running" until the process is gone (GPU still busy)
            proc = self.proc
            if proc is None:
                self.status, self.phase, self.finished = "cancelled", "cancelled", time.time()
                return True
        threading.Thread(target=self._kill, args=(proc,), daemon=True).start()
        return True

    def _kill(self, proc):
        if proc.poll() is not None:
            return
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, timeout=20)
            else:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(KILL_GRACE)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
        except (OSError, subprocess.SubprocessError):
            pass
        if proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass


JOBS = {}
JOBS_LOCK = threading.Lock()


def current():
    with JOBS_LOCK:
        return next((j for j in JOBS.values() if j.status in ("queued", "running")), None)


def stored_status(name):
    """Status of a run this runner process didn't start (it was restarted): from runner_status.json + the folder."""
    out_dir = OUTPUT_ROOT / name
    if not out_dir.is_dir():
        return None
    ck, sm = list_outputs(out_dir)
    snap = {}
    try:
        snap = json.loads((out_dir / "runner_status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    if not snap and not ck and not sm:
        return None
    status = snap.get("status") or ("done" if f"{name}.safetensors" in ck else "failed")
    if status in ("queued", "running"):
        status = "failed"
        snap["error"] = "the trainer restarted while this run was in progress"
    tail = []
    try:
        tail = (out_dir / "runner.log").read_text(encoding="utf-8", errors="replace").splitlines()[-TAIL:]
    except OSError:
        pass
    return {"name": name, "status": status, "phase": status, "step": snap.get("step", 0), "total_steps": snap.get("total_steps"),
            "it_per_s": None, "eta_s": None, "loss": snap.get("loss"), "log_tail": tail or snap.get("log_tail", []),
            "checkpoints": ck, "samples": sm, "started": snap.get("started"), "finished": snap.get("finished"),
            "returncode": snap.get("returncode"), "error": snap.get("error"), "config_path": snap.get("config_path")}


def job_dir(name):
    with JOBS_LOCK:
        j = JOBS.get(name)
    return j.output_dir if j else OUTPUT_ROOT / name


# ----------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "sprite-studio-trainer/" + VERSION

    def log_message(self, fmt, *args):            # quiet: only errors
        if args and str(args[1] if len(args) > 1 else "").startswith(("4", "5")):
            sys.stderr.write("runner: %s %s\n" % (self.address_string(), fmt % args))

    def send_json(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            d = json.loads(self.rfile.read(min(n, 1 << 20)) or b"{}")
            return d if isinstance(d, dict) else {}
        except ValueError:
            return None

    def parts(self):
        return [unquote(p) for p in urlparse(self.path).path.split("/") if p]

    def do_GET(self):
        p = self.parts()
        try:
            if p == ["health"]:
                cur = current()
                return self.send_json(200, {"ok": True, "busy": cur is not None, "job": cur.name if cur else None,
                                            "version": VERSION, "output_root": str(OUTPUT_ROOT)})
            if p == ["jobs"]:
                with JOBS_LOCK:
                    js = list(JOBS.values())
                return self.send_json(200, [{"name": j.name, "status": j.status, "step": j.step, "total_steps": j.total_steps} for j in js])
            if len(p) == 2 and p[0] == "jobs":
                with JOBS_LOCK:
                    j = JOBS.get(p[1])
                snap = j.snapshot() if j else (stored_status(p[1]) if NAME_RE.match(p[1]) else None)
                return self.send_json(200, snap) if snap else self.send_json(404, {"error": f"no job {p[1]}"})
            if len(p) >= 3 and p[0] == "files":
                return self.send_file(p[1], p[2:])
            return self.send_json(404, {"error": "not found"})
        except (BrokenPipeError, ConnectionResetError):
            pass

    def send_file(self, name, rel):
        if not NAME_RE.match(name) or any(x in ("", ".", "..") or "\\" in x for x in rel):
            return self.send_json(400, {"error": "bad path"})
        root = job_dir(name)
        path = root.joinpath(*rel)
        if not inside(root, path) or not path.is_file() or path.suffix.lower() not in SERVE_EXT:
            return self.send_json(404, {"error": "no such file"})
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        size = path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        self.end_headers()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                self.wfile.write(chunk)

    def do_POST(self):
        p = self.parts()
        try:
            if p == ["jobs"]:
                return self.start_job()
            if len(p) == 3 and p[0] == "jobs" and p[2] == "cancel":
                with JOBS_LOCK:
                    j = JOBS.get(p[1])
                if not j:
                    return self.send_json(404, {"error": f"no job {p[1]}"})
                ok = j.cancel()
                return self.send_json(200, {"ok": ok, **j.snapshot()})
            return self.send_json(404, {"error": "not found"})
        except (BrokenPipeError, ConnectionResetError):
            pass

    def start_job(self):
        b = self.body()
        if b is None:
            return self.send_json(400, {"error": "body must be JSON"})
        name, cfg = str(b.get("name") or ""), str(b.get("config_path") or "")
        if not NAME_RE.match(name):
            return self.send_json(400, {"error": "name: letters, digits, _ . - only"})
        cfg_path = Path(cfg)
        if not cfg or not inside(CONFIG_ROOT, cfg_path):
            return self.send_json(400, {"error": f"config_path must be under {CONFIG_ROOT}"})
        if not cfg_path.is_file():
            return self.send_json(400, {"error": f"no config at {cfg}"})
        out_dir = Path(b.get("output_dir") or OUTPUT_ROOT / name)
        if not inside(OUTPUT_ROOT, out_dir):
            return self.send_json(400, {"error": f"output_dir must be under {OUTPUT_ROOT}"})
        total = b.get("total_steps") or config_steps(cfg_path)
        with JOBS_LOCK:
            cur = next((j for j in JOBS.values() if j.status in ("queued", "running")), None)
            if cur:
                return self.send_json(409, {"error": f"busy: {cur.name} is training", "job": cur.name})
            j = Job(name, cfg_path, out_dir, int(total) if total else None)
            JOBS[name] = j
        j.start()
        return self.send_json(201, j.snapshot())


def main():
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    srv.daemon_threads = True
    print(f"sprite-studio trainer runner on {HOST}:{PORT}  (run: {' '.join(command('<config>'))}; outputs: {OUTPUT_ROOT})", flush=True)

    def stop(*_):
        cur = current()
        if cur:
            cur.cancel()
            time.sleep(1)
        raise SystemExit(0)
    if os.name != "nt":
        signal.signal(signal.SIGTERM, stop)
    try:
        srv.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        pass


if __name__ == "__main__":
    main()
