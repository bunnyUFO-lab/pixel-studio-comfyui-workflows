#!/usr/bin/env python3
"""Sprite Studio trainer runner: a tiny HTTP front for ostris ai-toolkit. Standard library only (it runs inside the
ostris/aitoolkit image with that image's python; nothing to install).

    python runner.py            # listens on 0.0.0.0:8675

API (JSON; with RUNNER_TOKEN set every route but a reduced /health needs `Authorization: Bearer <token>`):
  GET  /health                    -> {ok, busy, job, version, deadline_in_s, idle_in_s}; without a valid token only
                                      {ok, version, auth: "required"}
  GET  /jobs                      -> [{name, status, step, total_steps}]
  POST /jobs {name, config_path | config, output_dir?, total_steps?}
                                  starts `RUN_CMD <config_path>` (one job at a time; 409 while another runs);
                                  `config` is the YAML text itself, written to CONFIG_ROOT/<name>.yaml
  PUT  /datasets/<name>           body: a zip of images and .txt captions (flat) -> DATASETS_ROOT/<name>/
  GET  /datasets/<name>           -> {name, images, captions, bytes}
  POST /deadline {seconds}        moves the time limit (60-86400 s from now, never past the maximum)
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
  DATASETS_ROOT uploaded datasets (/workspace/datasets)
  RUNNER_TOKEN  required bearer token (32+ characters); read once and removed from the environment
  RUNNER_ALLOW_OPEN=1  allow a non-local RUNNER_HOST without a token (compose does; the port is local there)
  RUNNER_TIME_LIMIT_S / RUNNER_TIME_LIMIT_MAX_S / RUNNER_IDLE_LIMIT_S / RUNNER_STATE_DIR / RUNNER_UPLOAD_MAX_BYTES
                (see trainer/README.md, "On a rented machine")

Progress comes from ai-toolkit's tqdm bar ("<name>:  12%|#2   | 300/2500 [05:01<36:00,  1.02s/it, lr: 1.0e-04
loss: 4.1e-01]"), checkpoints and samples from the output folder (<name>_000000250.safetensors, <name>.safetensors,
samples/<time>_000000250_0.jpg). A copy of the log is kept in OUTPUT_ROOT/<name>/runner.log and the final status in
runner_status.json, so a restarted runner still reports finished runs.
"""
import collections
import hashlib
import hmac
import json
import mimetypes
import os
import re
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import zipfile
import zlib
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
DATASETS_ROOT = Path(os.environ.get("DATASETS_ROOT", "/workspace/datasets"))
STATE_DIR = Path(os.environ.get("RUNNER_STATE_DIR", "/workspace/runner-state"))
# Read once and removed, so the ai-toolkit child and any later environment dump never see it.
TOKEN = os.environ.pop("RUNNER_TOKEN", "")
TOKEN_MIN = 32
LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}
CHILD_ENV_DROP = ("RUNNER_TOKEN", "HF_WRITE_TOKEN", "RUNPOD_API_KEY")   # HF_TOKEN stays: gated base models need it
SOCKET_TIMEOUT = 120                  # seconds per socket read
BODY_MAX = 2 << 20                    # JSON request bodies
CONFIG_MAX = 262144                   # inline config, bytes
# Upload limits (module constants so tests can override them)
UPLOAD_MAX_BYTES = int(os.environ.get("RUNNER_UPLOAD_MAX_BYTES", "268435456"))
UPLOAD_SPARE_BYTES = 1 << 30          # free space kept beyond the upload limit
MAX_ENTRIES = 1000
IMAGE_MAX = 32 << 20
TEXT_MAX = 64 << 10
TOTAL_MAX = 1 << 30
CHUNK = 1 << 20
# Time and idle limits (set by configure())
TIME_LIMIT_S = 0.0
TIME_LIMIT_MAX_S = 86400.0
IDLE_LIMIT_S = 0.0
WATCH_EVERY_S = 5.0
DEADLINE = None                       # epoch seconds, or None
FIRST_START = None
STATE_LOCK = threading.Lock()
LAST_ACTIVITY = time.time()
LOG_LINES = 400                       # kept in memory per job
TAIL = 40                             # lines returned in log_tail
KILL_GRACE = 15                       # seconds between SIGTERM and SIGKILL on cancel

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
ENTRY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\.(png|jpg|jpeg|webp|txt)$", re.I)
CONFIG_PATH_RE = re.compile(r"^\s*(?:-\s+)?(training_folder|folder_path):\s*(.+)$")
CONFIG_KEY_RE = re.compile(r"\b(?:training_folder|folder_path)\b")
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


# ----------------------------------------------------------------- hooks (ss-m10t8 fills these: checkpoints to Hugging Face)
def after_run(job):
    """Called at the end of Job._pump, after the final status is saved. No-op for now."""


def before_exit(reason, budget_s):
    """Called by the watchdog just before the runner exits for a time or idle limit. No-op for now."""


def touch():
    global LAST_ACTIVITY
    LAST_ACTIVITY = time.time()


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
        self.settled = threading.Event()        # set once the process is gone and the final status is saved
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
        for key in CHILD_ENV_DROP:
            env.pop(key, None)
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
            self.settled.set()
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
        touch()
        self.settled.set()
        try:
            after_run(self)
        except Exception as e:                       # noqa: BLE001 - a hook must never break the runner
            sys.stderr.write(f"runner: after_run failed: {type(e).__name__}\n")

    def cancel(self):
        with self.lock:
            if self.status not in ("queued", "running"):
                return False
            self.cancel_requested = True
            self.phase = "cancelling"               # status stays "running" until the process is gone (GPU still busy)
            proc = self.proc
            if proc is None:
                self.status, self.phase, self.finished = "cancelled", "cancelled", time.time()
                self.settled.set()
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


def busy_with(name):
    with JOBS_LOCK:
        j = JOBS.get(name)
    return j is not None and j.status in ("queued", "running")


def job_dir(name):
    with JOBS_LOCK:
        j = JOBS.get(name)
    return j.output_dir if j else OUTPUT_ROOT / name


# ----------------------------------------------------------------- token and bind rule
def bind_refusal(host, token, allow_open):
    """Why the runner must not bind `host` (a message), or None. No token on a non-local address is refused unless
    RUNNER_ALLOW_OPEN=1 (docker compose sets it: its port is published on 127.0.0.1 only)."""
    if token or host in LOCAL_HOSTS or allow_open is True or allow_open == "1":
        return None
    return (f"runner: refusing to listen on {host} without a token. Set RUNNER_TOKEN (32+ characters), bind "
            f"RUNNER_HOST=127.0.0.1, or set RUNNER_ALLOW_OPEN=1 if the port is already private.")


def token_ok(header):
    """Constant-time check of an `Authorization` header value against TOKEN (no token configured: always true)."""
    if not TOKEN:
        return True
    given = header[7:] if header and header.startswith("Bearer ") else ""
    return hmac.compare_digest(given.encode("utf-8", "replace"), TOKEN.encode())


# ----------------------------------------------------------------- datasets
DATASET_LOCK = threading.Lock()


def clear_incoming(root):
    inc = Path(root) / ".incoming"
    shutil.rmtree(inc, ignore_errors=True)
    inc.mkdir(parents=True, exist_ok=True)
    return inc


def image_kind(head):
    if head.startswith(b"\x89PNG\r\n\x1a\n") or head.startswith(b"\xff\xd8\xff"):
        return True
    return head[:4] == b"RIFF" and head[8:12] == b"WEBP"


def dataset_info(folder):
    """{images, captions, bytes} of a dataset folder."""
    images = captions = size = 0
    for p in Path(folder).iterdir():
        if p.is_file():
            size += p.stat().st_size
            if p.suffix.lower() == ".txt":
                captions += 1
            else:
                images += 1
    return {"images": images, "captions": captions, "bytes": size}


def extract_dataset(zip_path, root, name):
    """Validate the zip of images and captions and put it in root/<name>/ (replacing an earlier upload). Raises
    ValueError(message) and leaves nothing behind but an empty root/.incoming/ when anything is wrong."""
    root = Path(root)
    inc = root / ".incoming"
    inc.mkdir(parents=True, exist_ok=True)
    tag = secrets.token_hex(4)
    tmp, old = inc / f"{name}-{tag}", inc / f"{name}-{tag}.old"
    try:
        with zipfile.ZipFile(zip_path) as z:
            infos = z.infolist()
            if len(infos) > MAX_ENTRIES:
                raise ValueError(f"too many entries in the zip ({len(infos)}; at most {MAX_ENTRIES})")
            keep, seen = [], set()
            for info in infos:
                n = info.filename
                if n.endswith("/") or n.startswith("__MACOSX/") or n.split("/")[-1].startswith("._"):
                    continue
                if "/" in n or "\\" in n or not ENTRY_RE.match(n):
                    raise ValueError(f"entry not allowed: {n[:100]!r} (flat png, jpg, jpeg, webp and txt files only)")
                if info.flag_bits & 0x1:
                    raise ValueError(f"encrypted entry: {n}")
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError(f"symbolic link entry: {n}")
                if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                    raise ValueError(f"unsupported compression in entry: {n}")
                if n.lower() in seen:
                    raise ValueError(f"duplicate entry: {n}")
                seen.add(n.lower())
                keep.append(info)
            tmp.mkdir()
            total = 0
            for info in keep:
                n = info.filename
                limit = TEXT_MAX if n.lower().endswith(".txt") else IMAGE_MAX
                size = 0
                with z.open(info) as src, open(tmp / n, "xb") as dst:
                    while True:
                        chunk = src.read(CHUNK)
                        if not chunk:
                            break
                        size += len(chunk)
                        total += len(chunk)
                        if size > limit:
                            raise ValueError(f"entry too large: {n} (more than {limit} bytes)")
                        if total > TOTAL_MAX:
                            raise ValueError(f"the files add up to more than {TOTAL_MAX} bytes")
                        dst.write(chunk)
        images = captions = 0
        for info in keep:
            n = info.filename
            if n.lower().endswith(".txt"):
                try:
                    (tmp / n).read_bytes().decode("utf-8")
                except UnicodeDecodeError:
                    raise ValueError(f"caption is not UTF-8 text: {n}")
                captions += 1
            else:
                with open(tmp / n, "rb") as f:
                    if not image_kind(f.read(12)):
                        raise ValueError(f"not a png, jpeg or webp picture: {n}")
                images += 1
        if not images:
            raise ValueError("the zip holds no images")
        final, swapped = root / name, False
        with DATASET_LOCK:
            if busy_with(name):
                raise ValueError(f"a run named {name} is queued or running")
            if final.exists():
                final.rename(old)
                swapped = True
            try:
                tmp.rename(final)
            except OSError:
                if swapped:
                    old.rename(final)
                raise
        return {"name": name, "dataset_dir": str(final), "images": images, "captions": captions, "bytes": total}
    except (zipfile.BadZipFile, zlib.error, EOFError, RuntimeError, NotImplementedError) as e:
        raise ValueError(f"not a readable zip ({type(e).__name__})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(old, ignore_errors=True)


# ----------------------------------------------------------------- inline config
def check_inline_config(text):
    """None when the YAML text may run, else the reason. Every folder_path / training_folder must stay inside
    DATASETS_ROOT / OUTPUT_ROOT; a key written in a way these line checks cannot read is refused."""
    if len(text.encode("utf-8")) > CONFIG_MAX:
        return f"config is larger than {CONFIG_MAX} bytes"
    matched = 0
    for line in text.splitlines():
        m = CONFIG_PATH_RE.match(line)
        if not m:
            continue
        matched += 1
        value = m.group(2).strip()
        if value[:1] in ("\"", "'"):
            value = value[1:].split(value[0], 1)[0]
        else:
            value = value.split(" #", 1)[0].strip()
        root, key = (OUTPUT_ROOT, "training_folder") if m.group(1) == "training_folder" else (DATASETS_ROOT, "folder_path")
        if not value or not inside(root, value):
            return f"{key} must be inside {root}"
    if len(CONFIG_KEY_RE.findall(text)) != matched:
        return "training_folder and folder_path must each be on a line of their own (key: value)"
    return None


# ----------------------------------------------------------------- time limit and idle limit
def read_boot():
    try:
        d = json.loads((STATE_DIR / "boot.json").read_text(encoding="utf-8"))
        return float(d["first_start"]), float(d["deadline"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def write_boot():
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = STATE_DIR / "boot.json.tmp"
        tmp.write_text(json.dumps({"first_start": FIRST_START, "deadline": DEADLINE}), encoding="utf-8")
        tmp.replace(STATE_DIR / "boot.json")
        return True
    except OSError as e:
        sys.stderr.write(f"runner: warning: cannot save {STATE_DIR / 'boot.json'} ({type(e).__name__}); "
                         f"a restart would reset the time limit\n")
        return False


def env_number(name, default):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        v = float(raw)
        if v < 0 or v != v:
            raise ValueError
        return v
    except ValueError:
        sys.stderr.write(f"runner: {name} must be a number of seconds, 0 or more\n")
        raise SystemExit(2)


def configure():
    """Validate the token and limits from the environment (exit 2 on a bad value) and start the clock."""
    global TIME_LIMIT_S, TIME_LIMIT_MAX_S, IDLE_LIMIT_S, WATCH_EVERY_S, DEADLINE, FIRST_START
    if TOKEN and len(TOKEN) < TOKEN_MIN:
        sys.stderr.write(f"runner: RUNNER_TOKEN must be at least {TOKEN_MIN} characters\n")
        raise SystemExit(2)
    TIME_LIMIT_MAX_S = env_number("RUNNER_TIME_LIMIT_MAX_S", 86400.0)
    TIME_LIMIT_S = env_number("RUNNER_TIME_LIMIT_S", 10800.0 if TOKEN else 0.0)
    IDLE_LIMIT_S = env_number("RUNNER_IDLE_LIMIT_S", 1800.0 if TOKEN else 0.0)
    WATCH_EVERY_S = env_number("RUNNER_WATCH_EVERY_S", 5.0) or 5.0
    if TIME_LIMIT_S > TIME_LIMIT_MAX_S:
        sys.stderr.write(f"runner: RUNNER_TIME_LIMIT_S ({TIME_LIMIT_S:.0f}) is above RUNNER_TIME_LIMIT_MAX_S "
                         f"({TIME_LIMIT_MAX_S:.0f})\n")
        raise SystemExit(2)
    if TOKEN and TIME_LIMIT_S <= 0:
        sys.stderr.write("runner: with a token set, RUNNER_TIME_LIMIT_S must be above 0\n")
        raise SystemExit(2)
    if TIME_LIMIT_S > 0:
        boot = read_boot()
        now = time.time()
        if boot:
            FIRST_START, DEADLINE = boot
        else:
            FIRST_START, DEADLINE = now, now + TIME_LIMIT_S
            write_boot()
        if now >= DEADLINE:
            print("runner: time limit reached", flush=True)
            raise SystemExit(3)
    touch()


def stop_for_limit(reason):
    """Stop the training child, save its status and leave with code 3. Needs no request from the app."""
    print(f"runner: {reason}", flush=True)
    cur = current()
    if cur:
        with cur.lock:
            cur.error = f"stopped: {reason}"
        cur.cancel()
        cur.settled.wait(KILL_GRACE + 5)
        with cur.lock:
            cur.error = f"stopped: {reason}"
        cur.persist()
    try:
        before_exit(reason, budget_s=0)
    except Exception as e:                           # noqa: BLE001
        sys.stderr.write(f"runner: before_exit failed: {type(e).__name__}\n")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(3)


def watchdog():
    while True:
        time.sleep(WATCH_EVERY_S)
        now = time.time()
        with STATE_LOCK:
            deadline = DEADLINE
        if deadline is not None and now >= deadline:
            stop_for_limit("time limit reached")
        if IDLE_LIMIT_S > 0:
            if current():
                touch()
            elif now - LAST_ACTIVITY >= IDLE_LIMIT_S:
                stop_for_limit("idle limit reached")


def deadline_in():
    with STATE_LOCK:
        return None if DEADLINE is None else max(0, round(DEADLINE - time.time()))


def idle_in():
    if IDLE_LIMIT_S <= 0:
        return None
    return max(0, round(IDLE_LIMIT_S - (time.time() - LAST_ACTIVITY)))


def set_deadline(seconds):
    """Move the deadline to now + seconds, never past first start + the maximum. Returns the new deadline."""
    global DEADLINE
    with STATE_LOCK:
        DEADLINE = min(time.time() + seconds, FIRST_START + TIME_LIMIT_MAX_S)
        write_boot()
        return DEADLINE


# ----------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "sprite-studio-trainer/" + VERSION
    timeout = SOCKET_TIMEOUT                     # per socket read: a stalled client cannot hold a thread forever

    def log_message(self, fmt, *args):           # quiet: only errors, and only the request line (no headers, no query)
        if args and str(args[1] if len(args) > 1 else "").startswith(("4", "5")):
            shown = tuple(re.sub(r"\?\S*", "?", a) if isinstance(a, str) else a for a in args)
            sys.stderr.write("runner: %s %s\n" % (self.address_string(), fmt % shown))

    def send_json(self, code, data, close=False, headers=None):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        if close:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        self.wfile.write(body)

    def authed(self):
        """True when the request may go on. A refusal (401, the same for a missing and a wrong token) is sent here."""
        if token_ok(self.headers.get("Authorization")):
            touch()
            return True
        self.send_json(401, {"error": "token required"}, close=True, headers={"WWW-Authenticate": "Bearer"})
        return False

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        if n > BODY_MAX:
            return None
        try:
            d = json.loads(self.rfile.read(n) or b"{}")
            return d if isinstance(d, dict) else {}
        except ValueError:
            return None

    def parts(self):
        return [unquote(p) for p in urlparse(self.path).path.split("/") if p]

    def do_GET(self):
        p = self.parts()
        try:
            if p == ["health"] and not TOKEN:
                return self.send_health()
            if p == ["health"] and token_ok(self.headers.get("Authorization")):
                touch()
                return self.send_health()
            if p == ["health"]:
                return self.send_json(200, {"ok": True, "version": VERSION, "auth": "required"})
            if not self.authed():
                return
            if p == ["jobs"]:
                with JOBS_LOCK:
                    js = list(JOBS.values())
                return self.send_json(200, [{"name": j.name, "status": j.status, "step": j.step, "total_steps": j.total_steps} for j in js])
            if len(p) == 2 and p[0] == "jobs":
                with JOBS_LOCK:
                    j = JOBS.get(p[1])
                snap = j.snapshot() if j else (stored_status(p[1]) if NAME_RE.match(p[1]) else None)
                return self.send_json(200, snap) if snap else self.send_json(404, {"error": f"no job {p[1]}"})
            if len(p) == 2 and p[0] == "datasets":
                folder = DATASETS_ROOT / p[1]
                if not NAME_RE.match(p[1]) or not folder.is_dir():
                    return self.send_json(404, {"error": f"no dataset {p[1]}"})
                return self.send_json(200, {"name": p[1], **dataset_info(folder)})
            if len(p) >= 3 and p[0] == "files":
                return self.send_file(p[1], p[2:])
            return self.send_json(404, {"error": "not found"})
        except (BrokenPipeError, ConnectionResetError):
            pass

    def send_health(self):
        cur = current()
        return self.send_json(200, {"ok": True, "busy": cur is not None, "job": cur.name if cur else None,
                                    "version": VERSION, "output_root": str(OUTPUT_ROOT),
                                    "deadline_in_s": deadline_in(), "idle_in_s": idle_in()})

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
            if not self.authed():
                return
            if p == ["jobs"]:
                return self.start_job()
            if p == ["deadline"]:
                return self.move_deadline()
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

    def move_deadline(self):
        b = self.body()
        if b is None:
            return self.send_json(400, {"error": "body must be JSON"})
        secs = b.get("seconds")
        if deadline_in() is None:
            return self.send_json(409, {"error": "no time limit is active"})
        if isinstance(secs, bool) or not isinstance(secs, int) or not 60 <= secs <= 86400:
            return self.send_json(400, {"error": "seconds: a whole number from 60 to 86400"})
        deadline = set_deadline(secs)
        return self.send_json(200, {"deadline": deadline, "deadline_in_s": deadline_in()})

    def do_PUT(self):
        p = self.parts()
        try:
            if not self.authed():
                return
            if len(p) == 2 and p[0] == "datasets":
                return self.upload_dataset(p[1])
            return self.send_json(404, {"error": "not found"}, close=True)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def upload_dataset(self, name):
        if not NAME_RE.match(name):
            return self.send_json(400, {"error": "name: letters, digits, _ . - only"}, close=True)
        raw = self.headers.get("Content-Length")
        if raw is None or not raw.strip().isdigit():
            return self.send_json(411, {"error": "Content-Length required"}, close=True)
        length = int(raw)
        if length > UPLOAD_MAX_BYTES:
            return self.send_json(413, {"error": f"upload is larger than {UPLOAD_MAX_BYTES} bytes"}, close=True)
        if busy_with(name):
            return self.send_json(409, {"error": f"a run named {name} is queued or running"}, close=True)
        DATASETS_ROOT.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(DATASETS_ROOT).free < UPLOAD_MAX_BYTES + UPLOAD_SPARE_BYTES:
            return self.send_json(507, {"error": "not enough free disk space"}, close=True)
        inc = DATASETS_ROOT / ".incoming"
        inc.mkdir(parents=True, exist_ok=True)
        zpath = inc / f"{name}-{secrets.token_hex(4)}.zip"
        digest = hashlib.sha256()
        try:
            with open(zpath, "xb") as out:
                left = length
                while left:
                    chunk = self.rfile.read(min(CHUNK, left))
                    if not chunk:
                        return self.send_json(400, {"error": "the body is shorter than Content-Length"}, close=True)
                    digest.update(chunk)
                    out.write(chunk)
                    left -= len(chunk)
            sha = digest.hexdigest()
            want = (self.headers.get("X-Content-SHA256") or "").strip().lower()
            if want and want != sha:
                return self.send_json(400, {"error": "X-Content-SHA256 does not match the body"})
            try:
                info = extract_dataset(zpath, DATASETS_ROOT, name)
            except ValueError as e:
                return self.send_json(400, {"error": str(e)})
            return self.send_json(201, {**info, "sha256": sha})
        except OSError as e:                          # a stalled or dropped client, or a full disk
            sys.stderr.write(f"runner: upload of {name} failed: {type(e).__name__}\n")
            self.close_connection = True
        finally:
            try:
                zpath.unlink()
            except OSError:
                pass

    def start_job(self):
        b = self.body()
        if b is None:
            return self.send_json(400, {"error": "body must be JSON"})
        name = str(b.get("name") or "")
        cfg, inline = str(b.get("config_path") or ""), b.get("config")
        if not NAME_RE.match(name):
            return self.send_json(400, {"error": "name: letters, digits, _ . - only"})
        if inline is not None and cfg:
            return self.send_json(400, {"error": "give config or config_path, not both"})
        if inline is not None:
            if not isinstance(inline, str):
                return self.send_json(400, {"error": "config must be the YAML text (a string)"})
            why = check_inline_config(inline)
            if why:
                return self.send_json(400, {"error": why})
            cfg_path = CONFIG_ROOT / f"{name}.yaml"
        else:
            cfg_path = Path(cfg)
            if not cfg or not inside(CONFIG_ROOT, cfg_path):
                return self.send_json(400, {"error": f"config_path must be under {CONFIG_ROOT}"})
            if not cfg_path.is_file():
                return self.send_json(400, {"error": f"no config at {cfg}"})
        out_dir = Path(b.get("output_dir") or OUTPUT_ROOT / name)
        if not inside(OUTPUT_ROOT, out_dir):
            return self.send_json(400, {"error": f"output_dir must be under {OUTPUT_ROOT}"})
        with JOBS_LOCK:
            cur = next((j for j in JOBS.values() if j.status in ("queued", "running")), None)
            if cur:
                return self.send_json(409, {"error": f"busy: {cur.name} is training", "job": cur.name})
            if inline is not None:
                try:
                    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
                    tmp = CONFIG_ROOT / f"{name}.yaml.tmp"
                    tmp.write_text(inline, encoding="utf-8")
                    tmp.replace(cfg_path)
                except OSError as e:
                    return self.send_json(500, {"error": f"could not write the config ({type(e).__name__})"})
            total = b.get("total_steps") or config_steps(cfg_path)
            j = Job(name, cfg_path, out_dir, int(total) if total else None)
            JOBS[name] = j
        j.start()
        return self.send_json(201, j.snapshot())


def main():
    configure()
    refusal = bind_refusal(HOST, TOKEN, os.environ.get("RUNNER_ALLOW_OPEN") == "1")
    if refusal:
        sys.stderr.write(refusal + "\n")
        raise SystemExit(2)
    try:
        clear_incoming(DATASETS_ROOT)
    except OSError as e:
        sys.stderr.write(f"runner: warning: cannot prepare {DATASETS_ROOT} ({type(e).__name__})\n")
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    srv.daemon_threads = True
    limits = []
    if DEADLINE is not None:
        limits.append(f"time limit {deadline_in()} s left")
    if IDLE_LIMIT_S > 0:
        limits.append(f"idle limit {IDLE_LIMIT_S:.0f} s")
    print(f"sprite-studio trainer runner on {HOST}:{PORT}  (run: {' '.join(command('<config>'))}; outputs: {OUTPUT_ROOT}; "
          f"token: {'set' if TOKEN else 'none'}; {', '.join(limits) or 'no time limit'})", flush=True)
    if DEADLINE is not None or IDLE_LIMIT_S > 0:
        threading.Thread(target=watchdog, daemon=True, name="watchdog").start()

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
