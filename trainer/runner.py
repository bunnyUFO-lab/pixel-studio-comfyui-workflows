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
                                  `config` is the ai-toolkit config as a JSON object (not text), checked against an
                                  allowlist and written to CONFIG_ROOT/<name>.json
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
  HF_WRITE_TOKEN + HF_REPO   upload checkpoints, samples and the config to that private Hugging Face repo (cloud only; both
                read once and removed from the environment); RUNNER_UPLOAD_CMD / _EVERY_S / _SETTLE_S / _GRACE_S
                (see trainer/README.md, "Checkpoints to Hugging Face"); GET /jobs/<name> gains `upload`

Progress comes from ai-toolkit's tqdm bar ("<name>:  12%|#2   | 300/2500 [05:01<36:00,  1.02s/it, lr: 1.0e-04
loss: 4.1e-01]"), checkpoints and samples from the output folder (<name>_000000250.safetensors, <name>.safetensors,
samples/<time>_000000250_0.jpg). A copy of the log is kept in OUTPUT_ROOT/<name>/runner.log and the final status in
runner_status.json, so a restarted runner still reports finished runs.
"""
import collections
import math
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
import stat
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
# Hugging Face upload (ss-m10t8): read once and removed, like RUNNER_TOKEN, so no child ever inherits them
HF_WRITE_TOKEN = os.environ.pop("HF_WRITE_TOKEN", "")
HF_REPO = os.environ.pop("HF_REPO", "")
# Read once at start, like the token: set at machine creation, so nobody can change them while the run goes on.
UPLOAD_ALLOW_PUBLIC = os.environ.pop("RUNNER_UPLOAD_ALLOW_PUBLIC", "") == "1"      # allow a public target repository
UPLOAD_ALLOW_CLASSIC = os.environ.pop("RUNNER_UPLOAD_ALLOW_CLASSIC", "") == "1"    # allow a classic (account-wide) token
LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}
CHILD_ENV_DROP = ("RUNNER_TOKEN", "HF_WRITE_TOKEN", "RUNPOD_API_KEY")
# The token goes only to huggingface.co: no other host, certificate store or proxy can come from the environment.
# (The training child keeps HF_TOKEN: gated base models need it; only the upload child loses it.)
UPLOAD_CHILD_ENV_DROP = ("HF_TOKEN", "HF_ENDPOINT", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE", "SSL_CERT_DIR",
                         "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy")
TOKEN_CHECK_WAITS = (30.0, 90.0, 270.0)   # seconds before the 2nd, 3rd and 4th token check after a failed one
TOKEN_CHECK_TRIES = 4                     # failed checks in a row, then the run is refused
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
TIME_LIMIT_MAX_S = 10800.0
IDLE_LIMIT_S = 0.0
WATCH_EVERY_S = 5.0
DEADLINE = None                       # epoch seconds, or None
FIRST_START = None
STATE_LOCK = threading.Lock()
LAST_ACTIVITY = time.time()
LOG_LINES = 400                       # kept in memory per job
TAIL = 40                             # lines returned in log_tail
KILL_GRACE = float(os.environ.get("RUNNER_KILL_GRACE_S", "15"))   # seconds between SIGTERM and SIGKILL on cancel
_exit = os._exit                      # tests replace it
# Upload to Hugging Face (module constants so tests can override them)
UPLOAD_CMD = os.environ.get("RUNNER_UPLOAD_CMD", "python -I -u /runner/hf_upload.py upload")
UPLOAD_EVERY_S = 60.0                 # RUNNER_UPLOAD_EVERY_S, set by configure()
UPLOAD_SETTLE_S = 30.0                # RUNNER_UPLOAD_SETTLE_S: a file must be this old and unchanged between two passes
UPLOAD_GRACE_S = 300.0                # RUNNER_UPLOAD_GRACE_S: the last pass at a time or idle limit (at most 480)
UPLOAD_GRACE_MAX_S = 480.0            # + the 15 s kill grace and 5 s settle stays inside the backstop (limit + 600 s)
UPLOAD_PASS_TIMEOUT_S = 900.0         # one upload child
UPLOAD_FILE_MAX = 2 << 30
UPLOAD_RUN_MAX = 6 << 30
UPLOAD_COUNT_MAX = 300
MODEL_KEYS = {"Tongyi-MAI/Z-Image-Turbo": "z-image-turbo"}     # config model.name_or_path -> the app's model id
HF_BASE_URL = "https://huggingface.co"

NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")           # \Z, not $: "job\n" must not match
ENTRY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\.(png|jpg|jpeg|webp|txt)\Z", re.I)
TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{32,}\Z")                        # what secrets.token_urlsafe makes
HF_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{20,}\Z")
HF_REPO_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
UPLOAD_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\.(safetensors|jpg|jpeg|png|webp|json)\Z")
HEX40_RE = re.compile(r"[0-9a-f]{40}\Z")
HEX64_RE = re.compile(r"[0-9a-f]{64}\Z")
REFUSED_RE = re.compile(r"(?<![0-9])(401|403|404)(?![0-9])")
LENGTH_RE = re.compile(r"[0-9]{1,15}\Z")                              # ASCII digits only ("\u00b2".isdigit() is True)
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


# ----------------------------------------------------------------- checkpoints to Hugging Face (ss-m10t8)
def upload_command():
    raw = UPLOAD_CMD.strip()
    return json.loads(raw) if raw.startswith("[") else shlex.split(raw)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                return h.hexdigest()
            h.update(chunk)


def kill_tree(proc):
    """Stop a child and everything it started (no grace: an upload child holds nothing that needs saving)."""
    if proc is None or proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, timeout=20)
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        pass
    if proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass


def safetensors_complete(path):
    """True when the file is a whole safetensors file: the 8-byte header length, a JSON header, and a file size equal to
    8 + header length + the end of the last tensor. A file still being written, or cut off, fails this."""
    try:
        size = os.lstat(path).st_size
        with open(path, "rb") as fh:
            head = fh.read(8)
            if len(head) < 8:
                return False
            n = int.from_bytes(head, "little")
            if n < 2 or n > (100 << 20) or 8 + n > size:
                return False
            meta = json.loads(fh.read(n).decode("utf-8"))
    except (OSError, ValueError, RecursionError):
        return False
    if not isinstance(meta, dict):
        return False
    end = 0
    for key, tensor in meta.items():
        if key == "__metadata__":
            continue
        offs = tensor.get("data_offsets") if isinstance(tensor, dict) else None
        if not (isinstance(offs, list) and len(offs) == 2 and all(isinstance(o, int) and not isinstance(o, bool) for o in offs)
                and 0 <= offs[0] <= offs[1]):
            return False
        end = max(end, offs[1])
    return size == 8 + n + end


class Rec:
    """One file of a run: what it is, where it goes, and how far it got."""

    def __init__(self, rel, path, local, kind):
        self.rel, self.path, self.local, self.kind = rel, path, local, kind     # kind: checkpoint | sample | config
        self.sha256 = self.size = self.error = self.url = None
        self.state = "pending"
        self.sig = None                  # (size, mtime_ns) seen at the last scan
        self.hashed_sig = None           # the signature the sha256 belongs to
        self.counted = 0                 # the bytes this file adds to the run's total

    def public(self):
        return {"file": self.rel, "path": self.path, "sha256": self.sha256, "size": self.size, "state": self.state,
                "error": self.error, "url": self.url}


class UploadRun:
    def __init__(self, job_name, out_dir, prefix, error=None):
        self.job_name, self.out_dir, self.prefix = job_name, Path(out_dir), prefix
        self.recs = {}                   # local path -> Rec
        self.error = error               # a run-level failure: unknown base model, token or repo refused
        self.final_done = error is not None
        self.in_flight = False
        self.total_bytes = 0
        self.refused = None              # "token or repo refused (HTTP 401)": no more passes


class Uploader:
    """Sends each finished checkpoint, the samples and the config to a private Hugging Face repo while the run trains,
    and once more when it ends. The token is held here only: each pass hands it to a child on its stdin."""

    def __init__(self, token, repo, enabled=True):
        self.token, self.repo, self.enabled = token, repo, bool(enabled and token and repo)
        self.runs = {}
        self.last = None
        self.pass_lock = threading.Lock()          # one pass at a time
        self.state_lock = threading.RLock()
        self.persist_lock = threading.Lock()
        self.proc = None
        self.stopping = False                      # the limit is here: no new periodic pass
        self.timed_out = False                     # the grace ended while a pass was in flight
        self.token_ok = False                      # the token kind was checked and accepted: kept for the whole run
        self.check_fails = 0                       # failed token checks in a row (a failed whoami, not a refusal)
        self.check_retry_at = 0.0                  # no new token check before this time, except in the final pass
        self.check_error = "token check failed"

    # ---- helpers
    def redact(self, text):
        text = str(text)
        return text.replace(self.token, "***") if self.token else text

    def busy(self):
        return self.enabled and any(r.in_flight for r in list(self.runs.values()))

    def base_prefix(self, job):
        """`<base model>/<job>` from the config's model.name_or_path, or None."""
        try:
            text = Path(job.config_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        try:
            name = json.loads(text)["config"]["process"][0]["model"]["name_or_path"]
        except (ValueError, KeyError, IndexError, TypeError):
            m = re.search(r"^\s*name_or_path:\s*[\"']?([^\s\"']+)", text, re.M)
            name = m.group(1) if m else None
        key = MODEL_KEYS.get(name) if isinstance(name, str) else None
        return f"{key}/{job.name}" if key else None

    # ---- lifecycle and status
    def begin(self, job):
        if not self.enabled:
            return
        prefix = self.base_prefix(job)
        run = UploadRun(job.name, job.output_dir, prefix, None if prefix else "unknown base model")
        with self.state_lock:
            self.runs[job.name] = run
            self.last = run
            self.stopping = False
            self.timed_out = False
        self.persist(run)

    def status(self, name):
        if not self.enabled:
            return {"state": "off", "repo": None, "error": None, "files": []}
        run = self.runs.get(name)
        if run is None:
            return {"state": "waiting", "repo": self.repo, "error": None, "files": []}
        return self.public(run)

    def public(self, run):
        with self.state_lock:
            recs = sorted(run.recs.values(), key=lambda r: ({"checkpoint": 0, "sample": 1, "config": 2}[r.kind], r.rel))
            bad = [r for r in recs if r.kind == "checkpoint" and r.state not in ("done", "gone")]
            final = next((r for r in recs if r.rel == f"{run.job_name}.safetensors"), None)
            if run.error:
                state, error = "failed", run.error
            elif run.final_done:
                # "done" needs the final checkpoint uploaded and verified; a pruned step save (gone) never covers it
                if bad:
                    state, error = "failed", bad[0].error or "a checkpoint was not uploaded"
                elif final is None:
                    state, error = "failed", f"final checkpoint {run.job_name}.safetensors was never written"
                elif final.state != "done":
                    state, error = "failed", final.error or f"final checkpoint {run.job_name}.safetensors was not uploaded"
                else:
                    state, error = "done", None
            else:
                state, error = ("uploading" if run.in_flight else "waiting"), None
            return {"state": state, "repo": self.repo, "error": error, "files": [r.public() for r in recs]}

    def persist(self, run):
        with self.persist_lock:                      # three threads call this and share one tmp name
            try:
                run.out_dir.mkdir(parents=True, exist_ok=True)
                tmp = run.out_dir / "upload_status.json.tmp"
                tmp.write_text(json.dumps(self.public(run), indent=1), encoding="utf-8")
                tmp.replace(run.out_dir / "upload_status.json")
            except OSError:
                pass

    def tick(self):
        """One periodic pass for the job that is running (the loop thread calls this every UPLOAD_EVERY_S)."""
        if not self.enabled or self.stopping:
            return
        cur = current()
        run = self.runs.get(cur.name) if cur else None
        if run and not run.final_done and not run.refused:
            self.run_pass(run, final=False)

    def loop(self):
        while True:
            time.sleep(UPLOAD_EVERY_S)
            try:
                self.tick()
            except Exception as e:                       # noqa: BLE001 - the loop must never die
                sys.stderr.write(f"runner: upload pass failed: {type(e).__name__}\n")

    def finish(self, job):
        """After the process has exited: the last pass, with everything not yet uploaded."""
        run = self.runs.get(job.name) if self.enabled else None
        if run is None or run.final_done or self.stopping:
            return
        if run.refused:
            self.finalize(run, run.refused)
        elif not self.run_pass(run, final=True, wait_s=UPLOAD_PASS_TIMEOUT_S + 30):
            self.finalize(run, "the upload pass in flight did not end")   # cannot happen: the child is killed at its timeout

    def abort(self, job, why):
        """The trainer never started: nothing will be uploaded, so the upload ends failed now (not waiting for ever)."""
        run = self.runs.get(job.name) if self.enabled else None
        if run is None or run.final_done:
            return
        run.error = why
        self.finalize(run, why)

    def before_exit(self, reason, budget_s):
        """At a time or idle limit: wait for a pass in flight, make the last pass, kill the child when the budget ends."""
        if not self.enabled:
            return
        end = time.time() + max(0.0, budget_s)
        self.stopping = True
        run = self.last
        if run is None or run.final_done:
            return
        if not self.pass_lock.acquire(timeout=max(0.0, end - time.time())):
            self.timed_out = True                        # the pass in flight ran out of grace: stop its child
            kill_tree(self.proc)
            if self.pass_lock.acquire(timeout=15):
                self.pass_lock.release()
            self.finalize(run, "time limit")
            return
        self.pass_lock.release()
        if end - time.time() > 1 and not run.refused:
            self.run_pass(run, final=True, deadline=end)
        self.finalize(run, "time limit")

    def finalize(self, run, why):
        """The last pass is over (or was cut off): whatever is not uploaded failed, and the status is saved."""
        with self.state_lock:
            for r in run.recs.values():
                if r.state in ("pending", "failed"):
                    r.state, r.error = "failed", r.error or why
            run.final_done = True
            run.in_flight = False
        self.persist(run)

    # ---- one pass
    def scan(self, run, final):
        """Find the files, apply the caps and the safety rules, pick the ones to send. Never touches a local file."""
        now = time.time()
        d = run.out_dir
        found = []                                       # (rel, local path, kind)
        try:
            with os.scandir(d) as it:
                tops = sorted(e.name for e in it)
        except OSError:
            tops = []
        for n in tops:
            if n.lower().endswith(".safetensors"):
                found.append((n, d / n, "checkpoint"))
        try:
            with os.scandir(d / "samples") as it:
                samples = sorted(e.name for e in it)
        except OSError:
            samples = []
        for n in samples:
            if Path(n).suffix.lower() in SAMPLE_EXT:
                found.append((f"samples/{n}", d / "samples" / n, "sample"))
        cfg = CONFIG_ROOT / f"{run.job_name}.json"
        if os.path.lexists(cfg):
            found.append(("config.json", cfg, "config"))
        seen, picked = set(), []
        for rel, local, kind in found:
            key = str(local)
            seen.add(key)
            rec = run.recs.get(key)
            try:
                st = os.lstat(local)
            except OSError:
                if rec is not None and rec.state in ("pending", "failed"):
                    rec.state, rec.error = "gone", ("gone after a failed upload" if rec.state == "failed" else None)
                continue
            if rec is None:
                rec = run.recs[key] = Rec(rel, f"{run.prefix}/{rel}", local, kind)
                shown = "config.json" if kind == "config" else local.name
                if not UPLOAD_NAME_RE.match(shown):
                    rec.state, rec.error = "skipped", "name not allowed"
                elif not stat.S_ISREG(st.st_mode) or not inside(CONFIG_ROOT if kind == "config" else d, local):
                    rec.state, rec.error = "skipped", "not a regular file inside the job folder"
                elif len(run.recs) > UPLOAD_COUNT_MAX:
                    rec.state, rec.error = "skipped", f"more than {UPLOAD_COUNT_MAX} files in this run"
            if rec.state not in ("pending", "failed"):
                continue
            why = self.cap_reason(run, rec, st.st_size)          # the size caps count on every pass, with today's size
            if why:
                run.total_bytes -= rec.counted
                rec.counted = 0
                rec.state, rec.error = "skipped", why
                continue
            run.total_bytes += st.st_size - rec.counted
            rec.counted = st.st_size
            sig = (st.st_size, st.st_mtime_ns)
            stable = rec.sig == sig and now - st.st_mtime >= UPLOAD_SETTLE_S
            rec.sig = sig
            if final or stable:
                picked.append(rec)
        for key, rec in run.recs.items():
            if key not in seen and rec.state in ("pending", "failed"):
                rec.state, rec.error = "gone", ("gone after a failed upload" if rec.state == "failed" else None)
        return picked

    @staticmethod
    def cap_reason(run, rec, size):
        """Why a file of this size may not go (per-file and per-run caps), else None."""
        if size > UPLOAD_FILE_MAX:
            return f"larger than {UPLOAD_FILE_MAX} bytes"
        if run.total_bytes - rec.counted + size > UPLOAD_RUN_MAX:
            return f"this run would pass {UPLOAD_RUN_MAX} bytes"
        return None

    def hash_files(self, run, picked, final=False):
        """The runner's own streaming sha256 and size of each file, before it goes out. A checkpoint whose safetensors
        header does not match its size is still being written, or was cut off: it waits, and in the final pass it
        fails as incomplete (no link)."""
        ready = []
        for rec in picked:
            try:
                if rec.kind == "checkpoint" and not safetensors_complete(rec.local):
                    if final:
                        rec.state, rec.error = "failed", "incomplete file"
                    continue
                if rec.hashed_sig != rec.sig:
                    why = self.cap_reason(run, rec, rec.sig[0])         # the caps again, right before the upload
                    if why:
                        run.total_bytes -= rec.counted
                        rec.counted = 0
                        rec.state, rec.error = "skipped", why
                        continue
                    digest = sha256_file(rec.local)
                    after = os.lstat(rec.local)
                    if (after.st_size, after.st_mtime_ns) != rec.sig:     # it changed while we read it: next pass
                        rec.sig = (after.st_size, after.st_mtime_ns)
                        continue
                    rec.sha256, rec.hashed_sig, rec.size = digest, rec.sig, rec.sig[0]
                ready.append(rec)
            except FileNotFoundError:
                rec.state, rec.error = "gone", ("gone after a failed upload" if rec.state == "failed" else None)
            except OSError as e:
                rec.state, rec.error = "failed", f"cannot read ({type(e).__name__})"
        return ready

    def run_pass(self, run, final, deadline=None, wait_s=0.0):
        """One pass: pick, hash, one commit through the child, read its result lines. False when another pass runs
        (the final pass of a job waits up to wait_s for it: the job ended while a periodic pass was in flight)."""
        if not self.enabled:
            return False
        got = self.pass_lock.acquire(timeout=wait_s) if wait_s else self.pass_lock.acquire(blocking=False)
        if not got:
            return False
        picked = []
        if run.final_done and not final:             # a periodic pass that was waiting behind the final one
            self.pass_lock.release()
            return False
        try:
            with self.state_lock:
                run.in_flight = True
            self.persist(run)
            try:
                picked = self.hash_files(run, self.scan(run, final), final)
                if picked:
                    self.send(run, picked, deadline, final)
            except Exception as e:                       # noqa: BLE001 - a pass must never break the runner
                sys.stderr.write(f"runner: upload pass failed: {type(e).__name__}\n")
                for rec in picked:
                    if rec.state == "pending":
                        rec.state, rec.error = "failed", f"upload pass failed ({type(e).__name__})"
        finally:
            with self.state_lock:
                run.in_flight = False
            self.pass_lock.release()
        if final and not self.timed_out:
            self.finalize(run, "time limit" if deadline is not None else "not uploaded")
        self.persist(run)
        touch()
        return True

    def child_env(self):
        env = dict(os.environ)
        for key in (*CHILD_ENV_DROP, *UPLOAD_CHILD_ENV_DROP):
            env.pop(key, None)
        env.update(HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_DISABLE_PROGRESS_BARS="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
                   PYTHONUNBUFFERED="1")
        return env

    def send(self, run, recs, deadline, final=False):
        """Hand the files to the child (the token goes on its stdin, once) and record each file's result line.
        The child checks the token kind (whoami) only until one check was accepted; after a failed check the runner waits
        out a back-off (the final pass excepted) and gives up after TOKEN_CHECK_TRIES, so whoami is not called every pass."""
        check = not (self.token_ok or UPLOAD_ALLOW_CLASSIC)
        if check and not final and time.time() < self.check_retry_at:
            for rec in recs:                                     # closed meanwhile; picked again by the next pass
                rec.state, rec.error = "failed", self.check_error
            return
        spec = {"check_token": check, "token": self.token, "repo": self.repo, "message": f"{run.job_name}: {len(recs)} files",
                "allow_public": UPLOAD_ALLOW_PUBLIC, "allow_classic": UPLOAD_ALLOW_CLASSIC,
                "files": [{"local": str(r.local), "path": r.path, "sha256": r.sha256, "size": r.size} for r in recs]}
        timeout = UPLOAD_PASS_TIMEOUT_S if deadline is None else max(1.0, min(UPLOAD_PASS_TIMEOUT_S, deadline - time.time()))
        kw = {"start_new_session": True} if os.name != "nt" else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        results, why, data, checked = {}, None, b"", False
        try:
            proc = subprocess.Popen(upload_command(), env=self.child_env(), stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=RUN_CWD, **kw)
        except (OSError, ValueError) as e:
            proc, why = None, f"could not start the upload ({type(e).__name__})"
        if proc is not None:
            self.proc = proc
            try:
                data, _ = proc.communicate(json.dumps(spec).encode("utf-8"), timeout=timeout)
            except subprocess.TimeoutExpired:
                kill_tree(proc)
                try:
                    data, _ = proc.communicate(timeout=5)
                except (subprocess.TimeoutExpired, OSError, ValueError):
                    data = b""
                why = "time limit" if deadline is not None else "timeout"
            except (OSError, ValueError):
                kill_tree(proc)
                why = "upload process failed"
            finally:
                self.proc = None
            if self.timed_out:
                why = "time limit"
            for raw in data.decode("utf-8", "replace").splitlines():
                try:
                    obj = json.loads(raw)
                except ValueError:
                    continue
                if isinstance(obj, dict) and isinstance(obj.get("path"), str):
                    results[obj["path"]] = obj
                elif isinstance(obj, dict) and obj.get("token_check") == "ok":
                    checked = True
            if why is None and proc.returncode not in (0, None) and not results:
                why = f"upload process exited with code {proc.returncode}"
        spec = None
        ok = failed = 0
        for rec in recs:
            obj = results.get(rec.path)
            commit = obj.get("commit") if obj else None
            if obj and obj.get("state") == "done" and isinstance(commit, str) and HEX40_RE.match(commit):
                rec.state, rec.error = "done", None
                rec.url = f"{HF_BASE_URL}/{self.repo}/resolve/{commit}/{rec.path}"
                ok += 1
                continue
            error = obj.get("error") if obj else None
            if not (isinstance(error, str) and re.fullmatch(r"[A-Za-z0-9_ .:()-]{1,120}", error)):
                error = why or ("failed" if obj else "no result")
            rec.state, rec.error = "failed", self.redact(error)
            failed += 1
            m = REFUSED_RE.search(rec.error)
            if rec.error.startswith("refused: ") and run.refused is None:     # not private, or a classic token
                run.refused = rec.error
            elif m and run.refused is None:
                run.refused = f"token or repo refused (HTTP {m.group(1)})"
        if check and checked:
            self.token_ok, self.check_fails = True, 0
        elif check and not run.refused:                          # no answer about the token: closed, retried with back-off
            self.check_fails += 1
            self.check_error = next((r.error for r in recs if r.error and r.error.startswith("token check failed")), "token check failed")
            if self.check_fails >= TOKEN_CHECK_TRIES:
                run.refused = "refused: the token check failed"
            else:
                self.check_retry_at = time.time() + TOKEN_CHECK_WAITS[min(self.check_fails - 1, len(TOKEN_CHECK_WAITS) - 1)]
        if run.refused:
            with self.state_lock:
                run.error = run.refused
                for rec in recs:
                    if rec.state == "failed":
                        rec.error = run.refused
        print(f"runner: upload {len(recs)} files: {ok} done, {failed} failed", flush=True)


UPLOADER = Uploader("", "", enabled=False)


def after_run(job):
    """Called at the end of Job._pump, after the final status is saved: the last upload pass."""
    UPLOADER.finish(job)


def before_exit(reason, budget_s):
    """Called by stop_for_limit just before the runner exits for a time or idle limit: the last pass inside a grace."""
    UPLOADER.before_exit(reason, budget_s)


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
                    "config_path": self.config_path, "upload": UPLOADER.status(self.name)}

    def persist(self):
        snap = self.snapshot()
        snap.pop("checkpoints"), snap.pop("samples"), snap.pop("upload")     # the upload keeps its own file
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
            try:
                UPLOADER.abort(self, "the trainer did not start")     # not "waiting" for ever
            except Exception as e:                   # noqa: BLE001
                sys.stderr.write(f"runner: upload abort failed: {type(e).__name__}\n")
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
UPLOAD_LOCK = threading.Lock()


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
    upload = stored_upload(out_dir)
    return {"name": name, "status": status, "phase": status, "step": snap.get("step", 0), "total_steps": snap.get("total_steps"),
            "it_per_s": None, "eta_s": None, "loss": snap.get("loss"), "log_tail": tail or snap.get("log_tail", []),
            "checkpoints": ck, "samples": sm, "started": snap.get("started"), "finished": snap.get("finished"),
            "returncode": snap.get("returncode"), "error": snap.get("error"), "config_path": snap.get("config_path"), "upload": upload}


def stored_upload(out_dir):
    """The upload block of a run this runner process didn't start: upload_status.json, with an unfinished state turned
    into failed (nobody is uploading any more)."""
    if not UPLOADER.enabled:
        return {"state": "off", "repo": None, "error": None, "files": []}
    try:
        up = json.loads((Path(out_dir) / "upload_status.json").read_text(encoding="utf-8"))
        if isinstance(up, dict) and up.get("state") in ("off", "waiting", "uploading", "done", "failed"):
            if up["state"] in ("waiting", "uploading"):
                up["state"], up["error"] = "failed", "the trainer restarted before the upload ended"
            return up
    except (OSError, ValueError):
        pass
    return {"state": "failed", "repo": UPLOADER.repo, "error": "no upload status was saved for this run", "files": []}


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
# `config` arrives as a JSON object, is checked key by key against this allowlist and is written out again from the
# parsed object (json.dumps), so what is checked is exactly what ai-toolkit reads. No text is ever scanned: a text
# check and a YAML parser disagree about escapes, quoting and flow style. Any key not listed is refused, which keeps
# out every other path key (control_path, mask_path, ...) and push_to_hub / hf_repo_id. The keys are the ones the app
# writes (webapp/app/training.py config_dict); a new ai-toolkit option is added here on purpose, not by accident.
CONFIG_MODELS = ("Tongyi-MAI/Z-Image-Turbo",)        # remote model names the runner may download
CONFIG_ADAPTERS = ("ostris/zimage_turbo_training_adapter/zimage_turbo_training_adapter_v2.safetensors",)
_QTYPES = ("one_of", ("qfloat8", "qint8", "qint4", "float8", "uint8"))
_DTYPES = ("one_of", ("bf16", "fp16", "float16", "float32", "fp32"))
CONFIG_SCHEMA = {
    "job": ("one_of", ("extension",)),
    "config": {
        "name": "job_name",
        "process": ("list", 1, {
            "type": ("one_of", ("sd_trainer",)),
            "training_folder": "output_path",
            "device": ("one_of", ("cuda:0", "cuda")),
            "trigger_word": "str",
            "network": {"type": ("one_of", ("lora",)), "linear": "int", "linear_alpha": "int"},
            "save": {"dtype": _DTYPES, "save_every": "int", "max_step_saves_to_keep": "int"},
            "datasets": ("list", 8, {
                "folder_path": "dataset_path", "caption_ext": ("one_of", ("txt",)), "caption_dropout_rate": "num",
                "cache_latents_to_disk": "bool", "resolution": ("list", 8, "int"), "flip_x": "bool", "flip_y": "bool",
                "is_reg": "bool", "num_repeats": "int"}),
            "train": {
                "batch_size": "int", "steps": "int", "gradient_accumulation": "int", "gradient_accumulation_steps": "int",
                "train_unet": "bool", "train_text_encoder": "bool", "gradient_checkpointing": "bool",
                "noise_scheduler": ("one_of", ("flowmatch", "ddpm")), "timestep_type": "str", "content_or_style": "str",
                "optimizer": ("one_of", ("adamw8bit", "adamw", "adafactor", "prodigy")), "lr": "num",
                "optimizer_params": {"weight_decay": "num"}, "dtype": _DTYPES, "cache_text_embeddings": "bool",
                "unload_text_encoder": "bool", "skip_first_sample": "bool", "disable_sampling": "bool"},
            "model": {
                "name_or_path": ("one_of", CONFIG_MODELS), "arch": ("one_of", ("zimage",)),
                "assistant_lora_path": ("one_of", CONFIG_ADAPTERS), "quantize": "bool", "qtype": _QTYPES,
                "quantize_te": "bool", "qtype_te": _QTYPES, "low_vram": "bool"},
            "sample": {
                "sampler": ("one_of", ("flowmatch", "ddpm")), "sample_every": "int", "width": "int", "height": "int",
                "prompts": ("list", 40, "str"), "neg": "str", "seed": "int", "walk_seed": "bool", "guidance_scale": "num",
                "sample_steps": "int"},
        }),
    },
    "meta": {"name": "job_name", "version": "str"},
}


def confined_path(root, value, what):
    """None when `value` is an absolute path inside `root` (symlinks and .. resolved), else the reason."""
    if not isinstance(value, str) or not value or "\0" in value or not os.path.isabs(value):
        return f"{what} must be an absolute path inside {root}"
    if not inside(root, value):
        return f"{what} must be inside {root}"
    return None


def check_value(value, spec, where, job):
    """None when `value` fits `spec`, else a sentence naming `where`."""
    if isinstance(spec, dict):
        if not isinstance(value, dict):
            return f"{where}: must be an object"
        for k, v in value.items():
            if k not in spec:
                return f"{where}.{k}: not an allowed key"
            why = check_value(v, spec[k], f"{where}.{k}", job)
            if why:
                return why
        return None
    if isinstance(spec, tuple) and spec[0] == "list":
        if not isinstance(value, list) or len(value) > spec[1] or (spec[1] == 1 and len(value) != 1):
            return f"{where}: must be a list of {'exactly 1' if spec[1] == 1 else f'at most {spec[1]}'} item(s)"
        for n, v in enumerate(value):
            why = check_value(v, spec[2], f"{where}[{n}]", job)
            if why:
                return why
        return None
    if isinstance(spec, tuple) and spec[0] == "one_of":
        if not isinstance(value, str) or value not in spec[1]:
            return f"{where}: must be one of {', '.join(spec[1])}"
        return None
    if spec == "job_name":
        return None if value == job else f"{where}: must equal the job name {job}"
    if spec == "output_path":
        return confined_path(OUTPUT_ROOT, value, where)
    if spec == "dataset_path":
        why = confined_path(DATASETS_ROOT, value, where)
        if why:
            return why
        if Path(value).resolve() == Path(DATASETS_ROOT).resolve() or inside(Path(DATASETS_ROOT) / ".incoming", value):
            return f"{where}: must name one dataset folder inside {DATASETS_ROOT}"
        return None
    if spec == "str":
        return None if isinstance(value, str) and len(value) <= 4000 and "\0" not in value else f"{where}: must be text (4000 characters at most)"
    if spec == "int":
        return None if isinstance(value, int) and not isinstance(value, bool) and abs(value) <= 10**9 else f"{where}: must be a whole number"
    if spec == "num":
        ok = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        return None if ok else f"{where}: must be a number"
    if spec == "bool":
        return None if isinstance(value, bool) else f"{where}: must be true or false"
    return f"{where}: not allowed"


def check_inline_config(cfg, name):
    """None when the config object may run, else the reason (the first problem found)."""
    if not isinstance(cfg, dict):
        return "config must be a JSON object (the ai-toolkit config), not text"
    try:
        size = len(json.dumps(cfg, allow_nan=False).encode("utf-8"))
    except (ValueError, TypeError):
        return "config must be plain JSON"
    if size > CONFIG_MAX:
        return f"config is larger than {CONFIG_MAX} bytes"
    if not isinstance(cfg.get("config"), dict) or cfg["config"].get("name") != name:
        return f"config.name must equal the job name {name}"
    return check_value(cfg, CONFIG_SCHEMA, "$", name)


def config_total_steps(cfg):
    try:
        n = cfg["config"]["process"][0]["train"]["steps"]
        return n if isinstance(n, int) and n > 0 else None
    except (KeyError, IndexError, TypeError):
        return None


# ----------------------------------------------------------------- time limit and idle limit
def boot_id():
    """Who this clock belongs to: RUNNER_BOOT_ID (the app may set one) or RUNPOD_POD_ID. Empty on the PC."""
    return (os.environ.get("RUNNER_BOOT_ID") or os.environ.get("RUNPOD_POD_ID") or "").strip()


def read_boot():
    """(first_start, deadline) saved by THIS pod's earlier run, else None. A boot.json written by another pod (a network
    volume reused by a new machine) is ignored, so an old deadline never ends or shortens a new pod."""
    try:
        d = json.loads((STATE_DIR / "boot.json").read_text(encoding="utf-8"))
        if str(d.get("pod") or "") != boot_id():
            return None
        return float(d["first_start"]), float(d["deadline"])
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def write_boot():
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = STATE_DIR / "boot.json.tmp"
        tmp.write_text(json.dumps({"pod": boot_id(), "first_start": FIRST_START, "deadline": DEADLINE}), encoding="utf-8")
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
        if v < 0 or not math.isfinite(v):                  # no NaN, no inf (time.sleep(inf) kills the watchdog)
            raise ValueError
        return v
    except ValueError:
        sys.stderr.write(f"runner: {name} must be a finite number of seconds, 0 or more\n")
        raise SystemExit(2)


def configure():
    """Validate the token and limits from the environment (exit 2 on a bad value) and start the clock."""
    global TIME_LIMIT_S, TIME_LIMIT_MAX_S, IDLE_LIMIT_S, WATCH_EVERY_S, DEADLINE, FIRST_START
    if TOKEN and not TOKEN_RE.match(TOKEN):
        sys.stderr.write(f"runner: RUNNER_TOKEN must be at least {TOKEN_MIN} characters from A-Z a-z 0-9 _ -\n")
        raise SystemExit(2)
    TIME_LIMIT_MAX_S = env_number("RUNNER_TIME_LIMIT_MAX_S", 10800.0)
    TIME_LIMIT_S = env_number("RUNNER_TIME_LIMIT_S", 10800.0 if TOKEN else 0.0)
    IDLE_LIMIT_S = env_number("RUNNER_IDLE_LIMIT_S", 1800.0 if TOKEN else 0.0)
    WATCH_EVERY_S = min(env_number("RUNNER_WATCH_EVERY_S", 5.0) or 5.0, 60.0)
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
    configure_upload()
    touch()


def configure_upload():
    """Validate the Hugging Face upload settings (exit 2 on a bad one, naming the variable and never its value)."""
    global UPLOAD_EVERY_S, UPLOAD_SETTLE_S, UPLOAD_GRACE_S, UPLOADER
    if UPLOAD_ALLOW_CLASSIC and os.environ.get("RUNPOD_POD_ID"):
        sys.stderr.write("runner: RUNNER_UPLOAD_ALLOW_CLASSIC must not be set on a rented machine (RUNPOD_POD_ID is set)\n")
        raise SystemExit(2)
    UPLOAD_EVERY_S = env_number("RUNNER_UPLOAD_EVERY_S", 60.0) or 60.0
    UPLOAD_SETTLE_S = env_number("RUNNER_UPLOAD_SETTLE_S", 30.0)
    UPLOAD_GRACE_S = env_number("RUNNER_UPLOAD_GRACE_S", 300.0)
    if UPLOAD_GRACE_S > UPLOAD_GRACE_MAX_S:
        sys.stderr.write(f"runner: RUNNER_UPLOAD_GRACE_S must be at most {UPLOAD_GRACE_MAX_S:.0f} (the grace must end "
                         f"before the image's backstop at the time limit + 600 s)\n")
        raise SystemExit(2)
    if not HF_WRITE_TOKEN and not HF_REPO:
        UPLOADER = Uploader("", "", enabled=False)
        return
    for name, value in (("HF_WRITE_TOKEN", HF_WRITE_TOKEN), ("HF_REPO", HF_REPO)):
        if not value:
            sys.stderr.write(f"runner: {name} is not set (HF_WRITE_TOKEN and HF_REPO go together)\n")
            raise SystemExit(2)
    if not HF_TOKEN_RE.match(HF_WRITE_TOKEN):
        sys.stderr.write("runner: HF_WRITE_TOKEN is not a Hugging Face token (hf_ and at least 20 letters or digits)\n")
        raise SystemExit(2)
    if not HF_REPO_RE.match(HF_REPO):
        sys.stderr.write("runner: HF_REPO must look like owner/name (letters, digits, _ . -)\n")
        raise SystemExit(2)
    if not TOKEN:
        sys.stderr.write("runner: HF_WRITE_TOKEN needs RUNNER_TOKEN: the upload is for a rented machine only\n")
        raise SystemExit(2)
    UPLOADER = Uploader(HF_WRITE_TOKEN, HF_REPO)


def stop_for_limit(reason):
    """Stop the training child, save its status and leave with code 3. Needs no request from the app."""
    print(f"runner: {reason}", flush=True)
    UPLOADER.stopping = True                         # after_run of the cancelled job must not start a pass without a budget
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
        before_exit(reason, budget_s=UPLOAD_GRACE_S)
    except Exception as e:                           # noqa: BLE001
        sys.stderr.write(f"runner: before_exit failed: {type(e).__name__}\n")
    sys.stdout.flush()
    sys.stderr.flush()
    _exit(3)


def watchdog():
    """Checks the deadline and the idle limit. If this loop itself fails, the runner exits (code 3) instead of going on
    with no limit at all."""
    try:
        while True:
            time.sleep(WATCH_EVERY_S)
            now = time.time()
            with STATE_LOCK:
                deadline = DEADLINE
            if deadline is not None and now >= deadline:
                stop_for_limit("time limit reached")
            if IDLE_LIMIT_S > 0:
                if current() or UPLOADER.busy():          # an upload in flight is activity
                    touch()
                elif now - LAST_ACTIVITY >= IDLE_LIMIT_S:
                    stop_for_limit("idle limit reached")
    except BaseException as e:                        # noqa: BLE001  fail closed
        try:
            sys.stderr.write(f"runner: the watchdog failed ({type(e).__name__}); exiting\n")
            sys.stderr.flush()
        finally:
            _exit(3)


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
def _no_constant(name):
    raise ValueError(name)                            # NaN / Infinity are not JSON


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
        raw = (self.headers.get("Content-Length") or "0").strip()
        if not LENGTH_RE.match(raw):                  # negative, non-numeric or non-ASCII digits
            return None
        n = int(raw)
        if not n:
            return {}
        if n > BODY_MAX:
            return None
        try:
            d = json.loads(self.rfile.read(n) or b"{}", parse_constant=_no_constant)
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
        if raw is None or not LENGTH_RE.match(raw.strip()):
            return self.send_json(411, {"error": "Content-Length required"}, close=True)
        length = int(raw)
        if length > UPLOAD_MAX_BYTES:
            return self.send_json(413, {"error": f"upload is larger than {UPLOAD_MAX_BYTES} bytes"}, close=True)
        if busy_with(name):
            return self.send_json(409, {"error": f"a run named {name} is queued or running"}, close=True)
        if not UPLOAD_LOCK.acquire(blocking=False):   # one upload at a time: the free-space check is per upload
            return self.send_json(409, {"error": "another upload is running"}, close=True)
        try:
            return self.take_upload(name, length)
        finally:
            UPLOAD_LOCK.release()

    def take_upload(self, name, length):
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
            why = check_inline_config(inline, name)
            if why:
                return self.send_json(400, {"error": why})
            cfg_path = CONFIG_ROOT / f"{name}.json"
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
                    tmp = CONFIG_ROOT / f"{name}.json.tmp"
                    tmp.write_text(json.dumps(inline, indent=2, allow_nan=False), encoding="utf-8")
                    tmp.replace(cfg_path)
                except OSError as e:
                    return self.send_json(500, {"error": f"could not write the config ({type(e).__name__})"})
            total = b.get("total_steps") or (config_total_steps(inline) if inline is not None else config_steps(cfg_path))
            j = Job(name, cfg_path, out_dir, int(total) if total else None)
            JOBS[name] = j
        UPLOADER.begin(j)
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
    if UPLOADER.enabled:
        limits.append(f"uploads to {UPLOADER.repo}")
        threading.Thread(target=UPLOADER.loop, daemon=True, name="uploader").start()
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
