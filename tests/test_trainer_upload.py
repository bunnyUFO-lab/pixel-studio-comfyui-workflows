"""CPU tests for the trainer's checkpoint upload to Hugging Face (ss-m10t8): trainer/cloud/hf_upload.py against a fake
HfApi in process, and trainer/runner.py with a fake trainer and a fake upload child. No network beyond 127.0.0.1 (the
runner listens there, never on 0.0.0.0), no real token, no Hugging Face call. Every subprocess has a timeout and is
killed in `finally`.

    python tests/test_trainer_upload.py        (also collected by pytest as test_trainer_upload)
"""
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "trainer", "cloud"))
import test_trainer_runner as T  # noqa: E402  (its Box, call, TOKEN ... and the runner module)
import hf_upload as H  # noqa: E402

runner = T.runner
RESULTS = []
HF_TOKEN = "hf_" + "abcdefghij" * 3                      # made up: hf_ and 30 letters
REPO = "test/sprite-studio-loras"
MODEL = "Tongyi-MAI/Z-Image-Turbo"


def check(cond, msg):
    RESULTS.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + msg)


def sft(tag, n=200):
    """A whole safetensors file: 8-byte header length, a JSON header with one tensor of n bytes, n bytes of data."""
    head = json.dumps({"w": {"dtype": "U8", "shape": [n], "data_offsets": [0, n]}}).encode()
    return len(head).to_bytes(8, "little") + head + (str(tag).encode() * n)[:n]


def alive(pid):
    """Is the process there? (os.kill(pid, 0) would END it on Windows.)"""
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True, timeout=20).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


# ================================================================== fakes
def good_fine(repo, perms=("repo.access.read", "repo.content.read", "repo.write"), name=None):
    """whoami's fine-grained block for the owner's token: no global permissions, one scope, one model."""
    return {"canReadGatedRepos": False, "global": [],
            "scoped": [{"entity": {"_id": "6700000000000000", "type": "model", "name": name or repo}, "permissions": list(perms)}]}


class HfHubHTTPError(Exception):
    """Looks like the library's error: a class name, a message that holds the token, and response.status_code."""

    def __init__(self, status, message=""):
        super().__init__(message)
        self.response = types.SimpleNamespace(status_code=status)


class FakeApi:
    """Records its calls; can fail create_commit with given statuses first; can answer another size or sha256."""

    def __init__(self, fail=(), size_delta=0, lfs_sha=None, no_lfs=False, leak="", private=True, role="fineGrained", fine="default"):
        self.calls, self.fail, self.size_delta, self.lfs_sha, self.no_lfs, self.leak = [], list(fail), size_delta, lfs_sha, no_lfs, leak
        self.private, self.role = private, role
        self.fine = good_fine(REPO) if fine == "default" else fine
        self.commits = []
        self.local = {}

    def whoami(self):
        self.calls.append(("whoami",))
        return {"name": "owner", "auth": {"type": "access_token", "accessToken": {"role": self.role, "fineGrained": self.fine}}}

    def model_info(self, repo_id):
        self.calls.append(("model_info", repo_id))
        return types.SimpleNamespace(private=self.private)

    def create_commit(self, repo_id, repo_type, revision, operations, commit_message):
        self.calls.append(("create_commit", repo_id, repo_type, revision, [o.path_in_repo for o in operations], commit_message))
        if self.fail:
            raise HfHubHTTPError(self.fail.pop(0), "401 Client Error for url https://huggingface.co/api/x Bearer " + self.leak)
        for o in operations:
            self.local[o.path_in_repo] = o.path_or_fileobj
        oid = hashlib.sha1(str(len(self.commits)).encode()).hexdigest()
        self.commits.append(oid)
        return types.SimpleNamespace(oid=oid)

    def get_paths_info(self, repo_id, paths, repo_type, revision, expand):
        self.calls.append(("get_paths_info", repo_id, tuple(paths), revision, expand))
        out = []
        for p in paths:
            src = self.local[p]
            data = Path(src).read_bytes() if isinstance(src, (str, os.PathLike)) else src
            lfs = None if self.no_lfs else types.SimpleNamespace(sha256=self.lfs_sha or hashlib.sha256(data).hexdigest())
            out.append(types.SimpleNamespace(path=p, size=len(data) + self.size_delta, lfs=lfs))
        return out


class FakeHub(FakeApi):
    """What selftest talks to: refs, branches, whoami, model_info, a store of uploaded bytes per branch."""

    def __init__(self, private=True, role="fineGrained", download_fails=False, main="a" * 40, **kw):
        super().__init__(private=private, role=role, **kw)
        self.download_fails = download_fails
        self.branches = {"main": main}
        self.store = {}
        self.main = main

    def list_repo_refs(self, repo_id, repo_type):
        return types.SimpleNamespace(branches=[types.SimpleNamespace(name=n, target_commit=c) for n, c in self.branches.items()])

    def create_branch(self, repo_id, branch, revision, repo_type):
        self.calls.append(("create_branch", branch, revision))
        self.branches[branch] = revision

    def delete_branch(self, repo_id, branch, repo_type):
        self.calls.append(("delete_branch", branch))
        self.branches.pop(branch, None)

    def create_commit(self, repo_id, repo_type, revision, operations, commit_message):
        self.calls.append(("create_commit", repo_id, repo_type, revision, [o.path_in_repo for o in operations], commit_message))
        if self.fail:
            raise HfHubHTTPError(self.fail.pop(0))
        for o in operations:
            self.store[(revision, o.path_in_repo)] = o.path_or_fileobj
        return types.SimpleNamespace(oid="b" * 40)

    def download(self, repo, filename, revision, token, local_dir):
        self.calls.append(("download", revision, "anonymous" if token is False else "token"))
        if token is False:
            raise HfHubHTTPError(401)
        if self.download_fails:
            raise HfHubHTTPError(500)
        p = Path(local_dir) / "got.bin"
        p.write_bytes(self.store[(revision, filename)])
        return str(p)


def make_files(tmp, n=3):
    files = []
    for i in range(n):
        p = Path(tmp) / f"job_{i:09d}.safetensors"
        p.write_bytes(f"data-{i}".encode() * 100)
        data = p.read_bytes()
        files.append({"local": str(p), "path": f"z-image-turbo/job/{p.name}", "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)})
    return files


def upload_lines(api, files, token=HF_TOKEN, waits=None):
    buf = io.StringIO()
    out = H.Out(buf, [token])
    H.run_upload({"token": token, "repo": REPO, "files": files, "message": "job: %d files" % len(files)}, api, out,
                 sleep=(waits.append if waits is not None else (lambda s: None)))
    return [json.loads(x) for x in buf.getvalue().splitlines()], buf.getvalue()


# ================================================================== hf_upload in process
def test_hf_upload_in_process():
    with tempfile.TemporaryDirectory() as tmp:
        files = make_files(tmp)
        api = FakeApi()
        lines, _ = upload_lines(api, files)
        commits = [c for c in api.calls if c[0] == "create_commit"]
        check(len(commits) == 1 and commits[0][1:4] == (REPO, "model", "main") and commits[0][4] == [f["path"] for f in files],
              "upload: one commit with every file, repo type model, revision main, paths in the <base model>/<job>/ layout")
        check(commits[0][5] == "job: 3 files", f"upload: commit message '<job>: <n> files' ({commits[0][5]})")
        infos = [c for c in api.calls if c[0] == "get_paths_info"]
        check(len(infos) == 1 and infos[0][3] == api.commits[0] and infos[0][4] is True, "upload: get_paths_info(expand=True) at the new commit")
        check([x["state"] for x in lines] == ["done"] * 3 and all(x["commit"] == api.commits[0] for x in lines), f"upload: one done line per file with the commit ({lines[0]})")
        # a retry on 500 twice, then done
        waits = []
        api = FakeApi(fail=[500, 500])
        lines, _ = upload_lines(api, files, waits=waits)
        check([x["state"] for x in lines] == ["done"] * 3 and len([c for c in api.calls if c[0] == "create_commit"]) == 3 and waits == [10, 30],
              f"retry: HTTP 500 twice, then done on the third try, after 10 s and 30 s ({waits})")
        waits = []
        api = FakeApi(fail=[500, 500, 500, 500])
        lines, _ = upload_lines(api, files, waits=waits)
        check(all(x["state"] == "failed" and x["error"] == "HfHubHTTPError 500" for x in lines) and waits == [10, 30, 90],
              f"retry: 4 tries (10, 30, 90 s) then failed with the class name and status only ({lines[0]})")
        for status in (401, 403, 404):
            api = FakeApi(fail=[status, status])
            waits = []
            lines, _ = upload_lines(api, files, waits=waits)
            check(len([c for c in api.calls if c[0] == "create_commit"]) == 1 and not waits and lines[0]["error"] == f"HfHubHTTPError {status}",
                  f"retry: HTTP {status} is not retried")
        # the checksum rules
        api = FakeApi(size_delta=1)
        lines, _ = upload_lines(api, files)
        check(all(x["state"] == "failed" and x["error"] == "checksum differs" for x in lines), "checks: another size on the hub -> failed: checksum differs")
        api = FakeApi(lfs_sha="0" * 64)
        lines, _ = upload_lines(api, files)
        check(all(x["state"] == "failed" and x["error"] == "checksum differs" for x in lines), "checks: another lfs.sha256 -> failed: checksum differs")
        api = FakeApi(no_lfs=True)
        small = [dict(f, path=f["path"].replace(".safetensors", ".json")) for f in files]
        lines, _ = upload_lines(api, small)
        check(all(x["state"] == "done" for x in lines), "checks: no lfs block on a small non-checkpoint file -> the size alone decides")
        # the token never reaches an output line: the exception text holds it, so does a path (the last guard redacts it)
        leaky = [dict(files[0], path="z-image-turbo/job/" + HF_TOKEN + ".safetensors")]
        api = FakeApi(fail=[500, 500, 500, 500], leak=HF_TOKEN)
        lines, text = upload_lines(api, leaky)
        check(HF_TOKEN not in text and "***" in text, f"token: an exception whose message holds the token, and a path that holds it -> not in the output ({text.strip()[:120]})")
        # main(): stdin in, result lines out, exit code
        spec = {"token": HF_TOKEN, "repo": REPO, "files": files, "message": "m"}
        out, err = io.StringIO(), io.StringIO()
        api = FakeApi()
        code = H.main(io.StringIO(json.dumps(spec) + "\n"), out, lambda tok: api, argv=["upload"], stderr=err)
        check(code == 0 and out.getvalue().count('"done"') == 3, f"main: stdin JSON line -> three done lines, exit 0 ({code})")
        out, err = io.StringIO(), io.StringIO()
        code = H.main(io.StringIO("not json\n"), out, lambda tok: api, argv=["upload"], stderr=err)
        check(code == 1 and HF_TOKEN not in err.getvalue(), "main: bad stdin -> exit 1")
        code = H.main(io.StringIO(json.dumps(spec) + "\n"), io.StringIO(), lambda tok: api, argv=["upload", "extra"], stderr=io.StringIO())
        check(code == 1, "main: an unknown argument -> exit 1")

        def boom(tok):
            raise HfHubHTTPError(500, "message with " + tok)
        out = io.StringIO()
        code = H.main(io.StringIO(json.dumps(spec) + "\n"), out, boom, argv=["upload"], stderr=io.StringIO())
        check(code == 1 and HF_TOKEN not in out.getvalue(), f"main: an exception while building the client -> exit 1, no token in the output ({out.getvalue().strip()})")


def selftest(hub, **spec_extra):
    buf = io.StringIO()
    spec = {"token": HF_TOKEN, "read_token": "hf_" + "klmnopqrst" * 3, "repo_name": "sprite-studio-loras", **spec_extra}
    ok = H.run_selftest(spec, hub, hub.download, H.Out(buf, [spec["token"], spec["read_token"]]), read_api=hub)
    return ok, [json.loads(x) for x in buf.getvalue().splitlines()], buf.getvalue()


def test_selftest_in_process():
    hub = FakeHub()
    ok, lines, text = selftest(hub)
    steps = {x["step"]: x for x in lines}
    check(ok and steps["whoami"]["owner"] == "owner" and steps["whoami"]["role"] == "fineGrained" and steps["whoami"]["repo"] == "owner/sprite-studio-loras",
          f"selftest: whoami prints the owner and the token's role ({steps['whoami']})")
    names = [c[0] for c in hub.calls]
    check(names.index("create_branch") < names.index("create_commit") < names.index("download") < names.index("delete_branch"),
          f"selftest: branch created, file uploaded, read back, branch deleted ({names})")
    branch = next(c[1] for c in hub.calls if c[0] == "create_branch")
    check(re.fullmatch(r"selftest-[0-9a-f]{12}", branch) and [c for c in hub.calls if c[0] == "create_commit"][0][3] == branch
          and all(c[3] == branch for c in hub.calls if c[0] == "create_commit"), "selftest: the file goes only to the throwaway branch, never to main")
    path = [c for c in hub.calls if c[0] == "create_commit"][0][4][0]
    check(re.fullmatch(r"_selftest/[0-9a-f]{12}\.txt", path) and len(hub.store[(branch, path)]) == 64, f"selftest: _selftest/<12 hex>.txt with 64 characters ({path})")
    check(("download", branch, "anonymous") in hub.calls and ("download", branch, "token") in hub.calls, "selftest: the read-back with the read token, and a refused one without a token")
    check(branch not in hub.branches and hub.branches == {"main": "a" * 40} and steps["main unchanged"]["ok"], "selftest: the branch is gone and main is at the same commit")
    check(HF_TOKEN not in text and "klmnopqrst" not in text, "selftest: neither token is in the output")
    # the read-back fails: the branch is deleted anyway and the result is a failure
    hub = FakeHub(download_fails=True)
    ok, lines, _ = selftest(hub)
    check(not ok and "delete_branch" in [c[0] for c in hub.calls] and list(hub.branches) == ["main"], "selftest: when the read-back fails the branch is still deleted, result False")
    # the upload itself fails
    hub = FakeHub(fail=[500])
    ok, lines, _ = selftest(hub)
    check(not ok and list(hub.branches) == ["main"], "selftest: when the upload fails the branch is still deleted")
    # a public repository: stop before writing anything
    hub = FakeHub(private=False)
    ok, lines, _ = selftest(hub)
    check(not ok and not any(c[0] in ("create_branch", "create_commit") for c in hub.calls), "selftest: a public repository stops it before anything is written")
    # a branch that cannot be deleted is named, loudly
    hub = FakeHub()
    hub.delete_branch = lambda **kw: (_ for _ in ()).throw(HfHubHTTPError(500))
    ok, lines, text = selftest(hub)
    check(not ok and "BRANCH LEFT" in text, "selftest: a failed delete says BRANCH LEFT and names the branch")
    # main through the command line entry (stdin JSON, exit code)
    hub = FakeHub()
    out = io.StringIO()
    code = H.main(io.StringIO(json.dumps({"token": HF_TOKEN, "read_token": "hf_" + "klmnopqrst" * 3, "repo_name": "sprite-studio-loras"}) + "\n"),
                  out, lambda tok: hub, argv=["selftest"], stderr=io.StringIO(), download=hub.download)
    check(code == 0 and '"result"' in out.getvalue(), f"selftest: main(['selftest']) exits 0 when all held ({code})")


# ================================================================== runner end to end
TRAINER = r'''
import json, os, sys, time
from pathlib import Path
cfg = sys.argv[-1]
name = os.path.splitext(os.path.basename(cfg))[0]
steps, step_s = int(os.environ.get("FAKE_STEPS", "6")), float(os.environ.get("FAKE_STEP_S", "0.3"))
mode = os.environ.get("FAKE_TRAIN_MODE", "early")
out = Path(os.environ["FAKE_OUT"]) / name
(out / "samples").mkdir(parents=True, exist_ok=True)
print("hf-token-in-env: " + ("yes" if "HF_WRITE_TOKEN" in os.environ else "no"), flush=True)
def save(step, final=False):
    f = out / ("%s.safetensors" % name if final else "%s_%09d.safetensors" % (name, step))
    n = 200
    head = json.dumps({"w": {"dtype": "U8", "shape": [n], "data_offsets": [0, n]}}).encode()
    f.write_bytes(len(head).to_bytes(8, "little") + head + (("%s-%s" % (name, step)).encode() * n)[:n])
    (out / "samples" / ("1700000%d_%09d_0.jpg" % (step, step))).write_bytes(b"\xff\xd8\xff\xe0" + b"s" * 50 + bytes([step % 250]))
for i in range(1, steps + 1):
    print("%s: %3d%%|#####     | %d/%d [00:0%d<00:01, 1.00s/it, lr: 1.0e-04 loss: 4.1e-01]" % (name, 100 * i // steps, i, steps, i % 10), flush=True)
    if mode == "early" and i in (2, 4):
        save(i)
    if mode == "long" and i in (2, 4):
        save(i)
    time.sleep(step_s)
if mode in ("early", "late"):
    if mode == "late":
        save(2); save(4)
    save(steps, final=True)
'''

UPLOADER_SCRIPT = r'''
import hashlib, json, os, sys, time
spec = json.loads(sys.stdin.readline())
mode = os.environ.get("FAKE_UP_MODE", "ok")
tok = spec["token"]
rec = {"argv": sys.argv[1:], "env_keys": sorted(os.environ), "token_in_env_values": any(tok in v for v in os.environ.values()),
       "token_in_argv": any(tok in a for a in sys.argv), "token_on_stdin": bool(tok), "paths": [f["path"] for f in spec["files"]],
       "repo": spec["repo"], "message": spec["message"], "allow_public": spec.get("allow_public"), "allow_classic": spec.get("allow_classic")}
with open(os.environ["FAKE_UP_LOG"], "a") as f:
    f.write(json.dumps(rec) + "\n")
if mode == "sleep":
    with open(os.environ["FAKE_UP_PID"], "w") as f:
        f.write(str(os.getpid()))
    time.sleep(60)
if mode == "slow":
    time.sleep(float(os.environ.get("FAKE_UP_SLOW_S", "5")))
commit = hashlib.sha1(str(time.time_ns()).encode()).hexdigest()
for f in spec["files"]:
    p = f["path"]
    if mode == "refuse":
        print(json.dumps({"path": p, "state": "failed", "error": "HfHubHTTPError 401"}), flush=True)
    elif mode == "refuse_private":
        print(json.dumps({"path": p, "state": "failed", "error": "refused: the repository is not private"}), flush=True)
    elif mode == "leak":
        print(tok, flush=True)
        sys.stderr.write(tok + "\n")
        print(json.dumps({"path": p, "state": "failed", "error": "HfHubHTTPError " + tok}), flush=True)
    else:
        data = open(f["local"], "rb").read()
        good = hashlib.sha256(data).hexdigest() == f["sha256"] and len(data) == f["size"]
        print(json.dumps({"path": p, "state": "done", "commit": commit} if good else {"path": p, "state": "failed", "error": "checksum differs"}), flush=True)
'''


class UBox(T.Box):
    """The runner test box plus the upload fakes; HF variables in this PC's environment never leak in."""

    def __init__(self):
        super().__init__()
        self.trainer = self.dir / "fake_upload_trainer.py"
        self.trainer.write_text(TRAINER, encoding="utf-8")
        self.uploader = self.dir / "fake_uploader.py"
        self.uploader.write_text(UPLOADER_SCRIPT, encoding="utf-8")
        self.uplog, self.uppid = self.dir / "uploader_calls.jsonl", self.dir / "uploader.pid"

    def env(self, extra=None):
        env = {k: v for k, v in super().env().items() if not k.startswith("HF_")}
        env.update(RUN_CMD=json.dumps([sys.executable, str(self.trainer)]), FAKE_OUT=str(self.output),
                   RUNNER_UPLOAD_CMD=json.dumps([sys.executable, str(self.uploader)]), FAKE_UP_LOG=str(self.uplog),
                   FAKE_UP_PID=str(self.uppid), RUNNER_UPLOAD_EVERY_S="0.2", RUNNER_UPLOAD_SETTLE_S="0")
        env.update(extra or {})
        return env

    def start_hf(self, **extra):
        return self.start(token=T.TOKEN, HF_WRITE_TOKEN=HF_TOKEN, HF_REPO=REPO, **extra)

    def calls(self):
        try:
            return [json.loads(x) for x in self.uplog.read_text().splitlines()]
        except OSError:
            return []

    def post_job(self, p, name, steps=6, model=MODEL):
        cfg = T.inline_config(self, name, steps)
        cfg["config"]["process"][0]["model"] = {"name_or_path": model}
        return T.call(p, "POST", "/jobs", {"name": name, "config": cfg}, token=T.TOKEN)

    def upload_status(self, p, name):
        s, js, _ = T.call(p, "GET", f"/jobs/{name}", token=T.TOKEN)
        return (js or {}).get("upload") or {}

    def wait_upload(self, p, name, want=("done", "failed"), secs=30):
        end = time.time() + secs
        up = {}
        while time.time() < end:
            try:
                up = self.upload_status(p, name)
            except OSError:
                break
            if up.get("state") in want:
                return up
            time.sleep(0.2)
        return up


def hits(token, texts):
    return [i for i, t in enumerate(texts) if token in t]


def test_runner_end_to_end():
    box = UBox()
    try:
        p = box.start_hf(HF_TOKEN="hf_" + "gatedmodel" * 3)
        check(box.wait_up(p), "e2e: the runner starts with HF_WRITE_TOKEN and HF_REPO")
        s, js, _ = box.post_job(p, "lora")
        check(s == 201 and js["upload"]["state"] in ("waiting", "uploading") and js["upload"]["repo"] == REPO, f"e2e: POST /jobs answers with an upload block ({js.get('upload')})")
        check(T.wait_status(p, "lora", T.TOKEN) == "done", "e2e: the fake run reaches done")
        up = box.wait_upload(p, "lora")
        files = {f["file"]: f for f in up.get("files", [])}
        cks = [f for n, f in files.items() if n.endswith(".safetensors")]
        check(up.get("state") == "done" and len(cks) == 3 and all(f["state"] == "done" for f in cks), f"e2e: upload.state done, three checkpoints done ({up.get('state')}, {up.get('error')})")
        ok = True
        for f in cks:
            data = (box.output / "lora" / f["file"]).read_bytes()
            ok = ok and f["sha256"] == hashlib.sha256(data).hexdigest() and f["size"] == len(data)
            ok = ok and f["path"] == "z-image-turbo/lora/" + f["file"] and re.fullmatch(r"https://huggingface\.co/test/sprite-studio-loras/resolve/[0-9a-f]{40}/z-image-turbo/lora/[A-Za-z0-9_.-]+", f["url"]) is not None
        check(ok, "e2e: each checkpoint has sha256 and size equal to hashlib's, the layout path and a commit-id url")
        names = sorted(files)
        check("config.json" in files and files["config.json"]["path"] == "z-image-turbo/lora/config.json" and files["config.json"]["state"] == "done"
              and sum(n.startswith("samples/") for n in names) == len(list((box.output / "lora" / "samples").glob("*.jpg"))) == 3 and "runner.log" not in names,
              f"e2e: the config goes as config.json, the samples under samples/, runner.log never ({names})")
        sha_config = hashlib.sha256((box.configs / "lora.json").read_bytes()).hexdigest()
        check(files["config.json"]["sha256"] == sha_config, "e2e: config.json is CONFIG_ROOT/<job>.json")
        calls = box.calls()
        check(len(calls) >= 2 and all(c["message"] == "lora: %d files" % len(c["paths"]) and c["repo"] == REPO for c in calls),
              f"e2e: the files went in several passes (one child, one commit each) while the run trained ({len(calls)} passes)")
        sent = sorted(x for c in calls for x in c["paths"])
        check(len(sent) == len(set(sent)), "e2e: no file was sent twice")
        s, js, _ = T.call(p, "GET", "/jobs/lora", token=T.TOKEN)
        check("hf-token-in-env: no" in "\n".join(js["log_tail"]), "e2e: the training child does not have HF_WRITE_TOKEN")
        check(all(c["token_on_stdin"] and not c["token_in_env_values"] and not c["token_in_argv"] for c in calls),
              "e2e: the upload child got the token on stdin, not in its arguments or environment values")
        banned = {"HF_WRITE_TOKEN", "HF_TOKEN", "RUNNER_TOKEN", "HF_REPO", "RUNPOD_API_KEY"}
        keys = set(calls[0]["env_keys"]) if calls else set()
        check(not (banned & keys) and {"HF_HUB_DISABLE_TELEMETRY", "HF_HUB_DISABLE_PROGRESS_BARS", "HF_HUB_DISABLE_IMPLICIT_TOKEN"} <= keys,
              f"e2e: the child's environment has none of {sorted(banned)} and the three HF_HUB_DISABLE_* flags")
        # the token in no log, file or reply
        replies = []
        for path, tok in (("/health", None), ("/health", T.TOKEN), ("/jobs", T.TOKEN), ("/jobs/lora", T.TOKEN)):
            s, js, _ = T.call(p, "GET", path, token=tok)
            replies.append(json.dumps(js))
        texts = [box.logs(p)] + replies
        for name in ("runner.log", "runner_status.json", "upload_status.json"):
            texts.append((box.output / "lora" / name).read_text(encoding="utf-8", errors="replace"))
        check((box.output / "lora" / "upload_status.json").exists(), "e2e: upload_status.json is kept next to the run")
        check(not hits(HF_TOKEN, texts) and not hits("gatedmodel", texts[:1]),
              f"e2e: the token is in none of runner stdout/stderr, runner.log, runner_status.json, upload_status.json, /health (with and without the runner token), /jobs, /jobs/<n> ({hits(HF_TOKEN, texts)})")
        check("runner: upload " in box.logs(p) and re.search(r"runner: upload \d+ files: \d+ done, 0 failed", box.logs(p)) is not None, "e2e: one log line per pass: 'runner: upload <n> files: <ok> done, <failed> failed'")
        # the files on disk are untouched
        check(len(list((box.output / "lora").glob("*.safetensors"))) == 3, "e2e: the local checkpoints are still there")
    finally:
        box.close()


def test_runner_restart_keeps_status():
    box = UBox()
    try:
        p = box.start_hf()
        box.wait_up(p)
        box.post_job(p, "kept")
        up = box.wait_upload(p, "kept")
        p.kill()
        p.wait(10)
        p2 = box.start_hf()
        box.wait_up(p2)
        again = box.upload_status(p2, "kept")
        check(up.get("state") == "done" and again.get("state") == "done" and len(again.get("files", [])) == len(up["files"]),
              f"restart: a restarted runner still reports the upload from upload_status.json ({again.get('state')})")
    finally:
        box.close()


def test_runner_off_and_bad_variables():
    box = UBox()
    try:
        p = box.start(token=T.TOKEN)
        box.wait_up(p)
        s, js, _ = box.post_job(p, "plain")
        T.wait_status(p, "plain", T.TOKEN)
        s, js, _ = T.call(p, "GET", "/jobs/plain", token=T.TOKEN)
        check(js["upload"] == {"state": "off", "repo": None, "error": None, "files": []} and not box.uplog.exists() and not (box.output / "plain" / "upload_status.json").exists(),
              f"off: no HF variables -> upload.state off, the uploader never runs ({js['upload']['state']})")
    finally:
        box.close()
    cases = [
        ("only HF_WRITE_TOKEN", dict(token=T.TOKEN, HF_WRITE_TOKEN=HF_TOKEN), "HF_REPO"),
        ("only HF_REPO", dict(token=T.TOKEN, HF_REPO=REPO), "HF_WRITE_TOKEN"),
        ("a bad token form", dict(token=T.TOKEN, HF_WRITE_TOKEN="hf_TOP-SECRET-short", HF_REPO=REPO), "HF_WRITE_TOKEN"),
        ("a token with a newline", dict(token=T.TOKEN, HF_WRITE_TOKEN=HF_TOKEN + "\n", HF_REPO=REPO), "HF_WRITE_TOKEN"),
        ("a bad repo", dict(token=T.TOKEN, HF_WRITE_TOKEN=HF_TOKEN, HF_REPO="no-owner"), "HF_REPO"),
        ("a bad repo with a space", dict(token=T.TOKEN, HF_WRITE_TOKEN=HF_TOKEN, HF_REPO="a/b c"), "HF_REPO"),
        ("HF variables without RUNNER_TOKEN", dict(HF_WRITE_TOKEN=HF_TOKEN, HF_REPO=REPO), "RUNNER_TOKEN"),
        ("RUNNER_UPLOAD_GRACE_S=600", dict(token=T.TOKEN, HF_WRITE_TOKEN=HF_TOKEN, HF_REPO=REPO, RUNNER_UPLOAD_GRACE_S="600"), "RUNNER_UPLOAD_GRACE_S"),
        ("RUNNER_UPLOAD_GRACE_S=600 without HF variables", dict(token=T.TOKEN, RUNNER_UPLOAD_GRACE_S="600"), "RUNNER_UPLOAD_GRACE_S"),
    ]
    for label, kw, word in cases:
        box = UBox()
        try:
            p = box.start(RUNNER_HOST="127.0.0.1", **kw)
            code = T.wait_exit(p, 15)
            text = box.logs(p)
            check(code == 2 and word in text and HF_TOKEN not in text and "TOP-SECRET" not in text,
                  f"start: {label} -> exit 2, the message names {word} and never the value ({code})")
        finally:
            box.close()
    box = UBox()
    try:
        p = box.start(token=T.TOKEN, HF_WRITE_TOKEN=HF_TOKEN, HF_REPO=REPO, RUNNER_UPLOAD_GRACE_S="480", RUNNER_TIME_LIMIT_S="3600")
        check(box.wait_up(p), "start: RUNNER_UPLOAD_GRACE_S=480 (the most) starts")
    finally:
        box.close()


# ================================================================== caps and safety (the Uploader in process)
def test_caps_and_safety():
    saved = {k: getattr(runner, k) for k in ("UPLOAD_CMD", "UPLOAD_SETTLE_S", "UPLOAD_FILE_MAX", "CONFIG_ROOT")}
    box = UBox()
    try:
        runner.UPLOAD_CMD = json.dumps([sys.executable, str(box.uploader)])
        runner.UPLOAD_SETTLE_S, runner.UPLOAD_FILE_MAX = 0.0, 1 << 20
        runner.CONFIG_ROOT = box.configs
        os.environ.update(FAKE_UP_LOG=str(box.uplog), FAKE_UP_MODE="ok")
        out = box.output / "capjob"
        (out / "samples").mkdir(parents=True)
        box.configs.mkdir(parents=True)
        cfg = box.configs / "capjob.json"
        cfg.write_text(json.dumps(T.inline_config(box, "capjob")))
        job = types.SimpleNamespace(name="capjob", config_path=str(cfg), output_dir=out)
        up = runner.Uploader(HF_TOKEN, REPO)
        up.begin(job)
        st = up.status("capjob")
        check(st["state"] == "failed" and st["error"] == "unknown base model" and not box.uplog.exists(),
              f"caps: a config without a known base model -> upload failed: unknown base model, nothing sent ({st['state']}, {st['error']})")
        cfg.write_text(json.dumps(T.with_process(box, "capjob", model={"name_or_path": MODEL})))
        up = runner.Uploader(HF_TOKEN, REPO)
        up.begin(job)
        (out / "capjob_000000001.safetensors").write_bytes(sft("ok"))
        (out / "bad name.safetensors").write_bytes(sft("x"))
        (out / "capjob_big.safetensors").write_bytes(sft("B", 3 << 20))
        (out / "samples" / "s1.jpg").write_bytes(b"\xff\xd8\xff" + b"s" * 20)
        (out / "samples" / "notes.txt").write_text("not a sample")
        outside = box.dir / "outside.safetensors"
        outside.write_bytes(b"secret" * 10)
        linked = True
        try:
            os.symlink(outside, out / "linked.safetensors")
        except (OSError, NotImplementedError):
            linked = False
        def snapshot():
            return {p.name: p.read_bytes() for p in out.rglob("*") if p.is_file() and not p.is_symlink() and not p.name.startswith("upload_status")}
        before = snapshot()
        up.run_pass(up.runs["capjob"], final=False)          # first sight: nothing is stable yet
        check(not box.uplog.exists(), "caps: a file seen once is not sent (it must be unchanged between two passes)")
        up.run_pass(up.runs["capjob"], final=False)
        files = {f["file"]: f for f in up.status("capjob")["files"]}
        check(files["capjob_000000001.safetensors"]["state"] == "done", "caps: a stable checkpoint is sent on the second pass")
        check(files["bad name.safetensors"]["state"] == "skipped" and files["bad name.safetensors"]["error"] == "name not allowed", "caps: a name with a space is skipped")
        check(files["capjob_big.safetensors"]["state"] == "skipped" and "larger than" in files["capjob_big.safetensors"]["error"], "caps: a 3 MiB file against a 1 MiB cap is skipped with the reason")
        if linked:
            check(files["linked.safetensors"]["state"] == "skipped" and "regular" in files["linked.safetensors"]["error"], "caps: a symlinked .safetensors is skipped")
        else:
            check(True, "caps: symlinks cannot be made here (Windows without the privilege): the symlink case waits for a Linux run")
        check("samples/notes.txt" not in files and files["samples/s1.jpg"]["state"] == "done", "caps: only images count as samples")
        sent = [x for c in box.calls() for x in c["paths"]]
        check(all("bad name" not in x and "big" not in x and "linked" not in x for x in sent), f"caps: none of the skipped files was sent ({sent})")
        # a checkpoint deleted between passes
        (out / "capjob_000000002.safetensors").write_bytes(sft("two"))
        up.run_pass(up.runs["capjob"], final=False)
        (out / "capjob_000000002.safetensors").unlink()
        up.run_pass(up.runs["capjob"], final=False)
        files = {f["file"]: f for f in up.status("capjob")["files"]}
        check(files["capjob_000000002.safetensors"]["state"] == "gone", "caps: a checkpoint deleted before it was sent -> gone")
        after = snapshot()
        check(all(after.get(k) == v for k, v in before.items()) and set(after) == set(before),
              "caps: the uploader changed, moved or deleted no local file")
        check(outside.read_bytes() == b"secret" * 10, "caps: the file the symlink pointed at is untouched")
        # a final pass takes the rest and settles the status
        up.run_pass(up.runs["capjob"], final=True)
        st = up.status("capjob")
        check(st["state"] == "failed" and st["error"] in ("name not allowed", f"larger than {1 << 20} bytes", "not a regular file inside the job folder"),
              f"caps: a checkpoint that was skipped (name, size, link) means the run is failed, saying why ({st['state']}, {st['error']})")
        # a gone checkpoint (ai-toolkit removed an old step save before it was sent) does not fail the run
        out3 = box.output / "capjob3"
        out3.mkdir()
        job3 = types.SimpleNamespace(name="capjob3", config_path=str(cfg), output_dir=out3)
        up3 = runner.Uploader(HF_TOKEN, REPO)
        up3.begin(job3)
        (out3 / "capjob3_000000001.safetensors").write_bytes(sft("1"))
        (out3 / "capjob3.safetensors").write_bytes(sft("2"))
        up3.run_pass(up3.runs["capjob3"], final=False)
        (out3 / "capjob3_000000001.safetensors").unlink()
        up3.run_pass(up3.runs["capjob3"], final=True)
        st3 = up3.status("capjob3")
        check(st3["state"] == "done" and sorted(f["state"] for f in st3["files"] if f["file"].endswith(".safetensors")) == ["done", "gone"],
              f"caps: a gone old step save does not fail the run, the final file is done ({st3['state']})")
        # the per-run count cap
        runner.UPLOAD_COUNT_MAX, old = 3, runner.UPLOAD_COUNT_MAX
        try:
            up = runner.Uploader(HF_TOKEN, REPO)
            out2 = box.output / "capjob2"
            out2.mkdir()
            job2 = types.SimpleNamespace(name="capjob2", config_path=str(cfg), output_dir=out2)
            for i in range(5):
                (out2 / f"capjob2_{i:09d}.safetensors").write_bytes(sft("c%d" % i, 20))
            up.begin(job2)
            up.run_pass(up.runs["capjob2"], final=True)
            states = [f["state"] for f in up.status("capjob2")["files"]]
            check(states.count("skipped") == 2 and states.count("done") == 3, f"caps: more files than the count cap -> the extra ones are skipped ({states})")
        finally:
            runner.UPLOAD_COUNT_MAX = old
    finally:
        for k, v in saved.items():
            setattr(runner, k, v)
        for k in ("FAKE_UP_LOG", "FAKE_UP_MODE"):
            os.environ.pop(k, None)
        box.close()


# ================================================================== refusal, leaks
def test_refused_and_leaky_uploader():
    box = UBox()
    try:
        p = box.start_hf(FAKE_UP_MODE="refuse")
        box.wait_up(p)
        box.post_job(p, "refused", steps=10)
        T.wait_status(p, "refused", T.TOKEN, secs=30)
        up = box.wait_upload(p, "refused")
        check(up.get("state") == "failed" and up.get("error") == "token or repo refused (HTTP 401)", f"refused: HTTP 401 -> upload failed: token or repo refused ({up.get('state')}, {up.get('error')})")
        check(len(box.calls()) == 1, f"refused: later passes are skipped, the child ran once ({len(box.calls())})")
    finally:
        box.close()
    box = UBox()
    try:
        p = box.start_hf(FAKE_UP_MODE="leak")
        box.wait_up(p)
        box.post_job(p, "leaky")
        T.wait_status(p, "leaky", T.TOKEN)
        up = box.wait_upload(p, "leaky")
        texts = [box.logs(p), json.dumps(up)]
        for name in ("runner.log", "runner_status.json", "upload_status.json"):
            texts.append((box.output / "leaky" / name).read_text(encoding="utf-8", errors="replace"))
        s, js, _ = T.call(p, "GET", "/jobs/leaky", token=T.TOKEN)
        texts.append(json.dumps(js))
        check(up.get("state") == "failed" and not hits(HF_TOKEN, texts),
              f"leak: an upload child that prints the token (stdout line, stderr, an error field) -> it is in no log, file or reply ({hits(HF_TOKEN, texts)}, {up.get('state')})")
    finally:
        box.close()


# ================================================================== limit and idle
def test_limit_mid_upload():
    box = UBox()
    try:
        p = box.start_hf(RUNNER_TIME_LIMIT_S="3", RUNNER_UPLOAD_GRACE_S="2", FAKE_UP_MODE="sleep", FAKE_TRAIN_MODE="long", FAKE_STEPS="2000", FAKE_STEP_S="0.1")
        box.wait_up(p)
        t0 = time.time()
        s, js, _ = box.post_job(p, "limited", steps=2000)
        code = T.wait_exit(p, 20)
        took = time.time() - t0
        pid = T.read_pid(box.uppid, 1)
        check(s == 201 and code == 3 and took < 20, f"limit: a 60 s upload at the limit -> the runner exits 3 within 20 s ({code}, {took:.1f}s)")
        check(pid and not alive(pid), f"limit: the upload child is gone ({pid})")
        st = json.loads((box.output / "limited" / "upload_status.json").read_text(encoding="utf-8"))
        cks = [f for f in st["files"] if f["file"].endswith(".safetensors")]
        check(st["state"] == "failed" and cks and all(f["state"] == "failed" and "time limit" in f["error"] for f in cks) and "time limit" in (st["error"] or ""),
              f"limit: upload_status.json says failed with 'time limit' for what was not sent ({st['state']}, {st['error']})")
        check(len(list((box.output / "limited").glob("*.safetensors"))) == 2, "limit: the local checkpoints still exist")
        check(HF_TOKEN not in json.dumps(st) + box.logs(p), "limit: no token in the status or the log")
    finally:
        box.close()
    box = UBox()
    try:
        p = box.start_hf(RUNNER_TIME_LIMIT_S="3", RUNNER_UPLOAD_GRACE_S="5", FAKE_TRAIN_MODE="long", FAKE_STEPS="2000", FAKE_STEP_S="0.1")
        box.wait_up(p)
        box.post_job(p, "fastlimit", steps=2000)
        code = T.wait_exit(p, 20)
        st = json.loads((box.output / "fastlimit" / "upload_status.json").read_text(encoding="utf-8"))
        cks = [f for f in st["files"] if f["file"].endswith(".safetensors")]
        check(code == 3 and st["state"] == "failed" and "final checkpoint" in (st["error"] or "") and len(cks) == 2 and all(f["state"] == "done" for f in cks),
              f"limit: with a fast uploader both step saves are sent, and without a final file the status is failed and says so ({code}, {st['state']}, {st['error']})")
    finally:
        box.close()


def test_idle_with_upload_in_flight():
    box = UBox()
    try:
        p = box.start_hf(RUNNER_TIME_LIMIT_S="3600", RUNNER_IDLE_LIMIT_S="2", FAKE_UP_MODE="slow", FAKE_UP_SLOW_S="5", FAKE_TRAIN_MODE="late", FAKE_STEPS="3", FAKE_STEP_S="0.1")
        box.wait_up(p)
        t0 = time.time()
        box.post_job(p, "idlejob", steps=3)
        time.sleep(4.2)                                  # past the 2 s idle limit, the 5 s upload still runs; no request in between
        up_file = box.output / "idlejob" / "upload_status.json"
        mid = json.loads(up_file.read_text(encoding="utf-8")) if up_file.exists() else {}
        check(p.poll() is None and mid.get("state") == "uploading", f"idle: an upload in flight keeps the runner up past its 2 s idle limit ({p.poll()}, {mid.get('state')}, {time.time() - t0:.1f}s)")
        code = T.wait_exit(p, 20)
        check(code == 3 and "idle limit reached" in box.logs(p), f"idle: ... and once the upload ended the idle limit stops it (exit {code})")
        st = json.loads(up_file.read_text(encoding="utf-8"))
        check(st["state"] == "done", f"idle: the upload finished before the exit ({st['state']})")
    finally:
        box.close()



# ================================================================== review fixes (PR 25 review, 10 Nov 2026)
def pass_box(box, name="rv", **uploader_env):
    """An Uploader in process with the fake uploader as its child, and a job folder for it."""
    runner.UPLOAD_CMD = json.dumps([sys.executable, str(box.uploader)])
    runner.UPLOAD_SETTLE_S = 0.0
    runner.CONFIG_ROOT = box.configs
    os.environ.update(FAKE_UP_LOG=str(box.uplog), FAKE_UP_MODE="ok", **uploader_env)
    out = box.output / name
    (out / "samples").mkdir(parents=True)
    box.configs.mkdir(parents=True, exist_ok=True)
    cfg = box.configs / f"{name}.json"
    cfg.write_text(json.dumps(T.with_process(box, name, model={"name_or_path": MODEL})))
    job = types.SimpleNamespace(name=name, config_path=str(cfg), output_dir=out)
    up = runner.Uploader(HF_TOKEN, REPO)
    up.begin(job)
    return up, job, out


def restore(saved, box):
    for k, v in saved.items():
        setattr(runner, k, v)
    for k in ("FAKE_UP_LOG", "FAKE_UP_MODE", "FAKE_UP_SLOW_S"):
        os.environ.pop(k, None)
    box.close()


SAVED = ("UPLOAD_CMD", "UPLOAD_SETTLE_S", "UPLOAD_FILE_MAX", "UPLOAD_RUN_MAX", "CONFIG_ROOT", "UPLOAD_PASS_TIMEOUT_S")


def test_m1_final_pass_after_a_pass_in_flight():
    import threading
    saved = {k: getattr(runner, k) for k in SAVED}
    box = UBox()
    try:
        up, job, out = pass_box(box, "m1")
        run = up.runs["m1"]
        (out / "m1_000000001.safetensors").write_bytes(sft("one"))
        up.run_pass(run, final=False)                              # first sight
        os.environ.update(FAKE_UP_MODE="slow", FAKE_UP_SLOW_S="2")
        t = threading.Thread(target=up.run_pass, args=(run,), kwargs={"final": False})
        t.start()
        end = time.time() + 10
        while not up.busy() and time.time() < end:
            time.sleep(0.02)
        (out / "m1.safetensors").write_bytes(sft("final"))        # the job ends while that slow pass is in flight
        up.finish(job)
        t.join(10)
        st = up.status("m1")
        files = {f["file"]: f["state"] for f in st["files"]}
        check(st["state"] == "done" and files.get("m1.safetensors") == "done" and files.get("m1_000000001.safetensors") == "done",
              f"M1: the job ends while a pass is in flight -> a final pass still runs, the final LoRA is sent, the state leaves waiting ({st['state']}, {files})")
    finally:
        restore(saved, box)


def test_m2_done_needs_the_final_checkpoint():
    saved = {k: getattr(runner, k) for k in SAVED}
    box = UBox()
    try:
        up, job, out = pass_box(box, "m2a")
        up.run_pass(up.runs["m2a"], final=True)                    # nothing at all was written
        st = up.status("m2a")
        check(st["state"] == "failed" and "m2a.safetensors" in (st["error"] or ""), f"M2: a run with no checkpoint is failed and the error names the final file ({st['state']}, {st['error']})")
        up, job, out = pass_box(box, "m2b")
        (out / "m2b_000000001.safetensors").write_bytes(sft("s"))
        up.run_pass(up.runs["m2b"], final=False)
        up.run_pass(up.runs["m2b"], final=True)
        st = up.status("m2b")
        check(st["state"] == "failed" and "final checkpoint" in (st["error"] or ""), f"M2: only a step save is done, the final file never written -> failed ({st['state']}, {st['error']})")
        up, job, out = pass_box(box, "m2c")
        (out / "m2c.safetensors").write_bytes(sft("f"))
        up.run_pass(up.runs["m2c"], final=False)
        up.run_pass(up.runs["m2c"], final=False)                   # sent
        up.runs["m2c"].recs[str(out / "m2c.safetensors")].state = "gone"
        up.finalize(up.runs["m2c"], "x")
        st = up.status("m2c")
        check(st["state"] == "failed", f"M2: the final file marked gone never counts as done ({st['state']})")
        up, job, out = pass_box(box, "m2d")
        (out / "m2d.safetensors").write_bytes(sft("f"))
        up.run_pass(up.runs["m2d"], final=True)
        st = up.status("m2d")
        check(st["state"] == "done" and st["error"] is None, f"M2: the final file uploaded and verified -> done ({st['state']})")
    finally:
        restore(saved, box)


def test_m3_incomplete_checkpoint():
    saved = {k: getattr(runner, k) for k in SAVED}
    box = UBox()
    try:
        up, job, out = pass_box(box, "m3")
        whole = sft("w", 400)
        (out / "m3_000000001.safetensors").write_bytes(whole[:-10])      # cut off: the header promises 10 more bytes
        (out / "m3.safetensors").write_bytes(b"not safetensors at all")  # no header
        run = up.runs["m3"]
        up.run_pass(run, final=False)
        up.run_pass(run, final=False)
        check(not [x for c in box.calls() for x in c["paths"] if x.endswith(".safetensors")], "M3: a checkpoint cut off, or without a header, is not sent by a periodic pass even when its size is stable")
        (out / "m3_000000001.safetensors").write_bytes(whole)             # the writer finishes: now it goes
        up.run_pass(run, final=False)
        up.run_pass(run, final=False)
        files = {f["file"]: f for f in up.status("m3")["files"]}
        check(files["m3_000000001.safetensors"]["state"] == "done", f"M3: once the file is whole it is sent ({files['m3_000000001.safetensors']['state']})")
        up.run_pass(run, final=True)
        st = up.status("m3")
        f = {x["file"]: x for x in st["files"]}["m3.safetensors"]
        check(f["state"] == "failed" and f["error"] == "incomplete file" and f["url"] is None and st["state"] == "failed",
              f"M3: the final pass reports a file that fails the check as incomplete, with no link, and the run is not done ({f['state']}, {f['error']}, {st['state']})")
        sent = [x for c in box.calls() for x in c["paths"]]
        check(not any(x.endswith("/m3.safetensors") for x in sent), "M3: the incomplete final file was never sent")
        # the check itself
        d = Path(box.dir)
        cases = {"whole": (sft("a"), True), "one byte short": (sft("a")[:-1], False), "one byte long": (sft("a") + b"x", False),
                 "empty": (b"", False), "huge header length": ((1 << 60).to_bytes(8, "little") + b"{}", False),
                 "header not json": ((5).to_bytes(8, "little") + b"nope!", False)}
        for name, (data, want) in cases.items():
            (d / "c.safetensors").write_bytes(data)
            check(runner.safetensors_complete(d / "c.safetensors") is want, f"M3: safetensors_complete: {name} -> {want}")
    finally:
        restore(saved, box)


def test_m4_trainer_that_cannot_start():
    box = UBox()
    try:
        p = box.start_hf(RUN_CMD=json.dumps([str(box.dir / "no-such-trainer")]))
        box.wait_up(p)
        box.post_job(p, "nostart")
        T.wait_status(p, "nostart", T.TOKEN)
        up = box.wait_upload(p, "nostart", secs=10)
        check(up.get("state") == "failed" and "did not start" in (up.get("error") or ""), f"M4: a trainer that cannot start -> upload failed, not waiting for ever ({up.get('state')}, {up.get('error')})")
        s, js, _ = T.call(p, "GET", "/jobs/nostart", token=T.TOKEN)
        st = json.loads((box.output / "nostart" / "upload_status.json").read_text(encoding="utf-8"))
        check(js["status"] == "failed" and st["state"] == "failed", "M4: the status file says failed too")
        # a restarted runner with no upload_status.json for a finished run: failed, not waiting
        box.post_job(p, "lost")
        T.wait_status(p, "lost", T.TOKEN)
        box.wait_upload(p, "lost")
        p.kill()
        p.wait(10)
        (box.output / "lost" / "upload_status.json").unlink()
        p2 = box.start_hf()
        box.wait_up(p2)
        again = box.upload_status(p2, "lost")
        check(again.get("state") == "failed", f"M4: upload_status.json missing after a restart -> failed ({again.get('state')})")
    finally:
        box.close()


def test_m5_caps_on_every_pass():
    saved = {k: getattr(runner, k) for k in SAVED}
    box = UBox()
    try:
        runner.UPLOAD_FILE_MAX = 1 << 20
        up, job, out = pass_box(box, "m5")
        f = out / "m5_000000001.safetensors"
        f.write_bytes(sft("a", 10))
        run = up.runs["m5"]
        up.run_pass(run, final=False)                                 # first seen small
        f.write_bytes(sft("a", 3 << 20))                              # then it grows past the cap
        up.run_pass(run, final=False)
        up.run_pass(run, final=True)
        rec = {x["file"]: x for x in up.status("m5")["files"]}["m5_000000001.safetensors"]
        sent = [x for c in box.calls() for x in c["paths"] if x.endswith(".safetensors")]
        check(rec["state"] == "skipped" and "larger than" in rec["error"] and not sent, f"M5: a file first seen small that grows past the per-file cap is skipped, never sent ({rec['state']}, {sent})")
        counted = sum(r.counted for r in run.recs.values() if r.kind == "checkpoint")
        check(counted == 0, f"M5: the run total follows the file ({counted})")
        # the per-run cap, with growth
        runner.UPLOAD_FILE_MAX = 2 << 30
        runner.UPLOAD_RUN_MAX = 100000
        up, job, out = pass_box(box, "m5b")
        a, b = out / "m5b_000000001.safetensors", out / "m5b.safetensors"
        a.write_bytes(sft("a", 40000))
        b.write_bytes(sft("b", 10))
        run = up.runs["m5b"]
        up.run_pass(run, final=False)
        b.write_bytes(sft("b", 70000))                                # 40 k + 70 k is over the 100 k run cap
        up.run_pass(run, final=True)
        recs = {x["file"]: x for x in up.status("m5b")["files"]}
        check(recs["m5b.safetensors"]["state"] == "skipped" and "would pass" in recs["m5b.safetensors"]["error"] and recs["m5b_000000001.safetensors"]["state"] == "done",
              f"M5: a file that grows past the per-run cap is skipped, the earlier one stays done ({recs['m5b.safetensors']['state']})")
    finally:
        restore(saved, box)


def test_t2_private_repository():
    files = None
    with tempfile.TemporaryDirectory() as tmp:
        files = make_files(tmp, 2)
        api = FakeApi(private=False)
        lines, text = upload_lines(api, files)
        check(all(x["state"] == "failed" and x["error"] == "refused: the repository is not private" for x in lines) and len(lines) == 2
              and not any(c[0] == "create_commit" for c in api.calls), f"T2a: a public repository -> every file failed 'refused', nothing committed ({lines[0]})")
        check(api.calls[0][0] == "whoami" and ("model_info", REPO) in api.calls, "T2a: the check runs before the first commit")
        api = FakeApi(private=False)
        buf = io.StringIO()
        H.run_upload({"token": HF_TOKEN, "repo": REPO, "files": files, "message": "m", "allow_public": True}, api, H.Out(buf, [HF_TOKEN]), sleep=lambda s: None)
        check(buf.getvalue().count('"done"') == 2, "T2a: with the explicit setting a public repository is allowed")
        api = FakeApi()
        lines, _ = upload_lines(api, files)
        check([x["state"] for x in lines] == ["done", "done"], "T2a: a private repository is uploaded to")
        # a failing check is retried like the commit, a 404 is a refusal
        waits = []
        api = FakeApi()
        calls = {"n": 0}
        real = api.model_info

        def flaky(repo_id):
            calls["n"] += 1
            if calls["n"] < 3:
                raise HfHubHTTPError(500)
            return real(repo_id)
        api.model_info = flaky
        lines, _ = upload_lines(api, files, waits=waits)
        check(waits == [10, 30] and [x["state"] for x in lines] == ["done", "done"], f"T2a: the privacy check is retried on 500 ({waits})")


def test_t2_runner_stops_on_a_refusal_and_passes_the_settings():
    box = UBox()
    try:
        p = box.start_hf(FAKE_UP_MODE="refuse_private")
        box.wait_up(p)
        box.post_job(p, "pub", steps=10)
        T.wait_status(p, "pub", T.TOKEN, secs=30)
        up = box.wait_upload(p, "pub")
        check(up.get("state") == "failed" and up.get("error") == "refused: the repository is not private" and len(box.calls()) == 1,
              f"T2a: the runner reports the refusal and runs no more passes ({up.get('state')}, {up.get('error')}, {len(box.calls())} child runs)")
    finally:
        box.close()
    box = UBox()
    try:
        p = box.start_hf()
        box.wait_up(p)
        box.post_job(p, "flags")
        box.wait_upload(p, "flags")
        c = box.calls()
        check(c and all(x["allow_public"] is False and x["allow_classic"] is False for x in c), "T2: by default neither allow flag is set in the child's spec")
    finally:
        box.close()
    box = UBox()
    try:
        p = box.start_hf(RUNNER_UPLOAD_ALLOW_PUBLIC="1", RUNNER_UPLOAD_ALLOW_CLASSIC="1")
        box.wait_up(p)
        box.post_job(p, "flags2")
        box.wait_upload(p, "flags2")
        c = box.calls()
        check(c and all(x["allow_public"] is True and x["allow_classic"] is True for x in c), "T2: the two settings given at machine creation reach the child")
        check(not (set(c[0]["env_keys"]) & {"RUNNER_UPLOAD_ALLOW_PUBLIC", "RUNNER_UPLOAD_ALLOW_CLASSIC"}), "T2: and are not in the child's environment")
    finally:
        box.close()


def test_t3_endpoint_is_fixed():
    saved = {k: os.environ.get(k) for k in ("HF_ENDPOINT", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE")}
    try:
        os.environ["HF_ENDPOINT"] = "http://127.0.0.1:9"
        api = H.real_api(HF_TOKEN)
        check(api.endpoint == "https://huggingface.co", f"T3b: HF_ENDPOINT in the environment does not move the write token's host ({api.endpoint})")
        check(H.ENDPOINT == "https://huggingface.co" and H.TEST_ENDPOINT is None, "T3b: the endpoint is a constant, the test override is unset (a module value no environment variable or stdin field reaches)")
        src = (Path(ROOT) / "trainer" / "cloud" / "hf_upload.py").read_text(encoding="utf-8")
        check("HF_ENDPOINT\") or" not in src and "environ.get(\"HF_ENDPOINT" not in src, "T3b: hf_upload.py reads no endpoint from the environment")
        H.TEST_ENDPOINT = "http://127.0.0.1:9"
        try:
            check(H.real_api(HF_TOKEN).endpoint == "http://127.0.0.1:9", "T3b: the test-only override works when a test sets it in process")
        finally:
            H.TEST_ENDPOINT = None
        for k in saved:
            os.environ[k] = "http://127.0.0.1:9/x"
        old_environ = dict(os.environ)
        H.harden_env()
        check(not any(k in os.environ for k in saved), "T3b: the upload child drops HF_ENDPOINT and the CA bundle variables itself")
        os.environ.update(old_environ)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    box = UBox()
    try:
        p = box.start_hf(HF_ENDPOINT="http://127.0.0.1:9", REQUESTS_CA_BUNDLE="/x", CURL_CA_BUNDLE="/x", SSL_CERT_FILE="/x")
        box.wait_up(p)
        box.post_job(p, "ep")
        box.wait_upload(p, "ep")
        keys = set(box.calls()[0]["env_keys"])
        check(not (keys & {"HF_ENDPOINT", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE"}), f"T3b: the runner gives the child none of HF_ENDPOINT, REQUESTS_CA_BUNDLE, CURL_CA_BUNDLE, SSL_CERT_FILE ({sorted(keys)})")
    finally:
        box.close()


def test_t4_checksum_required():
    with tempfile.TemporaryDirectory() as tmp:
        files = make_files(tmp, 2)
        api = FakeApi(no_lfs=True)
        lines, _ = upload_lines(api, files)
        check(all(x["state"] == "failed" and x["error"] == "no checksum from the hub" for x in lines), f"T2c: a checkpoint without an LFS checksum from the hub is failed ({lines[0]})")
        api = FakeApi(lfs_sha="0" * 64)
        lines, _ = upload_lines(api, files)
        check(all(x["error"] == "checksum differs" for x in lines), "T2c: a checksum that differs is failed")
        cfg = [{"local": files[0]["local"], "path": "z-image-turbo/job/config.json", "sha256": files[0]["sha256"], "size": files[0]["size"]}]
        api = FakeApi(no_lfs=True)
        lines, _ = upload_lines(api, cfg)
        check(lines[0]["state"] == "done", "T2c: a small file the hub keeps outside LFS is checked by size")


def test_t5_token_kind():
    with tempfile.TemporaryDirectory() as tmp:
        files = make_files(tmp, 2)
        api = FakeApi(role="fineGrained")
        lines, _ = upload_lines(api, files)
        check([x["state"] for x in lines] == ["done", "done"], "T2d: whoami answers a fine-grained token -> uploaded")
        for role in ("write", "admin", "read", None):
            api = FakeApi(role=role)
            lines, _ = upload_lines(api, files)
            check(all(x["state"] == "failed" and x["error"].startswith("refused: only a fine-grained token") for x in lines) and not any(c[0] == "create_commit" for c in api.calls),
                  f"T2d: whoami answers a classic token (role {role}) -> refused before anything is sent")
        api = FakeApi(role="write")
        buf = io.StringIO()
        H.run_upload({"token": HF_TOKEN, "repo": REPO, "files": files, "message": "m", "allow_classic": True}, api, H.Out(buf, [HF_TOKEN]), sleep=lambda s: None)
        check(buf.getvalue().count('"done"') == 2, "T2d: the explicit setting lets a classic token through (for a test on the owner's own PC)")
        api = FakeApi(role="write", private=False)
        lines, _ = upload_lines(api, files)
        check(lines[0]["error"].startswith("refused: only a fine-grained"), "T2d: the token kind is checked first")


def test_token_scope():
    with tempfile.TemporaryDirectory() as tmp:
        files = make_files(tmp, 1)
        bad = {
            "another repository": good_fine(REPO, name="test/other-repo"),
            "a global permission": dict(good_fine(REPO), **{"global": ["discussion.write"]}),
            "two scopes": dict(good_fine(REPO), scoped=good_fine(REPO)["scoped"] + good_fine(REPO, name="test/other")["scoped"]),
            "an organization scope": {"global": [], "scoped": [{"entity": {"type": "org", "name": "test"}, "permissions": ["repo.write"]}]},
            "read only": good_fine(REPO, perms=("repo.access.read", "repo.content.read")),
            "more than the three permissions": good_fine(REPO, perms=("repo.access.read", "repo.content.read", "repo.write", "repo.config")),
            "no scope": {"global": [], "scoped": []},
            "no fine-grained block": None,
        }
        for label, fine in bad.items():
            api = FakeApi(fine=fine)
            lines, _ = upload_lines(api, files)
            check(lines[0]["state"] == "failed" and lines[0]["error"] == "refused: the token is not limited to this one repository" and not any(c[0] == "create_commit" for c in api.calls),
                  f"token scope: a fine-grained token with {label} -> refused before anything is sent ({lines[0].get('error')})")
        api = FakeApi()
        lines, _ = upload_lines(api, files)
        check(lines[0]["state"] == "done", "token scope: fine-grained, no global permission, one scope = the configured model, with write -> accepted")
        api = FakeApi(role="write", fine=None)
        lines, _ = upload_lines(api, files)
        check(lines[0]["error"].startswith("refused: only a fine-grained"), "token scope: a classic token is still refused")
        api = FakeApi(fine=good_fine(REPO, perms=("repo.write",)))
        lines, _ = upload_lines(api, files)
        check(lines[0]["state"] == "done", "token scope: write alone on the repository is enough")
        api = FakeApi(role="write", fine=None)
        buf = io.StringIO()
        H.run_upload({"token": HF_TOKEN, "repo": REPO, "files": files, "message": "m", "allow_classic": True}, api, H.Out(buf, [HF_TOKEN]), sleep=lambda s: None)
        check(buf.getvalue().count('"done"') == 1, "token scope: the explicit PC-only setting still lets a classic token through")


def test_nice_to_have():
    saved = {k: getattr(runner, k) for k in SAVED}
    box = UBox()
    try:
        up, job, out = pass_box(box, "n3")
        (out / "n3_000000001.safetensors").write_bytes(sft("a"))
        os.environ["FAKE_UP_MODE"] = "refuse"
        run = up.runs["n3"]
        run.refused = None
        up.run_pass(run, final=False)
        up.run_pass(run, final=False)
        (out / "n3_000000001.safetensors").unlink()
        up.run_pass(run, final=False)
        rec = {x["file"]: x for x in up.status("n3")["files"]}["n3_000000001.safetensors"]
        check(rec["state"] == "gone" and rec["error"] == "gone after a failed upload", f"N3: a checkpoint whose upload failed and that was then removed keeps its error ({rec['state']}, {rec['error']})")
    finally:
        restore(saved, box)


# ================================================================== files of the image, the README
def test_files_and_readme():
    docker = (Path(ROOT) / "trainer" / "cloud" / "Dockerfile").read_text(encoding="utf-8")
    check(re.search(r"^COPY .*trainer/cloud/hf_upload\.py .*/runner/", docker, re.M) is not None, "image: hf_upload.py is copied to /runner/")
    check("RUNNER_UPLOAD_GRACE_S=300" in docker and "HF_WRITE_TOKEN" not in docker and "HF_REPO" not in docker, "image: ENV RUNNER_UPLOAD_GRACE_S=300 and no token or repo in the Dockerfile")
    check("\r" not in docker, "image: the Dockerfile keeps LF line endings")
    text = (Path(ROOT) / "trainer" / "README.md").read_text(encoding="utf-8")
    check("Checkpoints to Hugging Face" in text and "huggingface_hub" in text and "fine-grained" in text.lower() and "classic" in text.lower(),
          "readme: the section 'Checkpoints to Hugging Face' names the library version, the classic token risk and the fine-grained advice")
    check("/proc/1/environ" in text and "RunPod keeps" in text, "readme: says the training child could read /proc/1/environ and that RunPod keeps the pod environment")
    check("HF_TOKEN_LORAS" in text and "never goes on a rented machine" in text, "readme: names HF_TOKEN_LORAS as the PC variable for a rented machine, and that the classic HF_TOKEN never goes on one")
    check("RUNNER_UPLOAD_ALLOW_CLASSIC" in text and "RUNNER_UPLOAD_ALLOW_PUBLIC" in text and "must not" in text.lower() and "whoami" in text.lower(),
          "readme: says how to override the token-kind and the public-repository checks, and that a rented machine must not")
    src = (Path(ROOT) / "trainer" / "runner.py").read_text(encoding="utf-8")
    check('"python -I -u /runner/hf_upload.py upload"' in src, "N7: the upload child runs with -I (no PYTHONPATH or user site while it holds the token)")
    check(not re.search(r"^\s*(import|from)\s+(huggingface_hub|requests|yaml)", src, re.M), "runner.py stays standard library only")


def run(only=None):
    del RESULTS[:]
    tests = [test_hf_upload_in_process, test_selftest_in_process, test_runner_end_to_end, test_runner_restart_keeps_status,
             test_runner_off_and_bad_variables, test_caps_and_safety, test_refused_and_leaky_uploader, test_limit_mid_upload,
             test_idle_with_upload_in_flight, test_m1_final_pass_after_a_pass_in_flight, test_m2_done_needs_the_final_checkpoint,
             test_m3_incomplete_checkpoint, test_m4_trainer_that_cannot_start, test_m5_caps_on_every_pass, test_t2_private_repository,
             test_t2_runner_stops_on_a_refusal_and_passes_the_settings, test_t3_endpoint_is_fixed, test_t4_checksum_required,
             test_t5_token_kind, test_token_scope, test_nice_to_have, test_files_and_readme]
    for fn in tests:
        if not only or any(o in fn.__name__ for o in only):
            fn()
    print("\n%d/%d passed" % (sum(RESULTS), len(RESULTS)))
    return all(RESULTS)


def test_trainer_upload():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run(sys.argv[1:]) else 1)
