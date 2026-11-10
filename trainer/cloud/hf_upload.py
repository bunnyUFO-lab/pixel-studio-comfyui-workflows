#!/usr/bin/env python3
"""The upload child of the trainer runner: puts finished files into a private Hugging Face model repository.

    python -u hf_upload.py upload      # default: reads ONE JSON line on stdin, uploads, prints one result line per file
    python -u hf_upload.py selftest    # a one-file round trip on a throwaway branch (see run_selftest)

It runs with the image's python and its `huggingface_hub` (ai-toolkit needs it; the runner itself is standard library only).
The write token arrives on stdin only, in the JSON line {"token", "repo", "files": [{"local", "path", "sha256", "size"}],
"message"}: never in an argument, an environment variable, a file or a URL. Nothing it prints holds the token: it prints
result lines only, `{"path", "state", "commit"?, "error"?}`, where `error` is the exception CLASS NAME plus the HTTP status
(`HfHubHTTPError 401`), never the message text, a URL or headers; and every line passes through a redaction of the token.

It only uses these huggingface_hub calls, which exist in 0.34 and in 1.x: HfApi(token=, endpoint=), create_commit,
CommitOperationAdd, get_paths_info(expand=True), create_branch, delete_branch, list_repo_refs, model_info,
hf_hub_download, whoami.
"""
import json
import os
import re
import secrets
import sys
import tempfile
import time

ENDPOINT = "https://huggingface.co"   # fixed in code: the write token only ever goes to this host
TEST_ENDPOINT = None                  # tests set this in process; no environment variable, no stdin field can change the host
RETRY_WAITS = (10, 30, 90)            # seconds before the 2nd, 3rd and 4th try (a module constant so tests shorten it)
NO_RETRY_STATUS = (401, 403, 404)     # the token or the repository is wrong: trying again cannot help
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
STDIN_MAX = 4 << 20                   # one JSON line


# ----------------------------------------------------------------- output that never holds a secret
class Out:
    """Writes JSON lines to a stream, with every secret replaced by *** first (the last guard)."""

    def __init__(self, stream, secrets_=()):
        self.stream = stream
        self.secrets = [s for s in secrets_ if s]

    def redact(self, text):
        for s in self.secrets:
            text = text.replace(s, "***")
        return text

    def line(self, obj):
        self.stream.write(self.redact(json.dumps(obj)) + "\n")
        try:
            self.stream.flush()
        except (OSError, ValueError):
            pass


def status_of(exc):
    resp = getattr(exc, "response", None)
    code = getattr(resp, "status_code", None)
    return code if isinstance(code, int) and not isinstance(code, bool) else None


def describe(exc):
    """'HfHubHTTPError 401': the class name and the HTTP status, never the message (it can hold a URL or a header)."""
    code = status_of(exc)
    name = re.sub(r"[^A-Za-z0-9_]", "", type(exc).__name__) or "Error"
    return f"{name} {code}" if code else name


def field(obj, key):
    """obj[key] for a dict, obj.key for an object (huggingface_hub 0.34 and 1.x differ here), else None."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


# ----------------------------------------------------------------- upload
def retried(call, sleep):
    """-> (value, None) or (None, error text): the call retried like the commit, no retry on 401, 403 or 404."""
    tries = len(RETRY_WAITS) + 1
    for attempt in range(tries):
        try:
            return call(), None
        except Exception as e:                                          # noqa: BLE001
            if status_of(e) in NO_RETRY_STATUS or attempt == tries - 1:
                return None, describe(e)
            sleep(RETRY_WAITS[attempt])


def preflight(spec, api, sleep):
    """Before anything is sent: the token must be a fine-grained one and the repository private, unless the machine was
    created with the explicit setting that allows it (spec allow_classic / allow_public, which the runner reads from
    its environment at start). -> None when the target may receive files, else the error text for every file."""
    repo = spec["repo"]
    who, err = retried(api.whoami, sleep)
    if err:
        return err if status_of_text(err) else "token check failed: " + err
    token = field(field(who, "auth"), "accessToken")
    if not spec.get("allow_classic"):
        if field(token, "role") != "fineGrained":
            return "refused: only a fine-grained token is accepted on a rented machine"
        if not scoped_to_repo(field(token, "fineGrained"), repo):
            return "refused: the token is not limited to this one repository"
    info, err = retried(lambda: api.model_info(repo_id=repo), sleep)
    if err:
        return err if status_of_text(err) else "repository check failed: " + err
    if field(info, "private") is not True and not spec.get("allow_public"):
        return "refused: the repository is not private"
    return None


TOKEN_PERMISSIONS = {"repo.access.read", "repo.content.read", "repo.write"}


def scoped_to_repo(fine, repo):
    """True for exactly: no global permissions, one scope, the model `repo`, with repo.write and nothing beyond
    repo.access.read, repo.content.read and repo.write. A broader token, or one for another repository, is not."""
    scoped = field(fine, "scoped")
    if field(fine, "global") or not isinstance(scoped, list) or len(scoped) != 1:
        return False
    entity, perms = field(scoped[0], "entity"), field(scoped[0], "permissions")
    if field(entity, "type") != "model" or field(entity, "name") != repo:
        return False
    return isinstance(perms, list) and "repo.write" in perms and set(perms) <= TOKEN_PERMISSIONS


def status_of_text(text):
    """True when describe()'s text ends in an HTTP status (the runner reads 401, 403 and 404 as a refusal)."""
    return bool(re.search(r" [0-9]{3}\Z", text))


def run_upload(spec, api, out, sleep=time.sleep):
    """One commit with every file of the spec, retried, then checked against the sha256 and size we were given.
    Prints one line per file."""
    repo, files, message = spec["repo"], spec["files"], spec.get("message") or "upload"
    revision = "main"
    tries = len(RETRY_WAITS) + 1

    def fail_all(error):
        for f in files:
            out.line({"path": f["path"], "state": "failed", "error": error})

    refusal = preflight(spec, api, sleep)
    if refusal:
        fail_all(refusal)
        return
    from huggingface_hub import CommitOperationAdd
    info = None
    for attempt in range(tries):
        try:
            ops = [CommitOperationAdd(path_in_repo=f["path"], path_or_fileobj=f["local"]) for f in files]
            info = api.create_commit(repo_id=repo, repo_type="model", revision=revision, operations=ops,
                                     commit_message=message)
            break
        except Exception as e:                                          # noqa: BLE001
            if status_of(e) in NO_RETRY_STATUS or attempt == tries - 1:
                fail_all(describe(e))
                return
            sleep(RETRY_WAITS[attempt])
    oid = field(info, "oid")
    if not isinstance(oid, str) or not HEX40.match(oid):
        fail_all("no commit id")
        return
    found, error = None, None
    for attempt in range(tries):
        try:
            found = api.get_paths_info(repo_id=repo, paths=[f["path"] for f in files], repo_type="model", revision=oid,
                                       expand=True)
            break
        except Exception as e:                                          # noqa: BLE001
            error = "checks failed: " + describe(e)
            if status_of(e) in NO_RETRY_STATUS or attempt == tries - 1:
                break
            sleep(RETRY_WAITS[attempt])
    if found is None:
        fail_all(error or "checks failed")
        return
    by_path = {field(p, "path"): p for p in found}
    for f in files:
        p = by_path.get(f["path"])
        if p is None:
            out.line({"path": f["path"], "state": "failed", "error": "missing after commit"})
            continue
        lfs = field(p, "lfs")
        lfs_sha = field(lfs, "sha256") if lfs else None
        if f["path"].lower().endswith(".safetensors") and not lfs_sha:       # a library change must not weaken the check
            out.line({"path": f["path"], "state": "failed", "error": "no checksum from the hub"})
        elif field(p, "size") != f["size"] or (lfs_sha is not None and lfs_sha != f["sha256"]):
            out.line({"path": f["path"], "state": "failed", "error": "checksum differs"})
        else:
            out.line({"path": f["path"], "state": "done", "commit": oid})


# ----------------------------------------------------------------- selftest
def branch_names(api, repo):
    refs = api.list_repo_refs(repo_id=repo, repo_type="model")
    return {field(b, "name"): field(b, "target_commit") for b in (field(refs, "branches") or [])}


def run_selftest(spec, api, download, out, read_api=None):
    """A one-file round trip that never writes to `main`: whoami, privacy check, a throwaway branch, one small file on it,
    read back with the read token (and refused without a token), branch deleted, refs checked. True when all of it held.
    `api` holds the write token, `read_api` the read token (the same object in tests), `download(repo, filename,
    revision, token, local_dir)` returns the downloaded path (token False = no token at all)."""
    read_api = read_api or api
    ok = True

    def step(name, good, **info):
        nonlocal ok
        ok = ok and bool(good)
        out.line({"step": name, "ok": bool(good), **info})
        return bool(good)

    try:
        who = api.whoami()
        owner = field(who, "name")
        role = field(field(field(who, "auth"), "accessToken"), "role")
    except Exception as e:                                               # noqa: BLE001
        step("whoami", False, error=describe(e))
        return False
    repo = f"{owner}/{spec['repo_name']}"
    step("whoami", bool(owner), owner=owner, role=role, repo=repo)
    try:
        private = field(read_api.model_info(repo_id=repo), "private")
    except Exception as e:                                               # noqa: BLE001
        step("private", False, error=describe(e), note="repository missing or unreadable: nothing written")
        return False
    if not step("private", private is True, private=private, note=None if private is True else "not private: nothing written"):
        return False
    try:
        main_before = branch_names(api, repo).get("main")
    except Exception as e:                                               # noqa: BLE001
        step("refs", False, error=describe(e))
        return False
    branch, name = "selftest-" + secrets.token_hex(6), None
    name = f"_selftest/{secrets.token_hex(6)}.txt"
    content = secrets.token_hex(32).encode("ascii")                      # 64 hex characters, in memory only
    created = False
    try:
        api.create_branch(repo_id=repo, branch=branch, revision=main_before, repo_type="model")
        created = True
        step("branch", True, branch=branch)
        from huggingface_hub import CommitOperationAdd
        api.create_commit(repo_id=repo, repo_type="model", revision=branch, commit_message="selftest",
                          operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=content)])
        step("upload", True, path=name)
        with tempfile.TemporaryDirectory(prefix="hf-selftest-") as tmp:
            got = download(repo, name, branch, spec["read_token"], tmp)
            with open(got, "rb") as fh:
                same = fh.read() == content
        step("read back", same)
        try:
            with tempfile.TemporaryDirectory(prefix="hf-selftest-") as tmp:
                download(repo, name, branch, False, tmp)
            step("anonymous download refused", False, note="the repository answered without a token: it is not private")
        except Exception as e:                                           # noqa: BLE001
            step("anonymous download refused", status_of(e) in NO_RETRY_STATUS, error=describe(e))
    except Exception as e:                                               # noqa: BLE001
        step("round trip", False, error=describe(e))
    finally:
        if created:
            try:
                api.delete_branch(repo_id=repo, branch=branch, repo_type="model")
            except Exception as e:                                       # noqa: BLE001
                step("delete branch", False, error=describe(e), branch=branch, note="BRANCH LEFT: delete it by hand")
        try:
            names = branch_names(api, repo)
            gone = branch not in names
            step("branch deleted", gone or not created, branch=branch if not gone else None)
            step("main unchanged", names.get("main") == main_before and main_before is not None,
                 main=(main_before or "")[:7])
        except Exception as e:                                           # noqa: BLE001
            step("refs after", False, error=describe(e))
    out.line({"step": "result", "ok": ok})
    return ok


# ----------------------------------------------------------------- real library
def real_api(token, endpoint=None):
    from huggingface_hub import HfApi
    return HfApi(token=token, endpoint=endpoint or TEST_ENDPOINT or ENDPOINT)


def real_download(repo, filename, revision, token, local_dir):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo, filename=filename, revision=revision, token=token, local_dir=local_dir,
                           endpoint=TEST_ENDPOINT or ENDPOINT)


def harden_env():
    """The library must use only the token it is given."""
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    for key in ("HF_TOKEN", "HF_ENDPOINT", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE"):
        os.environ.pop(key, None)


def valid_upload_spec(spec):
    if not isinstance(spec, dict) or not isinstance(spec.get("token"), str) or not spec["token"]:
        return False
    if not isinstance(spec.get("repo"), str) or not isinstance(spec.get("files"), list) or not spec["files"]:
        return False
    for f in spec["files"]:
        if not (isinstance(f, dict) and isinstance(f.get("local"), str) and isinstance(f.get("path"), str)
                and isinstance(f.get("sha256"), str) and isinstance(f.get("size"), int)):
            return False
    return True


def main(stdin=None, stdout=None, api_factory=None, argv=None, stderr=None, download=None):
    """-> exit code: 0 done, 1 a stdin or argument error (or a failed selftest)."""
    stdin, stdout, stderr = stdin or sys.stdin, stdout or sys.stdout, stderr or sys.stderr
    argv = sys.argv[1:] if argv is None else list(argv)
    cmd = argv[0] if argv else "upload"
    if cmd not in ("upload", "selftest") or len(argv) > 1:
        stderr.write("usage: hf_upload.py [upload|selftest]   (one JSON line on stdin)\n")
        return 1
    try:
        spec = json.loads(stdin.readline(STDIN_MAX) or "null")
    except ValueError:
        spec = None
    if not isinstance(spec, dict):
        stderr.write("hf_upload: stdin must hold one JSON object\n")
        return 1
    out = Out(stdout, [spec.get("token"), spec.get("read_token")] if isinstance(spec.get("token"), str) else [])
    factory = api_factory or real_api
    try:
        if cmd == "upload":
            if not valid_upload_spec(spec):
                stderr.write("hf_upload: the JSON line needs token, repo and files [{local, path, sha256, size}]\n")
                return 1
            run_upload(spec, factory(spec["token"]), out)
            return 0
        if not (isinstance(spec.get("token"), str) and isinstance(spec.get("read_token"), str)
                and isinstance(spec.get("repo_name"), str)):
            stderr.write("hf_upload: selftest needs token, read_token and repo_name\n")
            return 1
        return 0 if run_selftest(spec, factory(spec["token"]), download or real_download, out,
                                 read_api=factory(spec["read_token"])) else 1
    except Exception as e:                                               # noqa: BLE001
        out.line({"error": describe(e)})
        return 1


if __name__ == "__main__":
    harden_env()
    sys.exit(main())
