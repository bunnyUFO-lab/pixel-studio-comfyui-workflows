"""Tests for the Modal job gateway's bookkeeping (cloud/modal_comfy.py) without Modal's servers or fastapi.

Needs the `modal` client importable (it is only imported, nothing is deployed); without it the test prints SKIP and
passes. On this PC: C:/Users/danie/.venvs/modal/Scripts/python.exe tests/test_modal_gateway.py
FastAPI, the Dict and the function calls are replaced by small fakes.
"""
import asyncio
import os
import sys
import types
import warnings

warnings.simplefilter("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "cloud"))
try:
    import modal
except ImportError:
    print("SKIP: the modal client is not installed (use the Modal venv to run this test)")
    sys.exit(0)

RESULTS = []


def check(cond, msg):
    RESULTS.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + msg)


# ----------------------------------------------------------------------------- fakes
class Resp:
    def __init__(self, body, status_code=200):
        self.body, self.status_code = body, status_code


class FakeApi:
    def __init__(self, **kw):
        self.routes = {}

    def _route(self, method):
        def deco(path):
            def reg(fn):
                self.routes[(method, path)] = fn
                return fn
            return reg
        return deco

    def __getattr__(self, name):
        if name in ("get", "post", "delete"):
            return self._route(name)
        raise AttributeError(name)


stub = types.ModuleType("fastapi")
stub.FastAPI, stub.Request = FakeApi, object
responses = types.ModuleType("fastapi.responses")
responses.JSONResponse = lambda body, status_code=200: Resp(body, status_code)
sys.modules["fastapi"], sys.modules["fastapi.responses"] = stub, responses

import modal_comfy as mc  # noqa: E402


def aio(fn):
    return types.SimpleNamespace(aio=fn)


class FakeDict:
    def __init__(self):
        self.data = {}

        async def get(k):
            return self.data.get(k)

        async def put(k, v):
            self.data[k] = v

        async def pop(k):
            return self.data.pop(k)
        self.get, self.put, self.pop = aio(get), aio(put), aio(pop)

    def sync_pop(self, k):
        return self.data.pop(k)

    def put_sync(self, k, v):
        self.data[k] = v


class FakeCall:
    """outcome: a value (done), an exception instance (raised by get), or None (still running)."""
    calls = {}

    def __init__(self, job_id, outcome, cancel_error=None):
        self.object_id, self.outcome, self.cancel_error, self.cancelled = job_id, outcome, cancel_error, False

        async def get(timeout=None):
            if self.outcome is None:
                raise modal.exception.TimeoutError("not yet")
            if isinstance(self.outcome, BaseException):
                raise self.outcome
            return self.outcome

        async def cancel(terminate_containers=False):
            if self.cancel_error:
                raise self.cancel_error
            self.cancelled = terminate_containers
        self.get, self.cancel = aio(get), aio(cancel)


async def from_id(job_id):
    if job_id not in FakeCall.calls:
        raise modal.exception.NotFoundError("no such call")
    return FakeCall.calls[job_id]


modal.FunctionCall = types.SimpleNamespace(from_id=aio(from_id))
mc.state = FakeDict()
api = mc.jobs._raw_f_()
status, cancel = api.routes[("get", "/v1/jobs/{job_id}")], api.routes[("delete", "/v1/jobs/{job_id}")]


def call(job_id, outcome, cancel_error=None):
    FakeCall.calls[job_id] = FakeCall(job_id, outcome, cancel_error)
    return FakeCall.calls[job_id]


def run(coro):
    return asyncio.run(coro)


# ----------------------------------------------------------------------------- A6: bookkeeping
st = mc.state
st.data.clear()
call("fc-failed", RuntimeError("ComfyUI: boom"))
st.data.update({"job:fc-failed": {"node": "1"}, "cancelled:fc-failed": True})
r = run(status("fc-failed"))
check(r["status"] == "cancelled" and "job:fc-failed" not in st.data and "cancelled:fc-failed" not in st.data,
      "A6: failed (flag set) -> cancelled, job and cancel records deleted")
call("fc-failed2", RuntimeError("ComfyUI: boom"))
st.data["job:fc-failed2"] = {"node": "1"}
r = run(status("fc-failed2"))
check(r["status"] == "failed" and "boom" in r["error"] and st.data == {}, "A6: failed -> job record deleted")
call("fc-done", {"status": "success"})
st.data["job:fc-done"] = {"node": "1"}
r = run(status("fc-done"))
check(r["status"] == "done" and st.data == {}, "A6: done -> job record deleted")
call("fc-cx", modal.exception.InputCancellation("cancelled"))
st.data.update({"cancelled:fc-cx": True, "job:fc-cx": {}})
r = run(status("fc-cx"))
check(r["status"] == "cancelled" and st.data == {}, "A6: Modal reports the cancel -> flag and job records deleted")
call("fc-run", None)
st.data["job:fc-run"] = {"node": "3"}
check(run(status("fc-run"))["status"] == "running" and "job:fc-run" in st.data, "status: running keeps its progress record")

st.data.clear()
c = call("fc-live", None)
st.data["job:fc-live"] = {"node": "3"}
r = run(cancel("fc-live"))
check(r == {"id": "fc-live", "status": "cancelled"} and c.cancelled is True and st.data == {"cancelled:fc-live": True},
      "A6: cancel -> terminate_containers, flag set, progress record deleted")
check(run(status("fc-live"))["status"] == "cancelled" and "cancelled:fc-live" in st.data,
      "A6: until Modal reports it, status keeps answering cancelled from the flag")
st.data.clear()
call("fc-bad", None, cancel_error=RuntimeError("modal said no"))
try:
    run(cancel("fc-bad"))
    check(False, "A6: cancel that raises")
except RuntimeError:
    check(st.data == {}, "A6: cancel raised -> no cancel flag was written (status stays running)")
r = run(cancel("fc-made-up"))
check(r.status_code == 404 and st.data == {}, "A6: DELETE on a made-up id -> 404, nothing written")
call("fc-finished", {"status": "success"})
r = run(cancel("fc-finished"))
check(r.status_code == 409 and st.data == {}, "A6: DELETE on a finished job -> 409, no flag")

# ----------------------------------------------------------------------------- A2 / A4 wiring in ComfyJob.run
stopped = []
modal.experimental.stop_fetching_inputs = lambda: stopped.append(1)
mc.recycle("test")
check(stopped == [1], "A2: recycle() calls modal.experimental.stop_fetching_inputs")

seen = {}
import job_runner  # noqa: E402


def fake_guarded(doc, proc, recycle, **kw):
    seen.update(kw, recycle=recycle)
    return {"status": "success", "files": []}


class SyncState:
    """run() uses the sync put/pop of the Dict."""
    def put(self, k, v):
        st.data[k] = v

    def pop(self, k):
        return st.data.pop(k)


mc.state = SyncState()
Job = mc.ComfyJob._get_user_cls()
job = Job.__new__(Job)
job.proc, job.boot, job.boot_pending = object(), 12.0, True
modal.current_function_call_id = lambda: "fc-1"
real = job_runner.run_job_guarded
job_runner.run_job_guarded = fake_guarded
st.data.clear()
try:
    Job.run._get_raw_f()(job, {"workflow": {"1": {}}})
finally:
    job_runner.run_job_guarded = real
check(seen["max_timeout"] == mc.CFG["job_timeout"] and seen["recycle"] is mc.recycle and seen["boot"] == 12.0,
      "A4/A2: run() passes MODAL_JOB_TIMEOUT as max_timeout, recycle as the callback, boot to the first job")
check("job:fc-1" not in st.data, "run(): the progress record is deleted when the job ends")

print(f"\n{sum(RESULTS)}/{len(RESULTS)} passed")
sys.exit(0 if all(RESULTS) else 1)
