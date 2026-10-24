"""ComfyUI on a Modal cloud GPU, for Sprite Studio to use when the PC is busy or off.

The same ComfyUI as docker/Dockerfile (tag, custom nodes, the repo's own nodes and workflows); models live on a
Modal Volume, downloaded inside Modal by docker/models.py from the URLs in the workflows. Everything is locked with
Modal proxy auth (ComfyUI has no login): every request needs the Modal-Key / Modal-Secret headers of a proxy auth
token, which Sprite Studio sends (COMFY_SOURCES headers). It scales to zero when idle.

Two ways in:
  comfyui  the old single-container ComfyUI web endpoint (one GPU, all requests share it).
  jobs     the job API, protocol psc-job/v1 (docs/job-api.md): POST a job, poll it, get every file back. Each job runs
           alone in a ComfyUI container, up to MODAL_MAX_CONTAINERS at once, so jobs run in parallel.

Settings come from cloud/modal.env (see modal.env.example) and the process environment, resolved once at deploy time.

From the repo root, with the Modal client (see cloud/README.md):
    modal run cloud/modal_comfy.py::download                 # fetch every Sprite Studio model into the Volume
    modal run cloud/modal_comfy.py::download --workflows starter   # or "all", or a comma-separated list
    modal run cloud/modal_comfy.py::refresh_info             # cache the node and model lists (/v1/info)
    modal run cloud/modal_comfy.py::smoke                    # one image-to-pixel job on a job container
    modal deploy cloud/modal_comfy.py                        # publish both endpoints (prints their URLs)
    modal app stop sprite-studio-comfy                       # take it down (use your MODAL_APP name)
"""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# ----------------------------------------------------------------------------- config
DEFAULTS = {"app": "sprite-studio-comfy", "gpu": "RTX-PRO-6000,L40S", "max_containers": 3, "idle": 300,
            "preset": "pixel", "job_timeout": 3600}
ENV_KEYS = {"app": "MODAL_APP", "gpu": "MODAL_GPU", "max_containers": "MODAL_MAX_CONTAINERS",
            "idle": "MODAL_IDLE_SECONDS", "preset": "MODAL_PRESET", "job_timeout": "MODAL_JOB_TIMEOUT"}


def _env_file(path):
    values = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip("\"'")
    return values


def resolve_config():
    """process env > cloud/modal.env > defaults."""
    file_env = _env_file(HERE / "modal.env")
    cfg = {}
    for key, default in DEFAULTS.items():
        raw = os.environ.get(ENV_KEYS[key]) or file_env.get(ENV_KEYS[key])
        cfg[key] = default if raw in (None, "") else (int(raw) if isinstance(default, int) else raw)
    return cfg


# Resolved here when deploying; baked into the images as PSC_MODAL_CONFIG so the containers see the same values.
CFG = resolve_config() if modal.is_local() else json.loads(os.environ.get("PSC_MODAL_CONFIG") or json.dumps(DEFAULTS))
GPUS = [g.strip() for g in CFG["gpu"].split(",") if g.strip()]
GPU = GPUS if len(GPUS) > 1 else GPUS[0]       # a list = Modal's ordered fallback
IDLE = CFG["idle"]
COMFYUI_REF = os.environ.get("COMFYUI_REF", "v0.38.2")    # keep in step with docker-compose.yml
COMFY = "/opt/ComfyUI"
MODELS = "/models"
WORKFLOWS = f"{COMFY}/user/default/workflows"

# The fast job types (drafts, prompt to sprite, pixelize, Qwen edits): about 30 GB. The default for download is
# "pixel" (every workflow Sprite Studio uses, from env/pixel.env); "all" = every workflow in the repo.
STARTER = ",".join([
    "pixel-art/image-to-pixel", "pixel-art/prompt-to-pixel", "pixel-art/sprite-edit", "pixel-art/sprite-edit-ref",
    "pixel-art/sprite-edit-refs", "image-edit/qwen-edit",
])

# One line: each run_commands entry becomes a single Dockerfile RUN.
CLONE_NODES = ("set -eu; cd /opt/ComfyUI/custom_nodes; "
               r"tr -d '\r' < /tmp/custom_nodes.txt | grep -vE '^\s*(#|$)' | while read -r url ref; do "
               "dir=$(basename \"$url\" .git); git clone -q \"$url\" \"$dir\"; git -C \"$dir\" checkout -q \"$ref\"; "
               "if [ -f \"$dir/requirements.txt\" ]; then pip install -q -r \"$dir/requirements.txt\"; fi; done")

CONFIG_ENV = {"PSC_MODAL_CONFIG": json.dumps(CFG)}      # placed just before the add_local_* steps: no torch rebuild

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "aria2", "curl", "ca-certificates", "libgl1", "libglib2.0-0")
    .pip_install("torch", "torchvision", "torchaudio", index_url="https://download.pytorch.org/whl/cu128")
    .run_commands(f"git clone -q --depth 1 --branch {COMFYUI_REF} https://github.com/comfyanonymous/ComfyUI {COMFY}",
                  f"pip install -q -r {COMFY}/requirements.txt")
    .add_local_file(REPO / "docker" / "custom_nodes.txt", "/tmp/custom_nodes.txt", copy=True)
    .run_commands(CLONE_NODES)
    .add_local_file(REPO / "docker" / "requirements-extra.txt", "/tmp/requirements-extra.txt", copy=True)
    # pip 25 trips over the gVisor kernel version ("4.19.0-gvisor") in environment markers; newer pip does not.
    .run_commands("pip install -q -U pip && pip install -q --no-deps -r /tmp/requirements-extra.txt")
    .pip_install("websockets==13.1")                    # job progress over ComfyUI's WebSocket (job_runner.py)
    .env({"MODELS_DIR": MODELS, "WORKFLOWS_DIR": WORKFLOWS, **CONFIG_ENV})
    # Added at container start (not baked), so a workflow or node change needs no image rebuild: just deploy again.
    .add_local_file(REPO / "docker" / "models.py", "/usr/local/bin/models.py")
    .add_local_file(REPO / "env" / "pixel.env", "/opt/env/pixel.env")
    .add_local_file(HERE / "job_runner.py", "/root/job_runner.py")
    .add_local_file(HERE / "gpu_rates.json", "/root/gpu_rates.json")
    .add_local_dir(REPO / "workflows", WORKFLOWS)
    .add_local_dir(REPO / "custom_nodes" / "comfyui_workflows_utils", f"{COMFY}/custom_nodes/comfyui_workflows_utils",
                   ignore=["**/__pycache__/**"])
)

# The job gateway never touches a GPU or the ComfyUI image: a status poll must not wake anything expensive.
gateway_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]==0.115.6")
    .env(CONFIG_ENV)
    .add_local_file(HERE / "job_runner.py", "/root/job_runner.py")
)

app = modal.App(CFG["app"], image=image)
models = modal.Volume.from_name("sprite-studio-models", create_if_missing=True)
state = modal.Dict.from_name(f"{CFG['app']}-state", create_if_missing=True)    # job progress and the /v1/info cache


@app.function(volumes={MODELS: models}, timeout=6 * 3600)
def download(workflows: str = CFG["preset"]):
    """Download the models the given workflows need into the Volume (skips what is there, resumes partial files).
    workflows: "pixel" (every workflow Sprite Studio uses, as in env/pixel.env), "starter", "all", or a comma-separated
    list. Runs on a CPU container; HF_TOKEN is not needed for the public models the workflows list."""
    names = _workflows(workflows)
    rc = subprocess.call(["python", "/usr/local/bin/models.py", "download", *names])
    models.commit()
    subprocess.call(["du", "-sh", MODELS])
    try:
        state.pop("info")           # the model lists changed: /v1/info refreshes on its next call
    except KeyError:
        pass
    if rc:
        raise SystemExit(f"models.py exited with {rc} (see above); what was downloaded is kept")


@app.function(volumes={MODELS: models}, timeout=600)
def list_models(workflows: str = CFG["preset"]):
    """Which models the workflows need and whether the Volume has them."""
    subprocess.call(["python", "/usr/local/bin/models.py", "list", *_workflows(workflows)])


def _workflows(value):
    if value == "pixel":
        line = next(l for l in Path("/opt/env/pixel.env").read_text().splitlines() if l.startswith("MODELS="))
        value = line.split("=", 1)[1]
    elif value == "starter":
        value = STARTER
    return [w.strip() for w in value.split(",") if w.strip()]


def _link_models():
    # ComfyUI looks for models in its own models folder: point it at the Volume.
    subprocess.run(f"rm -rf {COMFY}/models && ln -s {MODELS} {COMFY}/models", shell=True, check=True)


@app.function(gpu=GPU, volumes={MODELS: models}, max_containers=1, scaledown_window=IDLE, timeout=24 * 3600)
@modal.concurrent(max_inputs=100)       # one GPU container answers all of Sprite Studio's requests and its WebSocket
@modal.web_server(8188, startup_timeout=600, requires_proxy_auth=True)
def comfyui():
    _link_models()
    subprocess.Popen(["python", f"{COMFY}/main.py", "--listen", "0.0.0.0", "--port", "8188"], cwd=COMFY)


# ----------------------------------------------------------------------------- job API
@app.cls(gpu=GPU, volumes={MODELS: models}, max_containers=CFG["max_containers"], scaledown_window=IDLE,
         timeout=CFG["job_timeout"] + 600)
class ComfyJob:
    """One ComfyUI per container, one job at a time: no @modal.concurrent on purpose. job_runner finds a job's files
    by comparing output/ and temp/ before and after, which is only right when nothing else runs on the same ComfyUI.
    A warm container takes the next job (models stay loaded); a job's inputs and outputs are deleted after it."""

    @modal.enter()
    def start(self):
        import job_runner
        t0 = time.time()
        _link_models()
        self.proc = job_runner.ComfyProcess(COMFY, 8188).start()
        self.proc.wait_ready(600)
        self.boot = time.time() - t0
        self.boot_pending = True            # the first job of this container pays for the start
        # The GPU container's view of nodes and models is the authoritative one: publish it (off the boot path).
        threading.Thread(target=lambda: state.put("info", self.proc.info()), daemon=True).start()

    @modal.exit()
    def stop(self):
        self.proc.stop()

    @modal.method()
    def run(self, doc: dict) -> dict:
        import job_runner
        call_id = modal.current_function_call_id()
        key = f"job:{call_id}"
        rates = json.loads((HERE / "gpu_rates.json").read_text())
        last = [0.0]

        def on_progress(p):
            if time.time() - last[0] >= 1:
                last[0] = time.time()
                try:
                    state.put(key, p)
                except Exception:
                    pass

        boot = self.boot if self.boot_pending else 0.0
        self.boot_pending = False
        state.put(key, {"node": None, "value": 0, "max": 0, "done": [], "total": len(doc.get("workflow") or {})})
        try:
            result = job_runner.run_job(doc, self.proc, on_progress=on_progress, boot=boot, rate=rates,
                                        worker=os.environ.get("MODAL_TASK_ID", ""))
            return job_runner.encode_result(result)
        except ValueError:
            raise
        except Exception as e:
            # Re-raised as one type: a TimeoutError would read as "still running" to the status poll.
            raise RuntimeError(f"{type(e).__name__}: {e}") from None
        finally:
            try:
                state.pop(key)
            except KeyError:
                pass


@app.function(volumes={MODELS: models}, timeout=600)
def refresh_info() -> dict:
    """Start ComfyUI on the CPU, collect /object_info and the model lists, store them for /v1/info. No GPU."""
    import job_runner
    _link_models()
    proc = job_runner.ComfyProcess(COMFY, 8188).start(["--cpu"])
    try:
        proc.wait_ready(500)
        info = proc.info()
    finally:
        proc.stop()
    state.put("info", info)
    return info


@app.function(image=gateway_image, scaledown_window=IDLE, timeout=900)
@modal.concurrent(max_inputs=100)
@modal.asgi_app(requires_proxy_auth=True)
def jobs():
    import job_runner
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    api = FastAPI(title="psc-job/v1")

    def reply(code, body):
        return JSONResponse(body, status_code=code)

    @api.get("/v1/health")
    async def health():
        return {"ok": True, "protocol": job_runner.PROTOCOL, "app": CFG["app"], "gpu": GPUS,
                "max_containers": CFG["max_containers"], "idle_seconds": IDLE}

    @api.get("/v1/info")
    async def info(refresh: int = 0):
        cached = None if refresh else await state.get.aio("info")
        if cached is None:
            cached = await refresh_info.remote.aio()
        return cached

    @api.post("/v1/jobs")
    async def submit(request: Request):
        if int(request.headers.get("content-length") or 0) > job_runner.MAX_INPUT_BYTES * 2:
            return reply(413, {"error": "request body is too large"})
        try:
            doc = json.loads(await request.body())
            job_runner.validate_job(doc)
            job_runner.decode_job(doc)
        except job_runner.InputTooLarge as e:
            return reply(413, {"error": str(e)})
        except ValueError as e:                 # includes bad JSON
            return reply(400, {"error": str(e)})
        call = await ComfyJob().run.spawn.aio(doc)
        return reply(202, {"id": call.object_id, "status": "queued"})

    async def call_of(job_id):
        try:
            return await modal.FunctionCall.from_id.aio(job_id)
        except Exception:
            return None

    @api.get("/v1/jobs/{job_id}")
    async def status(job_id: str):
        call = await call_of(job_id)
        if call is None:
            return reply(404, {"error": "unknown job id"})
        progress = await state.get.aio(f"job:{job_id}")
        cancelled = await state.get.aio(f"cancelled:{job_id}")      # set by DELETE: Modal reports the cancel late
        try:
            result = await call.get.aio(timeout=0)
        except (TimeoutError, modal.exception.TimeoutError):
            if cancelled:
                return {"id": job_id, "status": "cancelled", "progress": None}
            return {"id": job_id, "status": "running" if progress else "queued", "progress": progress}
        except modal.exception.OutputExpiredError:
            return reply(404, {"error": "result expired"})
        except modal.exception.InputCancellation:
            return {"id": job_id, "status": "cancelled", "progress": None}
        except Exception as e:                  # modal.exception.NotFoundError / InvalidError for a made-up id
            if isinstance(e, (modal.exception.NotFoundError, modal.exception.InvalidError)):
                return reply(404, {"error": "unknown job id"})
            if cancelled:
                return {"id": job_id, "status": "cancelled", "progress": None}
            return {"id": job_id, "status": "failed", "progress": None, "error": str(e)[:2000]}
        try:
            await state.pop.aio(f"job:{job_id}")
        except KeyError:
            pass
        return {"id": job_id, "status": "done", "progress": None, "result": result}

    @api.delete("/v1/jobs/{job_id}")
    async def cancel(job_id: str):
        call = await call_of(job_id)
        if call is None:
            return reply(404, {"error": "unknown job id"})
        # terminate_containers: otherwise the ComfyUI subprocess would keep running the prompt, and the next job
        # on that container would wait behind it. Costs one cold start; cancels are rare.
        await state.put.aio(f"cancelled:{job_id}", True)
        await call.cancel.aio(terminate_containers=True)
        return {"id": job_id, "status": "cancelled"}

    return api


@app.local_entrypoint()
def smoke():
    """One image-to-pixel job on a job container (no HTTP): prints GPU, timing, cost and the files."""
    sys.path.insert(0, str(HERE))
    import job_smoke
    info = refresh_info.remote()
    print(f"info: {len(info['object_info'])} node classes")
    result = ComfyJob().run.remote(job_smoke.make_job(info["object_info"]))
    print(job_smoke.summary_line("smoke", result))
    for e in result.get("errors", []):
        print("ERROR", e)
    for f in result["files"]:
        print(f"file  [{f['node'] or '-'}] {f['type']}/{f['subfolder']}{'/' if f['subfolder'] else ''}{f['filename']} {f['size']} bytes")
    for title, text in result["text"].items():
        print(f"text  [{title}] {text.splitlines()[0] if text else ''}")
    if result["status"] != "success":
        raise SystemExit(1)
