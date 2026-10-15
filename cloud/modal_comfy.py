"""ComfyUI on a Modal cloud GPU, for Sprite Studio to use when the PC is busy or off.

The same ComfyUI as docker/Dockerfile (tag, custom nodes, the repo's own nodes and workflows); models live on a
Modal Volume, downloaded inside Modal by docker/models.py from the URLs in the workflows. The endpoint is locked with
Modal proxy auth (ComfyUI has no login): every request needs the Modal-Key / Modal-Secret headers of a proxy auth
token, which Sprite Studio sends (COMFY_SOURCES headers). It scales to zero when idle.

From the repo root, with the Modal client (see cloud/README.md):
    modal run cloud/modal_comfy.py::download                 # fetch every Sprite Studio model into the Volume
    modal run cloud/modal_comfy.py::download --workflows starter   # or "all", or a comma-separated list
    modal deploy cloud/modal_comfy.py                        # publish the GPU endpoint (prints its URL)
    modal app stop sprite-studio-comfy                       # take it down
"""
import os
import subprocess
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
COMFYUI_REF = os.environ.get("COMFYUI_REF", "v0.38.2")    # keep in step with docker-compose.yml
GPU = os.environ.get("MODAL_GPU", "L40S")                # 48 GB: fits every pixel-art workflow
IDLE = int(os.environ.get("MODAL_IDLE_SECONDS", "300"))  # a warm GPU stays up this long after the last request
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
    .env({"MODELS_DIR": MODELS, "WORKFLOWS_DIR": WORKFLOWS})
    # Added at container start (not baked), so a workflow or node change needs no image rebuild: just deploy again.
    .add_local_file(REPO / "docker" / "models.py", "/usr/local/bin/models.py")
    .add_local_file(REPO / "env" / "pixel.env", "/opt/env/pixel.env")
    .add_local_dir(REPO / "workflows", WORKFLOWS)
    .add_local_dir(REPO / "custom_nodes" / "comfyui_workflows_utils", f"{COMFY}/custom_nodes/comfyui_workflows_utils",
                   ignore=["**/__pycache__/**"])
)

app = modal.App("sprite-studio-comfy", image=image)
models = modal.Volume.from_name("sprite-studio-models", create_if_missing=True)


@app.function(volumes={MODELS: models}, timeout=6 * 3600)
def download(workflows: str = "pixel"):
    """Download the models the given workflows need into the Volume (skips what is there, resumes partial files).
    workflows: "pixel" (every workflow Sprite Studio uses, as in env/pixel.env), "starter", "all", or a comma-separated
    list. Runs on a CPU container; HF_TOKEN is not needed for the public models the workflows list."""
    names = _workflows(workflows)
    rc = subprocess.call(["python", "/usr/local/bin/models.py", "download", *names])
    models.commit()
    subprocess.call(["du", "-sh", MODELS])
    if rc:
        raise SystemExit(f"models.py exited with {rc} (see above); what was downloaded is kept")


@app.function(volumes={MODELS: models}, timeout=600)
def list_models(workflows: str = "pixel"):
    """Which models the workflows need and whether the Volume has them."""
    subprocess.call(["python", "/usr/local/bin/models.py", "list", *_workflows(workflows)])


def _workflows(value):
    if value == "pixel":
        line = next(l for l in Path("/opt/env/pixel.env").read_text().splitlines() if l.startswith("MODELS="))
        value = line.split("=", 1)[1]
    elif value == "starter":
        value = STARTER
    return [w.strip() for w in value.split(",") if w.strip()]


@app.function(gpu=GPU, volumes={MODELS: models}, max_containers=1, scaledown_window=IDLE, timeout=24 * 3600)
@modal.concurrent(max_inputs=100)       # one GPU container answers all of Sprite Studio's requests and its WebSocket
@modal.web_server(8188, startup_timeout=600, requires_proxy_auth=True)
def comfyui():
    # ComfyUI looks for models in its own models folder: point it at the Volume.
    subprocess.run(f"rm -rf {COMFY}/models && ln -s {MODELS} {COMFY}/models", shell=True, check=True)
    subprocess.Popen(["python", f"{COMFY}/main.py", "--listen", "0.0.0.0", "--port", "8188"], cwd=COMFY)
