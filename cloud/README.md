# ComfyUI on Modal (cloud GPU)

`modal_comfy.py` runs the same ComfyUI as the Docker setup on a [Modal](https://modal.com) cloud GPU (L40S, 48 GB),
for when you have no local GPU or it is busy. Models are downloaded inside Modal, and the server scales to zero when
idle, so you pay only while it runs.

## Setup

1. Create a Modal account, add a payment method and set a workspace budget as a spending limit.
2. Install the Modal client and sign in:

   ```bash
   pip install modal
   modal setup
   ```

3. Download the models and deploy with the CLI (settings in `cloud/modal.env`, see below); it prints the
   `COMFY_SOURCES` entry for Pixel Studio:

   ```bash
   pixel-studio-comfy cloud modal deploy --download      # --gpu, --max-containers, --idle, --app, --live, --dry-run ...
   ```

   The CLI finds the Modal client on PATH or through the `PSC_MODAL` environment variable. The raw commands do the
   same, and print the server URLs:

   ```bash
   modal run cloud/modal_comfy.py::download              # --workflows pixel (default), all, or a comma-separated list
   modal deploy cloud/modal_comfy.py
   ```

5. In the Modal dashboard, create a **Proxy Auth Token**. Requests must send it as the `Modal-Key` and
   `Modal-Secret` headers.

## Use

Connect to the URL like any ComfyUI server, with the two headers. The first request after a quiet spell wakes the
GPU (about half a minute), and it shuts down again 5 minutes after the last request.

To take it down: `modal app stop sprite-studio-comfy`.

## Job API (parallel jobs)

`modal deploy` also publishes the **jobs** endpoint (`https://<workspace>--<app>-jobs.modal.run`): a stateless job API,
protocol `psc-job/v1` (see [docs/job-api.md](../docs/job-api.md)). Each job runs alone in its own ComfyUI container, up
to `MODAL_MAX_CONTAINERS` at once, and comes back with every file it wrote plus its timing and cost. The old
single-container `comfyui` endpoint above is unchanged and still deployed.

Settings live in `cloud/modal.env` (git-ignored; copy `cloud/modal.env.example`). Precedence: process environment >
`cloud/modal.env` > defaults, read once at deploy time:

| Key | Default | Meaning |
| --- | --- | --- |
| `MODAL_APP` | `sprite-studio-comfy` | app name; use another name to try changes without touching the live app. `cloud modal deploy` refuses this name unless you pass `--live` (a deploy changes the live endpoint's GPU and cost) |
| `MODAL_GPU` | `RTX-PRO-6000,L40S` | comma list, tried in order |
| `MODAL_MAX_CONTAINERS` | `3` | job containers at once |
| `MODAL_IDLE_SECONDS` | `300` | how long a warm container waits for the next job |
| `MODAL_PRESET` | `pixel` | default `--workflows` of `download` |
| `MODAL_JOB_TIMEOUT` | `3600` | per-job ceiling, seconds |

```bash
modal run cloud/modal_comfy.py::smoke          # one job on a job container: GPU, boot, run, cost, files
modal deploy cloud/modal_comfy.py              # prints the comfyui and jobs URLs
python cloud/job_smoke.py --url <jobs url> --n 1   # through HTTP; needs MODAL_KEY / MODAL_SECRET in the environment
```

Add the jobs endpoint to Sprite Studio as a `COMFY_SOURCES` entry: `pixel-studio-comfy cloud modal snippet` prints it
(`deploy` prints it too). Its headers are the `$MODAL_KEY` / `$MODAL_SECRET` references to your proxy auth token.

Cost: a job is billed from the moment its container starts. The first job of a container also pays the ComfyUI start
(`timing.boot`), and a container stays up `MODAL_IDLE_SECONDS` after its last job, which no job's `cost` includes.
Rates are in `cloud/gpu_rates.json` (read with `modal billing rates --json`). `cost.usd` counts the GPU only.

RunPod Serverless runs the same job API behind RunPod's `/run` + `/status`: see [runpod/README.md](runpod/README.md).
