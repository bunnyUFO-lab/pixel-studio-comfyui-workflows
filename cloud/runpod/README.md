# ComfyUI on RunPod Serverless

The same job API as the Modal one (protocol `psc-job/v1`, see [docs/job-api.md](../../docs/job-api.md)), behind
RunPod's native `/run` + `/status`. The worker is the PC's ComfyUI image plus `handler.py`; it keeps one ComfyUI warm,
runs one job at a time, and RunPod scales the number of workers. Models live on a network volume, not in the image.

| | |
| --- | --- |
| Image | `ghcr.io/bunnyufo-lab/psc-comfy-runpod:<ComfyUI ref>-<git sha>` (public package: open-source code only, no models, no secrets) |
| GPU | 24 GB 4090 first, then 48 GB (L40S / RTX 6000 Ada), then 32 GB 5090. The PC runs every Pixel Studio workflow on a 4090 |
| Cost | GPU only, flex workers, from `gpu_rates.json` (runpod.io/pricing, 24 Oct 2026): 4090 $1.10/h, 5090 $1.58/h, L40S $1.75/h. Active (always-on) workers are priced differently and are not modelled. A network volume is about $0.07 a GB a month: the `pixel` set (145 GB) needs 200 GB, about $14 a month on top of the GPU time |
| Limits | request 10 MB (inputs inline, at most 9 MiB), result kept 30 minutes, one job per worker |

## One-time setup

Steps 2 and 3 are the owner's: they publish an image and need a RunPod account and key.

1. Build the image from the local ComfyUI image (`docker compose build comfyui` first if `comfyui-local:<ref>` does not
   exist). Nothing is pushed:

   ```bash
   pixel-studio-comfy cloud runpod build          # --dry-run prints the docker command; --base / --tag change them
   ```

2. **Owner:** publish it. Log in to GHCR with a personal access token that has `write:packages`, push, then set the
   package to **public** in GitHub (Packages > psc-comfy-runpod > Package settings):

   ```bash
   docker login ghcr.io
   pixel-studio-comfy cloud runpod build --push
   ```

3. **Owner:** in the RunPod console create an API key (Settings > API Keys; export it as `RUNPOD_API_KEY` in the
   environment Pixel Studio runs with), a **network volume** (200 GB: the `pixel` set downloads 145 GB, measured 24 Oct 2026, so 160 GB is the least that fits; in a data center that has both 4090 and L40S) and
   a **queue-based Serverless endpoint** from the image. `pixel-studio-comfy cloud runpod settings` prints what to
   enter: GPU priority, 0 active and 3 max workers, idle timeout 300 s, **execution timeout 3900 s** (the default 600 s
   kills video jobs), FlashBoot on, scaling by request count (1), container disk 20 GB, the volume attached.
4. Fill the volume with models. The `download` job runs `models.py` on a worker (it resumes partial files: run it again
   if it hits the execution timeout). `pixel` is every workflow Sprite Studio uses (sizes are in `env/pixel.env`; `list_models` shows what is missing; or name
   workflows, `starter`, `all`):

   ```bash
   curl -s -X POST https://api.runpod.ai/v2/$ENDPOINT/run -H "Authorization: Bearer $RUNPOD_API_KEY" \
     -H "Content-Type: application/json" -d '{"input": {"action": "download", "workflows": "pixel"}}'
   curl -s https://api.runpod.ai/v2/$ENDPOINT/status/<id> -H "Authorization: Bearer $RUNPOD_API_KEY"
   ```

   `{"action": "list_models", "workflows": "pixel"}` shows what is there and what is missing.
5. Try it: `python cloud/job_smoke.py --runpod <endpoint id> --n 1` (the first run pulls the image onto a host, which
   takes minutes; later cold starts are faster with FlashBoot).
6. Add the endpoint to Pixel Studio: `pixel-studio-comfy cloud runpod snippet --endpoint <endpoint id>` prints the
   `COMFY_SOURCES` entry (its header is the `$RUNPOD_API_KEY` reference, never the key).

## Large outputs

RunPod documents a 10 MB request limit and keeps `/run` results for 30 minutes. The worker assumes the same 10 MB for
the result. When the encoded result is bigger (videos, many frames) it uploads each file to an S3-compatible bucket
and returns presigned URLs (`url` + `expires` in place of `data`; the client does a plain GET). Without a bucket the
job returns `status: "error"` saying so. Set these on the endpoint (Environment variables); keep the keys in RunPod
**secrets** and refer to them as `{{ RUNPOD_SECRET_<name> }}`:

| Variable | Meaning |
| --- | --- |
| `PSC_S3_BUCKET` | bucket name; setting it turns the upload on |
| `PSC_S3_ENDPOINT` | optional: the S3 API URL of R2, MinIO or RunPod's S3-compatible storage |
| `PSC_S3_PREFIX` | key prefix, default `psc-jobs/` (keys are `<prefix><job id>/<type>/<subfolder>/<filename>`) |
| `PSC_S3_URL_SECONDS` | how long a URL works, default 3600 |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` | credentials (secrets) |

Give that key `s3:PutObject` and `s3:GetObject` on the prefix only, and add a 1-day lifecycle rule that deletes it.

## Notes

- **Inputs** are inline base64; the request JSON may be at most 9 MiB (`PSC_INLINE_MAX_BYTES`, default 9437184). A
  larger job fails with an error naming its size; use the Modal source for it.
- **Cancels and timeouts**: the worker registers an async handler that runs the job in a thread (the SDK runs a sync
  handler on its own event loop, where a cancel could not arrive until the job ended). When RunPod cancels or times
  out a job, the handler interrupts ComfyUI and clears its queue at once; the job's thread ends and the worker is free.
  If the thread is still finishing when the next job arrives, that job interrupts again and waits up to 120 s.
  Not yet verified on real RunPod.
- **Crashes**: if ComfyUI died, did not answer or timed out, the worker stops it and the next job starts a fresh one
  (one more boot, charged to that job); the failed job reads FAILED with the reason.
- **Boot** (starting ComfyUI) is charged to the worker's first job, as on Modal; `info` is a job, so it wakes a worker
  that the next real job reuses (idle timeout 300 s).
- **Models**: with a volume attached the worker links `/opt/ComfyUI/models` to `/runpod-volume/models`; without one it
  uses the image's empty folder (the local test below).

## Test the worker locally

No RunPod account, key or spend. The SDK serves the handler on a local port (`/runsync`); use an isolated container name
and not the live ComfyUI stack:

```bash
docker run --rm -d --name psc-runpod-test --gpus all -p 127.0.0.1:8010:8000 psc-comfy-runpod:dev \
  python -u /opt/psc/handler.py --rp_serve_api --rp_api_host 0.0.0.0 --rp_api_port 8000
curl -s -X POST localhost:8010/runsync -H "Content-Type: application/json" \
  -d '{"input": {"action": "download", "workflows": "pixel-art/image-to-pixel"}}'    # 444 MB into the container
python cloud/job_smoke.py --local http://127.0.0.1:8010 --n 1
docker stop psc-runpod-test
```

`-e PSC_INLINE_MAX_BYTES=1000` shows the over-the-limit error. On Windows with Git Bash, set `MSYS_NO_PATHCONV=1` so the
`/opt/psc/handler.py` path is not rewritten. Unit tests (no Docker): `python tests/test_runpod_handler.py`,
`python tests/test_psc_runpod.py`.
