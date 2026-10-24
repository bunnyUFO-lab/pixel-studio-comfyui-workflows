# ComfyUI job protocol `psc-job/v1`

A small, provider-neutral way to run one ComfyUI workflow on a cloud GPU and get everything it wrote back. A client
sends a **job document** (an API-format graph plus its input files) and receives a **result document** (every output
file, text outputs, errors, timing and cost). Nothing in the documents names a provider. Modal serves it over HTTP
today (`cloud/modal_comfy.py`); other hosts implement the same two documents behind their own transport, reusing
`cloud/job_runner.py`, which has no provider code.

Why not just expose ComfyUI? Its `/prompt`, WebSocket and `/history` must reach the same process, and a
load-balanced endpoint with several containers gives no such guarantee. Here the whole job travels in one call and the
whole result comes out of it, so any number of containers can run jobs side by side.

## Job document

```json
{"protocol": "psc-job/v1",
 "workflow": {"<node id>": {"class_type": "...", "inputs": {}, "_meta": {"title": "..."}}},
 "inputs":   [{"name": "ss_1a2b3c4d5e_cat.png", "data": "<base64>"}],
 "timeout":  3600}
```

- `workflow`: a ComfyUI **API-format** graph. It references each input file by `name`. The server does not convert UI
  workflows.
- `inputs[].name`: a plain file name (no `/`, `\`, `..` or NUL, no leading `.`, at most 255 characters, unique in the
  job). The file is written to ComfyUI's `input/` folder under that name and deleted after the job.
- `inputs[].data`: base64. The decoded total is capped at **64 MiB** (HTTP 413 above it).
- `timeout`: seconds, 1 to 3600, default 3600.

## Result document

```json
{"protocol": "psc-job/v1", "status": "success",
 "files": [{"node": "Save pixel art (native size)", "filename": "pixel_00001_.png", "subfolder": "", "type": "output",
            "size": 1234, "data": "<base64>"}],
 "text":  {"Check report": "PASS ..."},
 "errors": [],
 "timing": {"boot": 41.2, "run": 18.7, "billed": 59.9, "started_at": 1761300000.1, "finished_at": 1761300018.8},
 "cost":   {"gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "rate_per_hour": 3.03, "usd": 0.0504},
 "worker": "ta-01ABC..."}
```

- `status`: `"success"` or `"error"`. An error means ComfyUI rejected the graph or a node failed; `errors` then has
  `"<node_type>: <message>"` lines. Infrastructure failures (no GPU, container crash, timeout) are **not** results:
  they surface as a failed job on the transport.
- `files`: the files ComfyUI's history lists (with the title of the node that saved them in `node`) **plus** every
  other file that appeared or changed under `output/` and `temp/` during the job (`node` is `""`). Clients need the
  second kind: nodes write siblings (`x.json`, `x.gif`) next to the listed file. `type` is `"output"` or `"temp"`.
  The decoded total is capped at **256 MiB**; above it the job returns `status: "error"` and no files. A file entry
  carries either `data` (base64) or `url` (with `expires`, epoch seconds): an HTTPS GET with no headers returns the
  bytes. Clients must accept both; Modal only ever sends `data`, RunPod sends `url` for large results (below).
  `size` is always the decoded size.
- `text`: text outputs of nodes, by node title.
- `timing`: seconds. `boot` is the container's ComfyUI start time, charged to the first job that container ran (0 for
  later ones); `billed = boot + run`. It is a lower bound: `boot` starts when ComfyUI starts, so container allocation,
  image load and Volume mount are not in it, and the idle time before a container scales down belongs to no job.
- `cost`: `usd = billed * rate_per_hour / 3600`, with the rate from `cloud/gpu_rates.json` matched on the GPU name the
  container reports. A GPU-only estimate: the provider's CPU and memory charges are a few percent more. `rate_per_hour`
  and `usd` are `null` for a GPU not in the file.
- `worker`: an opaque container id, so a test can show that jobs ran in different containers.

One job runs at a time on a ComfyUI process (new-file detection depends on it). A warm container takes the next job
with its models loaded; each job's inputs and outputs are deleted afterwards, so no job sees another's files.

Warm containers and failures:
- ComfyUI runs with `--cache-none` (`job_runner.JOB_ARGS`). Its execution cache would otherwise skip the save nodes
  of a repeated identical job (same graph, same input bytes), which then writes no files while the history still lists
  them. `--cache-none` runs every node of every job, whatever its type; the price is that model loaders re-read their
  weights on each job. If a listed file is still missing, the job is `status: "error"`, never a silent success.
- The job's `timeout` is cut to the deployment's ceiling (`MODAL_JOB_TIMEOUT`).
- After an infrastructure failure (ComfyUI died, did not answer, timed out) or when ComfyUI is found dead, the container
  takes no more jobs (`job_runner.run_job_guarded` calls the provider's `recycle`; Modal:
  `modal.experimental.stop_fetching_inputs`). The next job pays one cold start. A timed-out job's outputs are deleted.

## HTTP transport (Modal)

All requests carry the workspace's proxy auth token as `Modal-Key` / `Modal-Secret` headers (a Modal proxy auth
token; there is no token of our own). Without them every path answers 401.

| Method, path | Does |
| --- | --- |
| `POST /v1/jobs` | Validate and queue a job. `202 {"id": "<call id>", "status": "queued"}`, `400 {"error"}` for a bad document, `413` for inputs over the cap. |
| `GET /v1/jobs/{id}` | `{"id", "status", "progress", "result"?, "error"?}`. `status` is `queued` (waiting for or starting a container), `running`, `done` (with `result`), `failed` (with `error`) or `cancelled`. `404` for an unknown id or an expired result. A finished result stays readable for a while, so a dropped download is retried by asking again. |
| `DELETE /v1/jobs/{id}` | Cancel; the container is terminated so the next job does not wait behind the running prompt. `{"status": "cancelled"}`. |
| `GET /v1/info` | `{"object_info": {...}, "models": {"<folder>": [files]}, "comfyui_version", "updated"}` from a cache, so no GPU wakes. Refreshed when empty, with `?refresh=1`, and by every container start. |
| `GET /v1/health` | `{"ok", "protocol", "app", "gpu": [...], "max_containers", "idle_seconds"}`. Never wakes a GPU. |

Clients poll `GET /v1/jobs/{id}` (a blocking POST would hit the 150 s web request limit). `progress` is
`{"node": "<api node id>", "value": n, "max": m, "done": ["<node id>", ...], "total": <node count>}` or `null`.

`GET /v1/info` is what a client needs before it can submit: it converts UI workflows to API graphs with `object_info`
and checks `models` for what a workflow needs.

## RunPod transport

`cloud/runpod/handler.py` serves the same two documents on a RunPod Serverless queue endpoint
(`https://api.runpod.ai/v2/<endpoint id>`, header `Authorization: Bearer <RunPod API key>`). The job document is the
request's `input`, the result document is the status's `output`. Setup: [cloud/runpod/README.md](../cloud/runpod/README.md).

| Does | RunPod call |
| --- | --- |
| submit | `POST /run {"input": <job document>}` -> `{"id", "status": "IN_QUEUE"}` |
| status | `GET /status/{id}`: `IN_QUEUE` = queued, `IN_PROGRESS` = running (`output` is the progress object while it runs), `COMPLETED` = done (`output` is the result document), `FAILED` / `TIMED_OUT` = failed (`error`), `CANCELLED` = cancelled |
| cancel | `POST /cancel/{id}` |
| info | a job: `POST /run {"input": {"action": "info"}}`, `output` is the `info` document. RunPod has nothing that answers without a worker; the first call wakes one, which the next job reuses |
| health | `GET /health` (workers and queue; wakes nothing) |
| models | jobs `{"action": "download", "workflows": "pixel"}` (fills the network volume) and `{"action": "list_models", ...}` |

Limits that shape the documents: `/run` takes at most **10 MB** of request JSON and RunPod keeps a `/run` result for
**30 minutes**. No separate output limit is documented, so the worker assumes the same 10 MB on `output`:

- **Inputs** travel inline, base64, with the whole request JSON at most **9 MiB** (`PSC_INLINE_MAX_BYTES`, default
  9437184). A larger job is refused with an error naming the size (use the Modal source for it).
- **Outputs**: when the encoded result is over the same budget, the worker uploads every file to the S3-compatible
  bucket configured on the endpoint (`PSC_S3_*`) and returns presigned GET `url`s in place of `data`. Without a
  bucket the result is `status: "error"` and says so. The bucket credentials live on the endpoint as RunPod secrets;
  a client only does a plain GET.
- One job at a time per worker; scale-out is more workers. A cancelled job's prompt is interrupted when the worker's
  next job starts.

A failed job (`FAILED`) is an infrastructure failure, as on Modal; a rejected graph or a failed node is a `COMPLETED`
job whose result has `status: "error"`.

## What another provider implements

Take a job document, call `job_runner.run_job_guarded(doc, proc, recycle, boot=..., rate=..., worker=...,
max_timeout=...)` on a started `ComfyProcess` (`recycle` stops the worker taking more jobs), return
`job_runner.encode_result(result)` (a file may carry `url` instead of `data`); map the transport's own submit/status
calls onto the endpoints above (RunPod does: see "RunPod transport"); serve `info` from `ComfyProcess.info()`.

Try it: `python cloud/job_smoke.py --url <jobs url> --n 1` (needs `MODAL_KEY` / `MODAL_SECRET` in the environment), or
`python cloud/job_smoke.py --runpod <endpoint id> --n 1` (needs `RUNPOD_API_KEY`).
