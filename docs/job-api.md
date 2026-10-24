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
  The decoded total is capped at **256 MiB**; above it the job returns `status: "error"` and no files.
- `text`: text outputs of nodes, by node title.
- `timing`: seconds. `boot` is the container's ComfyUI start time, charged to the first job that container ran (0 for
  later ones); `billed = boot + run`. The idle time before a container scales down is billed by the provider but
  belongs to no job.
- `cost`: `usd = billed * rate_per_hour / 3600`, with the rate from `cloud/gpu_rates.json` matched on the GPU name the
  container reports. A GPU-only estimate: the provider's CPU and memory charges are a few percent more. `rate_per_hour`
  and `usd` are `null` for a GPU not in the file.
- `worker`: an opaque container id, so a test can show that jobs ran in different containers.

One job runs at a time on a ComfyUI process (new-file detection depends on it). A warm container takes the next job
with its models loaded; each job's inputs and outputs are deleted afterwards, so no job sees another's files.

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

## What another provider implements

Take a job document, call `job_runner.run_job(doc, proc, boot=..., rate=..., worker=...)` on a started
`ComfyProcess`, return `job_runner.encode_result(result)`; map the transport's own submit/status calls onto the
endpoints above (RunPod: `/run` and `/status`); serve `info` from `ComfyProcess.info()`.

Try it: `python cloud/job_smoke.py --url <jobs url> --n 1` (needs `MODAL_KEY` / `MODAL_SECRET` in the environment).
