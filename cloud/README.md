# ComfyUI on Modal (cloud GPU)

`modal_comfy.py` runs this repo's ComfyUI on a Modal L40S (48 GB) for Sprite Studio
to use when the PC is busy or off. It is the same ComfyUI as `docker/Dockerfile`
(tag, custom nodes, the repo's own nodes and workflows). Models live on a Modal
Volume and are downloaded inside Modal, so nothing large goes up from home.

Sprite Studio calls it over HTTPS like any ComfyUI. The Modal CLI is only needed
to deploy or update it, from this PC.

## One-time setup

1. Modal account: add a payment method (unlocks the monthly free credits), then
   set a **workspace budget** in the billing settings as a hard spending stop.
2. Client, in its own venv (not this repo's):

   ```bash
   py -m venv ~/.venvs/modal
   ~/.venvs/modal/Scripts/python.exe -m pip install modal
   ~/.venvs/modal/Scripts/modal.exe setup      # browser sign-in; token in ~/.modal.toml
   ```

3. Models (CPU container; resumable, skips what is already there):

   ```bash
   ~/.venvs/modal/Scripts/modal.exe run cloud/modal_comfy.py::download
   ~/.venvs/modal/Scripts/modal.exe run cloud/modal_comfy.py::list_models
   ```

   The default (`--workflows pixel`) is every workflow Sprite Studio uses, the
   same list as `env/pixel.env`. `--workflows starter` is just drafts, prompt to
   sprite, pixelize and Qwen edits (about 30 GB); `all` or a comma-separated list
   also work. Storage is free up to 1 TiB a month, then $0.09 per GiB-month.

4. Deploy (prints the endpoint URL, `https://<workspace>--sprite-studio-comfy-comfyui.modal.run`):

   ```bash
   ~/.venvs/modal/Scripts/modal.exe deploy cloud/modal_comfy.py
   ```

5. Proxy auth token: Modal dashboard, workspace settings, **Proxy Auth Tokens**,
   create one. Its key and secret go only into the server's environment
   (`MODAL_KEY`, `MODAL_SECRET`), never into Git. Requests without them get 401.

## Using it from Sprite Studio

```
COMFY_SOURCES=[{"name":"pc","url":"http://<pc address>:8188"},
               {"name":"cloud","url":"https://<workspace>--sprite-studio-comfy-comfyui.modal.run","probe":0,
                "cost_per_hour":1.95,"headers":{"Modal-Key":"$MODAL_KEY","Modal-Secret":"$MODAL_SECRET"}}]
```

`cost_per_hour` (the L40S price) lets the job queue show roughly what each cloud
job cost. The client turns WebSocket compression off: Modal's proxy accepts a
compressed socket and then closes it at once, which loses live progress.

`probe: 0` because any request wakes the GPU (and starts billing).

## Test by hand

```bash
curl -H "Modal-Key: $MODAL_KEY" -H "Modal-Secret: $MODAL_SECRET" https://<url>/system_stats
```

The first request after idle starts a container: expect a cold start of about
half a minute, then model loading on the first job.

## Costs and settings

- The GPU bills per second while a container runs, including the idle window
  after the last request (`MODAL_IDLE_SECONDS`, default 300).
- `max_containers=1`: one GPU at most, whatever is queued.
- Measured: cold start (asleep to ComfyUI answering) about 23 s; a sprite job
  on a warm GPU about 14 s; the first job after a wake also loads its models.
- `MODAL_GPU` (default `L40S`) and `COMFYUI_REF` (default `v0.38.2`, keep in
  step with `docker-compose.yml`) are read at deploy time.
- `~/.venvs/modal/Scripts/modal.exe app stop sprite-studio-comfy` takes it down.
