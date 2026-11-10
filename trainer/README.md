# Style trainer

An optional service that trains a style LoRA for Z-Image Turbo from a folder of images, using
[ostris ai-toolkit](https://github.com/ostris/ai-toolkit). The finished LoRA goes into ComfyUI's `models/loras/`
folder; load it with **LoraLoaderModelOnly** and put its trigger word in the prompt.

## Start it

```bash
docker compose --profile train up -d trainer
```

It does not start with a plain `docker compose up -d`. Stop it when you are done to free its memory:
`docker compose stop trainer`.

The first run downloads the trainer image (about 6.6 GB) and the base model files (about 33 GB). Set
`TRAINER_CACHE_DIR` in `.env` to keep those on another drive, and `TRAIN_DIR` for the datasets, configs and results.

## Requirements

- An NVIDIA GPU with 24 GB of memory (RTX 3090 / 4090). Nothing else should use the GPU while it trains.
- About 24 GB of RAM for Docker. On Windows, raise WSL's memory limit in `%UserProfile%\.wslconfig` if a run stops
  with "Killed".

Training 15-60 images takes about 30-80 minutes on an RTX 4090. A checkpoint and sample images are saved every
250 steps, so you can keep the best one.

## API

The trainer listens on port 8675:

| Route | |
|---|---|
| `GET /health` | Whether it is busy |
| `POST /jobs` `{name, config_path, output_dir?, total_steps?}` | Starts a training run (one at a time) |
| `GET /jobs/<name>` | Progress, log, checkpoints and samples |
| `POST /jobs/<name>/cancel` | Stops the run |
| `GET /files/<name>/<path>` | Downloads a checkpoint or sample |

## On a rented machine

`trainer/cloud/Dockerfile` puts the same runner into an image for a rented GPU (RunPod). The runner is open to the
internet there, so it needs a token, limits how long it lives and ends the machine when it stops. Nothing changes for
the PC: `docker compose --profile train` sets `RUNNER_ALLOW_OPEN=1` (its port is published on 127.0.0.1 only) and runs
with no token, no upload and no time limit unless you ask for them.

### Variables

| Variable | Default | |
|---|---|---|
| `RUNNER_TOKEN` | none | The bearer token: 32 or more characters from `A-Z a-z 0-9 _ -` (what `secrets.token_urlsafe(32)` makes). Read once and removed from the runner's own environment and from the training process's environment, which stops accidental leaks only (see "What the token protects"). Anything else exits 2. |
| `RUNNER_HOST` | `0.0.0.0` | Without a token, anything but `127.0.0.1`, `::1` or `localhost` is refused (exit 2) unless `RUNNER_ALLOW_OPEN=1`. The cloud image never sets that. |
| `DATASETS_ROOT` | `/workspace/datasets` | Where uploaded datasets land. |
| `RUNNER_UPLOAD_MAX_BYTES` | 268435456 (256 MiB) | Largest upload body. |
| `RUNNER_TIME_LIMIT_S` | none; 10800 (3 h) with a token | Seconds from the first start. With a token it must be above 0. |
| `RUNNER_TIME_LIMIT_MAX_S` | 10800 (3 h, the approved limit) | The limit may never go above this, not even by `POST /deadline`; the image's hung-runner backstop is this plus 10 minutes. A larger `RUNNER_TIME_LIMIT_S` exits 2. A higher ceiling must be set in the pod's environment when the machine is created. |
| `RUNNER_IDLE_LIMIT_S` | 1800 with a token, else off | No running job and no authenticated request for this long. |
| `RUNNER_STATE_DIR` | `/workspace/runner-state` | `boot.json` keeps the first start, so a restarted container does not get a new clock. |
| `RUNNER_ON_EXIT` | `none`; the image sets `runpod-terminate` | `none`, `runpod-terminate` or `runpod-stop`: what `shutdown.py` asks RunPod to do when the runner has exited. Needs `RUNPOD_POD_ID` and `RUNPOD_API_KEY`, which RunPod puts into every pod. |
| `RUNNER_SHUTDOWN_RETRY_S` | 10 | Seconds between the 3 tries of that call. |
| `RUNNER_BOOT_ID` | `RUNPOD_POD_ID` | Whose clock `boot.json` holds. A `boot.json` written by another pod (a network volume reused by a new machine) is ignored and a fresh clock starts; the same pod restarting keeps its first start. |
| `RUNNER_SHUTDOWN_HOLD_S` / `RUNNER_SHUTDOWN_HOLD_MAX_S` | 60 / 300 | When ending the machine fails, `shutdown.py` retries for ever: first after 60 s, then the wait doubles up to 300 s. A `RUNNER_SHUTDOWN_HOLD_S` below 5 s (zero, text) falls back to 60, and an attempt that throws counts as failed, so the loop neither spins nor stops. It never exits (see below). `RUNNER_SHUTDOWN_HOLD=0` turns that off. |

With a token the runner also drops `HF_WRITE_TOKEN` and `RUNPOD_API_KEY` from the training process's environment. `HF_TOKEN`
stays, because gated base models need it. `RUNNER_WATCH_EVERY_S` is capped at 60 and every number must be finite (`inf` and `nan` exit 2).

### Routes

Every route needs `Authorization: Bearer <token>` when `RUNNER_TOKEN` is set, except `GET /health`, which then answers only
`{ok, version, auth: "required"}` without a valid token. A refusal is `401 {"error": "token required"}`, the same for a
missing and a wrong token. The token is never taken from a query string or a body.

| Route | |
|---|---|
| `PUT /datasets/<name>` | Body: a zip (`Content-Type: application/zip`) of flat image (png, jpg, jpeg, webp) and `.txt` caption files. Lands in `DATASETS_ROOT/<name>/`, replacing an earlier upload. Optional header `X-Content-SHA256`. Reply `201 {name, dataset_dir, images, captions, bytes, sha256}`. Refused with 411 (no `Content-Length`), 413 (too large), 409 (a run of that name is queued or running), 507 (disk) or 400 (the zip: the message names the entry). |
| `GET /datasets/<name>` | `{name, images, captions, bytes}` or 404: check an upload without sending it again. |
| `POST /jobs` `{name, config}` | Like `config_path`, but `config` is the ai-toolkit config as a **JSON object** (not YAML text; up to 256 KiB), written to `CONFIG_ROOT/<name>.json`. It is checked key by key against an allowlist (the keys `webapp/app/training.py` writes): `config.name` must equal the job name, every `datasets[].folder_path` must be inside `DATASETS_ROOT` (and name one dataset), `training_folder` inside `OUTPUT_ROOT`, `model.name_or_path` and `assistant_lora_path` only from a short list, and anything else is refused with a message naming the key: other path keys (`control_path`, `mask_path`, ...), `push_to_hub`, `hf_repo_id`, unknown keys, `NaN`. The file is written from the parsed object, so what was checked is what ai-toolkit reads. Give `config` or `config_path`, not both. |
| `POST /deadline` `{seconds}` | A whole number from 60 to 86400. New deadline = now + seconds, but never past first start + `RUNNER_TIME_LIMIT_MAX_S`. It can shorten the deadline too. 409 when no limit is active. |
| `GET /health` | With a valid token (or none configured) it also has `deadline_in_s` and `idle_in_s` (null when off). |

A zip entry must be a flat name `^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\.(png|jpg|jpeg|webp|txt)$` (no folders). `__MACOSX/` entries
and `._` files are skipped. At most 1000 entries; no encrypted entries, links or duplicates (case-insensitive); an image
is at most 32 MiB, a caption 64 KiB, everything 1 GiB, counted on the bytes really read. Pictures are checked by their first
bytes and captions must be UTF-8. The files are written to a fresh folder first and moved into place, so a refused upload
leaves nothing behind.

### Limits and shut-down

At the time limit or the idle limit the runner prints `runner: time limit reached` (or `idle limit reached`), stops the training
process, saves its status (`cancelled`, error `stopped: time limit reached`) and exits with code 3. It needs no request
from the app. `entrypoint.sh` then runs `shutdown.py` whatever the exit code (a pod started without a token exits 2 and is
ended too), and a coreutils `timeout` kills a runner that hangs. `shutdown.py` is standard library only and prints the mode
and the HTTP status, never the key.

If the watchdog thread itself fails for any reason, the runner prints that and exits 3 rather than going on with no limit.

**When ending the machine fails** (wrong endpoint or key scope, RunPod down, no network), `shutdown.py` tries the REST call and then `runpodctl pod remove <id>` (`runpodctl pod stop <id>` in stop mode, and the legacy `runpodctl remove pod <id>` / `runpodctl stop pod <id>` if the current form fails; it ships in RunPod pods and uses the pod's own key, nothing secret on its command line), 3 times. If both fail it does **not exit**: it keeps trying for ever, every 1 to 5 minutes (60 s, doubling, capped at 300 s). An exited container is restarted by RunPod, and the restarted runner would find the deadline passed, exit 3, fail again, and loop (billing) for ever; a container that never exits cannot. A missing key, a bad pod id and an unknown mode hold the same way. Such a pod keeps billing until it is ended in the console or by the app, which must also end any pod of ours that is past its deadline (ss-m10t4).

**A RunPod "Stop" in the console ends in deleting the pod.** RunPod sends `SIGTERM`, the runner exits 0, and the entrypoint then asks for `runpod-terminate` (or `runpod-stop`, which keeps the disk, if that is what is set). After a stop signal each call gets a 5 s timeout, because Docker allows only about 10 s.

`RUNNER_ON_EXIT=runpod-terminate` leaves nothing billing but deletes the pod's disk: checkpoints that were not copied
elsewhere are lost. `runpod-stop` keeps the disk, which stays billed until someone removes the pod.

The two RunPod calls (`DELETE /pods/{id}`, `POST /pods/{id}/stop` on `https://rest.runpod.io/v1`) have not run against RunPod
yet; they are checked against RunPod's documentation when the first real pod runs.

### What the token protects

The token is full control of the machine: with it a caller can start training runs, upload datasets, read any output file and move the deadline. The config check (above) keeps a run inside `DATASETS_ROOT` and `OUTPUT_ROOT` and off the internet except for the listed models; it does not make the token harmless.

What the training process can see of the token: it cannot get it from its own environment (the runner removes `RUNNER_TOKEN`, `HF_WRITE_TOKEN` and `RUNPOD_API_KEY` there), but everything in the container runs as root, and Linux keeps the environment a process started with. `RUNNER_TOKEN` and `RUNPOD_API_KEY` are still in `/proc/1/environ` (the entrypoint, which RunPod started with them), in the `timeout` process, and in the runner's own start-up environment block. So code running in the training process, for instance code a model or config makes it run, could read them. The scrub covers accidental dumps (a printed environment, a crash report) only. Closing that gap needs the training child to run as a non-root user, which is not done here because the ostris image and its caches expect root. `HF_TOKEN` is passed to the training process on purpose.

### Before the first rented machine

Nothing has run on RunPod yet, and the app cannot yet end its own pods. Until it can, do this in order:

1. **Test on a cheap CPU machine first.** Rent the smallest CPU pod, run `shutdown.py` there once in each mode and see the pod disappear, both through the REST call and through `runpodctl` (ss-m10t10). Both calls are unverified.
2. **Keep a small balance with auto top-up off.** The prepaid balance is the last limit on what a forgotten pod can cost.
3. **Watch the first GPU run** and confirm the pod is gone after the limit (3 hours at most, 30 minutes idle).
4. **The app-side cleanup (ss-m10t4) is the missing second line of defence.** It ends any pod of ours that is past its deadline. Until it exists, nothing automatic stands behind the runner's limits and the two shut-down calls, so do not leave a run unattended.

### Build the image

```bash
pixel-studio-comfy cloud trainer build --dry-run     # print the docker commands
pixel-studio-comfy cloud trainer build               # build ghcr.io/bunnyufo-lab/psc-trainer:aitk-<base>-<sha> and :latest
pixel-studio-comfy cloud trainer build --push        # also push (needs `docker login ghcr.io`)
```

`--base IMAGE` changes the base (default `ostris/aitoolkit:0.13.23@sha256:35e46171bc6def5e745fc478befec2ab0a0551fe78ce3d5f76cb22271795bb52`, pinned by digest, about 6.6 GB to download); `--tag IMAGE` names the result.
The image is published as a public GHCR package, so RunPod needs no registry login. It holds no token and no key: the token
and the limits come as environment variables when the pod is rented.

### Threats and answers

| Threat | Answer |
|---|---|
| The port is public (RunPod's proxy URL) | Every route but a minimal `/health` needs a 32+ character token; with no token the runner refuses a non-local address. |
| The token in a log | Removed from the environment at start, taken from the header only, never printed, dropped from the training process's environment (not from `/proc`, see above); the tests search the logs. The request log shows the request line without its query. |
| Path traversal | Flat zip names on an allowlist, links refused, extraction into a fresh folder then a rename; the `/files` checks stay. |
| Zip bomb, full disk | Body cap, entry cap, per-file and total caps on bytes actually read, a free-space check. |
| A forgotten machine keeps billing | A time limit from the first start that survives restarts (and is tied to the pod, so a reused volume cannot carry an old deadline), an idle limit, a watchdog that exits the runner if it fails, a coreutils backstop, the entrypoint ends the machine on any exit by two methods and never exits if both fail, a pod without a token ends itself. The app (ss-m10t4) is the third line: it ends any pod of ours past its deadline. |
| A config that reads or writes elsewhere, or pushes to Hugging Face | `config` is a JSON object checked against a key allowlist with every path confined; nothing is scanned as text. |
| A replayed upload | It needs the token over RunPod's HTTPS proxy; an upload is idempotent (the same bytes give the same dataset and sha256) and refused while that run trains. |
