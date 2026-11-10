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

The RunPod calls use REST v2 (`https://api.runpod.io/v2`): `DELETE /v2/pods/{id}` to delete the machine (204 when done) and `POST /v2/pods/{id}/action` with `{"action": "stop"}` for stop mode. RunPod's older REST v1 retires on 15 November 2026; no v1 call is left and `shutdown.py` refuses a v1 base. The key is the pod-scoped one RunPod puts in the machine's environment; it goes only into an `Authorization` header. A real rehearsal on 10 Nov 2026 proved that this key deletes its own pod through v2 on a CPU machine (v1 answered 403 to it). This trainer image itself has still not run on RunPod.

### What the token protects

The token is full control of the machine: with it a caller can start training runs, upload datasets, read any output file and move the deadline. The config check (above) keeps a run inside `DATASETS_ROOT` and `OUTPUT_ROOT` and off the internet except for the listed models; it does not make the token harmless.

What the training process can see of the token: it cannot get it from its own environment (the runner removes `RUNNER_TOKEN`, `HF_WRITE_TOKEN` and `RUNPOD_API_KEY` there), but everything in the container runs as root, and Linux keeps the environment a process started with. `RUNNER_TOKEN` and `RUNPOD_API_KEY` are still in `/proc/1/environ` (the entrypoint, which RunPod started with them), in the `timeout` process, and in the runner's own start-up environment block. So code running in the training process, for instance code a model or config makes it run, could read them. The scrub covers accidental dumps (a printed environment, a crash report) only. Closing that gap needs the training child to run as a non-root user, which is not done here because the ostris image and its caches expect root. `HF_TOKEN` is passed to the training process on purpose.

### Checkpoints to Hugging Face

On a rented machine the runner can send each finished checkpoint, the samples and the run's config to a **private** Hugging Face model repository while the run trains, and make a last pass when it ends, so the LoRA outlives the machine (`RUNNER_ON_EXIT=runpod-terminate` deletes the disk). `trainer/cloud/hf_upload.py` does the sending: a child process run with the image's python and its `huggingface_hub` (version **1.23.0** in `ostris/aitoolkit:0.13.23`, checked with `docker run`; the code also works with 0.34.3). `runner.py` stays standard library only. Without the two variables below nothing changes: the PC and every run without them report `upload: {"state": "off"}`.

| Variable | Default | |
|---|---|---|
| `HF_WRITE_TOKEN` | none | A Hugging Face write token (`hf_` and 20 or more letters or digits). The app puts it in the pod's environment at rent time. Read once and removed from the runner's environment, like `RUNNER_TOKEN`. |
| `HF_REPO` | none | `owner/sprite-studio-loras`. Both variables or neither; one without the other, a bad form, or either without `RUNNER_TOKEN` exits 2 with a message that names the variable and never its value, so a misconfigured pod ends before it trains. |
| `RUNNER_UPLOAD_EVERY_S` | 60 | Seconds between passes while the job runs. A pass takes the files whose size and modification time did not change since the previous pass and whose modification time is at least `RUNNER_UPLOAD_SETTLE_S` (default 30) old. One pass at a time, one commit per pass. |
| `RUNNER_UPLOAD_GRACE_S` | 300 (the image sets it) | How long the runner may spend on the last pass at the time or idle limit. Above 480 exits 2, so the grace always ends before the image's backstop at limit + 600 s. |
| `RUNNER_UPLOAD_ALLOW_CLASSIC` | unset | `1` lets a **classic** (account-wide) token upload. For a test on the owner's own PC only; **must not be set on a rented machine**. See "Before a token goes on a rented machine". |
| `RUNNER_UPLOAD_ALLOW_PUBLIC` | unset | `1` lets the runner upload to a repository that is not private. Set only when a machine is created on purpose for that; never as a default. |
| `RUNNER_UPLOAD_CMD` | `python -I -u /runner/hf_upload.py upload` | The child (shell words or a JSON array, like `RUN_CMD`). Tests point it at a fake. |

**What is sent.** Every `*.safetensors` at the top of `OUTPUT_ROOT/<job>/` (each checkpoint the run keeps, and the final file), every image in `samples/`, and the config the runner wrote (`CONFIG_ROOT/<job>.json`, sent as `config.json`). Never `runner.log` (it holds whatever ai-toolkit prints). Layout: `<base model>/<job>/<file>` and `<base model>/<job>/samples/<file>`, on `main`. The base model comes from the config's `model.name_or_path` through a table (`MODEL_KEYS`, today `Tongyi-MAI/Z-Image-Turbo` -> `z-image-turbo`); a model not in it ends in `upload: failed, unknown base model`. Names must match `^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\.(safetensors|jpg|jpeg|png|webp|json)$`; only regular files (no symlinks) inside the job folder; at most 2 GiB per file, 6 GiB and 300 files per run. A file over a cap or with another name is `skipped` and says why. The uploader never deletes, moves or rewrites a local file.

**What `GET /jobs/<name>` reports.** A new `upload` object (also kept in `OUTPUT_ROOT/<job>/upload_status.json`, which holds no token, so a restarted runner still reports it):

```json
{"state": "off|waiting|uploading|done|failed", "repo": "owner/sprite-studio-loras", "error": null,
 "files": [{"file": "<job>_000000250.safetensors", "path": "z-image-turbo/<job>/<job>_000000250.safetensors",
            "sha256": "<64 hex>", "size": 171966464, "state": "done|pending|failed|skipped|gone", "error": null,
            "url": "https://huggingface.co/<repo>/resolve/<40-hex commit>/<path>"}]}
```

`sha256` and `size` come from the runner's own streaming hash of the local file before the upload. After the commit the child asks the hub for the file again (`get_paths_info(expand=True)`) and the size and, when present, `lfs.sha256` must match, else the file is `failed: checksum differs`. The `url` uses the commit id, so it never changes. `done` means the last pass ended and every checkpoint is `done` (a sample or `config.json` that failed only adds its `error`; `gone` is a step save that ai-toolkit removed before it was sent and does not count against the run); `failed` means a checkpoint was not sent (for instance skipped by a cap), the token or repository was refused (HTTP 401, 403 or 404 is not retried and later passes are skipped), or the time limit cut the last pass short. The app waits for `done`, `failed` or `off` before it deletes the machine (ss-m10t4, at most 15 minutes).

**Retries and limits.** Four tries per pass, after 10, 30 and 90 s; a child that runs more than 900 s is killed with its process group (`failed: timeout`). An upload in flight counts as activity for the idle limit. At the time or idle limit the runner stops training as before, then waits for a pass in flight, makes the last pass with everything left and kills the child when `RUNNER_UPLOAD_GRACE_S` ends; what was not sent becomes `failed: time limit`, the status is saved and the runner exits 3. A SIGTERM (someone deleting the pod) gets no grace; the app deletes only after the upload says `done`, `failed` or `off`. A failed upload never loses the local checkpoint before the machine goes: files stay where they are and the app can still fetch them over `/files`.

**The token.** It travels: the app (from the PC user variable `HF_TOKEN`) -> the pod's environment as `HF_WRITE_TOKEN` -> the runner pops it at start -> each upload child gets it as one JSON line on its **stdin**, then stdin is closed. Never in an argument, a file, a URL or a child's environment (the child's environment is the runner's minus `RUNNER_TOKEN`, `HF_WRITE_TOKEN`, `RUNPOD_API_KEY` and `HF_TOKEN`, plus `HF_HUB_DISABLE_TELEMETRY`, `HF_HUB_DISABLE_PROGRESS_BARS` and `HF_HUB_DISABLE_IMPLICIT_TOKEN`, so the library uses only the token it is given). The child prints result lines only, with the exception class name and HTTP status as the error (`HfHubHTTPError 401`), never a message, URL or header, and every line it prints passes through a replacement of the token by `***`. The runner stores only an error that looks like a short plain phrase, redacts it too, discards everything else the child prints (its stderr goes nowhere) and logs one line per pass (`runner: upload <n> files: <ok> done, <failed> failed`). `/health` and every reply carry no token.

**Before a token goes on a rented machine.** Each upload child checks these before it sends anything, and refuses (the upload ends `failed` with the reason, and later passes are skipped) when one does not hold:

- **Only a fine-grained token limited to the one repository.** The child asks Hugging Face what the token is (`whoami`, `auth.accessToken`), once per run: the runner keeps the verdict and sends `check_token: false` afterwards; a failed or timed-out call fails closed and is retried with back-off (30, 90, 270 s) up to 4 checks, then the run is refused. A missing or wrongly typed `global`, `scoped` or permission list is refused, never read as "none". The upload child also loses `SSL_CERT_DIR` and the proxy variables. It must be `fineGrained`, with no global permissions and exactly one scope that grants anything: the model `HF_REPO` (spelled exactly as whoami shows it; another case is refused), plus at most the empty user scope Hugging Face adds to every fine-grained token, with `repo.write` and nothing beyond `repo.access.read`, `repo.content.read` and `repo.write`. A classic token (`write`, `admin`, `read`) gives `failed: refused: only a fine-grained token is accepted on a rented machine`; a fine-grained one that is broader (a global permission, two scopes, an organization scope, extra permissions) or scoped to another repository gives `failed: refused: the token is not limited to this one repository`. On the owner's PC that token is the Windows user variable **`HF_TOKEN_LORAS`**: it is the one the app side passes to a rented machine as `HF_WRITE_TOKEN`. The classic `HF_TOKEN` never goes on a rented machine. To test on the owner's own PC with a classic token, start the runner with `RUNNER_UPLOAD_ALLOW_CLASSIC=1`. **Do not set it on a rented machine**: that is the one place a classic token must never go (the training process runs as root and the token is in RunPod's records). The runner now enforces it: with `RUNNER_UPLOAD_ALLOW_CLASSIC=1` and `RUNPOD_POD_ID` both set it exits 2 at start.
- **Only a private repository.** The child reads `model_info(repo).private` and refuses a repository that is not private: `failed: refused: the repository is not private`. `RUNNER_UPLOAD_ALLOW_PUBLIC=1`, set at machine creation, is the only way around it.
- **A fixed address.** The token goes to `https://huggingface.co` and nowhere else: the address is a constant in `hf_upload.py`, not read from `HF_ENDPOINT` (the runner removes `HF_ENDPOINT`, `REQUESTS_CA_BUNDLE`, `CURL_CA_BUNDLE` and `SSL_CERT_FILE` from the child's environment, and the child drops them again). The tests' override is a module value that only test code can set in process; no environment variable, request or stdin field reaches it.
- **The hub's checksum.** A checkpoint counts as uploaded only when the hub reports the same size and an LFS sha256 equal to the runner's own; a checkpoint with no checksum from the hub is `failed: no checksum from the hub`.

Both settings are read once at start and removed from the environment, like the tokens, so nothing that talks to the runner afterwards can change them.

**When a checkpoint counts as done.** A `.safetensors` is sent only when it is whole: its size is stable between passes and its header agrees with its size (8 + header length + the end of the last tensor). A file still being written, or cut off by the time limit or a crash, waits, and in the last pass is reported `failed: incomplete file` with no link. The size caps are checked again on every pass and before the upload, with the file's current size. `upload.state` is `done` only when the final checkpoint `<job>.safetensors` is uploaded and verified; otherwise it is `failed` and the error says what is missing (never written, not uploaded). A trainer that fails to start ends the upload `failed` at once.

**What a leaked write token can do.** A classic write token is the owner's whole account: whoever has it can push, overwrite and delete files in every model, dataset and Space of that account, create repositories, and switch a private repository to public, so private LoRAs and anything else could be published or destroyed. **Use a fine-grained token with write access to `sprite-studio-loras` only**: then a leak costs that one repository. The `selftest` command prints the token's role (`write` is a classic token, `fineGrained` is a fine-grained one) so you can see which kind you have.

**What is not closed.** The training process runs as root, so it can read `/proc/1/environ`, where RunPod put `HF_WRITE_TOKEN` at boot. Hiding it would need a non-root training user, which is not cheap on the ostris image and is not done here. What limits the risk: the config allowlist fixes the model, adapter and paths, and the training process never gets the token in its own environment (a test checks it). Also, RunPod keeps a pod's environment in its records and returns it in its pod API, so anyone with the RunPod key can read the token while the pod exists; a RunPod secret that you create in the RunPod console and the pod references would avoid that, at the cost of setting it up by hand.

**The one-file round trip (no machine needed).** `python trainer/cloud/hf_upload.py selftest`, with one JSON line on stdin `{"token", "read_token", "repo_name"}`, prints the token's owner and role, checks that the repository is private (it stops before writing anything if not), creates a throwaway branch `selftest-<12 hex>`, uploads one 64-character file `_selftest/<12 hex>.txt` to it, reads it back with the read token (and shows that a download without a token is refused), deletes the branch and checks that it is gone and that `main` did not move. Nothing is ever written to `main`.

### Before the first rented machine

This trainer image has not run on RunPod yet (a rehearsal on 10 Nov 2026 proved the v2 self-delete with a pod-scoped key on a CPU machine, using a different program), and the app cannot yet end its own pods. Until it can, do this in order:

1. **Test on a cheap CPU machine first.** Rent the smallest CPU pod, run `shutdown.py` there once in each mode and see the pod disappear, both through the REST v2 call and through `runpodctl` (ss-m10t10). The v2 delete is proved by the rehearsal; `runpodctl` and stop mode are not.
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
