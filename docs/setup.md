# Setup

The repository runs ComfyUI in Docker with these workflows and their custom nodes installed. On each start it
downloads any models the chosen workflows still need.

## Windows

1. Install the latest **NVIDIA driver** and **[Docker Desktop](https://www.docker.com/products/docker-desktop/)**
   (WSL2 backend, the default) and **[Python 3.9+](https://www.python.org/downloads/)** (`winget install Python.Python.3.12`).
2. In PowerShell, from the repository folder (or from anywhere to clone it into `~\pixel-studio-comfy` first):

   ```powershell
   powershell -ExecutionPolicy Bypass -File install.ps1
   ```

   It checks Docker and the GPU, writes `.env` and `.secrets.env`, starts ComfyUI, runs `doctor`, and adds
   `bin\` to your user PATH (open a new terminal to use `pixel-studio-comfy`). Options go after the script name,
   for example `--preset svg`, `--models pixel-art/image-to-pixel`, `--port 8190`, `--lan`, `--yes`, `-NoPath`
   (see [Installer options](#installer-options)).
3. Open <http://localhost:8188>.

The first run builds the image (about 10 minutes) and downloads the models, which can take a while.

## Linux or a cloud GPU

Any Linux machine with an NVIDIA GPU, Docker, the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
and Python 3.9+ works the same way:

```sh
sh install.sh                     # in a checkout; or copy install.sh anywhere: it clones into ~/pixel-studio-comfy
```

It does the same as the Windows installer and installs the command as `~/.local/bin/pixel-studio-comfy`. To run on
Modal's cloud GPUs instead, see [cloud/README.md](../cloud/README.md).

## Installer options

`install.sh` and `install.ps1` take `--dir PATH` (`-Dir` on Windows; default: this checkout, else `~/pixel-studio-comfy`),
`--ref BRANCH` (clone ref, default `main`) and `--no-path` (`-NoPath`). Everything else goes to `pixel-studio-comfy setup`:

| Option | Effect |
|---|---|
| `--preset NAME` | `pixel` (default), `svg`, `edit`, `video`, `all`, or `none` (see [Choosing workflows](#choosing-workflows)) |
| `--models a,b,c` | workflow names to download models for; wins over `--preset` |
| `--port N` | host port (default 8188) |
| `--lan` | listen on the network (`BIND_ADDRESS=0.0.0.0`). New installs listen on `127.0.0.1` only; there is no authentication yet |
| `--project-name NAME` | Compose project name, to run a second stack beside another |
| `--output-dir PATH` | where outputs go (`COMFY_OUTPUT_DIR`) |
| `--civitai-token`, `--hf-token` | fill an empty token in `.secrets.env` (else the `CIVITAI_TOKEN` / `HF_TOKEN` environment variables; interactive runs ask) |
| `--gateway` | put the [gateway](#gateway) in front of ComfyUI (new `.env` only) |
| `--no-up` | write the files, do not start or run `doctor` |
| `-y`, `--yes` | never prompt |

An existing `.env` is never edited: `setup` keeps it, says which options it ignored, and leaves a running stack running.

## Manual setup

Without the installer, from the repository folder:

```powershell
cp env/pixel.env .env
docker compose up -d
docker compose logs -f      # wait for "To see the GUI go to"
```

`bin/pixel-studio-comfy` works without installing anything (`python -m psc <command>` too).

## Choosing workflows

A preset picks which workflows' models are downloaded:

| Preset | Workflows |
|---|---|
| `env/pixel.env` | Pixel art, drafts and image edits |
| `env/svg.env` | Image to SVG |
| `env/edit.env` | Image editing |
| `env/video.env` | Image to video and video extension |
| `.env.all` | Everything |

`pixel-studio-comfy models presets` lists them. The installer writes the chosen one to `.env`; by hand, copy one to `.env`
and run `docker compose up -d`. Switching presets only adds downloads. `pixel-studio-comfy models pull NAME` downloads a
preset or workflow now, without editing `.env`.

## API keys (optional)

Some models are hosted on Civitai or gated on Hugging Face. The installer creates `.secrets.env`; by hand, copy `.secrets.env.example` to it and add
a `CIVITAI_TOKEN` and/or `HF_TOKEN` (both free). Without them those downloads are skipped and ComfyUI still starts.

## Storage

Set `COMFY_OUTPUT_DIR` and `COMFY_INPUT_DIR` in `.env` to keep inputs and outputs on another drive (examples in
`.env.all`). Models are kept in a Docker volume.

## Gateway

By default ComfyUI has no authentication, so new installs listen on `127.0.0.1` only. To reach it from other machines
(Tailscale, LAN) with a password, turn on the opt-in gateway, a Caddy proxy that requires `Authorization: Bearer <token>`:

```sh
pixel-studio-comfy setup --gateway --lan     # new install: writes PSC_GATEWAY=1, COMPOSE_FILE and a random PSC_TOKEN
pixel-studio-comfy token show                # print the token
pixel-studio-comfy token rotate              # new token, gateway restarted
```

With the gateway on, only `PORT` is published (by the gateway, not ComfyUI), a missing or wrong token gets **401**, and with
a valid token only the paths Pixel Studio uses pass (`/prompt`, `/history`, `/view`, `/upload/image`, `/object_info`, `/ws`,
`/queue`, `/interrupt`, `/system_stats`, `/pixelstudio/*`, each also under `/api`); everything else, including the ComfyUI
page, `/manager` and `/userdata`, is **404**. In Pixel Studio add the source with the header, which `setup --gateway` prints:

```json
[{"name": "pc", "url": "http://<this machine>:8188", "headers": {"Authorization": "Bearer <token>"}}]
```

`status` and `doctor` send the token themselves, and `doctor` checks that the gateway answers 401 without it and 404 for
`/manager`. The gateway serves plain HTTP: use a trusted network or Tailscale.

To turn it on for an existing install, add `PSC_GATEWAY=1` to `.env`, run `pixel-studio-comfy token show` once, then
`pixel-studio-comfy up`. `setup --gateway` also writes `COMPOSE_FILE=docker-compose.yml;docker/gateway.compose.yml` and
`COMPOSE_PATH_SEPARATOR=;` to `.env`, so plain `docker compose` uses the override too; on an existing install add those two lines
yourself, or add `-f docker-compose.yml -f docker/gateway.compose.yml` to every `docker compose` call. Without the override,
`docker compose up -d` would publish ComfyUI's own port and leave the gateway orphaned. The token must be 32 or more of
`A-Z a-z 0-9 _ -` (`token rotate` makes one); the gateway refuses to start with anything else. `status --url` and `doctor
--url` send the token only to `127.0.0.1` / `localhost`. To turn the gateway off, run `pixel-studio-comfy down`, remove
`PSC_GATEWAY`, `COMPOSE_FILE` and `COMPOSE_PATH_SEPARATOR` from `.env` and run `up` again (the ComfyUI page is then available on the host as before).

## Everyday commands

`bin/pixel-studio-comfy` (Linux, Git Bash) or `bin\pixel-studio-comfy.cmd` (Windows) wraps the Docker commands; once the
installer has put it on PATH, use `pixel-studio-comfy`. `python -m psc <command>` from the repository folder does the same.
`down` never removes the model volumes.

With the gateway on, use the CLI, or add `-f docker-compose.yml -f docker/gateway.compose.yml` to every `docker compose`
call (not needed when `.env` has the `COMPOSE_FILE` line from `setup --gateway`); a bare `docker compose` would otherwise
drop the gateway.

| Task | CLI | Docker |
|---|---|---|
| Start and wait for ComfyUI | `pixel-studio-comfy up` | `docker compose up -d` |
| Stop | `pixel-studio-comfy down` | `docker compose down` |
| Logs | `pixel-studio-comfy logs -f` | `docker compose logs -f comfyui` |
| Container, GPU memory and queue | `pixel-studio-comfy status` | |
| Check Python, Docker, GPU, disk and models | `pixel-studio-comfy doctor` | |
| Check which models are downloaded | `pixel-studio-comfy models list` | `docker compose run --rm comfyui models` |
| Download models now | `pixel-studio-comfy models pull pixel` | `docker compose run --rm comfyui models download pixel-art/image-to-pixel` |
| Update after pulling changes | `pixel-studio-comfy update` | `git pull --ff-only` then `docker compose up -d --build` |

## CLI reference

Global options go before the command: `--dir PATH` (the stack folder, env `PSC_DIR`) and `--url URL` (ComfyUI address,
env `PSC_URL`; default `http://127.0.0.1:<PORT from .env, else 8188>`).

| Command | Does |
|---|---|
| `setup [options]` | checks, writes `.env` and `.secrets.env` (never overwrites), `up`, `doctor` (see [Installer options](#installer-options)) |
| `doctor` | one block of checks (Python, Docker, GPU, disk, container, API, models); exit 1 on any FAIL |
| `status` | container, ComfyUI version, GPU memory, queue |
| `up [--wait N]` | `docker compose up -d`, then wait for the API (default 120 s, 0 = do not wait) |
| `down` | `docker compose down` |
| `logs [-f] [--tail N]` | ComfyUI container log |
| `models list [PRESET\|WORKFLOW ...]` | required models and whether they are present (default: the `.env` preset) |
| `models pull PRESET\|WORKFLOW ...` | download missing models (`models get` is the same); does not edit `.env` |
| `models presets` | the preset table |
| `token show` / `token rotate` | print or replace the gateway token (see [Gateway](#gateway)) |
| `update [--no-up]` | `git pull --ff-only` (refuses on local changes or without an upstream), then `up` |
| `cloud modal deploy [options]` / `cloud modal snippet` | deploy the [Modal job API](../cloud/README.md) and print the Pixel Studio `COMFY_SOURCES` entry; `--dry-run` shows what it would do; the live app `sprite-studio-comfy` needs `--live` |
