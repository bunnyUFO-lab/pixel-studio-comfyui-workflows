# Setup

The repository runs ComfyUI in Docker with these workflows and their custom nodes installed. On each start it
downloads any models the chosen workflows still need.

## Windows

1. Install the latest **NVIDIA driver** and **[Docker Desktop](https://www.docker.com/products/docker-desktop/)**
   (WSL2 backend, the default).
2. From the repository folder:

   ```powershell
   cp env/pixel.env .env
   docker compose up -d
   docker compose logs -f      # wait for "To see the GUI go to"
   ```

3. Open <http://localhost:8188>.

The first run builds the image (about 10 minutes) and downloads the models, which can take a while.

## Linux or a cloud GPU

Any Linux machine with an NVIDIA GPU, Docker and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
works the same way: clone the repository, copy a preset to `.env` and run `docker compose up -d`. To run on
Modal's cloud GPUs instead, see [cloud/README.md](../cloud/README.md).

## Choosing workflows

A preset picks which workflows' models are downloaded:

| Preset | Workflows |
|---|---|
| `env/pixel.env` | Pixel art, drafts and image edits |
| `env/svg.env` | Image to SVG |
| `env/edit.env` | Image editing |
| `env/video.env` | Image to video and video extension |
| `.env.all` | Everything |

Copy one to `.env` and run `docker compose up -d`. Switching presets only adds downloads.

## API keys (optional)

Some models are hosted on Civitai or gated on Hugging Face. Copy `.secrets.env.example` to `.secrets.env` and add
a `CIVITAI_TOKEN` and/or `HF_TOKEN` (both free). Without them those downloads are skipped and ComfyUI still starts.

## Storage

Set `COMFY_OUTPUT_DIR` and `COMFY_INPUT_DIR` in `.env` to keep inputs and outputs on another drive (examples in
`.env.all`). Models are kept in a Docker volume.

## Everyday commands

| Task | Command |
|---|---|
| Stop / start | `docker compose stop` / `docker compose up -d` |
| Logs | `docker compose logs -f comfyui` |
| Check which models are downloaded | `docker compose run --rm comfyui models` |
| Update after pulling changes | `docker compose up -d --build` |
