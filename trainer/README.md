# Trainer: train a style from your game's own assets

Sprite Studio can teach the image model (Z-Image Turbo) the look of *your* game. You pick 15-60 of your sprites,
animations, 8-direction sets and tiles in the **Styles** tab, the app turns them into a training dataset, and a
style LoRA trains on your GPU with [ostris ai-toolkit](https://github.com/ostris/ai-toolkit). The finished LoRA is
copied into ComfyUI's LoRA folder and added as a new style.

```
browser ── /api/training/* ──> webapp (routes_training.py, training.py)
                                 │  writes webapp/data/train/{datasets,configs}/<job>
                                 │  POST /jobs, GET /jobs/<job> ...
                                 v
                               trainer service: runner.py (port 8675) ── python /app/ai-toolkit/run.py <config>
                                 │  writes webapp/data/train/output/<job>/  (checkpoints, samples/)
                                 v
                               webapp copies the checkpoint you pick to ComfyUI models/loras/sprite-studio/
```

`runner.py` is a tiny standard-library HTTP server that runs inside the official `ostris/aitoolkit` image. It starts
ai-toolkit with the config the app wrote, parses its progress bar, lists checkpoints and samples, and can cancel the
run. The app pauses its own ComfyUI job queue while training runs and asks ComfyUI to unload its models first, so
both fit on one GPU.

## Start the trainer

From the repo folder:

```
docker compose --profile train up -d trainer
```

It belongs to the `train` profile, so a plain `docker compose up -d` doesn't start it. Stop it when you're done
training to free its memory: `docker compose stop trainer`.

### First run downloads

| What | Size | Where it's kept |
|---|---|---|
| `ostris/aitoolkit:0.13.23` Docker image | ~6.6 GB | Docker |
| `Tongyi-MAI/Z-Image-Turbo` (diffusers format: transformer 24.6 GB, text encoder 8 GB, VAE 0.2 GB) | ~33 GB | `hf_cache` volume |
| `ostris/zimage_turbo_training_adapter` v2 | ~0.2 GB | `hf_cache` volume |

The model files download the first time you start training (the panel shows "downloading the base model"), then
later runs reuse them. ComfyUI's own Z-Image files can't be reused: ai-toolkit needs the diffusers-format repository,
while ComfyUI keeps single-file checkpoints.


### Keeping the downloads off drive C:

- **Base model files (~33 GB):** set a folder on another drive in the repo's `.env`, then start the trainer:

  ```
  TRAINER_CACHE_DIR=G:/sprite-studio/hf_cache
  ```

  Without it they go to the `hf_cache` Docker volume. Loading them from a Windows folder is a little slower at the
  start of each run (a minute or two), then training speed is the same.
- **The trainer image (~6.6 GB)** and all other Docker images and volumes live in Docker Desktop's disk image, on C:
  by default. To move everything (including ComfyUI's models), use Docker Desktop → Settings → Resources → Advanced →
  Disk image location, pick a folder on another drive, and Apply (Docker restarts and moves the data).

## Memory

- **GPU**: rank 32 at 1024 px with the 8-bit (qfloat8) transformer and text encoder needs roughly 17-22 GB of VRAM.
  A 24 GB card (RTX 3090/4090) works. The app calls ComfyUI's `/free` and holds its job queue while training, so
  ComfyUI's models don't take VRAM. Jobs you queue in the meantime wait and run when training ends.
- **RAM**: loading and quantizing the model briefly reads it into system memory (about 24 GB at the peak). Docker
  Desktop on Windows runs in WSL2, which by default gets only half of the PC's RAM. Give it more in
  `%UserProfile%\.wslconfig`:

  ```ini
  [wsl2]
  memory=24GB
  swap=16GB
  ```

  Then run `wsl --shutdown` and restart Docker Desktop. With too little memory the run dies with "Killed" or exit
  code 137 in the log.

## How long it takes

On an RTX 4090 a step at 1024 px takes about 1-1.5 s:

| Images | Steps (auto: 50 per image, 1500-3000) | Training | Total with loading + samples |
|---|---|---|---|
| 15-30 | 1500 | 25-40 min | ~30-45 min |
| 40 | 2000 | 35-50 min | ~40-55 min |
| 60+ | 3000 | 50-75 min | ~55-80 min |

Add the one-time download on the first run. A checkpoint and 4 sample images are saved every 250 steps, so you can
compare the steps and keep the best one, even before training ends.

## The dataset the app makes

ai-toolkit never upscales and shrinks images into its size buckets with bicubic filtering, which would blur pixel
art. So the app hands it images that are already exactly 1024 x 1024:

1. Up to 2-3 distinct frames per animation (near-duplicate frames are skipped). An 8-direction set gives its 8
   views; tiles become 4x4 boards.
2. Native 1x pixels (an imported 4x export is shrunk back to 1x), transparent borders trimmed.
3. Transparency flattened onto white.
4. Each picture centred on a square canvas measured in art pixels (64, 128, 256 ... large enough for the biggest
   asset), then scaled up with nearest-neighbour by **one** integer factor for the whole dataset: 64 px canvas x16,
   128 px x8, and so on. Every art pixel becomes a clean k x k block.
5. Captions describe content only, never the look, starting with a made-up trigger word:
   `hntdstyl, character: a gaunt old grave digger in a tattered brown coat, side view facing right, full body, plain white background`.
   Words like "pixel art", "16-bit" or "outline" are removed, so the LoRA learns the style itself.

Training settings: resolution 1024 only, no flips, LoRA rank 32 (or 16), learning rate 1e-4, AdamW 8-bit,
Z-Image Turbo with the v2 training adapter (keeps the distilled model's 8-step speed). The full config is
`webapp/data/train/configs/<job>.yaml`.

## Using the result

In the panel, pick a checkpoint and press **Use this checkpoint**. The app copies it to
`models/loras/sprite-studio/<name>_zit.safetensors` (ComfyUI's LoRA folder), registers it as a LoRA with its trigger
word and creates a style that uses it. In ComfyUI directly, load it with **LoraLoaderModelOnly** (strength 0.8-1.0)
and put the trigger word in the prompt.

## Training elsewhere (the kit)

When the trainer isn't running, the panel offers **Download training kit (.zip)**: the dataset, a `config.yaml` with
paths relative to the kit folder, and a README. On any PC with an NVIDIA GPU and Docker:

```
cd <unzipped kit folder>
docker run --rm -it --gpus all --shm-size 8g -v "$PWD:/workspace/kit" -v aitk_hf_cache:/root/.cache/huggingface -w /workspace/kit ostris/aitoolkit:0.13.23 python /app/ai-toolkit/run.py config.yaml
```

With a local ai-toolkit install, run `python <ai-toolkit>/run.py config.yaml` from the kit folder. In the ai-toolkit
web UI (also on cloud GPUs such as the RunPod ai-toolkit template), make a dataset from the `dataset/` folder and copy
the settings from `config.yaml`. Results land in `output/<job>/`; copy the `.safetensors` you like into ComfyUI's
`models/loras/`.

## Troubleshooting

- **The panel says the trainer is offline**: `docker compose --profile train up -d trainer`, then
  `docker compose logs -f trainer` to see it listening on port 8675.
- **Training failed**: the panel shows the log tail. The full log is `webapp/data/train/output/<job>/runner.log`.
  **Resume** continues from the last saved checkpoint (ai-toolkit picks it up from the output folder).
- **Out of memory** (CUDA): close other GPU programs; make sure ComfyUI isn't running a job; try rank 16.
- **Killed / exit code 137**: not enough RAM for WSL2, see [Memory](#memory).

## runner.py API

| Route | |
|---|---|
| `GET /health` | `{ok, busy, job, version}` |
| `POST /jobs` `{name, config_path, output_dir?, total_steps?}` | starts `RUN_CMD <config_path>`; one job at a time (409 while busy) |
| `GET /jobs/<name>` | `{status: queued\|running\|done\|failed\|cancelled, phase, step, total_steps, it_per_s, eta_s, loss, log_tail, checkpoints, samples, error, ...}` |
| `POST /jobs/<name>/cancel` | terminates the process group (status turns `cancelled` once it has exited) |
| `GET /files/<name>/<path>` | a checkpoint or sample from the job's output folder |

Environment: `RUNNER_HOST` / `RUNNER_PORT` (0.0.0.0 / 8675), `RUN_CMD` (`python /app/ai-toolkit/run.py`; shell
words or a JSON array), `RUN_CWD` (`/app/ai-toolkit`), `CONFIG_ROOT` (`/workspace/configs`: configs must live
there), `OUTPUT_ROOT` (`/workspace/output`). Tests run it with a fake trainer: `tests/test_training.py`.
