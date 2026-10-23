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
