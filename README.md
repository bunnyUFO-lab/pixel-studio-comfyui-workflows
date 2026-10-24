# Pixel Studio ComfyUI workflows

ComfyUI workflows and custom nodes for Pixel Studio, with a Docker setup that runs ComfyUI and downloads the models
each workflow needs. They also work on their own, from ComfyUI's UI or its API.

Pixel Studio will have a public release later, with the option to use this repository to set up ComfyUI, self-host
these workflows and connect Pixel Studio to them.

## Workflows

### Pixel art

| Workflow | What it does |
|---|---|
| [`pixel-art/image-to-pixel`](workflows/pixel-art/image-to-pixel.json) | Turns any image into true pixel art at its native size, with a limited palette and clean transparency. |
| [`pixel-art/prompt-to-pixel`](workflows/pixel-art/prompt-to-pixel.json) | Makes a pixel-art sprite from a text prompt. |
| [`pixel-art/prompt-to-pixel-krea2`](workflows/pixel-art/prompt-to-pixel-krea2.json) | The same with the Krea 2 model (experimental). |
| [`pixel-art/image-to-directions`](workflows/pixel-art/image-to-directions.json) | Turns one character image into an 8-direction sprite set. |
| [`pixel-art/sprite-edit`](workflows/pixel-art/sprite-edit.json) | Edits a sprite (or animation frames) from a prompt, keeping its size, grid and palette. `sprite-edit-ref` / `sprite-edit-refs` also take reference images. |
| [`pixel-art/sprite-from-prompt`](workflows/pixel-art/sprite-from-prompt.json) | Animates a sprite from a motion prompt into a looping sprite sheet. |
| [`pixel-art/sprite-from-video`](workflows/pixel-art/sprite-from-video.json) | Animates a sprite by copying the motion of a video. |
| [`pixel-art/sprite-from-prompt-ti2v5b`](workflows/pixel-art/sprite-from-prompt-ti2v5b.json) | Fast: animates a sprite from a motion prompt with Wan 2.2 TI2V-5B (start frame only). |
| [`pixel-art/sprite-from-prompt-vace`](workflows/pixel-art/sprite-from-prompt-vace.json) | Fast: animates a sprite from a motion prompt with Wan 2.1 VACE 1.3B; first = last frame, so it loops. |
| [`pixel-art/sprite-from-video-vace`](workflows/pixel-art/sprite-from-video-vace.json) | Fast: animates a sprite from the pose skeletons of a video with Wan 2.1 VACE 1.3B. |
| [`pixel-art/sprite-inbetween`](workflows/pixel-art/sprite-inbetween.json) | Draws the frames between key frames of an animation. |
| [`pixel-art/prompt-to-tile`](workflows/pixel-art/prompt-to-tile.json) | Makes a seamless terrain tile from a prompt. |
| [`pixel-art/prompt-to-setpiece`](workflows/pixel-art/prompt-to-setpiece.json) | Makes a large scenery object (a tree, a cliff, a tower) from a prompt. |
| [`pixel-art/prompt-to-scenery-layer`](workflows/pixel-art/prompt-to-scenery-layer.json) | Makes a repeating parallax background layer from a prompt. |

Every pixel-art workflow ends with a **Pixel Art Check** that reports whether the result is game-ready.

### Images, vectors and video

| Workflow | What it does |
|---|---|
| [`image/prompt-to-image`](workflows/image/prompt-to-image.json) | Makes a batch of concept drafts from a prompt. |
| [`image-edit/qwen-edit`](workflows/image-edit/qwen-edit.json) | Edits an image from a prompt; paint a mask to change only part of it. |
| [`image-to-svg/vtracer-svg`](workflows/image-to-svg/vtracer-svg.json) | Traces an image into an SVG. |
| [`image-to-svg/layered-svg`](workflows/image-to-svg/layered-svg.json) | Traces an image into an SVG with editable layers (flat colours, shadows, highlights, ink). |
| [`image-to-svg/layered-svg-ai`](workflows/image-to-svg/layered-svg-ai.json) | Redraws the image as flat art first, then makes the layered SVG. |
| [`image-to-svg/prompt-to-svg`](workflows/image-to-svg/prompt-to-svg.json) | Makes a layered SVG (logo, icon, sticker, game item) from a prompt. |
| [`image-to-video/wan22-remix`](workflows/image-to-video/wan22-remix.json) | Turns an image into a short video. |
| [`video-to-video/wan22-svi-extend`](workflows/video-to-video/wan22-svi-extend.json) | Makes a video longer, segment by segment, from prompts. |
| [`pose/extract-pose`](workflows/pose/extract-pose.json) | Finds the pose skeletons in a picture or clip. |

Most workflows need an NVIDIA GPU with 24 GB of memory (for example an RTX 4090).

## Quick start

With Docker and an NVIDIA GPU:

```bash
cp env/pixel.env .env       # pick which workflows to set up (see env/)
docker compose up -d        # builds the image and downloads the models on the first run
```

Then open <http://localhost:8188>; the workflows are in the sidebar under **Workflows**. See
[docs/setup.md](docs/setup.md) for the full setup, a [cloud GPU option](cloud/README.md) and the optional
[style trainer](trainer/README.md).

## License

[MIT](LICENSE). The models the workflows download have their own licences.
