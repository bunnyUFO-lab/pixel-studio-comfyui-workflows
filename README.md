# Pixel Studio ComfyUI workflows

The ComfyUI workflows, custom nodes and Docker setup that [Pixel Studio](https://github.com/bunnyUFO-lab/pixel-studio)
runs its jobs on. Run them on any NVIDIA PC or cloud GPU, and point Pixel Studio at the server as a compute source.
They also work on their own, from ComfyUI's UI or its API.

> Split out of `bunnyUFO-lab/comfyui_wofkflows` on 15 October 2026 (at `9ce7357`). That repository is unchanged;
> this one starts with fresh history.

My custom ComfyUI workflows, plus a Docker setup that runs ComfyUI with them on any NVIDIA PC or cloud GPU.

| Workflow (`MODELS` name) | Type | Hardware | Custom nodes | Download | Notes |
|---|---|---|---|---|---|
| [`image-to-video/wan22-remix`](workflows/image-to-video/wan22-remix.json) | Image → video | RTX 4090 (24 GB) | VideoHelperSuite | ~36 GB | Wan 2.2 Remix v3.0, 8 steps |
| [`video-to-video/wan22-svi-extend`](workflows/video-to-video/wan22-svi-extend.json) | Video → longer video | RTX 4090 (24 GB) | VideoHelperSuite + repo nodes | +2.5 GB (SVI LoRAs) | Adds N segments (~4.75 s each) from prompts; saves full + new-only MP4s |
| [`image-edit/qwen-edit`](workflows/image-edit/qwen-edit.json) | Image → image (optional inpaint) | RTX 4090 (24 GB) | repo nodes | ~17.3 GB | Qwen-Image-2.1 prompt edits; paint a mask to change only that area |
| [`image-to-svg/vtracer-svg`](workflows/image-to-svg/vtracer-svg.json) | Image → SVG | Any (GPU speeds up the upscale) | ComfyUI-ToSVG | 18 MB | AI upscale + VTracer tracing for logos, flat art, stylized characters |
| [`image-to-svg/layered-svg`](workflows/image-to-svg/layered-svg.json) | Image → layered SVG | Any (GPU speeds up the upscale) | ComfyUI-ToSVG + repo nodes | 18 MB | Editable layers (flats, shadows, highlights, ink, details) with far fewer shapes; optional palette |
| [`image-to-svg/layered-svg-ai`](workflows/image-to-svg/layered-svg-ai.json) | Image → AI repaint → layered SVG | RTX 4090 (24 GB) | ComfyUI-ToSVG + repo nodes | ~17.3 GB | Qwen-Image-2.1 redraws the image flat first (Vector Art Prompt template: style, colours, detail); fewer shapes, less faithful. Non-commercial model licence. |
| [`image-to-svg/prompt-to-svg`](workflows/image-to-svg/prompt-to-svg.json) | Prompt → layered SVG | RTX 4090 (24 GB) | ComfyUI-ToSVG + repo nodes | ~21 GB (Z-Image) | Type a subject; the template asks for vector-friendly art (logo, icon, sticker, cartoon, line art, game item); Z-Image Turbo draws it, Layered Vectorize traces it |
| [`image/prompt-to-image`](workflows/image/prompt-to-image.json) | Prompt → N draft images | RTX 4090 (24 GB) | repo nodes | ~21 GB (Z-Image) | Concept drafts in one batch (pixel-styled or plain), via the Pixel Art Prompt template |
| [`pixel-art/sprite-edit`](workflows/pixel-art/sprite-edit.json) | Sprite (or frames) + prompt → sprite | RTX 4090 (24 GB) | repo nodes | ~17.3 GB (Qwen) | Qwen edit on an integer upscale, snapped back to the same grid/canvas; palette locked or + new colours |
| [`pixel-art/prompt-to-tile`](workflows/pixel-art/prompt-to-tile.json) | Prompt → seamless tile | RTX 4090 (24 GB) | repo nodes | ~21 GB (Z-Image) | Terrain texture → exact grid → seamless tile (min-cut wrap seams, own pixels only); palette can match a tileset |
| [`pixel-art/image-to-pixel`](workflows/pixel-art/image-to-pixel.json) | Image → pixel art | Any (GPU speeds up BiRefNet) | repo nodes | 444 MB | True pixel art at native size: `pixels` = subject height, limited/preset palette, hard alpha, PixelOE |
| [`pixel-art/prompt-to-pixel`](workflows/pixel-art/prompt-to-pixel.json) | Prompt → pixel art | RTX 4090 (24 GB) | repo nodes | ~21 GB | Type a subject; Pixel Art Prompt adds a game-art style, detail for the sprite size, colours; Z-Image Turbo + pixel-art LoRA (Apache-2.0), ~20 s |
| [`pixel-art/sprite-from-prompt`](workflows/pixel-art/sprite-from-prompt.json) | Pixel sprite + prompt → sprite sheet | RTX 4090 (24 GB) | VideoHelperSuite + repo nodes | Remix + 2.5 GB | Looping animation at the sprite's exact pixel size + palette; PNG sheet, Aseprite JSON, WebP |
| [`pixel-art/sprite-from-video`](workflows/pixel-art/sprite-from-video.json) | Pixel sprite + video → sprite sheet | RTX 4090 (24 GB) | VideoHelperSuite + repo nodes | ~18.6 GB + Wan encoder/VAE | Wan Animate 2 copies a driving video's motion; same pixel size + palette |
| [`pixel-art/sprite-from-video-scail2`](workflows/pixel-art/sprite-from-video-scail2.json) | Pixel sprite + video → sprite sheet | RTX 4090 (24 GB) | VideoHelperSuite + repo nodes | ~19.6 GB (SCAIL 2 int8 + DPO LoRA + SAM3.1) + Wan encoder/VAE | SCAIL 2 (Apache-2.0/MIT) arm of sprite-from-video: same inputs/outputs; SAM3 masks the person in the video, the sprite's own alpha masks the reference; ~2.5 min, peak ~17 GB |
| [`pixel-art/sprite-from-video-h3`](workflows/pixel-art/sprite-from-video-h3.json) | Pixel sprite (+ optional second view) + video → sprite sheet | RTX 4090 (24 GB), tight | VideoHelperSuite + repo nodes | ~35 GB (H3 Ref2VA w6a8 + NVFP4 Qwen3-VL 32B + VAEs) | MiniMax H3 arm of sprite-from-video: sprite = `<Picture 1>`, video = `<Video 1>`, six-section prompt; audio dropped. **Licence excludes the US/EU/UK/South Korea without a MiniMax commercial licence** |
| [`pixel-art/image-to-directions`](workflows/pixel-art/image-to-directions.json) | Character → 8-direction sprites | RTX 4090 (24 GB) | repo nodes | ~17.8 GB | Qwen-Image-2.1 draws 5 views (editable prompts), pixelized together, mirrored to 8 (or all 8 drawn from Sprite Studio) |

## Repo layout

```
docker-compose.yml        # the only compose file
docker/Dockerfile         # CUDA 12.8 + PyTorch + ComfyUI (pinned tag) + built-in Manager
docker/custom_nodes.txt   # custom nodes baked into the image, pinned to commits
docker/models.py          # lists/downloads the models workflows need (reads the workflow JSON)
custom_nodes/             # this repo's own small ComfyUI nodes (mounted into the container)
workflows/                # mounted as ComfyUI's workflow library; saves land here -> commit them
tools/                    # workflow generator scripts (build_*.py) + shared helpers (comfy_workflow.py)
input/ output/            # mounted into ComfyUI, gitignored
trainer/                  # style-LoRA trainer runner (optional compose profile `train`)
env/                      # presets: video.env, svg.env, edit.env, pixel.env (which workflows' models to download)
.env.all                  # preset: every workflow
.env                      # active preset (e.g. cp env/svg.env .env), gitignored
.secrets.env              # API keys (from .secrets.env.example), gitignored, never touched by presets
```

How it fits together:
- **One image, one compose file.** Workflows differ only in which models and custom nodes they need, so separate Dockerfiles would just duplicate a ~10 GB image.
- **Each workflow lists its own models.** Every loader node in the workflow JSON carries `properties.models` (file name, URL, folder). This is ComfyUI's own format: the missing-models dialog uses it, and so does `docker/models.py`. There's no separate manifest to keep in sync.
- **A preset picks which workflows are ready on this machine.** `cp env/svg.env .env` (or `env/video.env`, `.env.all`) sets `MODELS`. On every start, the container downloads any missing models for those workflows, then launches ComfyUI. Models live in the `comfyui_models` Docker volume, never in the image.
- **Custom nodes are pinned in git.** A rebuild produces the same setup on any machine.
- **Workflows are bind-mounted** from this repo, so whatever you save in the UI shows up in `git status`.

## Setup (Windows + Docker Desktop)

One-time:
1. Install or update the **NVIDIA driver** and **[Docker Desktop](https://www.docker.com/products/docker-desktop/)**, with the WSL2 backend enabled (the default). No CUDA toolkit or NVIDIA container toolkit is needed on Windows.
2. Start Docker Desktop and check that containers can see the GPU:
   ```powershell
   docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
   ```
3. Optional: in `%UserProfile%\.wslconfig`, give WSL enough RAM for the 14 GB models (e.g. `[wsl2]` → `memory=48GB`), then run `wsl --shutdown`.

From the repo folder:
```powershell
cp env/video.env .env       # pick a preset: env/video.env, env/svg.env or .env.all
docker compose up -d        # first run: builds the image (~10 min), downloads models, starts ComfyUI
docker compose logs -f      # watch progress; ComfyUI is ready at "To see the GUI go to"
```
Open <http://localhost:8188>. Your workflows are in the left sidebar under **Workflows**.

### Presets and credentials

| File | In git | What it's for |
|---|---|---|
| `env/video.env` | yes | Image-to-video + video extension (~38.5 GB) |
| `env/edit.env` | yes | Image editing with Qwen-Image-2.1 (~17.3 GB) |
| `env/svg.env` | yes | The SVG workflows (18 MB upscaler) |
| `env/pixel.env` | yes | The pixel-art workflows (~95 GB: Z-Image, Wan Remix, Wan Animate 2, Qwen-Image-2.1, BiRefNet) |
| `.env.all` | yes | Every workflow (~36 GB today) |
| `.env` | no | Your active preset: `cp env/svg.env .env`, then `docker compose up -d` |
| `.secrets.env` | no | API keys. Create it once with `cp .secrets.env.example .secrets.env` and fill it in. Switching presets never touches it. |

Both API keys are free to create:
- `CIVITAI_TOKEN`: civitai.com → Account settings → API Keys. Needed for Civitai-hosted models.
- `HF_TOKEN`: huggingface.co/settings/tokens. Only needed for gated models.

`.secrets.env` is optional; without it, Civitai downloads are reported as skipped and ComfyUI still starts. Switching presets only adds downloads, because models already in the volume are kept. A preset can also override `COMFYUI_REF`, `PORT` and `COMFY_ARGS` (see the comments in each file).

### Where files are stored

Source code can stay on a small system drive; generated files and libraries can live elsewhere. Set these in `.env`
(examples in `.env.all`), then `docker compose up -d`:

| Setting | What goes there | Default |
|---|---|---|
| `COMFY_OUTPUT_DIR` / `COMFY_INPUT_DIR` | ComfyUI's output and input folders. Sprite Studio keeps its own copies, so when one of its jobs ends it deletes that job's files there (only the outputs the job saved and the `ss_…` images it uploaded; nothing else in those folders) | `./output`, `./input` |
| `TRAIN_DIR` | The LoRA trainer's datasets, configs and output. Must be the same folder as Pixel Studio's `<data>/train` | `./train` |
| `TRAINER_CACHE_DIR` | The LoRA trainer's ~33 GB of base model files | a Docker volume |

To move one of these folders, copy it to the new place, set the variable and recreate the containers. The
models live in Docker's own disk image: move that with Docker Desktop → Settings → Resources → Advanced → Disk image
location.

### Day to day

| Task | Command / action |
|---|---|
| Stop / start | `docker compose stop` / `docker compose up -d` |
| Logs | `docker compose logs -f comfyui` |
| What models do my workflows need, and are they downloaded? | `docker compose run --rm comfyui models` (or `models all`, or `models <workflow>`) |
| Download models without restarting | `docker compose run --rm comfyui models download <workflow>` |
| Save a workflow | Save in the UI → it lands in `workflows/` → `git commit` |
| Add a custom node | Try it via Manager, then add `<url> <commit>` to `docker/custom_nodes.txt` and run `docker compose up -d --build`. Nodes installed only through Manager are lost when the container is recreated. |
| Drop in a one-off model | `docker compose cp my_lora.safetensors comfyui:/opt/ComfyUI/models/loras/` |
| Update ComfyUI | Bump the `COMFYUI_REF` default in `docker-compose.yml` (or set it in `.env`), then `docker compose up -d --build` |
| Extra launch flags | `COMFY_ARGS=--reserve-vram 1` in `.env` |
| Models from Civitai | Set `CIVITAI_TOKEN` in `.secrets.env` (civitai.com → Account settings → API Keys). Missing keys are reported and skipped; ComfyUI still starts. |
| Shell inside the container | `docker compose exec comfyui bash` |

> **Git Bash users:** Git Bash rewrites arguments that look like Linux paths (`/opt/...`) into Windows paths before Docker sees them. Prefix such commands with `MSYS_NO_PATHCONV=1`, e.g. `MSYS_NO_PATHCONV=1 docker compose exec comfyui ls /opt/ComfyUI/models`, or use PowerShell.

### Adding a new workflow

1. Save it into `workflows/<category>/<name>.json`, from the UI or by copying the file in.
2. Make sure each model loader node has its download URL. ComfyUI's official templates already include them. Workflows from Civitai etc. usually don't, so add a `models` entry to the node's `properties`:
   ```json
   "properties": { "models": [{ "name": "file.safetensors", "url": "https://huggingface.co/.../resolve/main/file.safetensors", "directory": "loras" }] }
   ```
   `directory` is the folder under `ComfyUI/models` (`diffusion_models`, `text_encoders`, `vae`, `loras`, `checkpoints`, …). Then check that `docker compose run --rm comfyui models <name>` lists them all.
3. Add any custom nodes it needs to `docker/custom_nodes.txt`.
4. Add a row to the table at the top, add it to `MODELS` in the matching preset(s) (`.env.all` picks it up automatically), and run `docker compose up -d --build`.

### Repo custom nodes

`custom_nodes/comfyui_workflows_utils/` holds small nodes written for these workflows. It's mounted into the container, so edits only need `docker compose restart`.

| Node | What it does |
|---|---|
| **Layered Vectorize** | Turns an image into a layered SVG (flats, shadows, highlights, ink, details); see the Layered SVG section. |
| **Wan SVI 2.0 Pro Conditioning** | SVI conditioning for Wan 2.2 I2V: anchor latent + the previous clip's last latents + padding, with only the anchor marked known (the same idea as KJNodes' `WanImageToVideoSVIPro`, reimplemented). |
| **Segment Prompt** | Picks line N of a multiline prompt for loop iteration N (the last line repeats) and outputs seed + N. |
| **Resize for Wan** | Resizes a frame batch so the longest side is N, both sides multiples of 16. |
| **Video Anchor + Tail Frames** | Splits out the anchor frame (first or last) and the last N frames of a video. |
| **Join Video Segments** | Appends SVI segments: drops each anchor frame, cross-fades the overlap, optional colour match; outputs the full video and the new frames. |
| **Masked Edit Composite** | Puts an edited image back into the original inside a (grown, feathered) mask; with no mask it returns the full edit. |
| **Pixelize (true pixel art)** | Image(s) → native-resolution pixel art: exact grid (`pixels` = subject height, longest side, image height, cell size, body height per frame, or auto-detect an upscaled grid), PixelOE contrast-aware sampling, optional Kuwahara smoothing, limited palette (automatic fewest colours within `color_error`, N colours, PICO-8, GameBoy, DB32, Endesga-32, or hex list) shared across a batch, hard alpha, stray-pixel cleanup, optional outline. |
| **Prepare Pixel Sprite for Video** | Integer nearest upscale of a sprite onto a flat background far from its palette, with room for motion; outputs the image, its size, and a `PIXEL_SPEC` (factor, offsets, palette, background). |
| **Re-pixelate Frames** | Video frames + `PIXEL_SPEC` → native frames on exactly the sprite's grid, background keyed out; palette = the sprite's (optionally merged to N of its colours) or a preset; temporal stability + deflicker against colour popping; optional height match (one scale for the clip, from the first frame). |
| **Sprite Animation Preview** | Plays frames as a looping animation inside the node: fps, pixel-perfect zoom, checker/white/black/grey/transparent background, loop or ping-pong. |
| **Save Sprite Sheet** | Frames → sheet PNG + Aseprite JSON (frames, durations, `frameTags`) + animated WebP and GIF + preview (plays in the node); outputs `fps` for the preview; frame step, max frames, loop trim, columns, trim. |
| **Save Directional Views** | Views (front, front-right, right, back-right, back) → one PNG per direction + sheet + JSON, mirroring the right side to get the left (8 directions). |
| **Pixel Art Check (game-ready)** | PASS/FAIL report: hard alpha, native ×1, palette size, consistent height across frames (can be turned off for action clips), expected height, stray pixels. |
| **Pixel Art Prompt** | Subject + game-art style (clean sprite, 16-bit RPG, 8-bit, chibi, top-down, side-scroller, isometric, item/icon, monster, prop) + view + `pixels` (sets the detail level) + colours/palette + outline → a full prompt for simple, easily pixelated art. Also outputs `pixels` / `colors` / `palette` to drive Pixelize. |
| **Vector Art Prompt** | Generate or "redraw `<image1>`" prompt for flat, easily vectorized art: style (faithful flat redraw, flat vector, cartoon, logo, icon, sticker, line art, game item / UI art, stylized game character), colours/palette, detail, outlines, shading. Also outputs `colors` / `palette` to drive Layered Vectorize. |
| **SVG Stats (shapes, file size)** | Takes an SVG string and outputs a summary (shapes, colours, saved file size, canvas size; for Layered Vectorize output, base colours with shadow/highlight shades listed separately), plus `shapes` (INT) and `size_kb` (FLOAT). The size includes the workflow metadata that ComfyUI's Save SVG node embeds. |

> Why a Docker volume for models instead of a Windows folder: Docker on Windows runs inside WSL2, and reading big model files from `C:\` goes through a slow file-sharing bridge. Users report model loads several times slower. The `comfyui_models` volume lives on WSL2's native filesystem. Workflows, input and output are small, so they're fine as Windows bind mounts.

### New PC or rented cloud GPU

On any Linux box with an NVIDIA GPU, Docker, and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (most GPU clouds that give you a VM have these preinstalled):
```bash
git clone https://github.com/bunnyUFO/comfyui_wofkflows && cd comfyui_wofkflows
cp env/svg.env .env         # pick the preset you need on this box
cp .secrets.env.example .secrets.env   # only if you need Civitai/HF keys; then fill it in
docker compose up -d
```
Then reach it through an SSH tunnel: `ssh -L 8188:localhost:8188 user@host`. Port 8188 is bound to localhost on purpose.

Container-only hosts such as RunPod pods can't run compose. For those, publish the image to a registry (e.g. GHCR); that isn't set up yet.

---

## Pixel art (`pixel-art/*`)

Five workflows that make **true pixel art** for games: 1 art pixel = 1 image pixel, one uniform grid, a limited
palette shared by every frame or view, hard alpha (0 or 255), and a stable size and feet line. Every workflow ends
with **Pixel Art Check**, which prints PASS/FAIL. The `pixel-art` Claude skill (`.claude/skills/pixel-art/`) runs
them and iterates until the check passes.

| Workflow | Input → output | Time (RTX 4090) |
|---|---|---|
| `image-to-pixel` | any image → pixel art (BiRefNet cut-out, Pixelize) | ~3 s |
| `prompt-to-pixel` | prompt → Z-Image Turbo + pixel-art LoRA → pixel art | ~20 s |
| `image-to-directions` | character image → 5 Qwen views → 8 directions (left = mirrored right) | ~4 min |
| `sprite-from-prompt` | pixel sprite + motion prompt → looping sprite sheet | ~1 min (49 frames) |
| `sprite-from-video` | pixel sprite + driving video → sprite sheet | ~1–2 min (81 frames) |
| `sprite-from-video-scail2` | same, with SCAIL 2 instead of Wan Animate 2 | ~2.5 min (81 frames, warm) |
| `sprite-from-video-h3` | same, with MiniMax H3 Ref2VA (can take a second view) | ~53 min with 14 GB VRAM free and 10 GB WSL RAM (weights stream from disk every step); faster with the card to itself |

### Pixel size and palette
- `pixels` (default **64**) sets the size in art pixels. `measure` picks what it means: **subject height** (the
  non-transparent area), longest side, image height, cell size (source pixels per art pixel), **body height (per
  frame)** (each image's body, ignoring thin props like staffs; used for views drawn at different scales), or
  **auto (detect grid)**, which recovers already-pixelated art (even upscaled) exactly.
- `colors` = **0 (default): automatic, the fewest colours that keep the look**. `color_error` sets how far colours
  may drift: 12 is faithful, **20 (default)** simplified, 30+ very few. The error is measured at the 90th percentile,
  so small distinct areas such as a blade, skin or eyes keep their own colour. The swordsman sprite came out at
  6 colours. You can also set an exact number.
- `palette` = `auto`, `PICO-8`, `GameBoy`, `DB32`, `Endesga-32`, or a hex list (`#1a1c2c #5d275d ...`). With a preset,
  `colors` N keeps only its N best-fitting entries. Batches share one palette (`shared_palette`).
- `method` = **pixeloe** (default): [PixelOE](https://github.com/KohakuBlueleaf/PixelOE)'s contrast-aware
  downscale with outline expansion, which keeps thin dark outlines and highlights. `dominant`, `median`, `center`
  and `mean` are simpler alternatives.
- `smooth` (0.6 in image-to-pixel, 0 in prompt-to-pixel) applies edge-preserving smoothing first. It flattens
  painted texture and noise into the solid areas pixel art needs. Use 0 for art that is already flat.
- `cleanup` removes isolated stray pixels (distinct ones like eyes are kept).

### Clean edges
Smudged, half-background edge pixels used to turn into light specks and break the outline. Three steps prevent this:
- **Clean Cutout** (after background removal):
  - `drop_shadow` (on) removes a shadow or ground ellipse that the model drew under the subject. It looks for a light,
    low-saturation, flat patch at the very bottom; dark grey parts such as armoured boots are kept.
  - `enclosed_background` (off; opt-in) makes background seen through gaps (between arm and body) transparent. It's
    off by default because white eye whites on a white background look the same, so check the result.
- **Pixelize `edge_cleanup`** (on):
  - Before pixelizing, edge pixels take the nearest solid interior colour (defringe). No halo colours reach the
    palette or the border cells. PixelOE no longer composites on white.
  - The palette is built from flat pixels only, so anti-aliased in-between shades never become palette colours.
  - After pixelizing, 1-pixel spurs and notches on the silhouette are smoothed.
- **Pixelize `outline`** draws on the sprite's edge pixels, so the size is unchanged. It's only applied where the shape
  is at least 3 px thick, so blades and staffs keep their colour.

  | Mode | What it does |
  |---|---|
  | `auto` (default) | Completes the art's own dark outline if it has one; otherwise uses darker shades. |
  | `darker shade` | Selective outline: a darker palette colour of the fill behind each edge pixel. |
  | `darkest` | Solid black, or the darkest palette colour. |
  | `outside darkest` | An extra 1 px ring around the sprite. |
  | `none` | No outline. |

  Re-pixelate Frames has the same `outline` setting for animations (off by default; the sprite's own outline is kept).
- The prompt templates have a `background` option (white, light grey, green or magenta chroma key) and ask for no
  drop shadow, floor or ground ellipse.

Tested on 13 generated subjects: wizard, slime, potion, cat knight, dragon, pine tree, robot, skeleton, key and
swordsmen. All passed the check with 3–6 colours, clean closed outlines and no halo pixels.

### Prompt templates (`prompt-to-pixel`)
Type just the subject ("a swordsman in a red tunic holding a long steel sword") into **Pixel Art Prompt** and pick:
- **style**: clean game sprite, 16-bit RPG, 8-bit retro, chibi, top-down RPG, side-scroller, isometric, item / icon,
  monster / creature, prop / tile, or none;
- **view**;
- **pixels**: the detail level follows the sprite size (≤ 24 px minimal, ≤ 48 low, ≤ 96 medium), unless `detail` is set;
- **colors / palette**;
- **outline**.

The template adds flat colours, one shadow and one highlight tone, thick outlines, one subject with margins, a plain
background and the LoRA trigger, so the drawing is already simple enough to pixelate cleanly. The same `pixels`,
`colors` and `palette` drive Pixelize and the check. On the swordsman, the template styles gave 3–5 colours and
1–11 stray pixels, with a readable face and sword even at 24 px.

### Animations keep the sprite's exact pixels
Video models can't output 64 px sprites, so the sprite workflows:
1. scale the sprite up by an **integer** factor (e.g. ×6) onto a flat background colour far from its palette;
2. let Wan generate at that size;
3. bring every frame back down **on the same grid** and snap it to the **sprite's own palette**, so the pixel size
   can't change and no new colours appear. `sprite-from-video` also scales the whole clip once, so the character
   in the first frame has the sprite's height whatever the size of the person in the video (start the video in a
   neutral standing pose). Motion such as steps and crouches is kept.

**Playback preview.** The animation workflows end with a **Sprite Animation Preview** node.
- It plays the sprite-sheet frames in a loop at the sheet's `fps`.
- Settings: `scale` (nearest-neighbour zoom), `background` (checkerboard, white, black, grey or transparent) and
  `playback` (loop or ping-pong).
- It writes a temporary preview only, so changing a setting re-runs instantly without regenerating the video.
- The Save Sprite Sheet node also shows its animated WebP playing.
- `image-to-directions` has a turnaround preview that cycles through all 8 directions.

**Colours in animations (Re-pixelate Frames).** These settings can be changed after a run: only the cheap last
step re-runs.

| Setting | Default | What it does |
|---|---|---|
| `colors` | 0 | 0 = the sprite's own palette. N = merge it down to N of the sprite's own colours (fewer colours = less popping). |
| `palette` | `sprite` | Or a preset/hex list, to recolor the animation. |
| `stability` | 0.7 | A pixel keeps last frame's colour unless a new colour clearly wins. Higher is steadier; 0 = off. |
| `deflicker` | on | Removes one-frame blinks to a similar colour. |
| `cleanup` | on | Removes isolated stray pixels in each frame. |

The info output and Pixel Art Check report **colour popping**: the share of pixels per frame that switch to a
similar colour. On the swordsman test, the standing frames went from 8% (16 colours, no stability) to 0.06%
(6 colours, stability 0.7).

Sprite sheets are saved as `*.png` (one row), `*.json` (Aseprite format with `frameTags`, imports into Aseprite,
Godot, Unity and Phaser), an animated `*.webp` and an animated `*.gif`. `frame_step` 2 keeps every second frame (8 fps from 16 fps).
`loop_trim` drops repeated end frames. `sprite-from-prompt` uses the sprite as the first and last frame, so idle,
walk-in-place and other cycles loop. For one-shot actions, disconnect `end_image`.

### Tips
- Inputs with transparency keep it (the workflows rejoin LoadImage's alpha). Plain backgrounds are removed
  automatically; busy ones go through BiRefNet.
- For `image-to-directions`, a full-body image on a plain background works best. Edit a single line of
  "View prompts" to redo one view; keep the line order the same as `names` on the save node.
- Motion prompts: describe the motion and keep "static camera" and "flat solid background".
- Driving videos: one person, whole body, static camera, starting in a neutral standing pose.

### Compared with other pixel-art tools

| | These workflows | Aseprite MCP (`pixel-mcp`, `aseprite-mcp`) | PixelLab | Retro Diffusion |
|---|---|---|---|---|
| Runs | Locally, on your GPU | Locally (Aseprite + an LLM drawing pixels) | Cloud API, MCP, Aseprite plugin | Cloud API, Aseprite extension |
| Cost | Free after the download | Aseprite licence | Subscription / credits | Credits |
| Image → pixel art | Yes (PixelOE, palettes, exact grid) | Manual | Yes | Yes |
| Prompt → pixel art | Yes (Z-Image + LoRA) | LLM-drawn; good for icons/tiles, weak for characters | Yes | Yes (strong pixel-native models) |
| Animation | From a prompt or a driving video; exact sprite pixels kept | Manual, frame by frame | Yes (skeleton/template animations) | Yes |
| Directions | 8 (5 generated + mirrored) | Manual | 4/8 | 4/8 (some models) |
| Control | Full: every node and setting editable, scriptable through the `comfyui` skill | Full, but slow | Product UI/API options | Product UI/API options |
| Licence of outputs | Yours (see the licences below) | Yours | Per the service's terms | Per the service's terms |

A good split: generate here, then finish in Aseprite. Clean single pixels, retime frames, and export with
`aseprite -b sheet.png --data` or an Aseprite MCP server.

### Licences
- Z-Image Turbo, the pixel-art LoRAs (tarn59, Pixel Art Redmond, styly-agents, Teluv), the ai-toolkit trainer and
  its Z-Image training adapter, Wan 2.2, Wan Animate 2, PixelOE and BiRefNet are Apache-2.0 or MIT. Style LoRAs you
  train on your own art are yours (make sure bought asset packs allow ML training).
- **Qwen-Image-2.1** (`image-to-directions`) uses the Qwen Research licence (non-commercial) for the *model*.
  On 21 Sept 2026 the Qwen team stated that outputs are not part of the licensed materials and users retain the
  rights to the images they generate ([QwenDevs](https://x.com/QwenDevs/status/2101917379785838660)). Sprites you
  generate can be used in a commercial game. Don't redistribute the model or build a service on it.
- The Wan 2.2 Remix checkpoints are community merges; check their model card before commercial use.

## Image to SVG (local, VTracer)

Converts a raster image into an editable SVG on your own machine. It's free and nothing is uploaded; it takes 2–10 seconds on a 4090. It works best on **logos, icons, flat art, stylized characters on plain or transparent backgrounds, and illustrations**. Photos and soft gradients come out posterized, because that's what tracing does.

```
Image ─► AI upscale x4 ─► Resize (longest side 4096 px) ─┐
  └─ Mask ─► resize ─► hard edge (alpha > 0.5) ──────────┴─► VTracer (color) ─┬─► Save SVG (with preview)
                                                                               └─► SVG Stats ─► shapes & file size
```

**Usage:**
1. Open `image-to-svg/vtracer-svg` and load your image. A transparent PNG keeps its transparency.
2. Run. The SVG is saved to `output/svg/image_to_svg_00001_.svg` (numbered), previewed in the **Save SVG** node, and listed in the queue/history. The **Shapes & file size** box shows the number of shapes, distinct fill colours, the saved file size (drawing plus embedded workflow), and the canvas size, so you can see the effect of each tweak.
3. With an opaque image, a plain background becomes a single shape you can delete in Inkscape, Illustrator or Figma.
4. **Defaults favour detail** (4096 px). For simple art or smaller files, set both resize nodes to 2048, `layer_difference` 40 and `filter_speckle` 10: about a third of the file size, but tiny details get lost. **Simple logos, or large clean sources:** bypass the AI upscale node (`Ctrl+B`) to save time.

To change where files go, edit `filename_prefix` on the Save SVG node: `svg/characters/tangle` saves to `output/svg/characters/tangle_00001_.svg`. Files can only go under `output/`.

**Why these defaults:**
- **No colour-reduction step.** A pixel-count palette gets dominated by the background and anti-aliased edge pixels, and small areas like eyes or cheeks turned the wrong colour in testing. VTracer's own colour clustering keeps them.
- **AI upscale before resizing** ([Real-ESRGAN x4plus anime 6B](https://github.com/xinntao/Real-ESRGAN), 18 MB, BSD licence). Plain resizing keeps JPEG noise and soft edges, which trace as stair-stepped outlines. The AI upscale redraws crisp edges first. On a detailed 1408×768 JPEG character this gave visibly smooth outlines and webbing, and halved the SVG (1.05 MB → 551 KB).
- **Smoothing settings** (`spline`, `corner_threshold` 90, `splice_threshold` 60, `length_threshold` 6, `filter_speckle` 10). Compared with polygon mode and low thresholds, these remove wobbles and spurious corners while keeping small details like spiders and web strands. Going further (corner 120, splice 90, speckle 16) smooths more but starts merging fine detail.
- **4096 px tracing with `layer_difference` 24, `filter_speckle` 6.** At 2048 px with `layer_difference` 40 and `filter_speckle` 10, small low-contrast details (dark-brown spiders on a black visor, their 3 px eyes) merged into blobs. These settings keep legs and eyes with smooth outlines. The file is larger (1.4 MB vs 0.55 MB for that image), and tracing takes about 10 s.
- **Hard transparency edge.** Semi-transparent edge pixels otherwise turn into a fringe of tiny sliver shapes around the subject. On the test character this cut the file from 742 KB to 172 KB and gave a clean outline.

**Tuning: common problems**

| Problem | Change |
|---|---|
| Small details lost (pupils, highlights) | `filter_speckle` 8, and/or `layer_difference` 24–32 |
| Too many shapes / big file | `layer_difference` 64, `filter_speckle` 24, or `length_threshold` 6–8 |
| Colours shifted or merged | keep `color_precision` at 6 or higher |
| Image collapses into a few big blobs | the input is too soft or blurry: lower `layer_difference` (16), or sharpen the source |
| Jagged or stair-stepped edges | `mode` = spline, keep the AI upscale enabled, `corner_threshold` 90–120, `splice_threshold` 60–90 |
| Shapes with holes behave oddly in an editor | `hierarchical` = cutout |
| Simple flat logo / want small files | resize 1024–2048, `layer_difference` 40, `filter_speckle` 10, and bypass the AI upscale |
| Fine detail lost | `filter_speckle` 4, `layer_difference` 16, or check that both resize nodes are 4096 |

**Parameter reference**

Pipeline nodes:

| Node / parameter | Default | What it does |
|---|---|---|
| AI upscale x4 | on | Redraws the image at 4× with crisp edges before resizing. Bypass it (`Ctrl+B`) for large, clean sources or simple logos. |
| Resize (longest side) → `largest_size` | 4096 | Image size before tracing. Bigger gives smoother curves but slower tracing and larger files. Pixel-based settings (`filter_speckle`) act on this size. |
| Hard transparency edge → `value` | 0.5 | Pixels more transparent than this are dropped. Lower keeps more of soft edges (glows, shadows); higher trims tighter. Has no effect on opaque images. |
| Save SVG → `filename_prefix` | `svg/image_to_svg` | Path under `output/`; files are numbered (`_00001_`). Subfolders are allowed, and it can't write outside `output/`. The workflow is embedded in the SVG as metadata, the same as ComfyUI's PNGs. |

VTracer (color) node:

| Parameter | Default | What it does |
|---|---|---|
| `hierarchical` | stacked | **stacked**: shapes are layered on top of each other, which gives smaller files and no gaps. **cutout**: shapes are cut out of each other so none overlap, easier to recolor or move individually in an editor, but can show hairline gaps. |
| `mode` | spline | Curve style. **spline**: smooth curves (illustrations). **polygon**: straight segments (geometric or pixel art). **none**: exact pixel staircase outlines. |
| `filter_speckle` | 6 | Removes patches smaller than N×N pixels (at the resized size) before tracing. Higher means fewer tiny shapes but can drop small details. |
| `color_precision` | 6 | Significant bits per colour channel used to group colours (max 8). Higher keeps more distinct colours. Going below 6 shifts and merges colours. |
| `layer_difference` | 24 | How different neighbouring colour layers must be to stay separate. Lower keeps more layers, including gradient bands; higher merges into fewer, flatter shapes. On very soft images, high values can chain everything into one shape. |
| `corner_threshold` | 90 | Angle in degrees above which a turn counts as a sharp corner. Higher gives fewer corners and smoother outlines; lower keeps more sharp points. |
| `length_threshold` | 6.0 | Curves are subdivided until segments are shorter than this, in pixels. Higher gives fewer, longer segments (smaller, smoother files); lower follows the outline more tightly. |
| `max_iterations` | 20 | Maximum smoothing passes when fitting curves. Rarely needs changing. |
| `splice_threshold` | 60 | Angle in degrees at which a curve is split into a new spline segment. Lower gives more splice points and follows wiggly outlines closely; higher gives fewer, smoother segments. |
| `path_precision` | 3 | Decimal places in the SVG coordinates. 2–3 is plenty; lower makes smaller files with slightly rougher shapes. |

Cloud alternative: ComfyUI's built-in **Quiver Image to SVG** partner node (needs a comfy.org account and credits; about $0.03–$1 per image) redraws shapes with AI instead of tracing them, and handles photos and gradients better.

---

## Layered SVG (flats, tones, ink, details)

Where `vtracer-svg` traces every colour patch (about 2,900 shapes for a detailed character), `layered-svg` builds the SVG the way an artist would, as named layers that Inkscape and Illustrator show as layers:

| Layer (bottom → top) | What it is |
|---|---|
| **Flats** | One base colour per region, extended under the ink so there are no gaps. Each palette colour is its own group. |
| **Shadows** | A darker shade of each region's base colour where the image is darker (grouped per base colour) |
| **Highlights** | A lighter shade of each base colour where it's lighter |
| **Ink** | The dark outlines, traced smoothly. If a base colour is near-black, outlines use exactly that colour. |
| **Details** | Small things the layers above miss (web strands, eyes, specks), always in one of the base colours |

Flats are **stacked** (the biggest colour at the bottom, each colour extending under the next), which removed all of the 23,587 see-through seam pixels the earlier cut-out flats had on the test character.

On the detailed test character this gave **~430 shapes in 5 layers** (vs ~2,900 from `vtracer-svg`), or **~11** in compound mode, in about 8 s. The shape count depends on the artwork: fine repeated detail like web strands is real content and becomes details.

**Palette:** leave `palette` empty for `colors` automatic colours (default 6), or type hex colours, e.g. `#4f4f52 #8c8c90 #e3d6bb #4b3426`, and every flat snaps to the nearest one. Ink is detected separately, so black isn't needed in the palette.

| Parameter | Default | What it does |
|---|---|---|
| `colors` | 6 | Number of automatic base colours (ignored when `palette` is set) |
| `palette` | empty | Hex colours to use for the flats |
| `min_region_px` | 2000 | Flat regions smaller than this merge into their surroundings. Higher means fewer shapes. |
| `ink_threshold` | 22 | Pixels darker than this lightness (0–100) become ink |
| `tone_threshold` | 6 | How much darker/lighter (lightness) than the base colour a pixel must be to become shadow/highlight |
| `detail_threshold` | 25 | Colour difference from the layers below needed to add a detail. Lower gives more details. |
| `detail_min_px` | 60 | Smallest detail or ink speck kept |
| `shadow_opacity` / `highlight_opacity` | 0.3 / 0.35 | Shadow/highlight strength: how much darker/lighter the shade is than its base colour (in `overlay` style, the black/white opacity) |
| `curve_smoothness` | 1.1 | 0 = sharp corners everywhere, 1.33 = all smooth curves |
| `objects` | per region | **per region**: each region is its own shape (easy to select one armor plate). **compound**: one shape per colour/layer (fewest objects, recolor a whole layer at once). |
| `shadows` / `highlights` / `details` / `ink` | on | Include each layer |
| `remove_background` | on | For opaque images: makes the plain background, and large enclosed gaps of the same colour, transparent |
| `flat_stacking` | stacked | **stacked**: the colour with the most area goes at the bottom, and each colour extends under the ones drawn above it, so edges between colours always sit on a filled shape. That means no gaps and larger, simpler shapes. **cutout**: each colour covers only its own area, which can leave hairline gaps where colours meet. |
| `overlap_px` | 2 | Grows each flat colour this many pixels into the one below it, to hide seams |
| `line_min_px` / `line_max_px` | 0 / 0 (off) | Clamp the stroke width of **ink and details** (pixels at the traced size; Tangle Heart's lines are 2–12 px at 2048). Thinner strokes are thickened; thicker strokes are slimmed around their centre. Solid dark fills (e.g. a visor) are kept. Example: 4 / 8. |
| `tone_style` | tinted | **tinted**: shadows and highlights are shades of each base colour, so they add no new colours and recoloring a base is easy. **overlay**: black/white shapes at the opacities above. |
| `inset_under_ink_px` | 2 | Keeps fills, tones and details this many pixels inside the outer edge of outlines (still under the ink), so no light rim peeks out past the outline. Edges without an outline aren't affected. 0 = off. |

**Fewer shapes:** `min_region_px` 4000, `detail_min_px` 100 (about 250 shapes on the test character), or turn `details` off. **More detail:** `min_region_px` 1000, `detail_min_px` 30.

---

### With an AI repaint first (`layered-svg-ai`)

Same as `layered-svg`, but first **Qwen-Image-2.1** (an image-editing model) redraws the image as simplified flat art. The repaint instruction comes from the **Vector Art Prompt** node (mode "redraw `<image1>`"). Its default style, *faithful flat redraw*, identifies each element, keeps it recognizable with all its details, gives similar elements the same base colour, and uses only simple hard-edged highlights and shadows. Change `style` to cartoon, logo, sticker, line art and so on. `colors` / `palette` are asked for in the prompt *and* passed to Layered Vectorize. `detail` minimal/low gives fewer shapes. Put specifics ("spiders dark brown") in `subject`. "Final prompt" shows the text. The repainted PNG is shown and saved (`output/svg/layered_ai_repaint_*.png`), then vectorized.

- On Tangle Heart: **~210–330 shapes** depending on the seed (vs ~457 without AI), 40–70 s. Composition and outlines are kept and colours get simpler. The model may reinterpret some colours (e.g. a lighter armor, a wooden staff), and very small coloured details like spider bodies can turn into outlines. Results vary per seed; set `control_after_generate` to `fixed` to keep one you like.
- **Compare with and without:** bypass the "Qwen repaint" KSampler (`Ctrl+B`).
- To force exact colours, put hex colours (or a preset name) in Vector Art Prompt's `palette`. They go into the prompt and into Layered Vectorize, so include a colour for every area (e.g. a cape base colour).
- Models: Qwen-Image-2.1 int8 (7.3 GB), Qwen3-VL 8B int8 text encoder (9.4 GB), VAE (0.7 GB), downloaded via `env/svg.env`. The settings follow ComfyUI's official Qwen Image 2.1 edit template. **Licence: Qwen Research, non-commercial.**

---

### From a prompt (`prompt-to-svg`)

Type a **subject** into **Vector Art Prompt** and pick a **style**. The template adds what makes art trace cleanly:
- flat base colours (`colors`, or a `palette`), with similar parts sharing a colour;
- hard-edged shading (or none), bold outlines (they become the ink layer) and a detail level;
- a centred subject with margins, on a plain white background.

Z-Image Turbo (Apache-2.0) draws it in ~10 s. The image is saved, upscaled ×4 to 2048 px, and traced by Layered
Vectorize with the same colours.

| Test (RTX 4090) | Style, colours, detail | SVG |
|---|---|---|
| "a friendly fox mascot head" | cartoon character, 6, low | 24 shapes |
| "a mountain and rising sun emblem…" | logo / emblem, 4, minimal, flat | 34 shapes |
| "a treasure chest overflowing with gold coins" | game item / UI art, 6, medium | 45 shapes |

Tips:
- `shading` = flat and `detail` = minimal give the fewest shapes (logos, icons).
- `line art` gives colouring-book outlines.
- Without an exact colour count, the drawing can still use a few extra shades; Layered Vectorize snaps them to `colors`.

---

## Extend a video (`wan22-svi-extend`)

Adds more video after the end of an existing clip, guided by prompts. It uses your Wan 2.2 Remix models plus the **SVI 2.0 Pro** ("Stable Video Infinity") LoRAs, which continue a clip from the **last latent of the previous clip** (real motion continuation, not just the last frame) and keep returning towards an **anchor** frame so the subject stays consistent.

```
Video ─► 16 fps, resize ─► anchor frame + last 5 frames ─► encode
                                         │
   ┌──── Loop × segments ────────────────┴──────────────────────────────┐
   │ prompt line N ─► SVI conditioning(anchor, previous latent) ─►      │
   │ Remix HIGH (0→4) ─► Remix LOW (4→8) ─► latent → next iteration     │
   │                                     └► decode → collected          │
   └────────────────────────────────────────────────────────────────────┘
        ─► join (drop anchor frame, cross-fade overlap, colour match) ─► full MP4 + new-only MP4
```

**Usage:**
1. Load a video in **Input video**. It's resampled to 16 fps and resized (longest side 832 = 480p; 1280 = 720p is ~2.5× slower).
2. Set **segments** (the Start Loop's `num_iterations`). Each segment adds about 4.75 s (81 frames, minus the overlap and the anchor frame).
3. Write the prompts in **Prompts**, **one line per segment**, describing what happens next. If there are more segments than lines, the last line repeats.
4. Run. Two MP4s are saved in `output/video/`:
   - `wan22_extend_full_*`: the original plus all new segments, cross-faded at each join
   - `wan22_extend_new_*`: only the new frames
5. To keep going, load the `_full` video and run again with new prompts.

| Setting | Default | What it does |
|---|---|---|
| `segments` (Start Loop) | 2 | Number of extensions in one run |
| `anchor` | first frame | The frame SVI keeps returning towards. **first frame** keeps the original's identity; **last frame** follows where the video ended up. |
| `max_side` | 832 | Working resolution (longest side) |
| `length` (SVI conditioning) | 81 | Frames generated per segment (4n+1) |
| `overlap` / `color_match` (Join) | 4 / on | Frames cross-faded at each join; colour matching against the previous part limits colour drift |
| `seed` (Segment Prompt) | random | Segment N uses seed + N, so segments don't repeat the same noise |
| SVI LoRA strength | 1.0 | Lower it (e.g. 0.6–0.8) if motion feels stuck to the anchor |

Tested on a 49-frame 480p clip with 2 segments: 201 frames (12.6 s) out, about 3.5 minutes on an RTX 4090 (~1¾ min per segment). There was no visible jump or brightness step at either join (frame-to-frame change at the joins was below the clip's typical motion).

Known SVI trade-offs: it favours steady motion (the anchor pulls things back), follows prompts less strongly than plain image-to-video, and colours can drift over many segments (colour matching helps). No audio in this version. Models: SVI 2.0 Pro HIGH/LOW LoRAs from [Kijai/WanVideo_comfy](https://huggingface.co/Kijai/WanVideo_comfy/tree/main/LoRAs/Stable-Video-Infinity/v2.0), 1.23 GB each, downloaded via `env/video.env`.

---

## Image edit with optional inpainting (`qwen-edit`)

Edits an image from an instruction using **Qwen-Image-2.1**, for example *"Make the cape dark red"*, *"Remove the spider on the helmet"* or *"Replace the scythe with a spear, keep the pose"*. Use it to tweak an image step by step, for example before converting it to SVG.

1. Load an image.
2. **Optional inpainting:** right-click the image node → **Open in MaskEditor**, paint over the area to change, and Save. Only that area changes; everything outside the mask stays pixel-identical. With no mask, the whole image is edited.
3. Write the instruction in **Edit prompt** and refer to the image as `<image1>`. With a mask, describe the change for that area and add "keep everything else the same".
4. Run. The result is saved to `output/edit/qwen_edit_*.png`, and **Before / after** shows a comparison slider.
5. To keep editing, use the result as the next input and paint a new mask.

Tested: an edit without a mask changes the requested parts across the image; with a mask over the helmet, only the helmet changed, and the maximum pixel difference outside the mask was 0. About 30 s per edit.

`grow_px` (8) enlarges the painted area and `feather_px` (12) softens its edge so the edit blends in. Licence: Qwen Research (non-commercial).

---

## Wan 2.2 Remix — image to video

Animates a still image into a 5 s, 16 fps MP4 clip at up to 720p. No extra LoRAs needed.

**Why this model:** Wan 2.2 14B is the strongest open image-to-video model that runs locally, with no content filter and the largest community LoRA ecosystem. [Wan2.2-Remix](https://huggingface.co/FX-FeiHou/wan2.2-Remix) is a community fine-tune of the Wan 2.2 I2V high/low-noise experts. It merges in community motion and concept LoRAs, has the lightx2v Lightning distill baked in, and ships as fp8 checkpoints that fit on a 24 GB card. v3.0 is the final I2V release. The workflow pairs it with a [fine-tuned UMT5 text encoder](https://huggingface.co/NSFW-API/NSFW-Wan-UMT5-XXL) that follows prompts more closely than the stock one.

LTX-2 and HunyuanVideo 1.5 were also considered. They're good models, but their community LoRA support is much weaker than Wan's and they're more restrictive about content.

### Requirements

- Docker: covered by the setup above (`cp env/video.env .env`).
- Without Docker: a recent ComfyUI (late 2025 or later) plus the **ComfyUI-VideoHelperSuite** custom node, with the models below placed manually.
- About 36 GB of disk space and 32 GB or more of system RAM (WSL memory, when using Docker). ComfyUI swaps the two 14 GB experts between VRAM and RAM.

### Models

| File | Folder | Size |
|---|---|---|
| [Wan2.2_Remix_NSFW_i2v_14b_high_lighting_fp8_e4m3fn_v3.0.safetensors](https://huggingface.co/FX-FeiHou/wan2.2-Remix/blob/main/NSFW/Wan2.2_Remix_NSFW_i2v_14b_high_lighting_fp8_e4m3fn_v3.0.safetensors) | `models/diffusion_models` | 14.3 GB |
| [Wan2.2_Remix_NSFW_i2v_14b_low_lighting_fp8_e4m3fn_v3.0.safetensors](https://huggingface.co/FX-FeiHou/wan2.2-Remix/blob/main/NSFW/Wan2.2_Remix_NSFW_i2v_14b_low_lighting_fp8_e4m3fn_v3.0.safetensors) | `models/diffusion_models` | 14.3 GB |
| [nsfw_wan_umt5-xxl_fp8_scaled.safetensors](https://huggingface.co/NSFW-API/NSFW-Wan-UMT5-XXL/blob/main/nsfw_wan_umt5-xxl_fp8_scaled.safetensors) | `models/text_encoders` | 6.7 GB |
| [wan_2.1_vae.safetensors](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/blob/main/split_files/vae/wan_2.1_vae.safetensors) | `models/vae` | 0.25 GB |

These URLs are also stored in the workflow's loader nodes, which is what the Docker downloader and ComfyUI's missing-models dialog read.

> Use the **Wan 2.1 VAE**, not `wan2.2_vae.safetensors`. The 2.2 VAE is only for the 5B TI2V model.

### How it works

```
Start image ─┐
Prompts ─────┼─► WanImageToVideo ─► KSampler HIGH (steps 0→4, adds noise)
             │                    ─► KSampler LOW  (steps 4→8, refines)
             │                    ─► VAE Decode ─► MP4 (16 fps)
```

Wan 2.2 is a two-expert model. The high-noise expert lays out motion and composition, and the low-noise expert adds detail. Two `KSamplerAdvanced` nodes split one 8-step schedule between them. The yellow **Total steps** and **Switch at step** nodes keep both samplers in sync.

### Usage

1. Load your start image.
2. Set **width/height** to match the image's aspect ratio. The image is scaled and center-cropped to that size.
   - Portrait: `720×1280` (quality) or `480×832` (fast)
   - Landscape: `1280×720` or `832×480`
   - Square: `960×960` or `640×640`
3. In the positive prompt, describe the **motion and camera**: what moves, how, and in what direction. Don't re-describe what's already in the image. Be specific; the fine-tuned text encoder follows detailed prompts well.
4. Set **frames** using 4n+1 values: 81 = 5 s, 49 = 3 s. Going above ~121 frames makes motion loop or degrade.
5. Click Run. The video is saved to `output/wan22_remix_i2v/`.

### Recommended settings (already set)

| Setting | Value | Why |
|---|---|---|
| Steps / switch | 8 / 4 | The Remix author's recommendation. Use 6 / 3 for faster drafts. |
| CFG | 1.0 | Lightning distill is baked in. Higher CFG burns the image. Note that the negative prompt has little effect at CFG 1. |
| Sampler / scheduler | euler / simple | |
| Shift | 8 | |
| LoRAs | none | Concept LoRAs and Lightning are already merged in. **Don't** add the lightx2v LoRAs on top. |

The two **Optional LoRA** nodes are bypassed. To use a Wan 2.2 I2V concept LoRA, select the node, press `Ctrl+B`, and pick the HIGH and LOW files. Start at strength 0.6–1.0.

### Performance on an RTX 4090 (approximate)

Rough figures for 81 frames at 8 steps; they vary with your system RAM and PyTorch/Comfy build:

- 480×832: ~1.5–3 min
- 720×1280: ~5–9 min

If you run out of memory at 720p, lower the frame count, drop to 480p, or launch ComfyUI with `--reserve-vram 1`. Adding SageAttention (`--use-sage-attention`) speeds generation up noticeably.

### Ideas for extending it

- **Smoother motion:** add RIFE frame interpolation (ComfyUI-Frame-Interpolation, `rife47`) after the VAE decode, set ×2, and save at 32 fps.
- **Longer clips:** chain runs by feeding the last frame of one clip in as the start image of the next.

### Content responsibility

Only animate images you have the right to use. Never use images of real people without their consent, and never use images of minors. Generated content is your responsibility under the laws where you live.
