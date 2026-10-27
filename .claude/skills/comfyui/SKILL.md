---
name: comfyui
description: Run and inspect this repo's ComfyUI workflows on the local ComfyUI server (http://localhost:8188, Docker). Use to check server/GPU status, list workflows, see a workflow's editable settings, run a workflow with changed settings and uploaded input files, collect output files and text (e.g. Pixel Art Check / SVG Stats reports), and to author new workflows with tools/build_*.py.
---

# ComfyUI (local)

For game assets use the `pixel-art` skill: it goes through Pixel Studio so routing applies. This skill stays direct: inspection, authoring (`tools/build_*.py`) and local-only runs.

ComfyUI runs in Docker from this repo (`docker compose up -d`). Its workflow library *is* this repo's
`workflows/` folder, and the repo's own nodes live in `custom_nodes/comfyui_workflows_utils/`.
Everything below goes through `scripts/comfy.py` (Python stdlib only), run from the repo root.

Setup and upkeep use the `pixel-studio-comfy` CLI (`bin/`, installed by `install.ps1` / `install.sh`;
[docs/setup.md](../../../docs/setup.md)): `doctor`, `up`, `down`, `status`, `models`, `logs`, `update`, `token`.
For a cloud GPU, `pixel-studio-comfy cloud modal deploy|snippet` deploys the Modal job API and prints the Pixel Studio
`comfy-job` source (the live app needs `--live`; see `cloud/README.md` and `docs/job-api.md`). This skill talks to
the local server only.

## Before running anything
- `python .claude/skills/comfyui/scripts/comfy.py status` checks the server, GPU, free VRAM and queue.
- **Don't start jobs or restart ComfyUI while the user has a job running** (queue shows running/pending).
  Ask first if the queue isn't empty.
- New or changed repo nodes need a restart: `docker compose restart` (only when the queue is empty). New
  workflows/README need no restart; reload the browser page.

## Commands
```bash
S=.claude/skills/comfyui/scripts/comfy.py
python $S list                                    # workflows (paths under workflows/, no .json)
python $S inspect pixel-art/image-to-pixel        # editable settings per node: '#id [Type] Title' + widget = value
python $S run pixel-art/image-to-pixel \
    --input "Input image=art/knight.png" \
    --set "Pixelize.pixels=48" --set "Pixelize.palette=PICO-8" \
    --out out/knight                              # waits, downloads output files, prints text outputs
python $S run <workflow> --dump-api prompt.json --no-wait   # queue without waiting; save the API prompt
python $S result <prompt_id> --out DIR            # outputs of a finished job
python $S queue | interrupt | free                # queue, stop current job, unload models (free VRAM)
```
- Nodes are addressed by **title** (as shown in the UI), else by type (`KSampler`), else `#id`. `--set` applies to
  every matching node. Values are parsed as JSON (`48`, `true`, `"text"`), otherwise used as strings; give
  multi-line text as a JSON string (`--set "Prompts.prompts=\"line 1\nline 2\""`). Dotted widgets such as Start
  Loop's `mode.num_iterations` work as shown by `inspect` (`--set "Segments (num_iterations).mode.num_iterations=3"`).
- `--input "Title=path"` uploads the file to ComfyUI's input folder and sets that node's image/video/audio widget.
- Text outputs (nodes like `PreviewAny`, e.g. "Pixel Art Check" / "Shapes & file size") are printed after the run.
  Use them to decide whether to iterate.
- Seeds: workflows usually have `control_after_generate = randomize` in the UI, but `run` sends the saved seed.
  Use `--set "KSampler.seed=123"` (or the node's title) to vary it.

## How the conversion works (and its limits)
`comfy.py` converts the UI-format workflow JSON to ComfyUI's API format using `/object_info`. It handles:
- widget order, including `control_after_generate` slots and dynamic combos such as Start Loop's `mode.num_iterations`
- dict-style `widgets_values` (VideoHelperSuite)
- links, Reroute, PrimitiveNode
- muted nodes (mode 2, dropped) and bypassed nodes (mode 4, inputs passed through by type)
- `%date:…%` filename placeholders

It was checked against the frontend's own `graphToPrompt` for every workflow in the repo; the only difference
was a timestamp. **Subgraphs are not expanded**: keep repo workflows flat.

## Authoring workflows in this repo
- Workflows are **generated**: edit or add `tools/build_<name>.py`, then run it (`cd tools && python build_<name>.py`).
  Don't hand-edit `workflows/*.json`.
- Use the helpers in `tools/comfy_workflow.py`: `Graph()`, `node(id, type, pos, size, inputs, outputs, widgets, title, mode, color, props)`, `link(src, slot, dst, slot)`, `inp`, `out`, colours, and shared model constants.
- Conventions:
  - Every model loader node gets `props={"models": [{"name", "url", "directory"}]}`. `docker/models.py` downloads these and ComfyUI's missing-model dialog uses them.
  - Each workflow has a `Note` titled "READ ME".
  - Colours: blue for inputs, green for prompts and outputs, yellow for key settings.
- After generating:
  - `python docker/models.py <workflow>` with `WORKFLOWS_DIR=workflows MODELS_DIR=/x` lists the required models.
  - `docker compose run --rm comfyui models download <workflow>` downloads them.
  - Then `inspect` and `run` it with this skill.
- Add new presets to `env/*.env` and document every workflow in `README.md` (table row, section, settings table).
