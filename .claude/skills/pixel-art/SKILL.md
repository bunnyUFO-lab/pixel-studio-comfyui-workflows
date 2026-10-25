---
name: pixel-art
description: Make game-ready true pixel art through Pixel Studio (so Run on / cloud routing, the queue and the library apply), falling back to this repo's local ComfyUI pixel-art workflows - image to pixel art, prompt to pixel art, 8-direction character sprites, and animated sprite sheets from a motion prompt or a driving video - and iterate until the result is clean. Use for sprites, characters, items, tiles, sprite sheets, walk/idle/attack animations, palettes (PICO-8, DB32, Endesga-32, GameBoy) and pixel size questions. Also use when a Sprite Studio sprite or animation is made Final, or the user asks to make one final, production / game ready, polished, clean or high quality: do the pixel cleanup pass through the Pixel Studio MCP first.
---

# Pixel art (through Pixel Studio)

Generation goes through Pixel Studio (its UI may still say Sprite Studio), so its Run on setting, `source` routing, queue
and library apply. Its MCP is the interface; its REST API is used only to import a local file. Whatever the MCP cannot
do keeps the direct `comfy.py` path, labelled **local only** below.

```bash
P=.claude/skills/pixel-art/scripts/studio.py      # Pixel Studio client (stdlib only)
C=.claude/skills/comfyui/scripts/comfy.py         # direct ComfyUI (see the comfyui skill)
```
Base URL: `PIXEL_STUDIO_URL` (default `http://localhost:8190`; `--server` overrides it).

## Pick the path (once per request)
1. `python $P status`. If it answers `ok:true`, use the **Studio path**.
2. If it is unreachable and the user asked for neither the cloud nor a named source: `python $C status`. When ComfyUI is
   up, go **direct** and say in the reply: "Pixel Studio isn't reachable at <url>, so this ran on local ComfyUI
   directly: no Run on or cloud routing, and nothing is in the library. The outputs are in <dir>."
3. **Stop** and say what to start if both are down, or if the user asked for the cloud or a named source (only Pixel
   Studio can route those).

On the Studio path, use the MCP tools if they are registered (server `pixel-studio`, or `sprite-studio` in older
setups). Otherwise `python $P call <tool> --arg k=v ... [--json '{...}']` calls the same tools on the same server
(values are parsed as JSON when possible). Read the MCP server instructions and tool descriptions; they are not
repeated here.

Rules:
- **Job discipline:** queue, then `wait_for_job` (or `python $P wait <job_id>`). A timeout means the job is still running:
  wait again, never resubmit. Pixel Studio queues jobs, so a running job is no reason to hold back. (For direct
  `comfy.py` runs, still don't start a job while the user has one running.)
- **Local only only for a gap** in the table below, and say "local only: <gap>" in the reply. A job that **failed** in
  Pixel Studio is not a reason to fall back: report the failure.
- **`source`:** pass it only when the user names one (`auto`, or a name from `list_sources`, usually `pc` or `cloud`).
  The `pixel-studio` skill has the batch and cost guidance for choosing; don't copy it here. If the tool answers
  "unknown argument source", the app predates it: run again without `source` and say "routing followed the app's Run on".

## Mapping: goal to Pixel Studio

| Goal | Studio path | Local only (direct `comfy.py` workflow) |
|---|---|---|
| Prompt to pixel art | `generate_sprite(target, prompt, pixels, view, style, palette, variants, references, canvas_w/h)`; the target is `generate/characters` or `generate/objects`. Styles come from `list_styles`, not the node's list. | a fixed seed; the raw non-pixel image; `Pixel Art Prompt` `detail` / `outline` / `background` -> `pixel-art/prompt-to-pixel` |
| Image to pixel art | `python $P import <subject> art.png --kind image`, then `make_sprite(image, pixels, palette, remove_background)` | the Pixelize settings (`method`, `smooth`, `colors`/`color_error`, `outline`, `cleanup`, `edge_cleanup`, `padding`), Clean cutout `drop_shadow` / `enclosed_background`, and **recovering existing or upscaled pixel art** (`measure=auto (detect grid)`) -> `pixel-art/image-to-pixel` |
| 8 directions | `make_directions(sprite, mirror)`. Its input must be a **sprite**: import raw art, run `make_sprite`, then `make_directions`. To fix one view use `edit` on the directions set. | per-view prompt lines (`View prompts`), `num_iterations` -> `pixel-art/image-to-directions` |
| Sheet from a motion prompt | `animate(asset, motion, name, frames, fps, loop, speed)`; the asset is a sprite or directions set | Repixelate settings (`colors`, `palette`, `stability`, `deflicker`, `cleanup`), `Loop ... length`, `padding`, seeds -> `pixel-art/sprite-from-prompt` |
| Sheet from a driving video | `python $P import <subject> clip.mp4`, then `animate(asset, motion_reference=<video id>, mirror_motion, extra_prompt)`. For a web link use `add_reference(url, type="motion")` instead of importing. | trimming (`frame_load_cap`, `skip_first_frames`) and `height_match`; trim with ffmpeg first, or use the local path -> `pixel-art/sprite-from-video` |
| Outputs to a folder | `get(ref)` lists `files`; `python $P download <ref> --out DIR` | `comfy.py run --out DIR` |
| Quality check | the result's `warnings`, `view`, `read_pixels` | the full Pixel Art Check PASS/FAIL report |

## Studio-path examples
```bash
python $P import generate/characters art/knight.png          # -> {"id", "kind", "ref"}; --kind image is the default
python $P call make_sprite --arg image=<ref> --arg pixels=48 --arg palette=Endesga-32 --arg remove_background=true
python $P call generate_sprite --arg target=generate/objects --arg 'prompt=a red potion bottle' --arg pixels=24
python $P call make_directions --arg sprite=<sprite ref> --arg mirror=true
python $P call animate --arg asset=<sprite ref> --arg 'motion=the character walks in place, legs and arms swinging' \
    --arg name=walk --arg frames=16 --arg loop=true
python $P import character/12 clip/attack.mp4                # a video asset, whatever --kind says
python $P call animate --arg asset=<sprite ref> --arg motion_reference=<video id>
python $P wait <job_id>                                        # then: python $P download <result ref> --out out/knight
```
`import` sends `.gif` and `.webp` as `kind=auto`, so an animated one lands as a reference (or whatever kind the server picks), never a video: if `import` prints that note, ask the user to convert it to mp4. A static one stays an image.
Then `view` the result and read its `warnings`; iterate with the checklist below.

## Local only
Used only for these gaps, none of which the Pixel Studio MCP covers yet (each would need op params and workflows in
Pixel Studio itself):
1. No local-file upload in the MCP: `studio.py import` uses the REST import instead, so the work still goes through
   Pixel Studio. (A follow-up would add an MCP `import` taking base64 `data`.)
2. No seed parameter on the generation tools.
3. Fine Pixelize, Repixelate and Clean cutout settings.
4. Grid recovery of upscaled pixel art.
5. Video trimming.
6. Per-view direction prompts.
7. The full Pixel Art Check report.

These run the `workflows/pixel-art/*` workflows through the **comfyui** skill's `scripts/comfy.py` (read that skill for
server checks, `inspect`, `run`, `--set`, `--input`). Run `status` first and don't start jobs while the user has one
running. They are also the fallback when Pixel Studio is down.

| Goal | Workflow | Inputs (title=...) | Typical time |
|---|---|---|---|
| Existing image/art -> pixel art | `pixel-art/image-to-pixel` | `Input image` | ~3 s |
| Text -> pixel art sprite | `pixel-art/prompt-to-pixel` | `Pixel Art Prompt.subject` (+ style, view, pixels, colors, palette) | ~20 s |
| Character -> 8 directions | `pixel-art/image-to-directions` | `Character image` | ~4 min |
| Pixel sprite + motion text -> looping sheet | `pixel-art/sprite-from-prompt` | `Pixel sprite`, `Motion prompt.text` | ~1 min |
| Pixel sprite + video -> sheet | `pixel-art/sprite-from-video` | `Pixel sprite`, `Driving video` | ~1-2 min |

Chains: prompt-to-pixel -> image-to-directions (on the raw image `prompt_to_pixel_raw_*.png`) or -> sprite-from-prompt
(on the native `prompt_to_pixel_*.png`). The sprite workflows want a **native** sprite (upscaled copies also work;
the grid is detected).

```bash
python $C run pixel-art/image-to-pixel --input "Input image=art/knight.png" \
    --set "Pixelize.pixels=48" --set "Pixelize.palette=Endesga-32" --set "Pixel Art Check.expected_height=48" --out out/knight
python $C run pixel-art/prompt-to-pixel --set 'Pixel Art Prompt.subject="a red potion bottle"' \
    --set "Pixel Art Prompt.style=item / icon" --set "Pixel Art Prompt.pixels=24" --set "KSampler.seed=5" --out out/potion
python $C run pixel-art/sprite-from-prompt --input "Pixel sprite=out/knight/image_to_pixel_00001_.png" \
    --set 'Motion prompt.text="Pixel art sprite animation. The character walks in place, legs and arms swinging. Static camera, the character stays in place, flat solid background."' \
    --set "Save sprite sheet.tag=walk" --out out/knight_walk
python $C run pixel-art/sprite-from-video --input "Pixel sprite=out/knight.png" --input "Driving video=clips/attack.mp4" --out out/attack
python $C run pixel-art/image-to-directions --input "Character image=art/hero.png" --out out/hero_dirs
```

### Key settings (node title . widget; all local only)
- `Pixel Art Prompt` (prompt-to-pixel): `subject` (just what to draw), `style` (clean game sprite, 16-bit RPG,
  8-bit retro, chibi, top-down RPG, side-scroller, isometric, item / icon, monster / creature, prop / tile, none),
  `view`, `pixels` (also sets detail: <=24 minimal, <=48 low, <=96 medium), `detail`, `colors`, `palette`, `outline`,
  `extra`. Its pixels/colors/palette drive Pixelize and the check; "Final prompt" shows the text.
- `Pixelize.pixels` (64): size in art pixels. `Pixelize.measure`: `subject height` (default), `longest side`,
  `image height`, `cell size`, `body height (per frame)` (ignores thin props; directions), `auto (detect grid)`
  (recover existing/upscaled pixel art exactly).
- `Pixelize.colors` (0 = automatic fewest colours) + `Pixelize.color_error` (20; higher = fewer colours: 12 faithful,
  30 very few), or an exact number; `Pixelize.palette` (`auto`, `PICO-8`, `GameBoy`, `DB32`, `Endesga-32`, or hex list).
- Animations: `Repixelate ....colors` (0 = sprite palette, N = merge to N of its colours), `.palette` (`sprite` or a
  preset/hex), `.stability` (0.7), `.deflicker`, `.cleanup`. These re-run cheaply on the cached video.
- `Pixelize.method`: `pixeloe` (default, cleanest outlines), `dominant`, `median`, `center`, `mean`.
- `Pixelize.smooth`: 0.6 for painted/photo/textured sources, 0 for flat or already pixel-styled art.
- `Pixelize.cleanup` (stray pixels), `Pixelize.edge_cleanup` (defringe + silhouette smoothing), `Pixelize.outline`
  (`auto`, `darker shade`, `darkest`, `outside darkest`, `none`), `Pixelize.padding`.
- `Clean cutout (drop shadow, background pockets).drop_shadow` (on) / `.enclosed_background` (off; turn on for
  background visible through gaps, but check that eye whites survive).
- `Pixel Art Prompt.background`: plain white (default), light grey, green / magenta chroma key.
- Sprite sheets: `Save sprite sheet.frame_step` (2), `max_frames`, `loop_trim`, `columns`, `fps`, `tag`.
- Playback: `Animation preview (playback).scale` / `.background` / `.playback` (fps comes from the sheet); `run --out`
  downloads the preview WebP too (`sprite_preview_*.webp`) - open it to check the motion.
- Animation: `Loop (first = last frame) / frames.length` (49, 4n+1), `Prepare sprite (integer upscale).padding`
  (room for motion), seed on `KSampler HIGH noise (seed here).noise_seed` / `Sampler (seed here).noise_seed`.
- Directions: `View prompts (one line per view).prompts` (one line per view) + `Views (...).mode.num_iterations`,
  `Save directions (mirror -> 8).names` / `mirror`.
- Video: `Repixelate (height match, same grid + palette).height_match`, `Driving video.frame_load_cap` /
  `skip_first_frames`.

## Game-ready checklist (iterate until all true)
1. **Check the result.** Studio path: the result's `warnings` are empty and `view` looks right. Local-only runs:
   **Pixel Art Check = PASS** (printed after `run`): hard alpha, native x1, palette <= max, consistent height
   (off in sprite-from-video: poses change the height), expected height.
2. Look at the output (`view`, or the `_x8` / `_preview.png` file with the Read tool). Check for broken outlines,
   speckle (texture noise), missing thin parts, a leftover background, and readable faces/eyes.
3. Animations: frame 1 matches the sprite; colour popping well under 1% on still parts; the character stays in place;
   the loop is seamless (sprite-from-prompt). Check the `*_raw*.mp4` when motion looks wrong (local-only runs).
4. Directions: the same character in every view, facing the right way, with feet on the same line.

Fixes (rows marked **local only** need the direct workflow; on the Studio path use the tool's own parameters, `edit`,
or a new variant):
| Problem | Change |
|---|---|
| Speckled/noisy areas | **local only:** `smooth` 0.6-1.0, `colors` 0 with higher `color_error`. Studio: fewer `colors`/`palette` in `make_sprite` |
| Colours popping in animations | **local only:** fewer colours (Re-pixelate `colors` 4-8), `stability` 0.8-0.9; check the popping % in the info |
| Lost thin highlights/details | **local only:** `smooth` 0, `method` pixeloe. Studio: more `pixels` |
| Broken outline | **local only:** `outline` auto / darkest, keep `edge_cleanup` on. Studio: the cleanup pass below |
| Light specks / halo on edges | **local only:** keep `edge_cleanup` on (defringe); background visible through gaps -> Clean cutout `enclosed_background` |
| Shadow blob under feet | **local only:** Clean cutout `drop_shadow` on (default) |
| Background left over | Studio: `remove_background`. Local only: keep BiRefNet on; raise `bg_tolerance` for plain backgrounds |
| Wrong motion / camera moves | rewrite the motion prompt, another variant, more room (`extend`) |
| One bad direction | Studio: `edit` that view. Local only: edit just that prompt line, rerun |
| Colours drift between frames | they can't (palette is locked to the sprite); if the sprite palette is bad, fix the sprite first |

## Pixel Studio: poses, in-betweens and masks (MCP)
Pixel Studio's MCP (`$PIXEL_STUDIO_URL/mcp`) also runs Wan 2.2 VACE jobs (about 3-9 min each):
- **A pose from any picture (humanoids):** `add_reference` (type pose) a photo or GIF frame, then `animate_between`
  with `pose_reference`. Only the pose is used: the reference's limb angles go onto the sprite's own skeleton (SDPose),
  so it keeps its look, facing and proportions. Its last frame is the sprite in that pose: `view` it before building
  on it. Keep `method=guide`; `keyframe` (Qwen draws the pose first) drifts from the character. (`match_pose` was
  removed: it handed non-humanoids back unchanged and still said it worked.)
- **Animate into a pose or along a clip:** `animate_between` (to another sprite, a pose reference, or with only a
  `motion_reference`, the clip's skeletons; `loop` ends on the sprite). Steadier for humanoid swings and cycles than
  `animate` with a video (Wan Animate copies the whole clip and may turn the character).
- **Fix an animation:** `ai_animate` inserts frames or redraws frames between two frames or picked ones, optionally
  following a `motion_reference` or passing through a `pose_reference`.
- **Add an object to every frame:** `ai_animate mode=masked` with `mask_colors` (found per frame) or `mask_box`;
  `mask_up` leaves room above a `mask_colors` mask (a flame). (`mask_region` was removed: it missed too often.) The first frame is drawn by the edit model, then carried to the others. `edit` takes the same masks.
- Pose skeletons need a humanoid. For creatures use `animate` with a prompt or a similar clip, and
  masks by box or colour.
- `view` every result; if the character drifted, run again with another seed instead of fixing it by hand.

## Pixel Studio: the cleanup pass before Final
Do this when a draft or kept sprite / animation in Pixel Studio becomes **Final**, or when the user asks for it to
be final, production ready, game ready, shippable, polished, cleaned up or high quality. Generated frames often have
small flaws the models leave behind; fix them by hand through the Pixel Studio MCP (`$PIXEL_STUDIO_URL/mcp`),
not with the AI edit tools. Frame numbers are as the app shows them (1 = first).

1. **Look.** `view` the asset, then `read_pixels` every frame (a text map: one letter per pixel, a colour legend,
   x across / y down from 0,0). Also `read_pixels` the source sprite (the result's `made_from`) to learn which colours
   belong where and what the outline colour is.
2. **Find the flaws**, frame by frame:
   - colours from another part where they don't belong (e.g. the beak's brown on wing edges or feather tips);
   - the silhouette outline: it should be the sprite's outline colour (usually the darkest, black) all round; lighter
     pixels on the edge are flaws, unless the source sprite has them there too;
   - eyes and faces readable in every frame; when the eyes close (a blink, a kiss), a clear closed-eye shape (a small
     arc or ^ in the outline colour) instead of a few leftover specks;
   - props and effects (hearts, sparkles, slashes) crisp and in the palette, not blobs; nothing cut off at the frame
     edge (`extend`: "redraw with more room" is the best fix for a generated animation);
   - the same part keeps the same colours from frame to frame.
   Keep what is meant to be there: the beak, claws, eye shines, highlights, small loose sparkles.
3. **Fix** with `paint`: `pixels` ([[x, y, "#rrggbb" | "transparent"]]) and / or `replace` (colour swaps inside a
   `box`). The first call saves a new version (`mode=new`); paint the other frames onto that version with
   `mode=replace`, so the original stays as it was. `read_pixels` again to check every change landed.
4. **QA in the browser** (your own tab): play the animation, step through the frames, compare with the original.
   Rename the new version to say what it is (e.g. "walk (clean outlines)"), then mark it Final if that was the ask.
5. Tell the user what was fixed (how many pixels, which frames) and what was deliberately kept.

## Finishing (optional)
Sheets come with an Aseprite JSON (`frameTags`). Import with Aseprite (File -> Import Sprite Sheet, or
`aseprite -b --sheet-type rows ...`) for manual touch-ups (for Pixel Studio assets, prefer the cleanup pass above);
Godot/Unity/Phaser read the PNG + JSON directly. Native PNGs must be scaled with nearest-neighbour only.

## Licence note
Qwen-Image-2.1 (image-to-directions) has a non-commercial *model* licence, but the generated images belong to the
user (QwenDevs, 21 Sept 2026). All other models used here are Apache-2.0/MIT; Wan 2.2 Remix is a community merge.
