---
name: pixel-art
description: Make game-ready true pixel art with this repo's local ComfyUI pixel-art workflows - image to pixel art, prompt to pixel art, 8-direction character sprites, and animated sprite sheets from a motion prompt or a driving video - and iterate until Pixel Art Check passes. Use for sprites, characters, items, tiles, sprite sheets, walk/idle/attack animations, palettes (PICO-8, DB32, Endesga-32, GameBoy) and pixel size questions. Also use when a Sprite Studio sprite or animation is made Final, or the user asks to make one final, production / game ready, polished, clean or high quality: do the pixel cleanup pass through the Sprite Studio MCP first.
---

# Pixel art (local ComfyUI)

Runs the `workflows/pixel-art/*` workflows through the **comfyui** skill's `scripts/comfy.py` (read that skill for
server checks, `inspect`, `run`, `--set`, `--input`). Always run `status` first and don't start jobs while the user has
one running.

```bash
C=.claude/skills/comfyui/scripts/comfy.py
python $C status
```

## Pick the workflow

| Goal | Workflow | Inputs (title=…) | Typical time |
|---|---|---|---|
| Existing image/art → pixel art | `pixel-art/image-to-pixel` | `Input image` | ~3 s |
| Text → pixel art sprite | `pixel-art/prompt-to-pixel` | `Pixel Art Prompt.subject` (+ style, view, pixels, colors, palette) | ~20 s |
| Character → 8 directions | `pixel-art/image-to-directions` | `Character image` | ~4 min |
| Pixel sprite + motion text → looping sheet | `pixel-art/sprite-from-prompt` | `Pixel sprite`, `Motion prompt.text` | ~1 min |
| Pixel sprite + video → sheet | `pixel-art/sprite-from-video` | `Pixel sprite`, `Driving video` | ~1–2 min |

Chains: prompt-to-pixel → image-to-directions (on the raw image `prompt_to_pixel_raw_*.png`) or → sprite-from-prompt
(on the native `prompt_to_pixel_*.png`). The sprite workflows want a **native** sprite (upscaled copies also work;
the grid is detected).

## Examples

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

## Key settings (node title . widget)
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
1. **Pixel Art Check = PASS** (printed after `run`): hard alpha, native ×1, palette ≤ max, consistent height
   (off in sprite-from-video: poses change the height), expected height.
2. Look at the output: open the `_x8` / `_preview.png` file (Read tool). Check for broken outlines, speckle
   (texture noise), missing thin parts, a leftover background, and readable faces/eyes.
3. Animations: frame 1 matches the sprite; colour popping well under 1% on still parts; the character stays in place; the loop is seamless
   (sprite-from-prompt). Check the `*_raw*.mp4` when motion looks wrong.
4. Directions: the same character in every view, facing the right way, with feet on the same line.

Fixes:
| Problem | Change |
|---|---|
| Speckled/noisy areas | `smooth` 0.6–1.0, `colors` 0 with higher `color_error` |
| Colours popping in animations | fewer colours (Re-pixelate `colors` 4–8), `stability` 0.8–0.9; check the popping % in the info |
| Lost thin highlights/details | `smooth` 0, more `pixels`, `method` pixeloe |
| Broken outline | `outline` auto / darkest, keep `edge_cleanup` on |
| Light specks / halo on edges | keep `edge_cleanup` on (defringe); background visible through gaps -> Clean cutout `enclosed_background` |
| Shadow blob under feet | Clean cutout `drop_shadow` on (default) |
| Background left over | keep BiRefNet on; raise `bg_tolerance` for plain backgrounds |
| Wrong motion / camera moves | rewrite the motion prompt, new seed, more `padding` |
| One bad direction | edit just that prompt line, rerun |
| Colours drift between frames | they can't (palette is locked to the sprite); if the sprite palette is bad, fix the sprite first |

## Sprite Studio: poses, in-betweens and masks (MCP)
Sprite Studio's MCP (`http://localhost:8190/mcp`) also runs Wan 2.2 VACE jobs (about 3-9 min each, one at a time):
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

## Sprite Studio: the cleanup pass before Final
Do this when a draft or kept sprite / animation in Sprite Studio becomes **★ Final**, or when the user asks for it to
be final, production ready, game ready, shippable, polished, cleaned up or high quality. Generated frames often have
small flaws the models leave behind; fix them by hand through the Sprite Studio MCP (`http://localhost:8190/mcp`),
not with the AI edit tools. Frame numbers are as the app shows them (1 = first).

1. **Look.** `view` the asset, then `read_pixels` every frame (a text map: one letter per pixel, a colour legend,
   x across / y down from 0,0). Also `read_pixels` the source sprite (the result's `made_from`) to learn which colours
   belong where and what the outline colour is.
2. **Find the flaws**, frame by frame:
   - colours from another part where they don't belong (e.g. the beak's brown on wing edges or feather tips);
   - the silhouette outline: it should be the sprite's outline colour (usually the darkest, black) all round; lighter
     pixels on the edge are flaws, unless the source sprite has them there too;
   - eyes and faces readable in every frame; when the eyes close (a blink, a kiss), a clear closed-eye shape (a small
     ◠ or ^ in the outline colour) instead of a few leftover specks;
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
Sheets come with an Aseprite JSON (`frameTags`). Import with Aseprite (File → Import Sprite Sheet, or
`aseprite -b --sheet-type rows ...`) for manual touch-ups (for Sprite Studio assets, prefer the cleanup pass above);
Godot/Unity/Phaser read the PNG + JSON directly. Native PNGs must be scaled with nearest-neighbour only.

## Licence note
Qwen-Image-2.1 (image-to-directions) has a non-commercial *model* licence, but the generated images belong to the
user (QwenDevs, 21 Sept 2026). All other models used here are Apache-2.0/MIT; Wan 2.2 Remix is a community merge.
