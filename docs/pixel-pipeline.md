# Pixel art with diffusion models: what we do before and after the model

Diffusion models do not draw on a pixel grid. They draw smooth pictures that look like pixel art. So the work is in two places: prepare the input so the model has the best chance, then force the output back onto a real grid and palette. The model in the middle can be swapped.

The code is one file in this repository, [`custom_nodes/comfyui_workflows_utils/pixel.py`](../custom_nodes/comfyui_workflows_utils/pixel.py). It is plain NumPy, so it works outside ComfyUI too.

There are two pipelines: a picture into a sprite, and a sprite into an animation.

## A. Picture to sprite (image models)

### Before the model

Node `PixelArtPrompt` in [`prompts.py`](../custom_nodes/comfyui_workflows_utils/prompts.py) writes the prompt.

1. **Ask for one subject on a plain flat background.** The prompt ends with a sentence such as "Isolated on a plain flat pure white background." The background is cut out later, so it must be easy to tell from the subject.
2. **Name the view.** One view per picture, for example side view or front three-quarter, with the full body in frame.
3. **Draw large.** Let the model work at its normal size (1024 by 1024). Do not ask it for a 64 px image. The shrinking is ours to do.

### After the model

Node `CleanCutout`, then function `pixelize_frames` (node `Pixelize`).

1. **Cut out the background.** Flood from the border over pixels close to the border colour.
2. **Defringe.** Edge pixels are a blend of subject and background. Replace them with the nearest solid interior colour, so the blend never reaches the palette.
3. **Find the grid.** A model's "pixel art" has blocks of a regular size. Measure where the strong colour steps line up and use that pitch. If there is no grid, choose the cell size from the subject's height and the sprite size you want.
4. **Build the palette.** Use the fewest colours that keep the colour error low (measured in Lab, on flat areas only). Then rescue small accents such as glowing eyes, which are too few pixels to move an error score.
5. **One colour per cell, picked, not averaged.** An average is a new colour that was never in the drawing. Take the dominant colour of the cell.
6. **Hard alpha.** A cell is opaque if at least half of it is covered.
7. **Tidy.** Remove single stray pixels, fill one-pixel notches in the silhouette, remove background trapped between an arm and the body, and remove a drawn ground shadow.
8. **Outline.** If most of the silhouette edge is already one dark colour, make all of it that colour.

## B. Sprite to animation (video models)

This is the part that took the most testing. The model here is an image-to-video model with the sprite as its first frame.

### Before the model

Node `PrepareSpriteForVideo`.

1. **Recover the native sprite.** If the sprite was saved enlarged, detect the factor and sample one pixel per cell to get the true grid back.
2. **Give it room.** Pad the canvas so a raised weapon stays in frame: about half the figure's height above, a quarter of its width at each side, two pixels below.
3. **Enlarge the whole canvas by a whole number, nearest neighbour.** We use 8. Every art pixel becomes an exact 8 by 8 block. Never use a smooth resize and never a fractional factor.
4. **Put it on a flat neutral grey.** Choose the grey or white farthest from the sprite's colours. A magenta or green key bleeds into the body of the clip; a neutral does not.
5. **Make width and height divisible by what the model needs** (16 or 32), by adding whole art pixels to the canvas.
6. **Remember the spec:** the factor, the key colour, the offsets and the sprite's own colours. The clean-up needs all of it.
7. **Caption the action only.** Two or three short sentences of beats in order: wind-up, action, recovery. Do not describe the character or the style. The first frame already shows them.

Why 8: we measured how much of the first frame comes back correct after the round trip. At 4 it is 75%, at 6 it is 93%, at 8 it is 98.7%, at 10 it is 99.7%. At 4 the eyes, belt and straps blur away.

We generate 39 frames with a few-step sampler and no CFG, then keep 13.

### After the model

Function `clean_video_frames`, node `CleanVideoFrames`.

1. **Pick the frames.** Keep 13 of the 39, evenly spaced across the whole action. For a loop, make the last frame equal the first.
2. **Collapse each block to one pixel.** A block is background if at least half of its pixels are near the key colour. Otherwise take the pixel at the block's centre. Taking the centre keeps single-pixel detail; a vote or an average smears it.
3. **Trim blended edges.** Drop a silhouette-edge pixel whose colour is nearer the key than any sprite colour.
4. **Snap to the sprite's own palette.** Every pixel takes the nearest of the colours the sprite started with. No new shades.
5. **Keep effects as effects.** A pixel far from every sprite colour is a spark, a trail or a flash. Keep it, limited to 4 new colours for the whole clip. The distance that counts as "far" is set from the first frame's own error, since the first frame is the sprite itself and its error is only video noise. Effect groups smaller than 3 pixels are noise and go.
6. **Leave small gaps alone.** A key-coloured region inside the figure is cut out only if it has at least 4 pixels. Smaller ones stay body.
7. **Part-following hold** (function `follow_hold`). For each pixel, find where its 5 by 5 neighbourhood was in the sprite, or in the frame before, within 3 pixels. Where it matches and the video agrees, take that colour. This removes most flicker without smearing motion, because it follows a part as it moves instead of comparing fixed positions.
8. **Feet rule** (function `feet_band_rule`). Remove effect blobs that reach the bottom 8 rows, unless their colour also shows higher up in the clip as a trail.
9. **Clear the colour under transparent pixels** at the very end. The hold needs it while it runs; nothing after should see it.

## What we learned the hard way

- **Scale beats clean-up.** No clean-up setting at factor 4 fixed the blur. Moving to 8 did.
- **A hold on still pixels does nothing visible.** On a walk only 2% of changing pixels change while the figure is still. Most change because a part moved. The hold has to follow the part.
- **Never average colours.** At every step, pick an existing colour.
- **The first frame is a free calibration.** It should come back as the sprite. Its error tells you the noise level for that clip.
- **Not fixed by any of this:** the upper body bobbing by a pixel on walks, and a limb drawn behind a large head. Both come from the model.

## Where to look

| What | Where in `pixel.py` |
| --- | --- |
| Picture to sprite | `pixelize_frames`, class `Pixelize` |
| Palette choice | `auto_palette`, `rescue_accents`, `reduce_palette` |
| Grid detection | `find_grid`, `detect_scale` |
| Sprite to video input | class `PrepareSpriteForVideo`, `pick_neutral_background`, `recover_native` |
| Frame picking | class `PickEvenFrames` |
| Video clean-up | `clean_video_frames`, `_collapse_centre`, `follow_hold`, `feet_band_rule` |
| Ready workflows | [`sprite-from-prompt-h3`](../workflows/pixel-art/sprite-from-prompt-h3.json) (animation, MiniMax H3) and [`prompt-to-pixel`](../workflows/pixel-art/prompt-to-pixel.json) (sprite) |
