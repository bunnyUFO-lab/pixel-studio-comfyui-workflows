# One reference for a pixel environment

Draw the location once, then use that actual image to condition a prop and two terrain materials. The example is a side-on sandstone coast with turquoise water under a peach sky. The output includes an opaque background, transparent prop, two repeatable 32px tiles, a 16-tile corner atlas, and an editable Tiled map.

## Draw the background

Run from the workflows checkout using the normal ComfyUI server. Replace the scene prompt to choose a different location and keep its light direction, colours, and camera consistent in the later prompts. In these examples, `$S` is `.claude/skills/comfyui/scripts/comfy.py`.

```bash
python "$S" run workflows/pixel-art/environment-master.json --out out/coast/master
```

The graph draws at 1536×640 with Z-Image Turbo and the pixel-art LoRA. `Master pixels.pixels=160` preserves the whole frame as 384×160. It preserves the sky and does no cutout or wrapping. Use the native `environment_master_*.png` as the visual reference and copy the hex string printed by `Master palette`. The raw drawing is available separately for inspecting composition or choosing a new seed. The default sampler seed is 2601.

## Draw the matching asset set

Run one asset at a time. `$MASTER` is the native master PNG and `$PALETTE` is the copied hex palette. Both asset graphs use Qwen-Image-Edit-2511 with the four-step Lightning adapter, a 768×768 draw, and the actual master connected as an image input. Props and materials use separate prompts and postprocessing. Each saves a raw image so framing and style can be checked before pixel reduction.

```bash
python "$S" run workflows/pixel-art/environment-prop.json \
  --input "Environment reference=$MASTER" \
  --set "Prop pixels.palette=$PALETTE" \
  --out out/coast/prop

python "$S" run workflows/pixel-art/environment-material.json \
  --input "Environment reference=$MASTER" \
  --set "Material pixels.palette=$PALETTE" \
  --out out/coast/sandstone

python "$S" run workflows/pixel-art/environment-material.json \
  --input "Environment reference=$MASTER" \
  --set "Asset prompt.prompt=Use the input landscape only as a pixel-art style and lighting reference. Draw a square edge-to-edge dry dark muted gray-green slate paving material with broad broken flat rock plates, irregular interlocking cracks and charcoal crevices. Low contrast and large clean pixel clusters under soft even light. Show only the matte material viewed directly overhead, with no horizon, scenery, objects, shine, text or borders." \
  --set "Asset sample.seed=2604" \
  --set "Material pixels.palette=#29362d #414d41 #515f52 #657663" \
  --set "Material pixels.smooth=0.3" \
  --out out/coast/slate
```

| Control | Purpose |
| --- | --- |
| `Scene prompt.text` | Describe the full landscape, its camera, light and dominant colours. |
| `Canvas size.width/height` | Keep the drawing near 2.4:1 and derive a map size that fits the intended native grid. |
| `Master sample.seed`, `Asset sample.seed` | Freeze a chosen design for reuse. Default asset seeds are 2602 for the prop and 2603 for material A. |
| `Asset prompt.prompt` | Ask for the prop or the material alone. Use the reference for style and light. For props ask for a centred object on flat white. For materials ask for a flat surface filling the whole frame. |
| `Prop pixels.pixels` | Set the prop's longest native dimension. The default is 56. |
| `Prop pixels.palette`, `Material pixels.palette` | Paste the same master hex colours for both. When a material needs shades missing from the backdrop, try a short list sampled from its raw draw. |
| `Seamless material.tile_size` | Set the native tile dimension. Keep it at most half the height of `Material pixels.pixels`, and use the same value for both materials. The defaults are 32 and 64. |

Use `environment_prop_*.png` for the transparent object. It is cut out with BiRefNet and snapped to the requested pixel count and palette. Use `environment_tile_*.png` for each terrain material and inspect `environment_tile_repeat_*.png` at 3×3 before accepting it. A repeated shape can remain noticeable even if its edge joins fit. The background and raw assets need no transparency treatment.

If the default texture looks busy at game size, give each material a short list of three to five compatible master swatches. Try `Material pixels.pixels=32` and `Material pixels.smooth=0.8`, then `Seamless material.tile_size=16` and `Output scale.scale_by=2`. That samples a 16px wrap tile and enlarges whole pixels to a 32px map tile. Set `Seamless material.preview_scale=4` to keep the 3×3 check at 192×192. Apply these settings to sandstone with `#a35b4e #b94934 #c47b38 #d1785f #dc9637` to reproduce the quieter ground in the example. Keep both exported tiles at 32px.

The slate example preserves four muted colours from the raw material. Constraining it to the coastal master colours alone produced dark teal and hid the cracked plates. Check the composed scene before deciding whether to extend a palette.

## Assemble and edit the map

This step uses Pillow. Specify layout in a tiny JSON file. Each list entry is one row of **corners**, with a row and a column more than the map's tile grid. `A` and `B` choose materials, and any corner marked `.` leaves its adjoining tile cells empty. This 13×6 corner grid makes a 12×5 tile map (384×160 at 32 pixels per tile). The optional prop path is relative to the layout JSON. `x` is horizontal centre and `ground_y` is the first row below the aligned prop.

```json
{
  "corners": [
    ".............",
    ".............",
    ".............",
    ".............",
    "AAAAABBAABBBB",
    "AAAABBBBABBBB"
  ],
  "props": [
    {"image": "prop/environment_prop_00001_.png", "x": 112, "ground_y": 130}
  ]
}
```

```bash
python tools/compose_pixel_environment.py \
  --background "$MASTER" --material-a "$SANDSTONE" --material-b "$SLATE" \
  --layout out/coast/layout.json --out out/coast/map
```

Open `environment.tmj` in Tiled with the PNGs in the same directory. It contains ground cells, an embedded corner terrain set, and separate backdrop and prop image layers. `environment-scene.png` is a flattened native preview. `environment-ground.png` and `environment-tiles.png` can also be used separately. The metrics JSON measures average RGB-channel jumps across each material's two repeat edges and between its ordinary adjacent pixels. Lower repeat jumps are a useful numeric screen but check the 3×3 render for visual repetition too.

The corner atlas has 16 deterministic masks and preserves the input material colours. Layout, walkable geometry, and collision remain authoring decisions. This workflow does not infer a playable level or create an isometric/hex map from the master picture. Keep the flat PNG and edit ground or image-layer positions in Tiled, or edit the corner grid and regenerate it.

Tiled's [JSON map format](https://doc.mapeditor.org/en/stable/reference/json-map-format/) documents the atlas and terrain metadata.
