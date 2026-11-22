"""Compose a Tiled-ready environment from two repeating materials and a hand-authored corner map."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageStat


CORNERS = ("top-left", "top-right", "bottom-right", "bottom-left")
MATERIAL_B = "B"


def open_rgba(path):
    with Image.open(path) as image:
        return image.convert("RGBA")


def load_layout(path):
    layout = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(layout, dict):
        raise ValueError("layout must be an object")
    rows = layout.get("corners")
    if (not isinstance(rows, list) or len(rows) < 2 or not isinstance(rows[0], str) or len(rows[0]) < 2
            or any(not isinstance(row, str) or len(row) != len(rows[0]) or set(row) - {"A", "B", "."}
                   for row in rows)):
        raise ValueError("corners must be a rectangular grid of A, B, and dots, at least 2×2")
    props = layout.get("props", [])
    if not isinstance(props, list):
        raise ValueError("props must be a list")
    return rows, props


def transition_mask(size, index):
    """Four-corner Wang mask; adjacent tiles agree along their shared edge."""
    a, b, c, d = ((index >> i) & 1 for i in range(len(CORNERS)))
    last = size - 1
    threshold = last * last
    pixels = []
    for y in range(size):
        # Integer bilinear interpolation avoids diagonal cracks from floating-point roundoff.
        luma_left = a * (last - y) + d * y
        luma_right = b * (last - y) + c * y
        pixels.extend(255 if 2 * (luma_left * (last - x) + luma_right * x) >= threshold else 0
                      for x in range(size))
    mask = Image.new("L", (size, size))
    mask.putdata(pixels)
    return mask


def atlas_and_cells(a, b, rows):
    size = a.width
    atlas = Image.new("RGBA", (size * 4, size * 4))
    for index in range(16):
        atlas.paste(Image.composite(b, a, transition_mask(size, index)), (index % 4 * size, index // 4 * size))
    cells = []
    for top, bottom in zip(rows, rows[1:]):
        for x in range(len(top) - 1):
            corners = top[x], top[x + 1], bottom[x + 1], bottom[x]
            cells.append(0 if "." in corners else 1 + sum(1 << i for i, c in enumerate(corners) if c == MATERIAL_B))
    return atlas, cells


def image_layer(id_, name, file, position=(0, 0)):
    return {"type": "imagelayer", "name": name, "id": id_, "image": file, "x": 0, "y": 0,
            "offsetx": position[0], "offsety": position[1], "opacity": 1, "visible": True}


def tiled_map(cells, width, height, size, prop_layers):
    colors = [{"name": "Material A", "color": "#c8855d", "probability": 1, "tile": 0},
              {"name": "Material B", "color": "#699b81", "probability": 1, "tile": 15}]
    wangtiles = []
    for index in range(16):
        tl, tr, br, bl = (1 + ((index >> i) & 1) for i in range(len(CORNERS)))
        wangtiles.append({"tileid": index, "wangid": [0, tr, 0, br, 0, bl, 0, tl]})
    return {
        "type": "map", "version": "1.10", "orientation": "orthogonal", "renderorder": "right-down",
        "width": width, "height": height, "tilewidth": size, "tileheight": size, "infinite": False,
        "nextlayerid": 3 + len(prop_layers), "nextobjectid": 1,
        "layers": [
            image_layer(1, "Backdrop", "environment-backdrop.png"),
            {"type": "tilelayer", "name": "Ground", "id": 2, "width": width, "height": height,
             "x": 0, "y": 0, "opacity": 1, "visible": True, "data": cells},
            *prop_layers,
        ],
        "tilesets": [{"firstgid": 1, "name": "Environment terrain", "image": "environment-tiles.png",
                      "imagewidth": size * 4, "imageheight": size * 4, "tilewidth": size, "tileheight": size,
                      "spacing": 0, "margin": 0, "columns": 4, "tilecount": 16,
                      "wangsets": [{"name": "Environment materials", "type": "corner", "tile": -1,
                                    "colors": colors, "wangtiles": wangtiles}]}],
    }


def rgb_diff(a, b):
    return round(sum(ImageStat.Stat(ImageChops.difference(a.convert("RGB"), b.convert("RGB"))).mean) / 3, 2)


def colors(image):
    return {pixel for _, pixel in image.getcolors(image.width * image.height)}


def material_metrics(tile):
    w, h = tile.size
    return {
        "colors": len(colors(tile)),
        "left_right_edge_delta": rgb_diff(tile.crop((0, 0, 1, h)), tile.crop((w - 1, 0, w, h))),
        "top_bottom_edge_delta": rgb_diff(tile.crop((0, 0, w, 1)), tile.crop((0, h - 1, w, h))),
        "interior_horizontal_delta": rgb_diff(tile.crop((0, 0, w - 1, h)), tile.crop((1, 0, w, h))),
        "interior_vertical_delta": rgb_diff(tile.crop((0, 0, w, h - 1)), tile.crop((0, 1, w, h))),
    }


def resolve_props(props, directory, canvas_size):
    items = []
    for prop in props:
        if (not isinstance(prop, dict) or not isinstance(prop.get("image"), str)
                or type(prop.get("x")) is not int or type(prop.get("ground_y")) is not int):
            raise ValueError("each prop needs an image path, integer x, and integer ground_y")
        image = open_rgba(directory / prop["image"])
        alpha = image.getchannel("A")
        bbox = alpha.getbbox()
        if bbox is None or alpha.getextrema() != (0, 255) or any(alpha.histogram()[1:255]):
            raise ValueError(f"{prop['image']} must have transparent and solid pixels with hard alpha")
        image = image.crop(bbox)
        dx, dy = prop["x"] - image.width // 2, prop["ground_y"] - image.height
        if dx < 0 or dy < 0 or dx + image.width > canvas_size[0] or dy + image.height > canvas_size[1]:
            raise ValueError(f"{prop['image']} placement falls outside the canvas")
        items.append((image, (dx, dy)))
    return items


def compose(background, material_a, material_b, layout_path, out):
    a, b, scene = (open_rgba(p) for p in (material_a, material_b, background))
    if a.size != b.size or a.width != a.height or a.width < 2:
        raise ValueError("materials must be equal square tiles at native resolution")
    if any(im.getchannel("A").getextrema() != (255, 255) for im in (a, b, scene)):
        raise ValueError("materials and background must be fully opaque")
    rows, props = load_layout(layout_path)
    size = a.width
    width, height = len(rows[0]) - 1, len(rows) - 1
    if scene.size != (width * size, height * size):
        raise ValueError(f"background must be {width * size}×{height * size} for this corner grid and tile size")
    prop_images = resolve_props(props, layout_path.parent, scene.size)
    atlas, cells = atlas_and_cells(a, b, rows)
    terrain = Image.new("RGBA", scene.size)
    for offset, gid in enumerate(cells):
        if gid:
            col, row = (gid - 1) % 4 * size, (gid - 1) // 4 * size
            tile = atlas.crop((col, row, col + size, row + size))
            terrain.paste(tile, (offset % width * size, offset // width * size))
    master_palette = colors(scene)
    out.mkdir(parents=True, exist_ok=True)
    scene.save(out / "environment-backdrop.png")
    scene.alpha_composite(terrain)
    prop_layers = []
    for n, (image, position) in enumerate(prop_images, 1):
        file = f"environment-prop-{n}.png"
        image.save(out / file)
        scene.alpha_composite(image, position)
        prop_layers.append(image_layer(2 + n, f"Prop {n}", file, position))
    atlas.save(out / "environment-tiles.png")
    terrain.save(out / "environment-ground.png")
    scene.save(out / "environment-scene.png")
    (out / "environment.tmj").write_text(json.dumps(tiled_map(cells, width, height, size, prop_layers), indent=2) + "\n",
                                          encoding="utf-8")
    metrics = {
        "native_size": list(scene.size), "tile_size": size, "ground_cells": sum(gid != 0 for gid in cells),
        "atlas_cells": 16, "props_placed": len(prop_images), "scene_colours": len(colors(scene)),
        "master_colours": len(master_palette),
        "material_a": material_metrics(a), "material_b": material_metrics(b),
        "asset_colours_outside_master": len((colors(a) | colors(b) |
            {px for img, _ in prop_images for px in colors(img) if px[3]}) - master_palette),
    }
    (out / "environment-metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("background", "material-a", "material-b", "layout", "out"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    try:
        metrics = compose(args.background, args.material_a, args.material_b, args.layout, args.out)
    except (KeyError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

