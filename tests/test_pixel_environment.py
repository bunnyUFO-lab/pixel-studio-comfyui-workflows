"""Environment recipe checks without a ComfyUI server or model downloads."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from compose_pixel_environment import compose, transition_mask  # noqa: E402


class EnvironmentMapTests(unittest.TestCase):
    def test_every_matching_pair_shares_edge_mask(self):
        size = 8
        masks = [transition_mask(size, i) for i in range(16)]
        self.assertEqual(masks[0].getextrema(), (0, 0))
        self.assertEqual(masks[15].getextrema(), (255, 255))
        for i, a in enumerate(masks):
            for j, b in enumerate(masks):
                if (i >> 1) & 1 == j & 1 and (i >> 2) & 1 == (j >> 3) & 1:
                    self.assertEqual(a.crop((size - 1, 0, size, size)).tobytes(),
                                     b.crop((0, 0, 1, size)).tobytes())
                if (i >> 3) & 1 == j & 1 and (i >> 2) & 1 == (j >> 1) & 1:
                    self.assertEqual(a.crop((0, size - 1, size, size)).tobytes(),
                                     b.crop((0, 0, size, 1)).tobytes())

    def test_editable_map_and_flat_scene_have_same_pixels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a, b, backdrop, prop = (root / f"{x}.png" for x in ("a", "b", "bg", "prop"))
            Image.new("RGBA", (8, 8), "#bb5533").save(a)
            Image.new("RGBA", (8, 8), "#226677").save(b)
            Image.new("RGBA", (16, 16), "#eed0aa").save(backdrop)
            icon = Image.new("RGBA", (3, 3))
            icon.putpixel((1, 1), (255, 255, 255, 255))
            icon.save(prop)
            layout = root / "layout.json"
            layout.write_text(json.dumps({
                "corners": ["...", "ABA", "ABB"],
                "props": [{"image": "prop.png", "x": 4, "ground_y": 9}],
            }), encoding="utf-8")
            out = root / "map"
            metrics = compose(backdrop, a, b, layout, out)
            self.assertEqual(metrics["native_size"], [16, 16])
            self.assertEqual(metrics["ground_cells"], 2)
            tmj = json.loads((out / "environment.tmj").read_text(encoding="utf-8"))
            bg, ground, mark = tmj["layers"]
            rebuilt = Image.open(out / bg["image"]).convert("RGBA")
            self.assertEqual(ground["data"], [0, 0, 7, 14])
            with Image.open(out / tmj["tilesets"][0]["image"]) as atlas:
                for index, gid in enumerate(ground["data"]):
                    if gid:
                        x, y = (gid - 1) % 4 * 8, (gid - 1) // 4 * 8
                        rebuilt.alpha_composite(atlas.crop((x, y, x + 8, y + 8)), (index % 2 * 8, index // 2 * 8))
            with Image.open(out / mark["image"]) as sprite:
                rebuilt.alpha_composite(sprite, (mark["offsetx"], mark["offsety"]))
            with Image.open(out / "environment-scene.png") as flattened:
                self.assertEqual(flattened.tobytes(), rebuilt.tobytes())
            Image.new("RGBA", (8, 8)).save(b)
            with self.assertRaisesRegex(ValueError, "fully opaque"):
                compose(backdrop, a, b, layout, out)


if __name__ == "__main__":
    unittest.main()
