"""Environment recipe checks without a ComfyUI server or model downloads."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from compose_pixel_environment import compose, transition_mask  # noqa: E402


class EnvironmentGraphTests(unittest.TestCase):
    def graph(self, name):
        path = Path(__file__).resolve().parents[1] / "workflows" / "pixel-art" / f"{name}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def nodes(self, name):
        return {node.get("title"): node for node in self.graph(name)["nodes"] if node.get("title")}

    def test_detailed_master_retains_the_native_contract(self):
        classic = self.nodes("environment-master")
        detailed = self.nodes("environment-master-detailed")
        old_graph = self.graph("environment-master")
        new_graph = self.graph("environment-master-detailed")
        self.assertEqual(old_graph["links"], new_graph["links"])
        self.assertEqual(len(old_graph["nodes"]), len(new_graph["nodes"]))
        for old_node, new_node in zip(old_graph["nodes"], new_graph["nodes"]):
            if old_node.get("title") in {"Scene prompt", "Master pixels", "Save native master", "Save raw master"}:
                self.assertEqual({**old_node, "widgets_values": new_node["widgets_values"]}, new_node)
            else:
                self.assertEqual(old_node, new_node)
        self.assertEqual(detailed["Pixel-art LoRA"]["widgets_values"][1], 1.0)
        self.assertEqual(detailed["Master sample"]["widgets_values"][:4], [2601, "fixed", 8, 1.0])
        old, new = (nodes["Master pixels"]["widgets_values"] for nodes in (classic, detailed))
        self.assertEqual(old[:4], new[:4])
        self.assertEqual((old[4], new[4]), (32, 48))
        self.assertEqual(old[5:], new[5:])
        self.assertEqual(new[8], "keep")
        self.assertIn("Save native master", detailed)
        self.assertIn("Save raw master", detailed)

    def test_quieter_material_retains_the_repeat_contract(self):
        classic = self.nodes("environment-material")
        detailed = self.nodes("environment-material-detailed")
        self.assertEqual(detailed["Seamless material"]["widgets_values"], [32, 0, 3, 2, "both", 0])
        self.assertEqual(detailed["Asset sample"]["widgets_values"], classic["Asset sample"]["widgets_values"])
        self.assertEqual(detailed["Material pixels"]["widgets_values"][3], 0.3)
        self.assertIn("Save native repeating tile", detailed)
        self.assertIn("Save repeat check", detailed)


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
