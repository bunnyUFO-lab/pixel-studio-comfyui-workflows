"""CPU tests for tools/sprite_checks.py (numpy + Pillow only).

Run from the repo root:  python tests/test_sprite_checks.py   (or: python -m unittest tests.test_sprite_checks)
Sprites are drawn in code; bad sheets are made from good ones by blur, sub-pixel shift, noise, colour jitter, resizing.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import sprite_checks as sc  # noqa: E402

S = 8          # pixel size
N = 24         # native frame size
FRAMES = 6


def draw_frame(seed, i):
    """One native NxN RGBA frame of a little walking figure; limbs swing with i, feet stay on row N-3."""
    rng = np.random.RandomState(seed)
    pal = rng.randint(40, 230, size=(4, 3))
    body, head, legs, outline = (tuple(int(v) for v in p) for p in pal)
    im = np.zeros((N, N, 4), np.uint8)

    def rect(x0, y0, x1, y1, c):
        im[y0:y1, x0:x1] = (*c, 255)

    rect(8, 4, 16, 8, outline)           # head with outline
    rect(9, 5, 15, 7, head)
    rect(9, 8, 15, 16, body)             # torso
    sw = (0, 1, 2, 1, 0, -1)[i % 6]
    rect(9 + sw, 16, 11 + sw, N - 3, legs)       # legs swing sideways, feet row fixed
    rect(13 - sw, 16, 15 - sw, N - 3, legs)
    rect(7, 9 + abs(sw), 9, 13 + abs(sw), head)  # arms
    rect(15, 9 + abs(sw), 17, 13 + abs(sw), head)
    return im


def up(native):
    return np.kron(native, np.ones((S, S, 1), np.uint8))


def good_frames(seed):
    return [up(draw_frame(seed, i)) for i in range(FRAMES)]


def to_sheet(frames):
    return np.concatenate(frames, axis=1)


def save(arr, path):
    Image.fromarray(arr).save(path)


# ---- corruptions (each takes the list of good frames and a seed, returns a list of frames)
def blur(frames, seed):
    return [np.array(Image.fromarray(f).filter(ImageFilter.GaussianBlur(2.0))) for f in frames]


def subpixel_shift(frames, seed):
    rng = np.random.RandomState(seed)
    out = []
    for f in frames:
        dx, dy = int(rng.randint(1, S)), int(rng.randint(1, S))
        out.append(np.roll(np.roll(f, dx, axis=1), dy, axis=0))
    return out


def noise(frames, seed):
    rng = np.random.RandomState(seed)
    out = []
    for f in frames:
        g = f.astype(np.int16)
        g[..., :3] += rng.randint(-10, 11, size=g[..., :3].shape)
        out.append(np.clip(g, 0, 255).astype(np.uint8))
    return out


def colour_jitter(frames, seed):
    rng = np.random.RandomState(seed)
    out = []
    for f in frames:
        g = f.astype(np.int16)
        mask = g[..., 3] > 0
        g[mask, :3] += rng.randint(-25, 26, size=3)
        out.append(np.clip(g, 0, 255).astype(np.uint8))
    return out


def resized_frame(frames, seed):
    out = list(frames)
    k = len(out) // 2
    h, w = out[k].shape[:2]
    big = Image.fromarray(out[k]).resize((int(w * 1.15), int(h * 1.15)), Image.NEAREST)
    big = big.crop((0, big.height - h, w, big.height))   # keep the frame size, feet near the bottom
    out[k] = np.array(big)
    return out


BAD = [blur, subpixel_shift, noise, colour_jitter, resized_frame]


class SpriteChecksTest(unittest.TestCase):
    def test_good_sheet_scores(self):
        r = sc.score_frames(good_frames(1))
        self.assertEqual(r["grid_scale"], S)
        self.assertGreaterEqual(r["grid_purity"], 0.97)
        self.assertEqual(r["grid_stability"], 1.0)
        self.assertEqual(r["colour_flicker"], 0.0)
        self.assertLessEqual(r["same_character"]["palette_overlap_min"], 1.0)
        self.assertGreaterEqual(r["same_character"]["palette_overlap_min"], 0.9)
        self.assertLessEqual(r["palette_size"]["sheet"], 12)
        g = r["gates"]
        self.assertLessEqual(g["ground_shift"], 1)
        self.assertLess(g["zoom_variation"], 0.1)
        self.assertEqual(g["background_clean"], 1.0)
        self.assertTrue(g["loop_ok"])

    def test_each_corruption_hurts(self):
        base = sc.score_frames(good_frames(2))["overall"]
        for fn in BAD:
            with self.subTest(fn.__name__):
                bad = sc.score_frames(fn(good_frames(2), 7))["overall"]
                self.assertLess(bad, base)

    def test_five_good_rank_above_five_bad(self):
        good = [sc.score_frames(good_frames(s))["overall"] for s in range(10, 15)]
        bad = [sc.score_frames(fn(good_frames(20 + i), i))["overall"] for i, fn in enumerate(BAD)]
        self.assertGreater(min(good), max(bad), f"good={good} bad={bad}")

    def test_single_image_works(self):
        r = sc.score_frames([good_frames(3)[0]])
        self.assertEqual(r["grid_scale"], S)
        self.assertIsNone(r["colour_flicker"])
        self.assertIsNone(r["same_character"])
        self.assertGreater(r["overall"], 0.9)

    def test_given_scale(self):
        r = sc.score_frames(good_frames(4), scale=S)
        self.assertEqual(r["grid_scale"], S)
        self.assertGreaterEqual(r["grid_purity"], 0.97)

    def test_no_grid_found_is_reported(self):
        r = sc.score_frames(noise(good_frames(5), 1))
        self.assertFalse(r["grid_found"])
        self.assertIsNone(r["grid_scale"])
        self.assertEqual(r["quality"]["grid_purity"], 0.0)

    def test_cli_sheet_and_folder_json(self):
        with tempfile.TemporaryDirectory() as d:
            frames = good_frames(6)
            sheet = os.path.join(d, "sheet.png")
            save(to_sheet(frames), sheet)
            folder = os.path.join(d, "frames")
            os.makedirs(folder)
            for i, f in enumerate(frames):
                save(f, os.path.join(folder, f"f{i + 1}.png"))
            tool = os.path.join(ROOT, "tools", "sprite_checks.py")
            out1 = subprocess.run([sys.executable, tool, sheet, "--frames", f"{FRAMES}x1", "--json"],
                                  capture_output=True, text=True, check=True).stdout
            out2 = subprocess.run([sys.executable, tool, folder, "--scale", str(S), "--json"],
                                  capture_output=True, text=True, check=True).stdout
            r1, r2 = json.loads(out1), json.loads(out2)
            self.assertEqual(r1["frames"], FRAMES)
            self.assertEqual(r1["kind"], "sheet")
            self.assertEqual(r2["kind"], "folder")
            self.assertAlmostEqual(r1["overall"], r2["overall"], places=3)
            table = subprocess.run([sys.executable, tool, sheet, "--frames", f"{FRAMES}x1"],
                                   capture_output=True, text=True, check=True).stdout
            self.assertIn("grid purity", table)

    def test_bad_sheet_division_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.png")
            save(good_frames(1)[0], p)
            self.assertEqual(sc.main([p, "--frames", "5x1", "--json"]), 2)


if __name__ == "__main__":
    unittest.main()
