"""CPU tests for tools/sprite_checks.py (numpy + Pillow only).

Run from the repo root:  python tests/test_sprite_checks.py   (or: python -m unittest tests.test_sprite_checks)
Sprites are drawn in code; bad sheets are made from good ones by blur, sub-pixel shift, noise, colour jitter, resizing.
"""
import contextlib
import io
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


def draw_frame(seed, i, amp=1):
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
    sw = (0, 1, 2, 1, 0, -1)[i % 6] * amp
    rect(9 + sw, 16, 11 + sw, N - 3, legs)       # legs swing sideways, feet row fixed
    rect(13 - sw, 16, 15 - sw, N - 3, legs)
    rect(7, 9 + abs(sw), 9, 13 + abs(sw), head)  # arms
    rect(15, 9 + abs(sw), 17, 13 + abs(sw), head)
    return im


def up(native):
    return np.kron(native, np.ones((S, S, 1), np.uint8))


def good_frames(seed, amp=1):
    return [up(draw_frame(seed, i, amp)) for i in range(FRAMES)]


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


def on_background(frames, colour=(255, 0, 255), margin=S * 12):
    """Frames composited on a flat opaque colour, on a bigger canvas (the sprite is a small part of it)."""
    out = []
    for f in frames:
        h, w = f.shape[:2]
        canvas = np.zeros((h + 2 * margin, w + 2 * margin, 4), np.uint8)
        canvas[..., :3] = colour
        canvas[..., 3] = 255
        a = f[..., 3:4].astype(np.float32) / 255
        region = canvas[margin:margin + h, margin:margin + w]
        region[..., :3] = (f[..., :3] * a + region[..., :3] * (1 - a)).round().astype(np.uint8)
        out.append(canvas)
    return out


def flip_pixels(frames, share, seed):
    rng = np.random.RandomState(seed)
    out = []
    for f in frames:
        g = f.copy()
        m = (g[..., 3] > 0) & (rng.rand(*g.shape[:2]) < share)
        g[m, :3] = rng.randint(0, 256, size=(int(m.sum()), 3))
        out.append(g)
    return out


def vertical_shift(frames, seed):
    out = list(frames)
    out[2] = np.roll(out[2], -3 * S, axis=0)
    return out


def border_speckle(frames, seed):
    out = []
    for f in frames:
        g = f.copy()
        for bx in range(0, g.shape[1] // S, 2):
            g[0:S, bx * S:(bx + 1) * S] = (200, 30, 30, 255)
        out.append(g)
    return out


def palette_bloat(frames, seed):
    """Grid kept intact, but every native pixel gets its own random colour shift."""
    rng = np.random.RandomState(seed)
    out = []
    for f in frames:
        h, w = f.shape[:2]
        shift = rng.randint(-40, 41, size=(h // S, w // S, 1)).repeat(S, 0).repeat(S, 1)
        g = f.astype(np.int16)
        g[..., :3] += shift
        g[f[..., 3] == 0] = 0
        out.append(np.clip(g, 0, 255).astype(np.uint8))
    return out


def loop_break(frames, seed):
    out = list(frames)
    out[-1] = np.roll(out[-1], 3 * S, axis=1)
    return out


def zoomed(frames, seed):
    out = list(frames)
    k = len(out) // 2
    h, w = out[k].shape[:2]
    big = Image.fromarray(out[k]).resize((int(w * 1.4), int(h * 1.4)), Image.NEAREST)
    out[k] = np.array(big.crop((0, big.height - h, w, big.height)))
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
        good = [sc.score_frames(good_frames(s, 1 + s % 2))["overall"] for s in range(10, 15)]
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


def _score(frames, **kw):
    return sc.score_frames(frames, **kw)


# quality term -> (corruption built to break it, extra score_frames kwargs)
SINGLE_SCORE_BREAKERS = {
    "grid_purity": (blur, dict(scale=S)),
    "grid_stability": (subpixel_shift, {}),
    "colour_flicker": (colour_jitter, {}),
    "palette_size": (palette_bloat, {}),
    "same_character": (colour_jitter, {}),
    "ground": (vertical_shift, {}),
    "zoom": (zoomed, {}),
    "background": (border_speckle, {}),
    "loop": (loop_break, {}),
}


class EachScoreTest(unittest.TestCase):
    def test_each_score_catches_its_own_bad_sheet(self):
        """A score stuck at 1.0 (or stuck at 0) must fail here: good sheets read high, the matching bad sheet low."""
        goods = [_score(good_frames(s, 1 + s % 2))["quality"] for s in range(10, 14)]
        for term, (fn, kw) in SINGLE_SCORE_BREAKERS.items():
            with self.subTest(term):
                for g in goods:
                    self.assertGreaterEqual(g[term], 0.9, f"good sheet scores low on {term}: {g}")
                bad = _score(fn(good_frames(21, 1), 7), **kw)["quality"]
                self.assertLess(bad[term], 0.5, f"{fn.__name__} not caught by {term}: {bad}")

    def test_every_quality_term_is_covered(self):
        terms = set(_score(good_frames(1))["quality"])
        self.assertEqual(terms, set(SINGLE_SCORE_BREAKERS))

    def test_bad_sheets_move_the_raw_numbers_as_named(self):
        g = good_frames(2)
        self.assertFalse(_score(blur(g, 0))["grid_found"])
        r = _score(subpixel_shift(g, 7))
        self.assertEqual(r["grid_purity"], 1.0)
        self.assertLess(r["grid_stability"], 0.5)
        r = _score(colour_jitter(g, 7))
        self.assertGreater(r["colour_flicker"], 0.5)
        self.assertLess(r["same_character"]["palette_overlap_min"], 0.5)
        self.assertGreater(_score(resized_frame(g, 0))["gates"]["zoom_variation"], 0.1)
        self.assertFalse(_score(loop_break(g, 0))["gates"]["loop_ok"])


class BackgroundTest(unittest.TestCase):
    def test_blurred_sprite_scores_the_same_on_magenta_as_on_transparency(self):
        blurred = blur(good_frames(1), 0)
        r_t = _score(blurred, scale=S)
        r_m = _score(on_background(blurred), scale=S)
        self.assertEqual(r_m["background"]["colour"], "#ff00ff")
        self.assertEqual(r_t["background"]["colour"], "transparent")
        self.assertAlmostEqual(r_m["grid_purity"], r_t["grid_purity"], delta=0.05)
        self.assertAlmostEqual(r_m["overall"], r_t["overall"], delta=0.05)

    def test_flipped_pixels_purity_not_inflated_by_background(self):
        flipped = flip_pixels(good_frames(3), 0.1, 1)
        r_t = _score(flipped, scale=S)
        r_m = _score(on_background(flipped), scale=S)
        self.assertLess(r_t["grid_purity"], 0.93)
        self.assertAlmostEqual(r_m["grid_purity"], r_t["grid_purity"], delta=0.01)

    def test_clean_sprite_same_numbers_on_any_background(self):
        g = good_frames(4)
        a = _score(g, scale=S)
        for colour in [(255, 0, 255), (255, 255, 255), (0, 0, 0)]:
            b = _score(on_background(g, colour), scale=S)
            self.assertEqual(b["palette_size"]["sheet"], a["palette_size"]["sheet"])
            self.assertEqual(b["grid_purity"], a["grid_purity"])
            self.assertEqual(b["gates"]["ground_shift"], a["gates"]["ground_shift"])
            self.assertEqual(b["gates"]["zoom_variation"], a["gates"]["zoom_variation"])
            self.assertEqual(b["gates"]["background_clean"], 1.0)
            self.assertAlmostEqual(b["overall"], a["overall"], places=4)

    def test_gates_see_the_sprite_not_the_whole_frame_on_magenta(self):
        r = _score(on_background(resized_frame(good_frames(5), 0)), scale=S)
        self.assertGreater(r["gates"]["zoom_variation"], 0.1)
        r = _score(on_background(vertical_shift(good_frames(5), 0)), scale=S)
        self.assertGreater(r["gates"]["ground_shift"], 2)

    def test_background_option(self):
        frames = on_background(good_frames(6))
        none = _score(frames, background="none")
        self.assertEqual(none["background"]["colour"], "none")
        self.assertGreater(none["palette_size"]["sheet"], _score(frames)["palette_size"]["sheet"])
        given = _score(frames, background="#ff00ff")
        self.assertEqual(given["background"]["mode"], "colour")
        self.assertEqual(given["grid_purity"], _score(frames)["grid_purity"])
        wrong = _score(frames, background="#00ff00")
        self.assertTrue(any("nothing was removed" in w for w in wrong["warnings"]))
        with self.assertRaises(ValueError):
            _score(frames, background="pink")

    def test_background_colour_inside_the_sprite_stays(self):
        g = good_frames(7)
        for f in g:                       # a patch the same colour as the background, inside the body
            f[88:96, 80:88] = (255, 0, 255, 255)
        r = _score(on_background(g), scale=S)
        self.assertEqual(r["palette_size"]["sheet"], _score(g, scale=S)["palette_size"]["sheet"])


class ScaleTest(unittest.TestCase):
    def test_given_scale_that_differs_from_detected_warns(self):
        r = _score(good_frames(1), scale=4)
        self.assertEqual(r["detected_scale"], S)
        self.assertTrue(any("detected" in w and "divides" in w for w in r["warnings"]), r["warnings"])
        r = _score(good_frames(1), scale=16)
        self.assertTrue(any("multiple" in w for w in r["warnings"]), r["warnings"])

    def test_matching_scale_has_no_warning(self):
        self.assertEqual(_score(good_frames(1), scale=S)["warnings"], [])
        self.assertEqual(_score(good_frames(1))["warnings"], [])

    def test_impossible_scales_are_errors(self):
        for bad in (0, -3, 500, 150):  # 150: fits the 192 frame but not the sprite
            with self.subTest(bad):
                with self.assertRaises(ValueError) as cm:
                    _score(good_frames(1), scale=bad)
                self.assertIn("--scale", str(cm.exception))

    def test_cli_warning_on_stderr_and_error_exit(self):
        tool = os.path.join(ROOT, "tools", "sprite_checks.py")
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.png")
            save(to_sheet(good_frames(1)), p)
            r = subprocess.run([sys.executable, tool, p, "--frames", f"{FRAMES}x1", "--scale", "4", "--json"],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0)
            self.assertIn("warning:", r.stderr)
            self.assertTrue(json.loads(r.stdout)["warnings"])
            for bad in ("0", "500", "-2"):
                r = subprocess.run([sys.executable, tool, p, "--frames", f"{FRAMES}x1", "--scale", bad, "--json"],
                                   capture_output=True, text=True)
                self.assertEqual(r.returncode, 2)
                self.assertTrue(r.stderr.startswith("error: --scale"), r.stderr)
                self.assertNotIn("Traceback", r.stderr)

    def test_native_scale_one_sprite(self):
        native = draw_frame(8, 0)
        r = _score([native])
        self.assertEqual(r["grid_scale"], 1)
        self.assertTrue(r["grid_found"])
        self.assertEqual(r["grid_purity"], 1.0)
        self.assertIn("native", r["grid_note"])
        self.assertEqual(r["warnings"], [])
        self.assertGreater(r["overall"], 0.9)
        r = _score([draw_frame(8, i) for i in range(FRAMES)])
        self.assertEqual(r["grid_scale"], 1)
        self.assertEqual(r["grid_stability"], 1.0)
        self.assertEqual(_score([native], scale=1)["grid_purity"], 1.0)

    def test_scale_one_on_a_noisy_image_warns(self):
        r = _score(noise(good_frames(5), 1), scale=1)
        self.assertTrue(any("--scale 1" in w for w in r["warnings"]), r["warnings"])

    def test_noisy_image_is_not_native(self):
        self.assertIsNone(_score(noise(good_frames(5), 1))["grid_scale"])


class OtherFixesTest(unittest.TestCase):
    def test_mixed_size_folder_ground_gate(self):
        """Same feet position after padding at the bottom: ground shift stays 0 although frame heights differ."""
        frames = good_frames(9)
        taller = np.zeros((frames[0].shape[0] + 2 * S, frames[0].shape[1], 4), np.uint8)
        taller[2 * S:] = frames[3]
        frames[3] = taller
        with tempfile.TemporaryDirectory() as d:
            for i, f in enumerate(frames):
                save(f, os.path.join(d, f"f{i + 1}.png"))
            r = sc.run(d, scale=S)
        self.assertLessEqual(r["gates"]["ground_shift"], 0.01)
        self.assertIn("note", r)

    def test_bad_frames_values_are_clear_errors(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.png")
            save(good_frames(1)[0], p)
            for bad in ("0x1", "4", "axb", "1x0", "4x", "1.5x1"):
                with self.subTest(bad):
                    with self.assertRaises(ValueError) as cm:
                        sc.run(p, frames=bad)
                    self.assertIn("--frames must look like 4x1", str(cm.exception))
                    self.assertEqual(sc.main([p, "--frames", bad, "--json"]), 2)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(sc.main([p, "--frames", "1x1", "--json"]), 0)


if __name__ == "__main__":
    unittest.main()
