"""Regression pin for tools/sprite_checks.py: transparent-background sprites with an explicit --scale.

The numbers in tests/sprite_checks_pinned.json were taken from the first version of the tool (PR #23 as first
merged for review). They are used to compare training runs, so they must not move: this test must keep passing
unchanged. Keys added to the JSON output later are ignored; every pinned key must stay identical.

Regenerate only on purpose:  python tests/test_sprite_checks_pinned.py --write
"""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
import sprite_checks as sc  # noqa: E402
import test_sprite_checks as T  # noqa: E402

PIN = os.path.join(HERE, "sprite_checks_pinned.json")


def cases():
    """name -> (array, run() keyword arguments). All transparent background, explicit scale 8."""
    return {
        "good_sheet": (T.to_sheet(T.good_frames(31)), dict(scale=8, frames="6x1")),
        "blurred_sheet": (T.to_sheet(T.blur(T.good_frames(32), 0)), dict(scale=8, frames="6x1")),
        "shifted_single": (T.subpixel_shift(T.good_frames(33), 5)[2], dict(scale=8)),
    }


def compute():
    out = {}
    with tempfile.TemporaryDirectory() as d:
        for name, (arr, kw) in cases().items():
            p = os.path.join(d, name + ".png")
            T.save(arr, p)
            r = sc.run(p, **kw)
            r.pop("input")
            out[name] = r
    return json.loads(json.dumps(out))


class PinnedTest(unittest.TestCase):
    def test_numbers_unchanged(self):
        with open(PIN) as f:
            pinned = json.load(f)
        now = compute()
        for name, old in pinned.items():
            for key, val in old.items():
                with self.subTest(case=name, key=key):
                    self.assertEqual(now[name][key], val)


if __name__ == "__main__":
    if "--write" in sys.argv:
        with open(PIN, "w") as f:
            json.dump(compute(), f, indent=1, sort_keys=True)
        print("wrote", PIN)
    else:
        unittest.main()
