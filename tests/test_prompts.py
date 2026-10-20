"""CPU tests for the prompt-template nodes: python tests/test_prompts.py"""
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "custom_nodes", "comfyui_workflows_utils"))
sys.modules.setdefault("folder_paths", types.ModuleType("folder_paths"))
import prompts  # noqa: E402

RESULTS = []


def check(cond, msg):
    RESULTS.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + msg)


pp = prompts.PixelArtPrompt()
p, px, c, pal = pp.run("a swordsman", "chibi", "side (facing right)", 24, "auto (from pixels)", 6, "auto", "dark outline", "plain white", True, "")
print(p)
check(p.startswith("Pixel art style.") and "chibi" in p and "side view facing right" in p, "pixel prompt: trigger, style, view")
check("Minimal detail" in p and "24 pixels" in p and "6 flat colours" in p, "pixel prompt: detail from pixels, colour count")
check((px, c, pal) == (24, 6, "auto"), "pixel prompt passes pixels/colors/palette through")
p2 = pp.run("a potion", "item / icon", "as described", 32, "auto (from pixels)", 0, "PICO-8", "no outline", "solid green (chroma key)", False, "glowing")[0]
check("green background" in p2 and "PICO-8 palette" in p2 and not p2.startswith("Pixel art style") and p2.endswith("glowing"), "pixel prompt: preset palette named, no trigger, extra")
check("red" in prompts.palette_phrase(0, "#d82828 #f0f0f0"), "hex palette -> colour names")

vp = prompts.VectorArtPrompt()
r, vc, vpal = vp.run("redraw <image1>", "", "faithful flat redraw", 6, "", "medium", "bold dark outlines", "one shadow + one highlight", "plain white", "")
print(r)
check(r.startswith("Redraw <image1>") and "Do not remove details" in r and "6 flat colours" in r, "vector redraw keeps the generic redraw intent")
g, gc, gpal = vp.run("generate", "a fox head", "logo / emblem", 4, "PICO-8", "minimal", "no outlines", "flat (no shading)", "plain white", "")
print(g)
check("logo" in g and "fox head" in g and "PICO-8" in g and gpal.startswith("000000"), "vector generate: style, subject, preset expanded for the vectorizer")

sp = prompts.SceneryPrompt()
s1, spx, sc, spal, sstep = sp.run("a big gnarled oak tree", "setpiece", 96, 0, "auto", "white", True, "")
print(s1)
check(s1.startswith("Pixel art style.") and "One large big gnarled oak tree" in s1 and "standing on its base" in s1,
      "scenery prompt: trigger and the set-piece template, article stripped")
check("touching the bottom edge" in s1 and "wide horizontal strip" not in s1, "a set piece stands on the bottom edge, not a strip")
check("flat-on side elevation" in s1 and "no plinth" in s1
      and "flat-on side elevation" not in sp.run("hills", "mid", 180, 0, "auto", "white", True, "")[0],
      "a set piece is drawn as a flat-on elevation with no plinth; a layer is not")
check((spx, sc, spal) == (96, 0, "auto"), "scenery prompt passes pixels/colors/palette through")
far = sp.run("mountain range", "far", 180, 0, "auto", "white", True, "", "Fantasy landscape at dusk.")[0]
check("Fantasy landscape at dusk" in far and "distant mountain range along the horizon" in far and "hazy" in far,
      "far layer: the style sentence replaces the generic one, template wording intact")
check("wide horizontal strip" in far and "continues past both the left and the right edge" in far, "a layer is a wide strip")
check("5 flat colours" in far and "3 flat colours" in sp.run("small flat-bottomed peach", "clouds", 180, 0, "auto", "white", True, "")[0],
      "each slot has its own default colour count (far 5, clouds 3)")
check(sp.run("tall grass", "foreground", 180, 7, "auto", "white", True, "")[0].count("7 flat colours") == 1,
      "an explicit colour count wins over the slot default")
key = sp.run("drifting clouds", "clouds", 180, 0, "auto", "magenta (chroma key)", False, "")[0]
check("solid magenta" in key and not key.startswith("Pixel art style"), "the key colour is named in the prompt; the trigger can be off")
check(all(s.endswith("No characters, no text, no frame, no user interface.")
          for s in [sp.run("x", k, 64, 0, "auto", "white", True, "")[0] for k in prompts.SCENERY_SLOTS]),
      "every slot ends with the same exclusions")
check(set(prompts.SCENERY_SLOTS) == {"sky", "clouds", "far", "mid", "near", "foreground", "setpiece"},
      "the seven slots of the design")

# The sky has to be transparent above a layer's skyline or the layer hides the ones behind it, which is the whole
# point of a stack. Two things get it there, and both were measured rather than reasoned about.
check(sstep == 0.0 and all(v == 0.0 for v in prompts.SLOT_SKY_STEP.values()),
      "ClearSky is off for every slot: growing a region down from the top edge ate the content it was meant to "
      "keep - a far layer's mountain, a near layer's trunks, a mid layer's hills")
far_t = sp.run("mountain range", "far", 180, 0, "auto", "white", True, "", "lush green forest, dappled light, pixel art")[0]
check("lush green forest" in far_t, "a layer still gets the full style sentence, setting and all")
for slot in ("mid", "near", "foreground"):
    t = sp.run("x", slot, 180, 0, "auto", "white", True, "")[0]
    check("Nothing behind it" in t and "no sky, no horizon" in t,
          f"{slot} is told outright there is nothing behind the subject (it painted a sky without it)")
check("Nothing behind it" not in sp.run("x", "sky", 180, 0, "auto", "white", True, "")[0],
      "the sky slot is not told there is no sky behind it")

# Krea 2's k2-pixel LoRAs: their own trigger and a bigger subject; the defaults keep today's prompt
kp = pp.run("a knight", "clean game sprite", "front", 64, "auto (from pixels)", 0, "auto", "dark outline", "plain white", True, "", "", "pixel art", 85)[0]
check(kp.startswith("pixel art. A clean") and "filling about 85% of its height" in kp and "small empty margin" in kp
      and "Pixel art style" not in kp, "pixel prompt: another LoRA trigger and a fuller frame")
dp = pp.run("a knight", "clean game sprite", "front", 64, "auto (from pixels)", 0, "auto", "dark outline", "plain white", True, "")[0]
check(dp.startswith("Pixel art style.") and "small in the middle of the canvas, filling about 60% of it, with a wide empty margin" in dp,
      "pixel prompt: defaults unchanged (tarn59 trigger, 60% framing)")

print()
print(f"{sum(RESULTS)}/{len(RESULTS)} passed")
sys.exit(0 if all(RESULTS) else 1)
