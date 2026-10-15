"""Prompt templates: turn a short request ("a swordsman", "a fox logo") into a full prompt for art that converts
cleanly, and pass the same settings (size, colours, palette) on to the converter node (Pixelize / Layered Vectorize),
so the drawing and the conversion agree.

Text-to-image models here run at CFG 1 (Z-Image Turbo) or as edit models (Qwen), so everything is said in the
positive prompt, mostly as what to do rather than what to avoid.
"""
import re

import numpy as np

try:
    from .pixel import PALETTES, parse_palette, rgb_to_lab
except ImportError:                                        # tests import the module directly
    from pixel import PALETTES, parse_palette, rgb_to_lab

# ----------------------------------------------------------------------------- colour names (for palettes in prompts)
COLOR_NAMES = {
    "black": (0, 0, 0), "charcoal": (54, 54, 58), "dark grey": (90, 90, 95), "grey": (128, 128, 128),
    "light grey": (190, 190, 190), "white": (250, 250, 250), "cream": (245, 230, 200), "beige": (215, 195, 160),
    "dark brown": (75, 45, 25), "brown": (130, 80, 40), "tan": (195, 150, 100), "maroon": (100, 20, 30),
    "dark red": (140, 25, 25), "red": (215, 40, 40), "pink": (240, 140, 170), "salmon": (240, 130, 110),
    "orange": (240, 130, 30), "gold": (215, 170, 40), "yellow": (250, 220, 50), "olive": (120, 120, 40),
    "lime green": (150, 210, 60), "green": (50, 150, 60), "dark green": (25, 80, 40), "teal": (30, 130, 130),
    "cyan": (60, 200, 220), "sky blue": (120, 180, 240), "blue": (40, 90, 200), "navy": (25, 35, 90),
    "purple": (120, 60, 170), "violet": (170, 110, 220), "magenta": (200, 50, 160), "skin tone": (240, 190, 150),
    "steel grey": (150, 165, 185),
}
_NAMES = list(COLOR_NAMES)
_NAME_LAB = rgb_to_lab(np.array([COLOR_NAMES[n] for n in _NAMES], np.uint8))


def color_names(palette):
    """Nearest plain colour name for each palette colour, duplicates removed, order kept."""
    out = []
    for lab in rgb_to_lab(np.asarray(palette, np.uint8)):
        n = _NAMES[int(np.sqrt(((_NAME_LAB - lab) ** 2).sum(-1)).argmin())]
        if n not in out:
            out.append(n)
    return out


def palette_phrase(colors, palette):
    """How the colours are described in the prompt."""
    p = (palette or "").strip()
    if p.lower() not in ("", "auto", "sprite"):
        name = next((k for k in PALETTES if k.lower() == p.lower()), None)
        cols = parse_palette(p)
        names = color_names(cols) if cols is not None else []
        if name:
            return f"only colours from the {name} palette ({', '.join(names[:8])})"
        if names:
            return f"only these colours: {', '.join(names)}"
    if colors and colors > 0:
        return f"a strictly limited palette of {colors} flat colours"
    return "a small limited palette of flat colours"


def _join(parts):
    text = " ".join(p.strip() for p in parts if p and p.strip())
    return re.sub(r"\s+", " ", text).replace(" .", ".").strip()


# ----------------------------------------------------------------------------- shared
BACKGROUNDS = {
    "plain white": "Isolated on a plain flat pure white background.",
    "plain light grey": "Isolated on a plain flat light grey background.",
    "solid green (chroma key)": "Isolated on a plain flat solid bright green background.",
    "solid magenta (chroma key)": "Isolated on a plain flat solid magenta background.",
}
NO_SHADOW = "No drop shadow, no cast shadow, no ground shadow or ellipse under it, no floor, no text."


# ----------------------------------------------------------------------------- pixel art
PIXEL_STYLES = {
    "clean game sprite": "A clean stylized video game sprite with simple readable shapes and a bold silhouette.",
    "16-bit RPG": "A 16-bit console era RPG sprite, like classic SNES role-playing games, with clear shapes and a "
                  "strong readable silhouette.",
    "8-bit retro": "An 8-bit NES era sprite with very simple blocky shapes and chunky proportions.",
    "chibi": "A cute chibi game character: an oversized head about a third of its height with a clearly drawn face, big "
             "eyes and hair or a hat, a small body and short limbs, simple readable shapes.",
    "top-down RPG": "A top-down RPG overworld sprite seen from a high three-quarter angle, compact and simple.",
    "side-scroller": "A side-scrolling platformer game sprite in clear side view, compact and readable.",
    "isometric": "An isometric game asset viewed from a 30 degree isometric angle, simple geometric shapes.",
    "item / icon": "A single game item icon, a simple centred object with an iconic, instantly readable shape.",
    "monster / creature": "A video game monster sprite with a distinctive, simple, readable silhouette.",
    "prop / tile": "A game environment prop with simple shapes and a clear outline, as used in tile-based games.",
    "terrain tile (seamless)": "A seamless, tileable top-down terrain texture for a tile-based game, seen straight from "
                               "above, filling the whole image edge to edge.",
    "none": "",
}
PIXEL_VIEWS = {
    "front three-quarter": "front three-quarter view, facing the viewer, face visible",
    "front": "front view, facing the viewer, face visible",
    "side (facing right)": "side view facing right",
    "back": "back view",
    "top-down": "top-down view",
    "isometric": "isometric view",
    "as described": "",
}
DETAIL_TEXT = {
    "minimal": "Minimal detail: big simple shapes, keeping only the key features as simple solid shapes (a simple "
               "face with clear dot eyes, hands, the main prop). No tiny parts, no thin lines, no texture.",
    "low": "Low detail: big simple shapes plus a few key details (a readable face with clear eyes, hands, belt, the "
           "main prop). No tiny parts or fine texture.",
    "medium": "Moderate detail: clear main shapes, a readable face, and a handful of small details.",
    "high": "Rich but clean detail, every detail still a clear solid shape.",
}


def pixel_detail(detail, pixels):
    if detail != "auto (from pixels)":
        return detail
    return "minimal" if pixels <= 24 else "low" if pixels <= 48 else "medium" if pixels <= 96 else "high"


class PixelArtPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "subject": ("STRING", {"multiline": True, "default": "a swordsman in a red tunic holding a long steel sword",
                                   "tooltip": "What to draw (one subject). The style, view, colours and detail are added for you."}),
            "style": (list(PIXEL_STYLES), {"tooltip": "Game-art style template."}),
            "view": (list(PIXEL_VIEWS), {"tooltip": "Camera angle. 'as described' = say it in the subject."}),
            "pixels": ("INT", {"default": 64, "min": 8, "max": 512, "tooltip": "Target subject height in art pixels; sets the detail level (auto) and is passed on to Pixelize."}),
            "detail": (["auto (from pixels)", "minimal", "low", "medium", "high"], {"tooltip": "auto: <=24 px minimal, <=48 low, <=96 medium, larger high."}),
            "colors": ("INT", {"default": 0, "min": 0, "max": 64, "tooltip": "Palette size asked for in the prompt and passed on to Pixelize (0 = automatic fewest colours)."}),
            "palette": ("STRING", {"default": "auto", "tooltip": "auto, a preset (PICO-8, DB32, Endesga-32, GameBoy) or hex colours. Named in the prompt and passed on to Pixelize."}),
            "outline": (["dark outline", "no outline", "coloured outline"], {"tooltip": "Outline style asked for."}),
            "background": (list(BACKGROUNDS), {"tooltip": "Background to draw on (removed afterwards). A colour far from the subject's colours separates best; white is the most reliable for this model."}),
            "lora_trigger": ("BOOLEAN", {"default": True, "tooltip": "Start with 'Pixel art style.' (the pixel-art LoRA's trigger)."}),
            "extra": ("STRING", {"multiline": True, "default": "", "tooltip": "Anything else to add at the end."}),
        }, "optional": {
            "style_text": ("STRING", {"multiline": True, "default": "", "tooltip": "Custom style sentence; replaces the style template when not empty (used by Sprite Studio's style library)."}),
        }}

    RETURN_TYPES = ("STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("prompt", "pixels", "colors", "palette")
    FUNCTION = "run"
    CATEGORY = "comfyui_workflows/prompts"
    DESCRIPTION = "Build a prompt for simple, easily pixelated game art from a short subject, and pass size/colours on to Pixelize."

    def run(self, subject, style, view, pixels, detail, colors, palette, outline, background, lora_trigger, extra,
            style_text=""):
        d = pixel_detail(detail, pixels)
        outline_text = {"dark outline": "Thick, bold, dark outlines around the silhouette and the main shapes.",
                        "no outline": "No outlines; shapes are separated by colour only.",
                        "coloured outline": "Thin outlines in a darker shade of each colour."}[outline]
        v = PIXEL_VIEWS[view]
        if style == "terrain tile (seamless)":
            return (_join([
                "Pixel art style." if lora_trigger else "",
                f"Overhead photo looking straight down at {subject.strip().rstrip('.')}.",
                "The surface fills the entire frame edge to edge, evenly lit, natural and irregular, with clear "
                "medium-sized detail and good contrast, no single dominant object, no horizon, no sky, no text.",
                extra,
            ]), pixels, colors, palette)
        prompt = _join([
            "Pixel art style." if lora_trigger else "",
            (style_text or "").strip() or PIXEL_STYLES[style],
            f"Subject: {subject.strip().rstrip('.')}. Exactly one subject, drawn once.",
            (f"Full body, {v}, centred. " if v else "Full body, centred. ") + "Show the complete subject small in the middle of the canvas, filling about 60% of it, with a wide empty margin on every side; nothing touches or is cut off by the edges.",
            f"Designed to read clearly at about {pixels} pixels tall.", DETAIL_TEXT[d],
            f"Flat solid colours, {palette_phrase(colors, palette)}, simple cel shading with at most one shadow "
            "and one highlight tone per colour, no gradients, no texture, no noise, no dithering, crisp hard edges.",
            outline_text,
            BACKGROUNDS[background], NO_SHADOW,
            extra,
        ])
        return (prompt, pixels, colors, palette)


# ----------------------------------------------------------------------------- vector art (SVG)
VECTOR_STYLES = {
    "faithful flat redraw": "a clean, simplified flat illustration that is easy to vectorize, keeping every element "
                            "clearly recognizable with the same composition, pose, proportions, outlines and details",
    "flat vector illustration": "a flat vector illustration with clean geometric shapes",
    "cartoon character": "a cartoon character illustration with bold clean shapes and expressive simple forms",
    "logo / emblem": "a simple, bold logo emblem with strong geometric shapes, balanced and symmetrical where it fits",
    "icon": "a single icon of one simple object made of a few bold shapes, drawn on its own with no tile, button or "
            "frame behind it",
    "sticker": "a die-cut sticker design with bold shapes and a thick white border around the whole subject",
    "line art": "clean black line art like a colouring book page, even line weight, closed shapes, no fills",
    "game item / UI art": "a polished video game item illustration with simple bold shapes and clear edges, the object "
                          "alone on the plain background",
    "game character (stylized)": "a stylized game character in a modern flat art style with simple readable shapes",
    "none": "",
}


class VectorArtPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mode": (["generate", "redraw <image1>"], {"tooltip": "generate = text-to-image (prompt-to-svg). redraw = instruction for an edit model to repaint the input image (layered-svg-ai)."}),
            "subject": ("STRING", {"multiline": True, "default": "", "tooltip": "What to draw (generate), or specifics for the redraw (e.g. 'spiders dark brown'). Optional for redraw."}),
            "style": (list(VECTOR_STYLES), {"tooltip": "Vector-friendly style template."}),
            "colors": ("INT", {"default": 6, "min": 2, "max": 16, "tooltip": "Number of base colours asked for and passed on to Layered Vectorize."}),
            "palette": ("STRING", {"default": "", "tooltip": "Optional hex colours or a preset name; named in the prompt and passed on to Layered Vectorize. Empty = automatic."}),
            "detail": (["minimal", "low", "medium", "high"], {"default": "medium", "tooltip": "Fewer details = fewer, larger SVG shapes."}),
            "outline": (["bold dark outlines", "thin dark outlines", "no outlines"], {"tooltip": "Outlines become the SVG ink layer."}),
            "shading": (["flat (no shading)", "one shadow + one highlight", "keep original"], {"default": "one shadow + one highlight", "tooltip": "Hard-edged tones become the SVG shadow/highlight layers."}),
            "background": (list(BACKGROUNDS), {"tooltip": "Background to draw on (removed by the vectorizer). White is the most reliable."}),
            "extra": ("STRING", {"multiline": True, "default": "", "tooltip": "Anything else to add at the end."}),
        }}

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("prompt", "colors", "palette")
    FUNCTION = "run"
    CATEGORY = "comfyui_workflows/prompts"
    DESCRIPTION = "Build a prompt for flat, easily vectorized art (generate or redraw), and pass colours/palette on to Layered Vectorize."

    def run(self, mode, subject, style, colors, palette, detail, outline, shading, background, extra):
        redraw = mode.startswith("redraw")
        style_text = VECTOR_STYLES[style]
        detail_text = {"minimal": "Use as few shapes as possible: only the essential large shapes, no small details.",
                       "low": "Keep only the main shapes and a few key details.",
                       "medium": "Keep the main shapes and the important details." if not redraw else
                                 "Do not remove details or small objects, but simplify their shapes.",
                       "high": "Keep all details, each as a clean solid shape."}[detail]
        shading_text = {"flat (no shading)": "Each element is one flat colour with no shading at all.",
                        "one shadow + one highlight": "Use at most one simple, hard-edged shadow and one simple, "
                                                       "hard-edged highlight per base colour.",
                        "keep original": "Keep the original lighting, but as hard-edged flat tones."}[shading]
        outline_text = {"bold dark outlines": "Bold, clean, dark outlines.",
                        "thin dark outlines": "Thin, clean, dark outlines.",
                        "no outlines": "No outlines."}[outline]
        colour_text = (f"Simplify the colours to {palette_phrase(colors, palette)}: give each element one flat base "
                       "colour, and give similar or repeated elements (the same material, part or kind of object) the "
                       "same base colour.")
        if redraw:
            prompt = _join([
                f"Redraw <image1> as {style_text}." if style_text else "Redraw <image1> as flat art that is easy to vectorize.",
                "Identify each distinct element and object in the image and keep every one clearly recognizable.",
                colour_text, shading_text,
                "Remove gradients, texture, noise and subtle colour variation.", detail_text, outline_text,
                subject.strip(), BACKGROUNDS[background], NO_SHADOW, extra,
            ])
        else:
            prompt = _join([
                f"{style_text[0].upper() + style_text[1:]}." if style_text else "",
                f"Subject: {subject.strip().rstrip('.')}." if subject.strip() else "",
                "Centred. Show the complete subject small in the middle of the canvas, filling about 60% of it, with a wide empty margin on every side; nothing touches or is cut off by the edges.",
                colour_text, shading_text,
                "Smooth clean curves, no gradients, no texture, no noise, no photographic detail.", detail_text,
                outline_text, BACKGROUNDS[background], NO_SHADOW, extra,
            ])
        pal = palette.strip()
        name = next((k for k in PALETTES if k.lower() == pal.lower()), None)
        if name:
            pal = PALETTES[name]
        return (prompt, colors, pal)


# ----------------------------------------------------------------------------- scenery (parallax layers, set pieces)
# PixelArtPrompt asks for one centred subject with a wide empty margin, which is the opposite of what a background
# layer needs: a strip that runs edge to edge and is cut off deliberately. Scenery therefore gets its own templates,
# one per depth slot, from the pixel-scenery design (workspace card "Pixel Scenery").
#
# {style} is the style sentence, {subject} the user's words and {key} the flat colour that is removed afterwards
# (white by default). The "no ..." parts are written into the prompt rather than kept as a negative prompt, because
# Z-Image Turbo runs at CFG 1 and never sees one.
SCENERY_SLOTS = {
    "sky": "{style}. Only the sky: {subject}. No ground, no land, no objects. Wide horizontal bands of colour.",
    "clouds": "{style}. A row of {subject} clouds spread across the full width, side view. Plain flat {key} everywhere else.",
    "far": "{style}. A distant {subject} along the horizon, side view, spanning the full width. Low contrast, hazy, "
           "pale colours close to the sky, very little detail. Plain flat {key} above the skyline.",
    "mid": "{style}. {subject} at middle distance, side view, spanning the full width, medium detail and contrast. "
           "Plain flat {key} above the skyline. Nothing behind it: no sky, no horizon, no distant scenery, only plain flat {key}.",
    "near": "{style}. {subject} close by, side view, along the bottom of the image, strong contrast, dark values, "
            "high detail. Plain flat {key} above. Nothing behind it: no sky, no horizon, no distant scenery, only plain flat {key}.",
    "foreground": "{style}. Dark silhouettes of {subject} along the bottom edge, almost black, big simple shapes, "
                  "no detail. Plain flat {key} above. Nothing behind it: no sky, no horizon, no distant scenery, only plain flat {key}.",
    # the subject leads here, unlike every layer slot: a scenery style sentence names a scene ("lush green
    # forest"), and in front of a small subject it replaces it - a 48 px "mossy grey boulder" came back a tree
    "setpiece": "One large {subject}, side view, the whole object in frame, standing on its base. "
                "Drawn in this style: {style}. Plain flat {key} background.",
}
SCENERY_NO = "No characters, no text, no frame, no user interface."
# How many flat colours each slot is asked for when `colors` is left at 0 (automatic). Fewer colours further back is
# what gives the atmospheric perspective: far layers sit close to the sky's values, near layers carry the contrast.
SLOT_COLORS = {"sky": 6, "clouds": 3, "far": 5, "mid": 9, "near": 10, "foreground": 3, "setpiece": 0}
# ClearSky's `step` per slot. All zero: the node is off by default, and this is why.
#
# Growing a region down from the top edge until it stops changing slowly cannot tell a skyline from a layer's own
# internal contrast, and every slot was damaged by it. On `far`, whose mountains are ASKED to be "hazy, pale
# colours close to the sky", a step big enough to clear the haze walked straight through the mountain and left a
# jagged outline. On `near` it leaked between the pine trunks and carved transparent holes through the
# silhouettes while leaving the sky on the other half. On `mid` it gouged triangles out of the hills.
#
# The sky is cleared by asking for a flat key colour and removing it, as the design originally said. What was
# actually wrong was the style sentence: it named the sky ("purple and peach sky"), so the model painted one over
# the key colour. The look-only style sentence fixes that, and the slot templates now say outright that there is
# nothing behind the subject.
#
# The node stays because it is the right tool when a sky really is smooth and the content really is contrasty -
# a hand-made layer, a photograph - but nothing asks for it by default.
SLOT_SKY_STEP = {"sky": 0.0, "clouds": 0.0, "far": 0.0, "mid": 0.0, "near": 0.0, "foreground": 0.0,
                 "setpiece": 0.0}
# The flat colour drawn where nothing should be, named in the prompt and filled from the edges afterwards.
KEY_COLORS = {"white": "pure white", "light grey": "light grey", "green (chroma key)": "solid bright green",
              "magenta (chroma key)": "solid magenta"}
# A layer is a wide strip; a set piece is one object on a plain background. Only the strip needs the edge-to-edge
# wording, so it is added for every slot but "setpiece".
SCENERY_WIDE = ("The image is a wide horizontal strip: the scene runs the full width and continues past both the "
                "left and the right edge, with no border, no vignette and nothing centred as a single subject.")
SCENERY_GROUND = ("The object stands on flat ground at the very bottom of the image, its base touching the bottom "
                  "edge, nothing cut off at the sides or the top. It rests directly on the ground with no plinth, "
                  "pedestal or platform beneath it. "
                  # "Side view" alone is not enough for anything with flat faces: the model draws buildings at a
                  # three-quarter angle, and a cut-out cube with one unlit side face reads as a wide block on a
                  # narrow plinth. Measured on the castle preset; the organic presets are unaffected by it.
                  "Strict flat-on side elevation, seen straight from the front at eye level: no perspective, no "
                  "three-quarter angle, no visible side faces, no view from above.")


class SceneryPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "subject": ("STRING", {"multiline": True, "default": "mountain range with snow on the peaks",
                                   "tooltip": "What this layer or object is. The slot wording, colours and detail are added for you."}),
            "slot": (list(SCENERY_SLOTS), {"default": "far", "tooltip": "Which depth slot: sky / clouds / far / mid / near / foreground are parallax layers; setpiece is one large object with a ground line."}),
            "pixels": ("INT", {"default": 180, "min": 8, "max": 1024, "tooltip": "Final size in art pixels (a layer's height, a set piece's longest side); sets the detail level and is passed on to Pixelize."}),
            "colors": ("INT", {"default": 0, "min": 0, "max": 64, "tooltip": "Flat colours asked for and passed on to Pixelize. 0 = the slot's default (3 for clouds and foreground, 10 for near)."}),
            "palette": ("STRING", {"default": "auto", "tooltip": "auto, a preset (PICO-8, DB32, Endesga-32, GameBoy) or hex colours. Named in the prompt and passed on to Pixelize."}),
            "key_color": (list(KEY_COLORS), {"tooltip": "The flat colour to draw where nothing should be (removed afterwards by filling from the edges)."}),
            "lora_trigger": ("BOOLEAN", {"default": True, "tooltip": "Start with 'Pixel art style.' (the pixel-art LoRA's trigger)."}),
            "extra": ("STRING", {"multiline": True, "default": "", "tooltip": "Anything else to add at the end."}),
        }, "optional": {
            "style_text": ("STRING", {"multiline": True, "default": "", "tooltip": "Scenery style sentence; replaces the generic one (used by Sprite Studio's style library)."}),
        }}

    RETURN_TYPES = ("STRING", "INT", "INT", "STRING", "FLOAT")
    RETURN_NAMES = ("prompt", "pixels", "colors", "palette", "sky_step")
    FUNCTION = "run"
    CATEGORY = "comfyui_workflows/prompts"
    DESCRIPTION = ("Build a prompt for a pixel-art background layer or set piece, and pass size, colours and the "
                   "slot's sky-clearing sensitivity on to Pixelize and Clear Sky.")

    DEFAULT_STYLE = ("16-bit side-scrolling platformer background, SNES-era pixel art, clean readable shapes, "
                     "limited palette")

    def run(self, subject, slot, pixels, colors, palette, key_color, lora_trigger, extra, style_text=""):
        slot = slot if slot in SCENERY_SLOTS else "far"
        n = int(colors) if int(colors) > 0 else SLOT_COLORS.get(slot, 0)
        style = (style_text or "").strip().rstrip(".") or self.DEFAULT_STYLE
        body = SCENERY_SLOTS[slot].format(style=style, subject=_subject_text(subject),
                                          key=KEY_COLORS.get(key_color, "pure white"))
        prompt = _join([
            "Pixel art style." if lora_trigger else "",
            body,
            SCENERY_GROUND if slot == "setpiece" else SCENERY_WIDE,
            f"Designed to read clearly at about {int(pixels)} pixels.",
            f"Flat solid colours, {palette_phrase(n, palette)}, simple cel shading with hard edges, no gradients, "
            "no texture, no noise, no dithering.",
            SCENERY_NO,
            extra,
        ])
        return (prompt, int(pixels), n, palette, SLOT_SKY_STEP.get(slot, 14.0))


def _subject_text(text):
    """The subject as the slot templates want it: tidied and without a leading article, because most of them put
    their own words in front of it ("One large {subject}", "A distant {subject}", "Dark silhouettes of {subject}")."""
    t = " ".join(str(text or "").split()).strip().rstrip(".")
    return re.sub(r"^(?:a|an|the)\s+", "", t, flags=re.I)
