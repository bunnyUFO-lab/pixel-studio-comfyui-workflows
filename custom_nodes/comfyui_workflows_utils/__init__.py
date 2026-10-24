"""Small utility nodes used by the workflows in this repo (mounted into ComfyUI by docker-compose.yml)."""
import json
import re

SHAPE_TAGS = re.compile(r"<(path|rect|circle|ellipse|polygon|polyline|line)\b")
FILL_COLORS = re.compile(r'fill="(#[0-9A-Fa-f]{3,8}|rgb\([^)]*\))"')
METADATA_WRAPPER = 109  # bytes the core Save SVG node adds around the embedded JSON
ROLE_FILL = re.compile(r'<g\b[^>]*\bdata-role="(\w+)"[^>]*\bfill="(#[0-9A-Fa-f]{3,8})"')
SVG_SIZE = re.compile(r'<svg[^>]*\bwidth="([\d.]+)[^"]*"[^>]*\bheight="([\d.]+)[^"]*"')


def human_size(n):
    return f"{n / 1024 / 1024:.2f} MB" if n >= 1024 * 1024 else f"{n / 1024:.0f} KB"


class SVGStats:
    """Reports how many shapes an SVG has and how big the saved file will be."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"SVG_String": ("STRING", {"forceInput": True})},
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING", "INT", "FLOAT")
    RETURN_NAMES = ("info", "shapes", "size_kb")
    FUNCTION = "stats"
    CATEGORY = "comfyui_workflows/svg"
    DESCRIPTION = "Counts the shapes and colors in an SVG string and estimates the saved file size."

    def stats(self, SVG_String, prompt=None, extra_pnginfo=None):
        svg = SVG_String
        shapes = len(SHAPE_TAGS.findall(svg))
        colors = len({c.lower() for c in FILL_COLORS.findall(svg)})
        drawing = len(svg.encode("utf-8"))

        # The core Save SVG node (comfy_extras/nodes_images.py) embeds the prompt + workflow as
        # json.dumps(..., indent=2) inside a fixed <metadata><![CDATA[...]]> wrapper; mirror it.
        metadata = {}
        if prompt is not None:
            metadata["prompt"] = prompt
        if isinstance(extra_pnginfo, dict):
            metadata.update(extra_pnginfo)
        embedded = len(json.dumps(metadata, indent=2).encode("utf-8")) + METADATA_WRAPPER if metadata else 0

        lines = [f"Shapes: {shapes:,}"]
        roles = {}
        for role, fill in ROLE_FILL.findall(svg):            # Layered Vectorize output: colours by role
            roles.setdefault(role, set()).add(fill.lower())
        if roles:
            base = roles.get("base", set()) | roles.get("ink", set()) | roles.get("detail", set())
            shades = (roles.get("shadow", set()) | roles.get("highlight", set())) - base
            lines.append(f"Base colours: {len(base)}" + (f" (+ {len(shades)} shadow/highlight shades)" if shades else ""))
        else:
            lines.append(f"Colors: {colors:,}")
        lines.append(f"File size: ~{human_size(drawing + embedded)}")
        if embedded:
            lines.append(f"  (drawing {human_size(drawing)} + embedded workflow {human_size(embedded)})")
        size = SVG_SIZE.search(svg)
        if size:
            lines.append(f"Canvas: {float(size.group(1)):.0f} x {float(size.group(2)):.0f} px")
        return ("\n".join(lines), shapes, round((drawing + embedded) / 1024, 1))


from .edit import CropForMaskedEdit, MaskedEditComposite  # noqa: E402
from .layered import LayeredVectorize  # noqa: E402
from .pixel import (CleanCutout, ClearSky, CropToStrip, GridCutout, MakeTileable, PixelArtCheck, Pixelize, RestoreSpriteCanvas, PrepareSpriteForVideo, RepixelateFrames, SaveDirections,  # noqa: E402
                    SaveSpriteSheet, SpriteAnimationPreview, SpriteVideoMask)
from .prompts import PixelArtPrompt, SceneryPrompt, VectorArtPrompt  # noqa: E402
from .video import JoinVideoSegments, ResizeForWan, SegmentPrompt, VideoAnchorAndTail, WanSVIProConditioning  # noqa: E402

NODE_CLASS_MAPPINGS = {
    "SVGStats": SVGStats,
    "LayeredVectorize": LayeredVectorize,
    "WanSVIProConditioning": WanSVIProConditioning,
    "SegmentPrompt": SegmentPrompt,
    "ResizeForWan": ResizeForWan,
    "VideoAnchorAndTail": VideoAnchorAndTail,
    "JoinVideoSegments": JoinVideoSegments,
    "MaskedEditComposite": MaskedEditComposite,
    "CropForMaskedEdit": CropForMaskedEdit,
    "Pixelize": Pixelize,
    "CleanCutout": CleanCutout,
    "GridCutout": GridCutout,
    "RestoreSpriteCanvas": RestoreSpriteCanvas,
    "MakeTileable": MakeTileable,
    "CropToStrip": CropToStrip,
    "ClearSky": ClearSky,
    "PrepareSpriteForVideo": PrepareSpriteForVideo,
    "SpriteVideoMask": SpriteVideoMask,
    "RepixelateFrames": RepixelateFrames,
    "SaveSpriteSheet": SaveSpriteSheet,
    "SpriteAnimationPreview": SpriteAnimationPreview,
    "PixelArtCheck": PixelArtCheck,
    "SaveDirections": SaveDirections,
    "PixelArtPrompt": PixelArtPrompt,
    "SceneryPrompt": SceneryPrompt,
    "VectorArtPrompt": VectorArtPrompt,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SVGStats": "SVG Stats (shapes, file size)",
    "LayeredVectorize": "Layered Vectorize (flats, tones, ink, details)",
    "WanSVIProConditioning": "Wan SVI 2.0 Pro Conditioning (extend video)",
    "SegmentPrompt": "Segment Prompt (one line per segment)",
    "ResizeForWan": "Resize for Wan (longest side, /16)",
    "VideoAnchorAndTail": "Video Anchor + Tail Frames",
    "JoinVideoSegments": "Join Video Segments (overlap blend)",
    "MaskedEditComposite": "Masked Edit Composite (optional inpaint)",
    "CropForMaskedEdit": "Crop For Masked Edit (enlarge masked area)",
    "Pixelize": "Pixelize (true pixel art)",
    "CleanCutout": "Clean Cutout (drop shadow, background pockets)",
    "GridCutout": "Grid Cutout (sprite from on-grid pixel art + cut-out)",
    "RestoreSpriteCanvas": "Restore Sprite Canvas (original size)",
    "MakeTileable": "Make Tileable (seamless pixel tile or sideways strip)",
    "CropToStrip": "Crop To Strip (full-width band for a parallax layer)",
    "ClearSky": "Clear Sky (transparent down to the skyline)",
    "PrepareSpriteForVideo": "Prepare Pixel Sprite for Video",
    "SpriteVideoMask": "Sprite Video Mask (sprite silhouette)",
    "RepixelateFrames": "Re-pixelate Frames (same grid + palette)",
    "SaveSpriteSheet": "Save Sprite Sheet (PNG + Aseprite JSON)",
    "SpriteAnimationPreview": "Sprite Animation Preview (playback)",
    "PixelArtCheck": "Pixel Art Check (game-ready)",
    "SaveDirections": "Save Directional Views (mirror to 8)",
    "PixelArtPrompt": "Pixel Art Prompt (game-art template)",
    "SceneryPrompt": "Scenery Prompt (background layer / set piece)",
    "VectorArtPrompt": "Vector Art Prompt (SVG-friendly template)",
}

from . import cleanup  # noqa: E402,F401  - registers POST /spritestudio/forget (Sprite Studio's job cleanup)
