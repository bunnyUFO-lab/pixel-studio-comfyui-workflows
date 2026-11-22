"""Build the opaque environment master and the two reference-conditioned asset workflows."""
from build_pixel_qwen2511 import encoders, model_chain, models, sampler
from comfy_workflow import (BIREFNET, BIREFNET_URL, BLUE, GREEN, PIXEL_LORA, PIXEL_LORA_URL, WORKFLOWS,
                            YELLOW, ZIMAGE_HF, ZIMAGE_MODEL, ZIMAGE_TE, ZIMAGE_VAE, Graph, inp, out)

ROOT = WORKFLOWS / "pixel-art"
MASTER_PROMPT = (
    "Pixel art style. An original wide side-on background for a 16 bit adventure game: Cinder Coast at late "
    "afternoon, with cool turquoise water, terracotta sandstone sea cliffs, and small muted olive shrubs. "
    "The upper third is an open warm pale peach sky with a few broad flat cloud bands. The middle third has a "
    "layered receding shoreline and graphic cliff silhouettes, clear large shapes. Light from the upper left. "
    "The lower quarter is calmer blue-green water and low reddish reefs with no foreground objects. Full-bleed "
    "wide game environment filling the frame edge to edge, flat deliberate pixel clusters, no UI, frame, text, "
    "characters, gradients or lettering."
)
DETAILED_MASTER_PROMPT = (
    "Pixel art style. Painterly pixel-art environment for a side-view game. Cinder Coast at late afternoon: "
    "a wide terracotta sandstone coastline bordering a cool turquoise sea, seen at eye level. A near cliff at "
    "the left edge frames the view with rough layered stone and sparse olive shrubs. A curved middle-distance "
    "shoreline recedes toward a distant hazy island silhouette on the right. Upper-left sunlight cuts across "
    "the irregular shelves, with warm rock highlights, cool violet shadows and softer blue-grey distant land. "
    "Use softened organic contours and nuanced, stepped colour ramps. Shape the rock with irregular clustered "
    "brushwork, and keep only the focal silhouette edges crisp. Leave a quiet lower quarter for a playable "
    "foreground and keep the horizon readable. Full-bleed opaque background, no text, frame, UI, characters, "
    "blur, or airbrushed gradients."
)
PROP_PROMPT = (
    "Use the whole input landscape as an art style and material reference, not as the output composition. "
    "Draw only one squat terracotta sandstone waymarker with a turquoise enamel inset and a tuft of olive "
    "plants at the base, viewed straight from the side. Match the input's chunky pixel clusters, terracotta "
    "and turquoise colours and upper-left light. Show the entire object with margins on flat solid white, "
    "with no scenery, horizon, ground plane, cast shadow, text or border."
)
MATERIAL_PROMPT = (
    "Use the input landscape only as a visual style and material reference. Make a square, edge-to-edge swatch "
    "of that terracotta sandstone as a game ground texture, irregular tiny rust and salmon stone pixels, "
    "shallow horizontal grain, even lighting, subtle contrast. The only thing in frame is the material "
    "surface: flat orthographic texture, no perspective, objects, skyline, typography, gradients or borders."
)
QUIET_MATERIAL_PROMPT = (
    "Use the input landscape for the palette, material and shading. Draw one matte sandstone surface seen directly "
    "from above, filling the square edge to edge. Use broad painterly patches with subtle stepped shades, "
    "shallow irregular cracks and a few larger stone chips. Keep the contrast low so sprites and path markers "
    "remain easy to read. Avoid scattered single-pixel specks, evenly spaced decoration, hard lines along outer "
    "edges, highlights that look wet, perspective, objects, text, and borders."
)


def save_image(g, id_, x, y, source, prefix, title, slot=0):
    g.node(id_, "SaveImage", [x, y], [320, 300], inputs=[inp("images", "IMAGE")],
           widgets=["pixel/" + prefix], title=title, color=GREEN)
    g.link(source, slot, id_, 0)


def pixelize(g, id_, x, y, source, *, title, size, measure, colors, background, edge_cleanup, smooth=0.0):
    g.node(id_, "Pixelize", [x, y], [350, 440], inputs=[inp("images", "IMAGE")],
           outputs=[out("pixel_art", "IMAGE"), out("preview", "IMAGE"), out("palette", "STRING"), out("info", "STRING")],
           widgets=[size, measure, "median", smooth, colors, 20.0, "auto", True, background, 12.0,
                    False, edge_cleanup, "none", 0, 2],
           title=title, color=YELLOW)
    g.link(source, 0, id_, 0)


def preview_text(g, id_, x, y, source, slot, title):
    g.node(id_, "PreviewAny", [x, y], [320, 175], inputs=[inp("source", "*")], title=title)
    g.link(source, slot, id_, 0)


def build_master(detailed=False):
    g = Graph()
    node, link = g.node, g.link
    suffix = "-detailed" if detailed else ""
    prefix = "environment_master_detailed" if detailed else "environment_master"
    prompt = DETAILED_MASTER_PROMPT if detailed else MASTER_PROMPT
    node(1, "UNETLoader", [0, 0], [350, 82], outputs=[out("MODEL", "MODEL")], widgets=[ZIMAGE_MODEL, "default"],
         title="Z-Image Turbo", props=models(ZIMAGE_MODEL, f"{ZIMAGE_HF}/diffusion_models/{ZIMAGE_MODEL}", "diffusion_models",
                                               "UNETLoader"))
    node(2, "LoraLoaderModelOnly", [0, 120], [350, 82], inputs=[inp("model", "MODEL")], outputs=[out("MODEL", "MODEL")],
         widgets=[PIXEL_LORA, 0.55 if detailed else 1.0], title="Pixel-art LoRA", props=models(PIXEL_LORA, PIXEL_LORA_URL, "loras",
                                                                       "LoraLoaderModelOnly"))
    node(3, "ModelSamplingAuraFlow", [0, 240], [350, 58], inputs=[inp("model", "MODEL")], outputs=[out("MODEL", "MODEL")],
         widgets=[3.0])
    node(4, "CLIPLoader", [0, 340], [350, 106], outputs=[out("CLIP", "CLIP")], widgets=[ZIMAGE_TE, "lumina2", "default"],
         props=models(ZIMAGE_TE, f"{ZIMAGE_HF}/text_encoders/{ZIMAGE_TE}", "text_encoders", "CLIPLoader"))
    node(5, "VAELoader", [0, 485], [350, 58], outputs=[out("VAE", "VAE")], widgets=[ZIMAGE_VAE],
         props=models(ZIMAGE_VAE, f"{ZIMAGE_HF}/vae/{ZIMAGE_VAE}", "vae", "VAELoader"))
    node(6, "CLIPTextEncode", [400, 0], [420, 225], inputs=[inp("clip", "CLIP")],
         outputs=[out("CONDITIONING", "CONDITIONING")], widgets=[prompt], title="Scene prompt", color=GREEN)
    node(7, "ConditioningZeroOut", [400, 265], [240, 45], inputs=[inp("conditioning", "CONDITIONING")],
         outputs=[out("CONDITIONING", "CONDITIONING")])
    node(8, "EmptySD3LatentImage", [400, 350], [320, 106], outputs=[out("LATENT", "LATENT")],
         widgets=[1536, 640, 1], title="Canvas size", color=BLUE)
    node(9, "KSampler", [870, 0], [320, 262],
         inputs=[inp("model", "MODEL"), inp("positive", "CONDITIONING"), inp("negative", "CONDITIONING"),
                 inp("latent_image", "LATENT")],
         outputs=[out("LATENT", "LATENT")], widgets=[2601, "fixed", 8, 1.0, "res_multistep", "simple", 1.0],
         title="Master sample")
    node(10, "VAEDecode", [870, 300], [210, 46], inputs=[inp("samples", "LATENT"), inp("vae", "VAE")],
         outputs=[out("IMAGE", "IMAGE")])
    for wire in [(1, 0, 2, 0), (2, 0, 3, 0), (3, 0, 9, 0), (4, 0, 6, 0), (6, 0, 7, 0), (6, 0, 9, 1),
                 (7, 0, 9, 2), (8, 0, 9, 3), (9, 0, 10, 0), (5, 0, 10, 1)]:
        link(*wire)
    pixelize(g, 11, 1250, 0, 10, title="Master pixels", size=160, measure="whole frame",
             colors=48 if detailed else 32, background="keep", edge_cleanup=False)
    save_image(g, 12, 1650, 0, 11, prefix, "Save native master")
    save_image(g, 13, 1650, 345, 10, prefix + "_raw", "Save raw master")
    preview_text(g, 14, 1250, 485, 11, 2, "Master palette")
    node(15, "Note", [400, 510], [420, 230], widgets=[
        "Draw a full-bleed scene at about 2.4:1. Whole-frame pixelizing keeps the exact composition and the opaque sky. "
        "Default: 1536×640 → 384×160. Use the native scene and its hex palette in both environment asset workflows."
    ], title="READ ME", props={})
    g.save(ROOT / f"environment-master{suffix}.json", f"pixel-art-environment-master{suffix}",
           app=([(6, "text"), (8, "width"), (8, "height"), (9, "seed"), (11, "pixels"),
                 (11, "colors"), (11, "palette")], [12, 13, 14]))


def build_asset(kind, prompt, quiet=False):
    g = Graph()
    node, link = g.node, g.link
    node(1, "LoadImage", [0, 0], [335, 335], outputs=[out("IMAGE", "IMAGE"), out("MASK", "MASK")],
         widgets=["example.png", "image"], title="Environment reference", color=BLUE)
    node(2, "ImageScaleBy", [0, 385], [335, 82], inputs=[inp("image", "IMAGE")], outputs=[out("IMAGE", "IMAGE")],
         widgets=["nearest-exact", 2.0], title="Reference ×2 (keep the pixel clusters)")
    link(1, 0, 2, 0)
    model_chain(g, 380, 0)
    encoders(g, 770, 0, 70, 72, prompt)
    g.by_id[70]["title"] = "Asset prompt"
    for id_ in (70, 72):
        link(2, 0, id_, 2)
    node(3, "EmptySD3LatentImage", [770, 600], [320, 106], outputs=[out("LATENT", "LATENT")],
         widgets=[768, 768, 1], title="Asset canvas", color=BLUE)
    sampler(g, 4, 1230, 0, 2602 if kind == "prop" else 2603, "fixed")
    g.by_id[4]["title"] = "Asset sample"
    node(5, "VAEDecode", [1230, 300], [210, 46], inputs=[inp("samples", "LATENT"), inp("vae", "VAE")],
         outputs=[out("IMAGE", "IMAGE")])
    for wire in [(63, 0, 4, 0), (71, 0, 4, 1), (73, 0, 4, 2), (3, 0, 4, 3), (4, 0, 5, 0), (65, 0, 5, 1)]:
        link(*wire)
    save_image(g, 6, 1230, 390, 5, "environment_" + kind + "_raw", "Save raw asset")
    if kind == "prop":
        node(10, "LoadBackgroundRemovalModel", [1600, 0], [335, 58],
             outputs=[out("bg_model", "BACKGROUND_REMOVAL")], widgets=[BIREFNET], title="BiRefNet",
             props=models(BIREFNET, BIREFNET_URL, "background_removal", "LoadBackgroundRemovalModel"))
        node(11, "RemoveBackground", [1600, 95], [335, 46],
             inputs=[inp("bg_removal_model", "BACKGROUND_REMOVAL"), inp("image", "IMAGE")], outputs=[out("mask", "MASK")])
        node(12, "InvertMask", [1600, 175], [335, 46], inputs=[inp("mask", "MASK")], outputs=[out("mask", "MASK")])
        node(13, "JoinImageWithAlpha", [1600, 255], [335, 46],
             inputs=[inp("image", "IMAGE"), inp("alpha", "MASK")], outputs=[out("IMAGE", "IMAGE")])
        node(14, "CleanCutout", [1600, 336], [335, 165], inputs=[inp("images", "IMAGE")],
             outputs=[out("image", "IMAGE"), out("removed", "MASK"), out("info", "STRING")],
             widgets=[True, True, 0.02, 8.0, 0.2], title="Clean cutout")
        for wire in [(10, 0, 11, 0), (5, 0, 11, 1), (11, 0, 12, 0), (5, 0, 13, 0), (12, 0, 13, 1), (13, 0, 14, 0)]:
            link(*wire)
        pixelize(g, 20, 1985, 0, 14, title="Prop pixels", size=56, measure="longest side",
                 colors=0, background="alpha only", edge_cleanup=True)
        save_image(g, 21, 2385, 0, 20, "environment_prop", "Save native transparent prop")
        preview = []
    else:
        prefix = "environment_detailed_tile" if quiet else "environment_tile"
        pixelize(g, 20, 1600, 0, 5, title="Material pixels", size=64, measure="whole frame",
                 colors=0, background="keep", edge_cleanup=False, smooth=0.3 if quiet else 0.0)
        node(22, "MakeTileable", [1990, 0], [335, 180], inputs=[inp("images", "IMAGE")],
             outputs=[out("tile", "IMAGE"), out("preview", "IMAGE")],
             widgets=[32, 0, 3, 2, "both", 0], title="Seamless material", color=YELLOW)
        link(20, 0, 22, 0)
        node(24, "ImageScaleBy", [1990, 220], [335, 82], inputs=[inp("image", "IMAGE")],
             outputs=[out("IMAGE", "IMAGE")], widgets=["nearest-exact", 1.0], title="Output scale")
        link(22, 0, 24, 0)
        save_image(g, 21, 2385, 0, 24, prefix, "Save native repeating tile")
        save_image(g, 23, 2385, 345, 22, prefix + "_repeat", "Save repeat check", slot=1)
        preview = [23]
    node(30, "Note", [0, 510], [335, 250], widgets=[
        "Load the native master PNG as an actual visual reference. In Asset prompt ask for one new asset, "
        "rather than a remake of the whole scene. Paste the Master palette output in the pixels node to share "
        "its colours. The raw image is saved so you can inspect framing before quantization."
    ], title="READ ME", props={})
    suffix = "-detailed" if quiet else ""
    g.save(ROOT / f"environment-{kind}{suffix}.json", f"pixel-art-environment-{kind}{suffix}",
           app=([(1, "image"), (70, "prompt"), (4, "seed"), (20, "pixels"), (20, "palette")], [21, 6, *preview]))


if __name__ == "__main__":
    build_master()
    build_master(detailed=True)
    build_asset("prop", PROP_PROMPT)
    build_asset("material", MATERIAL_PROMPT)
    build_asset("material", QUIET_MATERIAL_PROMPT, quiet=True)
