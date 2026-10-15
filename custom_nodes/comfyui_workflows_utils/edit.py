"""Image-edit helpers: optional inpainting for whole-image edit models (crop -> edit -> stitch).

Edit models like Qwen-Image don't see a mask: they edit the whole input image, which is first scaled to
about 1 megapixel. Small masked details (a 3 px spider eye) are then too small to change. So:
  CropForMaskedEdit   crops the masked area plus context and enlarges it to ~target_megapixels
  (edit model)        edits the crop
  MaskedEditComposite shrinks the edit back, places it at the crop position and blends it in only
                      inside the (grown, feathered) mask; pixels outside stay exactly as they were.
With no mask painted, the crop is the whole image and the full edit is returned.
"""
import math

import comfy.utils
import torch
import torch.nn.functional as F


def _has_mask(mask):
    return mask is not None and float(mask.max()) > 0.0


def _mask_hw(mask, H, W, ref):
    m = mask.reshape(-1, 1, *mask.shape[-2:]).to(ref)[:1]
    if m.shape[-2:] != (H, W):
        m = F.interpolate(m, size=(H, W), mode="bilinear", align_corners=False)
    return (m > 0.5).to(ref.dtype)


def _resize(img, w, h):
    return comfy.utils.common_upscale(img.movedim(-1, 1), w, h, "lanczos", "disabled").movedim(1, -1).clamp(0, 1)


class CropForMaskedEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "context": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 5.0, "step": 0.05,
                                      "tooltip": "Extra area around the mask, as a fraction of its size, so the model sees what surrounds it."}),
                "min_context_px": ("INT", {"default": 64, "min": 0, "max": 2048,
                                           "tooltip": "At least this many pixels of context on each side."}),
                "target_megapixels": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05,
                                                "tooltip": "The crop is resized to about this size before editing (Qwen-Image works at ~1 MP)."}),
            },
            "optional": {"mask": ("MASK",)},
        }

    RETURN_TYPES = ("IMAGE", "CROP_INFO")
    RETURN_NAMES = ("image", "crop_info")
    FUNCTION = "run"
    CATEGORY = "comfyui_workflows/edit"
    DESCRIPTION = "Crop to the painted mask (+context) and enlarge it, so small areas get the edit model's full resolution."

    def run(self, image, context, min_context_px, target_megapixels, mask=None):
        H, W = image.shape[1:3]
        image = image[..., :3]
        if not _has_mask(mask):
            return (image, {"box": (0, 0, W, H), "size": (W, H), "masked": False})
        m = _mask_hw(mask, H, W, image)[0, 0]
        ys, xs = torch.nonzero(m > 0, as_tuple=True)
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        py = max(min_context_px, int((y1 - y0) * context))
        px = max(min_context_px, int((x1 - x0) * context))
        y0, y1, x0, x1 = max(0, y0 - py), min(H, y1 + py), max(0, x0 - px), min(W, x1 + px)
        crop = image[:, y0:y1, x0:x1]
        ch, cw = crop.shape[1:3]
        s = math.sqrt(target_megapixels * 1e6 / (cw * ch))
        tw, th = max(32, round(cw * s / 16) * 16), max(32, round(ch * s / 16) * 16)
        return (_resize(crop, tw, th), {"box": (x0, y0, x1, y1), "size": (W, H), "masked": True})


class MaskedEditComposite:
    """Composite an edited image (or edited crop) into the original inside a mask.

    No mask painted -> the full edit is returned. Mask painted -> the edit is resized to the original (or,
    with crop_info, to the crop box and placed there), the mask is grown and feathered, and only the masked
    area changes; pixels outside the mask stay exactly as in the original.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original": ("IMAGE",),
                "edited": ("IMAGE",),
                "grow_px": ("INT", {"default": 8, "min": 0, "max": 256, "tooltip": "Grow the painted mask by this many pixels."}),
                "feather_px": ("INT", {"default": 12, "min": 0, "max": 256, "tooltip": "Soft edge width in pixels, so the edit blends in."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Painted area to change (Load Image's mask). Empty = edit the whole image."}),
                "crop_info": ("CROP_INFO", {"tooltip": "From Crop For Masked Edit, when the edit was made on a crop."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "used_mask")
    FUNCTION = "run"
    CATEGORY = "comfyui_workflows/edit"

    def run(self, original, edited, grow_px, feather_px, mask=None, crop_info=None):
        original = original[..., :3]
        H, W = original.shape[1:3]
        edited = edited[..., :3].to(original).clamp(0, 1)
        if crop_info is not None and crop_info.get("masked"):
            x0, y0, x1, y1 = crop_info["box"]
            canvas = original.clone()
            canvas[:, y0:y1, x0:x1] = _resize(edited, x1 - x0, y1 - y0)[:1]
            edited = canvas
        elif edited.shape[1:3] != (H, W):
            edited = _resize(edited, W, H)
        if not _has_mask(mask):
            return (edited, torch.ones((1, H, W), dtype=original.dtype, device=original.device))
        m = _mask_hw(mask, H, W, original)
        if grow_px > 0:
            m = F.max_pool2d(m, kernel_size=2 * grow_px + 1, stride=1, padding=grow_px)
        if feather_px > 0:
            k = 2 * feather_px + 1
            m = F.avg_pool2d(F.pad(m, (feather_px,) * 4, mode="replicate"), kernel_size=k, stride=1)
            m = m * (m > 1e-3)
        m = m.clamp(0, 1)
        alpha = m[0].unsqueeze(-1)                       # (H, W, 1)
        out = original * (1 - alpha) + edited * alpha
        return (out, m[:, 0])
