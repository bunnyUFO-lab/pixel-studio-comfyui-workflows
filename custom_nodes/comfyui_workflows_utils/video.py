"""Nodes for extending a video with Wan 2.2 + SVI 2.0 Pro (Stable Video Infinity) LoRAs.

SVI continues a clip from the *latent* of the previous clip (motion context) plus an anchor latent
that keeps identity. WanSVIProConditioning re-implements the conditioning described for SVI 2.0 Pro
(as in KJNodes' WanImageToVideoSVIPro): image conditioning = [anchor, last motion latents, padding],
with only the first (anchor) latent marked as known.
"""
import comfy.latent_formats
import comfy.model_management
import comfy.utils
import node_helpers
import torch


class WanSVIProConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "anchor_samples": ("LATENT", {"tooltip": "VAE-encoded anchor frame (keeps identity)."}),
                "length": ("INT", {"default": 81, "min": 5, "max": 1024, "step": 4, "tooltip": "Frames to generate (4n+1)."}),
                "motion_latent_count": ("INT", {"default": 1, "min": 0, "max": 16, "tooltip": "How many of the previous clip's last latents (4 frames each) to continue from. 1 is recommended."}),
            },
            "optional": {
                "prev_samples": ("LATENT", {"tooltip": "Latent of the previous clip (or the encoded end of the input video)."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "run"
    CATEGORY = "comfyui_workflows/video"
    DESCRIPTION = "SVI 2.0 Pro conditioning for Wan 2.2 I2V: continue from the previous latent, anchored to a reference frame."

    def run(self, positive, negative, anchor_samples, length, motion_latent_count, prev_samples=None):
        anchor = anchor_samples["samples"][:, :, :1].clone()
        B, C, _, H, W = anchor.shape
        total = (length - 1) // 4 + 1
        parts = [anchor]
        if prev_samples is not None and motion_latent_count > 0:
            parts.append(prev_samples["samples"][:, :, -motion_latent_count:].clone().to(anchor))
        used = sum(p.shape[2] for p in parts)
        if used >= total:
            raise ValueError(f"length {length} is too short for the anchor + {motion_latent_count} motion latents")
        padding = comfy.latent_formats.Wan21().process_out(torch.zeros(B, C, total - used, H, W, dtype=anchor.dtype,
                                                                        device=anchor.device))
        image_cond = torch.cat(parts + [padding], dim=2)
        mask = torch.ones((1, 1, total, H, W), dtype=anchor.dtype, device=anchor.device)
        mask[:, :, :1] = 0.0
        positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": image_cond, "concat_mask": mask})
        negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": image_cond, "concat_mask": mask})
        latent = torch.zeros([B, 16, total, H, W], device=comfy.model_management.intermediate_device())
        return (positive, negative, {"samples": latent})


class SegmentPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "prompts": ("STRING", {"multiline": True, "default": "", "tooltip": "One prompt per line = one per segment. If there are more segments than lines, the last line repeats."}),
            "index": ("INT", {"default": 0, "min": 0, "max": 10000, "forceInput": True}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True,
                             "tooltip": "Base seed; segment N uses seed + N, so segments don't repeat the same noise."}),
        }}

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("prompt", "segment_seed")
    FUNCTION = "run"
    CATEGORY = "comfyui_workflows/video"

    def run(self, prompts, index, seed):
        lines = [line.strip() for line in prompts.splitlines() if line.strip()]
        return (lines[min(index, len(lines) - 1)] if lines else "", (seed + index) % 0xffffffffffffffff)


class ResizeForWan:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "max_side": ("INT", {"default": 832, "min": 256, "max": 2048, "step": 16, "tooltip": "Longest side in px (832 = 480p, 1280 = 720p). Both sides are rounded to multiples of 16."}),
        }}

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("images", "width", "height")
    FUNCTION = "run"
    CATEGORY = "comfyui_workflows/video"

    def run(self, images, max_side):
        h, w = images.shape[1:3]
        s = max_side / max(h, w)
        nw, nh = max(16, round(w * s / 16) * 16), max(16, round(h * s / 16) * 16)
        if (nw, nh) != (w, h):
            images = comfy.utils.common_upscale(images.movedim(-1, 1), nw, nh, "lanczos", "disabled").movedim(1, -1)
        return (images.clamp(0, 1), nw, nh)


class VideoAnchorAndTail:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "anchor": (["first frame", "last frame"], {"tooltip": "Frame SVI keeps returning to. First frame = keeps the identity of the original; last frame = follows where the video ended up."}),
            "tail_frames": ("INT", {"default": 5, "min": 1, "max": 81, "tooltip": "Frames from the end of the video used as motion context (5 = 2 latents)."}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("anchor", "tail")
    FUNCTION = "run"
    CATEGORY = "comfyui_workflows/video"

    def run(self, images, anchor, tail_frames):
        a = images[:1] if anchor == "first frame" else images[-1:]
        return (a, images[-min(tail_frames, images.shape[0]):])


class JoinVideoSegments:
    """Append SVI segments to the original video.

    Each segment's first frame is the anchor (not new content) and is dropped. The next `overlap` frames
    re-create the end of the previous part (they were generated from its last latent), so they are
    cross-faded into it. Optional colour matching scales each segment's per-channel mean/std to match
    the previous part over the overlap, which limits colour drift between segments.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "original": ("IMAGE",),
            "segments": ("IMAGE",),
            "overlap": ("INT", {"default": 4, "min": 0, "max": 32, "tooltip": "Frames at the start of each segment that repeat the end of the previous part; cross-faded."}),
            "color_match": ("BOOLEAN", {"default": True, "tooltip": "Match each segment's colours to the previous part over the overlap."}),
        }}

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("full_video", "new_frames")
    FUNCTION = "run"
    CATEGORY = "comfyui_workflows/video"

    def run(self, original, segments, overlap, color_match):
        full = torch.cat(original, dim=0) if isinstance(original, list) else original
        overlap, color_match = overlap[0], color_match[0]
        new_parts = []
        for seg in segments:
            seg = seg[1:].to(full)
            if seg.shape[1:] != full.shape[1:]:
                seg = comfy.utils.common_upscale(seg.movedim(-1, 1), full.shape[2], full.shape[1], "lanczos",
                                                 "disabled").movedim(1, -1)
            n = min(overlap, seg.shape[0], full.shape[0])
            if color_match and n > 0:
                ref, src = full[-n:], seg[:n]
                m_ref, s_ref = ref.mean((0, 1, 2)), ref.std((0, 1, 2)) + 1e-4
                m_src, s_src = src.mean((0, 1, 2)), src.std((0, 1, 2)) + 1e-4
                seg = ((seg - m_src) / s_src * s_ref + m_ref).clamp(0, 1)
            if n > 0:
                w = torch.linspace(1, n, n, dtype=full.dtype, device=full.device).view(-1, 1, 1, 1) / (n + 1)
                full = torch.cat([full[:-n], full[-n:] * (1 - w) + seg[:n] * w, seg[n:]], dim=0)
            else:
                full = torch.cat([full, seg], dim=0)
            new_parts.append(seg[n:])
        new = torch.cat(new_parts, dim=0) if new_parts else full[:0]
        return (full, new)
