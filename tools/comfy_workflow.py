"""Small helpers for generating ComfyUI UI-format workflow JSON (used by tools/build_*.py)."""
import json
import uuid
from pathlib import Path

WORKFLOWS = Path(__file__).resolve().parent.parent / "workflows"

# Shared by the image-to-SVG workflows
UPSCALER = "RealESRGAN_x4plus_anime_6B.pth"   # 18 MB, BSD-3, made for illustrations / line art
UPSCALER_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
ALPHA_CUTOFF = 0.5   # pixels more transparent than this are dropped

# Wan 2.2 Remix (image-to-video and video extension)
HIGH = "Wan2.2_Remix_NSFW_i2v_14b_high_lighting_fp8_e4m3fn_v3.0.safetensors"
LOW = "Wan2.2_Remix_NSFW_i2v_14b_low_lighting_fp8_e4m3fn_v3.0.safetensors"
TEXT_ENCODER = "nsfw_wan_umt5-xxl_fp8_scaled.safetensors"
VAE = "wan_2.1_vae.safetensors"

# Download sources, stored on each loader node as properties.models (read by ComfyUI's
# missing-models dialog and by docker/models.py).
HF = "https://huggingface.co"
URLS = {
    HIGH: f"{HF}/FX-FeiHou/wan2.2-Remix/resolve/main/NSFW/{HIGH}",
    LOW: f"{HF}/FX-FeiHou/wan2.2-Remix/resolve/main/NSFW/{LOW}",
    TEXT_ENCODER: f"{HF}/NSFW-API/NSFW-Wan-UMT5-XXL/resolve/main/{TEXT_ENCODER}",
    VAE: f"{HF}/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/{VAE}",
}

# Official Wan negative prompt (the model was trained with it) plus a few English extras.
NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走, "
    "slow motion, static, frozen, watermark, subtitles, blurry, deformed, extra limbs"
)

# Qwen-Image-2.1 (edit / repaint); Qwen Research licence (non-commercial)
QWEN_HF = "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main"
QWEN_MODEL = "qwen_image_2.1_int8_convrot.safetensors"
QWEN_TE = "qwen3vl_8b_int8_convrot.safetensors"
QWEN_VAE = "qwen_image_2.1_vae_bf16.safetensors"

# Background removal (core LoadBackgroundRemovalModel), MIT
BIREFNET = "birefnet.safetensors"
BIREFNET_URL = "https://huggingface.co/Comfy-Org/BiRefNet/resolve/main/background_removal/birefnet.safetensors"

# Z-Image Turbo (Tongyi-MAI, Apache-2.0) text-to-image, Comfy-Org repack
ZIMAGE_HF = "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files"
# 8-bit builds (int8 model 5.8 GB, fp8 text encoder 5.3 GB) instead of bf16 (11.5 + 7.5 GB): on the RTX 4090 the
# sprite workflow peaks at 14.5 GB instead of 21.8 GB, loads ~40% faster and makes the same images, so a local
# LLM fits beside it. Measured 2026-10-05 (game-dev workspace card "ComfyUI Pixel-Art Efficiency").
ZIMAGE_MODEL = "z_image_turbo_int8_convrot.safetensors"
ZIMAGE_TE = "qwen_3_4b_fp8_mixed.safetensors"
ZIMAGE_VAE = "ae.safetensors"
# Pixel-art style LoRA for Z-Image Turbo (tarn59, Apache-2.0), trigger "Pixel art style."
PIXEL_LORA = "pixel_art_style_z_image_turbo.safetensors"
PIXEL_LORA_URL = "https://huggingface.co/tarn59/pixel_art_style_lora_z_image_turbo/resolve/main/" + PIXEL_LORA

# Pixel-art sprite animation LoRA for Wan 2.2 I2V (styly-agents, Apache-2.0); same file on HIGH and LOW
PIXEL_ANIM_LORA = "wan2.2_animate_adapter_model.safetensors"
PIXEL_ANIM_LORA_URL = "https://huggingface.co/styly-agents/Wan2-2-pixel-animate/resolve/main/" + PIXEL_ANIM_LORA

# Wan Animate 2 (Wan-Video, Apache-2.0) motion transfer, Comfy-Org repack
ANIMATE_HF = "https://huggingface.co/Comfy-Org/Wan-Animate-2/resolve/main"
ANIMATE_MODEL = "wan_animate_2_int8_convrot.safetensors"
ANIMATE_LORA = "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
CLIP_VISION_H = "clip_vision_h.safetensors"

# SCAIL 2 (Z.ai, Apache-2.0 / MIT) character animation on Wan 2.1 14B, Comfy-Org repack; shares the lightx2v LoRA,
# umt5, clip_vision_h and Wan 2.1 VAE above. SAM3.1 (Meta, SAM licence) masks the person in the driving video.
SCAIL_HF = "https://huggingface.co/Comfy-Org/SCAIL-2/resolve/main"
SCAIL_MODEL = "wan2.1_14B_SCAIL_2_int8_convrot.safetensors"
SCAIL_DPO_LORA = "wan2.1_SCAIL_2_DPO_lora_bf16.safetensors"
SAM3 = "sam3.1_multiplex_fp16.safetensors"
URLS.update({
    SCAIL_MODEL: f"{SCAIL_HF}/diffusion_models/{SCAIL_MODEL}",
    SCAIL_DPO_LORA: f"{SCAIL_HF}/loras/{SCAIL_DPO_LORA}",
    SAM3: f"{HF}/Comfy-Org/sam3.1/resolve/main/checkpoints/{SAM3}",
})

# MiniMax H3 Ref2VA (MiniMax community licence: excludes the US, EU, UK and South Korea without a commercial licence),
# Comfy-Org repack. Smallest build: pruned w6a8 model (16 GB) + NVFP4 Qwen3-VL 32B text encoder (15.7 GB).
H3_HF = "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main"
H3_MODEL = "minimax_h3_ref2va_pruned_w6a8.safetensors"
H3_TE = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
H3_VAE = "minimax_h3_video_vae_int8_convrot.safetensors"
H3_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
URLS.update({
    H3_MODEL: f"{H3_HF}/diffusion_models/{H3_MODEL}",
    H3_TE: f"{H3_HF}/text_encoders/{H3_TE}",
    H3_VAE: f"{H3_HF}/vae/{H3_VAE}",
    H3_AUDIO_VAE: f"{H3_HF}/vae/{H3_AUDIO_VAE}",
})

# Node colors as (color, bgcolor)
GREEN, RED, YELLOW, BLUE = ("#232", "#353"), ("#322", "#533"), ("#432", "#653"), ("#223", "#335")


def inp(name, type_, widget=None):
    d = {"name": name, "type": type_, "link": None}
    if widget:
        d["widget"] = {"name": widget}
    return d


def out(name, type_, widget=None):
    d = {"name": name, "type": type_, "links": []}
    if widget:
        d["widget"] = {"name": widget}
    return d


class Graph:
    def __init__(self):
        self.nodes, self.links, self.by_id = [], [], {}

    def node(self, id_, type_, pos, size, inputs=(), outputs=(), widgets=None, title=None, mode=0, color=None,
             props=None):
        n = {
            "id": id_, "type": type_, "pos": pos, "size": size, "flags": {}, "order": 0, "mode": mode,
            "inputs": list(inputs), "outputs": list(outputs),
            "properties": props if props is not None else {"Node name for S&R": type_},
        }
        if widgets is not None:
            n["widgets_values"] = widgets
        if title:
            n["title"] = title
        if color:
            n["color"], n["bgcolor"] = color
        self.nodes.append(n)
        self.by_id[id_] = n

    def link(self, src, src_slot, dst, dst_slot):
        s, d = self.by_id[src], self.by_id[dst]
        link_id = len(self.links) + 1
        self.links.append([link_id, src, src_slot, dst, dst_slot, s["outputs"][src_slot]["type"]])
        s["outputs"][src_slot]["links"].append(link_id)
        s["outputs"][src_slot]["slot_index"] = src_slot
        d["inputs"][dst_slot]["link"] = link_id

    def save(self, path, name, groups=(), scale=0.7, offset=(60, 120), app=None):
        """Write the workflow; `name` seeds a stable workflow id. app = (inputs, outputs) for ComfyUI's App Mode:
        inputs = [(node_id, widget_name), ...], outputs = [node_id, ...] (the workflow then also opens as a simple
        form; Toggle App Mode shows the graph)."""
        for i, n in enumerate(sorted(self.nodes, key=lambda n: (n["pos"][0], n["pos"][1]))):
            n["order"] = i
        workflow = {
            "id": str(uuid.uuid5(uuid.NAMESPACE_URL, name)),
            "revision": 0,
            "last_node_id": max(self.by_id),
            "last_link_id": len(self.links),
            "nodes": self.nodes,
            "links": self.links,
            "groups": [{"id": i + 1, "title": t, "bounding": b, "color": c, "flags": {}}
                       for i, (t, b, c) in enumerate(groups)],
            "config": {},
            "extra": {"ds": {"scale": scale, "offset": list(offset)}},
            "version": 0.4,
        }
        if app is not None:
            ins, outs = app
            workflow["extra"]["linearMode"] = True
            workflow["extra"]["linearData"] = {"inputs": [[str(i), w] for i, w in ins], "outputs": [str(o) for o in outs]}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {path} ({len(self.nodes)} nodes, {len(self.links)} links)")
