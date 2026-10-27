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

# Fast local video models for sprites (tools/build_pixel_sprite_fast.py). Base models Apache-2.0, Comfy-Org repacks.
WAN21_HF = f"{HF}/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files"
WAN22_HF = f"{HF}/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files"
TI2V_5B = "wan2.2_ti2v_5B_fp16.safetensors"           # 10 GB, Wan 2.2 text/image-to-video 5B, 24 fps
WAN22_VAE = "wan2.2_vae.safetensors"                   # the 5B's own VAE (16x spatial); 14B models use wan_2.1_vae
VACE_1_3B = "wan2.1_vace_1.3B_fp16.safetensors"       # 4.3 GB, Wan 2.1 T2V 1.3B + VACE (first/last frame, pose)
# Step-distill LoRAs. FastWan (FastVideo, Apache-2.0) is the one fit to ship; Turbo (quanhaol, code CC-BY-NC-SA-4.0)
# and CausVid (tianweiy, CC-BY-NC-4.0) are non-commercial, kept for comparison only.
FASTWAN_5B_LORA = "Wan2_2_5B_FastWanFullAttn_lora_rank_128_bf16.safetensors"
TURBO_5B_LORA = "Wan22_TI2V_5B_Turbo_lora_rank_64_fp16.safetensors"
CAUSVID_1_3B_LORA = "Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors"
KIJAI_HF = f"{HF}/Kijai/WanVideo_comfy/resolve/main"
# Frame interpolation (core FrameInterpolate node), RIFE 4.26 (MIT), 22 MB
RIFE = "rife_v4.26.safetensors"
RIFE_URL = f"{HF}/Comfy-Org/Frame-Interpolation/resolve/main/frame_interpolation/{RIFE}"
URLS.update({
    TI2V_5B: f"{WAN22_HF}/diffusion_models/{TI2V_5B}",
    WAN22_VAE: f"{WAN22_HF}/vae/{WAN22_VAE}",
    VACE_1_3B: f"{WAN21_HF}/diffusion_models/{VACE_1_3B}",
    FASTWAN_5B_LORA: f"{KIJAI_HF}/FastWan/{FASTWAN_5B_LORA}",
    TURBO_5B_LORA: f"{HF}/hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF/resolve/main/{TURBO_5B_LORA}",
    CAUSVID_1_3B_LORA: f"{KIJAI_HF}/{CAUSVID_1_3B_LORA}",
    RIFE: RIFE_URL,
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
