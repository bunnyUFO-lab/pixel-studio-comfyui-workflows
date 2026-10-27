"""GET /spritestudio/loaded_models: the models ComfyUI holds right now and how much of each sits in GPU memory.

-> {"models": [{"name": "WAN21_Vace", "device": "cuda:0", "vram_mb": 14200, "size_mb": 16542, "in_use": false}, ...]}
ComfyUI keeps the last job's models loaded until another job needs the room or something posts /free
{"unload_models": true}. The Windows memory watchdog (tools/windows-memory-watchdog) shows this list in --status and
in its log when VRAM runs low. Read-only.
"""
import logging

MB = 1024 * 1024


def loaded_models(current):
    """current: comfy.model_management.current_loaded_models -> one dict per model, biggest share of VRAM first."""
    out = []
    for lm in list(current):                 # a copy: the prompt worker may change the list meanwhile
        try:
            patcher = lm.model
            if patcher is None:              # a weak reference whose model is already gone
                continue
            out.append({"name": type(getattr(patcher, "model", patcher)).__name__, "device": str(lm.device),
                        "vram_mb": round(lm.model_loaded_memory() / MB), "size_mb": round(lm.model_memory() / MB),
                        "in_use": bool(getattr(lm, "currently_used", False))})
        except Exception as e:               # noqa: BLE001 - one odd entry must not hide the others
            logging.debug("loaded_models: skipped an entry (%s)", e)
    return sorted(out, key=lambda m: -m["vram_mb"])


try:                                         # inside ComfyUI only; the module stays importable (and testable) without it
    import comfy.model_management as model_management
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/spritestudio/loaded_models")
    async def _loaded_models_route(request):
        return web.json_response({"models": loaded_models(model_management.current_loaded_models)})
except Exception as e:                       # noqa: BLE001 - not running inside ComfyUI
    logging.debug("comfyui_workflows_utils.memory: route not registered (%s)", e)
