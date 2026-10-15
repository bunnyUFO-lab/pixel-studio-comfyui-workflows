"""POST /spritestudio/forget: Sprite Studio deletes a finished job's files here when it can't reach ComfyUI's folders
(it runs on another machine, e.g. bunnylink, or ComfyUI runs in the cloud).

Body: {"outputs": [{"filename", "subfolder", "type"}], "uploads": ["ss_<10 hex>_<name>", ...]} -> {"deleted": n}.
Only the outputs a job's history listed (type "output") and the app's own uniquely named uploads are removed, and
only inside ComfyUI's output / input folders. Anything else is ignored.
"""
import logging
import os
import re

UPLOAD_NAME = re.compile(r"^ss_[0-9a-f]{10}_[^/\\]+$")


def _remove_inside(root, sub, name):
    root = os.path.realpath(root)
    p = os.path.realpath(os.path.join(root, sub or "", name))
    if not p.startswith(root + os.sep) or not os.path.isfile(p):
        return 0
    try:
        os.remove(p)
        return 1
    except OSError:
        return 0


def forget(body, output_dir, input_dir):
    n = 0
    for f in (body or {}).get("outputs") or []:
        if isinstance(f, dict) and f.get("type", "output") == "output" and isinstance(f.get("filename"), str):
            n += _remove_inside(output_dir, str(f.get("subfolder") or ""), f["filename"])
    for name in (body or {}).get("uploads") or []:
        if isinstance(name, str) and UPLOAD_NAME.match(name):
            n += _remove_inside(input_dir, "", name)
    return n


try:                                     # inside ComfyUI only; the module stays importable (and testable) without it
    import folder_paths
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.post("/spritestudio/forget")
    async def _forget_route(request):
        try:
            body = await request.json()
        except Exception:                # noqa: BLE001
            return web.json_response({"error": "expected JSON"}, status=400)
        n = forget(body, folder_paths.get_output_directory(), folder_paths.get_input_directory())
        return web.json_response({"deleted": n})
except Exception as e:                   # noqa: BLE001 - not running inside ComfyUI
    logging.debug("comfyui_workflows_utils.cleanup: route not registered (%s)", e)
