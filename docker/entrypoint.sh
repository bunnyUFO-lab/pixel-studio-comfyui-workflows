#!/usr/bin/env bash
set -euo pipefail

# `models ...` runs the model tool, e.g. `docker compose run --rm comfyui models`.
if [ "${1:-}" = "models" ]; then
    shift
    exec python /usr/local/bin/models.py "$@"
fi

# Any other arguments replace the default command, e.g. `docker compose run --rm comfyui bash`.
if [ "$#" -gt 0 ]; then
    exec "$@"
fi

# Fetch missing models for the workflows listed in MODELS (from .env) before starting.
# A failed download must not stop ComfyUI (restart: unless-stopped would loop forever).
if [ -n "${MODELS:-}" ]; then
    python /usr/local/bin/models.py download \
        || echo "WARNING: some models failed to download (see above); starting ComfyUI anyway"
fi

# shellcheck disable=SC2086  # COMFY_ARGS is intentionally word-split
exec python /opt/ComfyUI/main.py --listen 0.0.0.0 --port 8188 --enable-manager ${COMFY_ARGS:-}
