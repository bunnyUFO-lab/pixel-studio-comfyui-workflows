# ComfyUI for NVIDIA GPUs (RTX 30xx/40xx/50xx). Models are NOT baked in; see models/*.txt.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev build-essential \
        git ffmpeg aria2 curl ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

# PyTorch first: biggest layer, changes least often.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
RUN pip install torch torchvision torchaudio --index-url ${TORCH_INDEX}

ARG COMFYUI_REF=v0.38.2
RUN git clone --depth 1 --branch ${COMFYUI_REF} https://github.com/comfyanonymous/ComfyUI /opt/ComfyUI \
    && pip install -r /opt/ComfyUI/requirements.txt -r /opt/ComfyUI/manager_requirements.txt

# Custom nodes, pinned in custom_nodes.txt ("<git-url> <commit>" per line).
COPY custom_nodes.txt /tmp/custom_nodes.txt
RUN set -eu; cd /opt/ComfyUI/custom_nodes; \
    tr -d '\r' < /tmp/custom_nodes.txt | grep -vE '^\s*(#|$)' | while read -r url ref; do \
        dir=$(basename "$url" .git); \
        git clone "$url" "$dir"; git -C "$dir" checkout -q "$ref"; \
        if [ -f "$dir/requirements.txt" ]; then pip install -r "$dir/requirements.txt"; fi; \
    done

# Extra pip packages for the repo's own nodes (see the file for why --no-deps).
COPY requirements-extra.txt /tmp/requirements-extra.txt
RUN pip install --no-deps -r /tmp/requirements-extra.txt

COPY entrypoint.sh models.py /usr/local/bin/
RUN sed -i 's/\r$//' /usr/local/bin/*.sh && chmod +x /usr/local/bin/*.sh

WORKDIR /opt/ComfyUI
EXPOSE 8188
ENTRYPOINT ["entrypoint.sh"]
