# RunPod Serverless worker: the PC's ComfyUI image (docker/Dockerfile) plus the job handler. No models, no secrets.
# Build from the repo root:  psc cloud runpod build   (or: docker build -f cloud/runpod/Dockerfile -t IMAGE .)
ARG BASE=comfyui-local:v0.38.2
FROM ${BASE}
# runpod 1.12.0 and boto3 add only their own dependencies (checked with pip --dry-run against the base image on
# 24 Oct 2026: torch, numpy, Pillow and aiohttp stay as they are). websockets is the pin of the Modal image.
RUN pip install runpod==1.12.0 boto3==1.43.118 websockets==13.1
COPY custom_nodes/comfyui_workflows_utils /opt/ComfyUI/custom_nodes/comfyui_workflows_utils
COPY workflows /opt/ComfyUI/user/default/workflows
COPY env/pixel.env /opt/env/pixel.env
COPY docker/models.py /usr/local/bin/models.py
COPY cloud/job_runner.py cloud/runpod/handler.py cloud/runpod/gpu_rates.json /opt/psc/
ENTRYPOINT []
CMD ["python", "-u", "/opt/psc/handler.py"]
