# The style trainer for a rented machine: ostris ai-toolkit plus trainer/runner.py. No token and no keys in the
# image: the token comes as a pod environment variable at rent time. Build: pixel-studio-comfy cloud trainer build
ARG BASE=ostris/aitoolkit:0.13.23
FROM ${BASE}
COPY trainer/runner.py trainer/cloud/entrypoint.sh trainer/cloud/shutdown.py /runner/
ENV RUNNER_HOST=0.0.0.0 RUNNER_PORT=8675 CONFIG_ROOT=/workspace/configs OUTPUT_ROOT=/workspace/output \
    DATASETS_ROOT=/workspace/datasets RUNNER_STATE_DIR=/workspace/runner-state RUNNER_TIME_LIMIT_S=10800 \
    RUNNER_IDLE_LIMIT_S=1800 RUNNER_ON_EXIT=runpod-terminate HF_HOME=/workspace/hf-cache PYTHONUNBUFFERED=1
EXPOSE 8675
ENTRYPOINT ["bash", "/runner/entrypoint.sh"]
CMD []
