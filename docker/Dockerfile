# eos-switch development/experiment image.
#
# Base: CUDA 12.8 + cuDNN on Ubuntu 24.04. The RTX 5090 (Blackwell, sm_120)
# requires PyTorch >= 2.7 built against CUDA 12.8; torch is installed by uv
# from the official cu128 wheel index (pinned in uv.lock).
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# The project directory is bind-mounted at /workspace at runtime, so the
# virtualenv must live OUTSIDE it or the mount would shadow it.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy

RUN uv python install 3.12

WORKDIR /workspace
COPY pyproject.toml uv.lock .python-version ./
# Dependencies only; the project itself is installed (editable) from the
# bind mount with `uv sync --frozen` after the container starts.
RUN uv sync --frozen --no-install-project

ENV PATH="/opt/venv/bin:${PATH}"

CMD ["sleep", "infinity"]
