FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/cache/huggingface

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    git git-lfs curl wget build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Optional: install flash-attn if build environment supports it
ARG INSTALL_FLASH_ATTN=false
RUN if [ "$INSTALL_FLASH_ATTN" = "true" ]; then \
    pip install flash-attn --no-build-isolation; \
    fi

COPY . .

ENV PYTHONPATH=/workspace:$PYTHONPATH

CMD ["/bin/bash"]
