FROM nvidia/cuda:13.0.3-runtime-ubuntu24.04 AS runtime

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.13 \
    python3.13-venv \
    python3.13-dev \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python3.13 -m venv /app/.venv && /app/.venv/bin/pip install --no-cache-dir uv

ENV PATH="/app/.venv/bin:${PATH}" \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock README.md ./

RUN uv sync --frozen --no-dev

COPY . .

RUN uv sync --frozen --no-dev

ENTRYPOINT ["uv", "run", "--frozen"]
