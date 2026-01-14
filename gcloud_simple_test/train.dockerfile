# Base image
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /gcloud_simple_test/

COPY gcloud_simple_test/pyproject.toml pyproject.toml
COPY gcloud_simple_test/main.py main.py

RUN uv sync --no-cache

ENTRYPOINT ["uv", "run", "main.py"]
