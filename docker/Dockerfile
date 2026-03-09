# accept "cpu-only" or "gpu"
ARG IMAGE_TYPE="cpu-only"

FROM python:3.11-slim AS cpu-only-base
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS gpu-base
FROM ${IMAGE_TYPE}-base AS base

ARG CMS_UID=1000
ARG CMS_GID=1000
ARG CMS_MODEL_NAME
ARG CMS_MODEL_TYPE
ARG CMS_SERVE_EXTRA_OPTIONS
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
# accept comma saperated optional dependencies
ARG PIP_EXTRAS=""

ENV CMS_MODEL_NAME=$CMS_MODEL_NAME
ENV CMS_MODEL_TYPE=$CMS_MODEL_TYPE
ENV CMS_SERVE_EXTRA_OPTIONS=$CMS_SERVE_EXTRA_OPTIONS
ENV HTTP_PROXY=$HTTP_PROXY
ENV HTTPS_PROXY=$HTTPS_PROXY
ENV NO_PROXY=$NO_PROXY
ENV http_proxy=$HTTP_PROXY
ENV https_proxy=$HTTPS_PROXY
ENV no_proxy=$NO_PROXY
ENV PYTHONUNBUFFERED=1
ENV PATH="/home/cms/.local/bin:${PATH}"
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ARG IMAGE_TYPE

RUN if [ "$IMAGE_TYPE" = "gpu" ]; then \
        apt-get update && apt-get install -y --no-install-recommends \
        nano telnet software-properties-common && \
        add-apt-repository ppa:deadsnakes/ppa -y && \
        apt-get update && \
        apt-get install -y --no-install-recommends python3.11 python3.11-dev python3.11-venv python3-pip && \
        rm -rf /var/lib/apt/lists/*; \
    else \
        apt-get update && apt-get install -y --no-install-recommends \
        nano telnet build-essential g++ && \
        rm -rf /var/lib/apt/lists/*; \
    fi

RUN if [ -n "$PIP_EXTRAS" ]; then \
        apt-get update && apt-get install -y --no-install-recommends \
        build-essential g++ && \
        rm -rf /var/lib/apt/lists/*; \
    fi

RUN addgroup --gid $CMS_GID cms || true && \
    adduser --uid $CMS_UID --gid $CMS_GID --disabled-password --gecos "" cms || true && \
    echo "cms ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

COPY app /app
COPY pyproject.toml /pyproject.toml
COPY uv.lock /uv.lock
COPY uv-requirements.txt /uv-requirements.txt
COPY docker/entrypoint/serve.sh /app/entrypoint.sh
RUN mkdir -p /app/model/model && \
    mkdir -p /app/model/retrained && \
    chown -R $CMS_UID:$CMS_GID /app

RUN python3.11 -m pip install --require-hashes -r /uv-requirements.txt --no-cache-dir
RUN if [ -z "$PIP_EXTRAS" ]; then \
        uv sync --no-dev --python 3.11; \
    else \
        uv sync --no-dev --python 3.11 $(printf "%s" "$PIP_EXTRAS" | tr ',' ' ' | sed 's/\([^ ]*\)/--extra \1/g'); \
    fi
RUN /.venv/bin/python -m ensurepip && \
    chown -R $CMS_UID:$CMS_GID /.venv && \
    chmod +x /app/entrypoint.sh

WORKDIR /app
EXPOSE 8000
USER cms:cms
CMD ["bash", "-c", "./entrypoint.sh"]
