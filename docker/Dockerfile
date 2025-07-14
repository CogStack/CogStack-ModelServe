FROM python:3.10

ARG CMS_UID=1000
ARG CMS_GID=1000
ARG CMS_MODEL_NAME
ARG CMS_MODEL_TYPE
ARG CMS_STREAMABLE
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

ENV CMS_MODEL_NAME=$CMS_MODEL_NAME
ENV CMS_MODEL_TYPE=$CMS_MODEL_TYPE
ENV CMS_STREAMABLE=$CMS_STREAMABLE
ENV HTTP_PROXY=$HTTP_PROXY
ENV HTTPS_PROXY=$HTTPS_PROXY
ENV NO_PROXY=$NO_PROXY
ENV http_proxy=$HTTP_PROXY
ENV https_proxy=$HTTPS_PROXY
ENV no_proxy=$NO_PROXY
ENV PYTHONUNBUFFERED=1
ENV PATH="/home/cms/.local/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends nano telnet && \
    rm -rf /var/lib/apt/lists/*
RUN addgroup --gid $CMS_GID cms || true && \
    adduser --uid $CMS_UID --gid $CMS_GID --disabled-password --gecos "" cms || true && \
    echo "cms ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

COPY app /app
COPY pyproject.toml /pyproject.toml
COPY uv.lock /uv.lock
COPY docker/entrypoint/serve.sh /app/entrypoint.sh
RUN mkdir -p /app/model/model && \
    mkdir -p /app/model/retrained && \
    chown -R $CMS_UID:$CMS_GID /app
RUN pip install --no-cache-dir uv==0.7.20 && \
    uv sync --locked && \
    chmod +x /app/entrypoint.sh

WORKDIR /app
EXPOSE 8000
USER cms:cms
CMD ["uv", "run", "/app/entrypoint.sh"]
