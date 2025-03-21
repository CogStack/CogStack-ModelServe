FROM python:3.10
LABEL "org.cogstack.model-serve"="medcat_deid"

ARG CMS_MODEL_NAME
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG CMS_UID=1000
ARG CMS_GID=1000

ENV CMS_MODEL_NAME=$CMS_MODEL_NAME
ENV CMS_MODEL_TYPE=medcat_deid
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
COPY docker/medcat-deid/requirements.txt /app/requirements.txt
COPY docker/entrypoint/serve.sh /app/entrypoint.sh
RUN mkdir -p /app/model/model && \
    mkdir -p /app/model/retrained && \
    chown -R $CMS_UID:$CMS_GID /app
RUN pip install --no-cache-dir -U pip &&\
    pip install --no-cache-dir -r /app/requirements.txt && \
    python -m spacy download en_core_web_md
RUN chmod +x /app/entrypoint.sh

WORKDIR /app
EXPOSE 8000
USER cms:cms
CMD ["/app/entrypoint.sh"]
