FROM python:3.10

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

ENV HTTP_PROXY=$HTTP_PROXY
ENV HTTPS_PROXY=$HTTPS_PROXY
ENV NO_PROXY=$NO_PROXY
ENV http_proxy=$HTTP_PROXY
ENV https_proxy=$HTTPS_PROXY
ENV no_proxy=$NO_PROXY
ENV MLFLOW_HOME=/opt/mlflow
ENV PYTHONUNBUFFERED=1

RUN apt-get -y update && apt-get install -y --no-install-recommends cron && \
    rm -rf /var/lib/apt/lists/*

COPY docker/mlflow/server/crontab.txt /etc/cron.d/mlflow_gc
RUN chmod 0644 /etc/cron.d/mlflow_gc && \
    crontab /etc/cron.d/mlflow_gc && \
    touch /var/log/cron.log && \
    cron

COPY docker/mlflow/server/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r /requirements.txt

RUN mkdir -p ${MLFLOW_HOME}/../auth
COPY docker/mlflow/server/auth/basic_auth.ini ${MLFLOW_HOME}/../auth/basic_auth.ini
RUN mkdir -p ${MLFLOW_HOME}/../scripts
COPY docker/mlflow/server/scripts/run.sh ${MLFLOW_HOME}/../scripts/run.sh
RUN chmod +x ${MLFLOW_HOME}/../scripts/run.sh

WORKDIR ${MLFLOW_HOME}
ENTRYPOINT ["./../scripts/run.sh"]