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

COPY docker/mlflow/gateway/requirements.txt /requirements.txt

RUN pip install --no-cache-dir pip~=22.2.1 &&\
    pip install --no-cache-dir -r requirements.txt

RUN mkdir -p ${MLFLOW_HOME}/../config
COPY docker/mlflow/gateway/config/config.yml ${MLFLOW_HOME}/../config/config.yml
RUN mkdir -p ${MLFLOW_HOME}/../scripts
COPY docker/mlflow/gateway/scripts/run.sh ${MLFLOW_HOME}/../scripts/run.sh
RUN chmod +x ${MLFLOW_HOME}/../scripts/run.sh

WORKDIR ${MLFLOW_HOME}
ENTRYPOINT ["./../scripts/run.sh"]
