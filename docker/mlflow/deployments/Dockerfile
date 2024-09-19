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

COPY docker/mlflow/deployments/requirements.txt /requirements.txt

RUN pip install --no-cache-dir -U pip &&\
    pip install --no-cache-dir -r /requirements.txt

RUN mkdir -p ${MLFLOW_HOME}/../config
COPY docker/mlflow/deployments/config/config.yml ${MLFLOW_HOME}/../config/config.yml
RUN mkdir -p ${MLFLOW_HOME}/../scripts
COPY docker/mlflow/deployments/scripts/run.sh ${MLFLOW_HOME}/../scripts/run.sh
RUN chmod +x ${MLFLOW_HOME}/../scripts/run.sh

WORKDIR ${MLFLOW_HOME}
ENTRYPOINT ["./../scripts/run.sh"]
