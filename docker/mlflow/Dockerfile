FROM python:3.8

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

ENV HTTP_PROXY=$HTTP_PROXY
ENV HTTPS_PROXY=$HTTPS_PROXY
ENV NO_PROXY=$NO_PROXY
ENV http_proxy=$HTTP_PROXY
ENV https_proxy=$HTTPS_PROXY
ENV no_proxy=$NO_PROXY
ENV BACKEND_STORE_URI=$BACKEND_STORE_URI
ENV MLFLOW_HOME=/opt/mlflow

COPY docker/mlflow/requirements.txt requirements.txt

RUN pip install -U pip
RUN pip install -r requirements.txt

RUN mkdir -p ${MLFLOW_HOME}/scripts
COPY docker/mlflow/scripts/run.sh ${MLFLOW_HOME}/scripts/run.sh
RUN chmod +x ${MLFLOW_HOME}/scripts/run.sh

WORKDIR ${MLFLOW_HOME}
ENTRYPOINT ["./scripts/run.sh"]
