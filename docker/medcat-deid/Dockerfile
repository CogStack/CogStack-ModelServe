FROM python:3.10
LABEL "org.cogstack.model_serve"="medcat_deid"

ARG CMS_MODEL_NAME
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

ENV CMS_MODEL_NAME=$CMS_MODEL_NAME
ENV HTTP_PROXY=$HTTP_PROXY
ENV HTTPS_PROXY=$HTTPS_PROXY
ENV NO_PROXY=$NO_PROXY
ENV http_proxy=$HTTP_PROXY
ENV https_proxy=$HTTPS_PROXY
ENV no_proxy=$NO_PROXY
ENV PYTHONUNBUFFERED=1

COPY docker/medcat-deid/requirements.txt /requirements.txt
RUN pip install --no-cache-dir pip~=22.2.1 &&\
    pip install --no-cache-dir -r requirements.txt &&\
    pip install --no-cache-dir transformers==4.21.3 # pin down the version due to incompatibility with the base model

RUN python -m spacy download en_core_web_md

COPY app /app
WORKDIR /app
EXPOSE 8000
CMD ["sh", "-c", "python cli/cli.py serve --model-type medcat_deid --model-name \"${CMS_MODEL_NAME}\" --model-path /app/model/model.zip --host 0.0.0.0 --port 8000"]
