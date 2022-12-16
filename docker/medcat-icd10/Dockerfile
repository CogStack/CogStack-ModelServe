FROM python:3.8

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

COPY app /app
COPY docker/medcat-icd10/requirements.txt /requirements.txt

RUN pip install --no-cache-dir pip~=22.2.1 &&\
    pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_md

WORKDIR /app
EXPOSE 8000
CMD ["python", "cli/cli.py", "serve", "--model-type", "medcat_icd10", "--model-name", "'${CMS_MODEL_NAME}'", "--model-path", "/app/model/model.zip", "--host", "0.0.0.0", "--port", "8000"]
