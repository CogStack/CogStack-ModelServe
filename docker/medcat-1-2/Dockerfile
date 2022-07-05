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

COPY app /app
COPY docker/medcat-1-2/requirements.txt requirements.txt

RUN apt-get update && \
    apt-get install -qy curl

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN pip install -U pip
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_md

WORKDIR /app
EXPOSE 8000
CMD ["python", "serve.py", "--model", "medcat_1_2", "--host", "0.0.0.0", "--port", "8000"]
