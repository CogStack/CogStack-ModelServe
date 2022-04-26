FROM python:3.7

COPY ../.../app /app
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_md

RUN apt-get update && \
    apt-get -qy full-upgrade && \
    apt-get install -qy curl && \
    apt-get install -qy curl && \
    curl -sSL https://get.docker.com/ | sh

WORKDIR /app
EXPOSE 8000
CMD ["python", "serve.py", "--model", "medcat_1_2", "--host", "0.0.0.0", "--port", "8000"]
