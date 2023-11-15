
# Build an image that can serve mlflow models.
FROM ubuntu:20.04

# Customised
ARG MLFLOW_MODEL_NAME
ARG ARCH
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

ENV HTTP_PROXY=$HTTP_PROXY
ENV HTTPS_PROXY=$HTTPS_PROXY
ENV NO_PROXY=$NO_PROXY
ENV http_proxy=$HTTP_PROXY
ENV https_proxy=$HTTPS_PROXY
ENV no_proxy=$NO_PROXY
ENV PYTHONUNBUFFERED=1

RUN apt-get -y update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y --no-install-recommends          wget          curl          nginx          ca-certificates          bzip2          build-essential          cmake          openjdk-8-jdk          git-core          maven     && rm -rf /var/lib/apt/lists/*



# Setup pyenv
RUN apt -y update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get install -y \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
RUN git clone \
    --depth 1 \
    --branch $(git ls-remote --tags --sort=v:refname https://github.com/pyenv/pyenv.git | grep -o -E 'v[1-9]+(\.[1-9]+)+$' | tail -1) \
    https://github.com/pyenv/pyenv.git /root/.pyenv
ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PATH"
# Customised
RUN apt install -y python3.9 libpython3.9-dev python3.9-distutils
RUN ln -s -f $(which python3.9) /usr/bin/python
RUN wget https://bootstrap.pypa.io/get-pip.py -O /tmp/get-pip.py
RUN python /tmp/get-pip.py
RUN pip install virtualenv

# Customised
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-$ARCH
# Customised
ENV GUNICORN_CMD_ARGS="--timeout 120 -k gevent --workers=1 --bind=0.0.0.0:8000"
# Set up the program in the image
WORKDIR /opt/mlflow

# Customised
COPY docker/mlflow/models/requirements.txt /requirements.txt
RUN pip install --no-cache-dir pip~=22.2.1 &&\
    pip install --no-cache-dir -r /requirements.txt

RUN mvn --batch-mode dependency:copy -Dartifact=org.mlflow:mlflow-scoring:2.8.0:pom -DoutputDirectory=/opt/java 
RUN mvn --batch-mode dependency:copy -Dartifact=org.mlflow:mlflow-scoring:2.8.0:jar -DoutputDirectory=/opt/java/jars 
RUN cp /opt/java/mlflow-scoring-2.8.0.pom /opt/java/pom.xml
RUN cd /opt/java && mvn --batch-mode dependency:copy-dependencies -DoutputDirectory=/opt/java/jars



COPY docker/mlflow/models/model_dir/$MLFLOW_MODEL_NAME /opt/ml/model
RUN python -c                     'from mlflow.models.container import _install_pyfunc_deps;                    _install_pyfunc_deps(                        "/opt/ml/model",                         install_mlflow=False,                         enable_mlserver=False,                         env_manager="virtualenv")'
ENV MLFLOW_DISABLE_ENV_CREATION="true"
ENV ENABLE_MLSERVER=False
# Customised
ENV DISABLE_NGINX=true


# granting read/write access and conditional execution authority to all child directories
# and files to allow for deployment to AWS Sagemaker Serverless Endpoints
# (see https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html)
RUN chmod o+rwX /opt/mlflow/

ENTRYPOINT ["python", "-c", "from mlflow.models import container as C;C._serve('virtualenv')"]
