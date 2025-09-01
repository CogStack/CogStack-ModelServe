#!/bin/bash

if [ -z "${CMS_MODEL_TYPE}" ]; then
    echo "Error: CMS_MODEL_TYPE is required but not set."
    echo "Please set the CMS_MODEL_TYPE environment variable to the type of the model you want to serve."
    exit 1
fi

if [ -z "${CMS_MODEL_NAME}" ]; then
    echo "Error: CMS_MODEL_NAME is required but not set."
    echo "Please set the CMS_MODEL_NAME environment variable to your preferred model name."
    exit 1
fi

if [ -f "/app/model/model.zip" ]; then
    CMS_MODEL_FILE="/app/model/model.zip"
elif [ -f "/app/model/model.tar.gz" ]; then
    CMS_MODEL_FILE="/app/model/model.tar.gz"
else
    echo "Error: Neither /app/model/model.zip nor /app/model/model.tar.gz was found."
    echo "Did you correctly mount the model package to /app/model/ in the container?"
    exit 1
fi

if [ "${CMS_STREAMABLE}" = "true" ]; then
    streamable="--streamable"
else
    streamable=""
fi

source /.venv/bin/activate

exec /.venv/bin/python cli/cli.py serve \
  --model-type "${CMS_MODEL_TYPE}" \
  --model-name "${CMS_MODEL_NAME}" \
  --model-path "${CMS_MODEL_FILE}" \
  --host 0.0.0.0 \
  --port 8000 \
  $streamable