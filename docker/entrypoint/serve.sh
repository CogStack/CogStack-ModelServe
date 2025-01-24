#!/bin/bash

if [ -f "/app/model/model.zip" ]; then
    CMS_MODEL_FILE="/app/model/model.zip"
elif [ -f "/app/model/model.tar.gz" ]; then
    CMS_MODEL_FILE="/app/model/model.tar.gz"
else
    echo "Error: Neither /app/model/model.zip nor /app/model/model.tar.gz was found."
    echo "Did you correctly mount the model package to /app/model/ in the container?"
    exit 1
fi

python cli/cli.py serve --model-type ${CMS_MODEL_TYPE} --model-name "${CMS_MODEL_NAME}" --model-path "${CMS_MODEL_FILE}" --host 0.0.0.0 --port 8000