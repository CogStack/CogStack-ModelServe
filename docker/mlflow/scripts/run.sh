#!/bin/sh

mlflow server \
  --backend-store-uri "$BACKEND_STORE_URI" \
  --default-artifact-root "$DEFAULT_ARTIFACT_ROOT" \
  --host 0.0.0.0 \
  --port 5000