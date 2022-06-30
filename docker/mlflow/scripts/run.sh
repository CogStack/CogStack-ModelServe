#!/bin/sh

mlflow server \
  --backend-store-uri "postgresql://${MLFLOW_DB_USERNAME}:${MLFLOW_DB_PASSWORD}@mlflow-db:5432/mlflow-backend-store" \
  --default-artifact-root "$DEFAULT_ARTIFACT_ROOT" \
  --serve-artifacts \
  --host 0.0.0.0 \
  --port 5000