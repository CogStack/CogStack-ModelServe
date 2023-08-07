#!/bin/sh

mlflow server \
  --backend-store-uri "postgresql://${MLFLOW_DB_USERNAME}:${MLFLOW_DB_PASSWORD}@mlflow-db:5432/mlflow-backend-store" \
  --artifacts-destination "${ARTIFACTS_DESTINATION}" \
  --default-artifact-root mlflow-artifacts:/ \
  --app-name basic-auth \
  --serve-artifacts \
  --host 0.0.0.0 \
  --port 5000
