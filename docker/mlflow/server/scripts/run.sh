#!/bin/sh

if [ -n "$MLFLOW_BASIC_AUTH_ENABLED" ] && [ "$MLFLOW_BASIC_AUTH_ENABLED" = "true" ]; then
  app_name_option="--app-name basic-auth"
else
  app_name_option=""
fi

if [ -n "$MLFLOW_SERVER_DEBUG" ] && [ "$MLFLOW_SERVER_DEBUG" = "true" ]; then
  debug_option="--gunicorn-opts='--log-level=debug'"
else
  debug_option=""
fi

mlflow server \
  --backend-store-uri "postgresql://${MLFLOW_DB_USERNAME}:${MLFLOW_DB_PASSWORD}@mlflow-db:5432/mlflow-backend-store" \
  --artifacts-destination "${ARTIFACTS_DESTINATION}" \
  --default-artifact-root mlflow-artifacts:/ \
  $app_name_option \
  --serve-artifacts \
  --host 0.0.0.0 \
  --port 5000 \
  $debug_option
