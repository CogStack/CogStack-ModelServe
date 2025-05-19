#!/bin/sh

app_name_option=""
if [ -n "$MLFLOW_BASIC_AUTH_ENABLED" ] && [ "$MLFLOW_BASIC_AUTH_ENABLED" = "true" ]; then
  app_name_option="--app-name basic-auth"
fi

gunicorn_option=""
if [ -n "$MLFLOW_NUM_OF_WORKERS" ]; then
  gunicorn_option="$gunicorn_option --workers $MLFLOW_NUM_OF_WORKERS"
fi

if [ -n "$MLFLOW_WORKER_TIMEOUT_SECONDS" ]; then
  gunicorn_option="$gunicorn_option --timeout $MLFLOW_WORKER_TIMEOUT_SECONDS"
fi

if [ -n "$MLFLOW_SERVER_DEBUG" ] && [ "$MLFLOW_SERVER_DEBUG" = "true" ]; then
  gunicorn_option="$gunicorn_option --log-level debug"
fi

mlflow server \
  --backend-store-uri "postgresql://${MLFLOW_DB_USERNAME}:${MLFLOW_DB_PASSWORD}@mlflow-db:5432/mlflow-backend-store" \
  --artifacts-destination "${ARTIFACTS_DESTINATION}" \
  --default-artifact-root mlflow-artifacts:/ \
  $app_name_option \
  --serve-artifacts \
  --host 0.0.0.0 \
  --port 5000 \
  --gunicorn-opts "$gunicorn_option"
