#!/bin/sh

mlflow server --backend-store-uri $BACKEND_STORE_URI --host 0.0.0.0 --port 5000