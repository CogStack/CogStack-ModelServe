# Management

## MLflow Users
To enable user authentication and authorisation in MLflow, you will need to provide the following environement variables located in this [Docker Compose file](./../../docker-compose-auth.yml):

* MLFLOW_BASIC_AUTH_ENABLED=true
* MLFLOW_AUTH_CONFIG_PATH=/opt/auth/basic_auth.ini

Additionally, ensure you set the appropriate values in the default [basic auth file](./../../docker/mlflow/server/auth/basic_auth.ini) before firing up a container based off it. For detailed information on authentication, please refer to the [official documentation](https://mlflow.org/docs/2.6.0/auth/index.html).
