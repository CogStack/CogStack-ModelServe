import os
from pydantic import BaseSettings


class Settings(BaseSettings):
    BASE_MODEL_FILE: str = "model.zip"
    BASE_MODEL_FULL_PATH: str = ""
    DEVICE: str = "cpu"
    INCLUDE_SPAN_TEXT: str = "false"
    CONCAT_SIMILAR_ENTITIES: str = "true"
    ENABLE_TRAINING_APIS: str = "false"
    DISABLE_UNSUPERVISED_TRAINING: str = "false"
    DISABLE_METACAT_TRAINING: str = "true"
    ENABLE_EVALUATION_APIS: str = "false"
    ENABLE_PREVIEWS_APIS: str = "false"
    MLFLOW_TRACKING_URI: str = f'file:{os.path.join(os.path.abspath(os.path.dirname(__file__)), "mlruns")}'
    REDEPLOY_TRAINED_MODEL: str = "false"
    SKIP_SAVE_MODEL: str = "false"
    SKIP_SAVE_TRAINING_DATASET: str = "true"
    PROCESS_RATE_LIMIT: str = "180/minute"
    PROCESS_BULK_RATE_LIMIT: str = "90/minute"
    TYPE_UNIQUE_ID_WHITELIST: str = ""  # empty means all TUIs are whitelisted

    class Config:
        env_file = os.path.join(os.path.dirname(__file__), "envs", ".env")
        env_file_encoding = "utf-8"
