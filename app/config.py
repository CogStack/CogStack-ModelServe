import os
from pydantic import BaseSettings


class Settings(BaseSettings):
    BASE_MODEL_FILE: str = "model.zip"
    CODE_TYPE: str = "snomed"
    DEVICE: str = "cpu"
    INCLUDE_SPAN_TEXT: str = "false"
    CONCAT_SIMILAR_ENTITIES: str = "true"

    class Config:
        env_file = os.path.join(os.path.dirname(__file__), "envs", ".env")
        env_file_encoding = "utf-8"
