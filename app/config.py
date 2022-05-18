from pydantic import BaseSettings


class Settings(BaseSettings):
    BASE_MODEL_FILE: str = "model.zip"
    TEMP_FOLDER: str = "temp"
    CODE_TYPE: str = "snomed"
    DEVICE: str = "cpu"
    INCLUDE_ANNOTATION_TEXT: str = "false"
