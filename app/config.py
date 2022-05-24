from pydantic import BaseSettings


class Settings(BaseSettings):
    BASE_MODEL_FILE: str = "model.zip"
    TEMP_FOLDER: str = "temp"
    CODE_TYPE: str = "snomed"
    DEVICE: str = "cpu"
    INCLUDE_SPAN_TEXT: str = "false"
    CONCAT_SIMILAR_ENTITIES: str = "true"
