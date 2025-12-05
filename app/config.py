import os
import json
try:
    from pydantic import BaseSettings
except ImportError:
    from pydantic.v1 import BaseSettings


class Settings(BaseSettings):   # type: ignore
    BASE_MODEL_FILE: str = "model.zip"                # the base name of the model file
    BASE_MODEL_FULL_PATH: str = ""                    # the full path to the model file
    DEVICE: str = "default"                           # the device literal, either "default", "cpu[:X]", "cuda[:X]" or "mps[:X]"
    INCLUDE_SPAN_TEXT: str = "false"                  # if "true", include the text of the entity in the NER output
    CONCAT_SIMILAR_ENTITIES: str = "true"             # if "true", merge adjacent entities of the same type into one span
    ENABLE_TRAINING_APIS: str = "false"               # if "true", enable the APIs for model training
    DISABLE_UNSUPERVISED_TRAINING: str = "false"      # if "true", disable the API for unsupervised training
    DISABLE_METACAT_TRAINING: str = "true"            # if "true", disable the API for metacat training
    ENABLE_EVALUATION_APIS: str = "false"             # if "true", enable the APIs for evaluating the model being served
    ENABLE_PREVIEWS_APIS: str = "false"               # if "true", enable the APIs for previewing the NER output
    MLFLOW_TRACKING_URI: str = f'file:{os.path.join(os.path.abspath(os.path.dirname(__file__)), "mlruns")}'     # the mlflow tracking URI
    REDEPLOY_TRAINED_MODEL: str = "false"             # if "true", replace the running model with the newly trained one
    SKIP_SAVE_MODEL: str = "false"                    # if "true", newly trained models won't be saved but training metrics will be collected
    SKIP_SAVE_TRAINING_DATASET: str = "true"          # if "true", the dataset used for training won't be saved
    PROCESS_RATE_LIMIT: str = "180/minute"            # the rate limit on the /process route
    PROCESS_BULK_RATE_LIMIT: str = "90/minute"        # the rate limit on the /process_bulk route
    WS_IDLE_TIMEOUT_SECONDS: int = 60                 # the timeout in seconds on the WebSocket connection being idle
    TYPE_UNIQUE_ID_WHITELIST: str = ""                # the comma-separated TUIs used for filtering and if set to "", all TUIs are whitelisted
    AUTH_USER_ENABLED: str = "false"                  # if "true", enable user authentication on API access
    AUTH_JWT_SECRET: str = ""                         # the JWT secret and will be ignored if AUTH_USER_ENABLED is not "true"
    AUTH_ACCESS_TOKEN_EXPIRE_SECONDS: int = 3600      # the seconds after which the JWT will expire
    AUTH_DATABASE_URL: str = "sqlite+aiosqlite:///./cms-users.db"     # the URL of the authentication database
    SYSTEM_METRICS_LOGGING_INTERVAL_SECONDS: int = 30 # if set, enable the logging on system metrics and set the interval in seconds
    TRAINING_CONCEPT_ID_WHITELIST: str = ""           # the comma-separated concept IDs used for filtering annotations of interest
    TRAINING_METRICS_LOGGING_INTERVAL: int = 5        # the number of steps after which training metrics will be collected
    TRAINING_SAFE_MODEL_SERIALISATION: str = "false"  # if "true", serialise the trained model using safe tensors
    TRAINING_CACHE_DIR: str = os.path.join(os.path.abspath(os.path.dirname(__file__)), "cms_cache")           # the directory to cache the intermediate files created during training
    TRAINING_HF_TAGGING_SCHEME: str = "flat"          # the tagging scheme during the Hugging Face NER model training, either "flat", "iob" or "iobes"
    HF_PIPELINE_AGGREGATION_STRATEGY: str = "simple"  # the strategy used for aggregating the predictions of the Hugging Face NER model
    LOG_PER_CONCEPT_ACCURACIES: str = "false"         # if "true", per-concept accuracies will be exposed to the metrics scrapper. Switch this on with caution due to the potentially high number of concepts
    MEDCAT2_MAPPED_ONTOLOGIES: str = ""               # the comma-separated names of ontologies for MedCAT2 to map to
    DEBUG: str = "false"                              # if "true", the debug mode is switched on

    class Config:
        env_file = os.path.join(os.path.dirname(__file__), "envs", ".env")
        env_file_encoding = "utf-8"

    def __hash__(self) -> int:
        return hash(json.dumps(vars(self)))
