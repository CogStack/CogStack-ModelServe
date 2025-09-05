from app.api.api import get_model_server, get_stream_server, get_generative_server
from app.api.dependencies import ModelServiceDep
from app.utils import get_settings


def test_get_model_server():
    config = get_settings()
    config.ENABLE_TRAINING_APIS = "true"
    config.DISABLE_UNSUPERVISED_TRAINING = "false"
    config.ENABLE_EVALUATION_APIS = "true"
    config.ENABLE_PREVIEWS_APIS = "true"
    config.AUTH_USER_ENABLED = "true"

    model_service_dep = ModelServiceDep("medcat_snomed", config)
    app = get_model_server(config, model_service_dep)
    info = app.openapi()["info"]
    tags = app.openapi_tags
    paths = [route.path for route in app.routes]

    assert isinstance(info["title"], str)
    assert isinstance(info["summary"], str)
    assert isinstance(info["version"], str)
    assert {"name": "Metadata", "description": "Get the model card"} in tags
    assert {"name": "Annotations", "description": "Retrieve NER entities by running the model"} in tags
    assert {"name": "Redaction", "description": "Redact the extracted NER entities"} in tags
    assert {"name": "Rendering", "description": "Preview embeddable annotation snippet in HTML"} in tags
    assert {"name": "Training", "description": "Trigger model training on input annotations"} in tags
    assert {"name": "Evaluating", "description": "Evaluate the deployed model with trainer export"} in tags
    assert {"name": "Authentication", "description": "Authenticate registered users"} in tags
    assert "/info" in paths
    assert "/process" in paths
    assert "/process_jsonl" in paths
    assert "/process_bulk" in paths
    assert "/process_bulk_file" in paths
    assert "/redact" in paths
    assert "/redact_with_encryption" in paths
    assert "/preview" in paths
    assert "/preview_trainer_export" in paths
    assert "/train_supervised" in paths
    assert "/train_unsupervised" in paths
    assert "/train_unsupervised_with_hf_hub_dataset" in paths
    assert "/train_metacat" in paths
    assert "/train_eval_info" in paths
    assert "/train_eval_metrics" in paths
    assert "/cancel_training" in paths
    assert "/evaluate" in paths
    assert "/sanity-check" in paths
    assert "/iaa-scores" in paths
    assert "/concat_trainer_exports" in paths
    assert "/auth/jwt/login" in paths
    assert "/auth/jwt/logout" in paths
    assert "/healthz" in paths
    assert "/readyz" in paths
    assert "/metrics" not in paths


def test_get_stream_server():
    config = get_settings()
    config.AUTH_USER_ENABLED = "true"

    model_service_dep = ModelServiceDep("medcat_snomed", config)
    app = get_stream_server(config, model_service_dep)
    info = app.openapi()["info"]
    tags = app.openapi_tags
    paths = [route.path for route in app.routes]

    assert isinstance(info["title"], str)
    assert isinstance(info["summary"], str)
    assert isinstance(info["version"], str)
    assert {"name": "Streaming", "description": "Retrieve NER entities as a stream by running the model"} in tags
    assert "/info" in paths
    assert "/stream/process" in paths
    assert "/stream/ws" in paths
    assert "/auth/jwt/login" in paths
    assert "/auth/jwt/logout" in paths
    assert "/healthz" in paths
    assert "/readyz" in paths
    assert "/metrics" not in paths

def test_get_generative_server():
    config = get_settings()
    config.AUTH_USER_ENABLED = "true"

    model_service_dep = ModelServiceDep("huggingface_llm_model", config)
    app = get_generative_server(config, model_service_dep)
    info = app.openapi()["info"]
    tags = app.openapi_tags
    paths = [route.path for route in app.routes]

    assert isinstance(info["title"], str)
    assert isinstance(info["summary"], str)
    assert isinstance(info["version"], str)
    assert {"name": "Metadata", "description": "Get the model card"} in tags
    assert {"name": "Generative", "description": "Generate text based on the input prompt"} in tags
    assert "/info" in paths
    assert "/generate" in paths
    assert "/stream/generate" in paths
    assert "/healthz" in paths
    assert "/readyz" in paths
