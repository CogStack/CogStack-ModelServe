from api.api import get_model_server, get_stream_server
from api.dependencies import ModelServiceDep
from utils import get_settings


def test_get_model_server():
    config = get_settings()
    config.ENABLE_TRAINING_APIS = "true"
    config.DISABLE_UNSUPERVISED_TRAINING = "false"
    config.ENABLE_EVALUATION_APIS = "true"
    config.ENABLE_PREVIEWS_APIS = "true"
    config.AUTH_USER_ENABLED = "true"

    model_service_dep = ModelServiceDep("medcat_snomed", config)
    app = get_model_server(model_service_dep)
    info = app.openapi()["info"]
    tags = app.openapi_tags
    paths = [path for path in app.openapi()["paths"].keys()]

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
    assert "/process_stream" in paths
    assert "/process_bulk" in paths
    assert "/process_bulk_file" in paths
    assert "/redact" in paths
    assert "/redact_with_encryption" in paths
    assert "/preview" in paths
    assert "/preview_trainer_export" in paths
    assert "/train_supervised" in paths
    assert "/train_unsupervised" in paths
    assert "/train_metacat" in paths
    assert "/evaluate" in paths
    assert "/sanity-check" in paths
    assert "/iaa-scores" in paths
    assert "/concat_trainer_exports" in paths
    assert "/auth/jwt/login" in paths
    assert "/auth/jwt/logout" in paths
    assert "/metrics" not in paths
    assert "/healthz" not in paths
    assert "/readyz" not in paths


def test_get_stream_server():
    config = get_settings()
    config.AUTH_USER_ENABLED = "true"

    model_service_dep = ModelServiceDep("medcat_snomed", config)
    app = get_stream_server(model_service_dep)
    info = app.openapi()["info"]
    tags = app.openapi_tags
    paths = [path for path in app.openapi()["paths"].keys()]

    assert isinstance(info["title"], str)
    assert isinstance(info["summary"], str)
    assert isinstance(info["version"], str)
    assert {"name": "Streaming", "description": "Retrieve NER entities as a stream by running the model"} in tags
    assert "/stream/process" in paths
    assert "/auth/jwt/login" in paths
    assert "/auth/jwt/logout" in paths
    assert "/metrics" not in paths
    assert "/healthz" not in paths
    assert "/readyz" not in paths
