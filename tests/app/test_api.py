from app.config import Settings
from app.api import get_model_server
from app.dependencies import ModelServiceDep


def test_get_model_server():
    config = Settings()
    config.ENABLE_TRAINING_APIS = "true"
    config.DISABLE_UNSUPERVISED_TRAINING = "false"
    model_service_dep = ModelServiceDep("medcat_snomed", Settings())

    app = get_model_server(model_service_dep)
    info = app.openapi()["info"]
    tags = app.openapi_tags
    paths = [path for path in app.openapi()["paths"].keys()]

    assert isinstance(info["title"], str)
    assert isinstance(info["description"], str)
    assert isinstance(info["version"], str)
    assert {"name": "Metadata", "description": "Get the model card."} in tags
    assert {"name": "Annotations", "description": "Retrieve recognised entities by running the model."} in tags
    assert {"name": "Rendering", "description": "Get embeddable annotation snippet in HTML."} in tags
    assert {"name": "Training", "description": "Trigger model training on input annotations."} in tags
    assert "/info" in paths
    assert "/process" in paths
    assert "/process_bulk" in paths
    assert "/preview" in paths
    assert "/preview_trainer_export" in paths
    assert "/train_supervised" in paths
    assert "/train_unsupervised" in paths
