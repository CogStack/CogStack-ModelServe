import os
import pytest
import app.api.globals as cms_globals
from fastapi.testclient import TestClient
from unittest.mock import create_autospec
from app.api.api import get_model_server
from app.utils import get_settings, load_pydantic_object_from_dict
from app.model_services.huggingface_ner_model import HuggingFaceNerModel
from app.domain import ModelCard, ModelType

config = get_settings()
config.ENABLE_TRAINING_APIS = "true"
config.DISABLE_UNSUPERVISED_TRAINING = "false"
config.ENABLE_EVALUATION_APIS = "true"
config.ENABLE_PREVIEWS_APIS = "true"
config.AUTH_USER_ENABLED = "true"

TRAINER_EXPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "fixture", "trainer_export.json")
NOTE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "fixture", "note.txt")
ANOTHER_TRAINER_EXPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "fixture", "another_trainer_export.json")
TRAINER_EXPORT_MULTI_PROJS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "fixture", "trainer_export_multi_projs.json")
MULTI_TEXTS_FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "fixture", "sample_texts.json")
HF_DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "dataset", "huggingface_dataset.zip")

@pytest.fixture(scope="function")
def model_service():
    return create_autospec(HuggingFaceNerModel)


@pytest.fixture(scope="function")
def client(model_service):
    app = get_model_server(config, msd_overwritten=lambda: model_service)
    app.dependency_overrides[cms_globals.props.current_active_user] = lambda: None
    client = TestClient(app)
    yield client
    client.app.dependency_overrides.clear()


def test_train_unsupervised_with_hf_hub_dataset(model_service, client):
    model_card = load_pydantic_object_from_dict(
        ModelCard,
        {
            "api_version": "0.0.1",
            "model_description": "huggingface_ner_model_description",
            "model_type": ModelType.HUGGINGFACE_NER,
            "model_card": None,
            "labels": None,
        },
    )
    model_service.info.return_value = model_card
    model_service.train_unsupervised.return_value = (True, "experiment_id", "run_id")

    with open(HF_DATASET_PATH, "rb") as f:
        response = client.post(
            f"/train_unsupervised_with_hf_hub_dataset",
            files={"hf_dataset_package": ("huggingface_dataset.zip", f, "multipart/form-data")},
        )

    model_service.train_unsupervised.assert_called()
    assert response.json()["message"] == "Your training started successfully."
    assert all([key in response.json() for key in ["training_id", "experiment_id", "run_id"]])
