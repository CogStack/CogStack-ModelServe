import os
import tempfile
import pytest
from unittest.mock import patch
from medcat.cat import CAT
from app.config import Settings
from app.model_services.medcat_model import MedCATModel


MODEL_PARENT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "resources")


@pytest.fixture(scope="session", autouse=True)
def medcat_model():
    return MedCATModel(Settings(), MODEL_PARENT_DIR)


@pytest.mark.skipif(not os.path.exists(os.path.join(MODEL_PARENT_DIR, "model.zip")),
                    reason="requires the model file to be present in the resources folder")
def test_init_model(medcat_model):
    medcat_model.init_model()
    assert medcat_model.model is not None


@pytest.mark.skipif(not os.path.exists(os.path.join(MODEL_PARENT_DIR, "model.zip")),
                    reason="requires the model file to be present in the resources folder")
def test_load_model(medcat_model):
    cat = MedCATModel.load_model(os.path.join(os.path.dirname(__file__), "..", "..", "resources", "model.zip"))
    assert type(cat) is CAT


@pytest.mark.skipif(not os.path.exists(os.path.join(MODEL_PARENT_DIR, "model.zip")),
                    reason="requires the model file to be present in the resources folder")
def test_info(medcat_model):
    medcat_model.init_model()
    model_card = medcat_model.info()
    assert type(model_card.api_version) is str
    assert type(model_card.model_description) is str
    assert model_card.model_type == "MedCAT"


@pytest.mark.skipif(not os.path.exists(os.path.join(MODEL_PARENT_DIR, "model.zip")),
                    reason="requires the model file to be present in the resources folder")
def test_annotate(medcat_model):
    medcat_model.init_model()
    annotations = medcat_model.annotate("Spinal stenosis")
    assert len(annotations) == 1
    assert annotations[0]["label_name"] == "Spinal Stenosis"
    assert annotations[0]["start"] == 0
    assert annotations[0]["end"] == 15


@pytest.mark.skipif(not os.path.exists(os.path.join(MODEL_PARENT_DIR, "model.zip")),
                    reason="requires the model file to be present in the resources folder")
def test_train_supervised(medcat_model):
    medcat_model.init_model()
    medcat_model._config.REDEPLOY_TRAINED_MODEL = "false"
    medcat_model._config.SKIP_SAVE_MODEL = "true"
    with patch("app.model_services.medcat_model.MedCATModel._start_training", autospec=True) as start_training:
        with tempfile.TemporaryFile("r+") as f:
            medcat_model.train_supervised(f, 1, 1, "training_id", "input_file_name")
        start_training.assert_called()


@pytest.mark.skipif(not os.path.exists(os.path.join(MODEL_PARENT_DIR, "model.zip")),
                    reason="requires the model file to be present in the resources folder")
def test_train_unsupervised(medcat_model):
    medcat_model.init_model()
    medcat_model._config.REDEPLOY_TRAINED_MODEL = "false"
    medcat_model._config.SKIP_SAVE_MODEL = "true"
    with patch("app.model_services.medcat_model.MedCATModel._start_training", autospec=True) as start_training:
        medcat_model.train_unsupervised(["Spinal Stenosis"], 1, 1, "training_id", "input_file_name")
        start_training.assert_called()
