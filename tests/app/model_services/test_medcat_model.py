import os
import tempfile
import pytest
from unittest.mock import Mock, patch
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
    assert type(annotations[0]["label_name"]) is str
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
        with tempfile.TemporaryFile("r+") as f:
            medcat_model.train_unsupervised(f, 1, 1, "training_id", "input_file_name")
        start_training.assert_called()


def test_send_metrics(medcat_model):
    medcat_model._tracker_client = Mock()
    medcat_model.glean_and_log_metrics("Epoch: 0, Prec: 0.01, Rec: 0.01, F1: 0.01")
    medcat_model._tracker_client.send_model_stats.assert_called_once_with({"precision": 0.01, "recall": 0.01, "f1": 0.01}, 0)


def test_get_cuis_from_trainer_export(medcat_model):
    path = os.path.join(MODEL_PARENT_DIR, "fixture", "trainer_export.json")
    cuis = medcat_model.get_cuis_from_trainer_export(path)
    assert cuis == {'C0010068', 'C0011860', 'C0003864', 'C0011849', 'C0878544', 'C0020473', 'C0155626', 'C0007222',
                    'C0012634', 'C0020538', 'C0038454', 'C0042029', 'C0007787', 'C0027051', 'C0017168', 'C0338614',
                    'C0037284'}