import os
import tempfile
import pytest
from unittest.mock import Mock
from medcat.cat import CAT
from app.config import Settings
from app.model_services.medcat_model_deid import MedCATModelDeIdentification


MODEL_PARENT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "resources")


@pytest.fixture(scope="session", autouse=True)
def medcat_model():
    config = Settings()
    config.BASE_MODEL_FILE = "deid_model.zip"
    return MedCATModelDeIdentification(config, MODEL_PARENT_DIR, True)


def test_model_name(medcat_model):
    assert medcat_model.model_name == "De-Identification MedCAT model"


def test_api_version(medcat_model):
    assert medcat_model.api_version == "0.0.1"


def test_of(medcat_model):
    new_model_service = medcat_model.of(medcat_model.model)
    assert isinstance(new_model_service, MedCATModelDeIdentification)
    assert new_model_service.model == medcat_model.model


@pytest.mark.skipif(not os.path.exists(os.path.join(MODEL_PARENT_DIR, "deid_model.zip")),
                    reason="requires the model file to be present in the resources folder")
def test_init_model(medcat_model):
    medcat_model.init_model()
    assert medcat_model.model is not None


@pytest.mark.skipif(not os.path.exists(os.path.join(MODEL_PARENT_DIR, "deid_model.zip")),
                    reason="requires the model file to be present in the resources folder")
def test_load_model(medcat_model):
    cat = MedCATModelDeIdentification.load_model(os.path.join(os.path.dirname(__file__), "..", "..", "resources", "deid_model.zip"))
    assert type(cat) is CAT


@pytest.mark.skipif(not os.path.exists(os.path.join(MODEL_PARENT_DIR, "deid_model.zip")),
                    reason="requires the model file to be present in the resources folder")
def test_info(medcat_model):
    medcat_model.init_model()
    model_card = medcat_model.info()
    assert type(model_card.api_version) is str
    assert type(model_card.model_description) is str
    assert model_card.model_type == "MedCAT"


@pytest.mark.skipif(not os.path.exists(os.path.join(MODEL_PARENT_DIR, "deid_model.zip")),
                    reason="requires the model file to be present in the resources folder")
def test_annotate(medcat_model):
    medcat_model.init_model()
    annotations = medcat_model.annotate("This is a post code NW1 2DA")
    assert len(annotations) == 1
    assert type(annotations[0]["label_name"]) is str
    assert annotations[0]["start"] == 20
    assert annotations[0]["end"] == 27


@pytest.mark.skipif(not os.path.exists(os.path.join(MODEL_PARENT_DIR, "deid_model.zip")),
                    reason="requires the model file to be present in the resources folder")
def test_batch_annotate(medcat_model):
    medcat_model.init_model()
    annotation_list = medcat_model.batch_annotate(["This is a post code NW1 2DA", "This is a post code NW1 2DA"])
    assert len(annotation_list) == 2
    assert type(annotation_list[0][0]["label_name"]) is str
    assert type(annotation_list[1][0]["label_name"]) is str
    assert annotation_list[0][0]["start"] == annotation_list[1][0]["start"] == 20
    assert annotation_list[0][0]["end"] == annotation_list[1][0]["end"] == 27


@pytest.mark.skipif(not os.path.exists(os.path.join(MODEL_PARENT_DIR, "deid_model.zip")),
                    reason="requires the model file to be present in the resources folder")
def test_train_supervised(medcat_model):
    medcat_model.init_model()
    medcat_model._config.REDEPLOY_TRAINED_MODEL = "false"
    medcat_model._config.SKIP_SAVE_MODEL = "true"
    medcat_model._supervised_trainer = Mock()
    with tempfile.TemporaryFile("r+") as f:
        medcat_model.train_supervised(f, 1, 1, "training_id", "input_file_name")
    medcat_model._supervised_trainer.train.assert_called()