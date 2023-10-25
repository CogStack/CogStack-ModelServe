import os
import tempfile
import pytest
from unittest.mock import Mock
from medcat.cat import CAT
from config import Settings
from model_services.medcat_model_icd10 import MedCATModelIcd10


MODEL_PARENT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "resources")


@pytest.fixture(scope="function")
def medcat_model():
    config = Settings()
    config.BASE_MODEL_FILE = "icd10_model.zip"
    config.TYPE_UNIQUE_ID_WHITELIST = "T-9,T-11,T-18,T-39,T-40,T-45"
    return MedCATModelIcd10(config, MODEL_PARENT_DIR, True)


def test_model_name(medcat_model):
    assert medcat_model.model_name == "ICD-10 MedCAT model"


def test_api_version(medcat_model):
    assert medcat_model.api_version == "0.0.1"


def test_from_model(medcat_model):
    new_model_service = medcat_model.from_model(medcat_model.model)
    assert isinstance(new_model_service, MedCATModelIcd10)
    assert new_model_service.model == medcat_model.model


def test_get_records_from_doc(medcat_model):
    records = medcat_model.get_records_from_doc({"entities": {"0": {"pretty_name": "pretty_name", "cui": "cui", "types": ["type"], "icd10": [{"code": "code", "name": "name"}], "acc": 1.0, "meta_anns": {}}}})
    assert len(records) == 1
    assert records[0]["label_name"] == "name"
    assert records[0]["cui"] == "cui"
    assert records[0]["label_id"] == "code"
    assert records[0]["categories"] == ["type"]
    assert records[0]["accuracy"] == 1.0
    assert records[0]["meta_anns"] == {}


@pytest.mark.skipif(not os.path.exists(os.path.join(MODEL_PARENT_DIR, "icd10_model.zip")),
                    reason="requires the model file to be present in the resources folder")
def test_init_model_with_no_tui_filter(medcat_model):
    original = MedCATModelIcd10.load_model(os.path.join(os.path.dirname(__file__), "..", "..", "resources", "icd10_model.zip"))
    medcat_model._whitelisted_tuis = set([""])
    medcat_model.init_model()
    assert medcat_model.model is not None
    assert medcat_model.model.cdb.config.linking.filters.get("cuis") == original.cdb.config.linking.filters.get("cuis")


@pytest.mark.skipif(not os.path.exists(os.path.join(MODEL_PARENT_DIR, "icd10_model.zip")),
                    reason="requires the model file to be present in the resources folder")
def test_init_model(medcat_model):
    medcat_model.init_model()
    target_tuis = medcat_model._config.TYPE_UNIQUE_ID_WHITELIST.split(",")
    target_cuis = {cui for tui in target_tuis for cui in medcat_model.model.cdb.addl_info.get("type_id2cuis").get(tui, {})}
    assert medcat_model.model is not None
    assert medcat_model.model.cdb.config.linking.filters.get("cuis") == target_cuis


@pytest.mark.skipif(not os.path.exists(os.path.join(MODEL_PARENT_DIR, "icd10_model.zip")),
                    reason="requires the model file to be present in the resources folder")
def test_load_model(medcat_model):
    cat = MedCATModelIcd10.load_model(os.path.join(os.path.dirname(__file__), "..", "..", "resources", "icd10_model.zip"))
    assert type(cat) is CAT


@pytest.mark.skipif(not os.path.exists(os.path.join(MODEL_PARENT_DIR, "icd10_model.zip")),
                    reason="requires the model file to be present in the resources folder")
def test_info(medcat_model):
    medcat_model.init_model()
    model_card = medcat_model.info()
    assert type(model_card.api_version) is str
    assert type(model_card.model_description) is str
    assert model_card.model_type == "MedCAT"


@pytest.mark.skipif(not os.path.exists(os.path.join(MODEL_PARENT_DIR, "icd10_model.zip")),
                    reason="requires the model file to be present in the resources folder")
def test_annotate(medcat_model):
    medcat_model.init_model()
    annotations = medcat_model.annotate("Spinal stenosis")
    assert len(annotations) == 1
    assert type(annotations[0]["label_name"]) is str
    assert annotations[0]["start"] == 0
    assert annotations[0]["end"] == 15
    assert annotations[0]["accuracy"] > 0


@pytest.mark.skipif(not os.path.exists(os.path.join(MODEL_PARENT_DIR, "icd10_model.zip")),
                    reason="requires the model file to be present in the resources folder")
def test_train_supervised(medcat_model):
    medcat_model.init_model()
    medcat_model._config.REDEPLOY_TRAINED_MODEL = "false"
    medcat_model._config.SKIP_SAVE_MODEL = "true"
    medcat_model._supervised_trainer = Mock()
    with tempfile.TemporaryFile("r+") as f:
        medcat_model.train_supervised(f, 1, 1, "training_id", "input_file_name")
    medcat_model._supervised_trainer.train.assert_called()


@pytest.mark.skipif(not os.path.exists(os.path.join(MODEL_PARENT_DIR, "icd10_model.zip")),
                    reason="requires the model file to be present in the resources folder")
def test_train_unsupervised(medcat_model):
    medcat_model.init_model()
    medcat_model._config.REDEPLOY_TRAINED_MODEL = "false"
    medcat_model._config.SKIP_SAVE_MODEL = "true"
    medcat_model._unsupervised_trainer = Mock()
    with tempfile.TemporaryFile("r+") as f:
        medcat_model.train_unsupervised(f, 1, 1, "training_id", "input_file_name")
    medcat_model._unsupervised_trainer.train.assert_called()
