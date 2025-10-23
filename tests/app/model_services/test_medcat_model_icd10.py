import os
import tempfile
import pytest
from unittest.mock import Mock
from tests.app.conftest import MODEL_PARENT_DIR
from medcat.cat import CAT
from app import __version__
from app.domain import ModelType
from app.model_services.medcat_model_icd10 import MedCATModelIcd10


def test_model_name(medcat_icd10_model):
    assert medcat_icd10_model.model_name == "ICD-10 MedCAT model"


def test_api_version(medcat_icd10_model):
    assert medcat_icd10_model.api_version == __version__


def test_from_model(medcat_icd10_model):
    new_model_service = medcat_icd10_model.from_model(medcat_icd10_model.model)
    assert isinstance(new_model_service, MedCATModelIcd10)
    assert new_model_service.model == medcat_icd10_model.model


def test_get_records_from_doc(medcat_icd10_model):
    records = medcat_icd10_model.get_records_from_doc({
        "entities":
            {
                "0": {
                    "pretty_name": "pretty_name",
                    "cui": "cui",
                    "type_ids": ["type"],
                    "icd10": [{"code": "code", "name": "name"}],
                    "athena_ids": [{"name": "name_1", "code": "code_1"}, {"name": "name_2", "code": "code_2"}],
                    "acc": 1.0,
                    "meta_anns": {}
                }
            }
    })
    assert len(records) == 1
    assert records[0]["label_name"] == "name"
    assert records[0]["cui"] == "cui"
    assert records[0]["label_id"] == "code"
    assert records[0]["categories"] == ["type"]
    assert records[0]["athena_ids"] == ["code_1", "code_2"]
    assert records[0]["accuracy"] == 1.0
    assert records[0]["meta_anns"] == {}


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_PARENT_DIR, "icd10_model.zip")),
    reason="requires the model file to be present in the resources folder",
)
def test_init_model_with_no_tui_filter(medcat_icd10_model):
    original = MedCATModelIcd10.load_model(os.path.join(MODEL_PARENT_DIR, "icd10_model.zip"))
    medcat_icd10_model._whitelisted_tuis = set([""])
    medcat_icd10_model.init_model()
    assert medcat_icd10_model.model is not None
    assert medcat_icd10_model.model.config.components.linking.filters.cuis == original.config.components.linking.filters.cuis


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_PARENT_DIR, "icd10_model.zip")),
    reason="requires the model file to be present in the resources folder",
)
def test_init_model(medcat_icd10_model):
    medcat_icd10_model.init_model()
    target_tuis = medcat_icd10_model._config.TYPE_UNIQUE_ID_WHITELIST.split(",")
    target_cuis = {cui for tui in target_tuis for cui in medcat_icd10_model.model.cdb.addl_info.get("type_id2cuis").get(tui, {})}
    assert medcat_icd10_model.model is not None
    assert medcat_icd10_model.model.config.components.linking.filters.cuis == target_cuis


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_PARENT_DIR, "icd10_model.zip")),
    reason="requires the model file to be present in the resources folder",
)
def test_load_model(medcat_icd10_model):
    cat = MedCATModelIcd10.load_model(os.path.join(MODEL_PARENT_DIR, "icd10_model.zip"))
    assert type(cat) is CAT


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_PARENT_DIR, "icd10_model.zip")),
    reason="requires the model file to be present in the resources folder",
)
def test_info(medcat_icd10_model):
    medcat_icd10_model.init_model()
    model_card = medcat_icd10_model.info()
    assert type(model_card.api_version) is str
    assert type(model_card.model_description) is str
    assert model_card.model_type == ModelType.MEDCAT_ICD10


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_PARENT_DIR, "icd10_model.zip")),
    reason="requires the model file to be present in the resources folder",
)
def test_annotate(medcat_icd10_model):
    medcat_icd10_model.init_model()
    annotations = medcat_icd10_model.annotate("Spinal stenosis")
    assert len(annotations) == 1
    assert type(annotations[0]["label_name"]) is str
    assert annotations[0].start == 0
    assert annotations[0].end == 15
    assert annotations[0].accuracy > 0


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_PARENT_DIR, "icd10_model.zip")),
    reason="requires the model file to be present in the resources folder",
)
def test_train_supervised(medcat_icd10_model):
    medcat_icd10_model.init_model()
    medcat_icd10_model._config.REDEPLOY_TRAINED_MODEL = "false"
    medcat_icd10_model._config.SKIP_SAVE_MODEL = "true"
    medcat_icd10_model._supervised_trainer = Mock()
    with tempfile.TemporaryFile("r+") as f:
        medcat_icd10_model.train_supervised(f, 1, 1, "training_id", "input_file_name")
    medcat_icd10_model._supervised_trainer.train.assert_called()


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_PARENT_DIR, "icd10_model.zip")),
    reason="requires the model file to be present in the resources folder",
)
def test_train_unsupervised(medcat_icd10_model):
    medcat_icd10_model.init_model()
    medcat_icd10_model._config.REDEPLOY_TRAINED_MODEL = "false"
    medcat_icd10_model._config.SKIP_SAVE_MODEL = "true"
    medcat_icd10_model._unsupervised_trainer = Mock()
    with tempfile.TemporaryFile("r+") as f:
        medcat_icd10_model.train_unsupervised(f, 1, 1, "training_id", "input_file_name")
    medcat_icd10_model._unsupervised_trainer.train.assert_called()
