import os
import tempfile
import pytest
from unittest.mock import Mock
from tests.app.conftest import MODEL_PARENT_DIR
from medcat.cat import CAT
from app import __version__
from app.domain import ModelType
from app.model_services.medcat_model_opcs4 import MedCATModelOpcs4


def test_model_name(medcat_opcs4_model):
    assert medcat_opcs4_model.model_name == "OPCS-4 MedCAT model"


def test_api_version(medcat_opcs4_model):
    assert medcat_opcs4_model.api_version == __version__


def test_from_model(medcat_opcs4_model):
    new_model_service = medcat_opcs4_model.from_model(medcat_opcs4_model.model)
    assert isinstance(new_model_service, MedCATModelOpcs4)
    assert new_model_service.model == medcat_opcs4_model.model


def test_get_records_from_doc(medcat_opcs4_model):
    records = medcat_opcs4_model.get_records_from_doc({
        "entities":
            {
                "0": {
                    "pretty_name": "pretty_name",
                    "cui": "cui",
                    "types": ["type"],
                    "opcs4": [{"code": "code", "name": "name"}],
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
    not os.path.exists(os.path.join(MODEL_PARENT_DIR, "opcs4_model.zip")),
    reason="requires the model file to be present in the resources folder",
)
def test_init_model_with_no_tui_filter(medcat_opcs4_model):
    original = MedCATModelOpcs4.load_model(os.path.join(MODEL_PARENT_DIR, "opcs4_model.zip"))
    medcat_opcs4_model._whitelisted_tuis = set([""])
    medcat_opcs4_model.init_model()
    assert medcat_opcs4_model.model is not None
    assert medcat_opcs4_model.model.cdb.config.linking.filters.get("cuis") == original.cdb.config.linking.filters.get("cuis")


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_PARENT_DIR, "opcs4_model.zip")),
    reason="requires the model file to be present in the resources folder",
)
def test_init_model(medcat_opcs4_model):
    medcat_opcs4_model.init_model()
    target_tuis = medcat_opcs4_model._config.TYPE_UNIQUE_ID_WHITELIST.split(",")
    target_cuis = {cui for tui in target_tuis for cui in medcat_opcs4_model.model.cdb.addl_info.get("type_id2cuis").get(tui, {})}
    assert medcat_opcs4_model.model is not None
    assert medcat_opcs4_model.model.cdb.config.linking.filters.get("cuis") == target_cuis


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_PARENT_DIR, "opcs4_model.zip")),
    reason="requires the model file to be present in the resources folder",
)
def test_load_model(medcat_opcs4_model):
    cat = MedCATModelOpcs4.load_model(os.path.join(MODEL_PARENT_DIR, "opcs4_model.zip"))
    assert type(cat) is CAT


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_PARENT_DIR, "opcs4_model.zip")),
    reason="requires the model file to be present in the resources folder",
)
def test_info(medcat_opcs4_model):
    medcat_opcs4_model.init_model()
    model_card = medcat_opcs4_model.info()
    assert type(model_card.api_version) is str
    assert type(model_card.model_description) is str
    assert model_card.model_type == ModelType.MEDCAT_OPCS4


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_PARENT_DIR, "opcs4_model.zip")),
    reason="requires the model file to be present in the resources folder",
)
def test_annotate(medcat_opcs4_model):
    medcat_opcs4_model.init_model()
    annotations = medcat_opcs4_model.annotate("Spinal tap")
    assert len(annotations) == 1
    assert type(annotations[0]["label_name"]) is str
    assert annotations[0].start == 0
    assert annotations[0].end == 10
    assert annotations[0].accuracy > 0


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_PARENT_DIR, "opcs4_model.zip")),
    reason="requires the model file to be present in the resources folder",
)
def test_train_supervised(medcat_opcs4_model):
    medcat_opcs4_model.init_model()
    medcat_opcs4_model._config.REDEPLOY_TRAINED_MODEL = "false"
    medcat_opcs4_model._config.SKIP_SAVE_MODEL = "true"
    medcat_opcs4_model._supervised_trainer = Mock()
    with tempfile.TemporaryFile("r+") as f:
        medcat_opcs4_model.train_supervised(f, 1, 1, "training_id", "input_file_name")
    medcat_opcs4_model._supervised_trainer.train.assert_called()


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_PARENT_DIR, "opcs4_model.zip")),
    reason="requires the model file to be present in the resources folder",
)
def test_train_unsupervised(medcat_opcs4_model):
    medcat_opcs4_model.init_model()
    medcat_opcs4_model._config.REDEPLOY_TRAINED_MODEL = "false"
    medcat_opcs4_model._config.SKIP_SAVE_MODEL = "true"
    medcat_opcs4_model._unsupervised_trainer = Mock()
    with tempfile.TemporaryFile("r+") as f:
        medcat_opcs4_model.train_unsupervised(f, 1, 1, "training_id", "input_file_name")
    medcat_opcs4_model._unsupervised_trainer.train.assert_called()
