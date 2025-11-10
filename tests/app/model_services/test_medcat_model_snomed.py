import os
import tempfile
import pytest
from unittest.mock import Mock
from tests.app.conftest import MODEL_PARENT_DIR
from medcat.cat import CAT
from app import __version__
from app.domain import ModelType
from app.model_services.medcat_model_snomed import MedCATModelSnomed


def test_model_name(medcat_snomed_model):
    assert medcat_snomed_model.model_name == "SNOMED MedCAT model"


def test_api_version(medcat_snomed_model):
    assert medcat_snomed_model.api_version == __version__


def test_from_model(medcat_snomed_model):
    new_model_service = medcat_snomed_model.from_model(medcat_snomed_model.model)
    assert isinstance(new_model_service, MedCATModelSnomed)
    assert new_model_service.model == medcat_snomed_model.model


def test_get_records_from_doc(medcat_snomed_model):
    records = medcat_snomed_model.get_records_from_doc({
        "entities": {
            "0": {
                "pretty_name": "pretty_name",
                "cui": "cui",
                "type_ids": ["type"],
                "athena_ids": [{"name": "name_1", "code": "code_1"}, {"name": "name_2", "code": "code_2"}],
                "meta_anns": {}
            }
        }
    })
    assert len(records) == 1
    assert records[0]["label_name"] == "pretty_name"
    assert records[0]["label_id"] == "cui"
    assert records[0]["categories"] == ["type"]
    assert records[0]["athena_ids"] == ["code_1", "code_2"]
    assert records[0]["meta_anns"] == {}


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_PARENT_DIR, "snomed_model.zip")),
    reason="requires the model file to be present in the resources folder",
)
def test_init_model(medcat_snomed_model):
    medcat_snomed_model.init_model()
    target_tuis = medcat_snomed_model._config.TYPE_UNIQUE_ID_WHITELIST.split(",")
    target_cuis = {cui for tui in target_tuis for cui in medcat_snomed_model.model.cdb.addl_info.get("type_id2cuis").get(tui, {})}
    mapped_ontologies = medcat_snomed_model._config.MEDCAT2_MAPPED_ONTOLOGIES.split(",")
    assert medcat_snomed_model.model is not None
    assert medcat_snomed_model.model.config.components.linking.filters.cuis == target_cuis
    assert medcat_snomed_model.model.config.general.map_to_other_ontologies == mapped_ontologies


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_PARENT_DIR, "snomed_model.zip")),
    reason="requires the model file to be present in the resources folder",
)
def test_init_model_with_no_tui_filter(medcat_snomed_model):
    original = MedCATModelSnomed.load_model(os.path.join(MODEL_PARENT_DIR, "snomed_model.zip"))
    medcat_snomed_model._whitelisted_tuis = set([""])
    medcat_snomed_model.init_model()
    assert medcat_snomed_model.model is not None
    assert medcat_snomed_model.model.config.components.linking.filters.cuis == original.config.components.linking.filters.cuis


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_PARENT_DIR, "snomed_model.zip")),
    reason="requires the model file to be present in the resources folder",
)
def test_load_model(medcat_snomed_model):
    cat = MedCATModelSnomed.load_model(os.path.join(MODEL_PARENT_DIR, "snomed_model.zip"))
    assert type(cat) is CAT


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_PARENT_DIR, "snomed_model.zip")),
    reason="requires the model file to be present in the resources folder",
)
def test_info(medcat_snomed_model):
    medcat_snomed_model.init_model()
    model_card = medcat_snomed_model.info()
    assert type(model_card.api_version) is str
    assert type(model_card.model_description) is str
    assert model_card.model_type == ModelType.MEDCAT_SNOMED


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_PARENT_DIR, "snomed_model.zip")),
    reason="requires the model file to be present in the resources folder",
)
def test_annotate(medcat_snomed_model):
    medcat_snomed_model.init_model()
    annotations = medcat_snomed_model.annotate("Spinal stenosis")
    assert len(annotations) == 1
    assert type(annotations[0]["label_name"]) is str
    assert annotations[0].start == 0
    assert annotations[0].end == 15
    assert annotations[0].accuracy > 0


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_PARENT_DIR, "snomed_model.zip")),
    reason="requires the model file to be present in the resources folder",
)
def test_train_supervised(medcat_snomed_model):
    medcat_snomed_model.init_model()
    medcat_snomed_model._config.REDEPLOY_TRAINED_MODEL = "false"
    medcat_snomed_model._config.SKIP_SAVE_MODEL = "true"
    medcat_snomed_model._supervised_trainer = Mock()
    with tempfile.TemporaryFile("r+") as f:
        medcat_snomed_model.train_supervised(f, 1, 1, "training_id", "input_file_name")
    medcat_snomed_model._supervised_trainer.train.assert_called()


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_PARENT_DIR, "snomed_model.zip")),
    reason="requires the model file to be present in the resources folder",
)
def test_train_unsupervised(medcat_snomed_model):
    medcat_snomed_model.init_model()
    medcat_snomed_model._config.REDEPLOY_TRAINED_MODEL = "false"
    medcat_snomed_model._config.SKIP_SAVE_MODEL = "true"
    medcat_snomed_model._unsupervised_trainer = Mock()
    with tempfile.TemporaryFile("r+") as f:
        medcat_snomed_model.train_unsupervised(f, 1, 1, "training_id", "input_file_name")
    medcat_snomed_model._unsupervised_trainer.train.assert_called()
