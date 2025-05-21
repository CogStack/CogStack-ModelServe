import os
import tempfile
import pytest
from unittest.mock import Mock
from tests.app.conftest import MODEL_PARENT_DIR
from medcat.cat import CAT
from app import __version__
from app.domain import ModelType
from app.model_services.medcat_model_umls import MedCATModelUmls


def test_model_name(medcat_umls_model):
    assert medcat_umls_model.model_name == "UMLS MedCAT model"


def test_api_version(medcat_umls_model):
    assert medcat_umls_model.api_version == __version__


def test_from_model(medcat_umls_model):
    new_model_service = medcat_umls_model.from_model(medcat_umls_model.model)
    assert isinstance(new_model_service, MedCATModelUmls)
    assert new_model_service.model == medcat_umls_model.model


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_PARENT_DIR, "umls_model.zip")),
    reason="requires the model file to be present in the resources folder",
)
def test_init_model(medcat_umls_model):
    medcat_umls_model.init_model()
    assert medcat_umls_model.model is not None


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_PARENT_DIR, "umls_model.zip")),
    reason="requires the model file to be present in the resources folder",
)
def test_load_model(medcat_umls_model):
    cat = MedCATModelUmls.load_model(os.path.join(MODEL_PARENT_DIR, "umls_model.zip"))
    assert type(cat) is CAT


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_PARENT_DIR, "umls_model.zip")),
    reason="requires the model file to be present in the resources folder",
)
def test_info(medcat_umls_model):
    medcat_umls_model.init_model()
    model_card = medcat_umls_model.info()
    assert type(model_card.api_version) is str
    assert type(model_card.model_description) is str
    assert model_card.model_type == ModelType.MEDCAT_UMLS


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_PARENT_DIR, "umls_model.zip")),
    reason="requires the model file to be present in the resources folder",
)
def test_annotate(medcat_umls_model):
    medcat_umls_model.init_model()
    annotations = medcat_umls_model.annotate("Spinal stenosis")
    assert len(annotations) == 1
    assert type(annotations[0]["label_name"]) is str
    assert annotations[0].start == 0
    assert annotations[0].end == 15
    assert annotations[0].accuracy > 0


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_PARENT_DIR, "umls_model.zip")),
    reason="requires the model file to be present in the resources folder",
)
def test_train_supervised(medcat_umls_model):
    medcat_umls_model.init_model()
    medcat_umls_model._config.REDEPLOY_TRAINED_MODEL = "false"
    medcat_umls_model._config.SKIP_SAVE_MODEL = "true"
    medcat_umls_model._supervised_trainer = Mock()
    with tempfile.TemporaryFile("r+") as f:
        medcat_umls_model.train_supervised(f, 1, 1, "training_id", "input_file_name")
    medcat_umls_model._supervised_trainer.train.assert_called()


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_PARENT_DIR, "umls_model.zip")),
    reason="requires the model file to be present in the resources folder",
)
def test_train_unsupervised(medcat_umls_model):
    medcat_umls_model.init_model()
    medcat_umls_model._config.REDEPLOY_TRAINED_MODEL = "false"
    medcat_umls_model._config.SKIP_SAVE_MODEL = "true"
    medcat_umls_model._unsupervised_trainer = Mock()
    with tempfile.TemporaryFile("r+") as f:
        medcat_umls_model.train_unsupervised(f, 1, 1, "training_id", "input_file_name")
    medcat_umls_model._unsupervised_trainer.train.assert_called()
