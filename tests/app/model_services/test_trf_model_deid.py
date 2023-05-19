import os
import pytest
from transformers.models.bert.modeling_bert import BertForTokenClassification
from medcat.tokenizers.transformers_ner import TransformersTokenizerNER
from app.config import Settings
from app.model_services.trf_model_deid import TransformersModelDeIdentification


MODEL_PARENT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "resources")


@pytest.fixture(scope="function")
def trf_model():
    config = Settings()
    config.BASE_MODEL_FILE = "trf_deid_model.zip"
    return TransformersModelDeIdentification(config, MODEL_PARENT_DIR)


def test_model_name(trf_model):
    assert trf_model.model_name == "De-identification model"


def test_api_version(trf_model):
    assert trf_model.api_version == "0.0.1"


@pytest.mark.skipif(not os.path.exists(os.path.join(MODEL_PARENT_DIR, "trf_deid_model.zip")),
                    reason="requires the model file to be present in the resources folder")
def test_init_model(trf_model):
    trf_model.init_model()
    assert trf_model.model is not None


@pytest.mark.skipif(not os.path.exists(os.path.join(MODEL_PARENT_DIR, "trf_deid_model.zip")),
                    reason="requires the model file to be present in the resources folder")
def test_load_model(trf_model):
    tokenizer, model = TransformersModelDeIdentification.load_model(os.path.join(os.path.dirname(__file__), "..", "..", "resources", "trf_deid_model.zip"))
    assert type(tokenizer) is TransformersTokenizerNER
    assert type(model) is BertForTokenClassification


@pytest.mark.skipif(not os.path.exists(os.path.join(MODEL_PARENT_DIR, "trf_deid_model.zip")),
                    reason="requires the model file to be present in the resources folder")
def test_info(trf_model):
    trf_model.init_model()
    model_card = trf_model.info()
    assert type(model_card.api_version) is str
    assert type(model_card.model_description) is str
    assert model_card.model_type == "Transformers"


@pytest.mark.skipif(not os.path.exists(os.path.join(MODEL_PARENT_DIR, "trf_deid_model.zip")),
                    reason="requires the model file to be present in the resources folder")
def test_annotate(trf_model):
    trf_model.init_model()
    annotations = trf_model.annotate("NW1 2DA")
    assert len(annotations) == 1
    assert type(annotations[0]["label_name"]) is str
    assert annotations[0]["start"] == 0
    assert annotations[0]["end"] == 7


@pytest.mark.skipif(not os.path.exists(os.path.join(MODEL_PARENT_DIR, "trf_deid_model.zip")),
                    reason="requires the model file to be present in the resources folder")
def test_batch_annotate(trf_model):
    trf_model.init_model()
    annotation_list = trf_model.batch_annotate(["NW1 2DA", "NW1 2DA"])
    assert len(annotation_list) == 2
    assert type(annotation_list[0][0]["label_name"]) is str
    assert type(annotation_list[1][0]["label_name"]) is str
    assert annotation_list[0][0]["start"] == annotation_list[1][0]["start"] == 0
    assert annotation_list[0][0]["end"] == annotation_list[1][0]["end"] == 7
