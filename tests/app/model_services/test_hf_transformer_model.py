import os
import tempfile
import pytest
from unittest.mock import Mock
from tests.app.conftest import MODEL_PARENT_DIR
from transformers import PreTrainedModel, PreTrainedTokenizer
from domain import ModelType
from model_services.hf_transformer_model import HuggingfaceTransformerModel


def test_model_name(hf_transformer_model):
    assert hf_transformer_model.model_name == "Huggingface Transformer model"


def test_api_version(hf_transformer_model):
    assert hf_transformer_model.api_version == "0.0.1"


def test_from_model(hf_transformer_model):
    new_model_service = hf_transformer_model.from_model(hf_transformer_model.model, hf_transformer_model.tokenizer)
    assert isinstance(new_model_service, HuggingfaceTransformerModel)
    assert new_model_service.model == hf_transformer_model.model
    assert new_model_service.tokenizer == hf_transformer_model.tokenizer


@pytest.mark.skipif(not os.path.exists(os.path.join(MODEL_PARENT_DIR, "hf_transformer_model.zip")),
                    reason="requires the model file to be present in the resources folder")
def test_init_model(hf_transformer_model):
    hf_transformer_model.init_model()
    assert hf_transformer_model.model is not None
    assert hf_transformer_model.tokenizer is not None


@pytest.mark.skipif(not os.path.exists(os.path.join(MODEL_PARENT_DIR, "hf_transformer_model.zip")),
                    reason="requires the model file to be present in the resources folder")
def test_load_model(hf_transformer_model):
    model, tokenizer = HuggingfaceTransformerModel.load_model(os.path.join(MODEL_PARENT_DIR, "hf_transformer_model.zip"))
    assert isinstance(model, PreTrainedModel)
    assert isinstance(tokenizer, PreTrainedTokenizer)


@pytest.mark.skipif(not os.path.exists(os.path.join(MODEL_PARENT_DIR, "hf_transformer_model.zip")),
                    reason="requires the model file to be present in the resources folder")
def test_info(hf_transformer_model):
    hf_transformer_model.init_model()
    model_card = hf_transformer_model.info()
    assert type(model_card.api_version) is str
    assert type(model_card.model_description) is str
    assert model_card.model_type == ModelType.HF_TRANSFORMER


@pytest.mark.skipif(not os.path.exists(os.path.join(MODEL_PARENT_DIR, "hf_transformer_model.zip")),
                    reason="requires the model file to be present in the resources folder")
def test_annotate(hf_transformer_model):
    hf_transformer_model.init_model()
    annotations = hf_transformer_model.annotate(
        """The patient is a 60-year-old female, who complained of coughing during meals. """
        """ Her outpatient evaluation revealed a mild-to-moderate cognitive linguistic deficit, which was completed approximately"""
        """ 2 months ago.  The patient had a history of hypertension and TIA/stroke.  The patient denied history of heartburn"""
        """ and/or gastroesophageal reflux disorder.  A modified barium swallow study was ordered to objectively evaluate the"""
        """ patient's swallowing function and safety and to rule out aspiration.,OBJECTIVE: , Modified barium swallow study"""
        """ was performed in the Radiology Suite in cooperation with Dr. ABC.  The patient was seated upright in a video imaging"""
        """ chair throughout this assessment.  To evaluate the patient's swallowing function and safety, she was administered"""
        """ graduated amounts of liquid and food mixed with barium in the form of thin liquid (teaspoon x2, cup sip x2); nectar-thick"""
        """ liquid (teaspoon x2, cup sip x2); puree consistency (teaspoon x2); and solid food consistency (1/4 cracker x1).,ASSESSMENT,"""
        """ ORAL STAGE:,  Premature spillage to the level of the valleculae and pyriform sinuses with thin liquid.  Decreased"""
        """ tongue base retraction, which contributed to vallecular pooling after the swallow.,PHARYNGEAL STAGE: , No aspiration"""
        """ was observed during this evaluation.  Penetration was noted with cup sips of thin liquid only.  Trace residual on"""
        """ the valleculae and on tongue base with nectar-thick puree and solid consistencies.  The patient's hyolaryngeal"""
        """ elevation and anterior movement are within functional limits.  Epiglottic inversion is within functional limits.,"""
        """ CERVICAL ESOPHAGEAL STAGE:  ,The patient's upper esophageal sphincter opening is well coordinated with swallow and"""
        """ readily accepted the bolus.  Radiologist noted reduced peristaltic action of the constricted muscles in the esophagus,"""
        """ which may be contributing to the patient's complaint of globus sensation.,DIAGNOSTIC IMPRESSION:,  No aspiration was"""
        """ noted during this evaluation.  Penetration with cup sips of thin liquid.  The patient did cough during this evaluation,"""
        """ but that was noted related to aspiration or penetration.,PROGNOSTIC IMPRESSION: ,Based on this evaluation, the prognosis"""
        """ for swallowing and safety is good.,PLAN: , Based on this evaluation and following recommendations are being made:,1.  """
        """ The patient to take small bite and small sips to help decrease the risk of aspiration and penetration.,2.  The patient"""
        """ should remain upright at a 90-degree angle for at least 45 minutes after meals to decrease the risk of aspiration and"""
        """ penetration as well as to reduce her globus sensation.,3.  The patient should be referred to a gastroenterologist for"""
        """ further evaluation of her esophageal function.,The patient does not need any skilled speech therapy for her swallowing"""
        """ abilities at this time, and she is discharged from my services.). Dr. ABC""")
    assert len(annotations) == 2
    assert type(annotations[0]["label_name"]) is str
    assert type(annotations[1]["label_name"]) is str


@pytest.mark.skipif(not os.path.exists(os.path.join(MODEL_PARENT_DIR, "hf_transformer_model.zip")),
                    reason="requires the model file to be present in the resources folder")
def test_train_unsupervised(hf_transformer_model):
    hf_transformer_model.init_model()
    hf_transformer_model._config.REDEPLOY_TRAINED_MODEL = "false"
    hf_transformer_model._config.SKIP_SAVE_MODEL = "true"
    hf_transformer_model._unsupervised_trainer = Mock()
    with tempfile.TemporaryFile("r+") as f:
        hf_transformer_model.train_unsupervised(f, 1, 1, "training_id", "input_file_name")
    hf_transformer_model._unsupervised_trainer.train.assert_called()
