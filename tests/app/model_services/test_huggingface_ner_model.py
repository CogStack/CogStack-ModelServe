import os
import tempfile
from unittest.mock import Mock
import pandas as pd
import pytest
from tests.app.conftest import MODEL_PARENT_DIR
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from app import __version__
from app.domain import ModelType
from app.model_services.huggingface_ner_model import HuggingFaceNerModel


def test_model_name(huggingface_ner_model):
    assert huggingface_ner_model.model_name == "HuggingFace NER model"


def test_api_version(huggingface_ner_model):
    assert huggingface_ner_model.api_version == __version__


def test_from_model(huggingface_ner_model):
    new_model_service = huggingface_ner_model.from_model(huggingface_ner_model.model, huggingface_ner_model.tokenizer)
    assert isinstance(new_model_service, HuggingFaceNerModel)
    assert new_model_service.model == huggingface_ner_model.model
    assert new_model_service.tokenizer == huggingface_ner_model.tokenizer


def test_init_model(huggingface_ner_model):
    huggingface_ner_model.init_model()
    assert huggingface_ner_model.model is not None
    assert huggingface_ner_model.tokenizer is not None


def test_load_model(huggingface_ner_model):
    model, tokenizer = HuggingFaceNerModel.load_model(os.path.join(MODEL_PARENT_DIR, "huggingface_ner_model.tar.gz"))
    assert isinstance(model, PreTrainedModel)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)


def test_info(huggingface_ner_model):
    huggingface_ner_model.init_model()
    model_card = huggingface_ner_model.info()
    assert type(model_card.api_version) is str
    assert type(model_card.model_description) is str
    assert model_card.model_type == ModelType.HUGGINGFACE_NER


def test_annotate(huggingface_ner_model):
    huggingface_ner_model._confidence_threshold = 0.01
    annotations = huggingface_ner_model.annotate(
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
    assert isinstance(annotations, list)
    assert len(annotations) > 0
    assert annotations[0].start == 0
    assert annotations[0].end > annotations[0].start
    assert annotations[0].accuracy > 0
    assert len(annotations[0].text) > 0


def test_train_unsupervised(huggingface_ner_model):
    huggingface_ner_model.init_model()
    huggingface_ner_model._config.REDEPLOY_TRAINED_MODEL = "false"
    huggingface_ner_model._config.SKIP_SAVE_MODEL = "true"
    huggingface_ner_model._unsupervised_trainer = Mock()
    with tempfile.TemporaryFile("r+") as f:
        huggingface_ner_model.train_unsupervised(f, 1, 1, "training_id", "input_file_name")
    huggingface_ner_model._unsupervised_trainer.train.assert_called()


def test_train_supervised(huggingface_ner_model):
    huggingface_ner_model.init_model()
    huggingface_ner_model._config.REDEPLOY_TRAINED_MODEL = "false"
    huggingface_ner_model._config.SKIP_SAVE_MODEL = "true"
    huggingface_ner_model._supervised_trainer = Mock()
    with tempfile.TemporaryFile("r+") as f:
        huggingface_ner_model.train_supervised(f, 1, 1, "training_id", "input_file_name")
    huggingface_ner_model._supervised_trainer.train.assert_called()
