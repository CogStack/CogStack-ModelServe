import os
import tempfile
import pytest
from unittest.mock import Mock
from tests.app.conftest import MODEL_PARENT_DIR
from medcat.cat import CAT
from app import __version__
from app.domain import ModelType
from app.model_services.medcat_model_deid import MedCATModelDeIdentification


def test_model_name(medcat_deid_model):
    assert medcat_deid_model.model_name == "De-Identification MedCAT model"


def test_api_version(medcat_deid_model):
    assert medcat_deid_model.api_version == __version__


def test_from_model(medcat_deid_model):
    new_model_service = medcat_deid_model.from_model(medcat_deid_model.model)
    assert isinstance(new_model_service, MedCATModelDeIdentification)
    assert new_model_service.model == medcat_deid_model.model


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_PARENT_DIR, "deid_model.zip")),
    reason="requires the model file to be present in the resources folder",
)
def test_init_model(medcat_deid_model):
    medcat_deid_model.init_model()
    assert medcat_deid_model.model is not None


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_PARENT_DIR, "deid_model.zip")),
    reason="requires the model file to be present in the resources folder",
)
def test_load_model(medcat_deid_model):
    cat = MedCATModelDeIdentification.load_model(os.path.join(MODEL_PARENT_DIR, "deid_model.zip"))
    assert type(cat) is CAT


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_PARENT_DIR, "deid_model.zip")),
    reason="requires the model file to be present in the resources folder",
)
def test_info(medcat_deid_model):
    medcat_deid_model.init_model()
    model_card = medcat_deid_model.info()
    assert type(model_card.api_version) is str
    assert type(model_card.model_description) is str
    assert model_card.model_type == ModelType.ANONCAT


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_PARENT_DIR, "deid_model.zip")),
    reason="requires the model file to be present in the resources folder",
)
def test_annotate(medcat_deid_model):
    medcat_deid_model.init_model()
    annotations = medcat_deid_model.annotate(
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
    assert type(annotations[0].label_name) is str
    assert type(annotations[1].label_name) is str
    assert annotations[0].start == 598
    assert annotations[0].end == 601
    assert annotations[0].text == "ABC"
    assert annotations[0].accuracy > 0
    assert annotations[0].categories == ["PII"]
    assert annotations[1].start == 2839
    assert annotations[1].end == 2842
    assert annotations[1].accuracy > 0
    assert annotations[1].categories == ["PII"]


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_PARENT_DIR, "deid_model.zip")),
    reason="requires the model file to be present in the resources folder",
)
def test_annotate_with_local_chunking(medcat_deid_model):
    medcat_deid_model.init_model()
    annotations = medcat_deid_model.annotate_with_local_chunking(
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
    assert type(annotations[0].label_name) is str
    assert type(annotations[1].label_name) is str
    assert annotations[0].start == 598
    assert annotations[0].end == 601
    assert annotations[0].text == "ABC"
    assert annotations[0].accuracy > 0
    assert annotations[0].categories == ["PII"]
    assert annotations[1].start == 2839
    assert annotations[1].end == 2842
    assert annotations[1].text == "ABC"
    assert annotations[1].accuracy > 0
    assert annotations[1].categories == ["PII"]


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_PARENT_DIR, "deid_model.zip")),
    reason="requires the model file to be present in the resources folder",
)
def test_batch_annotate(medcat_deid_model):
    medcat_deid_model.init_model()
    annotations_list = medcat_deid_model.batch_annotate([
        "This is a post code NW1 2DA",
        "This is a post code NW1 2DB",
        "This is a post code NW1 2DC",
        "This is a post code NW1 2DD",
        "This is a post code NW1 2DE",
        "This is a post code NW1 2DF",
        "This is a post code NW1 2DG",
        "This is a post code NW1 2DH",
        "This is a post code NW1 2DI",
        "This is a post code NW1 2DJ",
    ])
    assert len(annotations_list) == 10
    for annotations in annotations_list:
        for annotation in annotations:
            assert type(annotation.label_name) is str
            assert type(annotation.label_id) is str
            assert annotation.categories == ["PII"]
            assert annotation.end > annotation.start
            assert annotation.accuracy > 0


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_PARENT_DIR, "deid_model.zip")),
    reason="requires the model file to be present in the resources folder",
)
def test_train_supervised(medcat_deid_model):
    medcat_deid_model.init_model()
    medcat_deid_model._config.REDEPLOY_TRAINED_MODEL = "false"
    medcat_deid_model._config.SKIP_SAVE_MODEL = "true"
    medcat_deid_model._supervised_trainer = Mock()
    with tempfile.TemporaryFile("r+") as f:
        medcat_deid_model.train_supervised(f, 1, 1, "training_id", "input_file_name")
    medcat_deid_model._supervised_trainer.train.assert_called()
