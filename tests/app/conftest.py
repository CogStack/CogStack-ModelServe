import os
import pytest
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForCausalLM
from unittest.mock import Mock, MagicMock
from app.config import Settings
from app.model_services.medcat_model_snomed import MedCATModelSnomed
from app.model_services.medcat_model_icd10 import MedCATModelIcd10
from app.model_services.medcat_model_umls import MedCATModelUmls
from app.model_services.medcat_model_deid import MedCATModelDeIdentification
from app.model_services.trf_model_deid import TransformersModelDeIdentification
from app.model_services.huggingface_ner_model import HuggingFaceNerModel
from app.model_services.huggingface_llm_model import HuggingFaceLlmModel

MODEL_PARENT_DIR = os.path.join(os.path.dirname(__file__), "..", "resources", "model")


@pytest.fixture
def mlflow_fixture(mocker):
    active_run = Mock()
    pyfunc_model = Mock()
    active_run.info = MagicMock()
    active_run.info.run_id = "run_id"
    active_run.data = MagicMock()
    active_run.data.metrics = {"precision": 0.9973285610540512, "recall": 0.9896606632947247, "f1": 0.9934285636532457}
    active_run.data.tags = {"tag": "tag"}
    mocker.patch("mlflow.set_tracking_uri")
    mocker.patch("mlflow.get_tracking_uri", return_value="http://localhost:5000")
    mocker.patch("mlflow.get_experiment_by_name", return_value=None)
    mocker.patch("mlflow.create_experiment", return_value="experiment_id")
    mocker.patch("mlflow.start_run", return_value=active_run)
    mocker.patch("mlflow.end_run")
    mocker.patch("mlflow.set_tags")
    mocker.patch("mlflow.set_tag")
    mocker.patch("mlflow.log_param")
    mocker.patch("mlflow.log_params")
    mocker.patch("mlflow.log_metrics")
    mocker.patch("mlflow.log_artifact")
    mocker.patch("mlflow.log_table")
    mocker.patch("mlflow.log_input")
    mocker.patch("mlflow.pyfunc.load_model", return_value=pyfunc_model)
    mocker.patch("mlflow.pyfunc.log_model")
    mocker.patch("mlflow.artifacts.download_artifacts")
    mocker.patch("mlflow.register_model")
    mocker.patch("mlflow.pyfunc.save_model")
    mocker.patch("mlflow.search_runs", return_value=[active_run])


@pytest.fixture(scope="function")
def medcat_snomed_model():
    config = Settings()
    config.BASE_MODEL_FILE = "snomed_model.zip"
    config.TYPE_UNIQUE_ID_WHITELIST = "T-9,T-11,T-18,T-39,T-40,T-45"
    return MedCATModelSnomed(config, MODEL_PARENT_DIR, True)


@pytest.fixture(scope="function")
def medcat_icd10_model():
    config = Settings()
    config.BASE_MODEL_FILE = "icd10_model.zip"
    config.TYPE_UNIQUE_ID_WHITELIST = "T-9,T-11,T-18,T-39,T-40,T-45"
    return MedCATModelIcd10(config, MODEL_PARENT_DIR, True)


@pytest.fixture(scope="function")
def medcat_umls_model():
    config = Settings()
    config.BASE_MODEL_FILE = "umls_model.zip"
    return MedCATModelUmls(config, MODEL_PARENT_DIR, True)


@pytest.fixture(scope="function")
def medcat_deid_model():
    config = Settings()
    config.BASE_MODEL_FILE = "deid_model.zip"
    return MedCATModelDeIdentification(config, MODEL_PARENT_DIR, True)


@pytest.fixture(scope="function")
def trf_model():
    config = Settings()
    config.BASE_MODEL_FILE = "trf_deid_model.zip"
    return TransformersModelDeIdentification(config, MODEL_PARENT_DIR)


@pytest.fixture(scope="function")
def huggingface_ner_model():
    config = Settings()
    config.BASE_MODEL_FILE = "huggingface_ner_model.zip"
    model_service = HuggingFaceNerModel(config, MODEL_PARENT_DIR)
    test_hf_repo = "hf-internal-testing/tiny-bert-for-token-classification"
    model_service.model = AutoModelForTokenClassification.from_pretrained(test_hf_repo)
    model_service.tokenizer = AutoTokenizer.from_pretrained(test_hf_repo)
    return model_service


@pytest.fixture(scope="function")
def huggingface_llm_model():
    config = Settings()
    config.BASE_MODEL_FILE = "huggingface_llm_model.zip"
    model_service = HuggingFaceLlmModel(config, MODEL_PARENT_DIR)
    test_hf_repo = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    model_service.model = AutoModelForCausalLM.from_pretrained(test_hf_repo)
    model_service.tokenizer = AutoTokenizer.from_pretrained(test_hf_repo)
    return model_service
