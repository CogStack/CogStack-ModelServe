from api.dependencies import ModelServiceDep
from config import Settings
from model_services.medcat_model import MedCATModel
from model_services.medcat_model_icd10 import MedCATModelIcd10
from model_services.medcat_model_umls import MedCATModelUmls
from model_services.medcat_model_deid import MedCATModelDeIdentification
from model_services.trf_model_deid import TransformersModelDeIdentification
from model_services.huggingface_ner_model import HuggingFaceNerModel


def test_medcat_snomed_dep():
    model_service_dep = ModelServiceDep("medcat_snomed", Settings())
    assert isinstance(model_service_dep(), MedCATModel)


def test_medcat_icd10_dep():
    model_service_dep = ModelServiceDep("medcat_icd10", Settings())
    assert isinstance(model_service_dep(), MedCATModelIcd10)


def test_medcat_umls_dep():
    model_service_dep = ModelServiceDep("medcat_umls", Settings())
    assert isinstance(model_service_dep(), MedCATModelUmls)


def test_medcat_deid_dep():
    model_service_dep = ModelServiceDep("medcat_deid", Settings())
    assert isinstance(model_service_dep(), MedCATModelDeIdentification)


def test_transformer_deid_dep():
    model_service_dep = ModelServiceDep("transformers_deid", Settings())
    assert isinstance(model_service_dep(), TransformersModelDeIdentification)


def test_huggingface_ner_dep():
    model_service_dep = ModelServiceDep("huggingface_ner", Settings())
    assert isinstance(model_service_dep(), HuggingFaceNerModel)
