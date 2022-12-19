from app.domain import ModelType
from app.registry import model_service_registry
from model_services.trf_model_deid import TransformersModelDeIdentification
from model_services.medcat_model import MedCATModel
from model_services.medcat_model_icd10 import MedCATModelIcd10
from model_services.medcat_model_deid import MedCATModelDeIdentification


def test_model_registry():
    assert model_service_registry[ModelType.MEDCAT_SNOMED.value] == MedCATModel
    assert model_service_registry[ModelType.MEDCAT_UMLS.value] == MedCATModel
    assert model_service_registry[ModelType.MEDCAT_ICD10.value] == MedCATModelIcd10
    assert model_service_registry[ModelType.MEDCAT_DEID.value] == MedCATModelDeIdentification
    assert model_service_registry[ModelType.TRANSFORMERS_DEID.value] == TransformersModelDeIdentification
