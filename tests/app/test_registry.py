from domain import ModelType
from registry import model_service_registry
from model_services.trf_model_deid import TransformersModelDeIdentification
from model_services.medcat_model_snomed import MedCATModelSnomed
from model_services.medcat_model_umls import MedCATModelUmls
from model_services.medcat_model_icd10 import MedCATModelIcd10
from model_services.medcat_model_deid import MedCATModelDeIdentification


def test_model_registry():
    assert model_service_registry[ModelType.MEDCAT_SNOMED.value] == MedCATModelSnomed
    assert model_service_registry[ModelType.MEDCAT_UMLS.value] == MedCATModelUmls
    assert model_service_registry[ModelType.MEDCAT_ICD10.value] == MedCATModelIcd10
    assert model_service_registry[ModelType.MEDCAT_DEID.value] == MedCATModelDeIdentification
    assert model_service_registry[ModelType.ANONCAT.value] == MedCATModelDeIdentification
    assert model_service_registry[ModelType.TRANSFORMERS_DEID.value] == TransformersModelDeIdentification
