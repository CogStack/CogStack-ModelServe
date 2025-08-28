from app.domain import ModelType
from app.registry import model_service_registry
from app.model_services.trf_model_deid import TransformersModelDeIdentification
from app.model_services.medcat_model_snomed import MedCATModelSnomed
from app.model_services.medcat_model_umls import MedCATModelUmls
from app.model_services.medcat_model_icd10 import MedCATModelIcd10
from app.model_services.medcat_model_opcs4 import MedCATModelOpcs4
from app.model_services.medcat_model_deid import MedCATModelDeIdentification
from app.model_services.huggingface_ner_model import HuggingFaceNerModel
from app.model_services.huggingface_llm_model import HuggingFaceLlmModel


def test_model_registry():
    assert model_service_registry[ModelType.MEDCAT_SNOMED.value] == MedCATModelSnomed
    assert model_service_registry[ModelType.MEDCAT_UMLS.value] == MedCATModelUmls
    assert model_service_registry[ModelType.MEDCAT_ICD10.value] == MedCATModelIcd10
    assert model_service_registry[ModelType.MEDCAT_OPCS4.value] == MedCATModelOpcs4
    assert model_service_registry[ModelType.MEDCAT_DEID.value] == MedCATModelDeIdentification
    assert model_service_registry[ModelType.ANONCAT.value] == MedCATModelDeIdentification
    assert model_service_registry[ModelType.TRANSFORMERS_DEID.value] == TransformersModelDeIdentification
    assert model_service_registry[ModelType.HUGGINGFACE_NER.value] == HuggingFaceNerModel
    assert model_service_registry[ModelType.HUGGINGFACE_LLM.value] == HuggingFaceLlmModel
