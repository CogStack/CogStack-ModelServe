from app.domain import ModelType
from app.model_services.trf_model_deid import TransformersModelDeIdentification
from app.model_services.medcat_model_snomed import MedCATModelSnomed
from app.model_services.medcat_model_umls import MedCATModelUmls
from app.model_services.medcat_model_icd10 import MedCATModelIcd10
from app.model_services.medcat_model_opcs4 import MedCATModelOpcs4
from app.model_services.medcat_model_deid import MedCATModelDeIdentification
from app.model_services.huggingface_ner_model import HuggingFaceNerModel
from app.model_services.huggingface_llm_model import HuggingFaceLlmModel

model_service_registry = {
    ModelType.MEDCAT_SNOMED: MedCATModelSnomed,
    ModelType.MEDCAT_UMLS: MedCATModelUmls,
    ModelType.MEDCAT_ICD10: MedCATModelIcd10,
    ModelType.MEDCAT_OPCS4: MedCATModelOpcs4,
    ModelType.MEDCAT_DEID: MedCATModelDeIdentification,
    ModelType.ANONCAT: MedCATModelDeIdentification,
    ModelType.TRANSFORMERS_DEID: TransformersModelDeIdentification,
    ModelType.HUGGINGFACE_NER: HuggingFaceNerModel,
    ModelType.HUGGINGFACE_LLM: HuggingFaceLlmModel,
}
