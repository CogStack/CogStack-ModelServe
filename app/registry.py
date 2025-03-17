from app.domain import ModelType
from app.model_services.trf_model_deid import TransformersModelDeIdentification
from app.model_services.medcat_model_snomed import MedCATModelSnomed
from app.model_services.medcat_model_umls import MedCATModelUmls
from app.model_services.medcat_model_icd10 import MedCATModelIcd10
from app.model_services.medcat_model_deid import MedCATModelDeIdentification
from app.model_services.huggingface_ner_model import HuggingFaceNerModel

model_service_registry = {
    ModelType.MEDCAT_SNOMED: MedCATModelSnomed,
    ModelType.MEDCAT_UMLS: MedCATModelUmls,
    ModelType.MEDCAT_ICD10: MedCATModelIcd10,
    ModelType.MEDCAT_DEID: MedCATModelDeIdentification,
    ModelType.ANONCAT: MedCATModelDeIdentification,
    ModelType.TRANSFORMERS_DEID: TransformersModelDeIdentification,
    ModelType.HUGGINGFACE_NER: HuggingFaceNerModel,
}
