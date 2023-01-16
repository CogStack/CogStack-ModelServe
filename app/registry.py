from domain import ModelType
from model_services.trf_model_deid import TransformersModelDeIdentification
from model_services.medcat_model_snomed import MedCATModelSnomed
from model_services.medcat_model_umls import MedCATModelUmls
from model_services.medcat_model_icd10 import MedCATModelIcd10
from model_services.medcat_model_deid import MedCATModelDeIdentification

model_service_registry = {
    ModelType.MEDCAT_SNOMED.value: MedCATModelSnomed,
    ModelType.MEDCAT_UMLS.value: MedCATModelUmls,
    ModelType.MEDCAT_ICD10.value: MedCATModelIcd10,
    ModelType.MEDCAT_DEID.value: MedCATModelDeIdentification,
    ModelType.TRANSFORMERS_DEID.value: TransformersModelDeIdentification
}
