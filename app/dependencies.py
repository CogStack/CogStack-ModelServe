import logging

from domain import ModelType
from config import Settings
from model_services.base import AbstractModelService
from model_services.deid_model import DeIdModel
from model_services.medcat_model import MedCATModel
from model_services.medcat_model_icd10 import MedCATModelIcd10
from model_services.medcat_model_deid import MedCATModelDeIdentification

logger = logging.getLogger(__name__)


class ModelServiceDep(object):

    @property
    def model_service(self) -> AbstractModelService:
        return self._model_sevice

    @model_service.setter
    def model_service(self, model_service):
        self._model_sevice = model_service

    def __init__(self, model_type: str, config: Settings):
        self._model_type = model_type
        self._config = config
        self._model_sevice = None

    def __call__(self) -> AbstractModelService:
        if self._model_sevice is not None:
            return self._model_sevice
        else:
            if self._model_type == ModelType.MEDCAT_SNOMED.value:
                self._model_sevice = MedCATModel(self._config)
            elif self._model_type == ModelType.MEDCAT_ICD10.value:
                self._model_sevice = MedCATModelIcd10(self._config)
            elif self._model_type == ModelType.MEDCAT_DEID.value:
                self._model_sevice = MedCATModelDeIdentification(self._config)
            elif self._model_type == ModelType.DE_ID.value:
                self._model_sevice = DeIdModel(self._config)
            else:
                logger.error(f"Unknown model type: {self._model_type}")
                exit(1)     # throw an exception?
            return self._model_sevice
