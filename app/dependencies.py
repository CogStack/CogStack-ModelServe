import logging

from config import Settings
from registry import model_service_registry
from model_services.base import AbstractModelService

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
            if self._model_type in model_service_registry.keys():
                self._model_sevice = model_service_registry[self._model_type](self._config)
            else:
                logger.error(f"Unknown model type: {self._model_type}")
                exit(1)     # throw an exception?
            return self._model_sevice
