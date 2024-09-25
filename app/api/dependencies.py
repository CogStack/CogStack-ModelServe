import logging

from typing import Optional
from config import Settings
from registry import model_service_registry
from model_services.base import AbstractModelService
from management.model_manager import ModelManager

logger = logging.getLogger("cms")


class ModelServiceDep(object):

    @property
    def model_service(self) -> AbstractModelService:
        return self._model_sevice

    @model_service.setter
    def model_service(self, model_service: AbstractModelService) -> None:
        self._model_sevice = model_service

    def __init__(self, model_type: str, config: Settings, model_name: Optional[str] = None) -> None:
        self._model_type = model_type
        self._config = config
        self._model_name = "Model" if model_name is None else model_name
        self._model_sevice = None

    def __call__(self) -> AbstractModelService:
        if self._model_sevice is not None:
            return self._model_sevice
        else:
            if self._model_type in model_service_registry.keys():
                self._model_sevice = model_service_registry[self._model_type](self._config)
            else:
                logger.error("Unknown model type: %s", self._model_type)
                exit(1)     # throw an exception?
            return self._model_sevice


class ModelManagerDep(object):

    def __init__(self, model_service: AbstractModelService) -> None:
        self._model_manager = ModelManager(model_service.__class__, model_service.service_config)
        self._model_manager.model_service = model_service

    def __call__(self) -> ModelManager:
        return self._model_manager
