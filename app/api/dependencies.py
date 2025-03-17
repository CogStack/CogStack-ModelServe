import logging
import re
from typing import Union
from typing_extensions import Annotated

from fastapi import HTTPException, Query
from starlette.status import HTTP_400_BAD_REQUEST

from typing import Optional
from app.config import Settings
from app.domain import ModelType
from app.exception import ConfigurationException
from app.registry import model_service_registry
from app.model_services.base import AbstractModelService
from app.management.model_manager import ModelManager

TRACKING_ID_REGEX = re.compile(r"^[a-zA-Z0-9][\w\-]{0,255}$")

logger = logging.getLogger("cms")


class ModelServiceDep(object):
    """Dependency class for injecting the CMS model service based on the given model type."""

    def __init__(self, model_type: ModelType, config: Settings, model_name: str = "Model") -> None:
        """
        Initialises a new ModelServiceDep instance.

        Args:
            model_type (ModelType): The type of the model service to be initialised.
            config (Settings): Configuration settings for the model service.
            model_name (str): Optional name for the model service. Defaults to "Model".
        """
        self._model_type = model_type
        self._config = config
        self._model_name = model_name
        self._model_service: Optional[AbstractModelService] = None

    @property
    def model_service(self) -> Optional[AbstractModelService]:
        """Getter for the model service."""

        return self._model_service

    @model_service.setter
    def model_service(self, model_service: AbstractModelService) -> None:
        """Setter for the model service."""
        self._model_service = model_service

    def __call__(self) -> AbstractModelService:
        """
        Gets the model service instance. If not yet initialised, it will be created based on the model type.

        Returns:
            The model service instance.

        Raises:
            ConfigurationException: If the model type is not known to the model service registry.
        """

        if self._model_service is not None:
            return self._model_service
        else:
            if self._model_type in model_service_registry.keys():
                self._model_service = model_service_registry[self._model_type](self._config)
            else:
                logger.error("Unknown model type: %s", self._model_type)
                raise ConfigurationException(f"Unknown model type: {self._model_type}")
            return self._model_service


class ModelManagerDep(object):
    """Dependency class for injecting the model manager."""

    def __init__(self, model_service: AbstractModelService) -> None:
        """
        Initialises a new ModelManagerDep instance.

        Args:
            model_service (AbstractModelService): The model service to which the model manager is associated.
        """
        self._model_manager = ModelManager(model_service.__class__, model_service.service_config)
        self._model_manager.model_service = model_service

    def __call__(self) -> ModelManager:
        """
        Gets the model manager instance.

        Returns:
            The model manager instance.
        """
        return self._model_manager


def validate_tracking_id(
    tracking_id: Annotated[Union[str, None], Query(description="The tracking ID of the requested task")] = None,
) -> Union[str, None]:
    """
    Validates a tracking ID against a predefined pattern.

    Args:
        tracking_id (str): The tracking ID to be validated.

    Returns:
        The validated tracking ID string, or None if no tracking ID was provided.

    Raises:
        HTTPException: If the tracking ID does not match the predefined pattern.
    """

    if tracking_id is not None and TRACKING_ID_REGEX.match(tracking_id) is None:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Invalid tracking ID '{tracking_id}', must be an alphanumeric string of length 1 to 256",
        )
    return tracking_id
