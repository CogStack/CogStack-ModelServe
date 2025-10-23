import logging
from typing import Optional, final
from app import __version__ as app_version
from app.model_services.medcat_model import MedCATModel
from app.config import Settings
from app.domain import ModelCard, ModelType

logger = logging.getLogger("cms")


@final
class MedCATModelSnomed(MedCATModel):
    """A model service for MedCAT SNOMED models."""

    def __init__(
        self,
        config: Settings,
        model_parent_dir: Optional[str] = None,
        enable_trainer: Optional[bool] = None,
        model_name: Optional[str] = None,
        base_model_file: Optional[str] = None,
    ) -> None:
        """
        Initialises the MedCAT SNOMED model service with specified configurations.

        Args:
            config (Settings): The configuration for the model service.
            model_parent_dir (Optional[str]): The directory where the model package is stored. Defaults to None.
            enable_trainer (Optional[bool]): The flag to enable or disable trainers. Defaults to None.
            model_name (Optional[str]): The name of the model. Defaults to None.
            base_model_file (Optional[str]): The model package file name. Defaults to None.
        """

        super().__init__(
            config,
            model_parent_dir=model_parent_dir,
            enable_trainer=enable_trainer,
            model_name=model_name,
            base_model_file=base_model_file,
        )
        self.model_name = model_name or "SNOMED MedCAT model"

    @property
    def api_version(self) -> str:
        """Getter for the API version of the model service."""

        # APP version is used although each model service could have its own API versioning
        return app_version

    def info(self) -> ModelCard:
        """
        Retrieves information about the MedCAT SNOMED model.

        Returns:
            ModelCard: A card containing information about the MedCAT SNOMED model.
        """

        assert self.model is not None, "Model is not initialised"
        return ModelCard(
            model_description=self.model_name,
            model_type=ModelType.MEDCAT_SNOMED,
            api_version=self.api_version,
            model_card=dict(self.model.get_model_card(as_dict=True)),
        )
