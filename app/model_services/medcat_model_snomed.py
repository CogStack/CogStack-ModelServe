import logging
from typing import Optional, final

from config import Settings
from domain import ModelCard, ModelType

from model_services.medcat_model import MedCATModel

logger = logging.getLogger("cms")


@final
class MedCATModelSnomed(MedCATModel):
    def __init__(
        self,
        config: Settings,
        model_parent_dir: Optional[str] = None,
        enable_trainer: Optional[bool] = None,
        model_name: Optional[str] = None,
        base_model_file: Optional[str] = None,
    ) -> None:
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
        return "0.0.1"

    def info(self) -> ModelCard:
        return ModelCard(
            model_description=self.model_name,
            model_type=ModelType.MEDCAT_SNOMED,
            api_version=self.api_version,
            model_card=self.model.get_model_card(as_dict=True),
        )
