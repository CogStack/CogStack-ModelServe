import logging
from typing import Optional
from model_services.medcat_model import MedCATModel
from config import Settings

logger = logging.getLogger(__name__)


class MedCATModelSnomed(MedCATModel):

    def __init__(self,
                 config: Settings,
                 model_parent_dir: Optional[str] = None,
                 enable_trainer: Optional[bool] = None,
                 model_name: Optional[str] = None) -> None:
        super().__init__(config, model_parent_dir, enable_trainer)
        self.model_name = model_name or "SNOMED MedCAT model"

    @property
    def api_version(self) -> str:
        return "0.0.1"
