from typing import Optional
from config import Settings
from model_services.medcat_model import MedCATModel


class MedCATModelUmls(MedCATModel):

    def __init__(self,
                 config: Settings,
                 model_parent_dir: Optional[str] = None,
                 enable_trainer: Optional[bool] = None,
                 model_name: Optional[str] = None) -> None:
        super().__init__(config, model_parent_dir, enable_trainer)
        self.model_name = model_name or "UMLS MedCAT model"
