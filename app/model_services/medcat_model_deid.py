import logging
from typing import Dict, List, TextIO, Optional
from medcat.cat import CAT
from config import Settings
from model_services.medcat_model import MedCATModel
from trainer.medcat_deid_trainer import MedcatDeIdentificationSupervisedTrainer
from domain import ModelCard
from exception import ConfigurationException

logger = logging.getLogger(__name__)


class MedCATModelDeIdentification(MedCATModel):

    def __init__(self,
                 config: Settings,
                 model_parent_dir: Optional[str] = None,
                 enable_trainer: Optional[bool] = None) -> None:
        super().__init__(config, model_parent_dir, enable_trainer)

    @property
    def model_name(self) -> str:
        return "De-Identification MedCAT model"

    @property
    def api_version(self) -> str:
        return "0.0.1"

    def info(self) -> ModelCard:
        return ModelCard(model_description=self.model_name,
                         model_type="MedCAT",
                         api_version=self.api_version)

    def batch_annotate(self, texts: List[str]) -> List[List[Dict]]:
        annotation_list = []
        for text in texts:
            annotation_list.append(self.annotate(text))
        return annotation_list

    def init_model(self) -> None:
        if hasattr(self, "_model") and isinstance(self._model, CAT):    # type: ignore
            logger.warning("Model service is already initialised and can be initialised only once")
        else:
            self._model = self.load_model(self._model_pack_path, meta_cat_config_dict=self._meta_cat_config_dict)
            if self._enable_trainer:
                self._supervised_trainer = MedcatDeIdentificationSupervisedTrainer(self)

    def train_supervised(self,
                         data_file: TextIO,
                         epochs: int,
                         log_frequency: int,
                         training_id: str,
                         input_file_name: str) -> bool:
        if self._supervised_trainer is None:
            raise ConfigurationException("Trainers are not enabled")
        return self._supervised_trainer.train(data_file, epochs, log_frequency, training_id, input_file_name)
