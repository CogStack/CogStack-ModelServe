from abc import ABC, abstractmethod
from typing import Any, List, Iterable, Tuple, Dict
from config import Settings
from domain import ModelCard


class AbstractModelService(ABC):

    @abstractmethod
    def __init__(self, config: Settings, *args, **kwargs) -> None:
        self._config = config
        self._model_name = "CMS model"

    @property
    def model_name(self) -> str:
        return self._model_name

    @model_name.setter
    def model_name(self, model_name: str) -> None:
        self._model_name = model_name

    @staticmethod
    @abstractmethod
    def load_model(model_file_path: str, *args, **kwargs) -> Any:
        raise NotImplementedError

    @staticmethod
    def _data_iterator(texts: List[str]) -> Iterable[Tuple[int, str]]:
        for idx, text in enumerate(texts):
            yield idx, text

    @abstractmethod
    def info(self) -> ModelCard:
        raise NotImplementedError

    @abstractmethod
    def annotate(self, text: str) -> List[Dict]:
        raise NotImplementedError

    @abstractmethod
    def batch_annotate(self, texts: List[str]) -> List[List[Dict]]:
        raise NotImplementedError

    @abstractmethod
    def init_model(self) -> None:
        raise NotImplementedError
