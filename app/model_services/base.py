from abc import ABC, abstractmethod
from typing import Any, List, Iterable, Tuple, Dict
from config import Settings
from domain import ModelCard


class AbstractModelService(ABC):

    def __init__(self, config: Settings) -> None:
        self._config = config

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

    def init_model(self) -> None:
        raise NotImplementedError
