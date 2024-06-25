import asyncio
from abc import ABC, abstractmethod
from typing import Any, List, Iterable, Tuple, Dict, final
from config import Settings
from domain import ModelCard


class AbstractModelService(ABC):

    @abstractmethod
    def __init__(self, config: Settings, *args: Tuple, **kwargs: Dict[str, Any]) -> None:
        self._config = config
        self._model_name = "CMS model"

    @final
    @property
    def service_config(self) -> Settings:
        return self._config

    @property
    def model_name(self) -> str:
        return self._model_name

    @model_name.setter
    def model_name(self, model_name: str) -> None:
        self._model_name = model_name

    @staticmethod
    @abstractmethod
    def load_model(model_file_path: str, *args: Tuple, **kwargs: Dict[str, Any]) -> Any:
        raise NotImplementedError

    @staticmethod
    @final
    def _data_iterator(texts: List[str]) -> Iterable[Tuple[int, str]]:
        for idx, text in enumerate(texts):
            yield idx, text

    @abstractmethod
    def info(self) -> ModelCard:
        raise NotImplementedError

    @abstractmethod
    def annotate(self, text: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

    async def async_annotate(self, text: str) -> Dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.annotate, text)  # type: ignore

    @abstractmethod
    def batch_annotate(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        raise NotImplementedError

    @abstractmethod
    def init_model(self) -> None:
        raise NotImplementedError

    def train_supervised(self, *args: Tuple, **kwargs: Dict[str, Any]) -> bool:
        return False

    def train_unsupervised(self, *args: Tuple, **kwargs: Dict[str, Any]) -> bool:
        return False

    def train_metacat(self, *args: Tuple, **kwargs: Dict[str, Any]) -> bool:
        return False
