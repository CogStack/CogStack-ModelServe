import asyncio
from abc import ABC, abstractmethod
from typing import Any, List, Iterable, Tuple, final, Optional, Generic, TypeVar, Protocol
from app.config import Settings
from app.domain import ModelCard, Annotation

class _TrainerCommon(Protocol):
    def stop_training(self) -> bool:
        ...

    @property
    def tracker_client(self) -> Any:
        ...

T = TypeVar("T", bound=_TrainerCommon)

class AbstractModelService(ABC, Generic[T]):

    @abstractmethod
    def __init__(self, config: Settings, *args: Any, **kwargs: Any) -> None:
        self._config = config
        self._model_name = "CMS model"
        self._supervised_trainer: Optional[T] = None
        self._unsupervised_trainer: Optional[T] = None
        self._metacat_trainer: Optional[T] = None

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
    def load_model(model_file_path: str, *args: Any, **kwargs: Any) -> Any:
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
    def annotate(self, text: str) -> List[Annotation]:
        raise NotImplementedError

    async def async_annotate(self, text: str) -> List[Annotation]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.annotate, text)

    @abstractmethod
    def batch_annotate(self, texts: List[str]) -> List[List[Annotation]]:
        raise NotImplementedError

    @abstractmethod
    def init_model(self) -> None:
        raise NotImplementedError

    def train_supervised(self, *args: Any, **kwargs: Any) -> Tuple[bool, str, str]:
        raise NotImplementedError

    def train_unsupervised(self, *args: Any, **kwargs: Any) -> Tuple[bool, str, str]:
        raise NotImplementedError

    def train_metacat(self, *args: Any, **kwargs: Any) -> Tuple[bool, str, str]:
        raise NotImplementedError

    def cancel_training(self) -> bool:
        st_stopped = False if self._supervised_trainer is None else self._supervised_trainer.stop_training()
        ut_stopped = False if self._unsupervised_trainer is None else self._unsupervised_trainer.stop_training()
        mt_stopped = False if self._metacat_trainer is None else self._metacat_trainer.stop_training()
        return st_stopped or ut_stopped or mt_stopped

    def get_tracker_client(self) -> Optional[Any]:
        if self._supervised_trainer is not None:
            return self._supervised_trainer.tracker_client
        elif self._unsupervised_trainer is not None:
            return self._unsupervised_trainer.tracker_client
        elif self._metacat_trainer is not None:
            return self._metacat_trainer.tracker_client
        else:
            return None
