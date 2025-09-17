import asyncio
from abc import ABC, abstractmethod
from typing import Any, List, Iterable, Tuple, final, Optional, Generic, TypeVar, Protocol, AsyncIterable, Union
from app.config import Settings
from app.domain import ModelCard, Annotation

class _TrainerCommon(Protocol):
    """A protocol for defining the common properties and methods that trainers should implement."""

    def stop_training(self) -> bool:
        ...

    @property
    def tracker_client(self) -> Any:
        ...

T = TypeVar("T", bound=_TrainerCommon)

class AbstractModelService(ABC, Generic[T]):
    """An abstract base class defining the common interface for NER model services."""

    @abstractmethod
    def __init__(self, config: Settings, *args: Any, **kwargs: Any) -> None:
        """
        Initialises the model service with the configuration and any additional implementation specific arguments.

        Args:
            config (Settings): The configuration for the model service.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.
        """

        self._config = config
        self._model_name = "CMS model"
        self._supervised_trainer: Optional[T] = None
        self._unsupervised_trainer: Optional[T] = None
        self._metacat_trainer: Optional[T] = None

    @final
    @property
    def service_config(self) -> Settings:
        """Getter for the model service configuration."""

        return self._config

    @property
    def model_name(self) -> str:
        """Getter for the model name."""

        return self._model_name

    @model_name.setter
    def model_name(self, model_name: str) -> None:
        """Setter for the model name."""

        self._model_name = model_name

    @staticmethod
    @abstractmethod
    def load_model(model_file_path: str, *args: Any, **kwargs: Any) -> Any:
        """
        Loads a model from a specified path to the model package.

        Args:
            model_file_path (str): The path to the file containing the model.
            *args (Any): Additional positional arguments to be passed to the model loader.
            **kwargs (Any): Additional keyword arguments to be passed to the model loader.

        Returns:
            Any: The loaded model components.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """

        raise NotImplementedError

    @staticmethod
    @final
    def _data_iterator(texts: List[str]) -> Iterable[Tuple[int, str]]:
        """
        Creates an generator from a list of texts.

        Args:
            texts (List[str]): The list of texts to generate.

        Yields:
            Tuple[int, str]: A tuple containing the index of the text and the text itself.
        """

        for idx, text in enumerate(texts):
            yield idx, text

    @abstractmethod
    def info(self) -> ModelCard:
        """
        Retrieves information about the model in the form of a ModelCard.

        Returns:
            ModelCard: A card containing information about the model.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """

        raise NotImplementedError

    @abstractmethod
    def annotate(self, text: str) -> List[Annotation]:
        """
        Annotates a given text and returns a list of annotations.

        Args:
            text (str): The text to be annotated.

        Returns:
            List[Annotation]: A list of annotations for the text.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """

        raise NotImplementedError

    async def annotate_async(self, text: str) -> List[Annotation]:
        """
        Asynchronously annotates a given text and returns a list of annotations.

        Args:
            text (str): The text to be annotated.

        Returns:
            List[Annotation]: A list of annotations for the text.
        """

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.annotate, text)

    @abstractmethod
    def batch_annotate(self, texts: List[str]) -> List[List[Annotation]]:
        """
        Annotates texts in batches and returns a list of lists of annotations.

        Args:
            texts (List[str]): The list of texts to be annotated.

        Returns:
            List[List[Annotation]]: A list where each element is a list of annotations for the corresponding text.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """

        raise NotImplementedError

    @abstractmethod
    def init_model(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialises the model and auxiliary resources.

        Args:
            *args (Any): Additional positional arguments to be passed to this method.
            **kwargs (Any): Additional keyword arguments to be passed to this method.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """

        raise NotImplementedError

    def generate(self, prompt: str, *args: Any, **kwargs: Any) -> str:
        """
        Generates a text based on a given prompt.

        Args:
            prompt (str): The text to be used as the prompt.
            *args (Any): Additional positional arguments to be passed to this method.
            **kwargs (Any): Additional keyword arguments to be passed to this method.

        Returns:
            srt: The string containing the generated text.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """

        raise NotImplementedError

    def generate_async(self, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterable:
        """
        Asynchronously generates a text stream based on a given prompt.

        Args:
            prompt (str): The text to be used as the prompt.
            *args (Any): Additional positional arguments to be passed to this method.
            **kwargs (Any): Additional keyword arguments to be passed to this method.

        Returns:
            AsyncIterable: The stream containing the generated text.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """

        raise NotImplementedError

    def create_embeddings(
        self,
        text: Union[str, List[str]],
        *args: Any,
        **kwargs: Any
    ) -> Union[List[float], List[List[float]]]:
        """
        Creates embeddings for a given text or list of texts.

        Args:
            text (Union[str, List[str]]): The text(s) to be embedded.
            *args (Any): Additional positional arguments to be passed to this method.
            **kwargs (Any): Additional keyword arguments to be passed to this method.

        Returns:
            Union[List[float], List[List[float]]]: The embedding vector(s) for the text(s).

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """

        raise NotImplementedError

    def train_supervised(self, *args: Any, **kwargs: Any) -> Tuple[bool, str, str]:
        """
        Initiates supervised training on the model.

        Args:
            *args (Any): Additional positional arguments to be passed to this method.
            **kwargs (Any): Additional keyword arguments to be passed to this method.

        Returns:
            Tuple[bool, str, str]: A tuple with the first element indicating success or failure, the second element
            being the experiment id, and the third element being the run id.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """

        raise NotImplementedError

    def train_unsupervised(self, *args: Any, **kwargs: Any) -> Tuple[bool, str, str]:
        """
        Initiates unsupervised training on the model.

        Args:
            *args (Any): Additional positional arguments to be passed to this method.
            **kwargs (Any): Additional keyword arguments to be passed to this method.

        Returns:
            Tuple[bool, str, str]: A tuple with the first element indicating success or failure, the second element
            being the experiment id, and the third element being the run id.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """

        raise NotImplementedError

    def train_metacat(self, *args: Any, **kwargs: Any) -> Tuple[bool, str, str]:
        """
        Initiates metacat training on the model.

        Args:
            *args (Any): Additional positional arguments to be passed to this method.
            **kwargs (Any): Additional keyword arguments to be passed to this method.

        Returns:
            Tuple[bool, str, str]: A tuple with the first element indicating success or failure, the second element
            being the experiment id, and the third element being the run id.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """

        raise NotImplementedError

    def cancel_training(self) -> bool:
        """
        Attempts to cancel the training process initiated by supervised, unsupervised, and metacat trainers.

        Returns:
            bool: True if the cancellation was acknowledged for any training processes, otherwise False.
        """

        st_stopped = False if self._supervised_trainer is None else self._supervised_trainer.stop_training()
        ut_stopped = False if self._unsupervised_trainer is None else self._unsupervised_trainer.stop_training()
        mt_stopped = False if self._metacat_trainer is None else self._metacat_trainer.stop_training()
        return st_stopped or ut_stopped or mt_stopped

    def get_tracker_client(self) -> Optional[Any]:
        """
        Retrieves the tracker client used by the model trainer.

        Returns:
            Optional[Any]: The tracker client if available, otherwise None.
        """

        if self._supervised_trainer is not None:
            return self._supervised_trainer.tracker_client
        elif self._unsupervised_trainer is not None:
            return self._unsupervised_trainer.tracker_client
        elif self._metacat_trainer is not None:
            return self._metacat_trainer.tracker_client
        else:
            return None
