import os
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Any, AsyncIterable
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TextIteratorStreamer,
)
from app import __version__ as app_version
from app.exception import ConfigurationException
from app.model_services.base import AbstractModelService
from app.domain import ModelCard, ModelType, Annotation
from app.config import Settings
from app.utils import (
    get_settings,
    non_default_device_is_available,
    unpack_model_data_package,
    ensure_tensor_contiguity,
)

logger = logging.getLogger("cms")


class HuggingFaceLlmModel(AbstractModelService):
    """A model service for Hugging Face generative LLMs."""

    def __init__(
        self,
        config: Settings,
        model_parent_dir: Optional[str] = None,
        enable_trainer: Optional[bool] = None,
        model_name: Optional[str] = None,
        base_model_file: Optional[str] = None,
    ) -> None:
        """
        Initialises the HuggingFace LLM model service with specified configurations.

        Args:
            config (Settings): The configuration for the model service.
            model_parent_dir (Optional[str]): The directory where the model package is stored. Defaults to None.
            enable_trainer (Optional[bool]): The flag to enable or disable trainers. Defaults to None.
            model_name (Optional[str]): The name of the model. Defaults to None.
            base_model_file (Optional[str]): The model package file name. Defaults to None.
        """

        super().__init__(config)
        self._config = config
        self._model_parent_dir = model_parent_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model"))
        self._model_pack_path = os.path.join(self._model_parent_dir, base_model_file or config.BASE_MODEL_FILE)
        self._enable_trainer = enable_trainer if enable_trainer is not None else config.ENABLE_TRAINING_APIS == "true"
        self._model: PreTrainedModel = None
        self._tokenizer: PreTrainedTokenizerBase = None
        self._whitelisted_tuis = set([tui.strip() for tui in config.TYPE_UNIQUE_ID_WHITELIST.split(",")])
        self._multi_label_threshold = 0.5
        self._text_generator = ThreadPoolExecutor(max_workers=50)
        self.model_name = model_name or "HuggingFace LLM model"

    @property
    def model(self) -> PreTrainedModel:
        """Getter for the HuggingFace pre-trained model."""

        return self._model

    @model.setter
    def model(self, model: PreTrainedModel) -> None:
        """Setter for the HuggingFace pre-trained model."""

        self._model = model

    @model.deleter
    def model(self) -> None:
        """Deleter for the HuggingFace pre-trained model."""

        del self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """Getter for the HuggingFace tokenizer."""

        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """Setter for the HuggingFace tokenizer."""

        self._tokenizer = tokenizer

    @tokenizer.deleter
    def tokenizer(self) -> None:
        """Deleter for the HuggingFace tokenizer."""

        del self._tokenizer

    @property
    def api_version(self) -> str:
        """Getter for the API version of the model service."""

        # APP version is used although each model service could have its own API versioning
        return app_version

    @classmethod
    def from_model(cls, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> "HuggingFaceLlmModel":
        """
        Creates a model service from a provided HuggingFace pre-trained model and its tokenizer.

        Args:
            model (PreTrainedModel): The HuggingFace pre-trained model.
            tokenizer (PreTrainedTokenizerBase): The tokenizer for the HuggingFace pre-trained model.

        Returns:
            HuggingFaceLlmModel: A HuggingFace Generative model service.
        """

        model_service = cls(get_settings(), enable_trainer=False)
        model_service.model = model
        model_service.tokenizer = tokenizer
        return model_service

    @staticmethod
    def load_model(model_file_path: str, *args: Tuple, **kwargs: Dict[str, Any]) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """
        Loads a pre-trained model and its tokenizer from a model package file.

        Args:
            model_file_path (str): The path to the model package file.
            *args (Tuple): Additional positional arguments.
            **kwargs (Dict[str, Any]): Additional keyword arguments.

        Returns:
            Tuple[PreTrainedModel, PreTrainedTokenizerBase]: A tuple containing the HuggingFace pre-trained model and its tokenizer.

        Raises:
            ConfigurationException: If the model package is not valid or not supported.
        """

        model_path = os.path.join(os.path.dirname(model_file_path), os.path.basename(model_file_path).split(".")[0])
        if unpack_model_data_package(model_file_path, model_path):
            try:
                model = AutoModelForCausalLM.from_pretrained(model_path)
                ensure_tensor_contiguity(model)
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    model_max_length=model.config.max_position_embeddings,
                    add_special_tokens=False,
                    do_lower_case=False,
                )
                logger.info("Model package loaded from %s", os.path.normpath(model_file_path))
                return model, tokenizer
            except ValueError as e:
                logger.error(e)
                raise ConfigurationException(f"Model package is not valid or not supported: {model_file_path}")
        else:
            raise ConfigurationException(f"Model package archive format is not supported: {model_file_path}")

    def init_model(self) -> None:
        """Initialises the HuggingFace model and its tokenizer based on the configuration."""

        if all([
            hasattr(self, "_model"),
            hasattr(self, "_tokenizer"),
            isinstance(self._model, PreTrainedModel),
            isinstance(self._tokenizer, PreTrainedTokenizerBase),
        ]):
            logger.warning("Model service is already initialised and can be initialised only once")
        else:
            self._model, self._tokenizer = self.load_model(self._model_pack_path)
            if self._enable_trainer:
                logger.error("Trainers are not yet implemented for HuggingFace Generative models")

    def info(self) -> ModelCard:
        """
        Retrieves a ModelCard containing information about the model.

        Returns:
            ModelCard: Information about the model.
        """
        return ModelCard(
            model_description=self.model_name,
            model_type=ModelType.HUGGINGFACE_LLM,
            api_version=self.api_version,
            model_card=self._model.config.to_dict(),
        )

    def annotate(self, text: str) -> List[Annotation]:
        raise NotImplementedError("Annotation is not yet implemented for HuggingFace Generative models")

    def batch_annotate(self, texts: List[str]) -> List[List[Annotation]]:
        raise NotImplementedError("Batch annotation is not yet implemented for HuggingFace Generative models")

    def generate(self, prompt: str, max_tokens: int = 512, **kwargs: Any) -> str:
        """
        Generates text based on the prompt.

        Args:
            prompt (str): The prompt for the text generation
            max_tokens (int): The maximum number of tokens to generate. Defaults to 512.
            **kwargs (Any): Additional keyword arguments to be passed to this method.

        Returns:
            Any: The string containing the generated text.
        """

        self.model.eval()

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if non_default_device_is_available(self._config.DEVICE):
            inputs.to(get_settings().DEVICE)

        generation_kwargs = dict(
            inputs=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        outputs = self.model.generate(**generation_kwargs)
        generated_text = self.tokenizer.decode(outputs[0], skip_prompt=True, skip_special_tokens=True)


        logger.debug("Response generation completed")

        return generated_text

    async def generate_async(self, prompt: str, max_tokens: int = 512, **kwargs: Any) -> AsyncIterable:
        """
        Asynchronously generates text stream based on the prompt.

        Args:
            prompt (str): The prompt for the text generation.
            max_tokens (int): The maximum number of tokens to generate. Defaults to 512.
            **kwargs (Any): Additional keyword arguments to be passed to the model loader.

        Returns:
            AsyncIterable: The stream containing the generated text.
        """

        self.model.eval()

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if non_default_device_is_available(self._config.DEVICE):
            inputs.to(get_settings().DEVICE)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        generation_kwargs = dict(
            inputs=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            streamer=streamer,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        try:
            _ = self._text_generator.submit(self.model.generate, **generation_kwargs)
            for content in streamer:
                yield content
                await asyncio.sleep(0.01)
        except Exception as e:
            logger.error("An error occurred while generating the response")
            logger.exception(e)
            return
        finally:
            logger.debug("Chat response generation completed")
