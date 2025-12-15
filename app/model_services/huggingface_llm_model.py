import os
import logging
import asyncio
import torch
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Any, AsyncIterable, TextIO, Callable, Union
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TextIteratorStreamer,
    BitsAndBytesConfig,
)
from app import __version__ as app_version
from app.exception import ConfigurationException
from app.model_services.base import AbstractModelService
from app.trainers.huggingface_llm_trainer import HuggingFaceLlmSupervisedTrainer
from app.domain import ModelCard, ModelType, Annotation, Device
from app.config import Settings
from app.utils import (
    get_settings,
    non_default_device_is_available,
    unpack_model_data_package,
    ensure_tensor_contiguity,
    get_model_data_package_base_name,
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
    def load_model(
        model_file_path: str,
        *args: Tuple,
        load_in_4bit: bool = False,
        **kwargs: Dict[str, Any]
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """
        Loads a pre-trained model and its tokenizer from a model package file.

        Args:
            model_file_path (str): The path to the model package file.
            *args (Tuple): Additional positional arguments.
            load_in_4bit (bool): Whether to load the model in 4-bit precision. Defaults to False.
            **kwargs (Dict[str, Any]): Additional keyword arguments.

        Returns:
            Tuple[PreTrainedModel, PreTrainedTokenizerBase]: A tuple containing the HuggingFace pre-trained model and its tokenizer.

        Raises:
            ConfigurationException: If the model package is not valid or not supported.
        """

        model_path = os.path.join(os.path.dirname(model_file_path), get_model_data_package_base_name(model_file_path))
        if unpack_model_data_package(model_file_path, model_path):
            try:
                if load_in_4bit:
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                    )
                    if get_settings().DEVICE == Device.DEFAULT.value:
                        model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            quantization_config=bnb_config,
                            device_map="auto",
                        )
                    else:
                        model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config)
                else:
                    if get_settings().DEVICE == Device.DEFAULT.value:
                        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
                    else:
                        model = AutoModelForCausalLM.from_pretrained(model_path)
                ensure_tensor_contiguity(model)
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    model_max_length=model.config.max_position_embeddings,
                    do_lower_case=False,
                )
                logger.info("Model package loaded from %s", os.path.normpath(model_file_path))
                return model, tokenizer
            except ValueError as e:
                logger.error(e)
                raise ConfigurationException(f"Model package is not valid or not supported: {model_file_path}")
        else:
            raise ConfigurationException(f"Model package archive format is not supported: {model_file_path}")

    def init_model(self, load_in_4bit: bool = False, *args: Any, **kwargs: Any) -> None:
        """Initialises the HuggingFace model and its tokenizer based on the configuration.

        Args:
            load_in_4bit (bool): Whether to load the model in 4-bit precision. Defaults to False.
            *args (Any): Additional positional arguments to be passed to this method.
            **kwargs (Any): Additional keyword arguments to be passed to this method.
        """

        if all([
            hasattr(self, "_model"),
            hasattr(self, "_tokenizer"),
            isinstance(self._model, PreTrainedModel),
            isinstance(self._tokenizer, PreTrainedTokenizerBase),
        ]):
            logger.warning("Model service is already initialised and can be initialised only once")
        else:
            self._model, self._tokenizer = self.load_model(self._model_pack_path, load_in_4bit=load_in_4bit)
            if non_default_device_is_available(get_settings().DEVICE):
                self._model.to(get_settings().DEVICE)
            if self._enable_trainer:
                self._supervised_trainer = HuggingFaceLlmSupervisedTrainer(self)

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

    def generate(
        self,
        prompt: str,
        min_tokens: int = 100,
        max_tokens: int = 512,
        num_beams: int = 5,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_sequences: Optional[List[str]] = None,
        report_tokens: Optional[Callable[[str], None]] = None,
        **kwargs: Any
    ) -> str:
        """
        Generates text based on the prompt.

        Args:
            prompt (str): The prompt for the text generation
            min_tokens (int): The minimum number of tokens to generate. Defaults to 100.
            max_tokens (int): The maximum number of tokens to generate. Defaults to 512.
            num_beams (int): The number of beams for beam search. Defaults to 5.
            temperature (float): The temperature for the text generation. Defaults to 0.7.
            top_p (float): The Top-P value for nucleus sampling. Defaults to 0.9.
            stop_sequences (Optional[List[str]]): List of strings that will stop generation when encountered. Defaults to None.
            report_tokens (Optional[Callable[[str], None]]): The callback function to send metrics. Defaults to None.
            **kwargs (Any): Additional keyword arguments to be passed to this method.

        Returns:
            Any: The string containing the generated text.
        """

        self.model.eval()

        inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        inputs.to(self.model.device)

        generation_kwargs = dict(
            inputs=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            min_new_tokens=min_tokens,
            max_new_tokens=max_tokens,
            num_beams=num_beams,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )

        outputs = self.model.generate(**generation_kwargs)
        generated_text = self.tokenizer.decode(outputs[0], skip_prompt=True, skip_special_tokens=True)

        if stop_sequences:
            for stop_seq in stop_sequences:
                if stop_seq in generated_text:
                    generated_text = generated_text.split(stop_seq)[0]
                    break

        logger.debug("Response generation completed")

        if report_tokens:
            report_tokens(
                prompt_token_num=inputs.input_ids.shape[-1],    # type: ignore
                completion_token_num=outputs[0].shape[-1],  # type: ignore
            )

        return generated_text

    async def generate_async(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_sequences: Optional[List[str]] = None,
        report_tokens: Optional[Callable[[str], None]] = None,
        **kwargs: Any
    ) -> AsyncIterable:
        """
        Asynchronously generates text stream based on the prompt.

        Args:
            prompt (str): The prompt for the text generation.
            max_tokens (int): The maximum number of tokens to generate. Defaults to 512.
            temperature (float): The temperature for the text generation. Defaults to 0.7.
            top_p (float): The Top-P value for nucleus sampling. Defaults to 0.9.
            stop_sequences (Optional[List[str]]): List of strings that will stop generation when encountered. Defaults to None.
            report_tokens (Optional[Callable[[str], None]]): The callback function to send metrics. Defaults to None.
            **kwargs (Any): Additional keyword arguments to be passed to the model loader.

        Returns:
            AsyncIterable: The stream containing the generated text.
        """

        self.model.eval()

        inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        inputs.to(self.model.device)

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
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )

        try:
            _ = self._text_generator.submit(self.model.generate, **generation_kwargs)
            output = ""
            for content in streamer:
                prev_output = output
                output += content
                if stop_sequences:
                    for stop_seq in stop_sequences:
                        if stop_seq in output:
                            remaining = output[len(prev_output):output.find(stop_seq)]
                            if remaining:
                                yield remaining
                            return
                yield content
                await asyncio.sleep(0.01)
            if report_tokens:
                report_tokens(
                    prompt_token_num=inputs.input_ids.shape[-1],    # type: ignore
                    completion_token_num=self.tokenizer(    # type: ignore
                        output,
                        add_special_tokens=False,
                        return_tensors="pt"
                    ).input_ids.shape[-1],
                )
        except Exception as e:
            logger.error("An error occurred while generating the response")
            logger.exception(e)
            return
        finally:
            logger.debug("Chat response generation completed")

    def create_embeddings(
        self,
        text: Union[str, List[str]],
        *args: Any,
        **kwargs: Any
    ) -> Union[List[float], List[List[float]]]:
        """
        Creates embeddings for a given text or list of texts using the model's hidden states.

        Args:
            text (Union[str, List[str]]): The text(s) to be embedded.
            *args (Any): Additional positional arguments to be passed to this method.
            **kwargs (Any): Additional keyword arguments to be passed to this method.

        Returns:
            List[float], List[List[float]]: The embedding vector(s) for the text(s).

        Raises:
            NotImplementedError: If the model doesn't support embeddings.
        """

        self.model.eval()

        inputs = self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        inputs.to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        last_hidden_state = outputs.hidden_states[-1]
        attention_mask = inputs["attention_mask"]
        masked_hidden_states = last_hidden_state * attention_mask.unsqueeze(-1)
        sum_hidden_states = masked_hidden_states.sum(dim=1)
        num_tokens = attention_mask.sum(dim=1, keepdim=True)
        embeddings = sum_hidden_states / num_tokens
        l2_normalised = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        results = l2_normalised.cpu().numpy().tolist()
        return results[0] if isinstance(text, str) else results

    def train_supervised(
        self,
        data_file: TextIO,
        epochs: int,
        log_frequency: int,
        training_id: str,
        input_file_name: str,
        raw_data_files: Optional[List[TextIO]] = None,
        description: Optional[str] = None,
        synchronised: bool = False,
        **hyperparams: Dict[str, Any],
    ) -> Tuple[bool, str, str]:
        """
        Initiates supervised training on the model.

        Args:
            data_file (TextIO): The file containing the trainer export data.
            epochs (int): The number of training epochs.
            log_frequency (int): The number of epochs after which training metrics will be logged.
            training_id (str): A unique identifier for the training process.
            input_file_name (str): The name of the input file to be logged.
            raw_data_files (Optional[List[TextIO]]): Additional raw data files to be logged. Defaults to None.
            description (Optional[str]): The description of the training or change logs. Defaults to empty.
            synchronised (bool): Whether to wait for the training to complete.
            **hyperparams (Dict[str, Any]): Additional hyperparameters for training.

        Returns:
            Tuple[bool, str, str]: A tuple with the first element indicating success or failure.

        Raises:
            ConfigurationException: If the supervised trainer is not enabled.
        """
        if self._supervised_trainer is None:
            raise ConfigurationException("The supervised trainer is not enabled")
        return self._supervised_trainer.train(
            data_file,
            epochs,
            log_frequency,
            training_id,
            input_file_name,
            raw_data_files,
            description,
            synchronised,
            **hyperparams,
        )
