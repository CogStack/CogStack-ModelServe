import asyncio
import os
import logging
import time
import hashlib
import json
import re
import torch
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Any, AsyncIterable, TextIO, Callable, Union, TYPE_CHECKING
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    AsyncTextIteratorStreamer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)
from app import __version__ as app_version
from app.exception import ConfigurationException, GenerationException, ExtraDependencyRequiredException
from app.model_services.base import AbstractModelService
from app.trainers.huggingface_llm_trainer import HuggingFaceLlmSupervisedTrainer, HuggingFaceLlmUnsupervisedTrainer
from app.domain import ModelCard, ModelType, Annotation, Device, GenerationResult
from app.config import Settings
from app.processors.data_batcher import MicroBatchScheduler
from app.processors.prefix_cache import PrefixCache
from app.utils import (
    get_settings,
    non_default_device_is_available,
    unpack_model_data_package,
    ensure_tensor_contiguity,
    get_model_data_package_base_name,
    ensure_pad_token,
    dump_pydantic_object_to_dict,
    extract_json_string,
    has_turing_generation_gpu,
    resolve_safe_max_model_length,
)
if TYPE_CHECKING:
    from lmformatenforcer import JsonSchemaParser as JsonSchemaParserType
else:
    JsonSchemaParserType = Any

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
        try:
            from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
        except ImportError:
            logger.error("Cannot import JsonSchemaParser. Please install it with `pip install '.[llm]'`.")
            raise ExtraDependencyRequiredException(
                "Cannot import JsonSchemaParser. Please install it with `pip install '.[llm]'`."
            )

        super().__init__(config)
        self._config = config
        self._model_parent_dir = model_parent_dir or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "model")
        )
        self._model_pack_path = os.path.join(self._model_parent_dir, base_model_file or config.BASE_MODEL_FILE)
        self._enable_trainer = enable_trainer if enable_trainer is not None else config.ENABLE_TRAINING_APIS == "true"
        self._model: PreTrainedModel = None
        self._tokenizer: PreTrainedTokenizerBase = None
        self._assistant_model: Optional[PreTrainedModel] = None
        self._assistant_tokenizer: Optional[PreTrainedTokenizerBase] = None
        self._whitelisted_tuis = set([tui.strip() for tui in config.TYPE_UNIQUE_ID_WHITELIST.split(",")])
        self._text_generator = ThreadPoolExecutor(max_workers=10)
        self._sentence_endings = ".。!！?？:：;；\n"
        self._generation_timeout_secs = config.GENERATION_TIMEOUT_SECONDS or 180
        self._prefix_kv_cache = PrefixCache()
        self._build_transformers_prefix_allowed_tokens_fn = build_transformers_prefix_allowed_tokens_fn
        self._micro_batch_scheduler = MicroBatchScheduler(
            process_batch_fn=self._process_batched_requests,
            batch_key_fn=lambda request: request["batch_key"],
            executor=self._text_generator,
            max_batch_size=8,
            batch_wait_milliseconds=500,
            on_start=lambda max_size, wait_ms: logger.debug(
                "Started micro batch scheduling worker (max_batch_size=%s, batch_wait_milliseconds=%s)",
                max_size,
                wait_ms,
            ),
        )
        self.model_name = model_name or "HuggingFace LLM model"
        self.is_quantised = False
        self.digest = "sha256:" + hashlib.sha256(f"{model_name} {base_model_file}".encode()).hexdigest()

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
        load_in_8bit: bool = False,
        **kwargs: Dict[str, Any]
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """
        Loads a pre-trained model and its tokenizer from a model package file.

        Args:
            model_file_path (str): The path to the model package file.
            *args (Tuple): Additional positional arguments.
            load_in_4bit (bool): Whether to load the model in 4-bit precision. Defaults to False.
            load_in_8bit (bool): Whether to load the model in 8-bit precision. Defaults to False.
            **kwargs (Dict[str, Any]): Additional keyword arguments.

        Returns:
            Tuple[PreTrainedModel, PreTrainedTokenizerBase]: A tuple containing the HuggingFace pre-trained model and its tokenizer.

        Raises:
            ConfigurationException: If the model package is not valid or not supported.
        """

        model_path = os.path.join(os.path.dirname(model_file_path), get_model_data_package_base_name(model_file_path))
        if unpack_model_data_package(model_file_path, model_path):
            try:
                config = AutoConfig.from_pretrained(model_path)
                enable_sdpa_attn = get_settings().ENABLE_SPDA_ATTN == "true"

                if "quantization_config" in config.to_dict():
                    logger.info("Model already quantised, loading by ignoring 'load_in_4bit' or 'load_in_8bit' flag")
                    if get_settings().DEVICE == Device.DEFAULT.value:
                        model = HuggingFaceLlmModel._load_causal_lm(
                            enable_sdpa_attn=enable_sdpa_attn,
                            model_path=model_path,
                            device_map="auto",
                            low_cpu_mem_usage=True,
                        )
                    else:
                        model = HuggingFaceLlmModel._load_causal_lm(
                            enable_sdpa_attn=enable_sdpa_attn,
                            model_path=model_path,
                            low_cpu_mem_usage=True,
                        )
                else:
                    if load_in_4bit:
                        bnb_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.float16 if has_turing_generation_gpu() else torch.bfloat16,
                            bnb_4bit_use_double_quant=True,
                        )
                        if get_settings().DEVICE == Device.DEFAULT.value:
                            model = HuggingFaceLlmModel._load_causal_lm(
                                enable_sdpa_attn=enable_sdpa_attn,
                                model_path=model_path,
                                quantization_config=bnb_config,
                                device_map="auto",
                                low_cpu_mem_usage=True,
                            )
                        else:
                            model = HuggingFaceLlmModel._load_causal_lm(
                                enable_sdpa_attn=enable_sdpa_attn,
                                model_path=model_path,
                                quantization_config=bnb_config,
                                low_cpu_mem_usage=True,
                            )
                    elif load_in_8bit:
                        bnb_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                            bnb_4bit_compute_dtype=torch.float16 if has_turing_generation_gpu() else torch.bfloat16,
                            llm_int8_threshold=6.0,
                            llm_int8_enable_fp32_cpu_offload=False
                        )
                        if get_settings().DEVICE == Device.DEFAULT.value:
                            model = HuggingFaceLlmModel._load_causal_lm(
                                enable_sdpa_attn=enable_sdpa_attn,
                                model_path=model_path,
                                quantization_config=bnb_config,
                                device_map="auto",
                                low_cpu_mem_usage=True,
                            )
                        else:
                            model = HuggingFaceLlmModel._load_causal_lm(
                                enable_sdpa_attn=enable_sdpa_attn,
                                model_path=model_path,
                                quantization_config=bnb_config,
                                low_cpu_mem_usage=True,
                            )
                    else:
                        if get_settings().DEVICE == Device.DEFAULT.value:
                            model = HuggingFaceLlmModel._load_causal_lm(
                                enable_sdpa_attn=enable_sdpa_attn,
                                model_path=model_path,
                                device_map="auto",
                                low_cpu_mem_usage=True,
                                dtype=torch.float16 if has_turing_generation_gpu() else torch.bfloat16,
                            )
                        else:
                            model = HuggingFaceLlmModel._load_causal_lm(
                                enable_sdpa_attn=enable_sdpa_attn,
                                model_path=model_path,
                                low_cpu_mem_usage=True,
                                dtype=torch.float16 if has_turing_generation_gpu() else torch.bfloat16,
                            )
                ensure_tensor_contiguity(model)
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    model_max_length=resolve_safe_max_model_length(model.config),
                    do_lower_case=False,
                )
                ensure_pad_token(model, tokenizer)
                logger.info("Model package loaded from %s", os.path.normpath(model_file_path))
                return model, tokenizer
            except ValueError as e:
                logger.error(e)
                raise ConfigurationException(f"Model package is not valid or not supported: {model_file_path}")
        else:
            raise ConfigurationException(f"Model package archive format is not supported: {model_file_path}")

    def init_model(self,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialises the HuggingFace model and its tokenizer based on the configuration.

        Args:
            load_in_4bit (bool): Whether to load the model in 4-bit precision. Defaults to False.
            load_in_8bit (bool): Whether to load the model in 8-bit precision. Defaults to False.
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
            self._model, self._tokenizer = self.load_model(
                self._model_pack_path, load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit
            )
            if self._config.OVERRIDE_CHAT_TEMPLATE:
                self._tokenizer.chat_template = self._config.OVERRIDE_CHAT_TEMPLATE

            if self._config.ASSISTANT_MODEL_FULL_PATH:
                assistant_model_path = (
                    self._config.ASSISTANT_MODEL_FULL_PATH
                    if os.path.isabs(self._config.ASSISTANT_MODEL_FULL_PATH)
                    else os.path.join(self._model_parent_dir, self._config.ASSISTANT_MODEL_FULL_PATH)
                )
                self._assistant_model, self._assistant_tokenizer = self.load_model(
                    assistant_model_path, load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit
                )

            if (non_default_device_is_available(get_settings().DEVICE) and
                not (
                    getattr(self._model, "is_loaded_in_8bit", False) or
                    getattr(self._model, "is_loaded_in_4bit", False)
                )
            ):
                self._model.to(get_settings().DEVICE)
            if self._assistant_model is not None:
                if self._assistant_tokenizer is not None and self._assistant_tokenizer is not self._tokenizer:
                    if getattr(self._assistant_tokenizer, "vocab_size", None) != getattr(self._tokenizer, "vocab_size", None):
                        logger.warning("Assistant model tokenizer vocab_size differs from the main tokenizer")
                if hasattr(self._assistant_model, "generation_config"):
                    self._assistant_model.generation_config.assistant_confidence_threshold = 0.4
                if (non_default_device_is_available(get_settings().DEVICE) and
                    not (
                        getattr(self._assistant_model, "is_loaded_in_8bit", False) or
                        getattr(self._assistant_model, "is_loaded_in_4bit", False)
                    )
                ):
                    self._assistant_model.to(get_settings().DEVICE)
            if self._enable_trainer:
                self._supervised_trainer = HuggingFaceLlmSupervisedTrainer(self)
                self._unsupervised_trainer = HuggingFaceLlmUnsupervisedTrainer(self)

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

    def shutdown(self) -> None:
        """Stops background workers owned by this model service."""
        try:
            self._micro_batch_scheduler.stop()
        except Exception:
            logger.debug("Failed to stop micro batch scheduler cleanly", exc_info=True)
        try:
            self._text_generator.shutdown(wait=True, cancel_futures=True)
        except Exception:
            logger.debug("Failed to shutdown text generator cleanly", exc_info=True)

    def generate(
        self,
        prompt: str,
        min_tokens: int = 64,
        max_tokens: int = 512,
        num_beams: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_sequences: Optional[List[str]] = None,
        report_tokens: Optional[Callable[..., None]] = None,
        ensure_full_sentences: bool = False,
        json_schema_parser: Optional[JsonSchemaParserType] = None,
        prefix_prompt: Optional[str] = None,
        *args: Tuple,
        **kwargs: Dict[str, Any],
    ) -> GenerationResult:
        """
        Generates text based on the prompt.

        Args:
            prompt (str): The prompt for the text generation
            min_tokens (int): The minimum number of tokens to generate. Defaults to 64.
            max_tokens (int): The maximum number of tokens to generate. Defaults to 512.
            num_beams (int): The number of beams for beam search. Defaults to 1.
            temperature (float): The temperature for the text generation. Defaults to 0.7.
            top_p (float): The Top-P value for nucleus sampling. Defaults to 0.9.
            stop_sequences (Optional[List[str]]): List of strings that will stop generation when encountered. Defaults to None.
            report_tokens (Optional[Callable[[str], None]]): The callback function to send metrics. Defaults to None.
            ensure_full_sentences (bool): Whether to generate full sentences only. Defaults to False.
            json_schema_parser (Optional[JsonSchemaParser]): The JSON schema parser for validating the generated text. Defaults to None.
            prefix_prompt (Optional[str]): The prefix prompt to be used for generation. Defaults to None.

        Returns:
            GenerationResult: The generation result object.
        """
        logger.debug("Prompt after chat template applied: %s", prompt[:200])
        max_tokens = max(min_tokens, max_tokens)
        request = {
            "prompt": prompt,
            "stop_sequences": stop_sequences,
            "ensure_full_sentences": ensure_full_sentences,
            "report_tokens": report_tokens,
            "json_schema_parser": json_schema_parser,
            "prefix_prompt": prefix_prompt,
            "batch_key": (
                min_tokens,
                max_tokens,
                num_beams,
                temperature,
                top_p,
                self._get_schema_hash(json_schema_parser),
                (PrefixCache.key(prefix_prompt) if prefix_prompt else None),
            ),
        }
        future = self._micro_batch_scheduler.submit(request)
        try:
            generation_result = future.result()
        except Exception as e:
            logger.error("Failed to generate text from the request")
            logger.exception(e)
            raise GenerationException(f"Failed to generate text from the request: {str(e)}") from e
        logger.debug("Response generation completed")
        return generation_result

    async def generate_async(
        self,
        prompt: str,
        min_tokens: int = 64,
        max_tokens: int = 512,
        num_beams: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_sequences: Optional[List[str]] = None,
        report_tokens: Optional[Callable[..., None]] = None,
        ensure_full_sentences: bool = False,
        json_schema_parser: Optional[JsonSchemaParserType] = None,
        prefix_prompt: Optional[str] = None,
        *args: Tuple,
        **kwargs: Dict[str, Any],
    ) -> AsyncIterable[Union[str, GenerationResult]]:
        """
        Asynchronously generates text stream based on the prompt.

        Args:
            prompt (str): The prompt for the text generation.
            max_tokens (int): The maximum number of tokens to generate. Defaults to 512.
            min_tokens (int): The minimum number of tokens to generate. Defaults to 64.
            num_beams (int): The number of beams for beam search. Defaults to 1.
            temperature (float): The temperature for the text generation. Defaults to 0.7.
            top_p (float): The Top-P value for nucleus sampling. Defaults to 0.9.
            stop_sequences (Optional[List[str]]): List of strings that will stop generation when encountered. Defaults to None.
            report_tokens (Optional[Callable[[str], None]]): The callback function to send metrics. Defaults to None.
            ensure_full_sentences (bool): Whether to generate full sentences only. Defaults to False.
            json_schema_parser (Optional[JsonSchemaParser]): The JSON schema parser for validating the generated text. Defaults to None.
            prefix_prompt  (Optional[str]): The prefix prompt to be used for generation. Defaults to None.

        Returns:
            AsyncIterable[Union[str, GenerationResult]]: The stream containing the generated text and the generation result object.
        """

        self.model.eval()
        logger.debug("Prompt after chat template applied: %s", prompt[:200])
        full_prompt_len = None
        past_key_values = None
        prefix_len = 0
        prefix_text = prefix_prompt or ""
        use_prefix_cache = bool(prefix_prompt) and prompt.startswith(prefix_text)
        if use_prefix_cache:
            prefix_entry = self._prefix_kv_cache.get_prefix_entry(
                prefix_text,
                self.model,
                self.tokenizer,
            )
            suffix_text = prompt[len(prefix_text):]
            if prefix_entry is None or not suffix_text:
                use_prefix_cache = False
            else:
                inputs = self.tokenizer(suffix_text, add_special_tokens=False, return_tensors="pt")
                inputs.to(self.model.device)
                if inputs.input_ids.shape[1] == 0:
                    use_prefix_cache = False
                elif inputs.attention_mask.sum().item() == 0:
                    use_prefix_cache = False
                else:
                    prefix_len = int(prefix_entry.input_ids.shape[1])
                    full_prompt_len = prefix_len + int(inputs.attention_mask.sum().item())
                    prefix_mask = torch.ones(
                        (inputs.input_ids.shape[0], prefix_len),
                        dtype=inputs.attention_mask.dtype,
                        device=inputs.input_ids.device,
                    )
                    attention_mask = torch.cat([prefix_mask, inputs.attention_mask], dim=1)
                    past_key_values = PrefixCache.expand_past_key_values(
                        prefix_entry.past_key_values,
                        inputs.input_ids.shape[0],
                    )
        prefix_len = prefix_len if use_prefix_cache else 0
        if not use_prefix_cache:
            inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
            inputs.to(self.model.device)
            attention_mask = inputs.attention_mask
            full_prompt_len = int(inputs.attention_mask.sum().item())
        inputs, attention_mask, _prompt_lens, full_prompt_len, past_key_values = self._ensure_non_empty_inputs(
            prompt=prompt,
            inputs=inputs,
            attention_mask=attention_mask,
            full_prompt_lens=full_prompt_len,
            past_key_values=past_key_values,
        )

        streamer = AsyncTextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            timeout=self._generation_timeout_secs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        max_tokens = max(min_tokens, max_tokens)
        use_constrained_tokens = json_schema_parser is not None
        generation_kwargs = dict(
            inputs=inputs.input_ids,
            attention_mask=attention_mask,
            streamer=streamer,
            min_new_tokens=min_tokens,
            max_new_tokens=max_tokens,
            use_cache=True,
            num_beams=num_beams,
            do_sample=(num_beams == 1 and not use_constrained_tokens),
            temperature=temperature,
            top_p=top_p,
            top_k=0,
            repetition_penalty=(1.0 if use_constrained_tokens else 1.2),
            no_repeat_ngram_size=(0 if use_constrained_tokens else 3),
            pad_token_id=self.tokenizer.pad_token_id,
            stopping_criteria=StoppingCriteriaList([TimeoutCriteria(float(self._generation_timeout_secs))]),
        )
        if past_key_values is not None:
            generation_kwargs["past_key_values"] = past_key_values
            cache_position = torch.arange(
                prefix_len,
                prefix_len + inputs.input_ids.shape[1],
                device=inputs.input_ids.device,
            )
            generation_kwargs["cache_position"] = cache_position
        if use_constrained_tokens:
            generation_kwargs["prefix_allowed_tokens_fn"] = self._build_transformers_prefix_allowed_tokens_fn(
                self.tokenizer,
                json_schema_parser,
            )
        if self._assistant_model is not None:
            generation_kwargs["assistant_model"] = self._assistant_model
            generation_kwargs["assistant_confidence_threshold"] = 0.4
            generation_kwargs["num_assistant_tokens"] = 5
            if self._assistant_tokenizer is not None:
                if self._assistant_tokenizer.vocab_size != self._tokenizer.vocab_size:  # type: ignore
                    generation_kwargs["assistant_tokenizer"] = self._assistant_tokenizer

        try:
            generation_start = time.monotonic()
            ttft_milliseconds = -1.0
            _ = self._text_generator.submit(self.model.generate, **generation_kwargs)
            buffer = ""
            full_output = ""

            output_is_formatted = use_constrained_tokens
            if not ensure_full_sentences:
                async for content in streamer:
                    if ttft_milliseconds == -1.0 and content:
                        ttft_milliseconds = (time.monotonic() - generation_start) * 1000.0
                    prev_output = full_output
                    full_output += content
                    if stop_sequences:
                        for stop_seq in stop_sequences:
                            if stop_seq in full_output:
                                remaining = full_output[len(prev_output):full_output.find(stop_seq)]
                                if remaining and not output_is_formatted:
                                    for out_chunk in self._split_stream_chunk(remaining):
                                        await asyncio.sleep(0.1)
                                        yield out_chunk
                                return
                    if not output_is_formatted:
                        for out_chunk in self._split_stream_chunk(content):
                            await asyncio.sleep(0.1)
                            yield out_chunk
            else:
                async for content in streamer:
                    if ttft_milliseconds == -1.0 and content:
                        ttft_milliseconds = (time.monotonic() - generation_start) * 1000.0
                    buffer += content

                    if stop_sequences:
                        stop_triggered = False
                        for stop_sequence in stop_sequences:
                            if stop_sequence in buffer:
                                remaining = buffer[:buffer.find(stop_sequence)]
                                if remaining and not output_is_formatted:
                                    await asyncio.sleep(0.1)
                                    yield remaining
                                    full_output += remaining
                                stop_triggered = True
                                break

                        if stop_triggered:
                            break

                    last_sentence_ending = -1
                    for ending in self._sentence_endings:
                        pos = buffer.rfind(ending)
                        if pos > last_sentence_ending:
                            last_sentence_ending = pos

                    if last_sentence_ending != -1:
                        new_sentences = buffer[:last_sentence_ending + 1]
                        buffer = buffer[last_sentence_ending + 1:]
                        if not output_is_formatted:
                            await asyncio.sleep(0.1)
                            yield new_sentences
                        full_output += new_sentences

            if output_is_formatted:
                yield extract_json_string(full_output)

            logger.debug("Decoded raw output: %s",full_output[:200])
            prompt_token_num = (
                full_prompt_len
                if isinstance(full_prompt_len, int)
                else inputs.input_ids.shape[-1]
            )
            completion_token_num = self.tokenizer(
                full_output,
                add_special_tokens=False,
                return_tensors="pt"
            ).input_ids.shape[-1]
            total_generation_ms = (time.monotonic() - generation_start) * 1000.0
            tpot_milliseconds = total_generation_ms / float(completion_token_num) if completion_token_num > 0 else -1
            if report_tokens:
                report_tokens(
                    prompt_token_num=prompt_token_num,  # type: ignore
                    completion_token_num=completion_token_num,  # type: ignore
                    ttft_milliseconds=int(ttft_milliseconds),  # type: ignore
                    tpot_milliseconds=int(tpot_milliseconds),  # type: ignore
                )
            yield GenerationResult(
                text=full_output,
                prompt_token_num=prompt_token_num,
                completion_token_num=completion_token_num,
                ttft_ms=int(ttft_milliseconds),
                tpot_ms=int(tpot_milliseconds),
            )
        except Exception as e:
            logger.error("An error occurred while generating the response")
            logger.exception(e)
            raise GenerationException(f"Failed to generate text from the request: {str(e)}") from e
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

        texts = [text] if isinstance(text, str) else text
        all_embeddings = []

        for txt in texts:
            inputs = self.tokenizer(txt, add_special_tokens=False, truncation=False, padding=False)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            window_size = max(resolve_safe_max_model_length(self.model.config) - 2, 1)
            chunk_embeddings = []

            for start in range(0, len(input_ids), window_size):
                end = min(start + window_size, len(input_ids))
                chunk_inputs = {
                    "input_ids": torch.tensor(
                        [input_ids[start:end]], dtype=torch.long
                    ).to(self.model.device),
                    "attention_mask": torch.tensor(
                        [attention_mask[start:end]], dtype=torch.long
                    ).to(self.model.device),
                }

                with torch.no_grad():
                    outputs = self.model(**chunk_inputs, output_hidden_states=True)

                last_hidden_state = outputs.hidden_states[-1]
                chunk_attention_mask = chunk_inputs["attention_mask"]
                masked_hidden_states = last_hidden_state * chunk_attention_mask.unsqueeze(-1)
                sum_hidden_states = masked_hidden_states.sum(dim=1)
                num_tokens = chunk_attention_mask.sum(dim=1, keepdim=True)
                chunk_embedding = sum_hidden_states / num_tokens
                chunk_embeddings.append(chunk_embedding)

                if end >= len(input_ids):
                    break

            final_embedding = torch.mean(torch.cat(chunk_embeddings, dim=0), dim=0, keepdim=True)
            l2_normalised = torch.nn.functional.normalize(final_embedding, p=2, dim=1)
            all_embeddings.append(l2_normalised.float().cpu().numpy().tolist()[0])

        return all_embeddings[0] if isinstance(text, str) else all_embeddings

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

    def train_unsupervised(
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
        Initiates unsupervised training on the model.

        Args:
            data_file (TextIO): The file containing a JSON list of texts.
            epochs (int): The number of training epochs.
            log_frequency (int): The number of epochs after which training metrics will be logged.
            training_id (str): A unique identifier for the training process.
            input_file_name (str): The name of the input file to be logged.
            raw_data_files (Optional[List[TextIO]]): Additional raw data files to be logged. Defaults to None.
            description (Optional[str]): The description of the training or change logs. Defaults to empty.
            synchronised (bool): Whether to wait for the training to complete.
            **hyperparams (Dict[str, Any]): Additional hyperparameters for training.

        Returns:
            Tuple[bool, str, str]:  A tuple with the first element indicating success or failure.

        Raises:
            ConfigurationException: If the unsupervised trainer is not enabled.
        """
        if self._unsupervised_trainer is None:
            raise ConfigurationException("The unsupervised trainer is not enabled")
        return self._unsupervised_trainer.train(
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

    @staticmethod
    def _load_causal_lm(
            enable_sdpa_attn: bool = False,
            model_path: Optional[str] = None,
            **kwargs: Any,
    ) -> PreTrainedModel:
        if enable_sdpa_attn:
            try:
                fa2_kwargs = dict(kwargs)
                fa2_kwargs.setdefault("dtype", torch.float16 if has_turing_generation_gpu() else torch.bfloat16)
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_math_sdp(False)
                return AutoModelForCausalLM.from_pretrained(
                    model_path,
                    attn_implementation="sdpa",
                    dtype=torch.float16 if has_turing_generation_gpu() else torch.bfloat16,
                    **fa2_kwargs,
                )
            except Exception as e:
                logger.debug(
                    "SDPA is enabled but unavailable for this model/runtime. Falling back due to error: %s", e
                )
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)


    @staticmethod
    def _get_schema_hash(json_schema_parser: Optional[JsonSchemaParserType]) -> Optional[str]:
        if json_schema_parser is None:
            return None
        schema_dict = dump_pydantic_object_to_dict(json_schema_parser.context.model_class)
        return hashlib.sha256(json.dumps(schema_dict).encode("utf-8")).hexdigest()

    def _postprocess_generated_text(
            self,
            generated_text: str,
            stop_sequences: Optional[List[str]],
            ensure_full_sentences: bool,
    ) -> str:
        if stop_sequences:
            for stop_seq in stop_sequences:
                if stop_seq in generated_text:
                    generated_text = generated_text.split(stop_seq)[0]
                    break

        if ensure_full_sentences and generated_text and generated_text[-1] not in self._sentence_endings:
            last_pos = -1
            for ending in self._sentence_endings:
                pos = generated_text.rfind(ending)
                if pos > last_pos:
                    last_pos = pos
            if last_pos != -1:
                generated_text = generated_text[:last_pos + 1]
        return generated_text

    def _split_stream_chunk(self, text: str, max_words_per_chunk: int = 4) -> List[str]:
        """Split text into phrase-like chunks while preserving spaces/newlines."""
        if not text:
            return []
        tokens = re.findall(r"\S+\s*", text)
        if not tokens:
            return [text]

        chunks: List[str] = []
        current: List[str] = []
        word_count = 0
        for token in tokens:
            current.append(token)
            word_count += 1

            if "\n" in token:
                chunks.append("".join(current))
                current = []
                word_count = 0
                continue

            if token.rstrip().endswith((".", "!", "?", ";", ":")) and word_count >= 2:
                chunks.append("".join(current))
                current = []
                word_count = 0
                continue

            if word_count >= max_words_per_chunk:
                chunks.append("".join(current))
                current = []
                word_count = 0

        if current:
            chunks.append("".join(current))
        return chunks

    def _process_batched_requests(self, requests: List[Dict[str, Any]]) -> None:
        try:
            self.model.eval()
            prompt_texts = [req["prompt"] for req in requests]
            prefix_prompt = requests[0].get("prefix_prompt")
            prefix_text = prefix_prompt or ""
            use_prefix_cache = bool(prefix_prompt) and all(
                prompt.startswith(prefix_text) for prompt in prompt_texts
            )
            prefix_entry = None
            suffix_inputs = None
            prefix_len = 0

            if use_prefix_cache:
                prefix_entry = self._prefix_kv_cache.get_prefix_entry(
                    prefix_text,
                    self.model,
                    self.tokenizer,
                )
                if prefix_entry is None:
                    use_prefix_cache = False
                else:
                    assert prefix_entry is not None
                    suffix_texts = [prompt[len(prefix_text):] for prompt in prompt_texts]
                    if any(not suffix for suffix in suffix_texts):
                        use_prefix_cache = False
                    else:
                        suffix_inputs = self.tokenizer(
                            suffix_texts,
                            add_special_tokens=False,
                            return_tensors="pt",
                            padding=True,
                        )
                        suffix_inputs.to(self.model.device)
                        if any([
                            suffix_inputs.input_ids.shape[1] == 0,
                            (suffix_inputs.attention_mask.sum(dim=1) == 0).any(),
                        ]):
                            use_prefix_cache = False
                        prefix_len = int(prefix_entry.input_ids.shape[1])

            if not use_prefix_cache:
                inputs = self.tokenizer(prompt_texts, add_special_tokens=False, return_tensors="pt", padding=True)
                inputs.to(self.model.device)
                prompt_lens = [int(x) for x in inputs.attention_mask.sum(dim=1).tolist()]
                full_prompt_lens = prompt_lens
                batch_input_ids = inputs.input_ids
                attention_mask = inputs.attention_mask
                past_key_values = None
            else:
                assert suffix_inputs is not None
                suffix_lens = [int(x) for x in suffix_inputs.attention_mask.sum(dim=1).tolist()]
                prompt_lens = suffix_lens
                full_prompt_lens = [prefix_len + length for length in suffix_lens]
                batch_input_ids = suffix_inputs.input_ids
                prefix_mask = torch.ones(
                    (batch_input_ids.shape[0], prefix_len),
                    dtype=suffix_inputs.attention_mask.dtype,
                    device=batch_input_ids.device,
                )
                attention_mask = torch.cat([prefix_mask, suffix_inputs.attention_mask], dim=1)
                assert prefix_entry is not None
                past_key_values = PrefixCache.expand_past_key_values(
                    prefix_entry.past_key_values,
                    batch_input_ids.shape[0],
                )
            prefix_len = prefix_len if use_prefix_cache else 0
            (
                batch_input_ids,
                attention_mask,
                new_prompt_lens,
                new_full_prompt_lens,
                past_key_values,
            ) = self._ensure_non_empty_inputs(
                prompt=prompt_texts,
                inputs=batch_input_ids,
                attention_mask=attention_mask,
                prompt_lens=prompt_lens,
                full_prompt_lens=full_prompt_lens,
                past_key_values=past_key_values,
            )
            if new_prompt_lens is not None:
                prompt_lens = new_prompt_lens
            if isinstance(new_full_prompt_lens, list):
                full_prompt_lens = new_full_prompt_lens
            assert prompt_lens is not None
            assert full_prompt_lens is not None
            min_tokens, max_tokens, num_beams, temperature, top_p, _, _ = requests[0]["batch_key"]
            json_schema_parser = requests[0].get("json_schema_parser")
            use_constrained_tokens = json_schema_parser is not None
            generation_kwargs = dict(
                inputs=batch_input_ids,
                attention_mask=attention_mask,
                min_new_tokens=min_tokens,
                max_new_tokens=max_tokens,
                use_cache=True,
                num_beams=num_beams,
                do_sample=(num_beams == 1 and not use_constrained_tokens),
                temperature=temperature,
                top_p=top_p,
                top_k=0,
                repetition_penalty=(1.0 if use_constrained_tokens else 1.2),
                no_repeat_ngram_size=(0 if use_constrained_tokens else 3),
                pad_token_id=self.tokenizer.pad_token_id,
                stopping_criteria=StoppingCriteriaList([TimeoutCriteria(float(self._generation_timeout_secs))]),
            )
            if past_key_values is not None:
                generation_kwargs["past_key_values"] = past_key_values
                cache_position = torch.arange(
                    prefix_len,
                    prefix_len + batch_input_ids.shape[1],
                    device=batch_input_ids.device,
                )
                generation_kwargs["cache_position"] = cache_position
            if use_constrained_tokens:
                generation_kwargs["prefix_allowed_tokens_fn"] = self._build_transformers_prefix_allowed_tokens_fn(
                    self.tokenizer,
                    json_schema_parser,
                )
            if self._assistant_model is not None:
                generation_kwargs["assistant_model"] = self._assistant_model
                generation_kwargs["assistant_confidence_threshold"] = 0.4
                generation_kwargs["num_assistant_tokens"] = 5
                if self._assistant_tokenizer.vocab_size != self._tokenizer.vocab_size:  # type: ignore
                    generation_kwargs["assistant_tokenizer"] = self._assistant_tokenizer

            generation_start = time.monotonic()
            outputs = self.model.generate(**generation_kwargs)
            total_generation_ms = (time.monotonic() - generation_start) * 1000.0
            for idx, req in enumerate(requests):
                completion_ids = outputs[idx][prompt_lens[idx]:]
                generated_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
                logger.debug("Decoded raw output (batched): %s",generated_text[:200])
                if use_constrained_tokens:
                    generated_text = extract_json_string(generated_text)
                else:
                    generated_text = self._postprocess_generated_text(
                        generated_text,
                        req["stop_sequences"],
                        req["ensure_full_sentences"],
                    )
                prompt_token_num = full_prompt_lens[idx]
                completion_token_num = self.tokenizer(
                    generated_text,
                    add_special_tokens=False,
                    return_tensors="pt",
                ).input_ids.shape[-1]
                tpot_milliseconds = (
                    total_generation_ms / completion_token_num
                    if completion_token_num > 0
                    else -1
                )
                ttft_milliseconds = (
                    total_generation_ms - tpot_milliseconds * (completion_token_num - 1)
                    if completion_token_num > 0
                    else -1
                )
                if req["report_tokens"]:
                    req["report_tokens"](
                        prompt_token_num=prompt_token_num,  # type: ignore
                        completion_token_num=completion_token_num,  # type: ignore
                        ttft_milliseconds=int(ttft_milliseconds),  # type: ignore
                        tpot_milliseconds=int(tpot_milliseconds),  # type: ignore
                    )
                if not req["future"].done():
                    req["future"].set_result(GenerationResult(
                        text=generated_text,
                        prompt_token_num=prompt_token_num,
                        completion_token_num=completion_token_num,
                        ttft_ms=-1,
                        tpot_ms=int(tpot_milliseconds),
                    ))
        except Exception as e:
            logger.error("Batched generation failed")
            logger.exception(e)
            for req in requests:
                if not req["future"].done():
                    req["future"].set_exception(e)

    def _ensure_non_empty_inputs(
        self,
        prompt: Union[str, List[str]],
        inputs: Any,
        attention_mask: torch.Tensor,
        past_key_values: Any,
        prompt_lens: Optional[List[int]] = None,
        full_prompt_lens: Optional[Union[int, List[int]]] = None,
    ) -> Tuple[Any, torch.Tensor, Optional[List[int]], Optional[Union[int, List[int]]], Any]:
        if isinstance(prompt, list):
            try:
                input_len = int(inputs.shape[1])
            except Exception:
                input_len = 1
            if input_len > 0:
                try:
                    zero_mask = attention_mask.sum(dim=1) == 0
                    if hasattr(zero_mask, "any"):
                        if not bool(zero_mask.any()):
                            return inputs, attention_mask, prompt_lens, full_prompt_lens, past_key_values
                    else:
                        return inputs, attention_mask, prompt_lens, full_prompt_lens, past_key_values
                except Exception:
                    return inputs, attention_mask, prompt_lens, full_prompt_lens, past_key_values
            rebuilt = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt", padding=True)
            rebuilt.to(self.model.device)
            prompt_lens = [int(x) for x in rebuilt.attention_mask.sum(dim=1).tolist()]
            full_prompt_lens = prompt_lens
            return rebuilt.input_ids, rebuilt.attention_mask, prompt_lens, full_prompt_lens, None
        try:
            input_len = int(inputs.input_ids.shape[1])
        except Exception:
            input_len = 1
        if input_len > 0:
            try:
                if attention_mask.sum().item() > 0:
                    return inputs, attention_mask, None, full_prompt_lens, past_key_values
            except Exception:
                return inputs, attention_mask, None, full_prompt_lens, past_key_values
        rebuilt = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        rebuilt.to(self.model.device)
        full_prompt_len = int(rebuilt.attention_mask.sum().item())
        return rebuilt, rebuilt.attention_mask, None, full_prompt_len, None


class TimeoutCriteria(StoppingCriteria):
    """Stop generation when the timeout is reached."""

    def __init__(self, timeout_in_secs: float) -> None:
        self._timeout_in_secs = timeout_in_secs
        self._deadline = time.monotonic() + timeout_in_secs

    def __call__(
        self, input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs: Dict[str, Any]
    ) -> bool:
        now = time.monotonic()
        if now >= self._deadline:
            logger.warning(f"Generation timed out after {str(self._timeout_in_secs)} seconds")
            return True
        return False
