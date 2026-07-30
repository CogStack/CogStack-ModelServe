import json
import hashlib
import logging
from typing import Any, Dict, Optional

from transformers import LogitsProcessor
from app.domain import LLMDecodingBackend
from app.exception import ConfigurationException, ClientException, ExtraDependencyRequiredException

try:
    from lmformatenforcer import JsonSchemaParser
except ImportError:
    JsonSchemaParser = None  # type: ignore[assignment]

try:
    from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
except ImportError:
    build_transformers_prefix_allowed_tokens_fn = None  # type: ignore[assignment]

try:
    import xgrammar as xgr
except ImportError:
    xgr = None  # type: ignore[assignment]

try:
    import llguidance
    from llguidance import LLMatcher
    from llguidance.hf import from_tokenizer as lg_from_tokenizer
    from llguidance.torch import (
        allocate_token_bitmask,
        fill_next_token_bitmask,
        apply_token_bitmask_inplace,
    )
except ImportError:
    llguidance = None  # type: ignore[assignment]
    LLMatcher = None  # type: ignore[assignment,misc]
    lg_from_tokenizer = None  # type: ignore[assignment]
    allocate_token_bitmask = None  # type: ignore[assignment]
    fill_next_token_bitmask = None  # type: ignore[assignment]
    apply_token_bitmask_inplace = None  # type: ignore[assignment]

from app.utils import dump_pydantic_object_to_dict

logger = logging.getLogger("cms")


class ConstrainedDecoder:
    """Encapsulates constrained decoding logic for JSON schema enforcement."""

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        vocab_size: Optional[int] = None,
        backend: str = LLMDecodingBackend.LM_FORMAT_ENFORCER.value,
    ) -> None:
        if build_transformers_prefix_allowed_tokens_fn is None and xgr is None and llguidance is None:
            logger.error("Cannot import lm-format-enforcer, xgrammar or llguidance. Please install it with `pip install '.[llm]'`.")
            raise ExtraDependencyRequiredException(
                "Cannot import lm-format-enforcer, xgrammar or llguidance. Please install it with `pip install '.[llm]'`."
            )
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.build_transformers_prefix_allowed_tokens_fn = build_transformers_prefix_allowed_tokens_fn
        self.backend = backend
        self._compiled_cache: Dict[str, Any] = {}
        self._compiled_cache_maxsize = 256

    @staticmethod
    def get_parser(
        schema: Dict[str, Any],
        model_service: Optional[Any] = None,
        source: str = "schema",
        backend: str = LLMDecodingBackend.LM_FORMAT_ENFORCER.value,
    ) -> Any:
        """
        Compiles a JSON schema into a parser or grammar object based on the specified backend.

        Args:
            schema (Dict[str, Any]): The JSON schema to compile.
            model_service (Optional[Any]): An instance of the model service.
            source (str): A string indicating the source of the schema.
            backend (str): The backend to use for constrained decoding, either "lm-format-enforcer", "xgrammar" or "llguidance"

        Returns:
            Any: A compiled parser or grammar object used for constrained decoding.
        """
        if backend == LLMDecodingBackend.LLGUIDANCE.value:
            if LLMatcher is None:
                raise ClientException("llguidance is not installed")
            if model_service is None or not hasattr(model_service, "tokenizer"):
                raise ClientException("model_service with tokenizer is required for llguidance backend")
            logger.debug("Using llguidance backend for constrained decoding (source=%s)", source)
            try:
                grammar = LLMatcher.grammar_from_json_schema(json.dumps(schema))
                logger.debug("llguidance grammar compiled: %s chars", len(grammar))
                return grammar
            except Exception as exc:
                logger.debug("llguidance schema compilation failed: %s", exc)
                raise ClientException(f"Invalid JSON schema for llguidance ({source})") from exc
        elif backend == LLMDecodingBackend.XGRAMMAR.value:
            if xgr is None:
                raise ClientException("xgrammar is not installed")
            if model_service is None or not hasattr(model_service, "tokenizer"):
                raise ClientException("model_service with tokenizer is required for xgrammar backend")
            logger.debug("Using xgrammar backend for constrained decoding (source=%s)", source)
            tokenizer_info = xgr.TokenizerInfo.from_huggingface(
                model_service.tokenizer,
                vocab_size=model_service.model.config.vocab_size,  # type: ignore
            )
            logger.debug(
                "xgrammar TokenizerInfo created: vocab_size=%s",
                tokenizer_info.vocab_size,
            )
            compiler = xgr.GrammarCompiler(tokenizer_info)
            logger.debug("xgrammar GrammarCompiler created")
            try:
                compiled = compiler.compile_json_schema(json.dumps(schema))
                logger.debug(
                    "xgrammar CompiledGrammar created: type=%s",
                    type(compiled).__name__,
                )
                return compiled
            except Exception as exc:
                logger.debug("xgrammar schema compilation failed: %s", exc)
                raise ClientException(f"Invalid JSON schema for xgrammar ({source})") from exc
        else:
            logger.debug("Using lm_format_enforcer backend for constrained decoding (source=%s)", source)
            try:
                parser = JsonSchemaParser(schema)
                setattr(parser, "schema", schema)
                logger.debug("lmfe JsonSchemaParser created successfully")
                return parser
            except Exception as exc:
                logger.debug("lmfe parser creation failed: %s", exc)
                raise ClientException(f"Invalid JSON schema ({source})") from exc

    def get_schema_hash(self, json_schema_parser: Optional[Any]) -> Optional[str]:
        if json_schema_parser is None:
            return None
        if self.backend == LLMDecodingBackend.LM_FORMAT_ENFORCER.value:
            schema_dict = dump_pydantic_object_to_dict(json_schema_parser.context.model_class)
            return hashlib.sha256(json.dumps(schema_dict).encode("utf-8")).hexdigest()
        if self.backend == LLMDecodingBackend.XGRAMMAR.value:
            grammar = getattr(json_schema_parser, "grammar", json_schema_parser)
            schema_json = grammar.serialize_json()
            schema_hash = hashlib.sha256(schema_json.encode("utf-8")).hexdigest()
            logger.debug("xgrammar schema hash computed: %s", schema_hash)
            return schema_hash
        if self.backend == LLMDecodingBackend.LLGUIDANCE.value:
            schema_hash = hashlib.sha256(json_schema_parser.encode("utf-8")).hexdigest()
            logger.debug("llguidance schema hash computed: %s", schema_hash)
            return schema_hash
        return None

    def apply_grammar_constraint(
        self,
        generation_kwargs: Dict[str, Any],
        json_schema_parser: Any,
        tokenizer: Optional[Any] = None,
        vocab_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Applies grammar constraints to the generation kwargs based on the specified backend.

        Args:
            generation_kwargs (Dict[str, Any]): The original generation keyword arguments.
            json_schema_parser (Any): The compiled schema parser/grammar object.
            tokenizer (Optional[Any]): The tokenizer to use for prefix allowed tokens.
            vocab_size (Optional[int]): The vocabulary size to validate against.

        Returns:
            Dict[str, Any]: The modified generation kwargs with grammar constraints applied.
        """
        tokenizer = tokenizer or self.tokenizer
        vocab_size = vocab_size if vocab_size is not None else self.vocab_size

        if json_schema_parser is None:
            return generation_kwargs

        schema_hash = self.get_schema_hash(json_schema_parser)
        cache_key = f"{self.backend}:{schema_hash}:{vocab_size}"
        cached = self._compiled_cache.get(cache_key)

        if self.backend == LLMDecodingBackend.LM_FORMAT_ENFORCER.value:
            build_fn = self.build_transformers_prefix_allowed_tokens_fn
            if build_fn is None:
                raise ConfigurationException(
                    "lm-format-enforcer is required for JSON schema enforcement"
                )
            if cached is None:
                cached = json_schema_parser
                self._compiled_cache[cache_key] = cached
                if len(self._compiled_cache) > self._compiled_cache_maxsize:
                    self._compiled_cache.pop(next(iter(self._compiled_cache)))
            logger.debug("Applying lm-format-enforcer constrained decoding via prefix_allowed_tokens_fn")
            generation_kwargs["prefix_allowed_tokens_fn"] = build_fn(tokenizer, cached)
        elif self.backend == LLMDecodingBackend.XGRAMMAR.value:
            if xgr is None:
                raise ConfigurationException(
                    "xgrammar is required for JSON schema enforcement"
                )
            if not hasattr(json_schema_parser, "tokenizer_info"):
                raise ConfigurationException(
                    "xgrammar backend requires a CompiledGrammar object with tokenizer_info"
                )
            if cached is None:
                cached = json_schema_parser
                self._compiled_cache[cache_key] = cached
                if len(self._compiled_cache) > self._compiled_cache_maxsize:
                    self._compiled_cache.pop(next(iter(self._compiled_cache)))
            compiled_vocab_size = cached.tokenizer_info.vocab_size
            logger.debug(
                "Applying xgrammar constrained decoding via LogitsProcessor "
                "(compiled_vocab_size=%s, model_vocab_size=%s)",
                compiled_vocab_size,
                vocab_size,
            )
            if vocab_size is not None and compiled_vocab_size != vocab_size:
                logger.warning(
                    "xgrammar vocab_size mismatch: compiled=%s vs model=%s. "
                    "This may cause AssertionErrors during generation.",
                    compiled_vocab_size,
                    vocab_size,
                )
            try:
                logits_processor = xgr.contrib.hf.LogitsProcessor(cached)
                logger.debug(
                    "xgrammar LogitsProcessor created: vocab_size=%s",
                    logits_processor.full_vocab_size,
                )
                generation_kwargs.setdefault("logits_processor", [])
                generation_kwargs["logits_processor"].append(logits_processor)
            except Exception as exc:
                logger.error("Failed to create xgrammar LogitsProcessor: %s", exc)
                raise ConfigurationException(
                    f"xgrammar LogitsProcessor initialization failed: {exc}. "
                    "This may indicate model/tokenizer incompatibility. "
                    "Try using lm-format-enforcer backend instead."
                ) from exc
        elif self.backend == LLMDecodingBackend.LLGUIDANCE.value:
            if llguidance is None:
                raise ConfigurationException("llguidance is required for JSON schema enforcement")
            if tokenizer is None:
                raise ConfigurationException("llguidance backend requires a tokenizer")
            if cached is None:
                cached = json_schema_parser
                self._compiled_cache[cache_key] = cached
                if len(self._compiled_cache) > self._compiled_cache_maxsize:
                    self._compiled_cache.pop(next(iter(self._compiled_cache)))
            logger.debug(
                "Applying llguidance constrained decoding via LogitsProcessor "
                "(model_vocab_size=%s)",
                vocab_size,
            )
            try:
                logits_processor = _LLGuidanceLogitsProcessor(cached, tokenizer, vocab_size)
                logger.debug("llguidance LogitsProcessor created")
                generation_kwargs.setdefault("logits_processor", [])
                generation_kwargs["logits_processor"].append(logits_processor)
            except Exception as exc:
                logger.error("Failed to create llguidance LogitsProcessor: %s", exc)
                raise ConfigurationException(
                    f"llguidance LogitsProcessor initialisation failed: {exc}. "
                    "This may indicate model/tokenizer incompatibility. "
                    "Try using lm-format-enforcer or xgrammar backend instead."
                ) from exc
        else:
            raise ConfigurationException(
                f"Unsupported grammar constraint type: {type(json_schema_parser).__name__}"
            )
        return generation_kwargs


class _LLGuidanceLogitsProcessor(LogitsProcessor):
    """Wraps an llguidance LLMatcher as a transformers LogitsProcessor."""

    def __init__(self, grammar: str, hf_tokenizer: Any, vocab_size: Optional[int]) -> None:
        self._ll_tokenizer = lg_from_tokenizer(hf_tokenizer, n_vocab=vocab_size)
        self._matcher = LLMatcher(self._ll_tokenizer, grammar)
        self._consumed = 0
        self._started = False
        self._bitmask = allocate_token_bitmask(1, self._ll_tokenizer.vocab_size)

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        seq = input_ids[0].tolist()
        if not self._started:
            self._prompt_len = len(seq)
            self._started = True
            self._consumed = len(seq)
        else:
            new_tokens = seq[self._consumed:]
            if new_tokens:
                self._matcher.consume_tokens(new_tokens)
                self._consumed = len(seq)
        bitmask = self._bitmask
        fill_next_token_bitmask(self._matcher, bitmask)
        if scores.device != bitmask.device:
            bitmask = bitmask.to(scores.device)
        apply_token_bitmask_inplace(scores, bitmask)
        return scores
