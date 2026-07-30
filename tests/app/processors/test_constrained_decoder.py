import hashlib
import json
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from pydantic import BaseModel
from app.domain import LLMDecodingBackend
from app.exception import ClientException, ConfigurationException, ExtraDependencyRequiredException
from app.processors.constrained_decoder import ConstrainedDecoder, _LLGuidanceLogitsProcessor


def test_init_raises_when_no_backends():
    with (
        patch("app.processors.constrained_decoder.JsonSchemaParser", None),
        patch("app.processors.constrained_decoder.build_transformers_prefix_allowed_tokens_fn", None),
        patch("app.processors.constrained_decoder.xgr", None),
        patch("app.processors.constrained_decoder.llguidance", None),
    ):
        with pytest.raises(ExtraDependencyRequiredException):
            ConstrainedDecoder()


def test_init_stores_backend_and_cache():
    decoder = _make_lmfe_decoder()
    assert decoder.backend == LLMDecodingBackend.LM_FORMAT_ENFORCER.value
    assert decoder._compiled_cache == {}
    assert decoder._compiled_cache_maxsize == 256


def test_get_parser_lm_format_enforcer_invalid_schema_raises():
    def _raise(schema):
        raise ValueError("bad schema")

    with patch("app.processors.constrained_decoder.JsonSchemaParser", _raise):
        with pytest.raises(ClientException):
            ConstrainedDecoder.get_parser({"type": "object"})


def test_get_parser_xgrammar():
    schema = {"type": "object", "properties": {"answer": {"type": "number"}}}
    compiled = SimpleNamespace(serialize_json=lambda: "{}")
    compiler = SimpleNamespace(compile_json_schema=MagicMock(return_value=compiled))
    tokenizer_info = SimpleNamespace(vocab_size=2)
    fake_xgr = SimpleNamespace(
        TokenizerInfo=SimpleNamespace(from_huggingface=MagicMock(return_value=tokenizer_info)),
        GrammarCompiler=MagicMock(return_value=compiler),
    )
    with patch("app.processors.constrained_decoder.xgr", fake_xgr):
        result = ConstrainedDecoder.get_parser(
            schema, model_service=_fake_model_service(), backend=LLMDecodingBackend.XGRAMMAR.value
        )
    assert result is compiled
    fake_xgr.TokenizerInfo.from_huggingface.assert_called_once()
    compiler.compile_json_schema.assert_called_once_with(json.dumps(schema))


def test_get_parser_xgrammar_invalid_schema_raises():
    compiler = SimpleNamespace(compile_json_schema=MagicMock(side_effect=ValueError("bad")))
    tokenizer_info = SimpleNamespace(vocab_size=2)
    fake_xgr = SimpleNamespace(
        TokenizerInfo=SimpleNamespace(from_huggingface=MagicMock(return_value=tokenizer_info)),
        GrammarCompiler=MagicMock(return_value=compiler),
    )
    with patch("app.processors.constrained_decoder.xgr", fake_xgr):
        with pytest.raises(ClientException):
            ConstrainedDecoder.get_parser(
                {}, model_service=_fake_model_service(), backend=LLMDecodingBackend.XGRAMMAR.value
            )


def test_get_parser_llguidance():
    schema = {"type": "object"}
    grammar = '{"grammar": "grammar"}'
    fake_llm = MagicMock(grammar_from_json_schema=MagicMock(return_value=grammar))
    with patch("app.processors.constrained_decoder.LLMatcher", fake_llm):
        result = ConstrainedDecoder.get_parser(
            schema, model_service=_fake_model_service(), backend=LLMDecodingBackend.LLGUIDANCE.value
        )
    assert result == grammar
    fake_llm.grammar_from_json_schema.assert_called_once_with(json.dumps(schema))


def test_get_parser_llguidance_invalid_schema_raises():
    fake_llm = MagicMock(grammar_from_json_schema=MagicMock(side_effect=ValueError("bad")))
    with patch("app.processors.constrained_decoder.LLMatcher", fake_llm):
        with pytest.raises(ClientException):
            ConstrainedDecoder.get_parser(
                {}, model_service=_fake_model_service(), backend=LLMDecodingBackend.LLGUIDANCE.value
            )


def test_get_schema_hash_lm_format_enforcer():
    decoder = _make_lmfe_decoder()
    parser = _FakeLmfeParser({})
    schema_dict = parser.context.model_class.model_dump(mode="json")
    expected = hashlib.sha256(json.dumps(schema_dict).encode("utf-8")).hexdigest()
    assert decoder.get_schema_hash(parser) == expected


def test_get_schema_hash_xgrammar():
    decoder = _make_xgr_decoder()
    grammar = SimpleNamespace(serialize_json=lambda: '{"grammar": "grammar"}')
    expected = hashlib.sha256('{"grammar": "grammar"}'.encode("utf-8")).hexdigest()
    assert decoder.get_schema_hash(grammar) == expected


def test_get_schema_hash_llguidance():
    decoder = _make_llg_decoder()
    grammar = '{"grammar": "grammar"}'
    expected = hashlib.sha256(grammar.encode("utf-8")).hexdigest()
    assert decoder.get_schema_hash(grammar) == expected


def test_get_schema_hash_unknown_backend():
    decoder = _make_lmfe_decoder()
    decoder.backend = "unknown"
    assert decoder.get_schema_hash(MagicMock()) is None


def test_apply_grammar_constraint_lmfe_sets_prefix_fn():
    decoder = _make_lmfe_decoder()
    prefix_fn = MagicMock()
    build_fn = MagicMock(return_value=prefix_fn)
    parser = _FakeLmfeParser({})
    with patch.object(decoder, "build_transformers_prefix_allowed_tokens_fn", build_fn):
        out = decoder.apply_grammar_constraint({"foo": "bar"}, parser, tokenizer=MagicMock(), vocab_size=10)
    assert "prefix_allowed_tokens_fn" in out
    assert out["prefix_allowed_tokens_fn"] is prefix_fn
    build_fn.assert_called_once()


def test_apply_grammar_constraint_cache_eviction():
    decoder = _make_lmfe_decoder()
    build_fn = MagicMock(return_value=MagicMock())
    with patch.object(decoder, "build_transformers_prefix_allowed_tokens_fn", build_fn):
        for i in range(300):
            parser = _FakeLmfeParser({})
            parser.context.model_class = _Person(name=f"name_{str(i)}", age=i)
            decoder.apply_grammar_constraint({}, parser, tokenizer=MagicMock(), vocab_size=5)
    assert len(decoder._compiled_cache) <= decoder._compiled_cache_maxsize


def test_apply_grammar_constraint_xgrammar_appends_logits_processor():
    decoder = _make_xgr_decoder()
    fake_lp = SimpleNamespace(full_vocab_size=2)
    fake_xgr = SimpleNamespace(
        contrib=SimpleNamespace(hf=SimpleNamespace(LogitsProcessor=MagicMock(return_value=fake_lp)))
    )
    compiled = SimpleNamespace(tokenizer_info=SimpleNamespace(vocab_size=2), serialize_json=lambda: "{}")
    with patch("app.processors.constrained_decoder.xgr", fake_xgr):
        out = decoder.apply_grammar_constraint({}, compiled, vocab_size=2)
    assert "logits_processor" in out
    assert out["logits_processor"][-1] is fake_lp


def test_apply_grammar_constraint_xgrammar_creation_failure():
    decoder = _make_xgr_decoder()
    fake_xgr = SimpleNamespace(
        contrib=SimpleNamespace(hf=SimpleNamespace(LogitsProcessor=MagicMock(side_effect=RuntimeError("boom!"))))
    )
    compiled = SimpleNamespace(tokenizer_info=SimpleNamespace(vocab_size=2), serialize_json=lambda: "{}")
    with patch("app.processors.constrained_decoder.xgr", fake_xgr):
        with pytest.raises(ConfigurationException):
            decoder.apply_grammar_constraint({}, compiled, vocab_size=2)


def test_apply_grammar_constraint_llguidance_appends_logits_processor():
    decoder = _make_llg_decoder()
    grammar = '{"grammar": "grammar"}'
    with (
        patch("app.processors.constrained_decoder.llguidance", SimpleNamespace()),
        patch("app.processors.constrained_decoder.lg_from_tokenizer", MagicMock(return_value=SimpleNamespace(vocab_size=2))),
        patch("app.processors.constrained_decoder.LLMatcher", MagicMock()),
        patch("app.processors.constrained_decoder.allocate_token_bitmask", MagicMock(return_value="bitmask")),
    ):
        out = decoder.apply_grammar_constraint({}, grammar, tokenizer=MagicMock(), vocab_size=2)
    assert isinstance(out["logits_processor"][-1], _LLGuidanceLogitsProcessor)


def test_apply_grammar_constraint_unsupported_backend():
    decoder = _make_lmfe_decoder()
    decoder.backend = "unsupported"
    with pytest.raises(ConfigurationException):
        decoder.apply_grammar_constraint({}, MagicMock())


def test_llguidance_logits_processor_applies_bitmask():
    scores = MagicMock()
    scores.device = "cpu"
    bitmask = MagicMock()
    bitmask.device = "cpu"
    _, mock_fill, mock_apply = _make_llguidance_processor('{"grammar": "grammar"}', scores, bitmask)
    mock_fill.assert_called_once()
    mock_apply.assert_called_once_with(scores, bitmask)


def test_llguidance_logits_processor_moves_bitmask_to_scores_device():
    scores = MagicMock()
    scores.device = "cuda"
    bitmask = MagicMock()
    bitmask.device = "cpu"
    bitmask.to.return_value = "moved"
    _, _, mock_apply = _make_llguidance_processor('{"grammar": "grammar"}', scores, bitmask)
    bitmask.to.assert_called_once_with(scores.device)
    mock_apply.assert_called_once_with(scores, "moved")


def test_llguidance_logits_processor_consumes_new_tokens():
    scores = MagicMock()
    scores.device = "cpu"
    bitmask = MagicMock()
    bitmask.device = "cpu"
    matcher = MagicMock()
    with (
        patch("app.processors.constrained_decoder.lg_from_tokenizer", MagicMock(return_value=SimpleNamespace(vocab_size=2))),
        patch("app.processors.constrained_decoder.LLMatcher", MagicMock(return_value=matcher)),
        patch("app.processors.constrained_decoder.allocate_token_bitmask", MagicMock(return_value=bitmask)),
        patch("app.processors.constrained_decoder.fill_next_token_bitmask"),
        patch("app.processors.constrained_decoder.apply_token_bitmask_inplace")
    ):
        logits_processor = _LLGuidanceLogitsProcessor('{"grammar": "grammar"}', MagicMock(), 2)
        logits_processor([SimpleNamespace(tolist=lambda: [5, 6, 7])], scores)
        logits_processor([SimpleNamespace(tolist=lambda: [5, 6, 7, 8])], scores)
    matcher.consume_tokens.assert_called_once_with([8])


class _Person(BaseModel):
    name: str
    age: int


class _FakeLmfeParser:
    def __init__(self, schema):
        self.schema = schema
        self.context = SimpleNamespace(model_class=_Person(name="a", age=1))


def _make_lmfe_decoder():
    with (
        patch("app.processors.constrained_decoder.JsonSchemaParser", lambda s: _FakeLmfeParser(s)),
        patch("app.processors.constrained_decoder.build_transformers_prefix_allowed_tokens_fn", MagicMock()),
    ):
        return ConstrainedDecoder(backend=LLMDecodingBackend.LM_FORMAT_ENFORCER.value)


def _make_xgr_decoder():
    with patch("app.processors.constrained_decoder.xgr", SimpleNamespace()):
        return ConstrainedDecoder(backend=LLMDecodingBackend.XGRAMMAR.value)


def _make_llg_decoder():
    with (
        patch("app.processors.constrained_decoder.llguidance", SimpleNamespace()),
        patch("app.processors.constrained_decoder.LLMatcher", MagicMock()),
        patch("app.processors.constrained_decoder.allocate_token_bitmask", MagicMock())
    ):
        return ConstrainedDecoder(backend=LLMDecodingBackend.LLGUIDANCE.value)


def _fake_model_service(vocab_size=2):
    return SimpleNamespace(
        tokenizer=MagicMock(),
        model=SimpleNamespace(config=SimpleNamespace(vocab_size=vocab_size)),
    )


def _make_llguidance_processor(grammar, scores, bitmask):
    matcher = MagicMock()
    input_ids = [SimpleNamespace(tolist=lambda: [5, 6, 7])]
    with (
        patch("app.processors.constrained_decoder.lg_from_tokenizer", MagicMock(return_value=SimpleNamespace(vocab_size=2))),
        patch("app.processors.constrained_decoder.LLMatcher", MagicMock(return_value=matcher)),
        patch("app.processors.constrained_decoder.allocate_token_bitmask", MagicMock(return_value=bitmask)),
        patch("app.processors.constrained_decoder.fill_next_token_bitmask") as mock_fill,
        patch("app.processors.constrained_decoder.apply_token_bitmask_inplace") as mock_apply,
    ):
        logits_processor = _LLGuidanceLogitsProcessor(grammar, MagicMock(), 2)
        logits_processor(input_ids, scores)
    return logits_processor, mock_fill, mock_apply
