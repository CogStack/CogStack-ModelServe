import os
import pytest
from unittest.mock import MagicMock, patch
from tests.app.conftest import MODEL_PARENT_DIR
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from app import __version__
from app.domain import ModelType
from app.model_services.huggingface_llm_model import HuggingFaceLlmModel, TimeoutCriteria


def test_model_name(huggingface_llm_model):
    assert huggingface_llm_model.model_name == "HuggingFace LLM model"


def test_api_version(huggingface_llm_model):
    assert huggingface_llm_model.api_version == __version__


def test_from_model(huggingface_llm_model):
    new_model_service = huggingface_llm_model.from_model(huggingface_llm_model.model, huggingface_llm_model.tokenizer)
    assert isinstance(new_model_service, HuggingFaceLlmModel)
    assert new_model_service.model == huggingface_llm_model.model
    assert new_model_service.tokenizer == huggingface_llm_model.tokenizer


def test_init_model(huggingface_llm_model):
    huggingface_llm_model.init_model()
    assert huggingface_llm_model.model is not None
    assert huggingface_llm_model.tokenizer is not None


def test_load_model(huggingface_llm_model):
    model, tokenizer = HuggingFaceLlmModel.load_model(os.path.join(MODEL_PARENT_DIR, "huggingface_llm_model.tar.gz"))
    assert isinstance(model, PreTrainedModel)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)


def test_info(huggingface_llm_model):
    huggingface_llm_model.init_model()
    model_card = huggingface_llm_model.info()
    assert type(model_card.api_version) is str
    assert type(model_card.model_description) is str
    assert model_card.model_type == ModelType.HUGGINGFACE_LLM


@pytest.mark.parametrize("ensure_full_sentences, expected_output", [
    (False, "Yeah."),
    (True, "Yeah."),
])
def test_generate(huggingface_llm_model, ensure_full_sentences, expected_output):
    huggingface_llm_model.init_model()
    huggingface_llm_model._micro_batch_scheduler._batch_wait_milliseconds = 1
    huggingface_llm_model.model = MagicMock()
    huggingface_llm_model.tokenizer = MagicMock()
    mock_send_metrics = MagicMock()
    inputs = MagicMock()
    inputs.input_ids = MagicMock(shape=[1, 2])
    inputs.attention_mask = MagicMock()
    inputs.attention_mask.sum.return_value.tolist.return_value = [2]
    huggingface_llm_model.tokenizer.return_value = inputs
    huggingface_llm_model.tokenizer.pad_token_id = 2
    outputs = [MagicMock(shape=[2])]
    huggingface_llm_model.model.generate.return_value = outputs
    completion_ids = MagicMock()
    completion_ids.shape = [2]
    outputs[0].__getitem__.return_value = completion_ids
    huggingface_llm_model.tokenizer.decode.return_value = "Yeah.[STOP] Hmm"
    huggingface_llm_model.tokenizer.apply_chat_template.return_value = "chat template text"

    result = huggingface_llm_model.generate(
        prompt="Alright?",
        min_tokens=50,
        max_tokens=128,
        num_beams=2,
        temperature=0.5,
        top_p=0.8,
        stop_sequences=["[STOP]"],
        report_tokens=mock_send_metrics,
        ensure_full_sentences=ensure_full_sentences,
    )

    huggingface_llm_model.tokenizer.assert_any_call(
        ["chat template text"],
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
    )
    huggingface_llm_model.model.generate.assert_called_once()
    call_kwargs = huggingface_llm_model.model.generate.call_args.kwargs
    assert call_kwargs["inputs"] == inputs.input_ids
    assert call_kwargs["attention_mask"] == inputs.attention_mask
    assert call_kwargs["min_new_tokens"] == 50
    assert call_kwargs["max_new_tokens"] == 128
    assert call_kwargs["use_cache"] is True
    assert call_kwargs["num_beams"] == 2
    assert call_kwargs["do_sample"] is False
    assert call_kwargs["temperature"] == 0.5
    assert call_kwargs["top_p"] == 0.8
    assert call_kwargs["repetition_penalty"] == 1.2
    assert call_kwargs["no_repeat_ngram_size"] == 3
    assert call_kwargs["pad_token_id"] == 2
    assert "stopping_criteria" in call_kwargs
    huggingface_llm_model.tokenizer.decode.assert_called_once_with(
        outputs[0][2:],
        skip_special_tokens=True,
    )
    mock_send_metrics.assert_called_once_with(
        prompt_token_num=2,
        completion_token_num=2,
    )
    assert result == expected_output
    assert "[STOP]" not in result


@pytest.mark.parametrize("ensure_full_sentences, stream_chunks, stop_sequences, expected_output, report_called", [
    (False, ["Yeah.", "[STOP]", "Hmm"], ["[STOP]"], "Yeah.", False),
    (True, ["Yeah.", "[STOP]", "Hmm"], ["[STOP]"], "Yeah.", True),
])
@pytest.mark.asyncio
async def test_generate_async(
    huggingface_llm_model,
    ensure_full_sentences,
    stream_chunks,
    stop_sequences,
    expected_output,
    report_called,
):
    huggingface_llm_model.init_model()
    huggingface_llm_model.model = MagicMock()
    huggingface_llm_model.tokenizer = MagicMock()
    mock_send_metrics = MagicMock()
    inputs = MagicMock()
    inputs.input_ids = MagicMock(shape=[1, 2])
    inputs.attention_mask = MagicMock()
    
    def mock_tokenizer_call(*args, **kwargs):
        if args and args[0] == "Alright?":
            return inputs
        mock_result = MagicMock()
        mock_result.input_ids = MagicMock(shape=[1, 2])
        return mock_result
    
    huggingface_llm_model.tokenizer.side_effect = mock_tokenizer_call
    streamer = FakeAsyncTextIteratorStreamer(stream_chunks)
    
    with patch("app.model_services.huggingface_llm_model.AsyncTextIteratorStreamer", return_value=streamer):
        huggingface_llm_model.model.generate.return_value = MagicMock(shape=[2])
        mock_future = MagicMock()
        huggingface_llm_model._text_generator.submit = MagicMock(return_value=mock_future)

        results = []
        async for chunk in huggingface_llm_model.generate_async(
            prompt="Alright?",
            min_tokens=50,
            max_tokens=128,
            num_beams=2,
            temperature=0.5,
            top_p=0.8,
            stop_sequences=stop_sequences,
            report_tokens=mock_send_metrics,
            ensure_full_sentences=ensure_full_sentences,
        ):
            results.append(chunk)
        result = "".join(results)
        submit_kwargs = huggingface_llm_model._text_generator.submit.call_args.kwargs
        assert "stopping_criteria" in submit_kwargs

    if report_called:
        mock_send_metrics.assert_called_once_with(
            prompt_token_num=2,
            completion_token_num=2,
        )
    else:
        mock_send_metrics.assert_not_called()
    assert result == expected_output
    for stop_sequence in stop_sequences:
        assert stop_sequence not in result


@pytest.mark.asyncio
async def test_generate_async_with_timeout(huggingface_llm_model):
    huggingface_llm_model.init_model()
    huggingface_llm_model._generation_timeout_secs = 2
    huggingface_llm_model.model = MagicMock()
    huggingface_llm_model.tokenizer = MagicMock()
    inputs = MagicMock()
    inputs.input_ids = MagicMock(shape=[1, 2])
    inputs.attention_mask = MagicMock()

    def _mock_tokenizer_call(*args, **kwargs):
        if args and args[0] == "Alright?":
            return inputs
        mock_result = MagicMock()
        mock_result.input_ids = MagicMock(shape=[1, 2])
        return mock_result

    huggingface_llm_model.tokenizer.side_effect = _mock_tokenizer_call
    streamer = FakeAsyncTextIteratorStreamer(["OK"])

    with patch(
        "app.model_services.huggingface_llm_model.AsyncTextIteratorStreamer", return_value=streamer
    ) as mock_streamer:
        huggingface_llm_model._text_generator.submit = MagicMock(return_value=MagicMock())
        results = []
        async for chunk in huggingface_llm_model.generate_async(prompt="Alright?"):
            results.append(chunk)

        submit_kwargs = huggingface_llm_model._text_generator.submit.call_args.kwargs
        mock_streamer.assert_called_once()
        assert "".join(results) == "OK"
        assert mock_streamer.call_args.kwargs["timeout"] == 2
        assert "stopping_criteria" in submit_kwargs
        assert len(submit_kwargs["stopping_criteria"]) == 1
        assert isinstance(submit_kwargs["stopping_criteria"][0], TimeoutCriteria)


@patch("torch.nn.functional.normalize")
@patch("torch.mean")
@patch("torch.cat")
@patch("torch.tensor")
def test_create_embeddings_single_text(mock_tensor, mock_cat, mock_mean, mock_normalise, huggingface_llm_model):
    def tensor_side_effect(*args, **kwargs):
        result = MagicMock()
        result.to.return_value = result
        return result

    huggingface_llm_model.init_model()
    huggingface_llm_model.model = MagicMock()
    huggingface_llm_model.model.config.max_position_embeddings = 10
    huggingface_llm_model.tokenizer = MagicMock()
    long_input_ids = list(range(25))
    long_attention_mask = [1] * 25
    huggingface_llm_model.tokenizer.return_value = {
        "input_ids": long_input_ids,
        "attention_mask": long_attention_mask
    }
    mock_outputs = MagicMock()
    mock_hidden_state = MagicMock()
    mock_outputs.hidden_states = [None, None, mock_hidden_state]
    huggingface_llm_model.model.return_value = mock_outputs
    mock_chunk_embedding = MagicMock()
    mock_final_embedding = MagicMock()
    mock_normalised = MagicMock()
    mock_concatenated = MagicMock()
    mock_cat.return_value = mock_concatenated
    mock_mean.return_value = mock_final_embedding
    mock_normalise.return_value = mock_normalised
    mock_normalised.float.return_value.cpu.return_value.numpy.return_value.tolist.return_value = [[0.1, 0.2, 0.3]]
    mock_tensor.side_effect = tensor_side_effect
    mock_masked = MagicMock()
    mock_summed = MagicMock()
    mock_hidden_state.__mul__.return_value = mock_masked
    mock_masked.sum.return_value = mock_summed
    mock_summed.__truediv__.return_value = mock_chunk_embedding

    result = huggingface_llm_model.create_embeddings(
        "This is a long text that should be chunked into multiple pieces"
    )

    assert huggingface_llm_model.model.call_count >= 3
    mock_cat.assert_called_once()
    mock_mean.assert_called_once()
    assert result == [0.1, 0.2, 0.3]


@patch("torch.nn.functional.normalize")
@patch("torch.mean")
@patch("torch.cat")
@patch("torch.tensor")
def test_create_embeddings_list_text(mock_tensor, mock_cat, mock_mean, mock_normalise, huggingface_llm_model):
    def tokenizer_side_effect(text, **kwargs):
        if isinstance(text, list):
            return {
                "input_ids": [list(range(10)), list(range(15))],
                "attention_mask": [[1]*10, [1]*15]
            }
        else:
            return {
                "input_ids": list(range(len(text.split()))),
                "attention_mask": [1] * len(text.split())
            }

    def tensor_side_effect(*args, **kwargs):
        result = MagicMock()
        result.to.return_value = result
        return result

    huggingface_llm_model.init_model()
    huggingface_llm_model.model = MagicMock()
    huggingface_llm_model.model.config.max_position_embeddings = 6
    huggingface_llm_model.tokenizer = MagicMock()
    huggingface_llm_model.tokenizer.side_effect = tokenizer_side_effect
    mock_outputs = MagicMock()
    mock_hidden_state = MagicMock()
    mock_outputs.hidden_states = [None, None, mock_hidden_state]
    huggingface_llm_model.model.return_value = mock_outputs
    mock_chunk_embedding = MagicMock()
    mock_final_embedding = MagicMock()
    mock_normalised = MagicMock()
    mock_concatenated = MagicMock()
    mock_cat.return_value = mock_concatenated
    mock_mean.return_value = mock_final_embedding
    mock_normalise.return_value = mock_normalised
    mock_normalised.float.return_value.cpu.return_value.numpy.return_value.tolist.return_value = [[0.1, 0.2, 0.3]]
    mock_tensor.side_effect = tensor_side_effect
    mock_masked = MagicMock()
    mock_summed = MagicMock()
    mock_hidden_state.__mul__.return_value = mock_masked
    mock_masked.sum.return_value = mock_summed
    mock_summed.__truediv__.return_value = mock_chunk_embedding

    result = huggingface_llm_model.create_embeddings([
        "Alright?",
        "This is a long text that should be chunked into multiple pieces",
    ])

    assert huggingface_llm_model.model.call_count >= 4
    assert mock_cat.call_count == 2
    assert mock_mean.call_count == 2
    assert result == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]


def test_load_model_quantization_check():
    mock_config = MagicMock()
    mock_config.to_dict.return_value = {"quantization_config": {}}
    mock_model = MagicMock(spec=PreTrainedModel)
    mock_model.config = MagicMock()
    mock_model.config.max_position_embeddings = 512
    mock_tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
    mock_tokenizer.pad_token_id = 2

    with patch("app.model_services.huggingface_llm_model.unpack_model_data_package", return_value=True), \
         patch("app.model_services.huggingface_llm_model.AutoConfig.from_pretrained", return_value=mock_config), \
         patch("app.model_services.huggingface_llm_model.AutoModelForCausalLM.from_pretrained", return_value=mock_model), \
         patch("app.model_services.huggingface_llm_model.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("app.model_services.huggingface_llm_model.BitsAndBytesConfig", return_value=MagicMock()), \
         patch("app.model_services.huggingface_llm_model.get_settings") as mock_get_settings, \
         patch("app.model_services.huggingface_llm_model.logger") as mock_logger:

        mock_settings = MagicMock()
        mock_settings.DEVICE = "cpu"
        mock_get_settings.return_value = mock_settings

        model, tokenizer = HuggingFaceLlmModel.load_model("dummy_path", load_in_4bit=True)

        mock_logger.info.assert_any_call("Model already quantised, loading by ignoring 'load_in_4bit' or 'load_in_8bit' flag")
        assert model == mock_model
        assert tokenizer == mock_tokenizer

    with patch("app.model_services.huggingface_llm_model.unpack_model_data_package", return_value=True), \
         patch("app.model_services.huggingface_llm_model.AutoConfig.from_pretrained", return_value=mock_config), \
         patch("app.model_services.huggingface_llm_model.AutoModelForCausalLM.from_pretrained", return_value=mock_model), \
         patch("app.model_services.huggingface_llm_model.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("app.model_services.huggingface_llm_model.BitsAndBytesConfig", return_value=MagicMock()), \
         patch("app.model_services.huggingface_llm_model.get_settings") as mock_get_settings, \
         patch("app.model_services.huggingface_llm_model.logger") as mock_logger:

        mock_settings = MagicMock()
        mock_settings.DEVICE = "cpu"
        mock_get_settings.return_value = mock_settings
        mock_config.to_dict.return_value = {"quantization_config": {}}

        model, tokenizer = HuggingFaceLlmModel.load_model("dummy_path", load_in_8bit=True)

        mock_logger.info.assert_any_call("Model already quantised, loading by ignoring 'load_in_4bit' or 'load_in_8bit' flag")
        assert model == mock_model
        assert tokenizer == mock_tokenizer

    with patch("app.model_services.huggingface_llm_model.unpack_model_data_package", return_value=True), \
         patch("app.model_services.huggingface_llm_model.AutoConfig.from_pretrained", return_value=mock_config), \
         patch("app.model_services.huggingface_llm_model.AutoModelForCausalLM.from_pretrained", return_value=mock_model), \
         patch("app.model_services.huggingface_llm_model.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("app.model_services.huggingface_llm_model.BitsAndBytesConfig", return_value=MagicMock()), \
         patch("app.model_services.huggingface_llm_model.get_settings") as mock_get_settings, \
         patch("app.model_services.huggingface_llm_model.logger") as mock_logger:

        mock_settings = MagicMock()
        mock_settings.DEVICE = "cpu"
        mock_get_settings.return_value = mock_settings
        mock_config.to_dict.return_value = {}

        model, tokenizer = HuggingFaceLlmModel.load_model("dummy_path", load_in_4bit=True)

        mock_logger.info.assert_called_once_with("Model package loaded from %s", "dummy_path")
        assert model == mock_model
        assert tokenizer == mock_tokenizer

    with patch("app.model_services.huggingface_llm_model.unpack_model_data_package", return_value=True), \
         patch("app.model_services.huggingface_llm_model.AutoConfig.from_pretrained", return_value=mock_config), \
         patch("app.model_services.huggingface_llm_model.AutoModelForCausalLM.from_pretrained", return_value=mock_model), \
         patch("app.model_services.huggingface_llm_model.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("app.model_services.huggingface_llm_model.BitsAndBytesConfig", return_value=MagicMock()), \
         patch("app.model_services.huggingface_llm_model.get_settings") as mock_get_settings, \
         patch("app.model_services.huggingface_llm_model.logger") as mock_logger:

        mock_settings = MagicMock()
        mock_settings.DEVICE = "cpu"
        mock_get_settings.return_value = mock_settings
        mock_config.to_dict.return_value = {}

        model, tokenizer = HuggingFaceLlmModel.load_model("dummy_path", load_in_8bit=True)

        mock_logger.info.assert_called_once_with("Model package loaded from %s", "dummy_path")
        assert model == mock_model
        assert tokenizer == mock_tokenizer


class FakeAsyncTextIteratorStreamer:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        self._iter = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration
