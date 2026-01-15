import os
from unittest.mock import MagicMock, patch
from tests.app.conftest import MODEL_PARENT_DIR
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from app import __version__
from app.domain import ModelType
from app.model_services.huggingface_llm_model import HuggingFaceLlmModel


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


def test_generate(huggingface_llm_model):
    huggingface_llm_model.init_model()
    huggingface_llm_model.model = MagicMock()
    huggingface_llm_model.tokenizer = MagicMock()
    mock_send_metrics = MagicMock()
    inputs = MagicMock()
    inputs.input_ids = MagicMock(shape=[1, 2])
    inputs.attention_mask = MagicMock()
    huggingface_llm_model.tokenizer.return_value = inputs
    outputs = [MagicMock(shape=[2])]
    huggingface_llm_model.model.generate.return_value = outputs
    huggingface_llm_model.tokenizer.decode.return_value = "Yeah."

    result = huggingface_llm_model.generate(
        prompt="Alright?",
        min_tokens=50,
        max_tokens=128,
        num_beams=2,
        temperature=0.5,
        top_p=0.8,
        stop_sequences=["end"],
        report_tokens=mock_send_metrics
    )

    huggingface_llm_model.tokenizer.assert_called_once_with(
        "Alright?",
        add_special_tokens=False,
        return_tensors="pt",
    )
    huggingface_llm_model.model.generate.assert_called_once_with(
        inputs=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        min_new_tokens=50,
        max_new_tokens=128,
        num_beams=2,
        do_sample=True,
        temperature=0.5,
        top_p=0.8,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
    )
    huggingface_llm_model.tokenizer.decode.assert_called_once_with(
        outputs[0],
        skip_prompt=True,
        skip_special_tokens=True,
    )
    mock_send_metrics.assert_called_once_with(
        prompt_token_num=2,
        completion_token_num=2,
    )
    assert result == "Yeah."


async def test_generate_async(huggingface_llm_model):
    huggingface_llm_model.init_model()
    huggingface_llm_model.model = MagicMock()
    huggingface_llm_model.tokenizer = MagicMock()
    mock_send_metrics = MagicMock()
    inputs = MagicMock()
    inputs.input_ids = MagicMock(shape=[1, 2])
    inputs.attention_mask = MagicMock()
    huggingface_llm_model.tokenizer.return_value = inputs
    outputs = [MagicMock(shape=[2])]
    huggingface_llm_model.model.generate.return_value = outputs
    huggingface_llm_model.tokenizer.decode.return_value = "Yeah."

    result = await huggingface_llm_model.generate_async(
        prompt="Alright?",
        min_tokens=50,
        max_tokens=128,
        num_beams=2,
        temperature=0.5,
        top_p=0.8,
        stop_sequences=["end"],
        report_tokens=mock_send_metrics
    )

    huggingface_llm_model.tokenizer.assert_called_once_with(
        "Alright?",
        add_special_tokens=False,
        return_tensors="pt",
    )
    huggingface_llm_model.model.generate_async.assert_called_once_with(
        inputs=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        min_new_tokens=50,
        max_new_tokens=128,
        num_beams=2,
        do_sample=True,
        temperature=0.5,
        top_p=0.8,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
    )
    huggingface_llm_model.tokenizer.decode.assert_called_once_with(
        outputs[0],
        skip_prompt=True,
        skip_special_tokens=True,
    )
    mock_send_metrics.assert_called_once_with(
        prompt_token_num=2,
        completion_token_num=2,
    )
    assert result == "Yeah."


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
    mock_normalised.cpu.return_value.numpy.return_value.tolist.return_value = [[0.1, 0.2, 0.3]]
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
    mock_normalised.cpu.return_value.numpy.return_value.tolist.return_value = [[0.1, 0.2, 0.3]]
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
