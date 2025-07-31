import os
from unittest.mock import MagicMock
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
        max_tokens=128,
        temperature=0.5,
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
        max_new_tokens=128,
        do_sample=False,
        temperature=0.5,
        top_p=0.9,
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
        max_tokens=128,
        temperature=0.5,
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
        max_new_tokens=128,
        do_sample=False,
        temperature=0.5,
        top_p=0.9,
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