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


def test_create_embeddings_single_text(huggingface_llm_model):
    """Test create_embeddings with single text input."""
    huggingface_llm_model.init_model()
    huggingface_llm_model.model = MagicMock()
    huggingface_llm_model.tokenizer = MagicMock()
    mock_hidden_states = [MagicMock(), MagicMock(), MagicMock()]
    mock_outputs = MagicMock()
    mock_outputs.hidden_states = mock_hidden_states
    mock_last_hidden_state = MagicMock()
    mock_last_hidden_state.shape = [1, 3, 768]
    mock_hidden_states[-1] = mock_last_hidden_state
    mock_attention_mask = MagicMock()
    mock_attention_mask.shape = [1, 3]
    mock_attention_mask.sum.return_value = MagicMock()
    mock_attention_mask.sum.return_value.unsqueeze.return_value = MagicMock()
    mock_inputs = MagicMock()
    mock_inputs.__getitem__.side_effect = lambda key: mock_attention_mask if key == "attention_mask" else MagicMock()
    huggingface_llm_model.tokenizer.return_value = mock_inputs
    huggingface_llm_model.model.return_value = mock_outputs
    expected_result = [0.1, 0.2, 0.3]
    mock_embeddings_batch = MagicMock()
    mock_first_embedding = MagicMock()
    mock_cpu_tensor = MagicMock()
    mock_numpy_array = MagicMock()
    mock_numpy_array.tolist.return_value = expected_result    
    mock_embeddings_batch.__getitem__.return_value = mock_first_embedding
    mock_first_embedding.cpu.return_value = mock_cpu_tensor
    mock_cpu_tensor.numpy.return_value = mock_numpy_array
    mock_masked_hidden_states = MagicMock()
    mock_sum_hidden_states = MagicMock()
    mock_num_tokens = MagicMock()
    mock_last_hidden_state.__mul__.return_value = mock_masked_hidden_states
    mock_masked_hidden_states.sum.return_value = mock_sum_hidden_states
    mock_attention_mask.sum.return_value = mock_num_tokens
    mock_sum_hidden_states.__truediv__.return_value = mock_embeddings_batch
    
    result = huggingface_llm_model.create_embeddings("Alright")
    
    huggingface_llm_model.model.eval.assert_called_once()    
    huggingface_llm_model.tokenizer.assert_called_once_with(
        "Alright",
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
        truncation=True
    )    
    huggingface_llm_model.model.assert_called_once_with(
        **mock_inputs,
        output_hidden_states=True
    )
    
    assert result is not None


def test_create_embeddings_list_text(huggingface_llm_model):
    huggingface_llm_model.init_model()
    huggingface_llm_model.model = MagicMock()
    huggingface_llm_model.tokenizer = MagicMock()
    mock_hidden_states = [MagicMock(), MagicMock(), MagicMock()]
    mock_outputs = MagicMock()
    mock_outputs.hidden_states = mock_hidden_states
    mock_last_hidden_state = MagicMock()
    mock_last_hidden_state.shape = [2, 3, 768]
    mock_hidden_states[-1] = mock_last_hidden_state
    mock_attention_mask = MagicMock()
    mock_attention_mask.shape = [2, 3]
    mock_attention_mask.sum.return_value = MagicMock()
    mock_attention_mask.sum.return_value.unsqueeze.return_value = MagicMock()
    mock_inputs = MagicMock()
    mock_inputs.__getitem__.side_effect = lambda key: mock_attention_mask if key == "attention_mask" else MagicMock()
    huggingface_llm_model.tokenizer.return_value = mock_inputs
    huggingface_llm_model.model.return_value = mock_outputs    
    mock_embeddings_batch = MagicMock()
    mock_first_embedding = MagicMock()
    mock_cpu_tensor = MagicMock()
    mock_numpy_array = MagicMock()
    mock_numpy_array.tolist.return_value = [[0.1, 0.2, 0.3],[0.1, 0.2, 0.3]]
    mock_embeddings_batch.__getitem__.return_value = mock_first_embedding
    mock_first_embedding.cpu.return_value = mock_cpu_tensor
    mock_cpu_tensor.numpy.return_value = mock_numpy_array
    mock_masked_hidden_states = MagicMock()
    mock_sum_hidden_states = MagicMock()
    mock_num_tokens = MagicMock()
    mock_last_hidden_state.__mul__.return_value = mock_masked_hidden_states
    mock_masked_hidden_states.sum.return_value = mock_sum_hidden_states
    mock_attention_mask.sum.return_value = mock_num_tokens
    mock_sum_hidden_states.__truediv__.return_value = mock_embeddings_batch
    
    result = huggingface_llm_model.create_embeddings(["Alright", "Alright"])
    
    huggingface_llm_model.tokenizer.assert_called_once_with(
        ["Alright", "Alright"],
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )    
    assert result is not None
