import pytest
from unittest.mock import Mock, patch
from app.exception import ManagedModelException
from app.processors.lora_adaptor import LoraAdaptor


@patch("app.processors.lora_adaptor.get_peft_model")
@patch("app.processors.lora_adaptor.LoraConfig")
def test_apply_uses_explicit_target_modules(mock_lora_config, mock_get_peft_model):
    model = Mock()
    peft_model = Mock()
    lora_config = Mock()
    mock_lora_config.return_value = lora_config
    mock_get_peft_model.return_value = peft_model

    result_model, result_config = LoraAdaptor.apply(
        model=model,
        task_type="TOKEN_CLS",
        target_modules=["q_proj", "k_proj"],
        r=16,
        lora_alpha=64,
        lora_dropout=0.2,
    )

    assert result_model is peft_model
    assert result_config is lora_config
    mock_lora_config.assert_called_once_with(
        task_type="TOKEN_CLS",
        r=16,
        lora_alpha=64,
        lora_dropout=0.2,
        target_modules=["q_proj", "k_proj"],
    )
    mock_get_peft_model.assert_called_once_with(model, lora_config)


@patch(
    "app.processors.lora_adaptor.TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING",
    {"bert": ["query", "key", "value"]},
)
@patch("app.processors.lora_adaptor.get_peft_model")
@patch("app.processors.lora_adaptor.LoraConfig")
def test_apply_uses_peft_mapping_when_target_modules_omitted(mock_lora_config, mock_get_peft_model):
    model = Mock()
    model.config.model_type = "bert"
    peft_model = Mock()
    lora_config = Mock()
    mock_lora_config.return_value = lora_config
    mock_get_peft_model.return_value = peft_model

    result_model, result_config = LoraAdaptor.apply(
        model=model,
        task_type="TOKEN_CLS",
    )

    assert result_model is peft_model
    assert result_config is lora_config
    mock_lora_config.assert_called_once_with(
        task_type="TOKEN_CLS",
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "key", "value"],
    )


@patch(
    "app.processors.lora_adaptor.TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING",
    {"bert": ["query", "key", "value"]},
)
@patch("app.processors.lora_adaptor.get_peft_model")
@patch("app.processors.lora_adaptor.LoraConfig")
def test_apply_falls_back_to_detected_target_modules_when_mapping_missing(mock_lora_config, mock_get_peft_model):
    class _LeafModule:
        def children(self):
            return iter(())

    class _DummyModel:
        config = Mock(model_type="unknown")

        def named_modules(self):
            return iter([
                ("", Mock()),
                ("encoder.layer.0.attention.q_proj", _LeafModule()),
                ("encoder.layer.0.attention.k_proj", _LeafModule()),
                ("encoder.layer.0.attention.v_proj", _LeafModule()),
            ])

    model = _DummyModel()
    peft_model = Mock()
    lora_config = Mock()
    mock_lora_config.return_value = lora_config
    mock_get_peft_model.return_value = peft_model

    result_model, result_config = LoraAdaptor.apply(
        model=model,  # type: ignore[arg-type]
        task_type="TOKEN_CLS",
    )

    assert result_model is peft_model
    assert result_config is lora_config
    mock_lora_config.assert_called_once_with(
        task_type="TOKEN_CLS",
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj"],
    )


@patch(
    "app.processors.lora_adaptor.TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING",
    {"bert": ["query", "key", "value"]},
)
@patch("app.processors.lora_adaptor.get_peft_model")
@patch("app.processors.lora_adaptor.LoraConfig")
def test_apply_with_detected_target_modules_when_peft_rejects_configured_modules(mock_lora_config, mock_get_peft_model):
    class _LeafModule:
        def children(self):
            return iter(())

    class _DummyModel:
        config = Mock(model_type="bert")

        def named_modules(self):
            return iter([
                ("", Mock()),
                ("encoder.layer.0.attention.q_proj", _LeafModule()),
                ("encoder.layer.0.attention.k_proj", _LeafModule()),
                ("encoder.layer.0.attention.v_proj", _LeafModule()),
            ])

    model = _DummyModel()
    peft_model = Mock()
    initial_lora_config = Mock()
    fallback_lora_config = Mock()
    mock_lora_config.side_effect = [initial_lora_config, fallback_lora_config]
    mock_get_peft_model.side_effect = [
        ValueError("Target modules {'value', 'key', 'query'} not found in the base model."),
        peft_model,
    ]

    result_model, result_config = LoraAdaptor.apply(
        model=model,  # type: ignore[arg-type]
        task_type="TOKEN_CLS",
    )

    assert result_model is peft_model
    assert result_config is fallback_lora_config
    assert mock_lora_config.call_count == 2
    assert mock_get_peft_model.call_count == 2
    fallback_call_kwargs = mock_lora_config.call_args_list[1].kwargs
    assert fallback_call_kwargs["target_modules"] == ["q_proj", "k_proj", "v_proj"]


@patch("app.processors.lora_adaptor.TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING", {})
def test_apply_raises_when_no_target_modules_can_be_resolved():
    model = Mock()
    model.config.model_type = "unknown"
    model.named_modules.return_value = iter([])

    with pytest.raises(ManagedModelException) as exc_info:
        LoraAdaptor.apply(
            model=model,
            task_type="TOKEN_CLS",
        )

    assert "Could not determine LoRA target modules" in str(exc_info.value)
