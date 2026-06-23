import torch
import pytest
from app.processors.prefix_cache import PrefixCache


class _DummyOutputs:
    def __init__(self, past_key_values):
        self.past_key_values = past_key_values


def _model(past_key_values):
    class _Model:
        device = torch.device("cpu")

        def __call__(self, **_kwargs):
            return _DummyOutputs(past_key_values)

    return _Model()


@pytest.fixture
def tokenizer():
    class _TokenBatch:
        def __init__(self):
            self.input_ids = torch.tensor([[1, 2, 3]])
            self.attention_mask = torch.tensor([[1, 1, 1]])

        def to(self, _device):
            return self

    class _Tokenizer:
        def __call__(self, _text, add_special_tokens=False, return_tensors="pt", padding=False):
            return _TokenBatch()

    return _Tokenizer()


def test_prefix_cache_get_prefix_entry(tokenizer):
    cache = PrefixCache(max_entries=2)
    past = ((torch.zeros(1, 2, 3), torch.ones(1, 2, 3)),)
    model = _model(past)

    first = cache.get_prefix_entry("system prompt", model, tokenizer)
    second = cache.get_prefix_entry("system prompt", model, tokenizer)

    assert first is not None
    assert second is not None
    assert first.past_key_values is second.past_key_values


def test_prefix_cache_returns_none_with_no_hit(tokenizer):
    cache = PrefixCache()
    model = _model(None)

    entry = cache.get_prefix_entry("system prompt", model, tokenizer)

    assert entry is None


def test_expand_past_key_values():
    past = (
        (
            torch.arange(6, dtype=torch.float32).reshape(1, 2, 3),
            torch.ones(1, 2, 3),
        ),
    )

    expanded = PrefixCache.expand_past_key_values(past, batch_size=3)

    assert expanded[0][0].shape[0] == 3
    assert expanded[0][1].shape[0] == 3

    for b in range(3):
        assert torch.equal(expanded[0][0][b], past[0][0][0])
        assert torch.equal(expanded[0][1][b], past[0][1][0])

    assert expanded[0][0].data_ptr() != past[0][0].data_ptr()
    assert expanded[0][1].data_ptr() != past[0][1].data_ptr()

    t0 = expanded[0][0][0]
    t1 = expanded[0][0][1]
    t2 = expanded[0][0][2]

    assert t0.data_ptr() != t1.data_ptr()
    assert t0.data_ptr() != t2.data_ptr()
    assert t1.data_ptr() != t2.data_ptr()

    t0_clone = t0.clone()
    expanded[0][0][0, 0, 0] += 999

    assert torch.equal(past[0][0][0], torch.arange(6).reshape(2, 3))
    assert torch.equal(expanded[0][0][1], t1)
    assert torch.equal(expanded[0][0][2], t2)
    assert not torch.equal(expanded[0][0][0], t0_clone)


def test_expand_past_key_values_batch_size_one():
    past = (
        (
            torch.arange(6, dtype=torch.float32).reshape(1, 2, 3),
            torch.ones(1, 2, 3),
        ),
    )

    expanded = PrefixCache.expand_past_key_values(past, batch_size=1)

    assert torch.equal(expanded[0][0], past[0][0])
    assert torch.equal(expanded[0][1], past[0][1])
    assert expanded[0][0].data_ptr() != past[0][0].data_ptr()
    assert expanded[0][1].data_ptr() != past[0][1].data_ptr()
