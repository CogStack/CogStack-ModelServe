from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional, Tuple
import hashlib

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class PrefixCacheEntry:
    input_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    past_key_values: Any
    device: torch.device


class PrefixCache:
    def __init__(self, max_entries: int = 10) -> None:
        self._max_entries = max_entries
        self._cache: "OrderedDict[str, PrefixCacheEntry]" = OrderedDict()

    @staticmethod
    def key(prefix_prompt: str) -> str:
        """
        Creates a hash key for the prefix prompt

        Args:
            prefix_prompt (str): The prefix prompt to hash.

        Returns:
            str: The hash key for the prefix prompt.
        """
        return hashlib.sha256(prefix_prompt.encode("utf-8")).hexdigest()

    @staticmethod
    def expand_past_key_values(past_key_values: Any, batch_size: int) -> Tuple:
        """
        Expands the past key values to the batch size

        Args:
            past_key_values (Any): The past key values to expand.
            batch_size (int): The batch size to expand to.

        Returns:
            Tuple: The expanded past key values.
        """
        if batch_size == 1:
            return tuple(
                tuple(
                    t.clone().contiguous() for t in layer
                ) for layer in past_key_values
            )
        else:
            return tuple(
                tuple(
                    tensor.expand(batch_size, *tensor.shape[1:]).contiguous() for tensor in layer
                ) for layer in past_key_values
            )

    def get_prefix_entry(
        self,
        prefix_prompt: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
    ) -> Optional[PrefixCacheEntry]:
        """
        Gets the prefix entry from the cache or creates it if it doesn't exist

        Args:
            prefix_prompt (str): The prefix prompt to get the entry for.
            model (PreTrainedModel): The model to create the prefix entry for.
            tokenizer (PreTrainedTokenizerBase): The tokenizer to use.

        Returns:
            Optional[PrefixCacheEntry]: The prefix entry if it exists, otherwise None.
        """
        if not prefix_prompt:
            return None
        key = PrefixCache.key(prefix_prompt)
        cached = self._cache.get(key)
        if cached is not None and cached.device == model.device:
            self._cache.move_to_end(key)
            return cached

        inputs = tokenizer(prefix_prompt, add_special_tokens=False, return_tensors="pt")
        inputs.to(model.device)
        with torch.no_grad():
            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                use_cache=True,
            )
        if outputs.past_key_values is None:
            return None

        entry = PrefixCacheEntry(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            past_key_values=outputs.past_key_values,
            device=model.device,
        )
        self._cache[key] = entry
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_entries:
            self._cache.popitem(last=False)
        return entry

    def clear(self) -> None:
        """Removes all entries from the prefix cache."""
        self._cache.clear()
