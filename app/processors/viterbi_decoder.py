import torch
from typing import Any, Dict, List, Sequence, Optional
from dataclasses import dataclass
from typing import Mapping

VITERBI_BIAS_KEYS = (
    "transition_bias_background_stay",
    "transition_bias_background_to_start",
    "transition_bias_inside_to_continue",
    "transition_bias_inside_to_end",
    "transition_bias_end_to_background",
    "transition_bias_end_to_start",
)


@dataclass(frozen=True)
class LabelInfo:
    """Label-space mappings used for inference and span decoding."""

    boundary_label_lookup: Mapping[str, Mapping[str, int]]
    token_to_span_label: Mapping[int, int]
    token_boundary_tags: Mapping[int, str | None]
    span_class_names: tuple[str, ...]
    span_label_lookup: Mapping[str, int]
    background_token_label: int
    background_span_label: int


class ViterbiDecoder:
    """CRF-style Viterbi decoder for BIOES token classification."""

    def __init__(
        self,
        label_info: LabelInfo,
        transition_bias_background_stay: float = 0.0,
        transition_bias_background_to_start: float = 0.0,
        transition_bias_inside_to_continue: float = 0.0,
        transition_bias_inside_to_end: float = 0.0,
        transition_bias_end_to_background: float = 0.0,
        transition_bias_end_to_start: float = 0.0,
    ):
        self.label_info = label_info
        self.transition_bias_background_stay = transition_bias_background_stay
        self.transition_bias_background_to_start = transition_bias_background_to_start
        self.transition_bias_inside_to_continue = transition_bias_inside_to_continue
        self.transition_bias_inside_to_end = transition_bias_inside_to_end
        self.transition_bias_end_to_background = transition_bias_end_to_background
        self.transition_bias_end_to_start = transition_bias_end_to_start

        boundary_tags = set(self.label_info.token_boundary_tags.values())
        self._has_explicit_end_labels = "E" in boundary_tags
        self._has_explicit_single_labels = "S" in boundary_tags
        self._precompute_scores()

    @classmethod
    def from_id2label(
        cls,
        id2label: Dict[int, str],
        viterbi_biases: Optional[Dict[str, float]] = None,
    ) -> Optional["ViterbiDecoder"]:
        """
        Constructs a Viterbi decoder from an id2label mapping.

        Args:
            id2label (Dict[int, str]): A mapping of label ids to label names.
            viterbi_biases (Optional[Dict[str, float]]): A mapping of Viterbi bias keys to bias values.

        Returns:
            Optional[ViterbiDecoder]: A configured Viterbi decoder, or None if tagging scheme is not IOB/IOBES.
        """
        O_LABEL = "O"
        BOUNDARY_PREFIXES = {"B", "I", "E", "S"}

        span_class_names: List[str] = [O_LABEL]
        span_label_lookup: Dict[str, int] = {O_LABEL: 0}
        boundary_label_lookup: Dict[str, Dict[str, int]] = {}
        token_to_span_label: Dict[int, int] = {}
        token_boundary_tags: Dict[int, str | None] = {}
        background_idx: Optional[int] = None

        for idx, name in id2label.items():
            idx = int(idx)
            if name == O_LABEL:
                background_idx = idx
                token_to_span_label[idx] = span_label_lookup[O_LABEL]
                token_boundary_tags[idx] = None
                continue

            parts = name.split("-", 1)
            if len(parts) != 2:
                if background_idx is None:
                    background_idx = idx
                token_to_span_label[idx] = span_label_lookup[O_LABEL]
                token_boundary_tags[idx] = None
                continue

            boundary, base_label = parts
            if boundary not in BOUNDARY_PREFIXES:
                if background_idx is None:
                    background_idx = idx
                token_to_span_label[idx] = span_label_lookup[O_LABEL]
                token_boundary_tags[idx] = None
                continue

            span_idx = span_label_lookup.get(base_label)
            if span_idx is None:
                span_idx = len(span_class_names)
                span_class_names.append(base_label)
                span_label_lookup[base_label] = span_idx

            token_to_span_label[idx] = span_idx
            token_boundary_tags[idx] = boundary
            mapping = boundary_label_lookup.setdefault(base_label, {})
            mapping[boundary] = idx

        if background_idx is None:
            return None

        for base_label, mapping in boundary_label_lookup.items():
            present = set(mapping)
            if "B" not in present or "I" not in present:
                return None
            if "E" in present and "S" not in present:
                return None
            if "S" in present and "E" not in present:
                return None

        label_info = LabelInfo(
            boundary_label_lookup={key: dict(value) for key, value in boundary_label_lookup.items()},
            token_to_span_label=dict(token_to_span_label),
            token_boundary_tags=dict(token_boundary_tags),
            span_class_names=tuple(span_class_names),
            span_label_lookup=dict(span_label_lookup),
            background_token_label=background_idx,
            background_span_label=span_label_lookup[O_LABEL],
        )
        biases = viterbi_biases or {}
        return cls(
            label_info=label_info,
            transition_bias_background_stay=biases.get("transition_bias_background_stay", 0.0),
            transition_bias_background_to_start=biases.get("transition_bias_background_to_start", 0.0),
            transition_bias_inside_to_continue=biases.get("transition_bias_inside_to_continue", 0.0),
            transition_bias_inside_to_end=biases.get("transition_bias_inside_to_end", 0.0),
            transition_bias_end_to_background=biases.get("transition_bias_end_to_background", 0.0),
            transition_bias_end_to_start=biases.get("transition_bias_end_to_start", 0.0),
        )

    def apply_viterbi_to_hf_pipeline_output(
        self,
        pipeline_output: List[Dict[str, Any]],
        id2label: Dict[int, str],
    ) -> List[Dict[str, Any]]:
        """
        Applies Viterbi decoding to a list of pipeline results.

        Args:
            pipeline_output (List[Dict[str, Any]]): A list of pipeline results.
            id2label (Dict[int, str]): A mapping of label ids to label names.

        Returns:
            List[Dict[str, Any]]: A list of pipeline results with Viterbi decoding applied.
        """
        label2id = {label: label_id for label_id, label in id2label.items()}
        num_tokens = len(pipeline_output)
        num_classes = len(id2label)
        log_probs = torch.full((num_tokens, num_classes), -1e9)

        def _resolve_label_id(entity_label: str) -> int:
            if entity_label in label2id:
                return label2id[entity_label]
            suffix = entity_label.split("-")[-1]
            if suffix.isdigit():
                suffix_id = int(suffix)
                if suffix_id in id2label:
                    return suffix_id
            return label2id.get("O", 0)

        for i, result in enumerate(pipeline_output):
            label_id = _resolve_label_id(result.get("entity", "O"))
            score = result.get("score", 0.5)
            log_probs[i, label_id] = torch.log(torch.tensor(score))

        viterbi_ids = self.decode(log_probs)
        corrected_results = []
        for result, viterbi_id in zip(pipeline_output, viterbi_ids):
            corrected_result = result.copy()
            corrected_result["entity"] = id2label.get(viterbi_id, "O")
            corrected_results.append(corrected_result)
        return corrected_results

    def decode(self, token_logprobs: torch.Tensor) -> List[int]:
        """
        Decodes one log probability tensor into label ids.

        Args:
            token_logprobs (torch.Tensor): A log probability tensor with the shape (seq_len, num_classes).
        Returns:
            List[int]: The list of label ids with length of (seq_len).
        """
        if token_logprobs.ndim != 2:
            raise ValueError("Token logprobs must have shape (seq_len, num_classes)")

        seq_len, num_classes = token_logprobs.shape
        if seq_len == 0:
            return []

        device = token_logprobs.device
        dtype = token_logprobs.dtype
        if self._start_scores.device == device and self._start_scores.dtype == dtype:
            start_scores = self._start_scores
            end_scores = self._end_scores
            transition_scores = self._transition_scores
        else:
            device_index = device.index if device.index is not None else -1
            cache_key = (device.type, device_index, dtype)
            cached_scores = self._score_cache.get(cache_key)
            if cached_scores is None:
                cached_scores = (
                    self._start_scores.to(device=device, dtype=dtype),
                    self._end_scores.to(device=device, dtype=dtype),
                    self._transition_scores.to(device=device, dtype=dtype),
                )
                self._score_cache[cache_key] = cached_scores
            start_scores, end_scores, transition_scores = cached_scores

        scores = token_logprobs[0] + start_scores
        backpointers = torch.empty((seq_len - 1, num_classes), device=device, dtype=torch.int64)

        for idx in range(1, seq_len):
            transitions = scores.unsqueeze(1) + transition_scores
            best_scores, best_paths = transitions.max(dim=0)
            scores = best_scores + token_logprobs[idx]
            backpointers[idx - 1] = best_paths

        if not torch.isfinite(scores).any():
            return token_logprobs.argmax(dim=1).tolist()

        scores = scores + end_scores
        last_label = scores.argmax()
        path = torch.empty((seq_len,), device=device, dtype=torch.int64)
        path[-1] = last_label
        for idx in range(seq_len - 2, -1, -1):
            last_label = backpointers[idx, last_label]
            path[idx] = last_label
        return path.tolist()

    def decode_many(
        self,
        token_logprobs_list: Sequence[torch.Tensor],
        device: Optional[torch.device] = None,
        batch_size: int = 128,
    ) -> List[List[int]]:
        """
        Decodes multiple log probability tensors into a list of label ids.

        Args:
            token_logprobs_list (Sequence[torch.Tensor]): A list of log probability tensors with the shape (seq_len, num_classes).
            device (tOptional[torch.device]): The device to run the decoding on.
            batch_size (int): The batch size used for GPU decoding. Defaults to 128.

        Returns:
            List[List[int]]: The list of label ids with the shape (batch_size, seq_len).

        """
        if not token_logprobs_list:
            return []
        if batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if device is None or device.type != "cuda":
            return [self.decode(scores) for scores in token_logprobs_list]

        lengths = [int(scores.shape[0]) for scores in token_logprobs_list]
        if any(scores.ndim != 2 for scores in token_logprobs_list):
            raise ValueError("decode_many expects [seq_len, num_classes] tensors")
        if any(length <= 0 for length in lengths):
            return [self.decode(scores) for scores in token_logprobs_list]

        order = sorted(
            range(len(token_logprobs_list)),
            key=lambda idx: lengths[idx],
            reverse=True,
        )
        results: List[List[int] | None] = [None] * len(token_logprobs_list)
        for start in range(0, len(order), batch_size):
            batch_indices = order[start:start + batch_size]
            batch_scores = [token_logprobs_list[idx] for idx in batch_indices]
            batch_lengths = [lengths[idx] for idx in batch_indices]
            batch_size = len(batch_scores)
            if batch_size == 0:
                return []
            num_classes = int(batch_scores[0].shape[1])
            max_len = int(max(batch_lengths))
            dtype = batch_scores[0].dtype
            for scores in batch_scores:
                if int(scores.shape[1]) != num_classes:
                    raise ValueError("All decode_many tensors must share the same class dimension")

            emissions = torch.full(
                (batch_size, max_len, num_classes),
                -float("inf"),
                device=device,
                dtype=dtype,
            )
            for row, (scores, length) in enumerate(zip(batch_scores, batch_lengths)):
                if scores.device != device or scores.dtype != dtype:
                    scores = scores.to(device=device, dtype=dtype)
                emissions[row, :length] = scores

            lengths_t = torch.tensor(batch_lengths, device=device, dtype=torch.long)
            device_index = device.index if device.index is not None else -1
            cache_key = (device.type, device_index, dtype)
            cached_scores = self._score_cache.get(cache_key)
            if cached_scores is None:
                cached_scores = (
                    self._start_scores.to(device=device, dtype=dtype),
                    self._end_scores.to(device=device, dtype=dtype),
                    self._transition_scores.to(device=device, dtype=dtype),
                )
                self._score_cache[cache_key] = cached_scores
            start_scores, end_scores, transition_scores = cached_scores

            scores = emissions[:, 0, :] + start_scores[None, :]
            backpointer_dtype = torch.int16 if num_classes <= 32767 else torch.int32
            backpointers = torch.zeros(
                (max_len - 1, batch_size, num_classes),
                device=device,
                dtype=backpointer_dtype,
            )
            batch_arange = torch.arange(batch_size, device=device, dtype=torch.long)

            for step in range(1, max_len):
                active = lengths_t > step
                if not bool(active.any().item()):
                    break
                active_idx = batch_arange[active]
                transitions = scores[active_idx].unsqueeze(2) + transition_scores
                best_scores, best_paths = transitions.max(dim=1)
                scores[active_idx] = best_scores + emissions[active_idx, step, :]
                backpointers[step - 1, active_idx] = best_paths.to(backpointer_dtype)

            bad_rows = ~torch.isfinite(scores).any(dim=1)
            scores = scores + end_scores[None, :]
            last_labels = scores.argmax(dim=1)
            paths = torch.zeros((batch_size, max_len), device=device, dtype=torch.long)
            paths[batch_arange, lengths_t - 1] = last_labels
            for step in range(max_len - 2, -1, -1):
                active = lengths_t > (step + 1)
                if not bool(active.any().item()):
                    continue
                active_idx = batch_arange[active]
                next_labels = paths[active_idx, step + 1]
                prev = backpointers[step, active_idx, next_labels].to(torch.long)
                paths[active_idx, step] = prev

            if bool(bad_rows.any().item()):
                fallback_paths = emissions.argmax(dim=2)
                bad_idx = batch_arange[bad_rows]
                for idx in bad_idx.tolist():
                    length = int(lengths_t[idx].item())
                    paths[idx, :length] = fallback_paths[idx, :length]

            decoded_batch: List[List[int]] = []
            for row, length in enumerate(batch_lengths):
                decoded_batch.append(paths[row, :length].tolist())

            for original_idx, decoded_seq in zip(batch_indices, decoded_batch):
                results[original_idx] = decoded_seq

        output: List[List[int]] = []
        for decoded in results:
            if decoded is None:
                raise RuntimeError("Internal decode_many failure: missing decoded sequence")
            assert decoded is not None
            output.append(decoded)
        return output

    def _is_valid_transition(
        self,
        prev_tag: str | None,
        prev_span: int | None,
        next_tag: str | None,
        next_span: int | None,
        background_token_idx: int,
        background_span_idx: int,
        next_idx: int,
    ) -> bool:
        next_is_background = next_span == background_span_idx or next_idx == background_token_idx
        if (next_span is None or next_tag is None) and not next_is_background:
            return False

        if prev_span is None or prev_tag is None:
            return next_is_background or next_tag in {"B", "S"}

        prev_is_background = prev_span == background_span_idx

        if prev_is_background:
            return next_is_background or next_tag in {"B", "S"}

        if prev_tag in {"E", "S"}:
            return next_is_background or next_tag in {"B", "S"}

        if prev_tag in {"B", "I"}:
            same_span = prev_span == next_span
            if same_span and next_tag in {"I", "E"}:
                return True
            if not self._has_explicit_end_labels:
                return next_is_background or next_tag in {"B", "S"}
            return False

        return False

    def _precompute_scores(self) -> None:
        num_classes = len(self.label_info.token_to_span_label)
        self._start_scores = torch.full((num_classes,), -1e9, dtype=torch.float32)
        self._end_scores = torch.full((num_classes,), -1e9, dtype=torch.float32)
        self._transition_scores = torch.full((num_classes, num_classes), -1e9, dtype=torch.float32)
        self._score_cache: dict[
            tuple[str, int, torch.dtype],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}

        background_token_idx = self.label_info.background_token_label
        background_span_idx = self.label_info.background_span_label
        token_boundary_tags = self.label_info.token_boundary_tags
        token_to_span_label = self.label_info.token_to_span_label

        for idx in range(num_classes):
            tag = token_boundary_tags.get(idx)
            span_label = token_to_span_label.get(idx)
            if tag in {"B", "S"} or idx == background_token_idx:
                self._start_scores[idx] = 0.0
            if tag in {"E", "S"} or idx == background_token_idx or (
                not self._has_explicit_end_labels and tag in {"B", "I"}
            ):
                self._end_scores[idx] = 0.0
            elif span_label == background_span_idx:
                self._start_scores[idx] = 0.0
                self._end_scores[idx] = 0.0

            for next_idx in range(num_classes):
                next_tag = token_boundary_tags.get(next_idx)
                next_span_label = token_to_span_label.get(next_idx)
                if self._is_valid_transition(
                    prev_tag=tag,
                    prev_span=span_label,
                    next_tag=next_tag,
                    next_span=next_span_label,
                    background_token_idx=background_token_idx,
                    background_span_idx=background_span_idx,
                    next_idx=next_idx,
                ):
                    self._transition_scores[idx, next_idx] = self._transition_bias(
                        prev_tag=tag,
                        prev_span=span_label,
                        next_tag=next_tag,
                        next_span=next_span_label,
                        background_token_idx=background_token_idx,
                        background_span_idx=background_span_idx,
                        prev_idx=idx,
                        next_idx=next_idx,
                    )

    def _transition_bias(
        self,
        *,
        prev_tag: str | None,
        prev_span: int | None,
        next_tag: str | None,
        next_span: int | None,
        background_token_idx: int,
        background_span_idx: int,
        prev_idx: int,
        next_idx: int,
    ) -> float:
        prev_is_background = (prev_span == background_span_idx) or (prev_idx == background_token_idx)
        next_is_background = (next_span == background_span_idx) or (next_idx == background_token_idx)

        if prev_is_background:
            if next_is_background:
                return self.transition_bias_background_stay
            if next_tag in {"B", "S"}:
                return self.transition_bias_background_to_start
            return 0.0

        if prev_tag in {"B", "I"}:
            if next_tag == "I" and prev_span == next_span:
                return self.transition_bias_inside_to_continue
            if next_tag == "E" and prev_span == next_span:
                return self.transition_bias_inside_to_end
            return 0.0

        if prev_tag in {"E", "S"}:
            if next_is_background:
                return self.transition_bias_end_to_background
            if next_tag in {"B", "S"}:
                return self.transition_bias_end_to_start
            return 0.0

        return 0.0
