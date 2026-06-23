import pandas as pd
from typing import Any, Dict, List, Optional, Iterable
from torch import nn, mean, cat
from transformers import PreTrainedModel
from app.domain import TaggingScheme
from app.utils import parse_label_into_id_and_name


class TagProcessor:

    @staticmethod
    def update_model_by_tagging_scheme(
        model: PreTrainedModel,
        concepts: List[str],
        tagging_scheme: TaggingScheme,
    ) -> PreTrainedModel:
        """
        Updates the token classification head of the model by appending new labels according to the tagging scheme.

        Args:
            model (PreTrainedModel): The Hugging Face token classification model to be updated.
            concepts (List[str]): The list of concept names to be added as new labels.
            tagging_scheme (TaggingScheme): The tagging scheme used for the model, either "flat","iob" or "iobes".

        Returns:
            PreTrainedModel: The updated model with new labels added to the classification head.
        """
        head_name, head = TagProcessor._get_classification_head(model)
        avg_weight = mean(head.weight, dim=0, keepdim=True)
        avg_bias = mean(head.bias, dim=0, keepdim=True)
        if tagging_scheme == TaggingScheme.IOB:
            for concept in concepts:
                b_label = f"B-{concept}"
                i_label = f"I-{concept}"
                head = TagProcessor._append_label_to_head(model, head_name, head, b_label, avg_weight, avg_bias)
                head = TagProcessor._append_label_to_head(model, head_name, head, i_label, avg_weight, avg_bias)
        elif tagging_scheme == TaggingScheme.IOBES:
            for concept in concepts:
                s_label = f"S-{concept}"
                b_label = f"B-{concept}"
                i_label = f"I-{concept}"
                e_label = f"E-{concept}"
                head = TagProcessor._append_label_to_head(model, head_name, head, s_label, avg_weight, avg_bias)
                head = TagProcessor._append_label_to_head(model, head_name, head, b_label, avg_weight, avg_bias)
                head = TagProcessor._append_label_to_head(model, head_name, head, i_label, avg_weight, avg_bias)
                head = TagProcessor._append_label_to_head(model, head_name, head, e_label, avg_weight, avg_bias)
        else:
            for concept in concepts:
                head = TagProcessor._append_label_to_head(model, head_name, head, concept, avg_weight, avg_bias)
        return model

    @staticmethod
    def generate_chuncks_by_tagging_scheme(
        annotations: List[Dict],
        tokenized: Dict[str, List],
        delfault_label_id: int,
        pad_token_id: int,
        pad_label_id: int,
        max_length: int,
        model: PreTrainedModel,
        tagging_scheme: TaggingScheme,
        window_size: int,
        stride: int,
    ) -> Iterable[Dict[str, Any]]:
        """
        Generates chunks of tokenized input along with corresponding labels and attention masks according to the tagging scheme.

        Args:
            annotations (List[Dict]): A list of annotations containing the entity spans and their corresponding CUIs.
            tokenized (Dict[str, List]): The tokenized input containing "input_ids", "attention_mask" and "offset_mapping".
            delfault_label_id (int): The label ID to be used for background tokens.
            pad_token_id (int): The token ID used for padding the input sequences.
            pad_label_id (int): The label ID used for padding the label sequences.
            max_length (int): The maximum length of the input sequences after tokenization.
            model (PreTrainedModel): The Hugging Face token classification model.
            tagging_scheme (TaggingScheme): The tagging scheme used for the model, either "flat","iob" or "iobes".
            window_size (int): The size of the sliding window for chunking the input sequences.
            stride (int): The stride of the sliding window for chunking the input sequences.

        Yields:
            Dict[str, Any]: A dictionary containing the chunked "input_ids", "labels" and "attention_mask" for the input sequence.
        """
        if tagging_scheme == TaggingScheme.IOB:
            labels = [delfault_label_id] * len(tokenized["input_ids"])
            for annotation in annotations:
                start = annotation["start"]
                end = annotation["end"]
                cui = annotation["cui"]
                b_label = f"B-{cui}"
                i_label = f"I-{cui}"
                b_label_id = model.config.label2id.get(b_label, delfault_label_id)
                i_label_id = model.config.label2id.get(i_label, delfault_label_id)
                first_token = True
                for idx, offset_mapping in enumerate(tokenized["offset_mapping"]):
                    if offset_mapping[0] == offset_mapping[1]:
                        continue
                    if start < offset_mapping[1] and offset_mapping[0] < end:
                        if first_token:
                            labels[idx] = b_label_id
                            first_token = False
                        else:
                            labels[idx] = i_label_id

            for start in range(0, len(tokenized["input_ids"]), stride):
                end = min(start + window_size, len(tokenized["input_ids"]))
                chunked_input_ids = tokenized["input_ids"][start:end]
                chunked_labels = labels[start:end]
                chunked_attention_mask = tokenized["attention_mask"][start:end]
                padding_length = max(0, max_length - len(chunked_input_ids))
                chunked_input_ids += [pad_token_id] * padding_length
                chunked_labels += [pad_label_id] * padding_length
                chunked_attention_mask += [0] * padding_length

                yield {
                        "input_ids": chunked_input_ids,
                        "labels": chunked_labels,
                        "attention_mask": chunked_attention_mask,
                }

        elif tagging_scheme == TaggingScheme.IOBES:
            labels = [delfault_label_id] * len(tokenized["input_ids"])
            for annotation in annotations:
                start = annotation["start"]
                end = annotation["end"]
                cui = annotation["cui"]

                span_token_indices = [
                    idx for idx, offset_mapping in enumerate(tokenized["offset_mapping"])
                    if offset_mapping[0] != offset_mapping[1] and start < offset_mapping[1] and offset_mapping[0] < end
                ]
                if not span_token_indices:
                    continue
                span_token_indices = list(range(span_token_indices[0], span_token_indices[-1] + 1))

                if len(span_token_indices) == 1:
                    s_label = f"S-{cui}"
                    s_id = model.config.label2id.get(s_label, delfault_label_id)
                    labels[span_token_indices[0]] = s_id
                else:
                    b_label = f"B-{cui}"
                    i_label = f"I-{cui}"
                    e_label = f"E-{cui}"
                    b_id = model.config.label2id.get(b_label, delfault_label_id)
                    i_id = model.config.label2id.get(i_label, delfault_label_id)
                    e_id = model.config.label2id.get(e_label, delfault_label_id)

                    labels[span_token_indices[0]] = b_id
                    for mid_idx in span_token_indices[1:-1]:
                        labels[mid_idx] = i_id
                    labels[span_token_indices[-1]] = e_id

            for start in range(0, len(tokenized["input_ids"]), stride):
                end = min(start + window_size, len(tokenized["input_ids"]))
                chunked_input_ids = tokenized["input_ids"][start:end]
                chunked_labels = labels[start:end]
                chunked_attention_mask = tokenized["attention_mask"][start:end]
                padding_length = max(0, max_length - len(chunked_input_ids))
                chunked_input_ids += [pad_token_id] * padding_length
                chunked_labels += [pad_label_id] * padding_length
                chunked_attention_mask += [0] * padding_length

                yield {
                        "input_ids": chunked_input_ids,
                        "labels": chunked_labels,
                        "attention_mask": chunked_attention_mask,
                }
        else:
            for start in range(0, len(tokenized["input_ids"]), stride):
                end = min(start + window_size, len(tokenized["input_ids"]))
                chunked_input_ids = tokenized["input_ids"][start:end]
                chunked_offsets_mapping = tokenized["offset_mapping"][start:end]
                chunked_labels = [0] * len(chunked_input_ids)
                chunked_attention_mask = tokenized["attention_mask"][start:end]
                for annotation in annotations:
                    start = annotation["start"]
                    end = annotation["end"]
                    label_id = model.config.label2id.get(annotation["cui"], delfault_label_id)
                    for idx, offset_mapping in enumerate(chunked_offsets_mapping):
                        if offset_mapping[0] == offset_mapping[1]:
                            continue
                        if start < offset_mapping[1] and offset_mapping[0] < end:
                            chunked_labels[idx] = label_id
                padding_length = max(0, max_length - len(chunked_input_ids))
                chunked_input_ids += [pad_token_id] * padding_length
                chunked_labels += [pad_label_id] * padding_length
                chunked_attention_mask += [0] * padding_length

                yield {
                        "input_ids": chunked_input_ids,
                        "labels": chunked_labels,
                        "attention_mask": chunked_attention_mask,
                }

    @staticmethod
    def aggregate_bioes_predictions(
        df: pd.DataFrame,
        text: str,
        include_span_text: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Aggregates token-level predictions into entity-level predictions according to the IOB/IOBES tagging scheme.

        Args:
            df (pd.DataFrame): A DataFrame containing the token-level predictions.
            text (str): The original input text from which the tokens were derived.
            include_span_text (bool): If True, include the text of the entity span in the output. Defaults to False.
s
        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the aggregated entity-level predictions.
        """
        aggregated_entities = []
        current_entity = None
        current_label = None
        current_score = 0.0
        token_count = 0

        for _, row in df.iterrows():
            entity_tag = str(row.get("entity", "")).strip()
            score = float(row.get("score", 0.0))
            start = int(row.get("start", 0))
            end = int(row.get("end", 0))

            if entity_tag.upper() == "O" or entity_tag == "":
                if current_entity is not None:
                    aggregated_entities.append(
                        TagProcessor._get_composed_entity(
                            text,
                            current_entity,
                            current_label,
                            current_score,
                            token_count,
                            include_span_text,
                        )
                    )
                    current_entity = None
                    current_label = None
                    current_score = 0.0
                    token_count = 0
                continue

            if "-" in entity_tag:
                prefix, label = entity_tag.split("-", 1)
                prefix = prefix.upper()
            else:
                prefix = None
                label = entity_tag

            if prefix == "B":
                if current_entity is not None:
                    aggregated_entities.append(
                        TagProcessor._get_composed_entity(
                            text,
                            current_entity,
                            current_label,
                            current_score,
                            token_count,
                            include_span_text,
                        )
                    )
                current_label = label
                current_entity = {"start": start, "end": end}
                current_score = score
                token_count = 1

            elif prefix == "I":
                if current_entity is None:
                    current_label = label
                    current_entity = {"start": start, "end": end}
                    current_score = score
                    token_count = 1
                else:
                    if label == current_label:
                        current_entity["end"] = end
                        current_score += score
                        token_count += 1
                    else:
                        aggregated_entities.append(
                            TagProcessor._get_composed_entity(
                                text,
                                current_entity,
                                current_label,
                                current_score,
                                token_count,
                                include_span_text,
                            )
                        )
                        current_label = label
                        current_entity = {"start": start, "end": end}
                        current_score = score
                        token_count = 1

            elif prefix == "E":
                if current_entity is None:
                    single_ent = {"start": start, "end": end}
                    aggregated_entities.append(
                        TagProcessor._get_composed_entity(
                            text, single_ent, label, score, 1, include_span_text
                        )
                    )
                else:
                    if label == current_label:
                        current_entity["end"] = end
                        current_score += score
                        token_count += 1
                        aggregated_entities.append(
                            TagProcessor._get_composed_entity(
                                text,
                                current_entity,
                                current_label,
                                current_score,
                                token_count,
                                include_span_text,
                            )
                        )
                        current_entity = None
                        current_label = None
                        current_score = 0.0
                        token_count = 0
                    else:
                        aggregated_entities.append(
                            TagProcessor._get_composed_entity(
                                text,
                                current_entity,
                                current_label,
                                current_score,
                                token_count,
                                include_span_text,
                            )
                        )

                        # Close current entity and discard the stray E- with mismatched label
                        current_entity = None
                        current_label = None
                        current_score = 0.0
                        token_count = 0

            elif prefix == "S" or prefix is None:
                if current_entity is not None:
                    aggregated_entities.append(
                        TagProcessor._get_composed_entity(
                            text,
                            current_entity,
                            current_label,
                            current_score,
                            token_count,
                            include_span_text,
                        )
                    )
                    current_entity = None
                    current_label = None
                    current_score = 0.0
                    token_count = 0
                single_ent = {"start": start, "end": end}
                aggregated_entities.append(
                    TagProcessor._get_composed_entity(text, single_ent, label, score, 1, include_span_text)
                )

            else:
                if current_entity is not None:
                    aggregated_entities.append(
                        TagProcessor._get_composed_entity(
                            text,
                            current_entity,
                            current_label,
                            current_score,
                            token_count,
                            include_span_text,
                        )
                    )
                    current_entity = None
                    current_label = None
                    current_score = 0.0
                    token_count = 0

        if current_entity is not None:
            aggregated_entities.append(
                TagProcessor._get_composed_entity(
                    text,
                    current_entity,
                    current_label,
                    current_score,
                    token_count,
                    include_span_text,
                )
            )

        return aggregated_entities

    @staticmethod
    def _get_composed_entity(
        text: str,
        entity: Dict,
        label: Optional[str],
        score: float,
        token_count: int,
        include_span_text: bool,
    ) -> Dict[str, Any]:
        label_id, label_name = parse_label_into_id_and_name(label)
        return {
            "entity_group": label,
            "label_name": label_name,
            "label_id": label_id,
            "start": entity["start"],
            "end": entity["end"],
            "score": score / token_count,
            "accuracy": score / token_count,
            "text": text[entity["start"]:entity["end"]] if include_span_text else None
        }

    @staticmethod
    def _get_classification_head(model: PreTrainedModel) -> tuple[str, nn.Linear]:
        head_name = ""
        head_module = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.out_features == model.config.num_labels:
                head_name = name
                head_module = module

        # Fallback for test doubles/mocks that do not implement named_modules() like real HF models.
        if (not head_name or head_module is None) and hasattr(model, "classifier"):
            return "classifier", model.classifier
        if (not head_name or head_module is None) and hasattr(model, "score"):
            return "score", model.score
        if not head_name or head_module is None:
            raise AttributeError("Unable to locate the token classification head from model.named_modules().")
        return head_name, head_module

    @staticmethod
    def _set_module_by_name(model: PreTrainedModel, module_name: str, module: nn.Module) -> None:
        if "." in module_name:
            parent_name, child_name = module_name.rsplit(".", 1)
            parent_module = model.get_submodule(parent_name)
            setattr(parent_module, child_name, module)
        else:
            setattr(model, module_name, module)

    @staticmethod
    def _append_label_to_head(
            model: PreTrainedModel,
            head_name: str,
            head: nn.Linear,
            label: str,
            avg_weight: Any,
            avg_bias: Any,
    ) -> nn.Linear:
        if label in model.config.label2id.keys():
            return head
        model.config.label2id[label] = len(model.config.label2id)
        model.config.id2label[len(model.config.id2label)] = label
        head.weight = nn.Parameter(cat((head.weight, avg_weight), 0))
        head.bias = nn.Parameter(cat((head.bias, avg_bias), 0))
        if hasattr(head, "out_features"):
            head.out_features += 1
        model.num_labels += 1
        TagProcessor._set_module_by_name(model, head_name, head)
        return head
