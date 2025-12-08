import pandas as pd
from typing import Any, Dict, List, Optional, Iterable
from torch import nn, mean, cat
from transformers import PreTrainedModel
from app.domain import TaggingScheme


class TagProcessor:

    @staticmethod
    def update_model_by_tagging_scheme(
        model: PreTrainedModel,
        concepts: List[str],
        tagging_scheme: TaggingScheme,
    ) -> PreTrainedModel:
        avg_weight = mean(model.classifier.weight, dim=0, keepdim=True)
        avg_bias = mean(model.classifier.bias, dim=0, keepdim=True)
        if tagging_scheme == TaggingScheme.IOB:
            for concept in concepts:
                b_label = f"B-{concept}"
                i_label = f"I-{concept}"
                if b_label not in model.config.label2id.keys():
                    model.config.label2id[b_label] = len(model.config.label2id)
                    model.config.id2label[len(model.config.id2label)] = b_label
                    model.classifier.weight = nn.Parameter(cat((model.classifier.weight, avg_weight), 0))
                    model.classifier.bias = nn.Parameter(cat((model.classifier.bias, avg_bias), 0))
                    model.classifier.out_features += 1
                    model.num_labels += 1
                if i_label not in model.config.label2id.keys():
                    model.config.label2id[i_label] = len(model.config.label2id)
                    model.config.id2label[len(model.config.id2label)] = i_label
                    model.classifier.weight = nn.Parameter(cat((model.classifier.weight, avg_weight), 0))
                    model.classifier.bias = nn.Parameter(cat((model.classifier.bias, avg_bias), 0))
                    model.classifier.out_features += 1
                    model.num_labels += 1
        elif tagging_scheme == TaggingScheme.IOBES:
            for concept in concepts:
                s_label = f"S-{concept}"
                b_label = f"B-{concept}"
                i_label = f"I-{concept}"
                e_label = f"E-{concept}"
                if s_label not in model.config.label2id.keys():
                    model.config.label2id[s_label] = len(model.config.label2id)
                    model.config.id2label[len(model.config.id2label)] = s_label
                    model.classifier.weight = nn.Parameter(cat((model.classifier.weight, avg_weight), 0))
                    model.classifier.bias = nn.Parameter(cat((model.classifier.bias, avg_bias), 0))
                    model.classifier.out_features += 1
                    model.num_labels += 1
                if b_label not in model.config.label2id.keys():
                    model.config.label2id[b_label] = len(model.config.label2id)
                    model.config.id2label[len(model.config.id2label)] = b_label
                    model.classifier.weight = nn.Parameter(cat((model.classifier.weight, avg_weight), 0))
                    model.classifier.bias = nn.Parameter(cat((model.classifier.bias, avg_bias), 0))
                    model.classifier.out_features += 1
                    model.num_labels += 1
                if i_label not in model.config.label2id.keys():
                    model.config.label2id[i_label] = len(model.config.label2id)
                    model.config.id2label[len(model.config.id2label)] = i_label
                    model.classifier.weight = nn.Parameter(cat((model.classifier.weight, avg_weight), 0))
                    model.classifier.bias = nn.Parameter(cat((model.classifier.bias, avg_bias), 0))
                    model.classifier.out_features += 1
                    model.num_labels += 1
                if e_label not in model.config.label2id.keys():
                    model.config.label2id[e_label] = len(model.config.label2id)
                    model.config.id2label[len(model.config.id2label)] = e_label
                    model.classifier.weight = nn.Parameter(cat((model.classifier.weight, avg_weight), 0))
                    model.classifier.bias = nn.Parameter(cat((model.classifier.bias, avg_bias), 0))
                    model.classifier.out_features += 1
                    model.num_labels += 1
        else:
            for concept in concepts:
                if concept not in model.config.label2id.keys():
                    model.config.label2id[concept] = len(model.config.label2id)
                    model.config.id2label[len(model.config.id2label)] = concept
                    model.classifier.weight = nn.Parameter(cat((model.classifier.weight, avg_weight), 0))
                    model.classifier.bias = nn.Parameter(cat((model.classifier.bias, avg_bias), 0))
                    model.classifier.out_features += 1
                    model.num_labels += 1
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
                    if start <= offset_mapping[0] and offset_mapping[1] <= end:
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
                ann_start = annotation["start"]
                ann_end = annotation["end"]
                cui = annotation["cui"]

                covered_indices = [
                    idx for idx, off in enumerate(tokenized["offset_mapping"])
                    if ann_start <= off[0] and off[1] <= ann_end
                ]
                if not covered_indices:
                    continue

                if len(covered_indices) == 1:
                    s_label = f"S-{cui}"
                    s_id = model.config.label2id.get(s_label, delfault_label_id)
                    labels[covered_indices[0]] = s_id
                else:
                    b_label = f"B-{cui}"
                    i_label = f"I-{cui}"
                    e_label = f"E-{cui}"
                    b_id = model.config.label2id.get(b_label, delfault_label_id)
                    i_id = model.config.label2id.get(i_label, delfault_label_id)
                    e_id = model.config.label2id.get(e_label, delfault_label_id)

                    labels[covered_indices[0]] = b_id
                    for mid_idx in covered_indices[1:-1]:
                        labels[mid_idx] = i_id
                    labels[covered_indices[-1]] = e_id

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
                    annotation_start = annotation["start"]
                    annotation_end = annotation["end"]
                    label_id = model.config.label2id.get(annotation["cui"], delfault_label_id)
                    for idx, offset_mapping in enumerate(chunked_offsets_mapping):
                        if annotation_start <= offset_mapping[0] and offset_mapping[1] <= annotation_end:
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
                        TagProcessor._get_composed_entitiy(
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
                        TagProcessor._get_composed_entitiy(
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
                            TagProcessor._get_composed_entitiy(
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
                        TagProcessor._get_composed_entitiy(text, single_ent, label, score, 1, include_span_text)
                    )
                else:
                    if label == current_label:
                        current_entity["end"] = end
                        current_score += score
                        token_count += 1
                        aggregated_entities.append(
                            TagProcessor._get_composed_entitiy(
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
                            TagProcessor._get_composed_entitiy(
                                text,
                                current_entity,
                                current_label,
                                current_score,
                                token_count,
                                include_span_text,
                            )
                        )
                        single_ent = {"start": start, "end": end}
                        aggregated_entities.append(
                            TagProcessor._get_composed_entitiy(text, single_ent, label, score, 1, include_span_text)
                        )
                        current_entity = None
                        current_label = None
                        current_score = 0.0
                        token_count = 0

            elif prefix == "S" or prefix is None:
                if current_entity is not None:
                    aggregated_entities.append(
                        TagProcessor._get_composed_entitiy(
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
                    TagProcessor._get_composed_entitiy(text, single_ent, label, score, 1, include_span_text)
                )

            else:
                if current_entity is not None:
                    aggregated_entities.append(
                        TagProcessor._get_composed_entitiy(
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
                TagProcessor._get_composed_entitiy(
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
    def _get_composed_entitiy(
        text: str,
        entity: Dict,
        label: Optional[str],
        score: float,
        token_count: int,
        include_span_text: bool,
    ) -> Dict[str, Any]:
        return {
            "entity_group": label,
            "label_name": label,
            "label_id": label,
            "start": entity["start"],
            "end": entity["end"],
            "score": score / token_count,
            "accuracy": score / token_count,
            "text": text[entity["start"]:entity["end"]] if include_span_text else None
        }
