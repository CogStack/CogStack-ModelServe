import os
import shutil
import numpy as np
from scipy.special import softmax
from transformers import AutoModelForTokenClassification, Trainer
from medcat.tokenizers.tokenizer_ner import TokenizerNER
from model_services import ModelServices


class DeIdModel(ModelServices):

    def __init__(self, config):
        self.config = config
        model_file_path = os.path.join(os.path.dirname(__file__), "model", config.base_model_file)
        self.tokenizer, self.model = self.load_model(model_file_path)
        self.trainer = Trainer(model=self.model, tokenizer=None)
        self.id2cui = {id:cui for cui, id in self.tokenizer.label_map.items()}

    @staticmethod
    def info():
        return {"model_description": "de-id model", "model_type": "bert"}
    
    @staticmethod
    def load_model(model_file_path, *args, **kwargs):
        model_file_dir = os.path.dirname(model_file_path)
        model_file_name = os.path.basename(model_file_path).replace(".zip", "")
        unpacked_model_dir = os.path.join(model_file_dir, model_file_name)
        if not os.path.isdir(unpacked_model_dir):
            shutil.unpack_archive(model_file_path, extract_dir=unpacked_model_dir)
        tokenizer_path = os.path.join(unpacked_model_dir, "tokenizer.dat")
        tokenizer = TokenizerNER.load(tokenizer_path)
        model = AutoModelForTokenClassification.from_pretrained(unpacked_model_dir)
        return tokenizer, model

    def annotate(self, text):
        return self._get_annotations(text)
    
    def batch_annotate(self, texts):
        annotation_list = []
        for text in texts:
            annotation_list.append(self._get_annotations(text))
        return annotation_list

    def _get_annotations(self, text):
        tokens = self.tokenizer.hf_tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        dataset, offset_mappings = self._chunck(tokens, pad_token_id=1, model_max_length=self.tokenizer.max_len)
        prediction_output = self.trainer.predict(dataset)
        predictions = np.array(prediction_output.predictions)
        predictions = softmax(predictions, axis=2)
        batched_cui_ids = np.argmax(predictions, axis=2)
        annotations = []

        for ps_idx, cui_ids in enumerate(batched_cui_ids):
            input_ids = dataset[ps_idx]["input_ids"]
            for t_idx, cur_cui_id in enumerate(cui_ids):
                if cur_cui_id not in [0, -100]:
                    annotation = {
                        "label_name": self.tokenizer.cui2name.get(self.id2cui[cur_cui_id]),
                        "label_id": self.id2cui[cur_cui_id],
                        "start": offset_mappings[ps_idx][t_idx][0],
                        "end": offset_mappings[ps_idx][t_idx][1],
                    }
                    if annotations:
                        token_type = self.tokenizer.id2type.get(input_ids[t_idx])
                        if (self._should_expand_with_partial(cur_cui_id, token_type, annotation, annotations) or
                            self._should_expand_with_whole(annotation, annotations)):
                            annotations[-1]["end"] = annotation["end"]
                            del annotation
                            continue
                        elif cur_cui_id != 1:
                            annotations.append(annotation)
                            continue
                    else:
                        if cur_cui_id != 1:
                            annotations.append(annotation)
                            continue
        return annotations

    def _chunck(self, tokens, pad_token_id, model_max_length):
        dataset = []
        offset_mappings = []
        for i in range(0, len(tokens["input_ids"]), model_max_length):
            dataset.append({
                "input_ids": tokens["input_ids"][i:i+model_max_length],
                "attention_mask": [1] * model_max_length,
            })
            offset_mappings.append(tokens["offset_mapping"][i:i+model_max_length])
        remainder = len(tokens["input_ids"]) % model_max_length
        if remainder and i >= model_max_length:
            del dataset[-1]
            del offset_mappings[-1]
            dataset.append({
                "input_ids": tokens["input_ids"][-remainder:] + [pad_token_id]*(model_max_length-remainder),
                "attention_mask": [1]*remainder + [1]*(model_max_length-remainder),
            })
            offset_mappings.append(tokens["offset_mapping"][-remainder:] +
                [(tokens["offset_mapping"][-1][1]+i, tokens["offset_mapping"][-1][1]+i+1) for i in range(model_max_length-remainder)])
        return dataset, offset_mappings

    @staticmethod
    def _should_expand_with_partial(cur_cui_id, cur_token_type, annotation, annotations):
        return all([cur_cui_id == 1, cur_token_type == "sub", (annotation["start"] - annotations[-1]["end"]) in [0, 1]])

    @staticmethod
    def _should_expand_with_whole(annotation, annotations):
        return annotation["label_id"] == annotations[-1]["label_id"] and (annotation["start"] - annotations[-1]["end"]) in [0, 1]
