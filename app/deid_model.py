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
        return self._get_annotation_list([text])[0]
    
    def batch_annotate(self, texts):
        return self._get_annotation_list(texts)

    def _get_annotation_list(self, texts):
        dataset = []
        for text in texts:
            tokens = self.tokenizer.hf_tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
            dataset.append(
                {
                    "input_ids": tokens["input_ids"],
                    "offset_mapping": tokens["offset_mapping"],
                    "labels": [0] * len(tokens["input_ids"])
                }
            )
        prediction_output = self.trainer.predict(dataset)
        predictions = np.array(prediction_output.predictions)
        predictions = softmax(predictions, axis=2)
        preds = np.argmax(predictions, axis=2)

        annotations_list = []
        for p_idx, predication in enumerate(predictions):
            annotations = []
            allow_expansion = False
            for t_idx, _ in enumerate(predication):
                offset_mapping = dataset[p_idx]["offset_mapping"]
                input_ids = dataset[p_idx]["input_ids"]

                if preds[p_idx][t_idx] != -100:
                    annotation = {
                        "label_name": self.tokenizer.cui2name.get(self.id2cui[preds[p_idx][t_idx]]),
                        "label_id": self.id2cui[preds[p_idx][t_idx]],
                        "start": offset_mapping[t_idx][0],
                        "end": offset_mapping[t_idx][1]
                    }
                    if self._should_expand_start(annotations, preds[p_idx][t_idx-1],
                                                 self.tokenizer.id2type.get(input_ids[t_idx-1]),
                                                 self.tokenizer.id2type.get(input_ids[t_idx])):
                        allow_expansion = True
                        annotations[-1]["end"] = annotation["end"]
                        continue
                    elif self._should_expand_middle(allow_expansion, self.tokenizer.id2type.get(input_ids[t_idx])):
                        annotations[-1]["end"] = annotation["end"]
                        continue
                    else:
                        allow_expansion = False
                    if preds[p_idx][t_idx] not in [0, 1]:
                        if (annotations and
                            annotation["label_id"] == annotations[-1]["label_id"] and
                            annotation["start"] == annotations[-1]["end"] + 1):
                                annotations[-1]["end"] = annotation["end"]
                        else:
                            annotations.append(annotation)
            annotations_list.append(annotations)
        return annotations_list

    @staticmethod
    def _should_expand_start(annotations, cur_label, last_token_type, cur_token_type):
        return all([annotations, cur_label not in [0, 1], last_token_type == "start", cur_token_type == "sub"])

    @staticmethod
    def _should_expand_middle(allow_expansion, cur_token_type):
        return allow_expansion and cur_token_type == "sub"
