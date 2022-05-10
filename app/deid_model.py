import os
import shutil
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from medcat.tokenizers.tokenizer_ner import TokenizerNER
from model_services import ModelServices


class DeIdModel(ModelServices):

    def __init__(self, config):
        self.config = config
        model_file_path = os.path.join(os.path.dirname(__file__), "model", config.base_model_file)
        self.tokenizer, self.model = self.load_model(model_file_path)

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
        doc = self.model.get_entities(text)
        return self._get_records_from_doc(doc)
    
    def _get_records_from_doc(self, doc):
        ...
