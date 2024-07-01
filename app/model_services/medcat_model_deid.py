import logging
import inspect
import threading
import torch
from typing import Dict, List, TextIO, Optional, Any, final, Callable
from functools import partial
from transformers import pipeline
from medcat.cat import CAT
from config import Settings
from model_services.medcat_model import MedCATModel
from trainers.medcat_deid_trainer import MedcatDeIdentificationSupervisedTrainer
from domain import ModelCard, ModelType
from exception import ConfigurationException

logger = logging.getLogger("cms")


@final
class MedCATModelDeIdentification(MedCATModel):

    CHUNK_SIZE = 500
    LEFT_CONTEXT_WORDS = 5

    def __init__(self,
                 config: Settings,
                 model_parent_dir: Optional[str] = None,
                 enable_trainer: Optional[bool] = None,
                 model_name: Optional[str] = None,
                 base_model_file: Optional[str] = None) -> None:
        super().__init__(config, model_parent_dir=model_parent_dir, enable_trainer=enable_trainer, model_name=model_name, base_model_file=base_model_file)
        self.model_name = model_name or "De-Identification MedCAT model"
        self._lock = threading.RLock()

    @property
    def api_version(self) -> str:
        return "0.0.1"

    def info(self) -> ModelCard:
        model_card = self.model.get_model_card(as_dict=True)
        model_card["Basic CDB Stats"]["Average training examples per concept"] = 0
        return ModelCard(model_description=self.model_name,
                         model_type=ModelType.MEDCAT_DEID,
                         api_version=self.api_version,
                         model_card=model_card)

    def annotate(self, text: str) -> Dict:
        tokenizer = self.model._addl_ner[0].tokenizer.hf_tokenizer
        leading_ws_len = len(text) - len(text.lstrip())
        text = text.lstrip()
        tokenized = self._with_lock(tokenizer, text, return_offsets_mapping=True, add_special_tokens=False)
        input_ids = tokenized["input_ids"]
        offset_mapping = tokenized["offset_mapping"]
        chunk = []
        aggregated_entities = {}
        ent_key = 0
        processed_char_len = leading_ws_len

        for input_id, (start, end) in zip(input_ids, offset_mapping):
            chunk.append((input_id, (start, end)))
            if len(chunk) == MedCATModelDeIdentification.CHUNK_SIZE:
                last_token_start_idx = 0
                window_overlap_start_idx = 0
                number_of_seen_words = 0
                for i in range(MedCATModelDeIdentification.CHUNK_SIZE-1, -1, -1):
                    if " " in tokenizer.decode([chunk[i][0]], skip_special_tokens=True):
                        if last_token_start_idx == 0:
                            last_token_start_idx = i
                        if number_of_seen_words < MedCATModelDeIdentification.LEFT_CONTEXT_WORDS:
                            window_overlap_start_idx = i
                        else:
                            break
                        number_of_seen_words += 1
                c_text = text[chunk[:last_token_start_idx][0][1][0]:chunk[:last_token_start_idx][-1][1][1]]
                doc = self._with_lock(self.model.get_entities, c_text)
                doc["entities"] = {_id: entity for _id, entity in doc["entities"].items() if entity["end"]+processed_char_len < chunk[window_overlap_start_idx][1][0]}
                for entity in doc["entities"].values():
                    entity["start"] += processed_char_len
                    entity["end"] += processed_char_len
                    entity["types"] = ["PII"]
                    aggregated_entities[ent_key] = entity
                    ent_key += 1
                processed_char_len = chunk[:window_overlap_start_idx][-1][1][1] + leading_ws_len + 1
                chunk = chunk[window_overlap_start_idx:]
        if chunk:
            c_text = text[chunk[0][1][0]:chunk[-1][1][1]]
            doc = self.model.get_entities(c_text)
            if doc["entities"]:
                for entity in doc["entities"].values():
                    entity["start"] += processed_char_len
                    entity["end"] += processed_char_len
                    entity["types"] = ["PII"]
                    aggregated_entities[ent_key] = entity
                    ent_key += 1
            processed_char_len += len(c_text)

        assert processed_char_len == (len(text)+leading_ws_len), f"{len(text)+leading_ws_len-processed_char_len} characters were not processed:\n{text}"

        return self.get_records_from_doc({"entities": aggregated_entities})

    def batch_annotate(self, texts: List[str]) -> List[Dict]:
        annotation_list = []
        for text in texts:
            annotation_list.append(self.annotate(text))
        return annotation_list

    def init_model(self) -> None:
        if hasattr(self, "_model") and isinstance(self._model, CAT):    # type: ignore
            logger.warning("Model service is already initialised and can be initialised only once")
        else:
            self._model = self.load_model(self._model_pack_path)
            self._model._addl_ner[0].tokenizer.hf_tokenizer._in_target_context_manager = getattr(self._model._addl_ner[0].tokenizer.hf_tokenizer, "_in_target_context_manager", False)
            self._model._addl_ner[0].tokenizer.hf_tokenizer.clean_up_tokenization_spaces = getattr(self._model._addl_ner[0].tokenizer.hf_tokenizer, "clean_up_tokenization_spaces", None)
            if (self._config.DEVICE.startswith("cuda") and torch.cuda.is_available()) or \
               (self._config.DEVICE.startswith("mps") and torch.backends.mps.is_available()) or \
               (self._config.DEVICE.startswith("cpu")):
                self._model.config.general["device"] = self._config.DEVICE
                self._model._addl_ner[0].model.to(torch.device(self._config.DEVICE))
                if self._config.DEVICE.startswith("cuda"):
                    device = 0 if len(self._config.DEVICE.split(":")) == 1 else self._config.DEVICE.split(":")[1]
                elif self._config.DEVICE.startswith("mps"):
                    device = "mps"
                else:
                    device = -1
                self._model._addl_ner[0].ner_pipe = pipeline(model=self._model._addl_ner[0].model,
                                                             framework="pt",
                                                             task="ner",
                                                             tokenizer=self._model._addl_ner[0].tokenizer.hf_tokenizer,
                                                             device=device)
            else:
                if self._config.DEVICE != "default":
                    logger.warning(
                        f"DEVICE is set to '{self._config.DEVICE}' but it is not available. Using 'default' instead.")
            _save_pretrained = self._model._addl_ner[0].model.save_pretrained
            if ("safe_serialization" in inspect.signature(_save_pretrained).parameters):
                self._model._addl_ner[0].model.save_pretrained = partial(_save_pretrained, safe_serialization=(self._config.TRAINING_SAFE_MODEL_SERIALISATION == "true"))
            if self._enable_trainer:
                self._supervised_trainer = MedcatDeIdentificationSupervisedTrainer(self)

    def train_supervised(self,
                         data_file: TextIO,
                         epochs: int,
                         log_frequency: int,
                         training_id: str,
                         input_file_name: str,
                         raw_data_files: Optional[List[TextIO]] = None,
                         description: Optional[str] = None,
                         synchronised: bool = False,
                         **hyperparams: Dict[str, Any]) -> bool:
        if self._supervised_trainer is None:
            raise ConfigurationException("Trainers are not enabled")
        return self._supervised_trainer.train(data_file, epochs, log_frequency, training_id, input_file_name, raw_data_files, description, synchronised, **hyperparams)

    def _with_lock(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        # Temporarily tackle https://github.com/huggingface/tokenizers/issues/537 but it reduces parallelism
        with self._lock:
            return func(*args, **kwargs)
