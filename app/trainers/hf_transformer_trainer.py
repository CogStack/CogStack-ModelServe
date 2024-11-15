import os
import logging
import torch
import gc
import json
import shutil
import datasets
import random
import tempfile
import numpy as np
from typing import final, Dict, TextIO, Optional, Any, List, Iterable, Tuple, Union
from torch import nn
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from scipy.special import softmax
from transformers import __version__ as transformers_version
from transformers import (
    BertForMaskedLM,
    BertForTokenClassification,
    RobertaForMaskedLM,
    RobertaForTokenClassification,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    EvalPrediction,
)
from management.model_manager import ModelManager
from management.tracker_client import TrackerClient
from model_services.base import AbstractModelService
from processors.metrics_collector import get_stats_from_trainer_export
from utils import filter_by_concept_ids, reset_random_seed
from trainers.base import UnsupervisedTrainer, SupervisedTrainer
from domain import ModelType, DatasetSplit


logger = logging.getLogger("cms")


@final
class HFTransformerUnsupervisedTrainer(UnsupervisedTrainer):

    def __init__(self, model_service: AbstractModelService) -> None:
        UnsupervisedTrainer.__init__(self, model_service._config, model_service.model_name)
        self._model_service = model_service
        self._model_name = model_service.model_name
        self._model_pack_path = model_service._model_pack_path
        self._retrained_models_dir = os.path.join(model_service._model_parent_dir, "retrained",
                                                  self._model_name.replace(" ", "_"))
        self._model_manager = ModelManager(type(model_service), model_service._config)
        self._max_length = model_service.model.config.max_position_embeddings
        os.makedirs(self._retrained_models_dir, exist_ok=True)


    @staticmethod
    def run(trainer: "HFTransformerUnsupervisedTrainer",
            training_params: Dict,
            data_file: Union[TextIO, tempfile.TemporaryDirectory],
            log_frequency: int,
            run_id: str,
            description: Optional[str] = None) -> None:
        copied_model_pack_path = None
        train_dataset = None
        eval_dataset = None
        redeploy = trainer._config.REDEPLOY_TRAINED_MODEL == "true"
        skip_save_model = trainer._config.SKIP_SAVE_MODEL == "true"
        results_path = os.path.abspath(os.path.join(trainer._config.TRAINING_CACHE_DIR, "results"))
        logs_path = os.path.abspath(os.path.join(trainer._config.TRAINING_CACHE_DIR, "logs"))
        reset_random_seed()
        try:
            logger.info("Loading a new model copy for training...")
            copied_model_pack_path = trainer._make_model_file_copy(trainer._model_pack_path, run_id)
            model, tokenizer = trainer._model_service.load_model(copied_model_pack_path)
            copied_model_directory = copied_model_pack_path.replace(".zip", "")
            mlm_model = trainer._get_mlm_model(model, copied_model_directory)

            if (trainer._config.DEVICE.startswith("cuda") and torch.cuda.is_available()) or \
                    (trainer._config.DEVICE.startswith("mps") and torch.backends.mps.is_available()) or \
                    (trainer._config.DEVICE.startswith("cpu")):
                mlm_model.to(trainer._config.DEVICE)
            test_size = 0.2 if training_params.get("test_size") is None else training_params["test_size"]
            if isinstance(data_file, tempfile.TemporaryDirectory):
                raw_dataset = datasets.load_from_disk(data_file.name)
                if DatasetSplit.VALIDATION.value in raw_dataset.keys():
                    train_texts = raw_dataset[DatasetSplit.TRAIN.value]["text"]
                    eval_texts = raw_dataset[DatasetSplit.VALIDATION.value]["text"]
                elif DatasetSplit.TEST.value in raw_dataset.keys():
                    train_texts = raw_dataset[DatasetSplit.TRAIN.value]["text"]
                    eval_texts = raw_dataset[DatasetSplit.TEST.value]["text"]
                else:
                    lines = raw_dataset[DatasetSplit.TRAIN.value]["text"]
                    random.shuffle(lines)
                    train_texts = [line.strip() for line in lines[:int(len(lines) * (1 - test_size))]]
                    eval_texts = [line.strip() for line in lines[int(len(lines) * (1 - test_size)):]]
            else:
                with open(data_file.name, "r") as f:
                    lines = json.load(f)
                    random.shuffle(lines)
                    train_texts = [line.strip() for line in lines[:int(len(lines) * (1-test_size))]]
                    eval_texts = [line.strip() for line in lines[int(len(lines) * (1-test_size)):]]

            dataset_features = datasets.Features({
                "input_ids": datasets.Sequence(datasets.Value("int32")),
                "attention_mask": datasets.Sequence(datasets.Value("int32")),
                "special_tokens_mask": datasets.Sequence(datasets.Value("int32")),
                "token_type_ids": datasets.Sequence(datasets.Value("int32"))
            })
            train_dataset = datasets.Dataset.from_generator(
                trainer._tokenize_and_chunk,
                features=dataset_features,
                gen_kwargs={"texts": train_texts, "tokenizer": tokenizer, "max_length": trainer._max_length},
                cache_dir=trainer._model_service._config.TRAINING_CACHE_DIR
            )
            eval_dataset = datasets.Dataset.from_generator(
                trainer._tokenize_and_chunk,
                features=dataset_features,
                gen_kwargs={"texts": eval_texts, "tokenizer": tokenizer, "max_length": trainer._max_length},
                cache_dir = trainer._model_service._config.TRAINING_CACHE_DIR
            )
            train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
            eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.2)

            training_args = TrainingArguments(
                output_dir=results_path,
                logging_dir=logs_path,
                eval_strategy="epoch",
                save_strategy="epoch",
                overwrite_output_dir=True,
                num_train_epochs=training_params["nepochs"],
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                gradient_accumulation_steps=2,
                logging_steps=log_frequency,
                save_steps=1000,
                load_best_model_at_end=True,
                save_total_limit=3,
            )

            if training_params.get("lr_override") is not None:
                training_args.learning_rate = training_params["lr_override"]

            hf_trainer = Trainer(
                model=mlm_model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                callbacks=[MLflowLoggingCallback(trainer._tracker_client)]
            )

            trainer._tracker_client.log_trainer_version(transformers_version)
            logger.info("Performing unsupervised training...")
            hf_trainer.train()

            model = trainer._get_final_model(model, mlm_model)
            if not skip_save_model:
                retrained_model_pack_path = os.path.join(trainer._retrained_models_dir, f"{ModelType.HF_TRANSFORMER.value}_{run_id}.zip")
                model.save_pretrained(copied_model_directory, safe_serialization=(trainer._config.TRAINING_SAFE_MODEL_SERIALISATION == "true"))
                shutil.make_archive(retrained_model_pack_path.replace(".zip", ""), "zip", copied_model_directory)
                model_uri = trainer._tracker_client.save_model(retrained_model_pack_path, trainer._model_name, trainer._model_manager)
                logger.info(f"Retrained model saved: {model_uri}")
            else:
                logger.info("Skipped saving on the retrained model")
            if redeploy:
                trainer.deploy_model(trainer._model_service, model, tokenizer)
            else:
                del model
                del mlm_model
                del tokenizer
                gc.collect()
                logger.info("Skipped deployment on the retrained model")
            logger.info("Unsupervised training finished")
            trainer._tracker_client.end_with_success()
        except Exception as e:
            logger.error("Unsupervised training failed")
            logger.exception(e)
            trainer._tracker_client.log_exceptions(e)
            trainer._tracker_client.end_with_failure()
        finally:
            if isinstance(data_file, TextIO):
                data_file.close()
            elif isinstance(data_file, tempfile.TemporaryDirectory):
                data_file.cleanup()
            if train_dataset is not None:
                train_dataset.cleanup_cache_files()
            if eval_dataset is not None:
                eval_dataset.cleanup_cache_files()
            with trainer._training_lock:
                trainer._training_in_progress = False
            _clean_up_training_cache(trainer._config.TRAINING_CACHE_DIR)
            trainer._housekeep_file(copied_model_pack_path)

    @staticmethod
    def deploy_model(model_service: AbstractModelService,
                     model: PreTrainedModel,
                     tokenizer: PreTrainedTokenizerBase) -> None:
        del model_service.model
        del model_service.tokenizer
        gc.collect()
        model_service.model = model
        model_service.tokenizer = tokenizer
        logger.info("Retrained model deployed")

    @staticmethod
    def _get_mlm_model(model: PreTrainedModel, copied_model_directory: str) -> PreTrainedModel:
        if isinstance(model, BertForMaskedLM) or isinstance(model, BertForTokenClassification):
            mlm_model = BertForMaskedLM.from_pretrained(copied_model_directory)
            mlm_model.bert = model.bert
        elif isinstance(model, RobertaForMaskedLM) or isinstance(model, RobertaForTokenClassification):
            mlm_model = RobertaForMaskedLM.from_pretrained(copied_model_directory)
            mlm_model.roberta = model.roberta
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")
        return mlm_model

    @staticmethod
    def _get_final_model(model: PreTrainedModel, mlm_model: PreTrainedModel) -> PreTrainedModel:
        if isinstance(model, BertForMaskedLM) or isinstance(model, RobertaForMaskedLM):
            model = mlm_model
        elif isinstance(model, BertForTokenClassification):
            model.bert = mlm_model.bert
        elif isinstance(model, RobertaForTokenClassification):
            model.roberta = mlm_model.roberta
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")
        return model

    @staticmethod
    def _tokenize_and_chunk(texts: Iterable[str], tokenizer: PreTrainedTokenizerBase, max_length: int) -> Iterable[Dict[str, Any]]:
        for text in texts:
            encoded = tokenizer(text, truncation=False, return_special_tokens_mask=True)

            for i in range(0, len(encoded["input_ids"]), max_length):
                chunked_input_ids = encoded["input_ids"][i:i + max_length]
                padding_length = max(0, max_length - len(chunked_input_ids))

                chunked_input_ids += [tokenizer.pad_token_id] * padding_length
                chunked_attention_mask = encoded["attention_mask"][i:i + max_length] + [0] * padding_length
                chunked_special_tokens = tokenizer.get_special_tokens_mask(chunked_input_ids,
                                                                           already_has_special_tokens=True)
                token_type_ids = [0] * len(chunked_input_ids)

                yield {
                    "input_ids": chunked_input_ids,
                    "attention_mask": chunked_attention_mask,
                    "special_tokens_mask": chunked_special_tokens,
                    "token_type_ids": token_type_ids,
                }


@final
class HFTransformerSupervisedTrainer(SupervisedTrainer):

    MIN_EXAMPLE_COUNT_FOR_TRAINABLE_CONCEPT = 5
    PAD_LABEL_ID = -100
    DEFAULT_LABEL_ID = 0

    def __init__(self, model_service: AbstractModelService) -> None:
        if not isinstance(model_service.tokenizer, PreTrainedTokenizerFast):
            logger.error("The supervised trainer requires a fast tokenizer to function correctly")
        SupervisedTrainer.__init__(self, model_service._config, model_service.model_name)
        self._model_service = model_service
        self._model_name = model_service.model_name
        self._model_pack_path = model_service._model_pack_path
        self._retrained_models_dir = os.path.join(model_service._model_parent_dir, "retrained",
                                                  self._model_name.replace(" ", "_"))
        self._model_manager = ModelManager(type(model_service), model_service._config)
        self._max_length = model_service.model.config.max_position_embeddings
        os.makedirs(self._retrained_models_dir, exist_ok=True)

    class _LocalDataCollator:

        def __init__(self, max_length: int, pad_token_id: int) -> None:
            self.max_length = max_length
            self.pad_token_id = pad_token_id

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            return {
                "input_ids": torch.tensor([self._add_padding(f["input_ids"], self.max_length, self.pad_token_id) for f in features], dtype=torch.long),
                "labels": torch.tensor([self._add_padding(f["labels"], self.max_length, HFTransformerSupervisedTrainer.PAD_LABEL_ID) for f in features], dtype=torch.long),
                "attention_mask": torch.tensor([self._add_padding(f["attention_mask"], self.max_length, 0) for f in features], dtype=torch.long),
            }

        @staticmethod
        def _add_padding(target: List[int], max_length: int, pad_token_id: int) -> List[int]:
            padding_length = max(0, max_length - len(target))
            paddings = [pad_token_id] * padding_length
            return target + paddings

    @staticmethod
    def run(trainer: "HFTransformerSupervisedTrainer",
            training_params: Dict,
            data_file: TextIO,
            log_frequency: int,
            run_id: str,
            description: Optional[str] = None) -> None:
        copied_model_pack_path = None
        redeploy = trainer._config.REDEPLOY_TRAINED_MODEL == "true"
        skip_save_model = trainer._config.SKIP_SAVE_MODEL == "true"
        results_path = os.path.abspath(os.path.join(trainer._config.TRAINING_CACHE_DIR, "results"))
        logs_path = os.path.abspath(os.path.join(trainer._config.TRAINING_CACHE_DIR, "logs"))
        reset_random_seed()
        try:
            logger.info("Loading a new model copy for training...")
            copied_model_pack_path = trainer._make_model_file_copy(trainer._model_pack_path, run_id)
            model, tokenizer = trainer._model_service.load_model(copied_model_pack_path)
            copied_model_directory = copied_model_pack_path.replace(".zip", "")

            if (trainer._config.DEVICE.startswith("cuda") and torch.cuda.is_available()) or \
                    (trainer._config.DEVICE.startswith("mps") and torch.backends.mps.is_available()) or \
                    (trainer._config.DEVICE.startswith("cpu")):
                model.to(trainer._config.DEVICE)

            filtered_training_data, filtered_concepts = trainer._filter_training_data_and_concepts(data_file)
            logger.debug(f"Filtered concepts: {filtered_concepts}")
            model = trainer._update_model_with_concepts(model, filtered_concepts)

            documents = [document for project in filtered_training_data["projects"] for document in project["documents"]]
            random.shuffle(documents)
            test_size = 0.2 if training_params.get("test_size") is None else training_params["test_size"]
            train_documents = [document for document in documents[:int(len(documents) * (1 - test_size))]]
            eval_documents = [document for document in documents[int(len(documents) * (1 - test_size)):]]

            dataset_features = datasets.Features({
                "input_ids": datasets.Sequence(datasets.Value("int32")),
                "labels": datasets.Sequence(datasets.Value("int32")),
                "attention_mask": datasets.Sequence(datasets.Value("int32")),
            })
            train_dataset = datasets.Dataset.from_generator(
                trainer._tokenize_and_chunk,
                features=dataset_features,
                gen_kwargs={"documents": train_documents, "tokenizer": tokenizer, "max_length": trainer._max_length, "model": model}
            )
            eval_dataset = datasets.Dataset.from_generator(
                trainer._tokenize_and_chunk,
                features=dataset_features,
                gen_kwargs={"documents": eval_documents, "tokenizer": tokenizer, "max_length": trainer._max_length, "model": model}
            )
            train_dataset.set_format(type=None, columns=["input_ids", "labels", "attention_mask"])
            eval_dataset.set_format(type=None, columns=["input_ids", "labels", "attention_mask"])

            data_collator = trainer._LocalDataCollator(max_length=trainer._max_length, pad_token_id=tokenizer.pad_token_id)

            training_args = TrainingArguments(
                output_dir=results_path,
                logging_dir=logs_path,
                eval_strategy="epoch",
                do_eval=True,
                save_strategy="epoch",
                logging_strategy="epoch",
                overwrite_output_dir=True,
                num_train_epochs=training_params["nepochs"],
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                eval_accumulation_steps=1,
                gradient_accumulation_steps=4,
                weight_decay=0.01,
                warmup_ratio=0.08,
                logging_steps=log_frequency,
                save_steps=1000,
                metric_for_best_model="eval_f1_avg",
                load_best_model_at_end=True,
                save_total_limit=3,
            )

            if training_params.get("lr_override") is not None:
                training_args.learning_rate = training_params["lr_override"]

            hf_trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=trainer._compute_token_level_metrics,
                callbacks=[MLflowLoggingCallback(trainer._tracker_client)]
            )

            trainer._tracker_client.log_trainer_version(transformers_version)
            logger.info("Performing supervised training...")
            hf_trainer.train()

            if not skip_save_model:
                retrained_model_pack_path = os.path.join(trainer._retrained_models_dir,
                                                         f"{ModelType.HF_TRANSFORMER.value}_{run_id}.zip")
                model.save_pretrained(copied_model_directory)
                shutil.make_archive(retrained_model_pack_path.replace(".zip", ""), "zip", copied_model_directory)
                model_uri = trainer._tracker_client.save_model(retrained_model_pack_path, trainer._model_name,
                                                               trainer._model_manager)
                logger.info(f"Retrained model saved: {model_uri}")
            else:
                logger.info("Skipped saving on the retrained model")
            if redeploy:
                trainer.deploy_model(trainer._model_service, model, tokenizer)
            else:
                del model
                del tokenizer
                gc.collect()
                logger.info("Skipped deployment on the retrained model")
            logger.info("Unsupervised training finished")
            trainer._tracker_client.end_with_success()
        except Exception as e:
            logger.error("Unsupervised training failed")
            logger.exception(e)
            trainer._tracker_client.log_exceptions(e)
            trainer._tracker_client.end_with_failure()
        finally:
            data_file.close()
            with trainer._training_lock:
                trainer._training_in_progress = False
            _clean_up_training_cache(trainer._config.TRAINING_CACHE_DIR)
            trainer._housekeep_file(copied_model_pack_path)

    @staticmethod
    def _filter_training_data_and_concepts(data_file: TextIO) -> Tuple[Dict, List]:
        with open(data_file.name, "r") as f:
            training_data = json.load(f)
            te_stats_df = get_stats_from_trainer_export(training_data, return_df=True)
            rear_concept_ids = te_stats_df[(te_stats_df["anno_count"] - te_stats_df["anno_ignorance_counts"]) < HFTransformerSupervisedTrainer.MIN_EXAMPLE_COUNT_FOR_TRAINABLE_CONCEPT]["concept"].unique()
            logger.debug(f"The following concept(s) will be excluded due to the low example count(s): {rear_concept_ids}")
            filtered_training_data = filter_by_concept_ids(training_data, ModelType.HF_TRANSFORMER,
                                                           extra_excluded=rear_concept_ids)
            filtered_concepts = get_stats_from_trainer_export(filtered_training_data, return_df=True)["concept"].unique()
            return filtered_training_data, filtered_concepts

    @staticmethod
    def _update_model_with_concepts(model: PreTrainedModel, concepts: List[str]) -> PreTrainedModel:
        if model.config.label2id == {"LABEL_0": 0, "LABEL_1": 1}:
            logger.debug("Cannot find existing labels and IDs, creating new ones...")
            model.config.label2id = {"O": HFTransformerSupervisedTrainer.DEFAULT_LABEL_ID, "X": 1}
            model.config.id2label = {HFTransformerSupervisedTrainer.DEFAULT_LABEL_ID: "O", 1: "X"}
        avg_weight = torch.mean(model.classifier.weight, dim=0, keepdim=True)
        avg_bias = torch.mean(model.classifier.bias, dim=0, keepdim=True)
        for concept in concepts:
            if concept not in model.config.label2id.keys():
                model.config.label2id[concept] = len(model.config.label2id)
                model.config.id2label[len(model.config.id2label)] = concept
                model.classifier.weight = nn.Parameter(torch.cat((model.classifier.weight, avg_weight), 0))
                model.classifier.bias = nn.Parameter(torch.cat((model.classifier.bias, avg_bias), 0))
                model.classifier.out_features += 1
                model.num_labels += 1
        return model

    @staticmethod
    def _tokenize_and_chunk(documents: List[Dict], tokenizer: PreTrainedTokenizerBase, max_length: int, model: PreTrainedModel) -> Iterable[Dict[str, Any]]:
        for document in documents:
            encoded = tokenizer(document["text"], truncation=False, return_offsets_mapping=True)
            document["annotations"] = sorted(document["annotations"], key=lambda annotation: annotation["start"])
            for i in range(0, len(encoded["input_ids"]), max_length):
                chunked_input_ids = encoded["input_ids"][i:i + max_length]
                chunked_offsets_mapping = encoded["offset_mapping"][i:i + max_length]
                chunked_labels = [0] * len(chunked_input_ids)
                for annotation in document["annotations"]:
                    start = annotation["start"]
                    end = annotation["end"]
                    label_id = model.config.label2id[annotation["cui"]]
                    for idx, offset_mapping in enumerate(chunked_offsets_mapping):
                        if start <= offset_mapping[0] and offset_mapping[1] <= end:
                            chunked_labels[idx] = label_id
                chunked_attention_mask = encoded["attention_mask"][i:i + max_length]

                yield {
                    "input_ids": chunked_input_ids,
                    "labels": chunked_labels,
                    "attention_mask": chunked_attention_mask,
                }

    @staticmethod
    def _compute_token_level_metrics(eval_pred: EvalPrediction) -> Dict[str, Any]:
        predictions = np.argmax(softmax(eval_pred.predictions, axis=2), axis=2)
        label_ids = eval_pred.label_ids
        non_default_or_padding_indices = np.where((label_ids != HFTransformerSupervisedTrainer.DEFAULT_LABEL_ID) & (label_ids != HFTransformerSupervisedTrainer.PAD_LABEL_ID))
        filtered_predictions = predictions[non_default_or_padding_indices]
        filtered_label_ids = label_ids[non_default_or_padding_indices]
        precision, recall, f1, support = precision_recall_fscore_support(filtered_label_ids, filtered_predictions)
        accuracy = accuracy_score(filtered_label_ids, filtered_predictions)
        metrics = {
            "accuracy": accuracy,
            "f1_avg": np.average(f1),
            "precision_avg": np.average(precision),
            "recall_avg": np.average(recall),
            "support_avg": np.average(support),
        }
        logger.info("Evaluation metrics: %s", metrics)
        return metrics


@final
class MLflowLoggingCallback(TrainerCallback):
    def __init__(self, tracker_client: TrackerClient) -> None:
        self.tracker_client = tracker_client

    def on_log(self,
               args: TrainingArguments,
               state: TrainerState,
               control: TrainerControl,
               logs: Dict[str, float],
               **kwargs: Dict[str, Any]) -> None:
        if logs is not None:
            self.tracker_client.send_hf_training_logs(logs)


def _clean_up_training_cache(directory: str) -> None:
    for root, dirs, files in os.walk(directory, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
                logger.debug("Housekept file: %s", file_path)
            except Exception as e:
                logger.error("Error deleting file: %s : %s",file_path, e)

        for dir in dirs:
            dir_path = os.path.join(root, dir)
            try:
                shutil.rmtree(dir_path)
                logger.debug("Housekept directory: %s", dir_path)
            except Exception as e:
                logger.error("Error deleting directory: %s : %s",dir_path, e)
