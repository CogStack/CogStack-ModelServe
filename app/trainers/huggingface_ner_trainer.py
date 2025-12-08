import os
import logging
import math
import torch
import gc
import json
import datasets
import random
import tempfile
import threading
import numpy as np
import pandas as pd
from functools import partial
from typing import final, Dict, TextIO, Optional, Any, List, Iterable, Tuple, Union, cast, TYPE_CHECKING
from torch import nn
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score as sklearn_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from seqeval.metrics import classification_report, accuracy_score as seqeval_accuracy_score
from scipy.special import softmax
from transformers import __version__ as transformers_version
from transformers import (
    AutoModelForMaskedLM,
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
    EarlyStoppingCallback,
)
from evaluate.visualization import radar_plot
from app.management.model_manager import ModelManager
from app.management.tracker_client import TrackerClient
from app.processors.metrics_collector import get_stats_from_trainer_export, sanity_check_model_with_trainer_export
from app.utils import (
    filter_by_concept_ids,
    reset_random_seed,
    non_default_device_is_available,
    create_model_data_package,
    get_model_data_package_extension,
    ensure_tensor_contiguity,
    get_model_data_package_base_name,
)
from app.trainers.base import UnsupervisedTrainer, SupervisedTrainer
from app.domain import ModelType, DatasetSplit, HfTransformerBackbone, Device, TrainerBackend, TaggingScheme
from app.processors.tagging import TagProcessor
from app.exception import AnnotationException, TrainingCancelledException, DatasetException
if TYPE_CHECKING:
    from app.model_services.huggingface_ner_model import HuggingFaceNerModel

logger = logging.getLogger("cms")


class _HuggingFaceNerTrainerCommon(object):

    @staticmethod
    def deploy_model(
        model_service: "HuggingFaceNerModel",
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        del model_service.model
        del model_service.tokenizer
        gc.collect()
        model_service.model = model
        model_service.tokenizer = tokenizer
        logger.info("Retrained model deployed")


@final
class HuggingFaceNerUnsupervisedTrainer(UnsupervisedTrainer, _HuggingFaceNerTrainerCommon):
    """
    An unsupervised trainer class for HuggingFace NER models.

    Args:
    model_service (HuggingFaceNerModel): An instance of the HuggingFace NER model service.
    """

    def __init__(self, model_service: "HuggingFaceNerModel") -> None:
        UnsupervisedTrainer.__init__(self, model_service._config, model_service.model_name)
        self._model_service = model_service
        self._model_name = model_service.model_name
        self._model_pack_path = model_service._model_pack_path
        self._retrained_models_dir = os.path.join(
            model_service._model_parent_dir,
            "retrained",
            self._model_name.replace(" ", "_"),
        )
        self._model_manager = ModelManager(type(model_service), model_service._config)
        self._max_length = model_service.model.config.max_position_embeddings
        os.makedirs(self._retrained_models_dir, exist_ok=True)

    def run(
        self,
        training_params: Dict,
        data_file: Union[TextIO, tempfile.TemporaryDirectory],
        log_frequency: int,
        run_id: str,
        description: Optional[str] = None,
    ) -> None:
        """
        Runs the unsupervised training loop for HuggingFace NER models.

        Args:
            training_params (Dict): A dictionary containing parameters for the training.
            data_file (Union[TextIO, tempfile.TemporaryDirectory]): The file-like object or temporary directory containing the training data.
            log_frequency (int): The frequency at which logs should be recorded (e.g, the number of processed documents or finished epochs).
            run_id (str): The run ID of the training job.
            description (Optional[str]): The optional description of the training or change logs.
        """

        copied_model_pack_path = None
        train_dataset = None
        eval_dataset = None
        redeploy = self._config.REDEPLOY_TRAINED_MODEL == "true"
        skip_save_model = self._config.SKIP_SAVE_MODEL == "true"
        results_path = os.path.abspath(os.path.join(self._config.TRAINING_CACHE_DIR, "results"))
        logs_path = os.path.abspath(os.path.join(self._config.TRAINING_CACHE_DIR, "logs"))
        reset_random_seed()

        eval_mode = training_params["nepochs"] == 0
        window_size = max(self._max_length - 2, 1)
        stride = max(window_size // 2, 1)
        self._tracker_client.log_trainer_mode(not eval_mode)
        if not eval_mode:
            try:
                logger.info("Loading a new model copy for training...")
                copied_model_pack_path = self._make_model_file_copy(self._model_pack_path, run_id)
                model, tokenizer = self._model_service.load_model(copied_model_pack_path)
                copied_model_directory = os.path.join(
                    os.path.dirname(copied_model_pack_path),
                    get_model_data_package_base_name(copied_model_pack_path),
                )
                mlm_model = self._get_mlm_model(model, copied_model_directory, self._config.DEVICE)

                if non_default_device_is_available(self._config.DEVICE):
                    mlm_model.to(self._config.DEVICE)
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
                    "token_type_ids": datasets.Sequence(datasets.Value("int32")),
                })
                train_dataset = datasets.Dataset.from_generator(
                    self._tokenize_and_chunk,
                    features=dataset_features,
                    gen_kwargs={
                        "texts": train_texts,
                        "tokenizer": tokenizer,
                        "max_length": self._max_length,
                        "window_size": window_size,
                        "stride": stride,
                    },
                    cache_dir=self._model_service._config.TRAINING_CACHE_DIR,
                )
                eval_dataset = datasets.Dataset.from_generator(
                    self._tokenize_and_chunk,
                    features=dataset_features,
                    gen_kwargs={
                        "texts": eval_texts,
                        "tokenizer": tokenizer,
                        "max_length": self._max_length,
                        "window_size": window_size,
                        "stride": stride,
                    },
                    cache_dir = self._model_service._config.TRAINING_CACHE_DIR,
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
                    use_cpu=self._config.DEVICE.lower() == Device.CPU.value if non_default_device_is_available(self._config.DEVICE) else False,
                )

                if training_params.get("lr_override") is not None:
                    training_args.learning_rate = training_params["lr_override"]

                mlflow_logging_callback = MLflowLoggingCallback(self._tracker_client)
                cancel_event_check_callback = CancelEventCheckCallback(self._cancel_event)
                trainer_callbacks = [mlflow_logging_callback, cancel_event_check_callback]
                early_stopping_patience = training_params.get("early_stopping_patience", -1)
                if early_stopping_patience > 0:
                    trainer_callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

                hf_trainer = Trainer(
                    model=mlm_model,
                    args=training_args,
                    data_collator=data_collator,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    callbacks=trainer_callbacks,
                )

                self._tracker_client.log_model_config(model.config.to_dict())
                self._tracker_client.log_trainer_version(TrainerBackend.TRANSFORMERS, transformers_version)
                logger.info("Performing unsupervised training...")
                hf_trainer.train()

                if cancel_event_check_callback.training_cancelled:
                    raise TrainingCancelledException("Training was cancelled by the user")

                model = self._get_final_model(model, mlm_model)
                if not skip_save_model:
                    model_pack_file_ext = get_model_data_package_extension(self._config.BASE_MODEL_FILE)
                    model_pack_file_name = f"{ModelType.HUGGINGFACE_NER.value}_{run_id}{model_pack_file_ext}"
                    retrained_model_pack_path = os.path.join(self._retrained_models_dir, model_pack_file_name)
                    model.save_pretrained(
                        copied_model_directory,
                        safe_serialization=(self._config.TRAINING_SAFE_MODEL_SERIALISATION == "true"),
                    )
                    create_model_data_package(copied_model_directory, retrained_model_pack_path)
                    model_uri = self._tracker_client.save_model(
                        retrained_model_pack_path,
                        self._model_name,
                        self._model_manager,
                        self._model_service.info().model_type.value,
                    )
                    logger.info(f"Retrained model saved: {model_uri}")
                else:
                    logger.info("Skipped saving on the retrained model")
                if redeploy:
                    self.deploy_model(self._model_service, model, tokenizer)
                else:
                    del model
                    del mlm_model
                    del tokenizer
                    gc.collect()
                    logger.info("Skipped deployment on the retrained model")
                logger.info("Unsupervised training finished")
                self._tracker_client.end_with_success()
            except TrainingCancelledException as e:
                logger.exception(e)
                logger.info("Unsupervised training was cancelled by the user")
                del model
                self._tracker_client.end_with_interruption()
            except Exception as e:
                logger.exception("Unsupervised training failed")
                self._tracker_client.log_exceptions(e)
                self._tracker_client.end_with_failure()
            finally:
                if isinstance(data_file, TextIO):
                    data_file.close()
                elif isinstance(data_file, tempfile.TemporaryDirectory):
                    data_file.cleanup()
                if train_dataset is not None:
                    train_dataset.cleanup_cache_files()
                if eval_dataset is not None:
                    eval_dataset.cleanup_cache_files()
                with self._training_lock:
                    self._training_in_progress = False
                self._clean_up_training_cache()
                self._housekeep_file(copied_model_pack_path)
        else:
            try:
                logger.info("Evaluating the running model...")
                self._tracker_client.log_model_config(self._model_service._model.config.to_dict())
                self._tracker_client.log_trainer_version(TrainerBackend.TRANSFORMERS, transformers_version)

                mlm_model = self._get_mlm_model(
                    self._model_service._model,
                    os.path.splitext(self._model_pack_path)[0],
                    self._config.DEVICE,
                )

                if non_default_device_is_available(self._config.DEVICE):
                    mlm_model.to(self._config.DEVICE)

                if isinstance(data_file, tempfile.TemporaryDirectory):
                    raw_dataset = datasets.load_from_disk(data_file.name)
                    if DatasetSplit.TEST.value in raw_dataset.keys():
                        eval_texts = raw_dataset[DatasetSplit.TEST.value]["text"]
                    elif DatasetSplit.VALIDATION.value in raw_dataset.keys():
                        eval_texts = raw_dataset[DatasetSplit.VALIDATION.value]["text"]
                    else:
                        raise DatasetException("No test or validation split found in the input dataset file")

                else:
                    with open(data_file.name, "r") as f:
                        eval_texts = [line.strip() for line in json.load(f)]

                dataset_features = datasets.Features({
                    "input_ids": datasets.Sequence(datasets.Value("int32")),
                    "attention_mask": datasets.Sequence(datasets.Value("int32")),
                    "special_tokens_mask": datasets.Sequence(datasets.Value("int32")),
                    "token_type_ids": datasets.Sequence(datasets.Value("int32")),
                })
                eval_dataset = datasets.Dataset.from_generator(
                    self._tokenize_and_chunk,
                    features=dataset_features,
                    gen_kwargs={"texts": eval_texts, "tokenizer": self._model_service._tokenizer, "max_length": self._max_length},
                    cache_dir=self._model_service._config.TRAINING_CACHE_DIR,
                )
                eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
                dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=1)
                device = torch.device(self._config.DEVICE) if non_default_device_is_available(self._config.DEVICE) else torch.device("cpu")

                self._model_service._model.eval()

                batch_size = 32

                def _create_iterative_masking(input_id: List[int], mask_token: int, pad_token_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
                    _input_id = torch.tensor(input_id)
                    attention_mask = torch.ones_like(_input_id)
                    attention_mask[_input_id == pad_token_id] = 0

                    n = _input_id.shape[0]
                    n_pad = _input_id[_input_id == pad_token_id].shape[0]

                    masked_sequence = _input_id.repeat(n - n_pad, 1)
                    attention_mask = attention_mask.repeat(n - n_pad, 1)

                    if mask_token != -999:
                        masked_sequence.fill_diagonal_(mask_token)

                    return masked_sequence, attention_mask

                all_input_ids = []
                all_attention_masks = []
                all_labels = []
                for batch in dataloader:
                    batch_input_ids = batch["input_ids"].tolist()
                    all_input_ids_ = []
                    all_attention_masks_ = []

                    for input_ids in batch_input_ids:
                        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)  # Ensure 2D shape
                        n = input_ids.size(-1)
                        n_windows = (n - 1) // stride + 1

                        input_ids_out = torch.full(
                            (n_windows, window_size),
                            self._model_service._tokenizer.pad_token_id,
                            dtype=torch.long,
                        )
                        attention_mask_out = torch.zeros((n_windows, window_size), dtype=torch.long)

                        for i, start in enumerate(range(0, n, stride)):
                            end = min(start + window_size, n)
                            length = end - start
                            input_ids_out[i, :length] = input_ids[0, start:end]
                            attention_mask_out[i, :length] = 1  # Mask only valid tokens

                        all_input_ids_.append(input_ids_out)
                        all_attention_masks_.append(attention_mask_out)

                    for input_id in all_input_ids_:
                        ls_rows_input_id: List[torch.Tensor] = []
                        ls_rows_attention_mask: List[torch.Tensor] = []
                        ls_labels: List[torch.Tensor] = []
                        for row in input_id:
                            input_id_out, attention_mask_out = _create_iterative_masking(
                                row,
                                mask_token=self._model_service._tokenizer.mask_token_id,
                                pad_token_id=self._model_service._tokenizer.pad_token_id,
                            )
                            labels, _ = _create_iterative_masking(
                                row,
                                mask_token=-999,
                                pad_token_id=self._model_service._tokenizer.pad_token_id,
                            )
                            ls_rows_input_id.extend(input_id_out)
                            ls_rows_attention_mask.extend(attention_mask_out)
                            ls_labels.extend(labels)

                        all_input_ids.append(torch.stack(ls_rows_input_id))
                        all_attention_masks.append(torch.stack(ls_rows_attention_mask))
                        all_labels.append(torch.stack(ls_labels))

                tensor_ids = torch.cat(all_input_ids, dim=0)
                tensor_attention_mask = torch.cat(all_attention_masks, dim=0)
                tensor_labels = torch.cat(all_labels, dim=0)

                n = tensor_ids.shape[0]
                losses = []
                for i in tqdm(range(0, n, batch_size)):
                    batch_input_ids = tensor_ids[i:i + batch_size].to(device)
                    batch_attention_mask = tensor_attention_mask[i:i + batch_size].to(device)
                    batch_tensor_labels = tensor_labels[i:i + batch_size].to(device)

                    with torch.no_grad():
                        output = mlm_model(
                            batch_input_ids,
                            attention_mask=batch_attention_mask,
                            labels=batch_tensor_labels,
                        )
                        losses.append(output.loss)
                        per_batch_metrics = {
                            "loss": losses[-1].item(),
                            "perplexity_mean": torch.exp(torch.stack(losses).mean()).item(),
                            "perplexity_median": torch.exp(torch.stack(losses).median()).item(),
                        }
                        logger.debug("Evaluation metrics: %s", per_batch_metrics)
                        self._tracker_client.send_model_stats(per_batch_metrics, i)

                self._tracker_client.end_with_success()
                logger.info("Model evaluation finished")
            except Exception as e:
                logger.exception("Model evaluation failed")
                self._tracker_client.log_exceptions(e)
                self._tracker_client.end_with_failure()
            finally:
                if isinstance(data_file, TextIO):
                    data_file.close()
                elif isinstance(data_file, tempfile.TemporaryDirectory):
                    data_file.cleanup()
                with self._training_lock:
                    self._training_in_progress = False


    @staticmethod
    def _get_mlm_model(model: PreTrainedModel, copied_model_directory: str, device: str) -> PreTrainedModel:
        if device.lower() == Device.DEFAULT.value:
            mlm_model = AutoModelForMaskedLM.from_pretrained(copied_model_directory, device_map="auto")
        else:
            mlm_model = AutoModelForMaskedLM.from_pretrained(copied_model_directory)
        ensure_tensor_contiguity(mlm_model)
        backbone_found = False
        for backbone in HfTransformerBackbone:
            if hasattr(model, backbone.value):
                setattr(mlm_model, backbone.value, getattr(model, backbone.value))
                backbone_found = True
                break
        if not backbone_found:
            raise ValueError(f"Unsupported model architecture: {type(model)}")
        return mlm_model

    @staticmethod
    def _get_final_model(model: PreTrainedModel, mlm_model: PreTrainedModel) -> PreTrainedModel:
        backbone_found = False
        for backbone in HfTransformerBackbone:
            if hasattr(model, backbone.value):
                setattr(model, backbone.value, getattr(mlm_model, backbone.value))
                backbone_found = True
                break
        if not backbone_found:
            raise ValueError(f"Unsupported model architecture: {type(model)}")
        return model

    @staticmethod
    def _tokenize_and_chunk(
        texts: Iterable[str],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        window_size: int,
        stride: int,
    ) -> Iterable[Dict[str, Any]]:
        for text in texts:
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                truncation=False,
                return_special_tokens_mask=True,
            )

            for start in range(0, len(encoded["input_ids"]), stride):
                end = min(start + window_size, len(encoded["input_ids"]))
                chunked_input_ids = encoded["input_ids"][start:end]
                padding_length = max(0, max_length - len(chunked_input_ids))

                chunked_input_ids += [tokenizer.pad_token_id] * padding_length
                chunked_attention_mask = encoded["attention_mask"][start:end] + [0] * padding_length
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
class HuggingFaceNerSupervisedTrainer(SupervisedTrainer, _HuggingFaceNerTrainerCommon):
    """
    A supervised trainer class for HuggingFace NER models.

    Args:
        model_service (HuggingFaceNerModel): An instance of the HuggingFace NER model service.
    """

    MIN_EXAMPLE_COUNT_FOR_TRAINABLE_CONCEPT = 5
    MAX_CONCEPTS_TO_TRACK = 20
    PAD_LABEL_ID = -100
    DEFAULT_LABEL_ID = 0
    CONTINUING_TOKEN_LABEL_ID = 1

    def __init__(self, model_service: "HuggingFaceNerModel") -> None:
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
                "labels": torch.tensor([self._add_padding(f["labels"], self.max_length, HuggingFaceNerSupervisedTrainer.PAD_LABEL_ID) for f in features], dtype=torch.long),
                "attention_mask": torch.tensor([self._add_padding(f["attention_mask"], self.max_length, 0) for f in features], dtype=torch.long),
            }

        @staticmethod
        def _add_padding(target: List[int], max_length: int, pad_token_id: int) -> List[int]:
            padding_length = max(0, max_length - len(target))
            paddings = [pad_token_id] * padding_length
            return target + paddings

    def run(
        self,
        training_params: Dict,
        data_file: TextIO,
        log_frequency: int,
        run_id: str,
        description: Optional[str] = None,
    ) -> None:
        """
        Runs the supervised training loop for HuggingFace NER models.

        Args:
            training_params (Dict): A dictionary containing parameters for the training.
            data_file (Union[TextIO, tempfile.TemporaryDirectory]): The file-like object or temporary directory containing the training data.
            log_frequency (int): The frequency at which logs should be recorded (e.g, the number of processed documents or finished epochs).
            run_id (str): The run ID of the training job.
            description (Optional[str]): The optional description of the training or change logs.
        """

        copied_model_pack_path = None
        redeploy = self._config.REDEPLOY_TRAINED_MODEL == "true"
        skip_save_model = self._config.SKIP_SAVE_MODEL == "true"
        results_path = os.path.abspath(os.path.join(self._config.TRAINING_CACHE_DIR, "results"))
        logs_path = os.path.abspath(os.path.join(self._config.TRAINING_CACHE_DIR, "logs"))
        reset_random_seed()
        eval_mode = training_params["nepochs"] == 0
        window_size = max(self._max_length - 2, 1)
        stride = max(window_size // 2, 1)
        self._tracker_client.log_trainer_mode(not eval_mode)
        tagging_scheme = TaggingScheme(self._model_service._config.TRAINING_HF_TAGGING_SCHEME.lower())
        if not eval_mode:
            try:
                logger.info("Loading a new model copy for training...")
                copied_model_pack_path = self._make_model_file_copy(self._model_pack_path, run_id)
                model, tokenizer = self._model_service.load_model(copied_model_pack_path)
                copied_model_directory = os.path.join(
                    os.path.dirname(copied_model_pack_path),
                    get_model_data_package_base_name(copied_model_pack_path),
                )

                if non_default_device_is_available(self._config.DEVICE):
                    model.to(self._config.DEVICE)

                filtered_training_data, filtered_concepts = self._filter_training_data_and_concepts(data_file)
                logger.debug(f"Filtered concepts: {filtered_concepts}")
                model = self._update_model_with_concepts(model, filtered_concepts, tagging_scheme)

                test_size = 0.2 if training_params.get("test_size") is None else training_params["test_size"]
                if test_size < 0:
                    logger.info("Using pre-defined train-validation-test split in trainer export...")
                    if len(filtered_training_data["projects"]) < 2:
                        raise AnnotationException("Not enough projects in the training data to provide a train-validation-test split")
                    train_documents = filtered_training_data["projects"][0]["documents"]
                    random.shuffle(train_documents)
                    eval_documents = filtered_training_data["projects"][1]["documents"]
                else:
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
                    self._tokenize_and_chunk,
                    features=dataset_features,
                    gen_kwargs={
                        "documents": train_documents,
                        "tokenizer": tokenizer,
                        "max_length": self._max_length,
                        "model": model,
                        "tagging_scheme": tagging_scheme,
                        "window_size": window_size,
                        "stride": stride,
                    },
                    cache_dir=self._config.TRAINING_CACHE_DIR,
                )
                eval_dataset = datasets.Dataset.from_generator(
                    self._tokenize_and_chunk,
                    features=dataset_features,
                    gen_kwargs={
                        "documents": eval_documents,
                        "tokenizer": tokenizer,
                        "max_length": self._max_length,
                        "model": model,
                        "tagging_scheme": tagging_scheme,
                        "window_size": window_size,
                        "stride": stride,
                    },
                    cache_dir = self._config.TRAINING_CACHE_DIR,
                )
                train_dataset.set_format(type=None, columns=["input_ids", "labels", "attention_mask"])
                eval_dataset.set_format(type=None, columns=["input_ids", "labels", "attention_mask"])

                data_collator = self._LocalDataCollator(max_length=self._max_length, pad_token_id=tokenizer.pad_token_id)
                training_args = self._get_training_args(results_path, logs_path, training_params, log_frequency)
                if training_params.get("lr_override") is not None:
                    training_args.learning_rate = training_params["lr_override"]

                mlflow_logging_callback = MLflowLoggingCallback(self._tracker_client)
                cancel_event_check_callback = CancelEventCheckCallback(self._cancel_event)
                trainer_callbacks = [mlflow_logging_callback, cancel_event_check_callback]
                early_stopping_patience = training_params.get("early_stopping_patience", -1)
                if early_stopping_patience > 0:
                    trainer_callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

                train_labels = []
                weights = torch.ones(model.num_labels, dtype=torch.float)
                for example in train_dataset:
                    train_labels.extend([label for label in example["labels"] if label != HuggingFaceNerSupervisedTrainer.PAD_LABEL_ID])
                unique_labels = np.unique(train_labels)
                class_weight_vect = compute_class_weight("balanced", classes=unique_labels, y=train_labels)
                for label_id, weight in zip(unique_labels, class_weight_vect):
                    weights[label_id] = weight

                if non_default_device_is_available(self._config.DEVICE):
                    weights = weights.to(self._config.DEVICE)
                else:
                    weights = weights.to(model.device)

                def _compute_loss(
                    outputs: Dict[str, Any],
                    labels: torch.Tensor,
                    num_items_in_batch: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
                    logits = outputs.get("logits")
                    loss_func = nn.CrossEntropyLoss(weight=weights, ignore_index=HuggingFaceNerSupervisedTrainer.PAD_LABEL_ID)
                    loss = loss_func(logits.view(-1, model.num_labels), labels.view(-1))    # type: ignore
                    return loss

                hf_trainer = Trainer(
                    model=model,
                    args=training_args,
                    data_collator=data_collator,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    compute_metrics=partial(
                        self._compute_metrics,
                        id2label=model.config.id2label,
                        tracker_client=self._tracker_client,
                        model_name=self._model_name,
                        token_level=True if tagging_scheme == TaggingScheme.FLAT else False,
                    ),
                    compute_loss_func=_compute_loss,
                    callbacks=trainer_callbacks,
                )

                self._tracker_client.log_model_config(model.config.to_dict())
                self._tracker_client.log_trainer_version(TrainerBackend.TRANSFORMERS, transformers_version)

                logger.info("Performing supervised training...")
                hf_trainer.train()

                if cancel_event_check_callback.training_cancelled:
                    raise TrainingCancelledException("Training was cancelled by the user")

                cui_counts, cui_unique_counts, cui_ignorance_counts, num_of_docs = get_stats_from_trainer_export(data_file.name)
                self._tracker_client.log_document_size(num_of_docs)
                self._save_trained_concepts(cui_counts, cui_unique_counts, cui_ignorance_counts, model)
                self._tracker_client.log_classes_and_names(model.config.id2label)
                self._sanity_check_model_and_save_results(data_file.name, self._model_service.from_model(model, tokenizer))

                if not skip_save_model:
                    model_pack_file_ext = get_model_data_package_extension(self._config.BASE_MODEL_FILE)
                    model_pack_file_name = f"{ModelType.HUGGINGFACE_NER.value}_{run_id}{model_pack_file_ext}"
                    retrained_model_pack_path = os.path.join(self._retrained_models_dir, model_pack_file_name)
                    model.save_pretrained(
                        copied_model_directory,
                        safe_serialization=(self._config.TRAINING_SAFE_MODEL_SERIALISATION == "true"),
                    )
                    create_model_data_package(copied_model_directory, retrained_model_pack_path)
                    logger.debug("Retrained model saved to: %s", retrained_model_pack_path)
                    model_uri = self._tracker_client.save_model(
                        retrained_model_pack_path,
                        self._model_name,
                        self._model_manager,
                        self._model_service.info().model_type.value,
                    )
                    logger.info(f"Retrained model saved: {model_uri}")
                else:
                    logger.info("Skipped saving on the retrained model")
                if redeploy:
                    self.deploy_model(self._model_service, model, tokenizer)
                else:
                    del model
                    del tokenizer
                    gc.collect()
                    logger.info("Skipped deployment on the retrained model")
                logger.info("Supervised training finished")
                self._tracker_client.end_with_success()
            except TrainingCancelledException as e:
                logger.exception(e)
                logger.info("Supervised training was cancelled")
                del model
                gc.collect()
                self._tracker_client.end_with_interruption()
            except Exception as e:
                logger.exception("Supervised training failed")
                self._tracker_client.log_exceptions(e)
                self._tracker_client.end_with_failure()
            finally:
                data_file.close()
                with self._training_lock:
                    self._training_in_progress = False
                self._clean_up_training_cache()
                self._housekeep_file(copied_model_pack_path)
        else:
            try:
                logger.info("Evaluating the running model...")
                self._tracker_client.log_model_config(self._model_service._model.config.to_dict())
                self._tracker_client.log_trainer_version(TrainerBackend.TRANSFORMERS, transformers_version)
                with open(data_file.name, "r") as f:
                    eval_data = json.load(f)
                eval_documents = [document for project in eval_data["projects"] for document in project["documents"]]
                dataset_features = datasets.Features({
                    "input_ids": datasets.Sequence(datasets.Value("int32")),
                    "labels": datasets.Sequence(datasets.Value("int32")),
                    "attention_mask": datasets.Sequence(datasets.Value("int32")),
                })
                eval_dataset = datasets.Dataset.from_generator(
                    self._tokenize_and_chunk,
                    features=dataset_features,
                    gen_kwargs={
                        "documents": eval_documents,
                        "tokenizer": self._model_service.tokenizer,
                        "max_length": self._max_length,
                        "model": self._model_service._model,
                        "tagging_scheme": tagging_scheme,
                        "window_size": window_size,
                        "stride": stride,
                    },
                    cache_dir=self._config.TRAINING_CACHE_DIR,
                )
                eval_dataset.set_format(type=None, columns=["input_ids", "labels", "attention_mask"])
                data_collator = self._LocalDataCollator(max_length=self._max_length, pad_token_id=self._model_service.tokenizer.pad_token_id)
                training_args = self._get_training_args(results_path, logs_path, training_params, log_frequency)
                training_args.eval_strategy = "no"
                hf_trainer = Trainer(
                    model=self._model_service.model,
                    args=training_args,
                    data_collator=data_collator,
                    train_dataset=None,
                    eval_dataset=None,
                    compute_metrics=partial(
                        self._compute_metrics,
                        id2label=self._model_service.model.config.id2label,
                        tracker_client=self._tracker_client,
                        model_name=self._model_name,
                        token_level=False,
                    ),
                    tokenizer=None,
                )
                eval_metrics = hf_trainer.evaluate(eval_dataset)
                logger.debug("Evaluation metrics: %s", eval_metrics)
                self._tracker_client.send_hf_metrics_logs(eval_metrics, 0)
                cui_counts, cui_unique_counts, cui_ignorance_counts, num_of_docs = get_stats_from_trainer_export(data_file.name)
                self._tracker_client.log_document_size(num_of_docs)
                self._sanity_check_model_and_save_results(data_file.name, self._model_service)
                self._tracker_client.end_with_success()
                logger.info("Model evaluation finished")
            except Exception as e:
                logger.exception("Model evaluation failed")
                self._tracker_client.log_exceptions(e)
                self._tracker_client.end_with_failure()
            finally:
                data_file.close()
                with self._training_lock:
                    self._training_in_progress = False

    @staticmethod
    def _filter_training_data_and_concepts(data_file: TextIO) -> Tuple[Dict, List]:
        with open(data_file.name, "r") as f:
            training_data = json.load(f)
            te_stats_df = get_stats_from_trainer_export(training_data, return_df=True)
            te_stats_df = cast(pd.DataFrame, te_stats_df)
            rear_concept_ids = te_stats_df[(te_stats_df["anno_count"] - te_stats_df["anno_ignorance_counts"]) < HuggingFaceNerSupervisedTrainer.MIN_EXAMPLE_COUNT_FOR_TRAINABLE_CONCEPT]["concept"].unique()
            logger.debug(f"The following concept(s) will be excluded due to the low example count(s): {rear_concept_ids}")
            filtered_training_data = filter_by_concept_ids(training_data, ModelType.HUGGINGFACE_NER, extra_excluded=rear_concept_ids)
            filtered_stats_df = get_stats_from_trainer_export(filtered_training_data, return_df=True)
            filtered_stats_df = cast(pd.DataFrame, filtered_stats_df)
            filtered_concepts = filtered_stats_df["concept"].unique()
            return filtered_training_data, filtered_concepts

    @staticmethod
    def _update_model_with_concepts(
        model: PreTrainedModel,
        concepts: List[str],
        tagging_scheme: TaggingScheme,
    ) -> PreTrainedModel:
        if model.config.label2id == {"LABEL_0": 0, "LABEL_1": 1}:
            logger.debug("Cannot find existing labels and IDs, creating new ones...")
            model.config.label2id = {"O": HuggingFaceNerSupervisedTrainer.DEFAULT_LABEL_ID, "X": HuggingFaceNerSupervisedTrainer.CONTINUING_TOKEN_LABEL_ID}
            model.config.id2label = {HuggingFaceNerSupervisedTrainer.DEFAULT_LABEL_ID: "O", HuggingFaceNerSupervisedTrainer.CONTINUING_TOKEN_LABEL_ID: "X"}
        return TagProcessor.update_model_by_tagging_scheme(model, concepts, tagging_scheme)

    @staticmethod
    def _tokenize_and_chunk(
        documents: List[Dict],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        model: PreTrainedModel,
        tagging_scheme: TaggingScheme,
        window_size: int,
        stride: int,
    ) -> Iterable[Dict[str, Any]]:
        for document in documents:
            encoded = tokenizer(
                document["text"],
                add_special_tokens=False,
                truncation=False,
                return_offsets_mapping=True,
            )
            document["annotations"] = sorted(document["annotations"], key=lambda annotation: annotation["start"])

            yield from TagProcessor.generate_chuncks_by_tagging_scheme(
                annotations=document["annotations"],
                tokenized=encoded,
                delfault_label_id=HuggingFaceNerSupervisedTrainer.DEFAULT_LABEL_ID,
                pad_token_id=tokenizer.pad_token_id,
                pad_label_id=HuggingFaceNerSupervisedTrainer.PAD_LABEL_ID,
                max_length=max_length,
                model=model,
                tagging_scheme=tagging_scheme,
                window_size=window_size,
                stride=stride,
            )

    @staticmethod
    def _compute_metrics(
        eval_pred: EvalPrediction,
        id2label: Dict[int, str],
        tracker_client: TrackerClient,
        model_name: str,
        token_level: bool,
    ) -> Dict[str, Any]:
        predictions = np.argmax(softmax(eval_pred.predictions, axis=2), axis=2)
        label_ids = eval_pred.label_ids
        non_padding_indices = np.where(label_ids != HuggingFaceNerSupervisedTrainer.PAD_LABEL_ID)
        non_padding_predictions = predictions[non_padding_indices].flatten()
        non_padding_label_ids = label_ids[non_padding_indices].flatten()
        labels = list(id2label.values())

        if token_level:
            # Get token level metrics
            precision, recall, f1, support = precision_recall_fscore_support(non_padding_label_ids, non_padding_predictions, labels=list(id2label.keys()), average=None)
            filtered_predictions, filtered_label_ids = zip(*[(a, b) for a, b in zip(non_padding_predictions, non_padding_label_ids) if not (a == b == HuggingFaceNerSupervisedTrainer.DEFAULT_LABEL_ID)])
            accuracy = sklearn_accuracy_score(filtered_label_ids, filtered_predictions)
            metrics = {
                "accuracy": accuracy,
                "f1_avg": np.average(f1[2:]),
                "precision_avg": np.average(precision[2:]),
                "recall_avg": np.average(recall[2:]),
                "support_avg": np.average(support[2:]),
            }
            aggregated_labels = []
            aggregated_metrics = []

            # Limit the number of labels to avoid excessive metrics logging
            for idx, (label, precision, recall, f1, support) in enumerate(zip(labels[2:HuggingFaceNerSupervisedTrainer.MAX_CONCEPTS_TO_TRACK+2],
                                                                            precision[2:HuggingFaceNerSupervisedTrainer.MAX_CONCEPTS_TO_TRACK+2],
                                                                            recall[2:HuggingFaceNerSupervisedTrainer.MAX_CONCEPTS_TO_TRACK+2],
                                                                            f1[2:HuggingFaceNerSupervisedTrainer.MAX_CONCEPTS_TO_TRACK+2],
                                                                            support[2:HuggingFaceNerSupervisedTrainer.MAX_CONCEPTS_TO_TRACK+2])):
                if support == 0:  # The concept has no true labels
                    continue
                metrics[f"{label}/precision"] = precision if precision is not None else 0.0
                metrics[f"{label}/recall"] = recall if recall is not None else 0.0
                metrics[f"{label}/f1"] = f1 if f1 is not None else 0.0
                metrics[f"{label}/support"] = support if support is not None else 0.0

                aggregated_labels.append(label)
                aggregated_metrics.append({
                    "per_concept_p": metrics[f"{label}/precision"],
                    "per_concept_r": metrics[f"{label}/recall"],
                    "per_concept_f1": metrics[f"{label}/f1"],
                })
        else:
            # Get entity level metrics
            y_true = []
            y_pred = []
            for i in range(label_ids.shape[0]):
                true_labels = []
                pred_labels = []
                for j in range(label_ids.shape[1]):
                    if label_ids[i, j] != HuggingFaceNerSupervisedTrainer.PAD_LABEL_ID:
                        true_labels.append(id2label[label_ids[i, j]])
                        pred_labels.append(id2label[predictions[i, j]])
                    else:
                        break
                y_true.append(true_labels)
                y_pred.append(pred_labels)
            report = classification_report(y_true, y_pred, output_dict=True)
            accuracy = seqeval_accuracy_score(y_true, y_pred)
            metrics = {
                "accuracy": accuracy,
                "f1_avg": np.mean([report[label]["f1-score"] for label in report]),
                "precision_avg": np.mean([report[label]["precision"] for label in report]),
                "recall_avg": np.mean([report[label]["recall"] for label in report]),
                "support_avg": np.mean([report[label]["support"] for label in report]),
            }
            aggregated_labels = []
            aggregated_metrics = []

            # Limit the number of labels to avoid excessive metrics logging
            label_keys = [k for k in report.keys() if k not in ['weighted avg', 'macro avg', 'micro avg']]
            for _, label in enumerate(label_keys[:HuggingFaceNerSupervisedTrainer.MAX_CONCEPTS_TO_TRACK]):
                if label not in report or report[label]['support'] == 0:  # The label has no true labels
                    continue
                metrics[f"{label}/precision"] = report[label]["precision"]
                metrics[f"{label}/recall"] = report[label]["recall"]
                metrics[f"{label}/f1"] = report[label]["f1-score"]
                metrics[f"{label}/support"] = report[label]["support"]

                aggregated_labels.append(label)
                aggregated_metrics.append({
                    "per_concept_p": metrics[f"{label}/precision"],
                    "per_concept_r": metrics[f"{label}/recall"],
                    "per_concept_f1": metrics[f"{label}/f1"],
                })

        HuggingFaceNerSupervisedTrainer._save_metrics_plot(
            aggregated_metrics,
            aggregated_labels,
            tracker_client,
            model_name,
        )
        logger.debug("Evaluation metrics: %s", metrics)
        return metrics

    @staticmethod
    def _save_metrics_plot(
        metrics: List[Dict],
        concepts: List[str],
        tracker_client: TrackerClient,
        model_name: str,
    ) -> None:
        try:
            plot = radar_plot(data=metrics, model_names=concepts)
            with tempfile.TemporaryDirectory() as d:
                with open(os.path.join(d, "metrics.png"), "w") as f:
                    plot.savefig(fname=f.name, format="png", bbox_inches="tight")
                    f.flush()
                    tracker_client.save_plot(f.name, model_name)
        except Exception as e:
            logger.error("Error occurred while plotting the metrics")
            logger.exception(e)

    def _get_training_args(
        self,
        results_path: str,
        logs_path: str,
        training_params: Dict,
        log_frequency: int,
    ) -> TrainingArguments:
        scaling_factor = 2
        cpu_count = os.cpu_count() or 1
        effective_batch_size = cpu_count * scaling_factor
        workers = max(1, cpu_count // scaling_factor)
        per_device_train_batch_size = max(1, effective_batch_size // workers)
        per_device_eval_batch_size = max(1, effective_batch_size // workers)
        eval_accumulation_steps = max(1, per_device_eval_batch_size // scaling_factor)
        torch.set_num_threads(workers)
        return TrainingArguments(
            output_dir=results_path,
            logging_dir=logs_path,
            eval_strategy="epoch",
            do_eval=True,
            save_strategy="epoch",
            logging_strategy="epoch",
            overwrite_output_dir=True,
            num_train_epochs=training_params["nepochs"],
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            eval_accumulation_steps=eval_accumulation_steps,
            gradient_accumulation_steps=1,
            weight_decay=0.01,
            warmup_ratio=0.08,
            logging_steps=log_frequency,
            save_steps=1000,
            metric_for_best_model="eval_f1_avg",
            greater_is_better=True,
            load_best_model_at_end=True,
            save_total_limit=3,
            use_cpu=self._config.DEVICE.lower() == Device.CPU.value if non_default_device_is_available(self._config.DEVICE) else False,
        )

    def _save_trained_concepts(
        self,
        training_concepts: Dict,
        training_unique_concepts: Dict,
        training_ignorance_counts: Dict,
        model: PreTrainedModel,
    ) -> None:
        if len(training_concepts.keys()) != 0:
            unknown_concepts = set(training_concepts.keys()) - set(model.config.label2id.keys())
            unknown_concept_pct = round(len(unknown_concepts) / len(training_concepts.keys()) * 100, 2)
            self._tracker_client.send_model_stats({
                "unknown_concept_count": len(unknown_concepts),
                "unknown_concept_pct": unknown_concept_pct,
            }, 0)
            if unknown_concepts:
                self._tracker_client.save_dataframe_as_csv(
                    "unknown_concepts.csv",
                    pd.DataFrame({"concept": list(unknown_concepts)}),
                    self._model_name,
                )
            annotation_count = []
            annotation_unique_count = []
            annotation_ignorance_count = []
            concepts = list(training_concepts.keys())
            for c in concepts:
                annotation_count.append(training_concepts[c])
                annotation_unique_count.append(training_unique_concepts[c])
                annotation_ignorance_count.append(training_ignorance_counts[c])
            self._tracker_client.save_dataframe_as_csv(
                "trained_concepts.csv",
                pd.DataFrame({
                    "concept": concepts,
                    "anno_count": annotation_count,
                    "anno_unique_count": annotation_unique_count,
                    "anno_ignorance_count": annotation_ignorance_count,
                }),
                self._model_name,
            )

    def _sanity_check_model_and_save_results(self, data_file_path: str, model_service: "HuggingFaceNerModel") -> None:
        self._tracker_client.save_dataframe_as_csv(
            "sanity_check_result.csv",
            sanity_check_model_with_trainer_export(
                data_file_path,
                model_service,
                return_df=True,
                include_anchors=True,
            ),
            self._model_name,
        )


@final
class MLflowLoggingCallback(TrainerCallback):
    """
    A callback class for logging training metrics to MLflow.

    Args:
        tracker_client (TrackerClient): An instance of TrackerClient used for logging.
    """

    def __init__(self, tracker_client: TrackerClient) -> None:
        self.tracker_client = tracker_client
        self.epoch = 0

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Dict[str, float],
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Logs metrics at the end of each epoch.

        Args:
            args (TrainingArguments): The arguments used for training.
            state (TrainerState): The current state of the Trainer.
            control (TrainerControl): The current control of the Trainer.
            logs (Dict[str, float]): A dictionary containing the metrics to log.
            **kwargs (Dict[str, Any]): Additional keyword arguments.
        """

        if logs is not None:
            if logs.get("eval_loss", None) is not None:
                logs["perplexity"] = math.exp(logs["eval_loss"])
            self.tracker_client.send_hf_metrics_logs(logs, self.epoch)
        self.epoch += 1


@final
class CancelEventCheckCallback(TrainerCallback):
    """
    A callback class for checking a cancellation event during training.

    Args:
        cancel_event (threading.Event): A threading event that signals whether training should be cancelled.
    """

    def __init__(self, cancel_event: threading.Event) -> None:
        self.cancel_event = cancel_event
        self.training_cancelled = False

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Checks if the training should be cancelled at the end of each training step.

        Args:
            args (TrainingArguments): The arguments used for training.
            state (TrainerState): The current state of the Trainer.
            control (TrainerControl): The current control of the Trainer.
            **kwargs (Dict[str, Any]): Additional keyword arguments.
        """

        if self.cancel_event.is_set():
            control.should_training_stop = True
            self.cancel_event.clear()
            self.training_cancelled = True
