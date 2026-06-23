import os
import logging
import math
import inspect
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
from app.processors.lora_adaptor import LoraAdaptor
from app.utils import (
    filter_by_concept_ids,
    reset_random_seed,
    non_default_device_is_available,
    create_model_data_package,
    get_model_data_package_extension,
    ensure_tensor_contiguity,
    get_model_data_package_base_name,
    freeze_hf_model_params_by_names,
    save_model_to_clean_directory,
)
from app.trainers.base import UnsupervisedTrainer, SupervisedTrainer
from app.domain import ModelType, DatasetSplit, HfTransformerBackbone, Device, TrainerBackend, TaggingScheme
from app.processors.tagging import TagProcessor
from app.exception import AnnotationException, TrainingCancelledException, DatasetException
if TYPE_CHECKING:
    from app.model_services.huggingface_ner_model import HuggingFaceNerModel

logger = logging.getLogger("cms")


class _HuggingFaceNerTrainerCommon(object):
    TRAINING_CONTEXT_CAP = 512

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

    @staticmethod
    def _create_training_arguments(**kwargs: Any) -> TrainingArguments:
        valid_args = set(inspect.signature(TrainingArguments.__init__).parameters.keys())

        # Handle naming differences across transformers versions
        if "eval_strategy" in kwargs and "eval_strategy" not in valid_args:
            kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
        if "evaluation_strategy" in kwargs and "evaluation_strategy" not in valid_args:
            kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")

        filtered = {k: v for k, v in kwargs.items() if k in valid_args}
        dropped = set(kwargs.keys()) - set(filtered.keys())
        if dropped:
            logger.debug(
                "Ignoring unsupported training arguments for this transformers installed: %s",sorted(dropped)
            )
        return TrainingArguments(**filtered)

    @staticmethod
    def _resolve_safe_max_length(model: PreTrainedModel, tokenizer: Optional[PreTrainedTokenizerBase]) -> int:
        model_max = int(
            getattr(model.config, "max_position_embeddings", None) or _HuggingFaceNerTrainerCommon.TRAINING_CONTEXT_CAP
        )
        tokenizer_max = int(
            getattr(tokenizer, "model_max_length", model_max) if tokenizer is not None else model_max
        )
        return max(1, min(model_max, tokenizer_max, _HuggingFaceNerTrainerCommon.TRAINING_CONTEXT_CAP))

    @staticmethod
    def _calculate_batch_sizes(training_params: Dict, device_config: str) -> Dict[str, int]:
        scaling_factor = max(1, int(training_params.get("scaling_factor", 1)))
        cpu_count = os.cpu_count() or 1
        effective_train_batch_size = 16
        effective_eval_batch_size = 16
        device = device_config.lower()
        cuda_available = device.startswith(Device.GPU.value) and torch.cuda.is_available()
        mps_available = device.startswith(Device.MPS.value) and torch.backends.mps.is_available()
        accelerator_available = cuda_available or mps_available

        if accelerator_available:
            workers = max(1, min(4, cpu_count))
            batch_size_cap = 16 if cuda_available else 8
            micro_batch_size = max(1, scaling_factor * 2)
        else:
            workers = max(1, min(4, cpu_count // scaling_factor))
            batch_size_cap = 16
            micro_batch_size = max(1, cpu_count // workers)

        per_device_train_batch_size = min(batch_size_cap, micro_batch_size)
        per_device_eval_batch_size = min(batch_size_cap, micro_batch_size)
        eval_accumulation_steps = max(1, math.ceil(effective_eval_batch_size / per_device_eval_batch_size))
        gradient_accumulation_steps = max(1, math.ceil(effective_train_batch_size / per_device_train_batch_size))
        return {
            "workers": workers,
            "per_device_train_batch_size": per_device_train_batch_size,
            "per_device_eval_batch_size": per_device_eval_batch_size,
            "eval_accumulation_steps": eval_accumulation_steps,
            "gradient_accumulation_steps": gradient_accumulation_steps,
        }


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
        self._max_length = self._resolve_safe_max_length(model_service.model, model_service.tokenizer)
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
        stride = min(max(int(window_size * 0.75), 1), window_size - 1)
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

                training_token_count = sum(len(example["input_ids"]) for example in train_dataset)
                logger.debug(f"Total training tokens: {training_token_count}")
                self._tracker_client.log_training_token_count(training_token_count)

                data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.2)

                batch_sizes = self._calculate_batch_sizes(training_params, self._config.DEVICE)
                per_device_train_batch_size = batch_sizes["per_device_train_batch_size"]
                per_device_eval_batch_size = batch_sizes["per_device_eval_batch_size"]
                gradient_accumulation_steps = batch_sizes["gradient_accumulation_steps"]
                torch.set_num_threads(batch_sizes["workers"])

                training_args = self._create_training_arguments(
                    output_dir=results_path,
                    logging_dir=logs_path,
                    eval_strategy="epoch",
                    save_strategy="epoch",
                    overwrite_output_dir=True,
                    num_train_epochs=training_params["nepochs"],
                    per_device_train_batch_size=per_device_train_batch_size,
                    per_device_eval_batch_size=per_device_eval_batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    logging_steps=log_frequency,
                    save_steps=1000,
                    load_best_model_at_end=True,
                    save_total_limit=3,
                    report_to="none",
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
                    save_model_to_clean_directory(
                        model,
                        tokenizer,
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
            mlm_model = AutoModelForMaskedLM.from_pretrained(
                copied_model_directory, device_map="auto", low_cpu_mem_usage=True
            )
        else:
            mlm_model = AutoModelForMaskedLM.from_pretrained(
                copied_model_directory, low_cpu_mem_usage=True
            )
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
        self._max_length = self._resolve_safe_max_length(model_service.model, model_service.tokenizer)
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
        stride = min(max(int(window_size * 0.75), 1), window_size - 1)
        self._tracker_client.log_trainer_mode(not eval_mode)
        tagging_scheme = TaggingScheme(self._model_service._config.TRAINING_HF_NER_TAGGING_SCHEME.lower())
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
                model = self._apply_lora_adapter_if_enabled(model)

                self._freeze_params_or_classifier(model, self._config.TRAINING_HF_NER_FROZEN_PARAM_NAMES)

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

                training_token_count = sum(len(example["input_ids"]) for example in train_dataset)
                logger.debug(f"Total training tokens: {training_token_count}")
                self._tracker_client.log_training_token_count(training_token_count)

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

                train_labels: List[str] = []
                weights = torch.ones(model.num_labels, dtype=torch.float)
                for example in train_dataset:
                    train_labels.extend(label for label in example["labels"] if label != HuggingFaceNerSupervisedTrainer.PAD_LABEL_ID)
                unique_labels = np.unique(train_labels)
                class_weight_vect = compute_class_weight(
                    class_weight="balanced",
                    classes=unique_labels,
                    y=train_labels,
                )
                for label_id, weight in zip(unique_labels, class_weight_vect):
                    weights[label_id] = weight
                weights = torch.sqrt(weights)
                weights = torch.clamp(weights, max=10.0)

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
                    loss_weights = weights.to(device=logits.device, dtype=logits.dtype) # type: ignore
                    loss_func = nn.CrossEntropyLoss(
                        weight=loss_weights,
                        ignore_index=HuggingFaceNerSupervisedTrainer.PAD_LABEL_ID,
                    )
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
                model = self._merge_lora_if_enabled(model)

                if not skip_save_model:
                    model_pack_file_ext = get_model_data_package_extension(self._config.BASE_MODEL_FILE)
                    model_pack_file_name = f"{ModelType.HUGGINGFACE_NER.value}_{run_id}{model_pack_file_ext}"
                    retrained_model_pack_path = os.path.join(self._retrained_models_dir, model_pack_file_name)
                    save_model_to_clean_directory(
                        model,
                        tokenizer,
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
                training_args = self._get_training_args(results_path, logs_path, training_params, log_frequency, "no")
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

    def _apply_lora_adapter_if_enabled(self, model: PreTrainedModel) -> PreTrainedModel:
        if self._config.TRAINING_HF_NER_ENABLE_LORA.lower() != "true":
            return model

        logger.info("Applying LoRA adapters for supervised training...")
        peft_model, _ = LoraAdaptor.apply(
            model=model,
            task_type="TOKEN_CLS",
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        if hasattr(peft_model, "print_trainable_parameters"):
            peft_model.print_trainable_parameters()
        return cast(PreTrainedModel, peft_model)

    def _merge_lora_if_enabled(self, model: PreTrainedModel) -> PreTrainedModel:
        if self._config.TRAINING_HF_NER_ENABLE_LORA.lower() != "true":
            return model

        if hasattr(model, "merge_and_unload"):
            logger.info("Merging LoRA adapters into the base model...")
            merged_model = model.merge_and_unload()
            return cast(PreTrainedModel, merged_model)
        return model

    @staticmethod
    def _freeze_params_or_classifier(model: PreTrainedModel, params_names_csv: str) -> None:
        param_names = [param_name.strip() for param_name in params_names_csv.split(",") if param_name.strip()]
        if not param_names:
            return

        if "except_classifier" in param_names:
            frozen_params, total_params = freeze_hf_model_params_by_names(
                model=model,
                params_names_csv="classifier,score",
                include=False,
            )
            logger.info(
                "Configured training on classification head only: %s; %s/%s parameter tensors remain trainable",
                ["classifier", "score"],
                total_params - frozen_params,
                total_params,
            )
            return

        frozen_params, total_params = freeze_hf_model_params_by_names(
            model=model,
            params_names_csv=params_names_csv,
            include=True,
        )
        logger.info(
            "Configured frozen parameters by names: %s; %s/%s parameter tensors remain trainable",
            param_names,
            total_params - frozen_params,
            total_params,
        )

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

        predictions = np.argmax(eval_pred.predictions, axis=2)
        label_ids = eval_pred.label_ids

        labels = list(id2label.values())
        ignored_labels = {"O", "X"}

        metric_indices = [
            idx for idx, label_id in enumerate(list(id2label.keys()))
            if id2label[label_id] not in ignored_labels
        ]

        if token_level:
            # Get token level metrics
            pred_labels = []
            true_labels = []

            for i in range(label_ids.shape[0]):
                for j in range(label_ids.shape[1]):
                    if label_ids[i, j] == HuggingFaceNerSupervisedTrainer.PAD_LABEL_ID:
                        continue

                    pred_labels.append(predictions[i, j])
                    true_labels.append(label_ids[i, j])

            precision, recall, f1, support = precision_recall_fscore_support(
                np.array(true_labels),
                np.array(pred_labels),
                labels=list(id2label.keys()),
                average=None
            )

            accuracy = sklearn_accuracy_score(np.array(true_labels), np.array(pred_labels))

            metrics = {
                "accuracy": accuracy,
                "f1_avg": np.average([f1[idx] for idx in metric_indices]) if metric_indices else 0.0,
                "precision_avg": np.average([precision[idx] for idx in metric_indices]) if metric_indices else 0.0,
                "recall_avg": np.average([recall[idx] for idx in metric_indices]) if metric_indices else 0.0,
                "support_avg": np.average([support[idx] for idx in metric_indices]) if metric_indices else 0.0,
            }

            aggregated_labels = []
            aggregated_metrics = []

            metric_rows = [
                (labels[idx], precision[idx], recall[idx], f1[idx], support[idx])
                for idx in metric_indices
            ]

            # Limit the number of labels to avoid excessive metrics logging
            for _, (label, p, r, f1_, support_) in enumerate(
                metric_rows[:HuggingFaceNerSupervisedTrainer.MAX_CONCEPTS_TO_TRACK]
            ):
                if support_ == 0:
                    continue

                metrics[f"{label}/precision"] = p
                metrics[f"{label}/recall"] = r
                metrics[f"{label}/f1"] = f1_
                metrics[f"{label}/support"] = support_

                aggregated_labels.append(label)
                aggregated_metrics.append({
                    "per_concept_p": p,
                    "per_concept_r": r,
                    "per_concept_f1": f1_,
                })
        else:
            # Get entity level metrics
            true_label_sequences = []
            pred_label_sequences = []

            for i in range(label_ids.shape[0]):

                entity_true_labels: List = []
                entity_pred_labels: List = []

                for j in range(label_ids.shape[1]):

                    if label_ids[i, j] == HuggingFaceNerSupervisedTrainer.PAD_LABEL_ID:
                        break

                    entity_true_labels.append(id2label[label_ids[i, j]])
                    entity_pred_labels.append(id2label[predictions[i, j]])

                true_label_sequences.append(entity_true_labels)
                pred_label_sequences.append(entity_pred_labels)

            report = classification_report(true_label_sequences, pred_label_sequences, output_dict=True)
            accuracy = seqeval_accuracy_score(true_label_sequences, pred_label_sequences)

            target_labels = [
                key for key in report.keys()
                if key not in {"weighted avg", "macro avg", "micro avg", "accuracy"}
                and key not in ignored_labels
            ]

            metrics = {
                "accuracy": accuracy,
                "f1_avg": np.mean([report[label]["f1-score"] for label in target_labels]) if target_labels else 0.0,
                "precision_avg": np.mean([report[label]["precision"] for label in target_labels]) if target_labels else 0.0,
                "recall_avg": np.mean([report[label]["recall"] for label in target_labels]) if target_labels else 0.0,
                "support_avg": np.mean([report[label]["support"] for label in target_labels]) if target_labels else 0.0,
            }

            aggregated_labels = []
            aggregated_metrics = []

            # Limit the number of labels to avoid excessive metrics logging
            for label in target_labels[:HuggingFaceNerSupervisedTrainer.MAX_CONCEPTS_TO_TRACK]:

                if label not in report or report[label]["support"] == 0:
                    continue

                metrics[f"{label}/precision"] = report[label]["precision"]
                metrics[f"{label}/recall"] = report[label]["recall"]
                metrics[f"{label}/f1"] = report[label]["f1-score"]
                metrics[f"{label}/support"] = report[label]["support"]

                aggregated_labels.append(label)
                aggregated_metrics.append({
                    "per_concept_p": report[label]["precision"],
                    "per_concept_r": report[label]["recall"],
                    "per_concept_f1": report[label]["f1-score"],
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
        eval_strategy: str = "epoch",
    ) -> TrainingArguments:
        batch_sizes = self._calculate_batch_sizes(training_params, self._config.DEVICE)
        workers = batch_sizes["workers"]
        per_device_train_batch_size = batch_sizes["per_device_train_batch_size"]
        per_device_eval_batch_size = batch_sizes["per_device_eval_batch_size"]
        eval_accumulation_steps = batch_sizes["eval_accumulation_steps"]
        gradient_accumulation_steps = batch_sizes["gradient_accumulation_steps"]
        torch.set_num_threads(workers)
        logger.debug("Training scaling arguments:")
        logger.debug("  - CPU workers: %d", workers)
        logger.debug("  - Per device train batch size: %d", per_device_train_batch_size)
        logger.debug("  - Per device eval batch size: %d", per_device_eval_batch_size)
        logger.debug("  - Eval accumulation steps: %d", eval_accumulation_steps)
        logger.debug("  - Gradient accumulation steps: %d", gradient_accumulation_steps)

        return self._create_training_arguments(
            output_dir=results_path,
            logging_dir=logs_path,
            eval_strategy=eval_strategy,
            do_eval=True,
            save_strategy="epoch",
            logging_strategy="epoch",
            overwrite_output_dir=True,
            num_train_epochs=training_params["nepochs"],
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            eval_accumulation_steps=eval_accumulation_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            weight_decay=0.01,
            warmup_ratio=0.08,
            max_grad_norm=1.0,
            logging_steps=log_frequency,
            save_steps=1000,
            metric_for_best_model="eval_f1_avg",
            greater_is_better=True,
            load_best_model_at_end=True,
            save_total_limit=3,
            report_to="none",
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
            labels = set(model.config.label2id.keys())
            model_concepts = set()
            for label in labels:
                if label in {"O", "X"}:
                    continue
                if len(label) > 2 and label[1] == "-" and label[0] in {"B", "I", "E", "S"}:
                    model_concepts.add(label[2:])
                else:
                    model_concepts.add(label)

            unknown_concepts = set(training_concepts.keys()) - model_concepts
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
