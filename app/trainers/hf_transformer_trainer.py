import os
import logging
import torch
import gc
import json
import shutil
import datasets
from typing import final, Dict, TextIO, Optional, Any, List, Iterable
from management.model_manager import ModelManager
from management.tracker_client import TrackerClient
from model_services.base import AbstractModelService
from trainers.base import UnsupervisedTrainer, SupervisedTrainer
from domain import ModelType
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
    PreTrainedTokenizer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)

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
        self._max_length = 512
        os.makedirs(self._retrained_models_dir, exist_ok=True)

    @staticmethod
    def run(trainer: "HFTransformerUnsupervisedTrainer",
            training_params: Dict,
            data_file: TextIO,
            log_frequency: int,
            run_id: str,
            description: Optional[str] = None) -> None:
        copied_model_pack_path = None
        redeploy = trainer._config.REDEPLOY_TRAINED_MODEL == "true"
        skip_save_model = trainer._config.SKIP_SAVE_MODEL == "true"
        results_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
        logs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
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
            with open(data_file.name, "r") as f:
                lines = json.load(f)
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
                gen_kwargs={"texts": train_texts, "tokenizer": tokenizer, "max_length": trainer._max_length}
            )
            eval_dataset = datasets.Dataset.from_generator(
                trainer._tokenize_and_chunk,
                features=dataset_features,
                gen_kwargs={"texts": eval_texts, "tokenizer": tokenizer, "max_length": trainer._max_length}
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
            data_file.close()
            with trainer._training_lock:
                trainer._training_in_progress = False
            if results_path and os.path.isdir(results_path):
                shutil.rmtree(results_path)
            if logs_path and os.path.isdir(logs_path):
                shutil.rmtree(logs_path)
            trainer._housekeep_file(copied_model_pack_path)

    @staticmethod
    def deploy_model(model_service: AbstractModelService,
                     model: PreTrainedModel,
                     tokenizer: PreTrainedTokenizer) -> None:
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
    def _tokenize_and_chunk(texts: List[str], tokenizer: PreTrainedTokenizer, max_length: int) -> Iterable[Dict[str, Any]]:
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

    def __init__(self, model_service: AbstractModelService) -> None:
        raise NotImplementedError(f"Supervised training is not supported for {type(model_service)}")


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
