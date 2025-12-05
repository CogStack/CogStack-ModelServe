import os
import logging
import shutil
import gc
import math
import mlflow
import tempfile
import inspect
import torch
import pandas as pd
import numpy as np
from collections import defaultdict
from functools import partial
from typing import Dict, TextIO, Any, Optional, List, final
from evaluate.visualization import radar_plot
from transformers import pipeline
from medcat import __version__ as medcat_version
from medcat.components.types import CoreComponentType
from medcat.components.ner.trf.transformers_ner import TransformersNERComponent
import json
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl, PreTrainedModel, Trainer
from app.domain import TrainerBackend
from app.utils import (
    get_settings,
    non_default_device_is_available,
    get_hf_pipeline_device_id,
    get_model_data_package_extension,
    dump_pydantic_object_to_dict,
)
from app.management.tracker_client import TrackerClient
from app.trainers.medcat_trainer import MedcatSupervisedTrainer
from app.processors.metrics_collector import get_stats_from_trainer_export
from app.exception import TrainingCancelledException

logger = logging.getLogger("cms")


@final
class MedcatDeIdentificationSupervisedTrainer(MedcatSupervisedTrainer):
    """A supervised trainer class for MedCAT de-identification (AnonCAT) models."""

    def run(
        self,
        training_params: Dict,
        data_file: TextIO,
        log_frequency: int,
        run_id: str,
        description: Optional[str] = None,
    ) -> None:
        """
        Runs the supervised training loop for MedCAT de-identification (AnonCAT) models.

        Args:
            training_params (Dict): A dictionary containing parameters for the training.
            data_file (TextIO): The file-like object containing the training data.
            log_frequency (int): The frequency at which logs should be recorded (e.g, the number of processed documents or finished epochs).
            run_id (str): The run ID of the training job.
            description (Optional[str]): The optional description of the training or change logs.
        """

        model_pack_path = None
        cdb_config_path = None
        copied_model_pack_path = None
        redeploy = self._config.REDEPLOY_TRAINED_MODEL == "true"
        skip_save_model = self._config.SKIP_SAVE_MODEL == "true"
        eval_mode = training_params["nepochs"] == 0
        self._tracker_client.log_trainer_mode(not eval_mode)
        if not eval_mode:
            try:
                logger.info("Loading a new model copy for training...")
                copied_model_pack_path = self._make_model_file_copy(self._model_pack_path, run_id)
                model = self._model_service.load_model(copied_model_pack_path)
                ner = model.pipe.get_component(CoreComponentType.ner)._component    # type: ignore
                ner.tokenizer.hf_tokenizer._in_target_context_manager = getattr(ner.tokenizer.hf_tokenizer, "_in_target_context_manager", False)
                ner.tokenizer.hf_tokenizer.clean_up_tokenization_spaces = getattr(ner.tokenizer.hf_tokenizer, "clean_up_tokenization_spaces", None)
                ner.tokenizer.hf_tokenizer.split_special_tokens = getattr(ner.tokenizer.hf_tokenizer, "split_special_tokens", False)
                _save_pretrained = ner.model.save_pretrained
                if ("safe_serialization" in inspect.signature(_save_pretrained).parameters):
                    ner.model.save_pretrained = partial(_save_pretrained, safe_serialization=(self._config.TRAINING_SAFE_MODEL_SERIALISATION == "true"))
                ner_config = {f"transformers.cat_config.{arg}": str(val) for arg, val in dump_pydantic_object_to_dict(ner.config.general).items()}
                ner_config.update({f"transformers.training.{arg}": str(val) for arg, val in ner.training_arguments.to_dict().items()})
                for key, val in ner_config.items():
                    ner_config[key] = "<EMPTY>" if val == "" else val
                self._tracker_client.log_model_config(ner_config)
                self._tracker_client.log_trainer_version(TrainerBackend.MEDCAT, medcat_version)

                eval_results: pd.DataFrame = None
                examples = None
                ner.training_arguments.num_train_epochs = 1
                ner.training_arguments.logging_steps = 1
                ner.training_arguments.overwrite_output_dir = False
                ner.training_arguments.save_strategy = "no"
                if training_params.get("lr_override") is not None:
                    ner.training_arguments.learning_rate = training_params["lr_override"]
                if training_params.get("test_size") is not None:
                    ner.config.general.test_size = training_params["test_size"]
                # This default evaluation strategy is "epoch"
                # ner.training_arguments.evaluation_strategy = "steps"
                # ner.training_arguments.eval_steps = 1
                logger.info("Performing supervised training...")
                model.config.meta.description = description or model.config.meta.description
                ner.config.general.description = description or ner.config.general.description
                dataset = None

                MetricsCallback.step = 0
                MetricsCallback.tracker_client = self._tracker_client
                for training in range(training_params["nepochs"]):

                    if dataset is not None:
                        dataset["train"] = dataset["train"].shuffle()
                        dataset["test"] = dataset["test"].shuffle()

                    ner = MedcatDeIdentificationSupervisedTrainer._customise_training_device(ner, self._config.DEVICE)
                    eval_results, examples, dataset = ner.train(
                        data_file.name,
                        ignore_extra_labels=True,
                        dataset=dataset,
                        trainer_callbacks=[MetricsCallback],
                    )
                    if (training + 1) % log_frequency == 0:
                        for _, row in eval_results.iterrows():
                            normalised_name = row["name"].replace(" ", "_").lower()
                            grouped_metrics = {
                                f"{normalised_name}/precision": row["p"] if row["p"] is not None else np.nan,
                                f"{normalised_name}/recall": row["r"] if row["r"] is not None else np.nan,
                                f"{normalised_name}/f1": row["f1"] if row["f1"] is not None else np.nan,
                                f"{normalised_name}/p_merged": row["p_merged"] if row["p_merged"] is not None else np.nan,
                                f"{normalised_name}/r_merged": row["r_merged"] if row["r_merged"] is not None else np.nan,
                            }
                            self._tracker_client.send_model_stats(grouped_metrics, training)

                        mean_metrics = {
                            "precision": eval_results["p"].mean(),
                            "recall": eval_results["r"].mean(),
                            "f1": eval_results["f1"].mean(),
                            "p_merged": eval_results["p_merged"].mean(),
                            "r_merged": eval_results["r_merged"].mean(),
                        }
                        self._tracker_client.send_model_stats(mean_metrics, training)

                    if (training + 1) == training_params["nepochs"] or self._cancel_event.is_set():
                        cui2names = {}
                        eval_results.sort_values(by=["cui"])
                        aggregated_metrics = []
                        for _, row in eval_results.iterrows():
                            if row["support"] == 0:  # the concept has not been used for annotation
                                continue
                            aggregated_metrics.append({
                                "per_concept_p": row["p"] if row["p"] is not None else 0.0,
                                "per_concept_r": row["r"] if row["r"] is not None else 0.0,
                                "per_concept_f1": row["f1"] if row["f1"] is not None else 0.0,
                                "per_concept_support": row["support"] if row["support"] is not None else 0.0,
                                "per_concept_p_merged": row["p_merged"] if row["p_merged"] is not None else 0.0,
                                "per_concept_r_merged": row["r_merged"] if row["r_merged"] is not None else 0.0,
                            })
                            cui2names[row["cui"]] = model.cdb.get_name(row["cui"]) if model.cdb is not None else ""
                        MedcatDeIdentificationSupervisedTrainer._save_metrics_plot(
                            aggregated_metrics,
                            list(cui2names.values()),
                            self._tracker_client,
                            self._model_name,
                        )
                        self._tracker_client.send_batched_model_stats(aggregated_metrics, run_id)
                        self._save_examples(examples, ["tp", "tn"])
                        self._tracker_client.log_classes_and_names(cui2names)
                        cui_counts, cui_unique_counts, cui_ignorance_counts, num_of_docs = get_stats_from_trainer_export(data_file.name)
                        self._tracker_client.log_document_size(num_of_docs)
                        self._save_trained_concepts(cui_counts, cui_unique_counts, cui_ignorance_counts, model)
                        self._sanity_check_model_and_save_results(data_file.name, self._model_service.from_model(model))

                    if self._cancel_event.is_set():
                        self._cancel_event.clear()
                        raise TrainingCancelledException("Training was cancelled by the user")

                if not skip_save_model:
                    model_pack_path = self.save_model_pack(
                        model,
                        self._retrained_models_dir,
                        self._config.BASE_MODEL_FILE,
                        description,
                    )
                    cdb_config_path = model_pack_path.replace(
                        get_model_data_package_extension(model_pack_path),
                        "_config.json",
                    )
                    with open(cdb_config_path, "w") as f:
                        json.dump(dump_pydantic_object_to_dict(model.config), f)
                    model_uri = self._tracker_client.save_model(
                        model_pack_path,
                        self._model_name,
                        self._model_manager,
                        self._model_service.info().model_type.value,
                    )
                    logger.info("Retrained model saved: %s", model_uri)
                    self._tracker_client.save_model_artifact(cdb_config_path, self._model_name)
                else:
                    logger.info("Skipped saving on the retrained model")
                if redeploy:
                    self.deploy_model(self._model_service, model, skip_save_model)
                else:
                    del model
                    gc.collect()
                    logger.info("Skipped deployment on the retrained model")
                logger.info("Supervised training finished")
                self._tracker_client.end_with_success()

                # Remove intermediate results folder on successful training
                results_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
                if results_path and os.path.isdir(results_path):
                    shutil.rmtree(results_path)
            except TrainingCancelledException as e:
                logger.exception(e)
                logger.info("Supervised training was cancelled by the user")
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
                self._housekeep_file(model_pack_path)
                self._housekeep_file(copied_model_pack_path)
                if cdb_config_path and os.path.exists(cdb_config_path):
                    os.remove(cdb_config_path)
        else:
            try:
                logger.info("Evaluating the running model...")
                if self._model_service._model is not None:
                    self._tracker_client.log_model_config(self.get_flattened_config(self._model_service._model))
                self._tracker_client.log_trainer_version(TrainerBackend.MEDCAT, medcat_version)
                ner = self._model.pipe.get_component(CoreComponentType.ner)._component  # type: ignore
                ner.tokenizer.hf_tokenizer._in_target_context_manager = getattr(ner.tokenizer.hf_tokenizer, "_in_target_context_manager", False)
                ner.tokenizer.hf_tokenizer.clean_up_tokenization_spaces = getattr(ner.tokenizer.hf_tokenizer, "clean_up_tokenization_spaces", None)
                ner.tokenizer.hf_tokenizer.split_special_tokens = getattr(ner.tokenizer.hf_tokenizer, "split_special_tokens", False)
                eval_results, examples = ner.eval(data_file.name)
                cui2names = {}
                eval_results.sort_values(by=["cui"])
                aggregated_metrics = []
                for _, row in eval_results.iterrows():
                    if row["support"] == 0:  # the concept has not been used for annotation
                        continue
                    aggregated_metrics.append({
                        "per_concept_p": row["p"] if row["p"] is not None else 0.0,
                        "per_concept_r": row["r"] if row["r"] is not None else 0.0,
                        "per_concept_f1": row["f1"] if row["f1"] is not None else 0.0,
                        "per_concept_support": row["support"] if row["support"] is not None else 0.0,
                        "per_concept_p_merged": row["p_merged"] if row["p_merged"] is not None else 0.0,
                        "per_concept_r_merged": row["r_merged"] if row["r_merged"] is not None else 0.0,
                    })
                    cui2names[row["cui"]] = self._model_service._model.cdb.get_name(row["cui"]) if self._model_service._model and self._model_service._model.cdb else ""
                self._tracker_client.send_batched_model_stats(aggregated_metrics, run_id)
                self._save_examples(examples, ["tp", "tn"])
                self._tracker_client.log_classes_and_names(cui2names)
                cui_counts, cui_unique_counts, cui_ignorance_counts, num_of_docs = get_stats_from_trainer_export(data_file.name)
                self._tracker_client.log_document_size(num_of_docs)
                self._sanity_check_model_and_save_results(data_file.name, self._model_service)
                logger.info("Model evaluation finished")
                self._tracker_client.end_with_success()
            except Exception as e:
                logger.exception("Model evaluation failed")
                self._tracker_client.log_exceptions(e)
                self._tracker_client.end_with_failure()
            finally:
                data_file.close()
                with self._training_lock:
                    self._training_in_progress = False

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

    @staticmethod
    def _customise_training_device(ner: TransformersNERComponent, device_name: str) -> TransformersNERComponent:
        if non_default_device_is_available(device_name):
            ner.model.to(torch.device(device_name))
            ner.ner_pipe = pipeline(
                model=ner.model,
                framework="pt",
                task="ner",
                tokenizer=ner.tokenizer.hf_tokenizer,
                device=get_hf_pipeline_device_id(device_name),
            )
        else:
            if device_name != "default":
                logger.warning("DEVICE is set to '%s' but it is not available. Using 'default' instead.", device_name)
        return ner


class MetricsCallback(TrainerCallback):
    """
    A callback class for logging metrics to Mlflow during training.

    Args:
        trainer (Trainer): The Hugging FaceTrainer object to which this callback is attached.
    """

    step: int = 0
    tracker_client: Optional[TrackerClient] = None

    def __init__(self, trainer: Trainer) -> None:
        self._trainer = trainer

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
            control (TrainerControl): The control object for the Trainer.
            logs (Dict[str, float]): A dictionary containing the metrics to log.
            **kwargs: Additional keyword arguments.
        """

        if logs is not None:
            if logs.get("eval_loss", None) is not None:
                logs["perplexity"] = math.exp(logs["eval_loss"])
            if self.tracker_client is not None:
                self.tracker_client.send_hf_metrics_logs(logs, step=MetricsCallback.step)
        MetricsCallback.step += 1


class LabelCountCallback(TrainerCallback):
    """
    A callback class for logging label counts to Mlflow during training.

    Args:
        trainer (Trainer): The Trainer object to which this callback is attached.
    """

    def __init__(self, trainer: Trainer) -> None:
        self._trainer = trainer
        self._label_counts: Dict = defaultdict(int)
        self._interval = get_settings().TRAINING_METRICS_LOGGING_INTERVAL

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Optional[PreTrainedModel] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Logs metrics at the end of a multiple of step interval.

        Args:
            args (TrainingArguments): The arguments used for training.
            state (TrainerState): The current state of the Trainer.
            control (TrainerControl): The control object for the Trainer.
            model: Optional[PreTrainedModel]: The model being trained.
            **kwargs: Additional keyword arguments.
        """

        step = state.global_step
        train_dataset = self._trainer.train_dataset
        batch_ids = train_dataset[step]["labels"]
        for id_ in batch_ids:
            self._label_counts[f"count_{model.config.id2label[id_]}"] += 1  # type: ignore
        self._label_counts.pop("count_O", None)
        self._label_counts.pop("count_X", None)

        if step % self._interval == 0:
            mlflow.log_metrics(self._label_counts, step=step)
