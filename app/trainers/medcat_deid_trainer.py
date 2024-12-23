import gc
import inspect
import logging
import os
import shutil
import tempfile
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Optional, TextIO, final

import mlflow
import numpy as np
import pandas as pd
import torch
from evaluate.visualization import radar_plot
from medcat import __version__ as medcat_version
from medcat.ner.transformers_ner import TransformersNER
from transformers import (
    PreTrainedModel,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    pipeline,
)

from utils import get_hf_pipeline_device_id, get_settings, non_default_device_is_available

from management import tracker_client
from processors.metrics_collector import get_stats_from_trainer_export
from trainers.medcat_trainer import MedcatSupervisedTrainer

logger = logging.getLogger("cms")


class MetricsCallback(TrainerCallback):
    def __init__(self, trainer: Trainer) -> None:
        self._trainer = trainer
        self._step = 0
        self._interval = get_settings().TRAINING_METRICS_LOGGING_INTERVAL

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Dict[str, Any],
    ) -> None:
        if self._step == 0:
            self._step += 1
            return
        if self._step % self._interval == 0 and state.log_history:
            metrics = state.log_history[-1]
            mlflow.log_metrics(metrics, step=self._step)
        self._step += 1


class LabelCountCallback(TrainerCallback):
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
        step = state.global_step
        train_dataset = self._trainer.train_dataset
        batch_ids = train_dataset[step]["labels"]
        for id_ in batch_ids:
            self._label_counts[f"count_{model.config.id2label[id_]}"] += 1  # type: ignore
        self._label_counts.pop("count_O", None)
        self._label_counts.pop("count_X", None)

        if step % self._interval == 0:
            mlflow.log_metrics(self._label_counts, step=step)


@final
class MedcatDeIdentificationSupervisedTrainer(MedcatSupervisedTrainer):
    @staticmethod
    def run(
        trainer: "MedcatDeIdentificationSupervisedTrainer",
        training_params: Dict,
        data_file: TextIO,
        log_frequency: int,
        run_id: str,
        description: Optional[str] = None,
    ) -> None:
        model_pack_path = None
        cdb_config_path = None
        copied_model_pack_path = None
        redeploy = trainer._config.REDEPLOY_TRAINED_MODEL == "true"
        skip_save_model = trainer._config.SKIP_SAVE_MODEL == "true"
        eval_mode = training_params["nepochs"] == 0
        trainer._tracker_client.log_trainer_mode(not eval_mode)
        if not eval_mode:
            try:
                logger.info("Loading a new model copy for training...")
                copied_model_pack_path = trainer._make_model_file_copy(
                    trainer._model_pack_path, run_id
                )
                model = trainer._model_service.load_model(copied_model_pack_path)
                ner = model._addl_ner[0]
                ner.tokenizer.hf_tokenizer._in_target_context_manager = getattr(
                    ner.tokenizer.hf_tokenizer, "_in_target_context_manager", False
                )
                ner.tokenizer.hf_tokenizer.clean_up_tokenization_spaces = getattr(
                    ner.tokenizer.hf_tokenizer, "clean_up_tokenization_spaces", None
                )
                ner.tokenizer.hf_tokenizer.split_special_tokens = getattr(
                    ner.tokenizer.hf_tokenizer, "split_special_tokens", False
                )
                _save_pretrained = ner.model.save_pretrained
                if "safe_serialization" in inspect.signature(_save_pretrained).parameters:
                    ner.model.save_pretrained = partial(
                        _save_pretrained,
                        safe_serialization=(
                            trainer._config.TRAINING_SAFE_MODEL_SERIALISATION == "true"
                        ),
                    )
                ner_config = {
                    f"transformers.cat_config.{arg}": str(val)
                    for arg, val in ner.config.general.dict().items()
                }
                ner_config.update(
                    {
                        f"transformers.training.{arg}": str(val)
                        for arg, val in ner.training_arguments.to_dict().items()
                    }
                )
                for key, val in ner_config.items():
                    ner_config[key] = "<EMPTY>" if val == "" else val
                trainer._tracker_client.log_model_config(ner_config)
                trainer._tracker_client.log_trainer_version(medcat_version)

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
                model.config.version.description = description or model.config.version.description
                ner.config.general.description = description or ner.config.general.description
                dataset = None

                for training in range(training_params["nepochs"]):
                    if dataset is not None:
                        dataset["train"] = dataset["train"].shuffle()
                        dataset["test"] = dataset["test"].shuffle()

                    ner = MedcatDeIdentificationSupervisedTrainer._customise_training_device(
                        ner, trainer._config.DEVICE
                    )
                    eval_results, examples, dataset = ner.train(
                        data_file.name,
                        ignore_extra_labels=True,
                        dataset=dataset,
                        # trainer_callbacks=[MetricsCallback, LabelCountCallback]
                    )
                    if (training + 1) % log_frequency == 0:
                        for _, row in eval_results.iterrows():
                            normalised_name = row["name"].replace(" ", "_").lower()
                            grouped_metrics = {
                                f"{normalised_name}/precision": row["p"]
                                if row["p"] is not None
                                else np.nan,
                                f"{normalised_name}/recall": row["r"]
                                if row["r"] is not None
                                else np.nan,
                                f"{normalised_name}/f1": row["f1"]
                                if row["f1"] is not None
                                else np.nan,
                                f"{normalised_name}/p_merged": row["p_merged"]
                                if row["p_merged"] is not None
                                else np.nan,
                                f"{normalised_name}/r_merged": row["r_merged"]
                                if row["r_merged"] is not None
                                else np.nan,
                            }
                            trainer._tracker_client.send_model_stats(grouped_metrics, training)

                        mean_metrics = {
                            "precision": eval_results["p"].mean(),
                            "recall": eval_results["r"].mean(),
                            "f1": eval_results["f1"].mean(),
                            "p_merged": eval_results["p_merged"].mean(),
                            "r_merged": eval_results["r_merged"].mean(),
                        }
                        trainer._tracker_client.send_model_stats(mean_metrics, training)

                cui2names = {}
                eval_results.sort_values(by=["cui"])
                aggregated_metrics = []
                for _, row in eval_results.iterrows():
                    if row["support"] == 0:  # the concept has not been used for annotation
                        continue
                    aggregated_metrics.append(
                        {
                            "per_concept_p": row["p"] if row["p"] is not None else 0.0,
                            "per_concept_r": row["r"] if row["r"] is not None else 0.0,
                            "per_concept_f1": row["f1"] if row["f1"] is not None else 0.0,
                            "per_concept_support": row["support"]
                            if row["support"] is not None
                            else 0.0,
                            "per_concept_p_merged": row["p_merged"]
                            if row["p_merged"] is not None
                            else 0.0,
                            "per_concept_r_merged": row["r_merged"]
                            if row["r_merged"] is not None
                            else 0.0,
                        }
                    )
                    cui2names[row["cui"]] = model.cdb.get_name(row["cui"])
                MedcatDeIdentificationSupervisedTrainer._save_metrics_plot(
                    aggregated_metrics,
                    list(cui2names.values()),
                    trainer._tracker_client,
                    trainer._model_name,
                )
                trainer._tracker_client.send_batched_model_stats(aggregated_metrics, run_id)
                trainer._save_examples(examples, ["tp", "tn"])
                trainer._tracker_client.log_classes_and_names(cui2names)
                cui_counts, cui_unique_counts, cui_ignorance_counts, num_of_docs = (
                    get_stats_from_trainer_export(data_file.name)
                )
                trainer._tracker_client.log_document_size(num_of_docs)
                trainer._save_trained_concepts(
                    cui_counts, cui_unique_counts, cui_ignorance_counts, model
                )
                trainer._sanity_check_model_and_save_results(
                    data_file.name, trainer._model_service.from_model(model)
                )

                if not skip_save_model:
                    model_pack_path = trainer.save_model_pack(
                        model, trainer._retrained_models_dir, description
                    )
                    cdb_config_path = model_pack_path.replace(".zip", "_config.json")
                    model.cdb.config.save(cdb_config_path)
                    model_uri = trainer._tracker_client.save_model(
                        model_pack_path, trainer._model_name, trainer._model_manager
                    )
                    logger.info("Retrained model saved: %s", model_uri)
                    trainer._tracker_client.save_model_artifact(
                        cdb_config_path, trainer._model_name
                    )
                else:
                    logger.info("Skipped saving on the retrained model")
                if redeploy:
                    trainer.deploy_model(trainer._model_service, model, skip_save_model)
                else:
                    del model
                    gc.collect()
                    logger.info("Skipped deployment on the retrained model")
                logger.info("Supervised training finished")
                trainer._tracker_client.end_with_success()

                # Remove intermediate results folder on successful training
                results_path = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..", "results")
                )
                if results_path and os.path.isdir(results_path):
                    shutil.rmtree(results_path)
            except Exception as e:
                logger.exception("Supervised training failed")
                trainer._tracker_client.log_exceptions(e)
                trainer._tracker_client.end_with_failure()
            finally:
                data_file.close()
                with trainer._training_lock:
                    trainer._training_in_progress = False
                trainer._housekeep_file(model_pack_path)
                trainer._housekeep_file(copied_model_pack_path)
                if cdb_config_path and os.path.exists(cdb_config_path):
                    os.remove(cdb_config_path)
        else:
            try:
                logger.info("Evaluating the running model...")
                trainer._tracker_client.log_model_config(
                    trainer.get_flattened_config(trainer._model_service._model)
                )
                trainer._tracker_client.log_trainer_version(medcat_version)
                ner = trainer._model_service._model._addl_ner[0]
                ner.tokenizer.hf_tokenizer._in_target_context_manager = getattr(
                    ner.tokenizer.hf_tokenizer, "_in_target_context_manager", False
                )
                ner.tokenizer.hf_tokenizer.clean_up_tokenization_spaces = getattr(
                    ner.tokenizer.hf_tokenizer, "clean_up_tokenization_spaces", None
                )
                ner.tokenizer.hf_tokenizer.split_special_tokens = getattr(
                    ner.tokenizer.hf_tokenizer, "split_special_tokens", False
                )
                eval_results, examples = ner.eval(data_file.name)
                cui2names = {}
                eval_results.sort_values(by=["cui"])
                aggregated_metrics = []
                for _, row in eval_results.iterrows():
                    if row["support"] == 0:  # the concept has not been used for annotation
                        continue
                    aggregated_metrics.append(
                        {
                            "per_concept_p": row["p"] if row["p"] is not None else 0.0,
                            "per_concept_r": row["r"] if row["r"] is not None else 0.0,
                            "per_concept_f1": row["f1"] if row["f1"] is not None else 0.0,
                            "per_concept_support": row["support"]
                            if row["support"] is not None
                            else 0.0,
                            "per_concept_p_merged": row["p_merged"]
                            if row["p_merged"] is not None
                            else 0.0,
                            "per_concept_r_merged": row["r_merged"]
                            if row["r_merged"] is not None
                            else 0.0,
                        }
                    )
                    cui2names[row["cui"]] = trainer._model_service._model.cdb.get_name(row["cui"])
                trainer._tracker_client.send_batched_model_stats(aggregated_metrics, run_id)
                trainer._save_examples(examples, ["tp", "tn"])
                trainer._tracker_client.log_classes_and_names(cui2names)
                cui_counts, cui_unique_counts, cui_ignorance_counts, num_of_docs = (
                    get_stats_from_trainer_export(data_file.name)
                )
                trainer._tracker_client.log_document_size(num_of_docs)
                trainer._sanity_check_model_and_save_results(data_file.name, trainer._model_service)
                logger.info("Model evaluation finished")
                trainer._tracker_client.end_with_success()
            except Exception as e:
                logger.exception("Model evaluation failed")
                trainer._tracker_client.log_exceptions(e)
                trainer._tracker_client.end_with_failure()
            finally:
                data_file.close()
                with trainer._training_lock:
                    trainer._training_in_progress = False

    @staticmethod
    def _save_metrics_plot(
        metrics: List[Dict],
        concepts: List[str],
        tracker_client: tracker_client.TrackerClient,
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
    def _customise_training_device(ner: TransformersNER, device_name: str) -> TransformersNER:
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
                logger.warning(
                    "DEVICE is set to '%s' but it is not available. Using 'default' instead.",
                    device_name,
                )
        return ner
