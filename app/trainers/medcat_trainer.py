import gc
import json
import logging
import os
import re
import tempfile
import ijson
import datasets
import pandas as pd
from contextlib import redirect_stdout
from typing import TextIO, Dict, Optional, Set, List, Union, final, TYPE_CHECKING
from medcat import __version__ as medcat_version
from medcat.cat import CAT
from medcat.stats.stats import get_stats
from app.management.log_captor import LogCaptor
from app.management.model_manager import ModelManager
from app.trainers.base import SupervisedTrainer, UnsupervisedTrainer
from app.processors.data_batcher import mini_batch
from app.processors.metrics_collector import sanity_check_model_with_trainer_export, get_stats_from_trainer_export
from app.utils import (
    get_func_params_as_dict,
    non_default_device_is_available,
    get_model_data_package_extension,
    create_model_data_package,
    dump_pydantic_object_to_dict,
)
from app.domain import DatasetSplit, TrainerBackend
from app.exception import TrainingCancelledException
if TYPE_CHECKING:
    from app.model_services.medcat_model import MedCATModel

logger = logging.getLogger("cms")


class _MedcatTrainerCommon(object):

    @staticmethod
    def get_flattened_config(model: CAT, prefix: Optional[str] = None) -> Dict:
        params = {}
        prefix = "" if prefix is None else f"{prefix}."
        for key, val in model.config.general.__dict__.items():
            params[f"{prefix}general.{key}"] = str(val)
        for key, val in model.config.cdb_maker.__dict__.items():
            params[f"{prefix}cdb_maker.{key}"] = str(val)
        for key, val in model.config.annotation_output.__dict__.items():
            params[f"{prefix}annotation_output.{key}"] = str(val)
        for key, val in model.config.preprocessing.__dict__.items():
            params[f"{prefix}preprocessing.{key}"] = str(val)
        for key, val in model.config.components.ner.__dict__.items():
            params[f"{prefix}components.ner.{key}"] = str(val)
        for key, val in model.config.components.linking.__dict__.items():
            params[f"{prefix}components.linking.{key}"] = str(val)
        for key, val in params.items():
            if val == "":
                params[key] = "<EMPTY>"
        return params

    @staticmethod
    def deploy_model(
        model_service: "MedCATModel",
        model: CAT,
        skip_save_model: bool,
        description: Optional[str] = None,
    ) -> None:
        if skip_save_model:
            model._versioning(change_description=description)
        if hasattr(model_service, "model"):
            del model_service.model
        gc.collect()
        model_service.model = model
        logger.info("Retrained model deployed")

    @staticmethod
    def save_model_pack(model: CAT, model_dir: str, base_model_file: str, description: Optional[str] = None) -> str:
        logger.info("Saving retrained model to %s...", model_dir)
        model.config.meta.description = description or model.config.meta.description
        model_pack_name = model.save_model_pack(model_dir, "model")
        if get_model_data_package_extension(base_model_file) == ".tar.gz":
            model_pack_path = f"{os.path.join(model_dir, model_pack_name)}.tar.gz"
            create_model_data_package(model_dir, model_pack_path)
        else:
            model_pack_path = f"{os.path.join(model_dir, model_pack_name)}.zip"
        logger.debug("Model package saved to %s", model_pack_path)
        return model_pack_path


class MedcatSupervisedTrainer(SupervisedTrainer, _MedcatTrainerCommon):
    """
    An supervised trainer class for MedCAT clinical-coding models.

    Args:
        model_service (MedCATModel): An instance of the MedCAT service.
    """

    _model_pack_path: str
    _model_parent_dir: str

    def __init__(self, model_service: "MedCATModel") -> None:
        SupervisedTrainer.__init__(self, model_service._config, model_service.model_name)
        self._model_service = model_service
        self._model_name = model_service.model_name
        self._model_pack_path = model_service._model_pack_path
        self._retrained_models_dir = os.path.join(model_service._model_parent_dir, "retrained", self._model_name.replace(" ", "_"))
        self._model_manager = ModelManager(type(model_service), model_service._config)
        os.makedirs(self._retrained_models_dir, exist_ok=True)

    def run(
        self,
        training_params: Dict,
        data_file: TextIO,
        log_frequency: int,
        run_id: str,
        description: Optional[str] = None,
    ) -> None:
        """
        Runs the supervised training loop for MedCAT clinical-coding models.

        Args:
            training_params (Dict): A dictionary containing parameters for the training.
            data_file (TextIO): The file-like object containing the training data.
            log_frequency (int): The frequency at which logs should be recorded (e.g, the number of processed documents or finished epochs).
            run_id (str): The run ID of the training job.
            description (Optional[str]): The optional description of the training or change logs.
        """

        training_params.update({"print_stats": log_frequency})
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

                if non_default_device_is_available(self._config.DEVICE):
                    model = self._model_service.load_model(copied_model_pack_path)
                    model.config.general.device = self._config.DEVICE   # type: ignore
                else:
                    model = self._model_service.load_model(copied_model_pack_path)
                self._tracker_client.log_model_config(self.get_flattened_config(model))
                self._tracker_client.log_trainer_version(TrainerBackend.MEDCAT, medcat_version)
                cui_counts, cui_unique_counts, cui_ignorance_counts, num_of_docs = get_stats_from_trainer_export(data_file.name)
                self._tracker_client.log_document_size(num_of_docs)
                training_params.update({"extra_cui_filter": self._get_concept_filter(cui_counts, model)})
                logger.info("Performing supervised training...")
                with open(data_file.name, "r") as f:
                    training_data = json.load(f)
                train_supervised_params = get_func_params_as_dict(model.trainer.train_supervised_raw)
                train_supervised_params = {p_key: training_params[p_key] if p_key in training_params else p_val for p_key, p_val in train_supervised_params.items()}
                model.config.meta.description = description or model.config.meta.description

                with redirect_stdout(LogCaptor(self._glean_and_log_metrics)):    # type: ignore
                    fps, fns, tps, p, r, f1, cc, examples = model.trainer.train_supervised_raw(training_data, **train_supervised_params)

                # This can be removed after the returned values are fixed in medcat trainer's train_supervised_raw()
                fps, fns, tps, p, r, f1, cc, examples = get_stats(model, training_data, training_params["nepochs"])

                self._save_examples(examples, ["tp", "tn"])
                del examples
                gc.collect()
                cuis = []
                f1 = {c: f for c, f in sorted(f1.items(), key=lambda item: item[0])}
                fp_accumulated = 0
                fn_accumulated = 0
                tp_accumulated = 0
                cc_accumulated = 0
                aggregated_metrics = []
                for cui, f1_val in f1.items():
                    fp_accumulated += fps.get(cui, 0)
                    fn_accumulated += fns.get(cui, 0)
                    tp_accumulated += tps.get(cui, 0)
                    cc_accumulated += cc.get(cui, 0)
                    cui_info = model.cdb.cui2info.get(cui)
                    aggregated_metrics.append({
                        "per_concept_fp": fps.get(cui, 0),
                        "per_concept_fn": fns.get(cui, 0),
                        "per_concept_tp": tps.get(cui, 0),
                        "per_concept_counts": cc.get(cui, 0),
                        "per_concept_count_train": cui_info.get("count_train", 0) if cui_info is not None else 0,
                        "per_concept_acc_fp": fp_accumulated,
                        "per_concept_acc_fn": fn_accumulated,
                        "per_concept_acc_tp": tp_accumulated,
                        "per_concept_acc_cc": cc_accumulated,
                        "per_concept_precision": p[cui],
                        "per_concept_recall": r[cui],
                        "per_concept_f1": f1_val,
                    })
                    cuis.append(cui)
                self._tracker_client.send_batched_model_stats(aggregated_metrics, run_id)
                self._save_trained_concepts(cui_counts, cui_unique_counts, cui_ignorance_counts, model)
                self._tracker_client.log_classes(cuis)
                self._sanity_check_model_and_save_results(
                    data_file.name,
                    self._model_service.from_model(model),
                )

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
                    self.deploy_model(self._model_service, model, skip_save_model, description)
                else:
                    del model
                    gc.collect()
                    logger.info("Skipped deployment on the retrained model")
                logger.info("Supervised training finished")
                self._tracker_client.end_with_success()
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
    def _get_concept_filter(training_concepts: Dict, model: CAT) -> Set[str]:
        return set(training_concepts.keys()).intersection(set(model.cdb.cui2info.keys()))

    def _glean_and_log_metrics(self, log: str) -> None:
        metric_lines = re.findall(
            r"Epoch: (\d+), Prec: (\d+\.\d+), Rec: (\d+\.\d+), F1: (\d+\.\d+)",
            log,
            re.IGNORECASE
        )
        for step, metric in enumerate(metric_lines):
            metrics = {
                "precision": float(metric[1]),
                "recall": float(metric[2]),
                "f1": float(metric[3]),
            }
            self._tracker_client.send_model_stats(metrics, int(metric[0]))
            if self._cancel_event.is_set():
                self._cancel_event.clear()
                raise TrainingCancelledException("Training cancelled by the user")

    def _save_trained_concepts(
            self,
            training_concepts: Dict,
            training_unique_concepts: Dict,
            training_ignorance_counts: Dict,
            model: CAT,
    ) -> None:
        if len(training_concepts.keys()) != 0:
            unknown_concepts = set(training_concepts.keys()) - set(model.cdb.cui2info.keys())
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
            train_count = []
            concept_names = []
            annotation_count = []
            annotation_unique_count = []
            annotation_ignorance_count = []
            concepts = list(training_concepts.keys())
            for c in concepts:
                cui_info = model.cdb.cui2info.get(c)
                train_count.append(cui_info.get("count_train", 0) if cui_info is not None else 0)
                concept_names.append(model.cdb.get_name(c))
                annotation_count.append(training_concepts[c])
                annotation_unique_count.append(training_unique_concepts[c])
                annotation_ignorance_count.append(training_ignorance_counts[c])
            self._tracker_client.save_dataframe_as_csv(
                "trained_concepts.csv",
                pd.DataFrame({
                   "concept": concepts,
                   "name": concept_names,
                   "train_count": train_count,
                   "anno_count": annotation_count,
                   "anno_unique_count": annotation_unique_count,
                   "anno_ignorance_count": annotation_ignorance_count,
                }),
                self._model_name,
            )

    def _sanity_check_model_and_save_results(self, data_file_path: str, medcat_model: "MedCATModel") -> None:
        self._tracker_client.save_dataframe_as_csv(
            "sanity_check_result.csv",
            sanity_check_model_with_trainer_export(
                data_file_path,
                medcat_model,
                return_df=True,
                include_anchors=True,
            ),
            self._model_name,
        )

    def _save_examples(self, examples: Dict, excluded_example_keys: List) -> None:
        for e_key, e_items in examples.items():
            if e_key in excluded_example_keys:
                continue
            rows: List = []
            columns: List = []
            for concept, items in e_items.items():
                if items and not columns:
                    # Extract column names from the first row
                    columns = ["concept"] + list(items[0].keys())
                for item in items:
                    rows.append([concept] + list(item.values())[:len(columns)-1])
            if rows:
                self._tracker_client.save_dataframe_as_csv(
                    f"{e_key}_examples.csv",
                    pd.DataFrame(rows, columns=columns),
                    self._model_name,
                )


@final
class MedcatUnsupervisedTrainer(UnsupervisedTrainer, _MedcatTrainerCommon):
    """
    An unsupervised trainer class for MedCAT clinical-coding models.

    Args:
        model_service (MedCATModel): An instance of the MedCAT service.
    """

    def __init__(self, model_service: "MedCATModel") -> None:
        UnsupervisedTrainer.__init__(self, model_service._config, model_service.model_name)
        self._model_service = model_service
        self._model_name = model_service.model_name
        self._model_pack_path = model_service._model_pack_path
        self._retrained_models_dir = os.path.join(model_service._model_parent_dir, "retrained", self._model_name.replace(" ", "_"))
        self._model_manager = ModelManager(type(model_service), model_service._config)
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
        Runs the unsupervised training loop for MedCAT clinical-coding models.

        Args:
            training_params (Dict): A dictionary containing parameters for the training.
            data_file (Union[TextIO, tempfile.TemporaryDirectory]): The file-like object or temporary directory containing the training data.
            log_frequency (int): The frequency at which logs should be recorded (e.g, the number of processed documents or finished epochs).
            run_id (str): The run ID of the training job.
            description (Optional[str]): The optional description of the training or change logs.
        """

        model_pack_path = None
        cdb_config_path = None
        copied_model_pack_path = None
        redeploy = self._config.REDEPLOY_TRAINED_MODEL == "true"
        skip_save_model = self._config.SKIP_SAVE_MODEL == "true"

        if isinstance(data_file, tempfile.TemporaryDirectory):
            raw_dataset = datasets.load_from_disk(data_file.name)
            texts = raw_dataset[DatasetSplit.TRAIN.value]["text"]
        else:
            texts = ijson.items(data_file, "item")

        try:
            logger.info("Loading a new model copy for training...")
            copied_model_pack_path = self._make_model_file_copy(self._model_pack_path, run_id)
            if non_default_device_is_available(self._config.DEVICE):
                model = self._model_service.load_model(copied_model_pack_path)
                model.config.general.device = self._config.DEVICE  # type: ignore
            else:
                model = self._model_service.load_model(copied_model_pack_path)
            self._tracker_client.log_model_config(self.get_flattened_config(model))
            self._tracker_client.log_trainer_version(TrainerBackend.MEDCAT, medcat_version)
            logger.info("Performing unsupervised training...")
            step = 0
            self._tracker_client.send_model_stats(dict(model.cdb.get_basic_info()), step)
            before_cui2count_train = model.cdb.get_cui2count_train()
            num_of_docs = 0
            train_unsupervised_params = get_func_params_as_dict(model.trainer.train_unsupervised)
            train_unsupervised_params = {p_key: training_params[p_key] if p_key in training_params else p_val for p_key, p_val in train_unsupervised_params.items()}

            for batch in mini_batch(texts, batch_size=log_frequency):
                if self._cancel_event.is_set():
                    self._cancel_event.clear()
                    raise TrainingCancelledException("Training cancelled by the user")
                step += 1
                model.trainer.train_unsupervised(batch, **train_unsupervised_params)
                num_of_docs += len(batch)
                self._tracker_client.send_model_stats(dict(model.cdb.get_basic_info()), step)

            self._tracker_client.log_document_size(num_of_docs)
            after_cui2count_train = {
                c: ct
                for c, ct in sorted(
                    model.cdb.get_cui2count_train().items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            }

            aggregated_metrics = []
            cui_step = 0
            for cui, train_count in after_cui2count_train.items():
                if cui_step >= 10000:  # large numbers will cause the mlflow page to hung on loading
                    break
                cui_step += 1
                aggregated_metrics.append({
                    "per_concept_train_count_before": before_cui2count_train.get(cui, 0),
                    "per_concept_train_count_after": train_count
                })
            self._tracker_client.send_batched_model_stats(aggregated_metrics, run_id)

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
                logger.info(f"Retrained model saved: {model_uri}")
                self._tracker_client.save_model_artifact(cdb_config_path, self._model_name)
            else:
                logger.info("Skipped saving on the retrained model")
            if redeploy:
                self.deploy_model(self._model_service, model, skip_save_model, description)
            else:
                del model
                gc.collect()
                logger.info("Skipped deployment on the retrained model")
            logger.info("Unsupervised training finished")
            self._tracker_client.end_with_success()
        except TrainingCancelledException as e:
            logger.exception(e)
            logger.info("Unsupervised training was cancelled by the user")
            del model
            gc.collect()
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
            with self._training_lock:
                self._training_in_progress = False
            self._clean_up_training_cache()
            self._housekeep_file(model_pack_path)
            self._housekeep_file(copied_model_pack_path)
            if cdb_config_path and os.path.exists(cdb_config_path):
                os.remove(cdb_config_path)
