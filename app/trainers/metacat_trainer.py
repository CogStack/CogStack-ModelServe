import os
import json
import logging
import shutil
import gc
import pandas as pd
from typing import Dict, TextIO, Optional, List
from medcat import __version__ as medcat_version
from medcat.components.addons.meta_cat.meta_cat import MetaCAT, MetaCATAddon
from app.domain import TrainerBackend
from app.trainers.medcat_trainer import MedcatSupervisedTrainer
from app.exception import TrainingFailedException, TrainingCancelledException
from app.utils import non_default_device_is_available, get_model_data_package_extension, dump_pydantic_object_to_dict

logger = logging.getLogger("cms")


class MetacatTrainer(MedcatSupervisedTrainer):
    """
    An supervised trainer class for MetaCAT models (will be deprecated).

    Args:
        model_service (MetaCAT): An instance of the MetaCAT service.
    """

    @staticmethod
    def get_flattened_metacat_config(model: MetaCAT, prefix: Optional[str] = None) -> Dict:
        """
        Flattens the configuration of a MetaCAT model into a dictionary with string values.

        Args:
            model (MetaCAT): The MetaCAT model instance whose configuration is to be flattened.
            prefix (Optional[str]): An optional prefix to prepend to each configuration key.

        Returns:
            Dict: A dictionary containing the flattened configuration. If a value is an empty string, it is replaced with "<EMPTY>".
        """

        params = {}
        prefix = "" if prefix is None else f"{prefix}."
        for key, val in model.config.general.__dict__.items():
            params[f"{prefix}general.{key}"] = str(val)
        for key, val in model.config.model.__dict__.items():
            params[f"{prefix}model.{key}"] = str(val)
        for key, val in model.config.train.__dict__.items():
            params[f"{prefix}train.{key}"] = str(val)
        for key, val in params.items():
            if val == "":
                params[key] = "<EMPTY>"
        return params

    def run(
        self,
        training_params: Dict,
        data_file: TextIO,
        log_frequency: int,
        run_id: str,
        description: Optional[str] = None,
    ) -> None:
        """
        Runs the supervised training loop for MetaCAT models.

        Args:
            training_params (Dict): A dictionary containing parameters for the training.
            data_file (TextIO): The file-like object containing the training data.
            log_frequency (int): The frequency at which logs should be recorded, currently not used.
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
                if non_default_device_is_available(self._config.DEVICE):
                    model = self._model_service.load_model(copied_model_pack_path)
                    model.config.general.device = self._config.DEVICE   # type: ignore
                else:
                    model = self._model_service.load_model(copied_model_pack_path)
                is_retrained = False
                model.config.meta.description = description or model.config.meta.description
                meta_cat_addons = model.get_addons_of_type(MetaCATAddon)
                for meta_cat_addon in meta_cat_addons:
                    if self._cancel_event.is_set():
                        self._cancel_event.clear()
                        raise TrainingCancelledException("Training was cancelled by the user")

                    meta_cat = meta_cat_addon.mc
                    category_name = meta_cat.config.general.category_name
                    assert category_name is not None, "Category name should not be None"
                    if meta_cat.config.general.alternative_class_names == [[]]:
                        class_name_mapping =  {
                            "Temporality": [["Past"], ["Recent", "Present"], ["Future"]],
                            "Time": [["Past"], ["Recent", "Present"], ["Future"]],
                            "Experiencer": [["Family"], ["Other"], ["Patient"]],
                            "Subject": [["Family"], ["Other"], ["Patient"]],
                            "Presence": [["Hypothetical (N/A)", "Hypothetical"], ["Not present (False)", "False"], ["Present (True)", "True"]],
                            "Status": [["Affirmed", "Confirmed"], ["Other"]],
                        }
                        meta_cat.config.general.alternative_class_names = class_name_mapping[category_name]

                    if training_params.get("lr_override") is not None:
                        meta_cat.config.train.lr = training_params["lr_override"]
                    if training_params.get("test_size") is not None:
                        meta_cat.config.train.test_size = training_params["test_size"]
                    meta_cat.config.train.nepochs = training_params["nepochs"]
                    self._tracker_client.log_model_config(self.get_flattened_metacat_config(meta_cat, category_name))
                    self._tracker_client.log_trainer_version(TrainerBackend.MEDCAT, medcat_version)
                    logger.info('Performing supervised training on category "%s"...', category_name)

                    try:
                        mp_ext = get_model_data_package_extension(copied_model_pack_path)
                        if non_default_device_is_available(self._config.DEVICE):
                            meta_cat.config.general.device = self._config.DEVICE
                        winner_report = meta_cat.train_from_json(
                            data_file.name,
                            os.path.join(copied_model_pack_path.replace(mp_ext, ""),f"meta_{category_name}"),
                        )
                        is_retrained = True
                        report_stats = {
                            f"{category_name}_macro_avg_precision": winner_report["report"]["macro avg"]["precision"],
                            f"{category_name}_macro_avg_recall": winner_report["report"]["macro avg"]["recall"],
                            f"{category_name}_macro_avg_f1": winner_report["report"]["macro avg"]["f1-score"],
                            f"{category_name}_macro_avg_support": winner_report["report"]["macro avg"]["support"],
                            f"{category_name}_weighted_avg_precision": winner_report["report"]["weighted avg"]["precision"],
                            f"{category_name}_weighted_avg_recall": winner_report["report"]["weighted avg"]["recall"],
                            f"{category_name}_weighted_avg_f1": winner_report["report"]["weighted avg"]["f1-score"],
                            f"{category_name}_weighted_avg_support": winner_report["report"]["weighted avg"]["support"],
                        }
                        self._tracker_client.send_model_stats(report_stats, winner_report["epoch"])
                    except Exception as e:
                        logger.exception("Failed on training meta model: %s. This could be benign if training data has no annotations belonging to this category.", category_name)
                        self._tracker_client.log_exceptions(e)

                if not is_retrained:
                    exception = TrainingFailedException("No metacat model has been retrained. Double-check the presence of metacat models and your annotations.")
                    logger.error("Error occurred while retraining the model: %s", exception, exc_info=True)
                    self._tracker_client.log_exceptions(exception)
                    self._tracker_client.end_with_failure()
                    return

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

                # Remove intermediate results folder on successful training
                results_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
                if results_path and os.path.isdir(results_path):
                    shutil.rmtree(results_path)
        else:
            try:
                logger.info("Evaluating the running model...")
                metrics: List[Dict] = []
                assert self._model_service.model is not None, "Model should not be None"
                meta_cat_addons = self._model_service.model.get_addons_of_type(MetaCATAddon)
                for meta_cat_addon in meta_cat_addons:
                    meta_cat = meta_cat_addon.mc
                    category_name = meta_cat.config.general.category_name
                    self._tracker_client.log_model_config(self.get_flattened_metacat_config(meta_cat, category_name))
                    self._tracker_client.log_trainer_version(TrainerBackend.MEDCAT, medcat_version)
                    result = meta_cat.eval(data_file.name)
                    metrics.append({"precision": result.get("precision"), "recall": result.get("recall"), "f1": result.get("f1")})

                if metrics:
                    self._tracker_client.save_dataframe_as_csv(
                        "sanity_check_result.csv",
                        pd.DataFrame(metrics, columns=["category", "precision", "recall", "f1"]),
                        self._model_service._model_name,
                    )
                    self._tracker_client.end_with_success()
                    logger.info("Model evaluation finished")
                else:
                    exception = TrainingFailedException("No metacat model has been evaluated. Double-check the presence of metacat models and your annotations.")
                    logger.error("Error occurred while evaluating the model: %s", exception, exc_info=True)
                    self._tracker_client.log_exceptions(exception)
                    self._tracker_client.end_with_failure()
                    return
            except Exception as e:
                logger.exception("Model evaluation failed")
                self._tracker_client.log_exceptions(e)
                self._tracker_client.end_with_failure()
            finally:
                data_file.close()
                with self._training_lock:
                    self._training_in_progress = False
