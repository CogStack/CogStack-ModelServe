import os
import logging
import shutil
import gc
import torch
from typing import Dict, TextIO, Optional, List

import pandas as pd
from medcat import __version__ as medcat_version
from medcat.meta_cat import MetaCAT
from trainers.medcat_trainer import MedcatSupervisedTrainer
from exception import TrainingFailedException

logger = logging.getLogger("cms")


class MetacatTrainer(MedcatSupervisedTrainer):

    @staticmethod
    def get_flattened_config(model: MetaCAT, prefix: Optional[str] = None) -> Dict:
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

    @staticmethod
    def run(trainer: "MetacatTrainer",
            training_params: Dict,
            data_file: TextIO,
            log_frequency: int,
            run_id: str,
            description: Optional[str] = None) -> None:
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
                copied_model_pack_path = trainer._make_model_file_copy(trainer._model_pack_path, run_id)
                if (trainer._config.DEVICE.startswith("cuda") and torch.cuda.is_available()) or \
                   (trainer._config.DEVICE.startswith("mps") and torch.backends.mps.is_available()) or \
                   (trainer._config.DEVICE.startswith("cpu")):
                    model = trainer._model_service.load_model(copied_model_pack_path,
                                                              meta_cat_config_dict={"general": {"device": trainer._config.DEVICE}})
                    model.config.general["device"] = trainer._config.DEVICE
                else:
                    model = trainer._model_service.load_model(copied_model_pack_path)
                is_retrained = False
                model.config.version.description = description or model.config.version.description
                for meta_cat in model._meta_cats:
                    category_name = meta_cat.config.general["category_name"]
                    if training_params.get("lr_override") is not None:
                        meta_cat.config.train.lr = training_params["lr_override"]
                    if training_params.get("test_size") is not None:
                        meta_cat.config.train.test_size = training_params["test_size"]
                    meta_cat.config.train.nepochs = training_params["nepochs"]
                    trainer._tracker_client.log_model_config(trainer.get_flattened_config(meta_cat, category_name))
                    trainer._tracker_client.log_trainer_version(medcat_version)
                    logger.info('Performing supervised training on category "%s"...', category_name)

                    try:
                        winner_report = meta_cat.train(data_file.name, os.path.join(copied_model_pack_path.replace(".zip", ""), f"meta_{category_name}"))
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
                        trainer._tracker_client.send_model_stats(report_stats, winner_report["epoch"])
                    except Exception as e:
                        logger.error("Failed on training meta model: %s. This could be benign if training data has no annotations belonging to this category.", category_name)
                        logger.exception(e)
                        trainer._tracker_client.log_exceptions(e)

                if not is_retrained:
                    raise TrainingFailedException(
                        "No metacat model has been retrained. Double-check the presence of metacat models and your annotations.")

                if not skip_save_model:
                    model_pack_path = trainer.save_model_pack(model, trainer._retrained_models_dir, description)
                    cdb_config_path = model_pack_path.replace(".zip", "_config.json")
                    model.cdb.config.save(cdb_config_path)
                    model_uri = trainer._tracker_client.save_model(model_pack_path, trainer._model_name, trainer._model_manager)
                    logger.info("Retrained model saved: %s", model_uri)
                    trainer._tracker_client.save_model_artifact(cdb_config_path, trainer._model_name)
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
                results_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
                if results_path and os.path.isdir(results_path):
                    shutil.rmtree(results_path)
            except Exception as e:
                logger.error("Supervised training failed")
                logger.exception(e)
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
                metrics: List[Dict] = []
                for meta_cat in trainer._model_service._model._meta_cats:
                    category_name = meta_cat.config.general["category_name"]
                    trainer._tracker_client.log_model_config(trainer.get_flattened_config(meta_cat, category_name))
                    trainer._tracker_client.log_trainer_version(medcat_version)
                    result = meta_cat.eval(data_file.name)
                    metrics.append({"precision": result.get("precision"), "recall": result.get("recall"), "f1": result.get("f1")})

                if metrics:
                    trainer._tracker_client.save_dataframe_as_csv("sanity_check_result.csv",
                                                                  pd.DataFrame(metrics, columns=["category", "precision", "recall", "f1"]),
                                                                  trainer._model_service._model_name)
                else:
                    raise TrainingFailedException(
                        "No metacat model has been evaluated. Double-check the presence of metacat models and your annotations.")
                trainer._tracker_client.end_with_success()
                logger.info("Model evaluation finished")
            except Exception as e:
                logger.error("Model evaluation failed")
                logger.exception(e)
                trainer._tracker_client.log_exceptions(e)
                trainer._tracker_client.end_with_failure()
            finally:
                data_file.close()
                with trainer._training_lock:
                    trainer._training_in_progress = False
