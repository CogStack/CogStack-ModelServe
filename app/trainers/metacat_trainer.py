import os
import logging
import shutil
import gc
from typing import Dict, TextIO, Optional
from medcat.meta_cat import MetaCAT
from trainers.medcat_trainer import MedcatSupervisedTrainer
from exception import TrainingFailedException

logger = logging.getLogger(__name__)


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
            if val == "":  # otherwise it will trigger an MLflow bug
                params[key] = "<EMPTY>"
        return params

    @staticmethod
    def run(trainer: MedcatSupervisedTrainer,
            training_params: Dict,
            data_file: TextIO,
            log_frequency: int,
            run_id: str) -> None:
        model_pack_path = None
        cdb_config_path = None
        copied_model_pack_path = None
        redeploy = trainer._config.REDEPLOY_TRAINED_MODEL == "true"
        skip_save_model = trainer._config.SKIP_SAVE_MODEL == "true"
        try:
            logger.info("Loading a new model copy for training...")
            copied_model_pack_path = trainer._make_model_file_copy(trainer._model_pack_path)
            model = trainer._model_service.load_model(copied_model_pack_path, meta_cat_config_dict=trainer._meta_cat_config_dict)

            is_retrained = False
            exceptions = []
            for meta_cat in model._meta_cats:
                category_name = meta_cat.config.general["category_name"]
                trainer._tracker_client.log_model_config(trainer.get_flattened_config(meta_cat, category_name))
                logger.info(f'Performing supervised training on category "{category_name}"...')

                for epoch in range(training_params["nepochs"]):
                    try:
                        winner_report = meta_cat.train(data_file.name, os.path.join(copied_model_pack_path.replace(".zip", ""), f"meta_{category_name}"))
                        is_retrained = True
                    except Exception as e:
                        logger.error(f"Failed on training meta model: {category_name}. This could be benign if training data has no annotations belonging to this category.")
                        logger.exception(e)
                        exceptions.append(e)
                        break
                    if (epoch + 1) % log_frequency == 0:
                        metrics = {
                            "macro_avg_precision": winner_report["report"]["macro avg"]["precision"],
                            "macro_avg_recall": winner_report["report"]["macro avg"]["recall"],
                            "macro_avg_f1": winner_report["report"]["macro avg"]["f1-score"],
                            "macro_avg_support": winner_report["report"]["macro avg"]["support"],
                            "weighted_avg_precision": winner_report["report"]["weighted avg"]["precision"],
                            "weighted_avg_recall": winner_report["report"]["weighted avg"]["recall"],
                            "weighted_avg_f1": winner_report["report"]["weighted avg"]["f1-score"],
                            "weighted_avg_support": winner_report["report"]["weighted avg"]["support"],
                        }
                        trainer._tracker_client.send_model_stats(metrics, epoch)
            trainer._tracker_client.log_exceptions(exceptions)

            if not is_retrained:
                raise TrainingFailedException("No metacat model has been retrained")

            if not skip_save_model:
                model_pack_path = trainer.save_model(model, trainer._retrained_models_dir)
                cdb_config_path = model_pack_path.replace(".zip", "_config.json")
                model.cdb.config.save(cdb_config_path)
                trainer._tracker_client.save_model(model_pack_path, trainer._model_name, trainer._model_manager)
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
            results_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "results")
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
            if cdb_config_path:
                os.remove(cdb_config_path)
