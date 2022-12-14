import os
import logging
import shutil
import gc
from typing import Dict, TextIO
from trainer.medcat_trainer import MedcatSupervisedTrainer
from processors.metrics_collector import get_cui_counts_from_trainer_export

logger = logging.getLogger(__name__)


class MedcatDeIdentificationSupervisedTrainer(MedcatSupervisedTrainer):

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
            ner = model._addl_ner[0]

            params = {f"transformers.{arg}": str(val) for arg, val in ner.training_arguments.to_dict().items()}
            for key, val in params.items():
                # Otherwise it will trigger an MLflow bug
                params[key] = "<EMPTY>" if val == "" else val
            trainer._tracker_client.log_model_config(params)

            logger.info("Performing supervised training...")
            for epoch in range(training_params["nepochs"]):
                epoch_results, _, _ = ner.train(data_file.name)
                if (epoch + 1) % log_frequency == 0:
                    metrics = {
                        "precision": epoch_results["p"].mean(),
                        "recall": epoch_results["r"].mean(),
                        "f1": epoch_results["f1"].mean(),
                    }
                    trainer._tracker_client.send_model_stats(metrics, epoch)

            logger.info("Evaluating the trained model...")
            eval_results, examples = ner.eval(data_file.name)
            cui2names = {}
            eval_results.sort_values(by=["cui"])
            aggregated_metrics = []
            for _, row in eval_results.iterrows():
                aggregated_metrics.append({
                    "per_concept_p": row["p"] if row["p"] is not None else 0.0,
                    "per_concept_r": row["r"] if row["r"] is not None else 0.0,
                    "per_concept_f1": row["f1"] if row["f1"] is not None else 0.0,
                    "per_concept_support": row["support"] if row["support"] is not None else 0.0,
                    "per_concept_p_merged": row["p_merged"] if row["p_merged"] is not None else 0.0,
                    "per_concept_r_merged": row["r_merged"] if row["r_merged"] is not None else 0.0,
                })
                cui2names[row["cui"]] = model.cdb.get_name(row["cui"])
            trainer._tracker_client.send_batched_model_stats(aggregated_metrics, run_id)
            trainer._save_examples(examples)
            trainer._tracker_client.log_classes_and_names(cui2names)
            cuis_in_data_file = get_cui_counts_from_trainer_export(data_file.name)
            trainer._save_trained_concepts(cuis_in_data_file, model)
            trainer._evaluate_model_and_save_results(data_file.name, trainer._model_service.from_model(model))
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
            logger.error(e, exc_info=True, stack_info=True)
            trainer._tracker_client.log_exception(e)
            trainer._tracker_client.end_with_failure()
        finally:
            data_file.close()
            with trainer._training_lock:
                trainer._training_in_progress = False
            trainer.housekeep_file(model_pack_path)
            trainer.housekeep_file(copied_model_pack_path)
            if cdb_config_path:
                os.remove(cdb_config_path)
