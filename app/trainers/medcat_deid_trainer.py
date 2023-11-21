import os
import logging
import shutil
import gc
import pandas as pd
from typing import Dict, TextIO
from medcat import __version__ as medcat_version
from trainers.medcat_trainer import MedcatSupervisedTrainer
from processors.metrics_collector import get_stats_from_trainer_export

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
        eval_mode = training_params["nepochs"] == 0
        trainer._tracker_client.log_trainer_mode(not eval_mode)
        if not eval_mode:
            try:
                logger.info("Loading a new model copy for training...")
                copied_model_pack_path = trainer._make_model_file_copy(trainer._model_pack_path)
                model = trainer._model_service.load_model(copied_model_pack_path, meta_cat_config_dict=trainer._meta_cat_config_dict)
                ner = model._addl_ner[0]

                ner_config = {f"transformers.cat_config.{arg}": str(val) for arg, val in ner.config.general.dict().items()}
                ner_config.update({f"transformers.training.{arg}": str(val) for arg, val in ner.training_arguments.to_dict().items()})
                for key, val in ner_config.items():
                    ner_config[key] = "<EMPTY>" if val == "" else val
                trainer._tracker_client.log_model_config(ner_config)
                trainer._tracker_client.log_trainer_version(medcat_version)

                eval_results: pd.DataFrame = None
                examples = None
                ner.training_arguments.num_train_epochs = 1
                logger.info("Performing supervised training...")
                dataset = None
                for training in range(training_params["nepochs"]):
                    if dataset is not None:
                        dataset["train"] = dataset["train"].shuffle()
                        dataset["test"] = dataset["test"].shuffle()
                    eval_results, examples, dataset = ner.train(data_file.name, ignore_extra_labels=True, dataset=dataset)
                    if (training + 1) % log_frequency == 0:
                        metrics = {
                            "precision": eval_results["p"].mean(),
                            "recall": eval_results["r"].mean(),
                            "f1": eval_results["f1"].mean(),
                            "p_merged": eval_results["p_merged"].mean(),
                            "r_merged": eval_results["r_merged"].mean(),
                        }
                        trainer._tracker_client.send_model_stats(metrics, training)

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
                    cui2names[row["cui"]] = model.cdb.get_name(row["cui"])
                trainer._tracker_client.send_batched_model_stats(aggregated_metrics, run_id)
                trainer._save_examples(examples, ["tp", "tn"])
                trainer._tracker_client.log_classes_and_names(cui2names)
                cui_counts, cui_unique_counts, cui_ignorance_counts, num_of_docs = get_stats_from_trainer_export(data_file.name)
                trainer._tracker_client.log_document_size(num_of_docs)
                trainer._save_trained_concepts(cui_counts, cui_unique_counts, cui_ignorance_counts, model)
                trainer._evaluate_model_and_save_results(data_file.name, trainer._model_service.from_model(model))

                if not skip_save_model:
                    model_pack_path = trainer.save_model_pack(model, trainer._retrained_models_dir)
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
                if cdb_config_path:
                    os.remove(cdb_config_path)
        else:
            try:
                logger.info("Evaluating the running model...")
                trainer._tracker_client.log_model_config(trainer.get_flattened_config(trainer._model_service._model))
                trainer._tracker_client.log_trainer_version(medcat_version)
                eval_results, examples = trainer._model_service._model._addl_ner[0].eval(data_file.name)
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
                    cui2names[row["cui"]] = trainer._model_service._model.cdb.get_name(row["cui"])
                trainer._tracker_client.send_batched_model_stats(aggregated_metrics, run_id)
                trainer._save_examples(examples, ["tp", "tn"])
                trainer._tracker_client.log_classes_and_names(cui2names)
                cui_counts, cui_unique_counts, cui_ignorance_counts, num_of_docs = get_stats_from_trainer_export(data_file.name)
                trainer._tracker_client.log_document_size(num_of_docs)
                trainer._evaluate_model_and_save_results(data_file.name, trainer._model_service)
                logger.info("Model evaluation finished")
                trainer._tracker_client.end_with_success()
            except Exception as e:
                logger.error("Model evaluation failed")
                logger.exception(e)
                trainer._tracker_client.log_exceptions(e)
                trainer._tracker_client.end_with_failure()
            finally:
                data_file.close()
                with trainer._training_lock:
                    trainer._training_in_progress = False
