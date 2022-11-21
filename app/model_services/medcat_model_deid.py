import os
import shutil
import logging
import gc
from typing import Dict, List, TextIO
from model_services.medcat_model import MedCATModel
from processors.metrics_collector import evaluate_model_with_trainer_export, get_cui_counts_from_trainer_export
from domain import ModelCard

logger = logging.getLogger(__name__)


class MedCATModelDeIdentification(MedCATModel):

    @property
    def model_name(self) -> str:
        return "De-Identification MedCAT model"

    @property
    def api_version(self) -> str:
        return "0.0.1"

    def info(self) -> ModelCard:
        return ModelCard(model_description=self.model_name,
                         model_type="MedCAT",
                         api_version=self.api_version)

    def batch_annotate(self, texts: List[str]) -> List[List[Dict]]:
        annotation_list = []
        for text in texts:
            annotation_list.append(self.annotate(text))
        return annotation_list

    @staticmethod
    def _train_supervised(medcat_model: "MedCATModel",
                          training_params: Dict,
                          data_file: TextIO,
                          log_frequency: int,
                          run_id: str) -> None:
        model_pack_path = None
        cdb_config_path = None
        copied_model_pack_path = None
        redeploy = medcat_model._config.REDEPLOY_TRAINED_MODEL == "true"
        skip_save_model = medcat_model._config.SKIP_SAVE_MODEL == "true"
        try:
            logger.info("Loading a new model copy for training...")
            copied_model_pack_path = medcat_model._make_model_file_copy(medcat_model._model_pack_path)
            model = medcat_model.load_model(copied_model_pack_path,
                                            meta_cat_config_dict=medcat_model._meta_cat_config_dict)
            ner = model._addl_ner[0]

            params = {f"transformers.{arg}": str(val) for arg, val in ner.training_arguments.to_dict().items()}
            for key, val in params.items():
                # Otherwise it will trigger an MLflow bug
                params[key] = "<EMPTY>" if val == "" else val
            medcat_model._tracker_client.log_model_config(params)

            logger.info("Performing supervised training...")
            for epoch in range(training_params["nepochs"]):
                epoch_results, _, _ = ner.train(data_file.name)
                metrics = {
                    "precision": epoch_results["p"].mean(),
                    "recall": epoch_results["r"].mean(),
                    "f1": epoch_results["f1"].mean(),
                }
                medcat_model._tracker_client.send_model_stats(metrics, epoch)

            logger.info("Evaluating the trained model...")
            eval_results, examples = ner.eval(data_file.name)
            cui2names = {}
            eval_results.sort_values(by=["cui"])
            aggregated_metrics = []
            for _, row in eval_results.iterrows():
                aggregated_metrics.append({
                    "per_concept_p": row["p"],
                    "per_concept_r": row["r"],
                    "per_concept_f1": row["f1"],
                    "per_concept_support": row["support"],
                    "per_concept_p_merged": row["p_merged"],
                    "per_concept_r_merged": row["r_merged"],
                })
                cui2names[row["cui"]] = model.cdb.cui2preferred_name[row["cui"]]
            medcat_model._tracker_client.send_batched_model_stats(aggregated_metrics, run_id)
            for e_key, e_items in examples.items():
                medcat_model._tracker_client.save_dict(f"{e_key}_examples.json", e_items, medcat_model.model_name)
            medcat_model._tracker_client.log_classes_and_names(cui2names)
            cuis_in_data_file = get_cui_counts_from_trainer_export(data_file.name)
            medcat_model._save_trained_concepts(cuis_in_data_file, medcat_model)
            medcat_model._evaluate_model_and_save_results(data_file.name, medcat_model.of(model))
            if not skip_save_model:
                model_pack_path = MedCATModel._save_model(medcat_model, model)
                cdb_config_path = model_pack_path.replace(".zip", "_config.json")
                model.cdb.config.save(cdb_config_path)
                medcat_model._tracker_client.save_model(model_pack_path,
                                                        medcat_model.info().model_description,
                                                        medcat_model._pyfunc_model)
                medcat_model._tracker_client.save_model_artifact(cdb_config_path,
                                                                 medcat_model.info().model_description)
            else:
                logger.info("Skipped saving on the retrained model")
            if redeploy:
                MedCATModel._deploy_model(medcat_model, model, skip_save_model)
            else:
                del model
                gc.collect()
                logger.info("Skipped deployment on the retrained model")
            logger.info("Supervised training finished")
            medcat_model._tracker_client.end_with_success()

            # Remove intermediate results folder on successful training
            results_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "results")
            if results_path and os.path.isdir(results_path):
                shutil.rmtree(results_path)
        except Exception as e:
            logger.error("Supervised training failed")
            logger.error(e, exc_info=True, stack_info=True)
            medcat_model._tracker_client.log_exception(e)
            medcat_model._tracker_client.end_with_failure()
        finally:
            data_file.close()
            with medcat_model._training_lock:
                medcat_model._training_in_progress = False
            medcat_model._housekeep_file(model_pack_path)
            medcat_model._housekeep_file(copied_model_pack_path)
            if cdb_config_path:
                os.remove(cdb_config_path)

    @staticmethod
    def _evaluate_model_and_save_results(data_file_path: str, medcat_model: "MedCATModel") -> None:
        medcat_model._tracker_client.save_dataframe("evaluation.csv",
                                                    evaluate_model_with_trainer_export(data_file_path, medcat_model, return_df=True),
                                                    medcat_model.model_name)
