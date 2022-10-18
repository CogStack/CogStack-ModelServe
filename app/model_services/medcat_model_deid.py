import os
import logging
import gc
from typing import Dict, List, TextIO
from model_services.medcat_model import MedCATModel
from domain import ModelCard

logger = logging.getLogger(__name__)


class MedCATModelDeIdentification(MedCATModel):

    def info(self) -> ModelCard:
        return ModelCard(model_description="De-Identification MedCAT model",
                         model_type="MedCAT",
                         api_version=self.api_version,
                         model_card=self.model.get_model_card(as_dict=True))

    def batch_annotate(self, texts: List[str]) -> List[List[Dict]]:
        annotation_list = []
        for text in texts:
            annotation_list.append(self.annotate(text))
        return annotation_list

    @staticmethod
    def _train_supervised(medcat_model: "MedCATModel",
                          training_params: Dict,
                          data_file: TextIO,
                          log_frequency: int) -> None:
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
                results, _, _ = ner.train(data_file.name)
                metrics = {
                    "precision": results["p"].mean(),
                    "recall": results["r"].mean(),
                    "f1": results["f1"].mean(),
                }
                medcat_model._tracker_client.send_model_stats(metrics, epoch)

            class_id = 0
            cui2names = {}
            results.sort_values(by=["cui"])
            for _, row in results.iterrows():
                metrics = {
                    "per_concept_p": row["p"],
                    "per_concept_r": row["r"],
                    "per_concept_f1": row["f1"],
                    "per_concept_support": row["support"],
                    "per_concept_p_merged": row["p_merged"],
                    "per_concept_r_merged": row["r_merged"],
                }
                medcat_model._tracker_client.send_model_stats(metrics, class_id)
                cui2names[row["cui"]] = model.cdb.cui2preferred_name[row["cui"]]
                class_id += 1
            medcat_model._tracker_client.log_classes_and_names(cui2names)
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
            if results_path and os.path.exists(results_path):
                os.remove(results_path)
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
