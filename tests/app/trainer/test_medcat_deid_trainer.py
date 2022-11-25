import os
from unittest.mock import create_autospec, patch
from app.config import Settings
from app.model_services.medcat_model_deid import MedCATModelDeIdentification
from app.trainer.medcat_deid_trainer import MedcatDeIdentificationSupervisedTrainer

model_service = create_autospec(MedCATModelDeIdentification,
                                _config=Settings(),
                                _model_parent_dir="model_parent_dir",
                                _enable_trainer=True,
                                _model_pack_path="model_parent_dir/mode.zip",
                                _meta_cat_config_dict={"general": {"device": "cpu"}})
supervised_trainer = MedcatDeIdentificationSupervisedTrainer(model_service)

data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "fixture")


def test_medcat_deid_supervised_trainer():
    with patch.object(supervised_trainer, "run", wraps=supervised_trainer.run) as run:
        with open(os.path.join(data_dir, "trainer_export.json"), "r") as f:
            supervised_trainer.train(f, 1, 1, "training_id", "input_file_name")
            supervised_trainer._tracker_client.end_with_success()
    run.assert_called_once()
