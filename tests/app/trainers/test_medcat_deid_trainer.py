import os
from unittest.mock import create_autospec, patch
from app.config import Settings
from app.model_services.medcat_model_deid import MedCATModelDeIdentification
from app.trainers.medcat_deid_trainer import MedcatDeIdentificationSupervisedTrainer
from ..utils import ensure_no_active_run

model_service = create_autospec(MedCATModelDeIdentification,
                                _config=Settings(),
                                _model_parent_dir="model_parent_dir",
                                _enable_trainer=True,
                                _model_pack_path="model_parent_dir/mode.zip",
                                _meta_cat_config_dict={"general": {"device": "cpu"}})
deid_trainer = MedcatDeIdentificationSupervisedTrainer(model_service)
deid_trainer.model_name = "deid_trainer"
data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "fixture")


def test_medcat_deid_supervised_trainer():
    ensure_no_active_run()
    with patch.object(deid_trainer, "run", wraps=deid_trainer.run) as run:
        with open(os.path.join(data_dir, "trainer_export.json"), "r") as f:
            deid_trainer.train(f, 1, 1, "training_id", "input_file_name")
            deid_trainer._tracker_client.end_with_success()
    run.assert_called_once()
