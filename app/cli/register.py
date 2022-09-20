import os
import argparse
import warnings
import uuid
import sys
import inspect

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0, parent_dir)

from config import Settings
from management.tracker import TrainingTracker
from management.model_manager import ModelManager


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    warnings.simplefilter("ignore")

    parser = argparse.ArgumentParser(
        description="This script pushes a pretrained NLP model to the Cogstack ModelServe registry",
    )

    parser.add_argument(
        "-mt",
        "--model_type",
        help="The type of the model to serve",
        choices=["medcat_snomed", "medcat_icd10", "de_id"],
    )

    parser.add_argument(
        "-mp",
        "--model_path",
        help="The file path to the model package",
        type=str,
        default="",
    )

    parser.add_argument(
        "-mn",
        "--model_name",
        help="The string representation of the registered model",
        type=str,
        default=""
    )

    args = parser.parse_args()
    config = Settings()
    training_tracker = TrainingTracker(config.MLFLOW_TRACKING_URI)

    if args.model_type == "medcat_snomed":
        from model_services.medcat_model import MedCATModel
        model_service_type = MedCATModel
    elif args.model_type == "medcat_icd10":
        from model_services.medcat_model_icd10 import MedCATModelIcd10
        model_service_type = MedCATModelIcd10
    elif args.model_type == "de_id":
        from model_services.deid_model import DeIdModel
        model_service_type = DeIdModel
    else:
        print(f"Unknown model type: {args.model_type}")
        exit(1)

    run_name = str(uuid.uuid4())
    training_tracker.save_pretrained_model(model_name=args.model_name,
                                           model_path=args.model_path,
                                           pyfunc_model=ModelManager(model_service_type, config),
                                           run_name=run_name)
    print(f"Pushed {args.model_path} as a new model version ({run_name})")
