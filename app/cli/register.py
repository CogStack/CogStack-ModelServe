import argparse
import warnings
import uuid

from parent_dir import parent_dir # noqa

from config import Settings
from management.tracker_client import TrackerClient
from management.model_manager import ModelManager
from domain import ModelType


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    warnings.simplefilter("ignore")

    parser = argparse.ArgumentParser(
        description="This script pushes a pretrained NLP model to the Cogstack ModelServe registry",
    )

    parser.add_argument(
        "-mt",
        "--model-type",
        help="The type of the model to serve",
        choices=["medcat_snomed", "medcat_icd10", "de_id"],
    )

    parser.add_argument(
        "-mp",
        "--model-path",
        help="The file path to the model package",
        type=str,
        default="",
    )

    parser.add_argument(
        "-mn",
        "--model-name",
        help="The string representation of the registered model",
        type=str,
        default=""
    )

    args = parser.parse_args()
    config = Settings()
    tracker_client = TrackerClient(config.MLFLOW_TRACKING_URI)

    if args.model_type == ModelType.MEDCAT_SNOMED.value:
        from model_services.medcat_model import MedCATModel
        model_service_type = MedCATModel
    elif args.model_type == ModelType.MEDCAT_ICD10.value:
        from model_services.medcat_model_icd10 import MedCATModelIcd10
        model_service_type = MedCATModelIcd10
    elif args.model_type == ModelType.DE_ID.value:
        from model_services.deid_model import DeIdModel
        model_service_type = DeIdModel
    else:
        print(f"Unknown model type: {args.model_type}")
        exit(1)

    run_name = str(uuid.uuid4())
    tracker_client.save_pretrained_model(model_name=args.model_name,
                                         model_path=args.model_path,
                                         pyfunc_model=ModelManager(model_service_type, config),
                                         run_name=run_name)
    print(f"Pushed {args.model_path} as a new model version ({run_name})")
