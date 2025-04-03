import logging
import pandas as pd
from typing import Dict, Optional, final, List
from app import __version__ as app_version
from app.model_services.medcat_model import MedCATModel
from app.config import Settings
from app.domain import ModelCard, ModelType

logger = logging.getLogger("cms")


@final
class MedCATModelIcd10(MedCATModel):
    """A model service for MedCAT ICD-10 models."""

    ICD10_KEY = "icd10"

    def __init__(
        self,
        config: Settings,
        model_parent_dir: Optional[str] = None,
        enable_trainer: Optional[bool] = None,
        model_name: Optional[str] = None,
        base_model_file: Optional[str] = None,
    ) -> None:
        """
        Initialises the MedCAT ICD-10 model service with specified configurations.

        Args:
            config (Settings): The configuration for the model service.
            model_parent_dir (Optional[str]): The directory where the model package is stored. Defaults to None.
            enable_trainer (Optional[bool]): The flag to enable or disable trainers. Defaults to None.
            model_name (Optional[str]): The name of the model. Defaults to None.
            base_model_file (Optional[str]): The model package file name. Defaults to None.
        """
        super().__init__(
            config,
            model_parent_dir=model_parent_dir,
            enable_trainer=enable_trainer,
            model_name=model_name,
            base_model_file=base_model_file,
        )
        self.model_name = model_name or "ICD-10 MedCAT model"

    @property
    def api_version(self) -> str:
        """Getter for the API version of the model service."""

        # APP version is used although each model service could have its own API versioning
        return app_version

    def info(self) -> ModelCard:
        """
        Retrieves information about the MedCAT ICD-10 model.

        Returns:
            ModelCard: A card containing information about the MedCAT ICD-10 model.
        """

        return ModelCard(
            model_description=self.model_name,
            model_type=ModelType.MEDCAT_ICD10,
            api_version=self.api_version,
            model_card=self.model.get_model_card(as_dict=True),
        )

    def get_records_from_doc(self, doc: Dict) -> List[Dict]:
        """
        Extracts and formats entity records from a document dictionary.

        Args:
            doc (Dict): The document dictionary containing extracted named entities.

        Returns:
            List[Dict]: A list of formatted entity records.
        """

        df = pd.DataFrame(doc["entities"].values())

        if df.empty:
            df = pd.DataFrame(columns=["label_name", "label_id", "start", "end", "accuracy"])
        else:
            new_rows = []
            for _, row in df.iterrows():
                if self.ICD10_KEY not in row or not row[self.ICD10_KEY]:
                    logger.debug("No mapped ICD-10 code associated with the entity: %s", row)
                else:
                    for icd10 in row[self.ICD10_KEY]:
                        output_row = row.copy()
                        if isinstance(icd10, str):
                            output_row[self.ICD10_KEY] = icd10
                        elif isinstance(icd10, dict):
                            output_row[self.ICD10_KEY] = icd10.get("code")
                            output_row["pretty_name"] = icd10.get("name")
                        elif isinstance(icd10, list) and icd10:
                            output_row[self.ICD10_KEY] = icd10[-1]
                        else:
                            logger.error("Unknown format for the ICD-10 code(s): %s", icd10)
                        if "athena_ids" in output_row and output_row["athena_ids"]:
                            output_row["athena_ids"] = [athena_id["code"] for athena_id in output_row["athena_ids"]]
                    new_rows.append(output_row)
            if new_rows:
                df = pd.DataFrame(new_rows)
                df.rename(columns={"pretty_name": "label_name", self.ICD10_KEY: "label_id", "types": "categories", "acc": "accuracy", "athena_ids": "athena_ids"}, inplace=True)
                df = self._retrieve_meta_annotations(df)
            else:
                df = pd.DataFrame(columns=["label_name", "label_id", "start", "end", "accuracy"])
        records = df.to_dict("records")
        return records
