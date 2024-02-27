import logging
import pandas as pd
from typing import Dict, Optional
from model_services.medcat_model import MedCATModel
from config import Settings
from domain import ModelCard, ModelType

logger = logging.getLogger(__name__)


class MedCATModelIcd10(MedCATModel):

    ICD10_KEY = "icd10"

    def __init__(self,
                 config: Settings,
                 model_parent_dir: Optional[str] = None,
                 enable_trainer: Optional[bool] = None,
                 model_name: Optional[str] = None,
                 base_model_file: Optional[str] = None) -> None:
        super().__init__(config, model_parent_dir=model_parent_dir, enable_trainer=enable_trainer, model_name=model_name, base_model_file=base_model_file)
        self.model_name = model_name or "ICD-10 MedCAT model"

    @property
    def api_version(self) -> str:
        return "0.0.1"

    def info(self) -> ModelCard:
        return ModelCard(model_description=self.model_name,
                         model_type=ModelType.MEDCAT_ICD10,
                         api_version=self.api_version,
                         model_card=self.model.get_model_card(as_dict=True))

    def get_records_from_doc(self, doc: Dict) -> Dict:
        df = pd.DataFrame(doc["entities"].values())

        if df.empty:
            df = pd.DataFrame(columns=["label_name", "label_id", "start", "end", "accuracy"])
        else:
            new_rows = []
            for _, row in df.iterrows():
                if self.ICD10_KEY not in row or not row[self.ICD10_KEY]:
                    logger.debug(f"No mapped ICD-10 code associated with the entity: {row}")
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
                            logger.error(f"Unknown format for the ICD-10 code(s): {icd10}")
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
