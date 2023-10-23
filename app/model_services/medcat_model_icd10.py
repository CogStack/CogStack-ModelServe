import logging
import pandas as pd
from typing import Dict, Optional
from model_services.medcat_model import MedCATModel
from config import Settings

logger = logging.getLogger(__name__)


class MedCATModelIcd10(MedCATModel):

    ICD10_KEY = "icd10"

    def __init__(self,
                 config: Settings,
                 model_parent_dir: Optional[str] = None,
                 enable_trainer: Optional[bool] = None,
                 model_name: Optional[str] = None) -> None:
        super().__init__(config, model_parent_dir, enable_trainer)
        self.model_name = model_name or "ICD-10 MedCAT model"

    @property
    def api_version(self) -> str:
        return "0.0.1"

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
                    new_rows.append(output_row)
            if new_rows:
                df = pd.DataFrame(new_rows)
                df.rename(columns={"pretty_name": "label_name", self.ICD10_KEY: "label_id", "types": "categories", "acc": "accuracy"}, inplace=True)
                df = self._retrieve_meta_annotations(df)
            else:
                df = pd.DataFrame(columns=["label_name", "label_id", "start", "end", "accuracy"])
        records = df.to_dict("records")
        return records
