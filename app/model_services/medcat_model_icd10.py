import logging
import pandas as pd
from typing import Dict
from model_services.medcat_model import MedCATModel
from domain import ModelCard

logger = logging.getLogger(__name__)


class MedCATModelIcd10(MedCATModel):

    @property
    def model_name(self) -> str:
        return "ICD-10 MedCAT model"

    @property
    def api_version(self) -> str:
        return "0.0.1"

    def info(self) -> ModelCard:
        return ModelCard(model_description=self.model_name,
                         model_type="MedCAT",
                         api_version=self.api_version,
                         model_card=self.model.get_model_card(as_dict=True))

    def get_records_from_doc(self, doc: Dict) -> Dict:
        df = pd.DataFrame(doc["entities"].values())

        if df.empty:
            df = pd.DataFrame(columns=["label_name", "label_id", "start", "end"])
        else:
            output = pd.DataFrame()
            for _, row in df.iterrows():
                if "icd10" not in row:
                    logger.error("No mapped ICD-10 code found in the record")
                if row["icd10"]:
                    for icd10 in row["icd10"]:
                        output_row = row.copy()
                        if isinstance(icd10, str):
                            output_row["icd10"] = icd10
                        else:
                            output_row["icd10"] = icd10["code"]
                            output_row["pretty_name"] = icd10["name"]
                        output = output.append(output_row, ignore_index=True)
            df = output
            df.rename(columns={"pretty_name": "label_name", "icd10": "label_id"}, inplace=True)
            df = self._retrieve_meta_annotations(df)
        records = df.to_dict("records")
        return records
