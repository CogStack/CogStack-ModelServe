import os
import json
import pandas as pd

from medcat.cat import CAT
from model_services import ModelServices


class NLPModel(ModelServices):

    def __init__(self, config):
        self.config = config
        model_pack_path = os.path.join(os.path.dirname(__file__), 'model', config.base_model_file)
        meta_cat_config_dict = {'general': {'device': 'cpu'}}
        self.model = self.load_model(model_pack_path, meta_cat_config_dict=meta_cat_config_dict)

    @staticmethod
    def info():
        return {'model_description': 'medmen model', 'model_type': 'medcat'}

    @staticmethod
    def load_model(model_file_path, *args, **kwargs):
        return CAT.load_model_pack(model_file_path, *args, **kwargs)

    def annotate(self, text):
        doc = self.model.get_entities(text)
        return self._get_records_from_doc(doc)

    def batch_annotate(self, texts):
        batch_size_chars = 500000

        docs = self.model.multiprocessing(self._data_iterator(texts),
                                          batch_size_chars=batch_size_chars,
                                          nproc=2)
        annotations_list = []
        for doc in docs:
            annotations_list.append(self._get_records_from_single_doc(doc))
        return annotations_list

    def train_supervised(self, annotations):

        temp_path = f'model/{self.config.temp_folder}/data.json'

        # Medcat only works with json files. Save to local dir and then retrain and delete
        with open(temp_path, 'w') as fp:
            json.dump(annotations, fp)

        self.model.train_supervised(data_path=temp_path,
                     nepochs=1,
                     reset_cui_count=False,
                     print_stats=True,
                     use_filters=True)

        data = json.load(open(temp_path))
        print(self.model._print_stats(data, extra_cui_filter=True))

    def train_unsupervised(self, texts):
        self.model.train(texts, progress_print=100)
        self.model.cdb.print_stats()

    def train_meta_models(self, annotations):
        pass

    def _get_records_from_doc(self, doc):
        df = pd.DataFrame(doc['entities'].values())

        if df.empty:
            df = pd.DataFrame(columns=['label_name', 'label_id', 'start', 'end'])
        else:
            if self.config.code_type == 'icd10':
                output = pd.DataFrame()
                for _, row in df.iterrows():
                    print(row)
                    if row['icd10']:
                        for icd10 in row['icd10']:
                            output_row = row.copy()
                            if isinstance(icd10, str):
                                output_row['icd10'] = icd10
                            else:
                                output_row['icd10'] = icd10['code']
                                output_row['pretty_name'] = icd10['name']
                            output = output.append(output_row, ignore_index=True)
                df = output
                df.rename(columns={'pretty_name': 'label_name', 'icd10': 'label_id'}, inplace=True)
            elif self.config.code_type == 'snomed':
                df.rename(columns={'pretty_name': 'label_name', 'cui': 'label_id'}, inplace=True)
            else:
                raise ValueError(f'Unknown coding type: {self.config.code_type}')
            df = self._retrieve_meta_annotations(df)
        records = df.to_dict('records')
        return records

    @staticmethod
    def _retrieve_meta_annotations(df):
        meta_annotations = []
        for i, r in df.iterrows():

            meta_dict = {}
            for k, v in r.meta_anns.items():
                meta_dict[k] = v['value']

            meta_annotations.append(meta_dict)

        df['new_meta_anns'] = meta_annotations
        return pd.concat([df.drop(['new_meta_anns'], axis=1), df['new_meta_anns'].apply(pd.Series)], axis=1)
