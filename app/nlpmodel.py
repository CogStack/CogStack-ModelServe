from medcat.vocab import Vocab
from medcat.cdb import CDB
from medcat.cat import CAT
from medcat.meta_cat import MetaCAT
import pandas as pd
import config
import json


class modelrunner:

    def __init__(self):

        model_pack_path = f'{config.all_models_path}/{config.base_model_path}'

        meta_cat_config_dict = {'general': {'device': 'cpu'}}
        self.model  = CAT.load_model_pack(model_pack_path, meta_cat_config_dict=meta_cat_config_dict)


    def annotate(self, text):
        doc = self.model.get_entities(text)
        df = pd.DataFrame(doc['entities'].values())

        if df.empty:
            df = pd.DataFrame(columns=['label_name', 'label_id', 'start', 'end'])
        else:
            df.rename(columns={'pretty_name': 'label_name', 'cui': 'label_id'}, inplace=True)
            df = self.retrievemetannotations(df)
        records = df.to_dict('records')

        return records

    def retrievemetannotations(self, df):

        meta_annotations = []
        for i,r in df.iterrows():
            
            meta_dict = {}
            for k,v in r.meta_anns.items():
                meta_dict[k] = v['value']
            
            meta_annotations.append(meta_dict)
        
        df['new_meta_anns'] = meta_annotations
        print(df)
        return pd.concat([df.drop(['new_meta_anns'], axis=1), df['new_meta_anns'].apply(pd.Series)], axis=1)


    def _data_iterator(self, texts):
        for idx, text in enumerate(texts):
            yield (id, text)

    def batchannotate(self, texts):
        batch_size_chars = 500000

        results = self.model.multiprocessing(self._data_iterator(texts), 
                              batch_size_chars = batch_size_chars,
                              nproc=2)

        print(results)

    def trainsupervised(self, annotations): 

        temp_path = f'{config.all_models_path}/{config.temp_folder}/data.json'

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

    def trainunsupervised(self, texts):

        self.model.train(texts, progress_print=100)
        self.model.cdb.print_stats()
        

    def trainmetamodels(self, annotations):
        pass

