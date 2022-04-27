import os

all_models_path = os.environ.get('ALL_MODELS_PATH', 'model')
base_model_path = os.environ.get('BASE_MODEL_PATH', 'medcatbase')
base_model_pack = os.environ.get('BASE_MODEL_PACK', 'mc_modelpack_blank_snomed_full_october_2021.zip')
temp_folder = os.environ.get('TEMP_FOLDER', 'temp')
code_type = os.environ.get('CODE_TYPE', 'snomed')
