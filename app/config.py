import os

base_model_file = os.environ.get("BASE_MODEL_FILE", "model.zip")
temp_folder = os.environ.get("TEMP_FOLDER", "temp")
code_type = os.environ.get("CODE_TYPE", "snomed")
