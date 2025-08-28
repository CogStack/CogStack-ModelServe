import os
import json
import tempfile
import uuid
import ijson
import logging
import datasets
from typing import List, Tuple, Union, cast
from typing_extensions import Annotated

from fastapi import APIRouter, Depends, UploadFile, Query, Request, File, Form
from fastapi.responses import JSONResponse
from starlette.status import HTTP_202_ACCEPTED, HTTP_503_SERVICE_UNAVAILABLE
import app.api.globals as cms_globals
from app.api.dependencies import validate_tracking_id
from app.domain import Tags, ModelType
from app.model_services.base import AbstractModelService
from app.utils import get_settings, unpack_model_data_package
from app.exception import ConfigurationException, ClientException

router = APIRouter()
logger = logging.getLogger("cms")

assert cms_globals.props is not None, "Current active user dependency not injected"
assert cms_globals.model_service_dep is not None, "Model service dependency not injected"

@router.post(
    "/train_unsupervised",
    status_code=HTTP_202_ACCEPTED,
    response_class=JSONResponse,
    tags=[Tags.Training.name],
    dependencies=[Depends(cms_globals.props.current_active_user)],
    description="Upload one or more files each containing a list of plain texts and trigger the unsupervised training",
)
async def train_unsupervised(
    request: Request,
    training_data: Annotated[List[UploadFile], File(description="One or more files to be uploaded and each contains a list of plain texts, in the format of [\"text_1\", \"text_2\", ..., \"text_n\"]")],
    epochs: Annotated[int, Query(description="The number of training epochs", ge=0)] = 1,
    lr_override: Annotated[Union[float, None], Query(description="The override of the initial learning rate", gt=0.0)] = None,
    test_size: Annotated[Union[float, None], Query(description="The override of the test size in percentage", ge=0.0)] = 0.2,
    log_frequency: Annotated[int, Query(description="The number of processed documents or epochs after which training metrics will be logged", ge=1)] = 1000,
    description: Annotated[Union[str, None], Form(description="The description of the training or change logs")] = None,
    tracking_id: Union[str, None] = Depends(validate_tracking_id),
    model_service: AbstractModelService = Depends(cms_globals.model_service_dep),
) -> JSONResponse:
    """
    Triggers unsupervised training on the running model using one or more files, each containing a list of plain texts.

    Args:
        request (Request): The request object.
        training_data (List[UploadFile]): A list of files uploaded, each containing a list of plain texts in the format of [\"text_1\", \"text_2\", ..., \"text_n\"].
        epochs (int): The number of training epochs to perform. Defaults to 1.
        lr_override (float, optional): The override of the initial learning rate. Defaults to the value used in previous training and must be greater than 0.0.
        test_size (float, optional): An override of the test size in percentage. Defaults to 0.2.
        log_frequency (int): The number of processed documents or epochs after which training metrics will be logged. Must be at least 1.
        description (str, optional): A description of the training or change logs. Defaults to empty.
        tracking_id (str, optional): n optional tracking ID of the requested task.
        model_service (AbstractModelService): The model service dependency.

    Returns:
        JSONResponse: A JSON response containing training response with the training ID.

    Raises:
        ClientException: If there is an issue with the file provided for training.
        ConfigurationException: If the running model does not support unsupervised training.
    """

    data_file = tempfile.NamedTemporaryFile(mode="r+")
    files = []
    file_names = []
    data_file.write("[")
    for td_idx, td in enumerate(training_data):
        temp_td = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8")
        items = ijson.items(td.file, "item")
        temp_td.write("[")
        for text_idx, text in enumerate(items):
            if text_idx > 0 or td_idx > 0:
                data_file.write(",")
            json.dump(text, data_file)
            if text_idx > 0:
                temp_td.write(",")
            json.dump(text, temp_td)
        temp_td.write("]")
        temp_td.flush()
        temp_td.seek(0)
        file_names.append("" if td.filename is None else td.filename)
        files.append(temp_td)
    data_file.write("]")
    logger.debug("Training data concatenated")
    data_file.flush()
    data_file.seek(0)
    training_id = tracking_id or str(uuid.uuid4())
    try:
        training_response = model_service.train_unsupervised(
            data_file,
            epochs,
            log_frequency,
            training_id,
            ",".join(file_names),
            raw_data_files=files,
            synchronised=(os.environ.get("CMS_CI", "false") == "true"),
            lr_override=lr_override,
            test_size=test_size,
            description=description,
        )
    finally:
        for file in files:
            file.close()

    return _get_training_response(training_response, training_id)


@router.post(
    "/train_unsupervised_with_hf_hub_dataset",
    status_code=HTTP_202_ACCEPTED,
    response_class=JSONResponse,
    tags=[Tags.Training.name],
    dependencies=[Depends(cms_globals.props.current_active_user)],
    description="Upload or specify an existing Hugging Face dataset and trigger the unsupervised training",
)
async def train_unsupervised_with_hf_dataset(
    request: Request,
    hf_dataset_repo_id: Annotated[Union[str, None], Query(description="The repository ID of the dataset to download from Hugging Face Hub, will be ignored when 'hf_dataset_package' is provided")] = None,
    hf_dataset_config: Annotated[Union[str, None], Query(description="The name of the dataset configuration, will be ignored when 'hf_dataset_package' is provided")] = None,
    hf_dataset_package: Annotated[Union[UploadFile, None], File(description="A ZIP file or Gzipped tarball containing the dataset to be uploaded, will disable the download of 'hf_dataset_repo_id'")] = None,
    trust_remote_code: Annotated[bool, Query(description="Whether to trust the remote code of the dataset")] = False,
    text_column_name: Annotated[str, Query(description="The name of the text column in the dataset")] = "text",
    epochs: Annotated[int, Query(description="The number of training epochs", ge=0)] = 1,
    lr_override: Annotated[Union[float, None], Query(description="The override of the initial learning rate", gt=0.0)] = None,
    test_size: Annotated[Union[float, None], Query(description="The override of the test size in percentage will only take effect if the dataset does not have predefined validation or test splits", ge=0.0)] = 0.2,
    log_frequency: Annotated[int, Query(description="The number of processed documents or epochs after which training metrics will be logged", ge=1)] = 1000,
    description: Annotated[Union[str, None], Query(description="The description of the training or change logs")] = None,
    tracking_id: Union[str, None] = Depends(validate_tracking_id),
    model_service: AbstractModelService = Depends(cms_globals.model_service_dep),
) -> JSONResponse:
    """
    Triggers unsupervised training on the running model using a dataset from the Hugging Face Hub.

    Args:
        request (Request): The request object.
        hf_dataset_repo_id (str, optional): The repository ID of the dataset to download from Hugging Face Hub, will be ignored when 'hf_dataset_package' is provided.
        hf_dataset_config (str, optional): The name of the dataset configuration, will be ignored when 'hf_dataset_package' is provided.
        hf_dataset_package (UploadFile, optional): A ZIP file or Gzipped tarball containing the dataset to be uploaded, will disable the download of 'hf_dataset_repo_id'.
        trust_remote_code (bool): Whether to trust the remote code of the dataset. Defaults to False.
        text_column_name (str): The name of the text column in the dataset. Defaults to "text".
        epochs (int): The number of training epochs to perform. Defaults tos 1.
        lr_override (float, optional): The override of the initial learning rate. Defaults to the value used in previous training and must be greater than 0.0.
        test_size (float, optional): An override of the test size in percentage. Defaults to 0.2.
        log_frequency (int): The number of processed documents or epochs after which training metrics will be logged. Must be at least 1.
        description (str, optional): A description of the training or change logs. Defaults to empty.
        tracking_id (str, optional): n optional tracking ID of the requested task.
        model_service (AbstractModelService): The model service dependency.

    Returns:
        JSONResponse: A JSON response containing training response with the training ID.
s
    Raises:
        ClientException: If 'hf_dataset_repo_id' and 'hf_dataset_package' are both None, or if the dataset does not contain the specified text column.
        ConfigurationException: If the running model does not support unsupervised training.
    """

    if hf_dataset_repo_id is None and hf_dataset_package is None:
        raise ClientException("Either 'hf_dataset_repo_id' or 'hf_dataset_package' must be provided")

    if model_service.info().model_type not in [ModelType.HUGGINGFACE_NER, ModelType.MEDCAT_SNOMED, ModelType.MEDCAT_ICD10, ModelType.MEDCAT_OPCS4, ModelType.MEDCAT_UMLS]:
        raise ConfigurationException(f"Currently this endpoint is not available for models of type: {model_service.info().model_type.value}")

    data_dir = tempfile.TemporaryDirectory()
    if hf_dataset_package is not None:
        input_file_name = cast(str, hf_dataset_package.filename)
        with tempfile.NamedTemporaryFile(
            suffix=".zip" if input_file_name.endswith(".zip") else ".tar.gz",
            mode="wb",
        ) as temp_file:
            temp_file.write(hf_dataset_package.file.read())
            temp_file.flush()
            if input_file_name and unpack_model_data_package(temp_file.name, data_dir.name):
                logger.debug("Training dataset uploaded and extracted")
            else:
                raise ClientException("Failed to extract the uploaded training dataset")
    else:
        input_file_name = cast(str, hf_dataset_repo_id)
        hf_dataset = datasets.load_dataset(hf_dataset_repo_id,
                                           cache_dir=get_settings().TRAINING_CACHE_DIR,
                                           trust_remote_code=trust_remote_code,
                                           name=hf_dataset_config)
        for split in hf_dataset.keys():
            if text_column_name not in hf_dataset[split].column_names:
                raise ClientException(f"The dataset does not contain a '{text_column_name}' column in the split(s)")
            if text_column_name != "text":
                hf_dataset[split] = hf_dataset[split].map(lambda x: {"text": x[text_column_name]}, batched=True)
            hf_dataset[split] = hf_dataset[split].remove_columns([col for col in hf_dataset[split].column_names if col != "text"])
        logger.debug("Training dataset downloaded and transformed")
        hf_dataset.save_to_disk(data_dir.name)

    training_id = tracking_id or str(uuid.uuid4())
    training_response = model_service.train_unsupervised(
        data_dir,
        epochs,
        log_frequency,
        training_id,
        input_file_name,
        raw_data_files=None,
        synchronised=(os.environ.get("CMS_CI", "false") == "true"),
        lr_override=lr_override,
        test_size=test_size,
        description=description,
    )
    return _get_training_response(training_response, training_id)


def _get_training_response(training_response: Tuple[bool, str, str], training_id: str) -> JSONResponse:
    training_accepted, experiment_id, run_id = training_response
    if training_accepted:
        logger.debug("Training accepted with ID: %s", training_id)
        return JSONResponse(
            content={
                "message": "Your training started successfully.",
                "training_id": training_id,
                "experiment_id": experiment_id,
                "run_id": run_id,
            }, status_code=HTTP_202_ACCEPTED
        )
    else:
        logger.debug("Training refused due to another active training or evaluation on this model")
        return JSONResponse(
            content={
                "message": "Another training or evaluation on this model is still active. Please retry later.",
                "experiment_id": experiment_id,
                "run_id": run_id,
            }, status_code=HTTP_503_SERVICE_UNAVAILABLE
        )
