import pytest
from fastapi import HTTPException

from app.api.dependencies import ModelServiceDep, validate_tracking_id
from app.config import Settings
from app.model_services.medcat_model import MedCATModel
from app.model_services.medcat_model_icd10 import MedCATModelIcd10
from app.model_services.medcat_model_opcs4 import MedCATModelOpcs4
from app.model_services.medcat_model_umls import MedCATModelUmls
from app.model_services.medcat_model_deid import MedCATModelDeIdentification
from app.model_services.trf_model_deid import TransformersModelDeIdentification
from app.model_services.huggingface_ner_model import HuggingFaceNerModel


def test_medcat_snomed_dep():
    model_service_dep = ModelServiceDep("medcat_snomed", Settings())
    assert isinstance(model_service_dep(), MedCATModel)


def test_medcat_icd10_dep():
    model_service_dep = ModelServiceDep("medcat_icd10", Settings())
    assert isinstance(model_service_dep(), MedCATModelIcd10)


def test_medcat_opcs4_dep():
    model_service_dep = ModelServiceDep("medcat_opcs4", Settings())
    assert isinstance(model_service_dep(), MedCATModelOpcs4)


def test_medcat_umls_dep():
    model_service_dep = ModelServiceDep("medcat_umls", Settings())
    assert isinstance(model_service_dep(), MedCATModelUmls)


def test_medcat_deid_dep():
    model_service_dep = ModelServiceDep("medcat_deid", Settings())
    assert isinstance(model_service_dep(), MedCATModelDeIdentification)


def test_transformer_deid_dep():
    model_service_dep = ModelServiceDep("transformers_deid", Settings())
    assert isinstance(model_service_dep(), TransformersModelDeIdentification)


def test_huggingface_ner_dep():
    model_service_dep = ModelServiceDep("huggingface_ner", Settings())
    assert isinstance(model_service_dep(), HuggingFaceNerModel)


@pytest.mark.parametrize(
    "run_id",
    [
        "a" * 32,
        "A" * 32,
        "a" * 256,
        "f0" * 16,
        "abcdef0123456789" * 2,
        "abcdefghijklmnopqrstuvqxyz",
        "123e4567-e89b-12d3-a456-426614174000",
        "123e4567e89b12d3a45642661417400",
    ],
)
def test_validate_tracking_id(run_id):
    assert validate_tracking_id(run_id) == run_id


@pytest.mark.parametrize("run_id", ["a/bc" * 8, "", "a" * 400, "*" * 5])
def test_validate_tracking_id_invalid(run_id):
    with pytest.raises(HTTPException) as exc_info:
        validate_tracking_id(run_id)
    assert exc_info.value.status_code == 400
    assert "Invalid tracking ID" in exc_info.value.detail
