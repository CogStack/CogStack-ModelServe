import urllib3
import cms_client
from unittest.mock import MagicMock
from cms_client.models.model_card import ModelCard
from cms_client.models.model_type import ModelType
from cms_client.models.text_with_annotations import TextWithAnnotations
from cms_client.models.annotation import Annotation
from app.mcp.tools.annotation.get_annotations import annotate
from app.mcp.tools.annotation.get_redaction import redact
from app.mcp.tools.metadata.get_model_info import model_info
from app.mcp.tools.train_eval.get_train_eval_info import get_train_eval_info
from app.mcp.tools.train_eval.get_train_eval_metrics import get_train_eval_metrics


class TestModelInfoTool:
    def test_model_info_success(self, mocker):
        mock_api_instance = mocker.MagicMock()
        mock_response = ModelCard(
            api_version="api_version",
            model_type=ModelType.MEDCAT_SNOMED,
            model_description="model_description",
            model_card={},
        )
        mock_api_instance.get_model_card.return_value = mock_response

        output = model_info(mock_api_instance)
        assert output == mock_response.to_dict()

    def test_model_info_api_exception(self, mocker):
        mock_api_instance = mocker.MagicMock()
        mock_api_instance.get_model_card.side_effect = cms_client.exceptions.ApiException(reason="API error")

        output = model_info(mock_api_instance)
        assert output == {"status": "error", "reason": "API error"}

    def test_model_info_timeout_error(self, mocker):
        mock_api_instance = mocker.MagicMock()
        mock_api_instance.get_model_card.side_effect = urllib3.exceptions.TimeoutError()

        output = model_info(mock_api_instance)
        assert output == {"status": "error", "reason": "Request timed out. Retrying or aborting gracefully."}

    def test_model_info_general_exception(self, mocker):
        mock_api_instance = mocker.MagicMock()
        mock_api_instance.get_model_card.side_effect = Exception("Unexpected error")

        output = model_info(mock_api_instance)
        assert output == {"status": "error", "reason": "Unexpected error: Unexpected error"}


class TestAnnotateTool:
    def test_annotate_success(self, mocker):
        cms_client.TextWithAnnotations = MagicMock()
        mock_api_instance = mocker.MagicMock()
        mock_response = TextWithAnnotations(
            text="Spinal stenosis",
            annotations=[
                Annotation.from_dict({
                    "label_name": "Spinal stenosis",
                    "label_id": "76107001",
                    "start": 0,
                    "end": 15,
                    "accuracy": 1.0,
                    "meta_anns": {
                        "Status": {
                            "value": "Affirmed",
                            "confidence": 0.9999833106994629,
                            "name": "Status"
                        }
                    },
                })
            ]
        )
        mock_api_instance.get_entities_from_text.return_value = mock_response

        output = annotate("Spinal stenosis", mock_api_instance)
        assert output == mock_response.to_dict()

    def test_annotate_api_exception(self, mocker):
        mock_api_instance = mocker.MagicMock()
        mock_api_instance.get_entities_from_text.side_effect = cms_client.exceptions.ApiException(reason="API error")

        output = annotate("Spinal stenosis", mock_api_instance)
        assert output == {"status": "error", "reason": "API error"}

    def test_annotate_timeout_error(self, mocker):
        mock_api_instance = mocker.MagicMock()
        mock_api_instance.get_entities_from_text.side_effect = urllib3.exceptions.TimeoutError()

        output = annotate("test text", mock_api_instance)
        assert output == {"status": "error", "reason": "Request timed out. Retrying or aborting gracefully."}

    def test_annotate_general_exception(self, mocker):
        mock_api_instance = mocker.MagicMock()
        mock_api_instance.get_entities_from_text.side_effect = Exception("Unexpected error")

        output = annotate("test text", mock_api_instance)
        assert output == {"status": "error", "reason": "Unexpected error: Unexpected error"}


class TestRedactTool:
    def test_redact_success(self, mocker):
        mock_api_instance = mocker.MagicMock()
        mock_api_instance.get_redacted_text.return_value = "[Spinal stenosis]"

        output = redact("Spinal stenosis", mock_api_instance)
        assert output == {"redact_text": "[Spinal stenosis]"}

    def test_redact_api_exception(self, mocker):
        mock_api_instance = mocker.MagicMock()
        mock_api_instance.get_redacted_text.side_effect = cms_client.exceptions.ApiException(reason="API error")

        output = redact("test text", mock_api_instance)
        assert output == {"status": "error", "reason": "API error"}

    def test_redact_timeout_error(self, mocker):
        mock_api_instance = mocker.MagicMock()
        mock_api_instance.get_redacted_text.side_effect = urllib3.exceptions.TimeoutError()

        output = redact("test text", mock_api_instance)
        assert output == {"status": "error", "reason": "Request timed out. Retrying or aborting gracefully."}

    def test_redact_general_exception(self, mocker):
        mock_api_instance = mocker.MagicMock()
        mock_api_instance.get_redacted_text.side_effect = Exception("Unexpected error")

        output = redact("test text", mock_api_instance)
        assert output == {"status": "error", "reason": "Unexpected error: Unexpected error"}


class TestGetTrainEvalInfoTool:
    def test_get_train_eval_info_success(self, mocker):
        train_eval_info = {
            "artifact_uri": "s3://my-bucket/path/to/artifact",
            "experiment_id": "experiment_id",
            "run_id": "run_id",
            "run_name": "run_name",
            "status": "RUNNING",
            "tags": {},
        }
        mock_api_instance = mocker.MagicMock()
        mock_api_instance.train_eval_info.return_value = [train_eval_info]

        output = get_train_eval_info("train_eval_id", mock_api_instance)
        assert output == train_eval_info

    def test_get_train_eval_info_api_exception(self, mocker):
        mock_api_instance = mocker.MagicMock()
        mock_api_instance.train_eval_info.side_effect = cms_client.exceptions.ApiException(reason="API error")

        output = get_train_eval_info("train_eval_id", mock_api_instance)
        assert output == {"status": "error", "reason": "API error"}

    def test_get_train_eval_info_timeout_error(self, mocker):
        mock_api_instance = mocker.MagicMock()
        mock_api_instance.train_eval_info.side_effect = urllib3.exceptions.TimeoutError()

        output = get_train_eval_info("train_eval_id", mock_api_instance)
        assert output == {"status": "error", "reason": "Request timed out. Retrying or aborting gracefully."}

    def test_get_train_eval_info_general_exception(self, mocker):
        mock_api_instance = mocker.MagicMock()
        mock_api_instance.train_eval_info.side_effect = Exception("Unexpected error")

        output = get_train_eval_info("train_eval_id", mock_api_instance)
        assert output == {"status": "error", "reason": "Unexpected error: Unexpected error"}


class TestGetTrainEvalMetricsTool:
    def test_get_train_eval_metrics_success(self, mocker):
        train_eval_metrics = {
            "per_concept_counts": 1,
            "per_concept_count_train": 1,
            "per_concept_acc_fn": 0,
            "per_concept_acc_fp": 1,
            "per_concept_acc_tp": 1,
            "per_concept_acc_cc": 1.0,
            "per_concept_precision": 1.0,
            "per_concept_recall": 1.0,
            "per_concept_f1": 1.0,
        }
        mock_api_instance = mocker.MagicMock()
        mock_api_instance.train_eval_metrics.return_value = [train_eval_metrics]

        output = get_train_eval_metrics("train_eval_id", mock_api_instance)
        assert output == train_eval_metrics

    def test_get_train_eval_metrics_api_exception(self, mocker):
        mock_api_instance = mocker.MagicMock()
        mock_api_instance.train_eval_metrics.side_effect = cms_client.exceptions.ApiException(reason="API error")

        output = get_train_eval_metrics("train_eval_id", mock_api_instance)
        assert output == {"status": "error", "reason": "API error"}

    def test_get_train_eval_metrics_timeout_error(self, mocker):
        mock_api_instance = mocker.MagicMock()
        mock_api_instance.train_eval_metrics.side_effect = urllib3.exceptions.TimeoutError()

        output = get_train_eval_metrics("train_eval_id", mock_api_instance)
        assert output == {"status": "error", "reason": "Request timed out. Retrying or aborting gracefully."}

    def test_get_train_eval_metrics_general_exception(self, mocker):
        mock_api_instance = mocker.MagicMock()
        mock_api_instance.train_eval_metrics.side_effect = Exception("Unexpected error")

        output = get_train_eval_metrics("train_eval_id", mock_api_instance)
        assert output == {"status": "error", "reason": "Unexpected error: Unexpected error"}
    