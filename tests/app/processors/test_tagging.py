import pytest
import pandas as pd
from unittest.mock import MagicMock
from torch import ones, zeros
from app.processors.tagging import TagProcessor
from app.domain import TaggingScheme


class TestAgregateBioesPredictions:

    def test_aggregate_bioes_predictions_empty_dataframe(self):
        empty_df = pd.DataFrame()
        text = "This is a test sentence."

        result = TagProcessor.aggregate_bioes_predictions(empty_df, text, True)

        assert result == []

    def test_aggregate_bioes_predictions_only_o_tags(self):
        df = pd.DataFrame([
            {"entity": "O", "score": 0.9, "start": 0, "end": 4},
            {"entity": "O", "score": 0.8, "start": 5, "end": 7},
            {"entity": "O", "score": 0.7, "start": 8, "end": 12},
        ])
        text = "This is a test"

        result = TagProcessor.aggregate_bioes_predictions(df, text, True)

        assert result == []

    def test_aggregate_bioes_predictions_single_token_entities(self):
        df = pd.DataFrame([
            {"entity": "S-DISEASE", "score": 0.9, "start": 0, "end": 7},
            {"entity": "O", "score": 0.8, "start": 8, "end": 12},
            {"entity": "S-MEDICATION", "score": 0.7, "start": 12, "end": 20},
        ])
        text = "Disease and medicine"

        result = TagProcessor.aggregate_bioes_predictions(df, text, True)

        assert len(result) == 2
        assert result[0]["entity_group"] == "DISEASE"
        assert result[0]["label_name"] == "DISEASE"
        assert result[0]["start"] == 0
        assert result[0]["end"] == 7
        assert result[0]["text"] == "Disease"
        assert result[0]["score"] == 0.9
        assert result[0]["accuracy"] == 0.9
        assert result[1]["entity_group"] == "MEDICATION"
        assert result[1]["label_name"] == "MEDICATION"
        assert result[1]["start"] == 12
        assert result[1]["end"] == 20
        assert result[1]["text"] == "medicine"
        assert result[1]["score"] == 0.7
        assert result[1]["accuracy"] == 0.7

    def test_aggregate_bioes_predictions_multi_token_entities(self):
        df = pd.DataFrame([
            {"entity": "B-DISEASE", "score": 0.9, "start": 0, "end": 4},
            {"entity": "I-DISEASE", "score": 0.8, "start": 4, "end": 11},
            {"entity": "E-DISEASE", "score": 0.7, "start": 11, "end": 18},
            {"entity": "O", "score": 0.8, "start": 19, "end": 27},
        ])
        text = "Heart disease and diabetes"

        result = TagProcessor.aggregate_bioes_predictions(df, text, True)

        assert len(result) == 1
        assert result[0]["entity_group"] == "DISEASE"
        assert result[0]["label_name"] == "DISEASE"
        assert result[0]["start"] == 0
        assert result[0]["end"] == 18
        assert result[0]["text"] == "Heart disease and "
        assert abs(result[0]["score"] - (0.9 + 0.8 + 0.7) / 3) < 1e-6
        assert abs(result[0]["accuracy"] - (0.9 + 0.8 + 0.7) / 3) < 1e-6

    def test_aggregate_bioes_predictions_beginning_entities(self):
        df = pd.DataFrame([
            {"entity": "B-DISEASE", "score": 0.9, "start": 0, "end": 11},
            {"entity": "O", "score": 0.8, "start": 12, "end": 16},
        ])
        text = "Heart disease"

        result = TagProcessor.aggregate_bioes_predictions(df, text, True)

        assert len(result) == 1
        assert result[0]["entity_group"] == "DISEASE"
        assert result[0]["start"] == 0
        assert result[0]["end"] == 11
        assert result[0]["score"] == 0.9
        assert result[0]["text"] == "Heart disea"

    def test_aggregate_bioes_predictions_inside_entities(self):
        df = pd.DataFrame([
            {"entity": "I-DISEASE", "score": 0.9, "start": 0, "end": 5},
            {"entity": "I-DISEASE", "score": 0.8, "start": 5, "end": 11},
            {"entity": "O", "score": 0.8, "start": 12, "end": 16},
        ])
        text = "Heart disease"

        result = TagProcessor.aggregate_bioes_predictions(df, text, True)

        assert len(result) == 1
        assert result[0]["entity_group"] == "DISEASE"
        assert result[0]["start"] == 0
        assert result[0]["end"] == 11
        assert abs(result[0]["score"] - (0.9 + 0.8) / 2) < 1e-6

    def test_aggregate_bioes_predictions_end_entities(self, huggingface_ner_model):
        df = pd.DataFrame([
            {"entity": "O", "score": 0.8, "start": 0, "end": 10},
            {"entity": "E-DISEASE", "score": 0.9, "start": 10, "end": 17},
        ])
        text = "has heart disease"

        result = TagProcessor.aggregate_bioes_predictions(df, text, True)

        assert len(result) == 1
        assert result[0]["entity_group"] == "DISEASE"
        assert result[0]["start"] == 10
        assert result[0]["end"] == 17
        assert result[0]["score"] == 0.9
        assert result[0]["text"] == "disease"

    def test_aggregate_bioes_predictions_no_prefix(self):
        df = pd.DataFrame([
            {"entity": "DISEASE", "score": 0.9, "start": 0, "end": 5},
            {"entity": "O", "score": 0.8, "start": 6, "end": 10},
        ])
        text = "heart disease patient"

        result = TagProcessor.aggregate_bioes_predictions(df, text, True)

        assert len(result) == 1
        assert result[0]["entity_group"] == "DISEASE"
        assert result[0]["start"] == 0
        assert result[0]["end"] == 5
        assert result[0]["score"] == 0.9
        assert result[0]["text"] == "heart"

    def test_aggregate_bioes_predictions_mixed_entities(self):
        df = pd.DataFrame([
            {"entity": "S-DISEASE", "score": 0.9, "start": 0, "end": 5},
            {"entity": "B-MEDICATION", "score": 0.8, "start": 6, "end": 10},
            {"entity": "I-MEDICATION", "score": 0.7, "start": 10, "end": 16},
            {"entity": "O", "score": 0.8, "start": 17, "end": 20},
            {"entity": "E-SYMPTOM", "score": 0.6, "start": 21, "end": 26},
        ])
        text = "heart aspirin and cough"

        result = TagProcessor.aggregate_bioes_predictions(df, text, True)

        assert len(result) == 3

        assert result[0]["entity_group"] == "DISEASE"
        assert result[0]["start"] == 0
        assert result[0]["end"] == 5
        assert result[0]["score"] == 0.9
        assert result[1]["entity_group"] == "MEDICATION"
        assert result[1]["start"] == 6
        assert result[1]["end"] == 16
        assert abs(result[1]["score"] - (0.8 + 0.7) / 2) < 1e-6
        assert result[2]["entity_group"] == "SYMPTOM"
        assert result[2]["start"] == 21
        assert result[2]["end"] == 26


class TestUpdateModelByTaggingScheme:

    @pytest.fixture
    def mock_model(self):
        model = MagicMock()
        model.config = MagicMock()
        model.config.label2id = {"O": 0, "B-PERSON": 1, "I-PERSON": 2}
        model.config.id2label = {0: "O", 1: "B-PERSON", 2: "I-PERSON"}
        model.classifier = MagicMock()
        model.classifier.weight = ones(3, 10)
        model.classifier.bias = zeros(3)
        model.classifier.out_features = 3
        model.num_labels = 3
        return model

    def test_update_model_by_iob_scheme_new_concepts(self, mock_model):
        concepts = ["DISEASE", "MEDICATION"]        
        initial_num_labels = mock_model.num_labels
        initial_out_features = mock_model.classifier.out_features
        updated_model = TagProcessor.update_model_by_tagging_scheme(
            mock_model, concepts, TaggingScheme.IOB
        )
        
        assert "B-DISEASE" in updated_model.config.label2id
        assert "I-DISEASE" in updated_model.config.label2id
        assert "B-MEDICATION" in updated_model.config.label2id
        assert "I-MEDICATION" in updated_model.config.label2id
        assert updated_model.config.id2label[3] == "B-DISEASE"
        assert updated_model.config.id2label[4] == "I-DISEASE"
        assert updated_model.config.id2label[5] == "B-MEDICATION"
        assert updated_model.config.id2label[6] == "I-MEDICATION"        
        assert updated_model.classifier.out_features == initial_out_features + 4
        assert updated_model.num_labels == initial_num_labels + 4

    def test_update_model_by_iob_scheme_existing_concepts(self, mock_model):
        concepts = ["PERSON", "DISEASE"]        
        initial_num_labels = mock_model.num_labels
        initial_out_features = mock_model.classifier.out_features
        
        updated_model = TagProcessor.update_model_by_tagging_scheme(
            mock_model, concepts, TaggingScheme.IOB
        )
        
        assert updated_model.num_labels == initial_num_labels + 2
        assert updated_model.classifier.out_features == initial_out_features + 2

    def test_update_model_by_iobes_scheme_new_concepts(self, mock_model):
        concepts = ["DISEASE", "MEDICATION"]        
        initial_num_labels = mock_model.num_labels
        initial_out_features = mock_model.classifier.out_features
        updated_model = TagProcessor.update_model_by_tagging_scheme(
            mock_model, concepts, TaggingScheme.IOBES
        )
        
        for concept in concepts:
            assert f"S-{concept}" in updated_model.config.label2id
            assert f"B-{concept}" in updated_model.config.label2id
            assert f"I-{concept}" in updated_model.config.label2id
            assert f"E-{concept}" in updated_model.config.label2id
        
        assert updated_model.classifier.out_features == initial_out_features + 8
        assert updated_model.num_labels == initial_num_labels + 8

    def test_update_model_by_iobes_scheme_existing_concepts(self, mock_model):
        mock_model.config.label2id["B-DISEASE"] = 3
        mock_model.config.label2id["I-DISEASE"] = 4
        mock_model.config.id2label[3] = "B-DISEASE"
        mock_model.config.id2label[4] = "I-DISEASE"        
        concepts = ["DISEASE"]
        initial_num_labels = mock_model.num_labels
        initial_out_features = mock_model.classifier.out_features
        
        updated_model = TagProcessor.update_model_by_tagging_scheme(
            mock_model, concepts, TaggingScheme.IOBES
        )
        
        assert updated_model.num_labels == initial_num_labels + 2
        assert updated_model.classifier.out_features == initial_out_features + 2

    def test_update_model_by_flat_scheme_new_concepts(self, mock_model):
        concepts = ["DISEASE", "MEDICATION"]
        initial_num_labels = mock_model.num_labels
        initial_out_features = mock_model.classifier.out_features
        
        updated_model = TagProcessor.update_model_by_tagging_scheme(
            mock_model, concepts, TaggingScheme.FLAT
        )
        
        assert "DISEASE" in updated_model.config.label2id
        assert "MEDICATION" in updated_model.config.label2id
        assert updated_model.config.id2label[3] == "DISEASE"
        assert updated_model.config.id2label[4] == "MEDICATION"
        assert updated_model.classifier.out_features == initial_out_features + 2
        assert updated_model.num_labels == initial_num_labels + 2

    def test_update_model_by_flat_scheme_existing_concepts(self, mock_model):
        mock_model.config.label2id["PERSON"] = 3
        mock_model.config.id2label[3] = "PERSON"
        mock_model.num_labels = 4
        mock_model.classifier.out_features = 4
        
        concepts = ["PERSON", "DISEASE"]
        initial_num_labels = mock_model.num_labels
        initial_out_features = mock_model.classifier.out_features
        
        updated_model = TagProcessor.update_model_by_tagging_scheme(
            mock_model, concepts, TaggingScheme.FLAT
        )
        
        assert updated_model.num_labels == initial_num_labels + 1
        assert updated_model.classifier.out_features == initial_out_features + 1

    def test_update_model_by_empty_concepts(self, mock_model):
        concepts = []        
        initial_num_labels = mock_model.num_labels
        initial_out_features = mock_model.classifier.out_features
        
        updated_model = TagProcessor.update_model_by_tagging_scheme(
            mock_model, concepts, TaggingScheme.IOB
        )
        
        assert updated_model.num_labels == initial_num_labels
        assert updated_model.classifier.out_features == initial_out_features


class TestGenerateChuncksByTaggingScheme:

    @pytest.fixture
    def mock_model(self):
        model = MagicMock()
        model.config = MagicMock()
        model.config.label2id = {
            "O": 0,
            "B-DISEASE": 1,
            "I-DISEASE": 2,
            "S-DISEASE": 3,
            "E-DISEASE": 4,
            "B-MEDICATION": 5,
            "I-MEDICATION": 6,
            "DISEASE": 7,
            "MEDICATION": 8,
        }
        return model

    def test_generate_chuncks_iob_scheme(self, mock_model):
        annotations = [
            {"start": 5, "end": 10, "cui": "DISEASE"},
            {"start": 15, "end": 20, "cui": "MEDICATION"},
        ]
        tokenized = {
            "input_ids": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "offset_mapping": [
                (0, 3), (3, 5), (5, 8), (8, 10), (10, 12), (12, 15), (15, 18), (18, 20), (20, 23), (23, 25)
            ],
        }
        
        chunks = list(TagProcessor.generate_chuncks_by_tagging_scheme(
            annotations=annotations,
            tokenized=tokenized,
            delfault_label_id=0,
            pad_token_id=0,
            pad_label_id=-100,
            max_length=16,
            model=mock_model,
            tagging_scheme=TaggingScheme.IOB,
            window_size=16,
            stride=16,
        ))
        
        assert len(chunks) == 1
        new_tokenized = chunks[0]
        assert new_tokenized["labels"][2] == 1
        assert new_tokenized["labels"][3] == 2
        assert new_tokenized["labels"][6] == 5
        assert new_tokenized["labels"][7] == 6

    def test_generate_chuncks_iobes_scheme(self, mock_model):
        annotations = [{"start": 5, "end": 15, "cui": "DISEASE"}]
        tokenized = {
            "input_ids": [101, 102, 103, 104, 105, 106, 107, 108, 109],
            "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1],
            "offset_mapping": [(0, 3), (3, 5), (5, 8), (8, 10), (10, 12), (12, 15), (15, 18), (18, 20), (20, 23)],
        }
        
        chunks = list(TagProcessor.generate_chuncks_by_tagging_scheme(
            annotations=annotations,
            tokenized=tokenized,
            delfault_label_id=0,
            pad_token_id=0,
            pad_label_id=-100,
            max_length=16,
            model=mock_model,
            tagging_scheme=TaggingScheme.IOBES,
            window_size=16,
            stride=16,
        ))
        
        assert len(chunks) == 1
        new_tokenized = chunks[0]
        assert new_tokenized["labels"][2] == 1
        assert new_tokenized["labels"][3] == 2
        assert new_tokenized["labels"][4] == 2
        assert new_tokenized["labels"][5] == 4

    def test_generate_chuncks_flat_scheme(self, mock_model):
        annotations = [
            {"start": 5, "end": 10, "cui": "DISEASE"},
            {"start": 15, "end": 20, "cui": "MEDICATION"},
        ]
        tokenized = {
            "input_ids": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "offset_mapping": [
                (0, 3), (3, 5), (5, 8), (8, 10), (10, 12), (12, 15), (15, 18), (18, 20), (20, 23), (23, 25)
            ],
        }
        
        chunks = list(TagProcessor.generate_chuncks_by_tagging_scheme(
            annotations=annotations,
            tokenized=tokenized,
            delfault_label_id=0,
            pad_token_id=0,
            pad_label_id=-100,
            max_length=16,
            model=mock_model,
            tagging_scheme=TaggingScheme.FLAT,
            window_size=16,
            stride=16,
        ))
        
        assert len(chunks) == 1
        new_tokenized = chunks[0]
        assert new_tokenized["labels"][2] == 7
        assert new_tokenized["labels"][3] == 7
        assert new_tokenized["labels"][6] == 8
        assert new_tokenized["labels"][7] == 8


    def test_generate_chuncks_empty_annotations(self, mock_model):
        """Test that empty annotations list results in all default labels"""
        annotations = []
        tokenized = {
            "input_ids": [101, 102, 103, 104, 105],
            "attention_mask": [1, 1, 1, 1, 1],
            "offset_mapping": [(0, 3), (3, 5), (5, 8), (8, 10), (10, 12)],
        }
        
        chunks = list(TagProcessor.generate_chuncks_by_tagging_scheme(
            annotations=annotations,
            tokenized=tokenized,
            delfault_label_id=0,
            pad_token_id=0,
            pad_label_id=-100,
            max_length=8,
            model=mock_model,
            tagging_scheme=TaggingScheme.IOB,
            window_size=8,
            stride=8,
        ))
        
        assert len(chunks) == 1
        new_tokenized = chunks[0]
        assert all(label == 0 for label in new_tokenized["labels"][:5])
        assert all(label == -100 for label in new_tokenized["labels"][5:])
        assert len(new_tokenized["input_ids"]) == 8
        assert len(new_tokenized["labels"]) == 8
        assert len(new_tokenized["attention_mask"]) == 8



    def test_generate_chuncks_multiple_chunks(self, mock_model):
        annotations = [{"start": 5, "end": 12, "cui": "DISEASE"}]
        tokenized = {
            "input_ids": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118],
            "attention_mask": [1] * 18,
            "offset_mapping": [(i*2, i*2+2) for i in range(18)],
        }
        
        chunks = list(TagProcessor.generate_chuncks_by_tagging_scheme(
            annotations=annotations,
            tokenized=tokenized,
            delfault_label_id=0,
            pad_token_id=0,
            pad_label_id=-100,
            max_length=8,
            model=mock_model,
            tagging_scheme=TaggingScheme.IOB,
            window_size=8,
            stride=4,
        ))
        

        assert len(chunks) == 5        
        for chunk in chunks:
            assert len(chunk["input_ids"]) == 8
            assert len(chunk["labels"]) == 8
            assert len(chunk["attention_mask"]) == 8        
        assert chunks[0]["input_ids"][:8] == tokenized["input_ids"][0:8]
        assert chunks[1]["input_ids"][:8] == tokenized["input_ids"][4:12]
        assert chunks[2]["input_ids"][:8] == tokenized["input_ids"][8:16]
        assert chunks[3]["input_ids"][6:] == [0, 0]
        assert chunks[4]["input_ids"][2:] == [0] * 6
