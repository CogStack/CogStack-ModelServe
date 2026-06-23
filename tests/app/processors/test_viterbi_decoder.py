from app.processors.viterbi_decoder import ViterbiDecoder


def test_from_id2label_iob() -> None:
    id2label = {
        0: "O",
        1: "B-LABEL",
        2: "I-LABEL",
        3: "E-LABEL",
        4: "S-LABEL",
    }

    decoder = ViterbiDecoder.from_id2label(
        id2label,
        viterbi_biases={"transition_bias_background_to_start": 1.5},
    )

    assert isinstance(decoder, ViterbiDecoder)
    assert decoder.label_info.background_token_label == 0
    assert decoder.label_info.boundary_label_lookup["LABEL"]["B"] == 1
    assert decoder.transition_bias_background_to_start == 1.5


def test_from_id2label_iobes() -> None:
    id2label = {
        0: "O",
        1: "B-LABEL",
        2: "I-LABEL",
    }

    decoder = ViterbiDecoder.from_id2label(
        id2label,
        viterbi_biases={"transition_bias_background_to_start": 1.5},
    )

    assert isinstance(decoder, ViterbiDecoder)
    assert decoder.label_info.background_token_label == 0
    assert decoder.label_info.boundary_label_lookup["LABEL"]["B"] == 1
    assert decoder.transition_bias_background_to_start == 1.5


def test_apply_viterbi_to_hf_pipeline_output_iob() -> None:
    id2label = {
        0: "O",
        1: "B-LABEL",
        2: "I-LABEL",
    }
    decoder = ViterbiDecoder.from_id2label(id2label)
    decoder.decode = lambda _: [1, 0]  # type: ignore[method-assign]

    pipeline_output = [
        {"entity": "LABEL", "score": 0.9, "index": 0, "start": 0, "end": 5},
        {"entity": "B-LABEL", "score": 0.8, "index": 1, "start": 6, "end": 10},
    ]

    corrected = decoder.apply_viterbi_to_hf_pipeline_output(pipeline_output, id2label)

    assert corrected[0]["entity"] == "B-LABEL"
    assert corrected[1]["entity"] == "O"
    assert corrected[0]["start"] == 0
    assert corrected[1]["end"] == 10


def test_apply_viterbi_to_hf_pipeline_output_iobes() -> None:
    id2label = {
        0: "O",
        1: "B-LABEL",
        2: "I-LABEL",
        3: "E-LABEL",
        4: "S-LABEL",
    }
    decoder = ViterbiDecoder.from_id2label(id2label)
    decoder.decode = lambda _: [4, 0]  # type: ignore[method-assign]

    pipeline_output = [
        {"entity": "LABEL", "score": 0.9, "index": 0, "start": 0, "end": 5},
        {"entity": "B-LABEL", "score": 0.8, "index": 1, "start": 6, "end": 10},
    ]

    corrected = decoder.apply_viterbi_to_hf_pipeline_output(pipeline_output, id2label)

    assert corrected[0]["entity"] == "S-LABEL"
    assert corrected[1]["entity"] == "O"
    assert corrected[0]["start"] == 0
    assert corrected[1]["end"] == 10
