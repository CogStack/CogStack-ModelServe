import os
import datasets
from app.data import anno_dataset


def test_load_dataset():
    trainer_export = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "fixture", "trainer_export_multi_projs.json")
    dataset = datasets.load_dataset(
        anno_dataset.__file__,
        data_files={"annotations": trainer_export},
        split="train",
        cache_dir="/tmp",
        trust_remote_code=True,
    )
    assert dataset.features.to_dict() == {
        "project": {"dtype": "string", "_type": "Value"},
        "name": {"dtype": "string", "_type": "Value"},
        "text": {"dtype": "string", "_type": "Value"},
        "starts": {"dtype": "string", "_type": "Value"},
        "ends": {"dtype": "string", "_type": "Value"},
        "labels": {"dtype": "string", "_type": "Value"},
    }
    assert len(dataset.to_list()) == 4
    assert dataset.to_list()[0]["project"] == "MT Samples (Clone)"
    assert dataset.to_list()[0]["name"] == "1687"
    assert dataset.to_list()[0]["starts"] == "332,255,276,272"
    assert dataset.to_list()[0]["ends"] == "355,267,282,275"
    assert dataset.to_list()[0]["labels"] == "C0017168,C0020538,C0038454,C0007787"


def test_generate_examples():
    example_gen = anno_dataset.generate_examples([
        os.path.join(os.path.dirname(__file__), "..", "..", "resources", "fixture", "trainer_export.json")
    ])
    example = next(example_gen)
    assert example[0] == "1"
    assert "project" in example[1]
    assert "name" in example[1]
    assert "text" in example[1]
    assert "starts" in example[1]
    assert "ends" in example[1]
    assert "labels" in example[1]
