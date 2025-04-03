import os
import datasets
from app.data import doc_dataset


def test_load_dataset():
    sample_texts = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "fixture", "sample_texts.json")
    dataset = datasets.load_dataset(
        doc_dataset.__file__,
        data_files={"documents": sample_texts},
        split="train",
        cache_dir="/tmp",
        trust_remote_code=True,
    )
    assert dataset.features.to_dict() == {"name": {"dtype": "string", "_type": "Value"}, "text": {"dtype": "string", "_type": "Value"}}
    assert len(dataset.to_list()) == 15
    assert dataset.to_list()[0]["name"] == "1"


def test_generate_examples():
    example_gen = doc_dataset.generate_examples([
        os.path.join(os.path.dirname(__file__), "..", "..", "resources", "fixture", "sample_texts.json")
    ])
    example = next(example_gen)
    assert example[0] == "1"
    assert "name" in example[1]
    assert "text" in example[1]
