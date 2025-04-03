import datasets
import ijson
from pathlib import Path
from typing import List, Iterable, Tuple, Dict


class TextDatasetConfig(datasets.BuilderConfig):
    pass


class TextDatasetBuilder(datasets.GeneratorBasedBuilder):
    """A builder class for creating text datasets from files containing text lists."""

    BUILDER_CONFIGS = [
        TextDatasetConfig(
            name="free_text",
            version=datasets.Version("0.0.1"),
            description="Documents with names and free texts",
        )
    ]

    def _info(self) -> datasets.DatasetInfo:
        return datasets.DatasetInfo(
            description="Free text Dataset. This is a dataset containing document records each of which has 'doc_name' and 'text' attributes",
            features=datasets.Features(
                {
                    "name": datasets.Value("string"),
                    "text": datasets.Value("string"),
                }
            ),
        )

    def _split_generators(self, _: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepaths": self.config.data_files["documents"]})
        ]

    def _generate_examples(self, filepaths: List[Path]) -> Iterable[Tuple[str, Dict]]:
        return generate_examples(filepaths)


def generate_examples(filepaths: List[Path]) -> Iterable[Tuple[str, Dict]]:
    """
    Generates examples from the files each containing a list of texts.

    This method reads JSON files containing text lists and yields each record as a tuple
    of a unique text ID and a dictionary with 'name' and 'text' attributes.

    Args:
        filepaths (List[Path]): A list of paths to the JSON files containing text lists.

    Yields:
        Tuple[str, Dict]: A tuple where the first element is a string representing the unique text ID
        and the second element is a dictionary with the following keys:
            - "name" (str): A string representing the text name, which is the same as the text ID.
            - "text" (str): A string containing the free text.
    """

    id_ = 1
    for filepath in filepaths:
        with open(str(filepath), "r") as f:
            texts = ijson.items(f, "item")
            for text in texts:
                yield str(id_), {"name": f"{str(id_)}", "text": text}
                id_ += 1
