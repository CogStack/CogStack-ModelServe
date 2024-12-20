from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import datasets
import ijson


class TextDatasetConfig(datasets.BuilderConfig):
    pass


class TextDatasetBuilder(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        TextDatasetConfig(
            name="free_text",
            version=datasets.Version("0.0.1"),
            description="Documents with names and free texts",
        )
    ]

    def _info(self) -> datasets.DatasetInfo:
        return datasets.DatasetInfo(
            description=(
                "Free text Dataset. This is a dataset containing document records each of which has"
                " 'doc_name' and 'text' attributes"
            ),
            features=datasets.Features(
                {
                    "name": datasets.Value("string"),
                    "text": datasets.Value("string"),
                }
            ),
        )

    def _split_generators(self, _: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepaths": self.config.data_files["documents"]},
            )
        ]

    def _generate_examples(self, filepaths: List[Path]) -> Iterable[Tuple[str, Dict]]:
        return generate_examples(filepaths)


def generate_examples(filepaths: List[Path]) -> Iterable[Tuple[str, Dict]]:
    id_ = 1
    for filepath in filepaths:
        with open(str(filepath), "r") as f:
            texts = ijson.items(f, "item")
            for text in texts:
                yield str(id_), {"name": f"{str(id_)}", "text": text}
                id_ += 1
