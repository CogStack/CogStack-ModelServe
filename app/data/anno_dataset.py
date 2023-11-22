import datasets
import json
from pathlib import Path
from typing import List, Iterable, Tuple, Dict
from utils import filter_by_concept_ids


class AnnotationDatasetConfig(datasets.BuilderConfig):
    pass


class AnnotationDatasetBuilder(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        AnnotationDatasetConfig(
            name="json_annotation",
            version=datasets.Version("0.0.1"),
            description="Flattened MedCAT Trainer export JSON",
        )
    ]

    def _info(self) -> datasets.DatasetInfo:
        return datasets.DatasetInfo(
            description="Annotation Dataset. This is a dataset containing flattened MedCAT Trainer export",
            features=datasets.Features(
                {
                    "name": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "starts": datasets.Value("string"), # Mlflow ColSpec schema does not support HF Dataset Sequence
                    "ends": datasets.Value("string"),   # Mlflow ColSpec schema does not support HF Dataset Sequence
                    "labels": datasets.Value("string"), # Mlflow ColSpec schema does not support HF Dataset Sequence
                }
            )
        )

    def _split_generators(self, _: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepaths": self.config.data_files["annotations"]})
        ]

    def _generate_examples(self, filepaths: List[Path]) -> Iterable[Tuple[str, Dict]]:
        id_ = 1
        for filepath in filepaths:
            with (open(str(filepath), "r") as f):
                annotations = json.load(f)
                filtered = filter_by_concept_ids(annotations)
                for project in filtered["projects"]:
                    for document in project["documents"]:
                        starts = []
                        ends = []
                        labels = []
                        for annotation in document["annotations"]:
                            starts.append(str(annotation["start"]))
                            ends.append(str(annotation["end"]))
                            labels.append(annotation["cui"])
                        yield str(id_), {
                            "name": document["name"],
                            "text": document["text"],
                            "starts": ",".join(starts),
                            "ends": ",".join(ends),
                            "labels": ",".join(labels),
                        }
                        id_ += 1
