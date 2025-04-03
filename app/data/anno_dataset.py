import datasets
import json
from pathlib import Path
from typing import List, Iterable, Tuple, Dict
from app.utils import filter_by_concept_ids


class AnnotationDatasetConfig(datasets.BuilderConfig):
    pass


class AnnotationDatasetBuilder(datasets.GeneratorBasedBuilder):
    """A builder class for creating annotation datasets from flattened trainer export files."""

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
                    "project": datasets.Value("string"),
                    "name":datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "starts": datasets.Value("string"), # Mlflow ColSpec schema does not support HF Dataset Sequence
                    "ends": datasets.Value("string"),   # Mlflow ColSpec schema does not support HF Dataset Sequence
                    "labels": datasets.Value("string"), # Mlflow ColSpec schema does not support HF Dataset Sequence
                }
            ),
        )

    def _split_generators(self, _: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepaths": self.config.data_files["annotations"]})
        ]

    def _generate_examples(self, filepaths: List[Path]) -> Iterable[Tuple[str, Dict]]:
        return generate_examples(filepaths)

def generate_examples(filepaths: List[Path]) -> Iterable[Tuple[str, Dict]]:
    """
    Generates examples from a list of trainer export files.

    This function reads annotation data from trainer export files and yields
    each annotated document as a tuple containing a unique doc ID and a dictionary
    with keys 'project', 'name', 'text', 'starts', 'ends', and 'labels'.

    Args:
        filepaths (List[Path]): A list of trainer export files.

    Yields:
        Iterable[Tuple[str, Dict]]: An iterable of tuples where each tuple contains:
            - A unique doc ID.
            - A dictionary with the following keys:
                - "project": The name of the annotation project.
                - "name": The name of the document.
                - "text": The text of the document.
                - "starts": A comma-separated string of start indices of annotations.
                - "ends": A comma-separated string of end indices of annotations.
                - "labels": A comma-separated string of concept IDs (CUIs) for the annotations.
    """

    id_ = 1
    for filepath in filepaths:
        with open(str(filepath), "r") as f:
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
                        "project": project.get("name"),
                        "name": document.get("name"),
                        "text": document.get("text"),
                        "starts": ",".join(starts),
                        "ends": ",".join(ends),
                        "labels": ",".join(labels),
                    }
                    id_ += 1
