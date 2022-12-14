import json
import pandas as pd
from typing import Tuple, Dict, List, Union, Optional
from collections import defaultdict
from tqdm.autonotebook import tqdm
from model_services.base import AbstractModelService


ANCHOR_DELIMITER = ";"


def evaluate_model_with_trainer_export(data_file_path: str,
                                       model_service: AbstractModelService,
                                       return_df: bool = False,
                                       include_anchors: bool = False) -> Union[pd.DataFrame, Tuple[float, float, float, Dict, Dict, Dict, Dict, Optional[Dict]]]:
    with open(data_file_path, "r") as f:
        data = json.load(f)

    correct_cuis: Dict = {}
    for project in data["projects"]:
        correct_cuis[project["id"]] = defaultdict(list)

        for document in project["documents"]:
            for entry in document["annotations"]:
                if entry["correct"]:
                    if document["id"] not in correct_cuis[project["id"]]:
                        correct_cuis[project["id"]][document["id"]] = []
                    correct_cuis[project["id"]][document["id"]].append([entry["start"], entry["end"], entry["cui"]])

    true_positives: Dict = {}
    false_positives: Dict = {}
    false_negatives: Dict = {}
    concept_names: Dict = {}
    concept_anchors: Dict = {}
    true_positive_count, false_positive_count, false_negative_count = 0, 0, 0

    for project in tqdm(data["projects"], desc="Evaluating projects", total=len(data["projects"]), leave=False):
        predictions: Dict = {}
        documents = project["documents"]
        true_positives[project["id"]] = {}
        false_positives[project["id"]] = {}
        false_negatives[project["id"]] = {}

        for document in tqdm(documents, desc="Evaluating documents", total=len(documents), leave=False):
            true_positives[project["id"]][document["id"]] = {}
            false_positives[project["id"]][document["id"]] = {}
            false_negatives[project["id"]][document["id"]] = {}

            annotations = model_service.annotate(document["text"])
            predictions[document["id"]] = []
            for annotation in annotations:
                predictions[document["id"]].append([annotation["start"], annotation["end"], annotation["label_id"]])
                concept_names[annotation["label_id"]] = annotation["label_name"]
                concept_anchors[annotation["label_id"]] = concept_anchors.get(annotation["label_id"], [])
                concept_anchors[annotation["label_id"]].append(f"P{project['id']}/D{document['id']}/S{annotation['start']}/E{ annotation['end']}")

            predicted = {tuple(x) for x in predictions[document["id"]]}
            actual = {tuple(x) for x in correct_cuis[project["id"]][document["id"]]}
            doc_tps = list(predicted.intersection(actual))
            doc_fps = list(predicted.difference(actual))
            doc_fns = list(actual.difference(predicted))
            true_positives[project["id"]][document["id"]] = doc_tps
            false_positives[project["id"]][document["id"]] = doc_fps
            false_negatives[project["id"]][document["id"]] = doc_fns
            true_positive_count += len(doc_tps)
            false_positive_count += len(doc_fps)
            false_negative_count += len(doc_fns)

    precision = true_positive_count / (true_positive_count + false_positive_count) if (true_positive_count + false_positive_count) != 0 else 0
    recall = true_positive_count / (true_positive_count + false_negative_count) if (true_positive_count + false_negative_count) != 0 else 0
    f1 = 2*((precision*recall) / (precision + recall)) if (precision + recall) != 0 else 0

    fp_counts: Dict = defaultdict(int)
    fn_counts: Dict = defaultdict(int)
    tp_counts: Dict = defaultdict(int)
    per_cui_prec = defaultdict(float)
    per_cui_rec = defaultdict(float)
    per_cui_f1 = defaultdict(float)
    per_cui_name = defaultdict(str)
    per_cui_anchors = defaultdict(str)

    for documents in false_positives.values():
        for spans in documents.values():
            for span in spans:
                fp_counts[span[2]] += 1

    for documents in false_negatives.values():
        for spans in documents.values():
            for span in spans:
                fn_counts[span[2]] += 1

    for documents in true_positives.values():
        for spans in documents.values():
            for span in spans:
                tp_counts[span[2]] += 1

    for cui in tp_counts.keys():
        per_cui_prec[cui] = tp_counts[cui] / (tp_counts[cui] + fp_counts[cui])
        per_cui_rec[cui] = tp_counts[cui] / (tp_counts[cui] + fn_counts[cui])
        per_cui_f1[cui] = 2*(per_cui_prec[cui]*per_cui_rec[cui]) / (per_cui_prec[cui] + per_cui_rec[cui])
        per_cui_name[cui] = concept_names[cui]
        per_cui_anchors[cui] = ANCHOR_DELIMITER.join(concept_anchors[cui])

    if return_df:
        df = pd.DataFrame({
            "concept": per_cui_prec.keys(),
            "name": per_cui_name.values(),
            "precision": per_cui_prec.values(),
            "recall": per_cui_rec.values(),
            "f1": per_cui_f1.values(),
        })
        if include_anchors:
            df["anchors"] = per_cui_anchors.values()
        return df
    else:
        return precision, recall, f1, per_cui_prec, per_cui_rec, per_cui_f1, per_cui_name, per_cui_anchors if include_anchors else None


def concat_trainer_exports(data_file_paths: List[str], combined_data_file_path: str) -> str:
    combined: Dict = {"projects": []}
    project_id = 0
    for path in data_file_paths:
        with open(path, "r") as f:
            data = json.load(f)
            for project in data["projects"]:
                project["id"] = project_id
                project_id += 1
        combined["projects"].extend(data["projects"])

    with open(combined_data_file_path, "w") as f:
        json.dump(combined, f)

    return combined_data_file_path


def get_cui_counts_from_trainer_export(file_path: str) -> Dict[str, int]:
    cuis: Dict = defaultdict(int)
    with open(file_path, "r") as f:
        export_object = json.load(f)
        for project in export_object["projects"]:
            for doc in project["documents"]:
                annotations = []
                if type(doc["annotations"]) == list:
                    annotations = doc["annotations"]
                elif type(doc["annotations"]) == dict:
                    annotations = list(doc["annotations"].values())
                for annotation in annotations:
                    cuis[annotation["cui"]] += 1
    return dict(cuis)
