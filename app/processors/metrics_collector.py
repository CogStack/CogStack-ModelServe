import json
from typing import Tuple, Dict, List
from collections import defaultdict
from model_services.base import AbstractModelService


def evaluate_model_with_trainer_export(data_file_path: str, model: AbstractModelService) -> Tuple[float, float, float, Dict, Dict, Dict]:
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
    true_positive_count, false_positive_count, false_negative_count = 0, 0, 0

    for project in data["projects"]:
        predictions = {}
        documents = project["documents"]
        true_positives[project["id"]] = {}
        false_positives[project["id"]] = {}
        false_negatives[project["id"]] = {}

        for document in documents:
            true_positives[project["id"]][document["id"]] = {}
            false_positives[project["id"]][document["id"]] = {}
            false_negatives[project["id"]][document["id"]] = {}

            annotations = model.annotate(document["text"])
            predictions[document["id"]] = [[a["start"], a["end"], a["label_id"]] for a in annotations]

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

    fps: Dict = defaultdict(int)
    fns: Dict = defaultdict(int)
    tps: Dict = defaultdict(int)
    per_cui_prec = defaultdict(int)
    per_cui_rec = defaultdict(int)
    per_cui_f1 = defaultdict(int)

    for documents in false_positives.values():
        for spans in documents.values():
            for span in spans:
                fps[span[2]] += 1

    for documents in false_negatives.values():
        for spans in documents.values():
            for span in spans:
                fns[span[2]] += 1

    for documents in true_positives.values():
        for spans in documents.values():
            for span in spans:
                tps[span[2]] += 1

    for cui in tps.keys():
        per_cui_prec[cui] = tps[cui] / (tps[cui] + fps[cui])
        per_cui_rec[cui] = tps[cui] / (tps[cui] + fns[cui])
        per_cui_f1[cui] = 2*(per_cui_prec[cui]*per_cui_rec[cui]) / (per_cui_prec[cui] + per_cui_rec[cui])

    return precision, recall, f1, dict(per_cui_prec), dict(per_cui_rec), dict(per_cui_f1)


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
