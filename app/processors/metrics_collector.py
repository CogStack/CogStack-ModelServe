import json
import hashlib
import pandas as pd
from typing import Tuple, Dict, List, Set, Union, Optional, TextIO
from collections import defaultdict
from sklearn.metrics import cohen_kappa_score
from tqdm.autonotebook import tqdm
from model_services.base import AbstractModelService


ANCHOR_DELIMITER = ";"
DOC_SPAN_DELIMITER = "_"
STATE_MISSING = hashlib.sha1("MISSING".encode("utf-8")).hexdigest()
META_STATE_MISSING = hashlib.sha1("{}".encode("utf-8")).hexdigest()


def evaluate_model_with_trainer_export(export_file: Union[str, TextIO],
                                       model_service: AbstractModelService,
                                       return_df: bool = False,
                                       include_anchors: bool = False) -> Union[pd.DataFrame, Tuple[float, float, float, Dict, Dict, Dict, Dict, Optional[Dict]]]:
    if isinstance(export_file, str):
        with open(export_file, "r") as file:
            data = json.load(file)
    else:
        data = json.load(export_file)

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


def concat_trainer_exports(data_file_paths: List[str], combined_data_file_path: Optional[str] = None) -> Union[Dict, str]:
    combined: Dict = {"projects": []}
    project_id = 0
    for path in data_file_paths:
        with open(path, "r") as f:
            data = json.load(f)
            for project in data["projects"]:
                project["id"] = project_id
                project_id += 1
        combined["projects"].extend(data["projects"])

    if isinstance(combined_data_file_path, str):
        with open(combined_data_file_path, "w") as f:
            json.dump(combined, f)

        return combined_data_file_path
    else:
        return combined


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


def get_iaa_scores_per_concept(export_file: Union[str, TextIO],
                               project_id: int,
                               another_project_id: int,
                               return_df: bool = False) -> Union[pd.DataFrame, Tuple[Dict, Dict]]:
    project_a, project_b = _extract_project_pair(export_file, project_id, another_project_id)
    filtered_projects = _filter_common_docs([project_a, project_b])

    state_keys = {"validated", "correct", "deleted", "alternative", "killed", "manually_created"}
    docspan2cui_a = {}
    docspan2state_proj_a = {}
    docspan2metastate_proj_a = {}
    for document in filtered_projects[0]["documents"]:
        for annotation in document["annotations"]:
            docspan_key = _get_docspan_key(document, annotation)
            docspan2cui_a[docspan_key] = annotation["cui"]
            docspan2state_proj_a[docspan_key] = _get_hashed_annotation_state(annotation, state_keys)
            docspan2metastate_proj_a[docspan_key] = _get_hashed_meta_annotation_state(annotation["meta_anns"])

    docspan2cui_b = {}
    docspan2state_proj_b = {}
    docspan2metastate_proj_b = {}
    for document in filtered_projects[1]["documents"]:
        for annotation in document["annotations"]:
            docspan_key = _get_docspan_key(document, annotation)
            docspan2cui_b[docspan_key] = annotation["cui"]
            docspan2state_proj_b[docspan_key] = _get_hashed_annotation_state(annotation, state_keys)
            docspan2metastate_proj_b[docspan_key] = _get_hashed_meta_annotation_state(annotation["meta_anns"])

    cui_states = {}
    cui_metastates = {}
    cuis = set(docspan2cui_a.values()).union(set(docspan2cui_b.values()))
    for cui in cuis:
        docspans = set(_filter_docspan_by_value(docspan2cui_a, cui).keys()).union(set(_filter_docspan_by_value(docspan2cui_b, cui).keys()))
        cui_states[cui] = [(docspan2state_proj_a.get(docspan, STATE_MISSING), docspan2state_proj_b.get(docspan, STATE_MISSING)) for docspan in docspans]
        cui_metastates[cui] = [(docspan2metastate_proj_a.get(docspan, META_STATE_MISSING), docspan2metastate_proj_b.get(docspan, META_STATE_MISSING)) for docspan in docspans]

    per_cui_anno_iia_pct = {}
    per_cui_anno_cohens_kappa = {}
    for cui, cui_state_pairs in cui_states.items():
        per_cui_anno_iia_pct[cui] = len([1 for csp in cui_state_pairs if csp[0] == csp[1]]) / len(cui_state_pairs) * 100
        per_cui_anno_cohens_kappa[cui] = _get_cohens_kappa_coefficient(*map(list, zip(*cui_state_pairs)))
    per_cui_metaanno_iia_pct = {}
    per_cui_metaanno_cohens_kappa = {}
    for cui, cui_metastate_pairs in cui_metastates.items():
        per_cui_metaanno_iia_pct[cui] = len([1 for csp in cui_metastate_pairs if csp[0] == csp[1]]) / len(cui_metastate_pairs) * 100
        per_cui_metaanno_cohens_kappa[cui] = _get_cohens_kappa_coefficient(*map(list, zip(*cui_metastate_pairs)))

    if return_df:
        df = pd.DataFrame({
            "concept": per_cui_anno_iia_pct.keys(),
            "iaa_percentage": per_cui_anno_iia_pct.values(),
            "cohens_kappa": per_cui_anno_cohens_kappa.values(),
            "iaa_percentage_meta": per_cui_metaanno_iia_pct.values(),
            "cohens_kappa_meta": per_cui_metaanno_cohens_kappa.values()
        }).sort_values(["concept"], ascending=True)
        return df.fillna("NaN")
    else:
        return per_cui_anno_iia_pct, per_cui_anno_cohens_kappa, per_cui_metaanno_iia_pct, per_cui_metaanno_cohens_kappa


def get_iaa_scores_per_doc(export_file: Union[str, TextIO],
                           project_id: int,
                           another_project_id: int,
                           return_df: bool = False) -> Union[pd.DataFrame, Tuple[Dict, Dict]]:
    project_a, project_b = _extract_project_pair(export_file, project_id, another_project_id)
    filtered_projects = _filter_common_docs([project_a, project_b])
    state_keys = {"validated", "correct", "deleted", "alternative", "killed", "manually_created", "cui"}

    docspan2doc_id_a = {}
    docspan2state_proj_a = {}
    docspan2metastate_proj_a = {}
    for document in filtered_projects[0]["documents"]:
        for annotation in document["annotations"]:
            docspan_key = _get_docspan_key(document, annotation)
            docspan2doc_id_a[docspan_key] = document["id"]
            docspan2state_proj_a[docspan_key] = _get_hashed_annotation_state(annotation, state_keys)
            docspan2metastate_proj_a[docspan_key] = _get_hashed_meta_annotation_state(annotation["meta_anns"])

    docspan2doc_id_b = {}
    docspan2state_proj_b = {}
    docspan2metastate_proj_b = {}
    for document in filtered_projects[1]["documents"]:
        for annotation in document["annotations"]:
            docspan_key = _get_docspan_key(document, annotation)
            docspan2doc_id_b[docspan_key] = document["id"]
            docspan2state_proj_b[docspan_key] = _get_hashed_annotation_state(annotation, state_keys)
            docspan2metastate_proj_b[docspan_key] = _get_hashed_meta_annotation_state(annotation["meta_anns"])

    doc_states = {}
    doc_metastates = {}
    doc_ids = sorted(set(docspan2doc_id_a.values()).union(set(docspan2doc_id_b.values())))
    for doc_id in doc_ids:
        docspans = set(_filter_docspan_by_value(docspan2doc_id_a, doc_id).keys()).union(
            set(_filter_docspan_by_value(docspan2doc_id_b, doc_id).keys()))
        doc_states[doc_id] = [(docspan2state_proj_a.get(docspan, STATE_MISSING), docspan2state_proj_b.get(docspan, STATE_MISSING)) for docspan in docspans]
        doc_metastates[doc_id] = [(docspan2metastate_proj_a.get(docspan, META_STATE_MISSING), docspan2metastate_proj_b.get(docspan, META_STATE_MISSING)) for docspan in docspans]

    per_doc_anno_iia_pct = {}
    per_doc_anno_cohens_kappa = {}
    for doc_id, doc_state_pairs in doc_states.items():
        per_doc_anno_iia_pct[str(doc_id)] = len([1 for dsp in doc_state_pairs if dsp[0] == dsp[1]]) / len(doc_state_pairs) * 100
        per_doc_anno_cohens_kappa[str(doc_id)] = _get_cohens_kappa_coefficient(*map(list, zip(*doc_state_pairs)))
    per_doc_metaanno_iia_pct = {}
    per_doc_metaanno_cohens_kappa = {}
    for doc_id, doc_metastate_pairs in doc_metastates.items():
        per_doc_metaanno_iia_pct[str(doc_id)] = len([1 for dsp in doc_metastate_pairs if dsp[0] == dsp[1]]) / len(doc_metastate_pairs) * 100
        per_doc_metaanno_cohens_kappa[str(doc_id)] = _get_cohens_kappa_coefficient(*map(list, zip(*doc_metastate_pairs)))

    if return_df:
        df = pd.DataFrame({
            "doc_id": per_doc_anno_iia_pct.keys(),
            "iaa_percentage": per_doc_anno_iia_pct.values(),
            "cohens_kappa": per_doc_anno_cohens_kappa.values(),
            "iaa_percentage_meta": per_doc_metaanno_iia_pct.values(),
            "cohens_kappa_meta": per_doc_metaanno_cohens_kappa.values()
        }).sort_values(["doc_id"], ascending=True)
        return df.fillna("NaN")
    else:
        return per_doc_anno_iia_pct, per_doc_anno_cohens_kappa, per_doc_metaanno_iia_pct, per_doc_metaanno_cohens_kappa


def get_iaa_scores_per_span(export_file: Union[str, TextIO],
                            project_id: int,
                            another_project_id: int,
                            return_df: bool = False) -> Union[pd.DataFrame, Tuple[Dict, Dict]]:
    project_a, project_b = _extract_project_pair(export_file, project_id, another_project_id)
    filtered_projects = _filter_common_docs([project_a, project_b])
    state_keys = {"validated", "correct", "deleted", "alternative", "killed", "manually_created", "cui"}

    docspan2state_proj_a = {}
    docspan2statemeta_proj_a = {}
    for document in filtered_projects[0]["documents"]:
        for annotation in document["annotations"]:
            docspan_key = _get_docspan_key(document, annotation)
            docspan2state_proj_a[docspan_key] = [str(annotation[key]) for key in state_keys]
            docspan2statemeta_proj_a[docspan_key] = [str(meta_ann) for meta_ann in annotation["meta_anns"].items()] if annotation["meta_anns"] else [META_STATE_MISSING]

    docspan2state_proj_b = {}
    docspan2statemeta_proj_b = {}
    for document in filtered_projects[1]["documents"]:
        for annotation in document["annotations"]:
            docspan_key = _get_docspan_key(document, annotation)
            docspan2state_proj_b[docspan_key] = [str(annotation[key]) for key in state_keys]
            docspan2statemeta_proj_b[docspan_key] = [str(meta_ann) for meta_ann in annotation["meta_anns"].items()] if annotation["meta_anns"] else [META_STATE_MISSING]

    docspans = set(docspan2state_proj_a.keys()).union(set(docspan2state_proj_b.keys()))
    docspan_states = {docspan: (docspan2state_proj_a.get(docspan, [STATE_MISSING]*len(state_keys)), docspan2state_proj_b.get(docspan, [STATE_MISSING]*len(state_keys))) for docspan in docspans}
    docspan_metastates = {}
    for docspan in docspans:
        if docspan in docspan2statemeta_proj_a and docspan not in docspan2statemeta_proj_b:
            docspan_metastates[docspan] = (docspan2statemeta_proj_a[docspan], [STATE_MISSING] * len(docspan2statemeta_proj_a[docspan]))
        elif docspan not in docspan2statemeta_proj_a and docspan in docspan2statemeta_proj_b:
            docspan_metastates[docspan] = ([STATE_MISSING] * len(docspan2statemeta_proj_b[docspan]), docspan2statemeta_proj_b[docspan])
        else:
            docspan_metastates[docspan] = (docspan2statemeta_proj_a[docspan], docspan2statemeta_proj_b[docspan])

    per_span_anno_iia_pct = {}
    per_span_anno_cohens_kappa = {}
    for docspan, docspan_state_pairs in docspan_states.items():
        per_span_anno_iia_pct[docspan] = len([1 for state_a, state_b in zip(docspan_state_pairs[0], docspan_state_pairs[1]) if state_a == state_b]) / len(state_keys) * 100
        per_span_anno_cohens_kappa[docspan] = _get_cohens_kappa_coefficient(docspan_state_pairs[0], docspan_state_pairs[1])
    per_doc_metaanno_iia_pct = {}
    per_doc_metaanno_cohens_kappa = {}
    for docspan, docspan_metastate_pairs in docspan_metastates.items():
        per_doc_metaanno_iia_pct[docspan] = len([1 for state_a, state_b in zip(docspan_metastate_pairs[0], docspan_metastate_pairs[1]) if state_a == state_b]) / len(docspan_metastate_pairs[0]) * 100
        per_doc_metaanno_cohens_kappa[docspan] = _get_cohens_kappa_coefficient(docspan_metastate_pairs[0], docspan_metastate_pairs[1])

    if return_df:
        df = pd.DataFrame({
            "doc_id": [int(key.split(DOC_SPAN_DELIMITER)[0]) for key in per_span_anno_iia_pct.keys()],
            "span_start": [int(key.split(DOC_SPAN_DELIMITER)[1]) for key in per_span_anno_iia_pct.keys()],
            "span_end": [int(key.split(DOC_SPAN_DELIMITER)[2]) for key in per_span_anno_iia_pct.keys()],
            "iaa_percentage": per_span_anno_iia_pct.values(),
            "cohens_kappa": per_span_anno_cohens_kappa.values(),
            "iaa_percentage_meta": per_doc_metaanno_iia_pct.values(),
            "cohens_kappa_meta": per_doc_metaanno_cohens_kappa.values()
        }).sort_values(["doc_id", "span_start", "span_end"], ascending=[True, True, True])
        return df.fillna("NaN")
    else:
        return per_span_anno_iia_pct, per_span_anno_cohens_kappa, per_doc_metaanno_iia_pct, per_doc_metaanno_cohens_kappa


def _extract_project_pair(export_file: Union[str, TextIO],
                          project_id: int,
                          another_project_id: int) -> Tuple[Dict, Dict]:
    if isinstance(export_file, str):
        with open(export_file, "r") as file:
            data = json.load(file)
    else:
        data = json.load(export_file)

    project_a = project_b = None
    for project in data["projects"]:
        if project_id == project["id"]:
            project_a = project
        if another_project_id == project["id"]:
            project_b = project
    if project_a is None:
        raise ValueError(f"Cannot find the project with ID: {project_id}")
    if project_b is None:
        raise ValueError(f"Cannot find the project with ID: {another_project_id}")

    return project_a, project_b


def _get_docspan_key(document: Dict, annotation: Dict) -> str:
    return f"{document['id']}{DOC_SPAN_DELIMITER}{annotation['start']}{DOC_SPAN_DELIMITER}{annotation['end']}"


def _filter_common_docs(projects: List[Dict]) -> List[Dict]:
    project_doc_ids = []
    for project in projects:
        project_doc_ids.append({doc["id"] for doc in project["documents"]})
    common_doc_ids = set.intersection(*project_doc_ids)
    filtered_projects = []
    for project in projects:
        project["documents"] = [doc for doc in project["documents"] if doc["id"] in common_doc_ids]
        filtered_projects.append(project)
    return filtered_projects


def _filter_docspan_by_value(docspan2value: Dict, value: str) -> Dict:
    return {docspan: val for docspan, val in docspan2value.items() if val == value}


def _get_hashed_annotation_state(annotation: Dict, state_keys: Set[str]) -> str:
    return hashlib.sha1("_".join([str(annotation[key]) for key in state_keys]).encode("utf-8")).hexdigest()


def _get_hashed_meta_annotation_state(meta_anno: Dict) -> str:
    meta_anno = {key: val for key, val in sorted(meta_anno.items(), key=lambda item: item[0])}  # may not be necessary
    return hashlib.sha1(str(meta_anno).encode("utf=8")).hexdigest()


def _get_cohens_kappa_coefficient(y1_labels: List, y2_labels: List) -> float:
    return cohen_kappa_score(y1_labels, y2_labels) if len(set(y1_labels).union(set(y2_labels))) != 1 else 1.0
