import inspect
import json
import hashlib
import pandas as pd
from typing import Tuple, Dict, List, Set, Union, Optional, IO, Any
from collections import defaultdict
from sklearn.metrics import cohen_kappa_score
from tqdm.autonotebook import tqdm
from app.model_services.base import AbstractModelService
from app.exception import AnnotationException


ANCHOR_DELIMITER = ";"
DOC_SPAN_DELIMITER = "_"

if "usedforsecurity" in inspect.signature(hashlib.sha1).parameters:
    STATE_MISSING = hashlib.sha1("MISSING".encode("utf-8"), usedforsecurity=False).hexdigest()  # type: ignore
    META_STATE_MISSING = hashlib.sha1("{}".encode("utf-8"), usedforsecurity=False).hexdigest()  # type: ignore
else:
    STATE_MISSING = hashlib.sha1("MISSING".encode("utf-8")).hexdigest()
    META_STATE_MISSING = hashlib.sha1("{}".encode("utf-8")).hexdigest()


def sanity_check_model_with_trainer_export(
    trainer_export: Union[str, IO, Dict],
    model_service: AbstractModelService,
    return_df: bool = False,
    include_anchors: bool = False,
) -> Union[pd.DataFrame, Tuple[float, float, float, Dict, Dict, Dict, Dict, Optional[Dict]]]:
    """
    Performs a sanity check on the model's performance against a trainer export.

    Args:
        trainer_export (Union[str, IO, Dict]): The trainer export data, which can be a file path, a file-like object, or a dictionary.
        model_service (AbstractModelService): An instance of the model service used for prediction.
        return_df (bool): If True, returns a pandas DataFrame with metrics. Defaults to False.
        include_anchors (bool): If True, includes anchor information in the output. Defaults to False.

    Returns:
        Union[pd.DataFrame, Tuple[float, float, float, Dict, Dict, Dict, Dict, Optional[Dict]]]: A pandas DataFrame or a tuple with collected metrics,
    """

    if isinstance(trainer_export, str):
        with open(trainer_export, "r") as file:
            data = json.load(file)
    elif isinstance(trainer_export, Dict):
        data = trainer_export
    else:
        data = json.load(trainer_export)

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
                predictions[document["id"]].append([annotation.start, annotation.end, annotation.label_id])
                concept_names[annotation.label_id] = annotation.label_name
                concept_anchors[annotation.label_id] = concept_anchors.get(annotation.label_id, [])
                concept_anchors[annotation.label_id].append(f"P{project['id']}/D{document['id']}/S{annotation.start}/E{ annotation.end}")

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


def concat_trainer_exports(
    data_file_paths: List[str],
    combined_data_file_path: Optional[str] = None,
    allow_recurring_project_ids: bool = False,
    allow_recurring_doc_ids: bool = True,
) -> Union[Dict[str, Any], str]:
    """
    Concatenates multiple trainer export files into a single combined file.

    Args:
        data_file_paths (List[str]): List of paths to files containing trainer export data.
        combined_data_file_path (Optional[str]): The file path where the combined data will be saved. If None, the combined data will be returned as a dictionary.
        allow_recurring_project_ids (bool): If set to False, raises an exception if multiple projects are found sharing the same ID.
        allow_recurring_doc_ids (bool): If set to False, raises an exception if multiple documents are found sharing the same ID.

    Returns:
        Union[Dict[str, Any], str]: The path to the combined data file if `combined_data_file_path` is provided, or the combined data as a dictionary otherwise.

    Raises:
        AnnotationException: If multiple projects or documents share the same ID, and either is not allowed.
    """

    combined: Dict = {"projects": []}
    project_ids = []
    for path in data_file_paths:
        with open(path, "r") as f:
            data = json.load(f)
            for project in data["projects"]:
                if project["id"] in project_ids and not allow_recurring_project_ids:
                    raise AnnotationException(f'Found multiple projects share the same ID: {project["id"]}')
                project_ids.append(project["id"])
        combined["projects"].extend(data["projects"])
    document_ids = [doc["id"] for project in combined["projects"] for doc in project["documents"]]
    if not allow_recurring_doc_ids and len(document_ids) > len(set(document_ids)):
        recurring_ids = list(set([doc_id for doc_id in document_ids if document_ids.count(doc_id) > 1]))
        raise AnnotationException(f'Found multiple documents share the same ID(s): {recurring_ids}')

    if isinstance(combined_data_file_path, str):
        with open(combined_data_file_path, "w") as f:
            json.dump(combined, f)

        return combined_data_file_path
    else:
        return combined


def concat_json_lists(
    data_file_paths: List[str],
    combined_data_file_path: Optional[str] = None,
) -> Union[List[Dict[str, Any]], str]:
    """
    Concatenates multiple json list files into a single combined file.

    Args:
        data_file_paths (List[str]): List of paths to files each containing a json list.
        combined_data_file_path (Optional[str]): The file path where the combined data will be saved. If None, the combined data will be returned as a list.


    Returns:
        Union[List[Dict[str, Any]], str]: The path to the combined data file if `combined_data_file_path` is provided, or the combined data as a list otherwise.
    """
    combined: List = []
    for path in data_file_paths:
        with open(path, "r") as f:
            data = json.load(f)
        combined.extend(data)

    if isinstance(combined_data_file_path, str):
        with open(combined_data_file_path, "w") as f:
            json.dump(combined, f)

        return combined_data_file_path
    else:
        return combined


def get_stats_from_trainer_export(
    trainer_export: Union[str, IO, Dict],
    return_df: bool = False,
) -> Union[pd.DataFrame, Tuple[Dict[str, int], Dict[str, int], Dict[str, int], int]]:
    """
    Collects statistics from a trainer export.

    Args:
        trainer_export (Union[str, IO, Dict]): The trainer export data, which can be a file path, a file-like object, or a dictionary.
        return_df (bool): If set to True, returns the statistics as a pandas DataFrame.

    Returns:
        Union[pd.DataFrame, Tuple[Dict[str, int], Dict[str, int], Dict[str, int], int]]: A pandas DataFrame if `return_df` is True,
        otherwise returns a tuple containing counts of annotations per concept, counts of unique annotations per concept, cunts of ignored
        annotations per concept, and the total number of documents.
    """

    if isinstance(trainer_export, str):
        with open(trainer_export, "r") as file:
            data = json.load(file)
    elif isinstance(trainer_export, Dict):
        data = trainer_export
    else:
        data = json.load(trainer_export)

    cui_values: Dict = defaultdict(list)
    cui_ignorance_counts: Dict = defaultdict(int)
    num_of_docs = 0

    for project in data["projects"]:
        for doc in project["documents"]:
            annotations = []
            if isinstance(doc["annotations"], list):
                annotations = doc["annotations"]
            elif isinstance(doc["annotations"], dict):
                annotations = list(doc["annotations"].values())
            for annotation in annotations:
                if any([not annotation.get("validated", True),
                       annotation.get("deleted", False),
                       annotation.get("killed", False),
                       annotation.get("irrelevant", False)]):
                    cui_ignorance_counts[annotation["cui"]] += 1
                cui_values[annotation["cui"]].append(doc["text"][annotation["start"]:annotation["end"]].lower())
            num_of_docs += 1

    cui_counts = {cui: len(values) for cui, values in cui_values.items()}
    cui_unique_counts = {cui: len(set(values)) for cui, values in cui_values.items()}

    if return_df:
        return pd.DataFrame({
            "concept": cui_counts.keys(),
            "anno_count": cui_counts.values(),
            "anno_unique_counts": cui_unique_counts.values(),
            "anno_ignorance_counts": [cui_ignorance_counts[c] for c in cui_counts.keys()],
        })
    else:
        return cui_counts, cui_unique_counts, cui_ignorance_counts, num_of_docs


def get_iaa_scores_per_concept(
    trainer_export: Union[str, IO],
    project_id: int,
    another_project_id: int,
    return_df: bool = False,
) -> Union[pd.DataFrame, Tuple[Dict, Dict]]:
    """
    Calculates Inter-Annotator Agreement (IAA) scores for annotations and meta-annotations per concept between two projects.

    Args:
        trainer_export (Union[str, IO]): The trainer export data, which can be a file path or a file-like object.
        project_id (int): The ID of the first project.
        another_project_id (int): The ID of the second project.
        return_df (bool): If set to True, returns the IAA scores as a pandas DataFrame.

    Returns:
        Union[pd.DataFrame, Tuple[Dict, Dict]]: A pandas DataFrame if `return_df` is True, otherwise returns a tuple containing
        the percentage of IAA for annotations per concept, the Cohen's Kappa score for annotations per concept, the percentage of IAA
        for meta-annotations per concept, and the Cohen's Kappa score for meta-annotations per concept.
    """

    project_a, project_b = _extract_project_pair(trainer_export, project_id, another_project_id)
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
        per_cui_metaanno_iia_pct[cui] = len([1 for cmp in cui_metastate_pairs if cmp[0] == cmp[1]]) / len(cui_metastate_pairs) * 100
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


def get_iaa_scores_per_doc(
    export_file: Union[str, IO],
    project_id: int,
    another_project_id: int,
    return_df: bool = False,
) -> Union[pd.DataFrame, Tuple[Dict, Dict]]:
    """
    Calculates Inter-Annotator Agreement (IAA) scores for annotations and meta-annotations per document between two projects.

    Args:
        export_file (Union[str, IO]): The trainer export data, which can be a file path or a file-like object.
        project_id (int): The ID of the first project.
        another_project_id (int): The ID of the second project.
        return_df (bool): If set to True, returns the IAA scores as a pandas DataFrame.

    Returns:
        Union[pd.DataFrame, Tuple[Dict, Dict]]: A pandas DataFrame if `return_df` is True, otherwise returns a tuple containing
        the percentage of IAA for annotations per document, the Cohen's Kappa score for annotations per document, the percentage
        of IAA for meta-annotations per document, and the Cohen's Kappa score for meta-annotations per document.
    """

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
        per_doc_metaanno_iia_pct[str(doc_id)] = len([1 for dmp in doc_metastate_pairs if dmp[0] == dmp[1]]) / len(doc_metastate_pairs) * 100
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


def get_iaa_scores_per_span(
    trainer_export: Union[str, IO],
    project_id: int,
    another_project_id: int,
    return_df: bool = False,
) -> Union[pd.DataFrame, Tuple[Dict, Dict]]:
    """
    Calculates Inter-Annotator Agreement (IAA) scores for annotations and meta-annotations per span between two projects.

    Args:
        trainer_export (Union[str, IO]): The trainer export data, which can be a file path or a file-like object.
        project_id (int): The ID of the first project.
        another_project_id (int): The ID of the second project.
        return_df (bool): If set to True, returns the IAA scores as a pandas DataFrame.

    Returns:
        Union[pd.DataFrame, Tuple[Dict, Dict]]: A pandas DataFrame if `return_df` is True, otherwise returns a tuple containing
        the percentage of IAA for annotations per span, the Cohen's Kappa score for annotations per span, the percentage of IAA
        for meta-annotations per span, and theCohen's Kappa score for meta-annotations per span.
    """

    project_a, project_b = _extract_project_pair(trainer_export, project_id, another_project_id)
    filtered_projects = _filter_common_docs([project_a, project_b])
    state_keys = {"validated", "correct", "deleted", "alternative", "killed", "manually_created", "cui"}

    docspan2state_proj_a = {}
    docspan2statemeta_proj_a = {}
    for document in filtered_projects[0]["documents"]:
        for annotation in document["annotations"]:
            docspan_key = _get_docspan_key(document, annotation)
            docspan2state_proj_a[docspan_key] = [str(annotation.get(key)) for key in state_keys]
            docspan2statemeta_proj_a[docspan_key] = [str(meta_ann) for meta_ann in annotation["meta_anns"].items()] if annotation["meta_anns"] else [META_STATE_MISSING]

    docspan2state_proj_b = {}
    docspan2statemeta_proj_b = {}
    for document in filtered_projects[1]["documents"]:
        for annotation in document["annotations"]:
            docspan_key = _get_docspan_key(document, annotation)
            docspan2state_proj_b[docspan_key] = [str(annotation.get(key)) for key in state_keys]
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


def _extract_project_pair(
    export_file: Union[str, IO],
    project_id: int,
    another_project_id: int,
) -> Tuple[Dict, Dict]:
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
        raise AnnotationException(f"Cannot find the project with ID: {project_id}")
    if project_b is None:
        raise AnnotationException(f"Cannot find the project with ID: {another_project_id}")

    return project_a, project_b


def _get_docspan_key(document: Dict, annotation: Dict) -> str:
    return f"{document['id']}{DOC_SPAN_DELIMITER}{annotation.get('start')}{DOC_SPAN_DELIMITER}{annotation.get('end')}"


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
    if "usedforsecurity" in inspect.signature(hashlib.sha1).parameters:
        return hashlib.sha1("_".join([
            str(annotation.get(key)) for key in state_keys
        ]).encode("utf-8"), usedforsecurity=False).hexdigest()  # type: ignore
    else:
        return hashlib.sha1("_".join([str(annotation.get(key)) for key in state_keys]).encode("utf-8")).hexdigest()


def _get_hashed_meta_annotation_state(meta_anno: Dict) -> str:
    meta_anno = {key: val for key, val in sorted(meta_anno.items(), key=lambda item: item[0])}  # may not be necessary

    if "usedforsecurity" in inspect.signature(hashlib.sha1).parameters:
        return hashlib.sha1(str(meta_anno).encode("utf-8"), usedforsecurity=False).hexdigest()  # type: ignore
    else:
        return hashlib.sha1(str(meta_anno).encode("utf-8")).hexdigest()


def _get_cohens_kappa_coefficient(y1_labels: List, y2_labels: List) -> float:
    return cohen_kappa_score(y1_labels, y2_labels) if len(set(y1_labels).union(set(y2_labels))) != 1 else 1.0
