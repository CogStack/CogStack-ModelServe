#!/usr/bin/env python

import json
import random
import sys
from argparse import ArgumentParser
from typing import List, Dict
from medcat.cat import CAT
from spacy.lang.en import English
from tqdm.autonotebook import tqdm


def generate_annotations(cuis: List, texts: List, minimum_words: int, cui2original_names: Dict) -> Dict:
    original_names = {cui: cui2original_names[cui] for cui in cuis if cui in cui2original_names}
    new_snames = {}
    for cui, names in original_names.items():
        new_names = set()
        for name in names:
            new_names.add(name.replace("~.~", ".").replace("~:~", ":").replace("~", " "))
        new_snames[cui] = new_names

    nlp = English()
    patterns = []
    for cui, names in new_snames.items():
        for name in names:
            if name not in nlp.Defaults.stop_words:
                pattern = {
                    "label": cui,
                    "pattern": name,
                }
                patterns.append(pattern)
    ruler = nlp.add_pipe("entity_ruler", config={"phrase_matcher_attr": "LOWER"})
    ruler.add_patterns(patterns)    # type: ignore

    documents = []
    for doc_id, text in enumerate(tqdm(texts, desc="Evaluating projects", total=len(texts), leave=False)):
        doc = nlp(text)
        annotations = []
        for ent in doc.ents:
            if len(ent.text.strip().split(" ")) < minimum_words:
                continue
            annotation = {
                "cui": ent.label_,
                "value": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "correct": True,
                "killed": False,
            }
            annotations.append(annotation)
        document = {
            "id": doc_id,
            "text": text,
            "annotations": annotations,
        }
        documents.append(document)

    return {"projects": [{"name": "Generated", "id": 0, "documents": documents}]}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--cuis",
        type=str,
        default="",
        help="The path to the file containing newline-separated CUIs"
    )
    parser.add_argument(
        "-t",
        "--texts",
        type=str,
        help="The path to the file containing texts as a JSON list",
    )
    parser.add_argument(
        "-s",
        "--sample-size",
        type=int,
        default=-1,
        help="The sample size of input texts",
    )
    parser.add_argument(
        "-n",
        "--min-words",
        type=int,
        default=1,
        help="The lowest number of words each generated annotation will have"
    )
    parser.add_argument(
        "-m",
        "--model-pack-path",
        type=str,
        help="The path to the first model package"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="The path to the output file of annotations",
    )
    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.cuis == "":
        print("ERROR: The path to the CUI file is empty. Use '-c' to pass in the file containing newline-separated CUIs.")
        sys.exit(1)
    if FLAGS.texts == "":
        print("ERROR: The path to the text file is empty. Use '-t' to pass in the file containing texts as a JSON list.")
        sys.exit(1)
    if FLAGS.model_pack_path == "":
        print("ERROR: The path to the model package is empty. Use '-m' to pass in the model pack path.")
        sys.exit(1)
    if FLAGS.output == "":
        print("ERROR: The path to the output file is empty. Use '-o' to pass in the file of annotations.")
        sys.exit(1)

    with open(FLAGS.cuis, "r") as f:
        cuis = json.load(f)
    with open(FLAGS.texts, "r") as f:
        texts = json.load(f)

    if 0 <= FLAGS.sample_size <= len(texts):
        texts = random.sample(texts, FLAGS.sample_size)

    cat = CAT.load_model_pack(FLAGS.model_pack_path)
    annotations = generate_annotations(cuis, texts, FLAGS.min_words, cat.cdb.addl_info["cui2original_names"])

    with open(FLAGS.output, "w") as f:
        json.dump(annotations, f, indent=4)
