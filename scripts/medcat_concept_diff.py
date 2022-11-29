#!/usr/bin/env python

import os
import sys
import difflib
import jsonpickle
from argparse import ArgumentParser
from medcat.cat import CAT

jsonpickle.set_encoder_options("json", sort_keys=True, indent=4)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-a",
        "--model-pack-path",
        type=str,
        default="",
        help="The path to the first model pack"
    )
    parser.add_argument(
        "-b",
        "--another-model-pack-path",
        type=str,
        help="The path to the second model pack",
    )
    parser.add_argument(
        "-p",
        "--with-preferred-name",
        action="store_true",
        help="Print preferred names of concepts as the second column"
    )
    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.model_pack_path == "":
        print("ERROR: The path to model A is empty. Use '-a' to pass in the model pack path.")
        sys.exit(1)
    if FLAGS.another_model_pack_path == "":
        print("ERROR: The path to model B is empty. Use '-b' to pass in the model pack path.")
        sys.exit(1)

    model_pack_path = os.path.abspath(FLAGS.model_pack_path)
    another_model_pack_path = os.path.abspath(FLAGS.another_model_pack_path)
    cat_a = CAT.load_model_pack(model_pack_path)
    cat_b = CAT.load_model_pack(another_model_pack_path)
    concepts_a = {k: v for k, v in sorted(cat_a.cdb.cui2names.items(), key=lambda item: item[0])}
    concepts_b = {k: v for k, v in sorted(cat_b.cdb.cui2names.items(), key=lambda item: item[0])}
    concepts_string_a = ""
    for concept in set(concepts_a.keys()):
        if FLAGS.with_preferred_name:
            concepts_string_a += concept + "," + cat_a.cdb.get_name(concept) + "\n"
        else:
            concepts_string_a += concept + "\n"
    concepts_string_b = ""
    for concept in set(concepts_b.keys()):
        if FLAGS.with_preferred_name:
            concepts_string_b += concept + "," + cat_b.cdb.get_name(concept) + "\n"
        else:
            concepts_string_b += concept + "\n"

    print(f"--- a|{model_pack_path}")
    print(f"+++ b|{another_model_pack_path}")
    diff = None
    for diff in difflib.unified_diff(concepts_string_a.split("\n"), concepts_string_b.split("\n")):
        if diff[:3] not in ("+++", "---"):
            print(diff)
    if diff is None:
        print("No diffs found.")
