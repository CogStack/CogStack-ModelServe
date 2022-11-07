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
    json_string_a = jsonpickle.encode(
            {field: getattr(cat_a.cdb.config, field) for field in cat_a.cdb.config.fields()})
    json_string_b = jsonpickle.encode(
            {field: getattr(cat_b.cdb.config, field) for field in cat_b.cdb.config.fields()})

    print(f"--- a|{model_pack_path}")
    print(f"+++ b|{another_model_pack_path}")
    diff = None
    for diff in difflib.unified_diff(json_string_a.split("\n"), json_string_b.split("\n")):
        if diff[:3] not in ("+++", "---"):
            print(diff)
    if diff is None:
        print("No diffs found.")
