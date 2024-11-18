import os
import sys
import shutil
from argparse import ArgumentParser
from datasets import load_dataset

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--hf-dataset-id",
        type=str,
        default="",
        help="The repository ID of the dataset to download from Hugging Face Hub, e.g., 'stanfordnlp/imdb'")
    parser.add_argument(
        "--hf-dataset-revision",
        type=str,
        default="",
        help="The revision of the dataset to download from Hugging Face Hub"
    )
    parser.add_argument(
        "--cached-dataset-dir",
        type=str,
        default="",
        help="Path to the cached dataset directory, will only be used if --hf-dataset-id is not provided"
    )
    parser.add_argument(
        "--output-dataset-package",
        type=str,
        default="",
        help="Path to save the dataset package, minus any format-specific extension, e.g., './dataset_packages/imdb'"
    )
    parser.add_argument(
        "--remove-cached",
        action="store_true",
        default=False,
        help="Whether to remove the downloaded cache after the dataset package is saved"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Whether to trust and use the remote script of the dataset"
    )

    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.hf_dataset_id == "" and FLAGS.cached_dataset_dir == "":
        print("ERROR: Neither the repository ID of the Hugging Face dataset nor the cached dataset directory is passed in.")
        sys.exit(1)
    if FLAGS.output_dataset_package == "":
        print("ERROR: The dataset package path is not passed in.")
        sys.exit(1)

    dataset_package_archive = os.path.abspath(os.path.expanduser(FLAGS.output_dataset_package))
    if FLAGS.hf_dataset_id != "":
        cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "cache"))
        cached_dataset_path = os.path.join(cache_dir, "datasets", FLAGS.hf_dataset_id.replace("/", "_"))
        try:
            if not FLAGS.hf_dataset_revision:
                dataset = load_dataset(path=FLAGS.hf_dataset_id,
                                       cache_dir=cache_dir,
                                       trust_remote_code=FLAGS.trust_remote_code)
            else:
                dataset = load_dataset(path=FLAGS.hf_dataset_id,
                                       cache_dir=cache_dir,
                                       revision=FLAGS.hf_repo_revision,
                                       trust_remote_code=FLAGS.trust_remote_code)

            dataset.save_to_disk(cached_dataset_path)
            shutil.make_archive(dataset_package_archive, "zip", cached_dataset_path)
        finally:
            if FLAGS.remove_cached:
                shutil.rmtree(cache_dir)
    elif FLAGS.cached_dataset_dir != "":
        cached_dataset_path = os.path.abspath(os.path.expanduser(FLAGS.cached_dataset_dir))
        shutil.make_archive(dataset_package_archive, "zip", cached_dataset_path)

    print(f"Dataset package saved to {dataset_package_archive}.zip")
