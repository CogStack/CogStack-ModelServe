import os
import sys
import shutil
from argparse import ArgumentParser
from huggingface_hub import snapshot_download

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default="",
        help="The repository ID of the model to download from Hugging Face Hub, e.g., 'google-bert/bert-base-cased'")
    parser.add_argument(
        "--hf-repo-revision",
        type=str,
        default="",
        help="The revision of the model to download from Hugging Face Hub"
    )
    parser.add_argument(
        "--output-model-package",
        type=str,
        default="",
        help="Path to save the model package, minus any format-specific extension, e.g., './bert-base-cased'"
    )
    parser.add_argument(
        "--remove-cached",
        action="store_true",
        default=False,
        help="Whether to remove the downloaded cache after the model package is saved"
    )

    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.hf_repo_id == "":
        print("ERROR: The repository ID of the Hugging Face model is not passed in.")
        sys.exit(1)
    if FLAGS.output_model_package == "":
        print("ERROR: The model package path is not passed in.")
        sys.exit(1)

    if not FLAGS.hf_repo_revision:
        download_path = snapshot_download(repo_id=FLAGS.hf_repo_id)
    else:
        download_path = snapshot_download(repo_id=FLAGS.hf_repo_id, revision=FLAGS.hf_repo_revision)

    model_package_archive = os.path.abspath(os.path.expanduser(FLAGS.output_model_package))
    cached_model_path = os.path.abspath(os.path.expanduser(os.path.join(download_path, "..", "..")))
    shutil.make_archive(model_package_archive, "zip", download_path)
    if FLAGS.remove_cached:
        shutil.rmtree(cached_model_path)

    print(f"Model package saved to {model_package_archive}.zip")
