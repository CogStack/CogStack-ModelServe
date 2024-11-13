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
        "--cached-model-dir",
        type=str,
        default="",
        help="Path to the cached model directory, will only be used if --hf-repo-id is not provided"
    )
    parser.add_argument(
        "--output-model-package",
        type=str,
        default="",
        help="Path to save the model package, minus any format-specific extension, e.g., './model_packages/bert-base-cased'"
    )
    parser.add_argument(
        "--remove-cached",
        action="store_true",
        default=False,
        help="Whether to remove the downloaded cache after the model package is saved"
    )

    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.hf_repo_id == "" and FLAGS.cached_model_dir == "":
        print("ERROR: Neither the repository ID of the Hugging Face model nor the cached model directory is passed in.")
        sys.exit(1)
    if FLAGS.output_model_package == "":
        print("ERROR: The model package path is not passed in.")
        sys.exit(1)

    model_package_archive = os.path.abspath(os.path.expanduser(FLAGS.output_model_package))
    if FLAGS.hf_repo_id != "":
        if not FLAGS.hf_repo_revision:
            download_path = snapshot_download(repo_id=FLAGS.hf_repo_id)
        else:
            download_path = snapshot_download(repo_id=FLAGS.hf_repo_id, revision=FLAGS.hf_repo_revision)

        cached_model_path = os.path.abspath(os.path.expanduser(os.path.join(download_path, "..", "..")))
        shutil.make_archive(model_package_archive, "zip", download_path)

        if FLAGS.remove_cached:
            shutil.rmtree(cached_model_path)
    elif FLAGS.cached_model_dir != "":
        cached_model_path = os.path.abspath(os.path.expanduser(FLAGS.cached_model_dir))
        shutil.make_archive(model_package_archive, "zip", cached_model_path)

    print(f"Model package saved to {model_package_archive}.zip")
