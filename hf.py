#!/usr/bin/env python3
"""
Hugging Face model utilities: download, upload, and unpack NeMo files.

Usage:
    # Download a model
    python hf.py download nvidia/parakeet-tdt-0.6b-v3 --local-dir models/nvidia

    # Upload a model
    python hf.py upload models/my-model username/my-model-mlx

    # Unpack a NeMo file (they're just tar archives)
    python hf.py unpack model.nemo --output-dir unpacked/
"""

import argparse
import os
import tarfile
from pathlib import Path

from huggingface_hub import HfApi, ModelCard, snapshot_download


def fetch_model_metadata(repo_id: str) -> dict:
    """
    Fetch model card metadata from Hugging Face Hub.

    Args:
        repo_id: Repository ID (e.g., 'nvidia/parakeet-tdt-0.6b-v3')

    Returns:
        Dictionary with model metadata including:
        - license, language, pipeline_tag, tags, datasets, base_model
    """
    # Sensible defaults
    defaults = {
        "license": "apache-2.0",
        "language": ["en"],
        "pipeline_tag": "automatic-speech-recognition",
        "tags": [],
        "datasets": [],
        "base_model": repo_id,
    }

    try:
        card = ModelCard.load(repo_id)
        data = card.data.to_dict() if card.data else {}
    except Exception as e:
        print(f"Warning: Could not fetch model card from {repo_id}: {e}")
        print("Using default metadata values.")
        return defaults

    # Extract metadata with fallbacks to defaults
    metadata = {
        "license": data.get("license", defaults["license"]),
        "language": data.get("language", defaults["language"]),
        "pipeline_tag": data.get("pipeline_tag", defaults["pipeline_tag"]),
        "tags": data.get("tags", defaults["tags"]),
        "datasets": data.get("datasets", defaults["datasets"]),
        "base_model": repo_id,
    }

    # Ensure language is always a list
    if isinstance(metadata["language"], str):
        metadata["language"] = [metadata["language"]]

    # Ensure tags is always a list
    if metadata["tags"] is None:
        metadata["tags"] = []

    # Ensure datasets is always a list
    if metadata["datasets"] is None:
        metadata["datasets"] = []

    return metadata


def download_model(
    repo_id: str, local_dir: str | None = None, revision: str | None = None
) -> str:
    """
    Download a model from Hugging Face Hub.

    Args:
        repo_id: Repository ID (e.g., 'nvidia/parakeet-tdt-0.6b-v3')
        local_dir: Local directory to save the model (default: models/<repo_id>)
        revision: Git revision to download (branch, tag, or commit hash)

    Returns:
        Path to the downloaded model directory
    """
    if local_dir is None:
        local_dir = f"models/{repo_id}"

    print(f"Downloading {repo_id} to {local_dir}...")

    path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        revision=revision,
    )

    print(f"Downloaded to {path}")
    return path


def upload_model(
    local_dir: str,
    repo_id: str,
    private: bool = False,
    commit_message: str | None = None,
) -> str:
    """
    Upload a model directory to Hugging Face Hub.

    Args:
        local_dir: Local directory containing the model files
        repo_id: Target repository ID (e.g., 'username/model-name')
        private: Whether to create a private repository
        commit_message: Commit message for the upload

    Returns:
        URL of the uploaded repository
    """
    api = HfApi()

    # Create repo if it doesn't exist
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)

    print(f"Uploading {local_dir} to {repo_id}...")

    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        commit_message=commit_message or f"Upload model from {local_dir}",
    )

    url = f"https://huggingface.co/{repo_id}"
    print(f"Uploaded to {url}")
    return url


def unpack_nemo(nemo_path: str, output_dir: str | None = None) -> str:
    """
    Unpack a NeMo file (which is just a tar archive).

    Args:
        nemo_path: Path to the .nemo file
        output_dir: Directory to extract to (default: same name without .nemo extension)

    Returns:
        Path to the extracted directory
    """
    nemo_path = Path(nemo_path)

    if not nemo_path.exists():
        raise FileNotFoundError(f"NeMo file not found: {nemo_path}")

    if output_dir is None:
        output_dir = str(nemo_path.parent / "nemo-files")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Unpacking {nemo_path} to {output_dir}...")

    with tarfile.open(nemo_path, "r:*") as tar:
        tar.extractall(path=output_dir)

    print(f"Extracted to {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Hugging Face model utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Download command
    dl_parser = subparsers.add_parser("download", help="Download a model from HF Hub")
    dl_parser.add_argument(
        "repo_id", help="Repository ID (e.g., nvidia/parakeet-tdt-0.6b-v3)"
    )
    dl_parser.add_argument(
        "--local-dir", "-d", help="Local directory to save the model"
    )
    dl_parser.add_argument(
        "--revision", "-r", help="Git revision (branch, tag, or commit)"
    )

    # Upload command
    up_parser = subparsers.add_parser("upload", help="Upload a model to HF Hub")
    up_parser.add_argument("local_dir", help="Local directory containing the model")
    up_parser.add_argument(
        "repo_id", help="Target repository ID (e.g., username/model-name)"
    )
    up_parser.add_argument(
        "--private", "-p", action="store_true", help="Create a private repo"
    )
    up_parser.add_argument("--message", "-m", help="Commit message")

    # Unpack command
    unpack_parser = subparsers.add_parser(
        "unpack", help="Unpack a NeMo file (tar archive)"
    )
    unpack_parser.add_argument("nemo_path", help="Path to the .nemo file")
    unpack_parser.add_argument("--output-dir", "-o", help="Output directory")

    args = parser.parse_args()

    if args.command == "download":
        download_model(args.repo_id, args.local_dir, args.revision)
    elif args.command == "upload":
        upload_model(args.local_dir, args.repo_id, args.private, args.message)
    elif args.command == "unpack":
        unpack_nemo(args.nemo_path, args.output_dir)


if __name__ == "__main__":
    main()
