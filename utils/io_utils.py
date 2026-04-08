"""Filesystem, CLI, and runtime environment helpers."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Mapping, Optional, Sequence

import tensorflow as tf


def handle_gpu_compatibility() -> None:
    """Enable TensorFlow memory growth for each detected GPU.

    Returns:
        None: TensorFlow is configured in place.
    """
    try:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as error:  # pragma: no cover - defensive runtime branch.
        print(error)


def get_files(path: str) -> Sequence[Dict[str, str]]:
    """Collect file metadata from a directory tree.

    Args:
        path (str): Root directory to walk.

    Returns:
        Sequence[Dict[str, str]]: File paths paired with their filenames.
    """
    files = []
    for root, _directories, filenames in os.walk(path):
        for filename in filenames:
            files.append(
                {
                    "path": os.path.join(root, filename),
                    "filename": filename,
                }
            )
    return files


def get_main_path(version: str) -> str:
    """Return the directory that stores artifacts for a model version.

    Args:
        version (str): Model version such as `v1` or `v2`.

    Returns:
        str: Existing directory path for the selected version.
    """
    main_path = os.path.join("models", version)
    if not os.path.exists(main_path):
        os.makedirs(main_path)
    return main_path


def get_model_path(main_path: str, trained_with: str) -> str:
    """Build the model checkpoint path for a training combination.

    Args:
        main_path (str): Version-specific model directory.
        trained_with (str): Dataset combination name.

    Returns:
        str: Checkpoint file path.
    """
    return os.path.join(main_path, trained_with + "_model.h5")


def save_results_as_json(main_path: str, results: Mapping[str, Any]) -> None:
    """Persist evaluation results in JSON format.

    Args:
        main_path (str): Directory where the JSON file is written.
        results (Mapping[str, Any]): Serializable evaluation payload.

    Returns:
        None: Results are written to disk.
    """
    with open(os.path.join(main_path, "results.json"), "w", encoding="utf-8") as jsonfile:
        json.dump(results, jsonfile)


def handle_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the trainer script.

    Args:
        argv (Optional[Sequence[str]]): Optional custom argument list for tests.

    Returns:
        argparse.Namespace: Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="CNN Digit Recognition Implementation")
    parser.add_argument("-handle-gpu", action="store_true", help="Tensorflow 2 GPU compatibility flag")
    parser.add_argument(
        "--version",
        required=False,
        default="v2",
        metavar="['v1', 'v2']",
        help="Which CNN model you want to use",
    )
    return parser.parse_args(args=argv)
