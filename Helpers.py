"""Shared training and dataset helpers for the CNN digit-recognition project."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from typing import Any, Dict, Mapping, MutableMapping, NoReturn, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as keras_backend
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential


DatasetSplit = Dict[str, np.ndarray]
DatasetCollection = Dict[str, DatasetSplit]
EvaluationResults = Dict[str, Dict[str, Any]]
ImageShape = Tuple[int, int, int]

MNIST_HEIGHT = 28
MNIST_WIDTH = 28
NUM_CLASSES = 10


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


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: Sequence[Any],
    normalize: bool = True,
    title: Optional[str] = None,
    cmap: Any = plt.cm.Blues,
) -> Any:
    """Plot a confusion matrix with optional row-wise normalization.

    Args:
        cm (np.ndarray): Confusion matrix values.
        labels (Sequence[Any]): Class labels shown on the axes.
        normalize (bool): Whether to normalize the matrix by row totals.
        title (Optional[str]): Custom plot title.
        cmap (Any): Matplotlib color map.

    Returns:
        Any: Matplotlib axes that contain the rendered matrix.
    """
    if not title:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix, without normalization"

    if normalize:
        numerator = cm.astype("float")
        denominator = cm.sum(axis=1)[:, np.newaxis]
        cm = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)

    fig, ax = plt.subplots()
    image = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.margins(0.05)

    value_format = ".2f" if normalize else "d"
    threshold = cm.max() / 2.0
    for row_index in range(cm.shape[0]):
        for column_index in range(cm.shape[1]):
            ax.text(
                column_index,
                row_index,
                format(cm[row_index, column_index], value_format),
                ha="center",
                va="center",
                color="white" if cm[row_index, column_index] > threshold else "black",
            )

    fig.tight_layout()
    return ax


def handle_confusion_matrix(
    actuals: Sequence[Any],
    predictions: Sequence[Any],
    labels: Sequence[Any],
    title: str,
    save_path: Optional[str] = None,
    print_results: bool = True,
) -> None:
    """Generate and optionally print confusion-matrix diagnostics.

    Args:
        actuals (Sequence[Any]): Ground-truth labels.
        predictions (Sequence[Any]): Predicted labels.
        labels (Sequence[Any]): Label order used by sklearn reports.
        title (str): Figure title.
        save_path (Optional[str]): Optional file path for the plot image.
        print_results (bool): Whether to print text diagnostics.

    Returns:
        None: Diagnostics are printed or written to disk.
    """
    cm = confusion_matrix(actuals, predictions)
    plot_confusion_matrix(cm, labels, title=title)
    if save_path:
        plt.savefig(save_path)
    if not print_results:
        return

    print("Classification Report")
    print(classification_report(actuals, predictions, labels=labels))
    print("Confusion Matrix")
    print(cm)
    print("Accuracy Score")
    print(accuracy_score(actuals, predictions))


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


def get_batch_size(train_data: Any) -> int:
    """Select a batch size based on the dataset size.

    Args:
        train_data (Any): Array-like training data with a leading batch dimension.

    Returns:
        int: Batch size tuned for the data volume.
    """
    batch_size = 1024
    length = train_data.shape[0]
    if length < 5000:
        batch_size = 2
    elif length < 25000:
        batch_size = 16
    elif length < 50000:
        batch_size = 128
    elif length < 250000:
        batch_size = 256
    elif length < 500000:
        batch_size = 512
    return batch_size


def handle_output_data(output: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert class labels into one-hot encoded vectors.

    Args:
        output (np.ndarray): Raw integer labels.
        num_classes (int): Number of classes in the output space.

    Returns:
        np.ndarray: One-hot encoded labels.
    """
    return keras.utils.to_categorical(output, num_classes)


def handle_input_data(input_data: np.ndarray, n: int = 255) -> np.ndarray:
    """Reshape and normalize image data.

    Args:
        input_data (np.ndarray): Input image tensors.
        n (int): Normalization denominator.

    Returns:
        np.ndarray: Normalized image tensors with an explicit channel axis.
    """
    input_data = handle_channel(input_data)
    input_data = input_data.astype("float32")
    input_data /= n
    return input_data


def get_input_shape() -> ImageShape:
    """Return the TensorFlow image shape expected by the CNN models.

    Returns:
        ImageShape: Channel-aware input shape tuple.
    """
    if keras_backend.image_data_format() == "channels_first":
        return (1, MNIST_HEIGHT, MNIST_WIDTH)
    return (MNIST_HEIGHT, MNIST_WIDTH, 1)


def handle_channel(data: Any) -> Any:
    """Add a channel dimension that matches the active Keras image format.

    Args:
        data (Any): Array-like image tensor.

    Returns:
        Any: Reshaped tensor with a channel axis.
    """
    if keras_backend.image_data_format() == "channels_first":
        return data.reshape(data.shape[0], 1, MNIST_HEIGHT, MNIST_WIDTH)
    return data.reshape(data.shape[0], MNIST_HEIGHT, MNIST_WIDTH, 1)


def convert_array_to_mnist_type_image(array: np.ndarray) -> np.ndarray:
    """Resize a source image into the MNIST-like single-channel format.

    Args:
        array (np.ndarray): Source image array.

    Returns:
        np.ndarray: Resized grayscale image with one channel.
    """
    image = Image.fromarray(array)
    image = image.resize((MNIST_HEIGHT, MNIST_HEIGHT))
    converted = np.array(image.convert("I"))
    return converted.reshape(converted.shape[0], converted.shape[1], 1)


def prepare_ORHD_to_MNIST_format(data: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Convert the sklearn digits dataset into the MNIST training format.

    Args:
        data (Any): Dataset object with `images` and `target` attributes.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Prepared images and one-hot labels.
    """
    images = []
    image_count = data.images.shape[0]
    for index in range(image_count):
        image = convert_array_to_mnist_type_image(data.images[index, :, :])
        images.append(image)
    x = np.asarray(images)
    y = handle_output_data(data.target, NUM_CLASSES)
    return handle_input_data(x, 16), y


def prepare_SVHN_to_MNIST_format(data: MutableMapping[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert the SVHN dataset dictionary into the MNIST training format.

    Args:
        data (MutableMapping[str, np.ndarray]): SVHN dataset dictionary.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Prepared images and one-hot labels.
    """
    images = []
    image_count = data["X"].shape[-1]
    for index in range(image_count):
        image = convert_array_to_mnist_type_image(data["X"][:, :, :, index])
        images.append(image)

    x = np.asarray(images)
    y = data["y"].reshape(data["y"].shape[0],)
    x = handle_input_data(x)
    y[y == 10] = 0
    return x, handle_output_data(y, NUM_CLASSES)


def prepare_final_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> DatasetSplit:
    """Create the final train/validation/test split dictionary.

    Args:
        x_train (np.ndarray): Training images before the validation split.
        y_train (np.ndarray): Training labels before the validation split.
        x_test (np.ndarray): Test images.
        y_test (np.ndarray): Test labels.

    Returns:
        DatasetSplit: Dictionary with train, validation, and test tensors.
    """
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)
    print(x_train.shape[0], "train samples")
    print(x_valid.shape[0], "valid samples")
    print(x_test.shape[0], "test samples")
    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_valid": x_valid,
        "y_valid": y_valid,
        "x_test": x_test,
        "y_test": y_test,
    }


def get_ARDIS_data() -> DatasetSplit:
    """Load, normalize, and split the ARDIS dataset.

    Returns:
        DatasetSplit: Prepared ARDIS dataset splits.
    """
    x_train = np.loadtxt("data/ARDIS_train_2828.csv", dtype="float")
    x_test = np.loadtxt("data/ARDIS_test_2828.csv", dtype="float")
    y_train = np.loadtxt("data/ARDIS_train_labels.csv", dtype="float")
    y_test = np.loadtxt("data/ARDIS_test_labels.csv", dtype="float")

    x_train = x_train.reshape(x_train.shape[0], 28, 28).astype("float32")
    x_test = x_test.reshape(x_test.shape[0], 28, 28).astype("float32")
    x_train = handle_input_data(x_train)
    x_test = handle_input_data(x_test)

    print("ARDIS data summary: ")
    return prepare_final_data(x_train, y_train, x_test, y_test)


def get_ORHD_data() -> DatasetSplit:
    """Load, normalize, and split the sklearn digits dataset.

    Returns:
        DatasetSplit: Prepared ORHD dataset splits.
    """
    data = load_digits()
    x, y = prepare_ORHD_to_MNIST_format(data)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    print("ORHD data summary: ")
    return prepare_final_data(x_train, y_train, x_test, y_test)


def get_SVHN_data() -> DatasetSplit:
    """Load, normalize, and split the SVHN dataset.

    Returns:
        DatasetSplit: Prepared SVHN dataset splits.
    """
    train_data = sio.loadmat("data/train_32x32.mat")
    test_data = sio.loadmat("data/test_32x32.mat")
    x_train, y_train = prepare_SVHN_to_MNIST_format(train_data)
    x_test, y_test = prepare_SVHN_to_MNIST_format(test_data)
    print("SVHN data summary: ")
    return prepare_final_data(x_train, y_train, x_test, y_test)


def get_MNIST_data() -> DatasetSplit:
    """Load, normalize, and split the built-in MNIST dataset.

    Returns:
        DatasetSplit: Prepared MNIST dataset splits.
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = handle_input_data(x_train)
    x_test = handle_input_data(x_test)
    y_train = handle_output_data(y_train, NUM_CLASSES)
    y_test = handle_output_data(y_test, NUM_CLASSES)

    print("MNIST data summary: ")
    return prepare_final_data(x_train, y_train, x_test, y_test)


def get_combined_dataset(datasets: Mapping[str, DatasetSplit], combination: Sequence[str]) -> DatasetSplit:
    """Concatenate dataset splits for a combined training run.

    Args:
        datasets (Mapping[str, DatasetSplit]): Available base datasets.
        combination (Sequence[str]): Dataset names included in the combination.

    Returns:
        DatasetSplit: Concatenated dataset splits.
    """
    return {
        "x_train": np.concatenate([datasets[dataset]["x_train"] for dataset in combination]),
        "y_train": np.concatenate([datasets[dataset]["y_train"] for dataset in combination]),
        "x_valid": np.concatenate([datasets[dataset]["x_valid"] for dataset in combination]),
        "y_valid": np.concatenate([datasets[dataset]["y_valid"] for dataset in combination]),
        "x_test": np.concatenate([datasets[dataset]["x_test"] for dataset in combination]),
        "y_test": np.concatenate([datasets[dataset]["y_test"] for dataset in combination]),
    }


def get_datasets(use_datasets: Sequence[str], n_combinations: int = 1) -> DatasetCollection:
    """Load the requested datasets and build combination datasets.

    Args:
        use_datasets (Sequence[str]): Dataset names to include.
        n_combinations (int): Highest combination size to generate.

    Returns:
        DatasetCollection: Base and combined datasets keyed by name.
    """
    datasets: DatasetCollection = {}
    for dataset in use_datasets:
        if dataset == "MNIST":
            datasets["MNIST"] = get_MNIST_data()
        elif dataset == "SVHN":
            datasets["SVHN"] = get_SVHN_data()
        elif dataset == "ORHD":
            datasets["ORHD"] = get_ORHD_data()
        elif dataset == "ARDIS":
            datasets["ARDIS"] = get_ARDIS_data()

    for index in range(n_combinations):
        combination_size = index + 1
        for combination in itertools.combinations(use_datasets, combination_size):
            dataset_name = "+".join(combination)
            if dataset_name in datasets:
                continue
            datasets[dataset_name] = get_combined_dataset(datasets, combination)

    return datasets


def evaluate(model: Any, datasets: Mapping[str, DatasetSplit], use_datasets: Sequence[str]) -> EvaluationResults:
    """Evaluate a trained model on each requested dataset.

    Args:
        model (Any): Keras-compatible model with an `evaluate` method.
        datasets (Mapping[str, DatasetSplit]): Dataset collection.
        use_datasets (Sequence[str]): Dataset names used for evaluation.

    Returns:
        EvaluationResults: Rounded loss and accuracy values for each dataset.
    """
    results: EvaluationResults = {}
    for dataset_name in use_datasets:
        dataset = datasets[dataset_name]
        loss, accuracy = model.evaluate(dataset["x_test"], dataset["y_test"])
        results[dataset_name] = {
            "dataset": dataset_name,
            "loss": round(float(loss), 5),
            "accuracy": round(float(accuracy), 5),
        }
    return results


def print_results(results: Mapping[str, Mapping[str, Any]], trained_with: str, version: str) -> None:
    """Print evaluation results for a trained model.

    Args:
        results (Mapping[str, Mapping[str, Any]]): Evaluation payload.
        trained_with (str): Dataset combination used for training.
        version (str): Model version name.

    Returns:
        None: Results are printed to standard output.
    """
    print("============================================================================")
    print("Trained with:", trained_with, "model version", version)
    print("============================================================================")
    for dataset_name in results:
        result = results[dataset_name]
        print(dataset_name, "Test loss:", result["loss"])
        print(dataset_name, "Test accuracy:", result["accuracy"])
        print("=========================================")
    print("============================================================================")


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


def raise_model_exception(version: str) -> NoReturn:
    """Raise an error for an unsupported model version.

    Args:
        version (str): Requested model version.

    Returns:
        NoReturn: This function always raises.
    """
    raise Exception("Please give a valid version for training model")


def get_model(input_shape: ImageShape, version: str) -> Sequential:
    """Create the requested CNN feature extractor.

    Args:
        input_shape (ImageShape): Model input shape.
        version (str): Model version key.

    Returns:
        Sequential: Model without the final classification layer.
    """
    model_builders = {
        "v1": get_model_v1,
        "v2": get_model_v2,
    }
    model_builder = model_builders.get(version)
    if model_builder is None:
        raise_model_exception(version)
    return model_builder(input_shape)


def get_model_v1(input_shape: ImageShape) -> Sequential:
    """Build the original lightweight CNN feature extractor.

    Args:
        input_shape (ImageShape): Model input shape.

    Returns:
        Sequential: Uncompiled Keras model.
    """
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    return model


def get_model_v2(input_shape: ImageShape) -> Sequential:
    """Build the deeper CNN feature extractor used by version two.

    Args:
        input_shape (ImageShape): Model input shape.

    Returns:
        Sequential: Uncompiled Keras model.
    """
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape, padding="same"))
    model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    return model


convert_array_to_MNIST_type_img = convert_array_to_mnist_type_image
