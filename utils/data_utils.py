"""Dataset loading and preprocessing helpers."""

from __future__ import annotations

import itertools
from typing import Any, MutableMapping, Sequence, Tuple

import numpy as np
import scipy.io as sio
import tensorflow.keras as keras
import tensorflow.keras.backend as keras_backend
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist

from utils.common import (
    DatasetCollection,
    DatasetSplit,
    ImageShape,
    MNIST_HEIGHT,
    MNIST_WIDTH,
    NUM_CLASSES,
)


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


def prepare_orhd_to_mnist_format(data: Any) -> Tuple[np.ndarray, np.ndarray]:
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


def prepare_svhn_to_mnist_format(data: MutableMapping[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
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


def get_ardis_data() -> DatasetSplit:
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


def get_orhd_data() -> DatasetSplit:
    """Load, normalize, and split the sklearn digits dataset.

    Returns:
        DatasetSplit: Prepared ORHD dataset splits.
    """
    data = load_digits()
    x, y = prepare_orhd_to_mnist_format(data)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    print("ORHD data summary: ")
    return prepare_final_data(x_train, y_train, x_test, y_test)


def get_svhn_data() -> DatasetSplit:
    """Load, normalize, and split the SVHN dataset.

    Returns:
        DatasetSplit: Prepared SVHN dataset splits.
    """
    train_data = sio.loadmat("data/train_32x32.mat")
    test_data = sio.loadmat("data/test_32x32.mat")
    x_train, y_train = prepare_svhn_to_mnist_format(train_data)
    x_test, y_test = prepare_svhn_to_mnist_format(test_data)
    print("SVHN data summary: ")
    return prepare_final_data(x_train, y_train, x_test, y_test)


def get_mnist_data() -> DatasetSplit:
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


def get_combined_dataset(
    datasets: dict[str, DatasetSplit],
    combination: Sequence[str],
) -> DatasetSplit:
    """Concatenate dataset splits for a combined training run.

    Args:
        datasets (dict[str, DatasetSplit]): Available base datasets.
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
    dataset_loaders = {
        "MNIST": get_mnist_data,
        "SVHN": get_svhn_data,
        "ORHD": get_orhd_data,
        "ARDIS": get_ardis_data,
    }

    for dataset_name in use_datasets:
        dataset_loader = dataset_loaders.get(dataset_name)
        if dataset_loader is not None:
            datasets[dataset_name] = dataset_loader()

    for index in range(n_combinations):
        combination_size = index + 1
        for combination in itertools.combinations(use_datasets, combination_size):
            dataset_name = "+".join(combination)
            if dataset_name in datasets:
                continue
            datasets[dataset_name] = get_combined_dataset(datasets, combination)

    return datasets
