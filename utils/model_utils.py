"""Model-construction helpers for the CNN architectures."""

from __future__ import annotations

from typing import NoReturn

from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential

from utils.common import ImageShape


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
