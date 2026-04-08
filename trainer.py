"""Train CNN digit-recognition models for every dataset combination."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense

from utils import common, data_utils, eval_utils, io_utils, model_utils


USE_DATASETS = ["ARDIS", "MNIST", "ORHD", "SVHN"]
EPOCHS = 30


def build_model(input_shape: common.ImageShape, version: str) -> Any:
    """Create and compile a training-ready CNN classifier.

    Args:
        input_shape (ImageShape): Model input shape.
        version (str): Model version to instantiate.

    Returns:
        Any: Compiled Keras model.
    """
    model = model_utils.get_model(input_shape, version)
    model.add(Dense(common.NUM_CLASSES, activation="softmax"))
    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(
            learning_rate=0.1,
        ),
        metrics=["accuracy"],
    )
    return model


def build_callbacks(main_path: str, trained_with: str) -> Sequence[Any]:
    """Create the callback set used during model training.

    Args:
        main_path (str): Version-specific artifact directory.
        trained_with (str): Dataset combination name.

    Returns:
        Sequence[Any]: Keras callback instances.
    """
    early_stopping = EarlyStopping(monitor="val_accuracy", patience=4, verbose=0, mode="auto")
    model_path = io_utils.get_model_path(main_path, trained_with)
    model_checkpoint = ModelCheckpoint(
        model_path,
        save_best_only=True,
        monitor="val_accuracy",
        mode="auto",
    )
    return [early_stopping, model_checkpoint]


def run_training(version: str, datasets: common.DatasetCollection) -> dict[str, Any]:
    """Train and evaluate all prepared dataset combinations.

    Args:
        version (str): Model version to train.
        datasets (DatasetCollection): Prepared base and combined datasets.

    Returns:
        dict[str, Any]: Aggregated evaluation results per training combination.
    """
    results = {}
    input_shape = data_utils.get_input_shape()
    main_path = io_utils.get_main_path(version)

    for trained_with, dataset in datasets.items():
        model = build_model(input_shape, version)
        x_train = dataset["x_train"]
        y_train = dataset["y_train"]
        x_valid = dataset["x_valid"]
        y_valid = dataset["y_valid"]
        batch_size = data_utils.get_batch_size(x_train)

        print("Training started with:", trained_with, "dataset", "model version", version)
        model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=EPOCHS,
            verbose=1,
            validation_data=(x_valid, y_valid),
            callbacks=build_callbacks(main_path, trained_with),
        )

        test_results = eval_utils.evaluate(model, datasets, USE_DATASETS)
        eval_utils.print_results(test_results, trained_with, version)
        results[trained_with] = test_results

    io_utils.save_results_as_json(main_path, results)
    return results


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Train and evaluate each configured model/dataset combination.

    Args:
        argv (Optional[Sequence[str]]): Optional CLI arguments for tests.

    Returns:
        None: Results are printed and persisted to disk.
    """
    args = io_utils.handle_args(argv)
    if args.handle_gpu:
        io_utils.handle_gpu_compatibility()

    datasets = data_utils.get_datasets(USE_DATASETS, n_combinations=4)
    run_training(args.version, datasets)


if __name__ == "__main__":
    main()
