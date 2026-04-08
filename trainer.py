"""Train CNN digit-recognition models for every dataset combination."""

from __future__ import annotations

from typing import Optional, Sequence

import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense

import helpers


USE_DATASETS = ["ARDIS", "MNIST", "ORHD", "SVHN"]
NUM_CLASSES = 10
EPOCHS = 30


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Train and evaluate each configured model/dataset combination.

    Args:
        argv (Optional[Sequence[str]]): Optional CLI arguments for tests.

    Returns:
        None: Results are printed and persisted to disk.
    """
    results = {}
    args = helpers.handle_args(argv)
    if args.handle_gpu:
        helpers.handle_gpu_compatibility()

    datasets = helpers.get_datasets(USE_DATASETS, n_combinations=4)
    input_shape = helpers.get_input_shape()
    version = args.version
    main_path = helpers.get_main_path(version)

    for trained_with, dataset in datasets.items():
        model = helpers.get_model(input_shape, version)
        model.add(Dense(NUM_CLASSES, activation="softmax"))
        model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adadelta(
                learning_rate=0.1,
            ),
            metrics=["accuracy"],
        )

        early_stopping = EarlyStopping(monitor="val_accuracy", patience=4, verbose=0, mode="auto")
        model_path = helpers.get_model_path(main_path, trained_with)
        model_checkpoint = ModelCheckpoint(
            model_path,
            save_best_only=True,
            monitor="val_accuracy",
            mode="auto",
        )

        x_train = dataset["x_train"]
        y_train = dataset["y_train"]
        x_valid = dataset["x_valid"]
        y_valid = dataset["y_valid"]
        batch_size = helpers.get_batch_size(x_train)

        print("Training started with:", trained_with, "dataset", "model version", version)
        model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=EPOCHS,
            verbose=1,
            validation_data=(x_valid, y_valid),
            callbacks=[early_stopping, model_checkpoint],
        )

        test_results = helpers.evaluate(model, datasets, USE_DATASETS)
        helpers.print_results(test_results, trained_with, version)
        results[trained_with] = test_results

    helpers.save_results_as_json(main_path, results)


if __name__ == "__main__":
    main()
