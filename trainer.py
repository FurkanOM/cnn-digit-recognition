import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import Helpers

use_datasets = ["ARDIS", "MNIST", "ORHD", "SVHN"]
num_classes = 10
epochs = 20
results = {}
#
datasets = Helpers.get_datasets(use_datasets, n_combinations=4)
input_shape = Helpers.get_input_shape()
version = Helpers.get_version()
main_path = Helpers.get_main_path(version)
for trained_with in datasets:
    model = Helpers.get_model(input_shape, version)
    model.add(Dense(num_classes, activation="softmax"))
    #
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=["accuracy"])
    #
    early_stopping = EarlyStopping(monitor="val_accuracy", patience=4, verbose=0, mode="auto")
    model_path = Helpers.get_model_path(main_path, trained_with)
    model_checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor="val_accuracy", mode="auto")
    #
    dataset = datasets[trained_with]
    x_train, y_train, x_valid, y_valid = dataset["x_train"], dataset["y_train"], dataset["x_valid"], dataset["y_valid"]
    batch_size = Helpers.get_batch_size(x_train)
    #
    print("Training started with:", trained_with, "dataset", "model version", version)
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_valid, y_valid),
              callbacks=[early_stopping, model_checkpoint])
    #
    test_results = Helpers.evaluate(model, datasets, use_datasets)
    Helpers.print_results(test_results, trained_with, version)
    results[trained_with] = test_results
#
Helpers.save_results_as_json(main_path, results)
