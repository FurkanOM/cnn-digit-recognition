import sys
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import Helpers

version = sys.argv[1] if len(sys.argv) > 1 else "v1"
use_datasets = ["ARDIS", "MNIST", "ORHD", "SVHN"]
batch_size = 256
num_classes = 10
epochs = 20
#
datasets = Helpers.get_datasets(use_datasets, n_combinations=4)
input_shape = Helpers.get_input_shape()
for trained_with in datasets:
    model = Helpers.get_model(input_shape, version=version)
    model.add(Dense(num_classes, activation="softmax"))
    #
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=["accuracy"])
    #
    early_stopping = EarlyStopping(monitor="val_accuracy", patience=4, verbose=0, mode="auto")
    model_path = Helpers.get_model_path(trained_with, version)
    model_checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor="val_accuracy", mode="auto")
    #
    dataset = datasets[trained_with]
    x_train, y_train, x_valid, y_valid = dataset["x_train"], dataset["y_train"], dataset["x_valid"], dataset["y_valid"]
    #
    print("Training started with:", trained_with, "dataset", "model version", version)
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_valid, y_valid),
              callbacks=[early_stopping, model_checkpoint])
    #
    print("============================================================================")
    print("Trained with:", trained_with, "model version", version)
    print("============================================================================")
    Helpers.evaluate(model, datasets, use_datasets)
    print("============================================================================")
