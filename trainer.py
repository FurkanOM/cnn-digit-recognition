import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import Helpers

use_datasets = ["ORHD", "SVHN", "MNIST"]
batch_size = 256
num_classes = 10
epochs = 20
#
datasets = Helpers.get_datasets(use_datasets, n_combinations=3)
input_shape = Helpers.get_input_shape()
for trained_with in datasets:
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation="relu",
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
    #
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=["accuracy"])
    #
    early_stopping = EarlyStopping(monitor="val_accuracy", patience=4, verbose=0, mode="auto")
    model_path = Helpers.get_model_path(trained_with)
    model_checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor="val_accuracy", mode="auto")
    #
    dataset = datasets[trained_with]
    x_train, y_train, x_valid, y_valid = dataset["x_train"], dataset["y_train"], dataset["x_valid"], dataset["y_valid"]
    #
    print("Training started with:", trained_with, "dataset")
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_valid, y_valid),
              callbacks=[early_stopping, model_checkpoint])
    #
    print("============================================================================")
    print("Trained with:", trained_with)
    print("============================================================================")
    Helpers.evaluate(model, datasets, use_datasets)
    print("============================================================================")
