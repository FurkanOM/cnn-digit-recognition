import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import Helpers

MNIST = Helpers.get_MNIST_data()
SVHN = Helpers.get_SVHN_data()
#
batch_size = 128
num_classes = 10
epochs = 12
input_shape = Helpers.get_input_shape()
trainable = ["SVHN", "MNIST", "SVHN+MNIST"]
for trained_with in trainable:
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
    early_stopping = EarlyStopping(monitor="val_accuracy", patience=3, verbose=0, mode="auto")
    model_checkpoint = ModelCheckpoint("models/"+trained_with+"_model.h5", save_best_only=True, monitor="val_accuracy", mode="auto")
    #
    if trained_with == "SVHN":
        x_train, y_train, x_valid, y_valid = SVHN["x_train"], SVHN["y_train"], SVHN["x_valid"], SVHN["y_valid"]
    elif trained_with == "MNIST":
        x_train, y_train, x_valid, y_valid = MNIST["x_train"], MNIST["y_train"], MNIST["x_valid"], MNIST["y_valid"]
    else:
        x_train, y_train, x_valid, y_valid = Helpers.merge_SVHN_MNIST_to_training(SVHN, MNIST)
    #
    print("Training started with:", trained_with, "dataset")
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_valid, y_valid),
              callbacks=[early_stopping, model_checkpoint])
    #
    MNIST_score = model.evaluate(MNIST["x_test"], MNIST["y_test"], verbose=0)
    print("Trained with:", trained_with, "MNIST Test loss:", MNIST_score[0])
    print("Trained with:", trained_with, "MNIST Test accuracy:", MNIST_score[1])
    #
    SVHN_score = model.evaluate(SVHN["x_test"], SVHN["y_test"], verbose=0)
    print("Trained with:", trained_with, "SVHN Test loss:", SVHN_score[0])
    print("Trained with:", trained_with, "SVHN Test accuracy:", SVHN_score[1])
    print("============================================================================")

import code
code.interact(local=locals())
