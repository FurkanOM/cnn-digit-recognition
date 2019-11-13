import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import keras
from keras.datasets import mnist
from keras import backend as K
from sklearn.model_selection import train_test_split

MNIST_height, MNIST_width = 28, 28

def get_files(path):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            files.append({
                "path": os.path.join(r, file),
                "filename": file
            })
    return files

def merge_SVHN_MNIST_to_training(SVHN, MNIST):
    x_train = np.concatenate((SVHN["x_train"], MNIST["x_train"]))
    y_train = np.concatenate((SVHN["y_train"], MNIST["y_train"]))
    x_valid = np.concatenate((SVHN["x_valid"], MNIST["x_valid"]))
    y_valid = np.concatenate((SVHN["y_valid"], MNIST["y_valid"]))
    return x_train, y_train, x_valid, y_valid

def handle_output_data(output, num_classes):
    # convert class vectors to binary class matrices
    return keras.utils.to_categorical(output, num_classes)

def handle_input_data(input):
    input = handle_channel(input)
    input = input.astype("float32")
    input /= 255
    return input

def get_input_shape():
    if K.image_data_format() == "channels_first":
        input_shape = (1, MNIST_height, MNIST_width)
    else:
        input_shape = (MNIST_height, MNIST_width, 1)
    return input_shape

def handle_channel(data):
    if K.image_data_format() == "channels_first":
        data = data.reshape(data.shape[0], 1, MNIST_height, MNIST_width)
    else:
        data = data.reshape(data.shape[0], MNIST_height, MNIST_width, 1)
    return data

def prepare_SVHN_to_MNIST_format(data):
    images = []
    num = data["X"].shape[-1]
    for i in range(num):
        image = Image.fromarray(data["X"][:,:,:,i])
        image = image.resize((MNIST_height, MNIST_height))
        image = np.array(image.convert("I"))
        image = image.reshape(image.shape[0], image.shape[1], 1)
        images.append(image)
    x = np.asarray(images)
    y = data["y"].reshape(data["y"].shape[0], )
    x = handle_input_data(x)
    y[y == 10] = 0
    y = handle_output_data(y, 10)
    return x, y

def prepare_final_data(x_train, y_train, x_test, y_test):
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = 0.2)
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

def get_SVHN_data():
    train_data = sio.loadmat("data/train_32x32.mat")
    test_data = sio.loadmat("data/test_32x32.mat")
    #
    x_train, y_train = prepare_SVHN_to_MNIST_format(train_data)
    x_test, y_test = prepare_SVHN_to_MNIST_format(test_data)
    #
    print("SVHN data summary: ")
    return prepare_final_data(x_train, y_train, x_test, y_test)

def get_MNIST_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #
    x_train = handle_input_data(x_train)
    x_test = handle_input_data(x_test)
    y_train = handle_output_data(y_train, 10)
    y_test = handle_output_data(y_test, 10)
    #
    print("MNIST data summary: ")
    return prepare_final_data(x_train, y_train, x_test, y_test)
