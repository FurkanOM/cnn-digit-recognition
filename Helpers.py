import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
import keras
from keras.datasets import mnist
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

MNIST_height, MNIST_width = 28, 28

def plot_confusion_matrix(cm, labels, normalize=True, title=None, cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    if normalize:
        a = cm.astype('float')
        b = cm.sum(axis=1)[:, np.newaxis]
        cm = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=labels, yticklabels=labels,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.margins(0.05)
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def handle_confusion_matrix(actuals, predictions, labels, title, save_path=None, print_results=True):
    cm = confusion_matrix(actuals, predictions)
    plot_confusion_matrix(cm, labels, title=title)
    if save_path:
        plt.savefig(save_path)
    if not print_results:
        return
    print('Classification Report')
    print(classification_report(actuals, predictions, labels=labels))
    print('Confusion Matrix')
    print(cm)
    print('Accuracy Score')
    print(accuracy_score(actuals, predictions))

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
