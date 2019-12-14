#################################################
## For GPU compatibility tf2
#################################################
import tensorflow as tf
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    print(e)
#################################################
import os
import itertools
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
from sklearn.datasets import load_digits

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

def get_model_path(trained_with):
    main_path = "models"
    if not os.path.exists(main_path):
        os.makedirs(main_path)
    model_path = main_path+"/"+trained_with+"_model.h5"
    return model_path

def handle_output_data(output, num_classes):
    # convert class vectors to binary class matrices
    return keras.utils.to_categorical(output, num_classes)

def handle_input_data(input, n=255):
    input = handle_channel(input)
    input = input.astype("float32")
    input /= n
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

def convert_array_to_MNIST_type_img(array):
    image = Image.fromarray(array)
    image = image.resize((MNIST_height, MNIST_height))
    image = np.array(image.convert("I"))
    image = image.reshape(image.shape[0], image.shape[1], 1)
    return image

def prepare_ORHD_to_MNIST_format(data):
    images = []
    num = data.images.shape[0]
    for i in range(num):
        image = convert_array_to_MNIST_type_img(data.images[i,:,:])
        images.append(image)
    x = np.asarray(images)
    x = handle_input_data(x, 16)
    y = handle_output_data(data.target, 10)
    return x, y

def prepare_SVHN_to_MNIST_format(data):
    images = []
    num = data["X"].shape[-1]
    for i in range(num):
        image = convert_array_to_MNIST_type_img(data["X"][:,:,:,i])
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

def get_ORHD_data():
    data = load_digits()
    x, y = prepare_ORHD_to_MNIST_format(data)
    #
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
    print("ORHD data summary: ")
    return prepare_final_data(x_train, y_train, x_test, y_test)

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

def get_combined_dataset(datasets, combination):
    x_train = np.concatenate([datasets[dataset]["x_train"] for dataset in combination])
    y_train = np.concatenate([datasets[dataset]["y_train"] for dataset in combination])
    x_valid = np.concatenate([datasets[dataset]["x_valid"] for dataset in combination])
    y_valid = np.concatenate([datasets[dataset]["y_valid"] for dataset in combination])
    x_test = np.concatenate([datasets[dataset]["x_test"] for dataset in combination])
    y_test = np.concatenate([datasets[dataset]["y_test"] for dataset in combination])
    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_valid": x_valid,
        "y_valid": y_valid,
        "x_test": x_test,
        "y_test": y_test,
    }

def get_datasets(use_datasets, n_combinations=1):
    datasets = {}
    for dataset in use_datasets:
        if dataset == "MNIST":
            datasets["MNIST"] = get_MNIST_data()
        elif dataset == "SVHN":
            datasets["SVHN"] = get_SVHN_data()
        elif dataset == "ORHD":
            datasets["ORHD"] = get_ORHD_data()
    for i in range(n_combinations):
        n = i + 1
        combinations = itertools.combinations(use_datasets, n)
        for combination in combinations:
            dataset = "+".join(combination)
            if dataset in datasets:
                continue
            datasets[dataset] = get_combined_dataset(datasets, combination)
    return datasets

def evaluate(model, datasets, use_datasets):
    for key in use_datasets:
        dataset = datasets[key]
        score = model.evaluate(dataset["x_test"], dataset["y_test"])
        print(key, "Test loss:", score[0])
        print(key, "Test accuracy:", score[1])
        print("=========================================")
